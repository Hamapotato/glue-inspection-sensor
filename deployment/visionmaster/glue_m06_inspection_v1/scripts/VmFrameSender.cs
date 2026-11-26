using System;
using System.Globalization;
using System.IO;
using System.Text;
using Script.Methods;

/// <summary>
/// 将 VM 原图、卡尺图和检测结果组装为一帧二进制数据，并通过 TCP 发送。
/// 对应 docs/algo/visionmaster-memory-frame.md。
/// </summary>
public partial class UserScript : ScriptMethods, IProcessMethods
{
    private const ushort ProtocolVersion = 1;
    private const ushort HeaderSize = 72;
    private const int DefectArrayCapacity = 64;
    private const int DefaultMaxFrameBytes = 64 * 1024 * 1024;
    private const string DefaultConfigPath =
        @"D:\WORK\glue-inspection-sensor\deployment\visionmaster\glue_m06_inspection_v1\config\vm_frame_sender.ini";
    private static readonly byte[] Magic = new byte[] { 0x47, 0x49, 0x46, 0x52 }; // GIFR

    private int tcpDeviceId = 1;
    private int maxFrameBytes = DefaultMaxFrameBytes;
    private string loadedConfigPath = string.Empty;
    private long frameSequence;

    /// <summary>预编译或方案加载时初始化。</summary>
    public void Init()
    {
        loadedConfigPath = string.Empty;
        frameSequence = 0;
    }

    /// <summary>每帧执行一次，组装并发送一帧图像与检测结果。</summary>
    public bool Process()
    {
        try
        {
            string verdict = string.Empty;
            int flawCount = 0;
            ImageData originalImage = new ImageData();
            ImageData caliperImage = new ImageData();

            RequireSuccess(GetImageValue("originalImage", ref originalImage), "originalImage");
            RequireSuccess(GetImageValue("caliperImage", ref caliperImage), "caliperImage");
            RequireSuccess(GetStringValue("verdict", ref verdict), "verdict");
            byte verdictCode = ParseVerdict(verdict);

            // [假设] OK 帧可能不产生缺陷结果；VM 对这类未生成结果返回非零读取码。
            int flawCountResult = GetIntValue("flawCount", ref flawCount);
            bool defectDetailsMissing =
                AcceptOptionalDefectResult(flawCountResult, verdictCode, "flawCount");
            int[] defectTypes = new int[DefectArrayCapacity];
            int defectTypeCount = 0;
            int defectTypesResult =
                GetIntArrayValue("defectTypes", ref defectTypes, out defectTypeCount);
            defectDetailsMissing =
                AcceptOptionalDefectResult(
                    defectTypesResult,
                    verdictCode,
                    "defectTypes") || defectDetailsMissing;
            ValidateDefectTypeCount(defectTypes, defectTypeCount);
            ValidateFlawCount(flawCount);

            LoadConfig(DefaultConfigPath);

            ImageInfo original = InspectImage(originalImage, "originalImage");
            ImageInfo caliper = InspectImage(caliperImage, "caliperImage");

            int validDefectCount = CountValidDefects(defectTypes, defectTypeCount);
            int packetLength = checked(
                HeaderSize
                + validDefectCount * sizeof(ushort)
                + original.Buffer.Length
                + caliper.Buffer.Length);
            if (packetLength > maxFrameBytes)
            {
                throw new InvalidOperationException(
                    string.Format(
                        CultureInfo.InvariantCulture,
                        "帧大小 {0} 字节超过配置上限 {1} 字节",
                        packetLength,
                        maxFrameBytes));
            }

            long timestampMs = ToUnixMilliseconds(DateTime.UtcNow);
            frameSequence++;
            long frameId = checked(timestampMs * 1000L + frameSequence % 1000L);
            byte[] packet = BuildPacket(
                frameId,
                timestampMs,
                verdictCode,
                flawCount,
                defectTypes,
                defectTypeCount,
                original,
                caliper);

            int sendResult = GlobalCommunicateModule
                .GetDevice(tcpDeviceId)
                .SendData(packet, DataType.ByteType);
            if (sendResult != 0)
            {
                SetStringValue("sendStatus", "SEND_FAILED");
                SetStringValue("lastFrameId", frameId.ToString(CultureInfo.InvariantCulture));
                ConsoleWrite(
                    string.Format(
                        CultureInfo.InvariantCulture,
                        "VM 图像帧发送失败：deviceId={0}, code={1}, bytes={2}",
                        tcpDeviceId,
                        sendResult,
                        packetLength));
                return false;
            }

            string sendStatus = "SENT";
            if (verdictCode == 2)
                sendStatus = "SENT_UNKNOWN_VERDICT";
            else if (defectDetailsMissing)
                sendStatus = "SENT_WITHOUT_DEFECT_DETAILS";
            SetStringValue("sendStatus", sendStatus);
            SetStringValue("lastFrameId", frameId.ToString(CultureInfo.InvariantCulture));
            return verdictCode != 2;
        }
        catch (Exception ex)
        {
            SetStringValue("sendStatus", "ERROR");
            ConsoleWrite("VM 图像帧组装或发送异常：" + ex.Message);
            return false;
        }
    }

    private void LoadConfig(string configPath)
    {
        if (string.IsNullOrWhiteSpace(configPath))
            throw new ArgumentException("configPath 不能为空");
        if (!Path.IsPathRooted(configPath))
            throw new ArgumentException("configPath 必须是绝对路径");

        string fullPath = Path.GetFullPath(configPath);
        if (string.Equals(loadedConfigPath, fullPath, StringComparison.OrdinalIgnoreCase))
            return;
        if (!File.Exists(fullPath))
            throw new FileNotFoundException("找不到帧发送配置", fullPath);

        int parsedDeviceId = 1;
        int parsedMaxFrameBytes = DefaultMaxFrameBytes;
        foreach (string rawLine in File.ReadAllLines(fullPath, Encoding.UTF8))
        {
            string line = rawLine.Trim();
            if (line.Length == 0 || line.StartsWith("#", StringComparison.Ordinal))
                continue;

            int separator = line.IndexOf('=');
            if (separator <= 0)
                continue;
            string key = line.Substring(0, separator).Trim();
            string value = line.Substring(separator + 1).Trim();
            if (key.Equals("tcp_device_id", StringComparison.OrdinalIgnoreCase))
            {
                if (!int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out parsedDeviceId))
                    throw new FormatException("tcp_device_id 必须是整数");
            }
            else if (key.Equals("max_frame_bytes", StringComparison.OrdinalIgnoreCase))
            {
                if (!int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out parsedMaxFrameBytes))
                    throw new FormatException("max_frame_bytes 必须是整数");
            }
        }

        if (parsedDeviceId <= 0)
            throw new FormatException("tcp_device_id 必须大于 0");
        if (parsedMaxFrameBytes <= HeaderSize)
            throw new FormatException("max_frame_bytes 必须大于协议帧头长度");

        tcpDeviceId = parsedDeviceId;
        maxFrameBytes = parsedMaxFrameBytes;
        loadedConfigPath = fullPath;
    }

    private static byte[] BuildPacket(
        long frameId,
        long timestampMs,
        byte verdictCode,
        int flawCount,
        int[] defectTypes,
        int defectTypeCount,
        ImageInfo original,
        ImageInfo caliper)
    {
        int validDefectCount = CountValidDefects(defectTypes, defectTypeCount);
        int packetLength = checked(
            HeaderSize
            + validDefectCount * sizeof(ushort)
            + original.Buffer.Length
            + caliper.Buffer.Length);
        byte[] packet = new byte[packetLength];

        using (MemoryStream stream = new MemoryStream(packet, 0, packet.Length, true, true))
        using (BinaryWriter writer = new BinaryWriter(stream, Encoding.UTF8, true))
        {
            writer.Write(Magic);
            writer.Write(ProtocolVersion);
            writer.Write(HeaderSize);
            writer.Write(packetLength);
            writer.Write(frameId);
            writer.Write(timestampMs);
            writer.Write(verdictCode);
            writer.Write((byte)0);
            writer.Write((ushort)Math.Max(0, flawCount));
            writer.Write((ushort)validDefectCount);
            writer.Write(original.PixelFormatCode);
            writer.Write(caliper.PixelFormatCode);
            writer.Write((uint)original.Width);
            writer.Write((uint)original.Height);
            writer.Write((uint)original.Stride);
            writer.Write((uint)original.Buffer.Length);
            writer.Write((uint)caliper.Width);
            writer.Write((uint)caliper.Height);
            writer.Write((uint)caliper.Stride);
            writer.Write((uint)caliper.Buffer.Length);
            writer.Write((ushort)0);

            for (int index = 0; index < defectTypeCount; index++)
            {
                int defectType = defectTypes[index];
                if (defectType > 0)
                    writer.Write((ushort)defectType);
            }

            writer.Write(original.Buffer);
            writer.Write(caliper.Buffer);
        }

        return packet;
    }

    private static ImageInfo InspectImage(ImageData image, string name)
    {
        if (image == null || image.Buffer == null || image.Buffer.Length == 0)
            throw new ArgumentException(name + " 没有有效图像数据");
        if (image.Width <= 0 || image.Height <= 0)
            throw new ArgumentException(name + " 的图像宽高无效");

        int bytesPerPixel;
        ushort pixelFormatCode;
        if (image.PixelFormat == ImagePixelFormate.MONO8)
        {
            bytesPerPixel = 1;
            pixelFormatCode = 1;
        }
        else if (image.PixelFormat == ImagePixelFormate.RGB24)
        {
            bytesPerPixel = 3;
            pixelFormatCode = 2;
        }
        else
        {
            throw new NotSupportedException(
                name + " 的像素格式不受支持：" + image.PixelFormat);
        }

        if (image.Buffer.Length % image.Height != 0)
            throw new ArgumentException(name + " 的像素缓冲长度不能按行整除");

        int stride = image.Buffer.Length / image.Height;
        int minimumStride = checked(image.Width * bytesPerPixel);
        if (stride < minimumStride)
        {
            throw new ArgumentException(
                string.Format(
                    CultureInfo.InvariantCulture,
                    "{0} 的行跨度 {1} 小于最小行跨度 {2}",
                    name,
                    stride,
                    minimumStride));
        }

        return new ImageInfo(image.Buffer, image.Width, image.Height, stride, pixelFormatCode);
    }

    private static int CountValidDefects(int[] defectTypes, int count)
    {
        ValidateDefectTypeCount(defectTypes, count);
        int validCount = 0;
        for (int index = 0; index < count; index++)
        {
            int defectType = defectTypes[index];
            if (defectType <= 0)
                continue;
            if (defectType > ushort.MaxValue)
                throw new ArgumentOutOfRangeException("defectTypes", "缺陷类型编码超过 uint16 范围");
            validCount++;
        }
        return validCount;
    }

    private static void ValidateDefectTypeCount(int[] defectTypes, int count)
    {
        if (count < 0)
            throw new ArgumentOutOfRangeException("defectTypes", "缺陷类型数量不能为负数");
        if (count > defectTypes.Length)
        {
            throw new InvalidOperationException(
                string.Format(
                    CultureInfo.InvariantCulture,
                    "缺陷类型数量 {0} 超过脚本数组容量 {1}，拒绝发送以避免静默截断",
                    count,
                    defectTypes.Length));
        }
    }

    private static void ValidateFlawCount(int flawCount)
    {
        if (flawCount < 0)
            throw new ArgumentOutOfRangeException("flawCount", "缺陷个数不能为负数");
        if (flawCount > ushort.MaxValue)
            throw new ArgumentOutOfRangeException("flawCount", "缺陷个数超过 uint16 范围");
    }

    private static byte ParseVerdict(string verdict)
    {
        string normalized = (verdict ?? string.Empty).Trim().ToUpperInvariant();
        if (normalized == "OK")
            return 0;
        if (normalized == "NG")
            return 1;
        return 2;
    }

    private static bool AcceptOptionalDefectResult(
        int result,
        byte verdictCode,
        string variableName)
    {
        if (result == 0)
            return false;
        if (verdictCode != 0)
            RequireSuccess(result, variableName);
        return true;
    }

    private static long ToUnixMilliseconds(DateTime utcTime)
    {
        DateTime epoch = new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc);
        return (long)(utcTime - epoch).TotalMilliseconds;
    }

    private static void RequireSuccess(int result, string variableName)
    {
        if (result != 0)
            throw new InvalidOperationException(variableName + " 读取失败，返回码 " + result);
    }

    private sealed class ImageInfo
    {
        public ImageInfo(
            byte[] buffer,
            int width,
            int height,
            int stride,
            ushort pixelFormatCode)
        {
            Buffer = buffer;
            Width = width;
            Height = height;
            Stride = stride;
            PixelFormatCode = pixelFormatCode;
        }

        public byte[] Buffer { get; private set; }
        public int Width { get; private set; }
        public int Height { get; private set; }
        public int Stride { get; private set; }
        public ushort PixelFormatCode { get; private set; }
    }
}
