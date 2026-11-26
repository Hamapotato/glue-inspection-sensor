using System.Buffers.Binary;
using System.IO;

namespace GlueInspect.VmViewer.Models;

/// <summary>内存帧中的一张原始像素图，对应 docs/algo/visionmaster-memory-frame.md。</summary>
public sealed record MemoryImage(
    int Width,
    int Height,
    int Stride,
    ushort PixelFormatCode,
    byte[] Pixels)
{
    public int BytesPerPixel => PixelFormatCode switch
    {
        1 => 1,
        2 => 3,
        _ => throw new InvalidDataException($"不支持的像素格式编码：{PixelFormatCode}"),
    };

    public string PixelFormatText => PixelFormatCode switch
    {
        1 => "Mono8",
        2 => "RGB24",
        _ => $"未知({PixelFormatCode})",
    };
}

/// <summary>VM 通过 TCP 发送的一帧双图检测结果。</summary>
public sealed record MemoryFrame(
    ulong FrameId,
    long TimestampUnixMs,
    byte VerdictCode,
    int FlawCount,
    IReadOnlyList<int> DefectTypes,
    MemoryImage Original,
    MemoryImage Caliper)
{
    public string Verdict => VerdictCode switch
    {
        0 => "OK",
        1 => "NG",
        _ => "UNKNOWN",
    };

    public DateTimeOffset TimestampUtc =>
        DateTimeOffset.FromUnixTimeMilliseconds(TimestampUnixMs);

    public string DefectTypeText => DefectTypeCatalog.Format(DefectTypes);

    public string DefectSummaryText =>
        FlawCount > 0 && DefectTypes.Count == 0
            ? $"缺陷 {FlawCount} · 类型未提供"
            : $"缺陷 {FlawCount} · {DefectTypeText}";

    public string FrameSummary =>
        $"帧 {FrameId} · {TimestampUtc.ToLocalTime():HH:mm:ss.fff} · {Verdict} · "
        + DefectTypeText;
}

/// <summary>
/// 解析 VisionMaster GIFR v1 二进制帧。
/// 协议说明见 docs/algo/visionmaster-memory-frame.md。
/// </summary>
public static class MemoryFrameProtocol
{
    public const int HeaderSize = 72;
    public const int DefaultMaxFrameBytes = 64 * 1024 * 1024;
    private const ushort ProtocolVersion = 1;
    private const int DefectTypeCapacity = 4096;
    private static ReadOnlySpan<byte> Magic => "GIFR"u8;

    public static async ValueTask<MemoryFrame?> ReadAsync(
        Stream stream,
        int maxFrameBytes = DefaultMaxFrameBytes,
        CancellationToken cancellationToken = default)
    {
        var header = new byte[HeaderSize];
        var firstByteCount = await stream.ReadAsync(
            header.AsMemory(0, 1),
            cancellationToken);
        if (firstByteCount == 0)
            return null;

        await stream.ReadExactlyAsync(header.AsMemory(1), cancellationToken);
        ValidateHeader(header, maxFrameBytes, out var frameLength, out var defectTypeCount);

        var payloadLength = frameLength - HeaderSize;
        var payload = new byte[payloadLength];
        await stream.ReadExactlyAsync(payload, cancellationToken);
        return ParseHeaderAndPayload(header, payload);
    }

    public static MemoryFrame Parse(byte[] packet, int maxFrameBytes = DefaultMaxFrameBytes)
    {
        if (packet.Length < HeaderSize)
            throw new InvalidDataException("二进制帧长度小于 72 字节帧头");

        ValidateHeader(
            packet.AsSpan(0, HeaderSize),
            maxFrameBytes,
            out var frameLength,
            out _);
        if (packet.Length != frameLength)
        {
            throw new InvalidDataException(
                $"二进制帧实际长度 {packet.Length} 与帧头声明长度 {frameLength} 不一致");
        }

        return ParseHeaderAndPayload(
            packet.AsSpan(0, HeaderSize),
            packet.AsSpan(HeaderSize));
    }

    private static void ValidateHeader(
        ReadOnlySpan<byte> header,
        int maxFrameBytes,
        out int frameLength,
        out int defectTypeCount)
    {
        if (!header[..4].SequenceEqual(Magic))
            throw new InvalidDataException("二进制帧 magic 不是 GIFR");

        var version = BinaryPrimitives.ReadUInt16LittleEndian(header[4..]);
        if (version != ProtocolVersion)
            throw new InvalidDataException($"不支持的帧协议版本：{version}");

        var headerSize = BinaryPrimitives.ReadUInt16LittleEndian(header[6..]);
        if (headerSize != HeaderSize)
            throw new InvalidDataException($"不支持的帧头长度：{headerSize}");

        var declaredLength = BinaryPrimitives.ReadUInt32LittleEndian(header[8..]);
        if (declaredLength < HeaderSize || declaredLength > int.MaxValue)
            throw new InvalidDataException($"无效的整帧长度：{declaredLength}");
        frameLength = (int)declaredLength;
        if (frameLength > maxFrameBytes)
            throw new InvalidDataException(
                $"整帧长度 {frameLength} 超过接收上限 {maxFrameBytes}");

        var verdictCode = header[28];
        if (verdictCode > 2)
            throw new InvalidDataException($"无效的判定编码：{verdictCode}");

        defectTypeCount = BinaryPrimitives.ReadUInt16LittleEndian(header[32..]);
        if (defectTypeCount > DefectTypeCapacity)
        {
            throw new InvalidDataException(
                $"缺陷类型数量 {defectTypeCount} 超过接收上限 {DefectTypeCapacity}");
        }

        var original = ReadImageMetadata(header, 34, 38, "原图");
        var caliper = ReadImageMetadata(header, 36, 54, "卡尺图");
        var expectedLength = checked(
            HeaderSize
            + defectTypeCount * sizeof(ushort)
            + original.Length
            + caliper.Length);
        if (expectedLength != frameLength)
        {
            throw new InvalidDataException(
                $"帧头计算长度 {expectedLength} 与声明整帧长度 {frameLength} 不一致");
        }
    }

    private static MemoryFrame ParseHeaderAndPayload(
        ReadOnlySpan<byte> header,
        ReadOnlySpan<byte> payload)
    {
        var frameId = BinaryPrimitives.ReadUInt64LittleEndian(header[12..]);
        var timestampMs = BinaryPrimitives.ReadInt64LittleEndian(header[20..]);
        var verdictCode = header[28];
        var flawCount = BinaryPrimitives.ReadUInt16LittleEndian(header[30..]);
        var defectTypeCount = BinaryPrimitives.ReadUInt16LittleEndian(header[32..]);
        var originalMeta = ReadImageMetadata(header, 34, 38, "原图");
        var caliperMeta = ReadImageMetadata(header, 36, 54, "卡尺图");

        try
        {
            _ = DateTimeOffset.FromUnixTimeMilliseconds(timestampMs);
        }
        catch (ArgumentOutOfRangeException ex)
        {
            throw new InvalidDataException("二进制帧时间戳超出有效范围", ex);
        }

        var offset = 0;
        var defectTypes = new int[defectTypeCount];
        for (var index = 0; index < defectTypeCount; index++)
        {
            defectTypes[index] = BinaryPrimitives.ReadUInt16LittleEndian(payload[offset..]);
            offset += sizeof(ushort);
        }

        var originalPixels = payload.Slice(offset, originalMeta.Length).ToArray();
        offset += originalMeta.Length;
        var caliperPixels = payload.Slice(offset, caliperMeta.Length).ToArray();

        return new MemoryFrame(
            frameId,
            timestampMs,
            verdictCode,
            flawCount,
            defectTypes,
            CreateImage(originalMeta, originalPixels),
            CreateImage(caliperMeta, caliperPixels));
    }

    private static ImageMetadata ReadImageMetadata(
        ReadOnlySpan<byte> header,
        int pixelFormatOffset,
        int sizeOffset,
        string name)
    {
        var pixelFormat = BinaryPrimitives.ReadUInt16LittleEndian(header[pixelFormatOffset..]);
        var bytesPerPixel = pixelFormat switch
        {
            1 => 1,
            2 => 3,
            _ => throw new InvalidDataException(
                $"{name}像素格式编码无效：{pixelFormat}"),
        };
        var width = ReadUInt32AsInt(header[sizeOffset..], $"{name}宽度");
        var height = ReadUInt32AsInt(header[(sizeOffset + 4)..], $"{name}高度");
        var stride = ReadUInt32AsInt(header[(sizeOffset + 8)..], $"{name}行跨度");
        var length = ReadUInt32AsInt(header[(sizeOffset + 12)..], $"{name}字节数");

        if (width <= 0 || height <= 0)
            throw new InvalidDataException($"{name}宽高必须大于 0");
        var minimumStride = checked(width * bytesPerPixel);
        if (stride < minimumStride)
            throw new InvalidDataException(
                $"{name}行跨度 {stride} 小于最小行跨度 {minimumStride}");
        if ((long)stride * height != length)
        {
            throw new InvalidDataException(
                $"{name}行跨度乘高度不等于图像字节数");
        }

        return new ImageMetadata(
            width,
            height,
            stride,
            (ushort)pixelFormat,
            length);
    }

    private static int ReadUInt32AsInt(ReadOnlySpan<byte> source, string name)
    {
        var value = BinaryPrimitives.ReadUInt32LittleEndian(source);
        if (value > int.MaxValue)
            throw new InvalidDataException($"{name}超出 int 范围：{value}");
        return (int)value;
    }

    private static MemoryImage CreateImage(ImageMetadata metadata, byte[] pixels) =>
        new(
            metadata.Width,
            metadata.Height,
            metadata.Stride,
            metadata.PixelFormatCode,
            pixels);

    private sealed record ImageMetadata(
        int Width,
        int Height,
        int Stride,
        ushort PixelFormatCode,
        int Length);
}
