using System;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using System.Windows.Media;
using GlueInspect.VmViewer.Models;
using GlueInspect.VmViewer.Services;

var testRoot = Path.Combine(Path.GetTempPath(), $"GlueVmViewerTests-{Guid.NewGuid():N}");
try
{
    TestDefectTypePresentation();
    await TestMemoryFrameReceiverAsync();
    await TestMemoryFramePersistenceAsync(testRoot);
    await TestTcpReceiverAsync();
    await TestTimeMatchingAndHistoryAsync(testRoot);
    await TestClearScopeAsync(testRoot);
    Console.WriteLine("PASS GlueInspect.VmViewer smoke tests");
}
finally
{
    if (Directory.Exists(testRoot))
        Directory.Delete(testRoot, recursive: true);
}

static void TestDefectTypePresentation()
{
    Assert(
        DefectTypeCatalog.Format([1, 2, 3, 4, 5])
        == "边缘对宽度缺陷、边缘对位置偏移、边缘对阶梯缺陷、"
        + "边缘对断裂缺陷、边缘对气泡缺陷",
        "缺陷类型中文映射错误");
    Assert(
        DefectTypeCatalog.Format([4, 4, 5])
        == "边缘对断裂缺陷 ×2、边缘对气泡缺陷",
        "重复缺陷类型计数错误");
    Assert(
        DefectTypeCatalog.Format([99]) == "未知缺陷",
        "未知缺陷中文显示错误");
    Assert(
        DefectTypeCatalog.Format([]) == "无",
        "无缺陷类型显示错误");
}

static async Task TestMemoryFrameReceiverAsync()
{
    var firstPacket = BuildFramePacket(
        frameId: 10,
        timestampMs: 1760000000123,
        verdictCode: 1,
        flawCount: 2,
        defectTypes: [4, 5],
        originalWidth: 2,
        originalHeight: 2,
        originalFormat: 1,
        originalPixels: [0, 64, 128, 255],
        caliperWidth: 2,
        caliperHeight: 1,
        caliperFormat: 2,
        caliperPixels: [255, 0, 0, 0, 255, 0]);
    var secondPacket = BuildFramePacket(
        frameId: 11,
        timestampMs: 1760000000223,
        verdictCode: 0,
        flawCount: 0,
        defectTypes: [],
        originalWidth: 2,
        originalHeight: 2,
        originalFormat: 1,
        originalPixels: [1, 2, 3, 4],
        caliperWidth: 2,
        caliperHeight: 1,
        caliperFormat: 2,
        caliperPixels: [0, 255, 0, 0, 0, 255]);

    var parsed = MemoryFrameProtocol.Parse(firstPacket);
    Assert(parsed.FrameId == 10, "内存帧号解析错误");
    Assert(parsed.Verdict == "NG", "内存帧判定解析错误");
    Assert(parsed.FlawCount == 2, "内存帧缺陷数量解析错误");
    Assert(parsed.DefectTypes.SequenceEqual([4, 5]), "内存帧缺陷类型解析错误");
    Assert(
        parsed.DefectTypeText == "边缘对断裂缺陷、边缘对气泡缺陷",
        "内存帧缺陷类型中文显示错误");
    Assert(parsed.Original.PixelFormatText == "Mono8", "内存帧原图像素格式错误");
    Assert(parsed.Caliper.PixelFormatText == "RGB24", "内存帧卡尺图像素格式错误");

    var originalBitmap = MemoryImageConverter.ToBitmapSource(parsed.Original);
    Assert(originalBitmap.Format == PixelFormats.Gray8, "Mono8 未转换为 Gray8");
    var originalPixels = new byte[4];
    originalBitmap.CopyPixels(originalPixels, 2, 0);
    Assert(originalPixels.SequenceEqual(new byte[] { 0, 64, 128, 255 }),
        "Mono8 内存像素转换错误");

    var caliperBitmap = MemoryImageConverter.ToBitmapSource(parsed.Caliper);
    Assert(caliperBitmap.Format == PixelFormats.Rgb24, "RGB24 未转换为 Rgb24");
    var caliperPixels = new byte[6];
    caliperBitmap.CopyPixels(caliperPixels, 6, 0);
    Assert(caliperPixels.SequenceEqual(new byte[] { 255, 0, 0, 0, 255, 0 }),
        "RGB24 内存像素转换错误");

    var probe = new TcpListener(IPAddress.Loopback, 0);
    probe.Start();
    var port = ((IPEndPoint)probe.LocalEndpoint).Port;
    probe.Stop();

    using var receiver = new MemoryFrameReceiver();
    var received = new List<MemoryFrame>();
    var receivedAll = new TaskCompletionSource(
        TaskCreationOptions.RunContinuationsAsynchronously);
    receiver.FrameReceived += frame =>
    {
        lock (received)
        {
            received.Add(frame);
            if (received.Count == 2)
                receivedAll.TrySetResult();
        }
        return Task.CompletedTask;
    };
    await receiver.StartAsync("127.0.0.1", port);

    TcpClient? connectedClient = null;
    for (var attempt = 0; attempt < 30 && connectedClient is null; attempt++)
    {
        var candidate = new TcpClient();
        try
        {
            await candidate.ConnectAsync(IPAddress.Loopback, port);
            connectedClient = candidate;
        }
        catch (SocketException)
        {
            candidate.Dispose();
            await Task.Delay(50);
        }
    }

    using (var client = connectedClient
        ?? throw new InvalidOperationException("内存帧测试客户端无法连接"))
    {
        var stream = client.GetStream();
        await stream.WriteAsync(firstPacket.AsMemory(0, 13));
        await Task.Delay(20);
        var combined = firstPacket.Concat(secondPacket).ToArray();
        await stream.WriteAsync(combined.AsMemory(13));
    }

    await receivedAll.Task.WaitAsync(TimeSpan.FromSeconds(3));
    lock (received)
    {
        Assert(received.Count == 2, "内存帧连续接收数量错误");
        Assert(received[0].FrameId == 10 && received[1].FrameId == 11,
            "内存帧连续接收顺序错误");
        Assert(received[1].Verdict == "OK", "第二帧 OK 判定错误");
    }
}

static async Task TestMemoryFramePersistenceAsync(string testRoot)
{
    var historyFolder = Path.Combine(testRoot, "memory-history");
    var store = new HistoryStore(historyFolder);
    var okFrame = BuildFrame(
        frameId: 20,
        timestampMs: 1760000001000,
        verdictCode: 0,
        flawCount: 0,
        defectTypes: [],
        originalPixels: [1, 2, 3, 4],
        caliperPixels: [255, 0, 0, 0, 255, 0]);
    var ngFrame = BuildFrame(
        frameId: 21,
        timestampMs: 1760000002000,
        verdictCode: 1,
        flawCount: 1,
        defectTypes: [4],
        originalPixels: [5, 6, 7, 8],
        caliperPixels: [0, 0, 255, 255, 255, 0]);

    var okRejected = false;
    try
    {
        await store.SaveNgMemoryFrameAsync(okFrame);
    }
    catch (InvalidOperationException)
    {
        okRejected = true;
    }
    Assert(okRejected, "OK 内存帧被错误写入历史");
    Assert(!Directory.Exists(historyFolder)
        || !Directory.EnumerateFiles(historyFolder, "*", SearchOption.AllDirectories).Any(),
        "OK 内存帧产生了磁盘文件");

    var record = await store.SaveNgMemoryFrameAsync(ngFrame);
    Assert(record.Transport == "memory_frame_v1", "内存帧历史来源标记错误");
    Assert(record.FrameId == 21, "内存帧历史帧号错误");
    Assert(record.FlawCount == 1, "内存帧历史缺陷数量错误");
    Assert(record.DefectTypes.SequenceEqual([4]), "内存帧历史缺陷类型错误");
    Assert(record.DefectTypeText == "边缘对断裂缺陷", "历史缺陷类型中文显示错误");
    Assert(File.Exists(Path.Combine(record.RecordDirectory, "original.png")),
        "NG 原图未保存");
    Assert(File.Exists(Path.Combine(record.RecordDirectory, "caliper.png")),
        "NG 卡尺图未保存");
    Assert(File.Exists(Path.Combine(record.RecordDirectory, "record.json")),
        "NG record.json 未保存");
    Assert(store.LoadAll().Count == 1, "历史中出现了非 NG 内存帧");
}

static MemoryFrame BuildFrame(
    ulong frameId,
    long timestampMs,
    byte verdictCode,
    int flawCount,
    int[] defectTypes,
    byte[] originalPixels,
    byte[] caliperPixels) =>
    MemoryFrameProtocol.Parse(BuildFramePacket(
        frameId,
        timestampMs,
        verdictCode,
        flawCount,
        defectTypes,
        originalWidth: 2,
        originalHeight: 2,
        originalFormat: 1,
        originalPixels,
        caliperWidth: 2,
        caliperHeight: 1,
        caliperFormat: 2,
        caliperPixels));

static byte[] BuildFramePacket(
    ulong frameId,
    long timestampMs,
    byte verdictCode,
    int flawCount,
    int[] defectTypes,
    int originalWidth,
    int originalHeight,
    ushort originalFormat,
    byte[] originalPixels,
    int caliperWidth,
    int caliperHeight,
    ushort caliperFormat,
    byte[] caliperPixels)
{
    var originalBytesPerPixel = originalFormat == 1 ? 1 : 3;
    var caliperBytesPerPixel = caliperFormat == 1 ? 1 : 3;
    var originalStride = originalWidth * originalBytesPerPixel;
    var caliperStride = caliperWidth * caliperBytesPerPixel;
    Assert(originalPixels.Length == originalStride * originalHeight,
        "测试原图缓冲长度错误");
    Assert(caliperPixels.Length == caliperStride * caliperHeight,
        "测试卡尺图缓冲长度错误");

    var frameLength = checked(
        MemoryFrameProtocol.HeaderSize
        + defectTypes.Length * sizeof(ushort)
        + originalPixels.Length
        + caliperPixels.Length);
    using var stream = new MemoryStream(frameLength);
    using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);
    writer.Write(Encoding.ASCII.GetBytes("GIFR"));
    writer.Write((ushort)1);
    writer.Write((ushort)MemoryFrameProtocol.HeaderSize);
    writer.Write((uint)frameLength);
    writer.Write(frameId);
    writer.Write(timestampMs);
    writer.Write(verdictCode);
    writer.Write((byte)0);
    writer.Write((ushort)flawCount);
    writer.Write((ushort)defectTypes.Length);
    writer.Write(originalFormat);
    writer.Write(caliperFormat);
    writer.Write((uint)originalWidth);
    writer.Write((uint)originalHeight);
    writer.Write((uint)originalStride);
    writer.Write((uint)originalPixels.Length);
    writer.Write((uint)caliperWidth);
    writer.Write((uint)caliperHeight);
    writer.Write((uint)caliperStride);
    writer.Write((uint)caliperPixels.Length);
    writer.Write((ushort)0);
    foreach (var defectType in defectTypes)
        writer.Write((ushort)defectType);
    writer.Write(originalPixels);
    writer.Write(caliperPixels);
    writer.Flush();
    return stream.ToArray();
}

static async Task TestTcpReceiverAsync()
{
    var parsed = TcpResultReceiver.ParseMessage("20261124153000123|NG")
        ?? throw new InvalidOperationException("TCP 时间消息未解析");
    Assert(parsed.Verdict == "NG", "TCP 判定解析错误");
    Assert(parsed.SourceTimestamp == DateTime.Parse("2026-11-24 15:30:00.123"),
        "TCP 时间戳解析错误");

    var probe = new TcpListener(IPAddress.Loopback, 0);
    probe.Start();
    var port = ((IPEndPoint)probe.LocalEndpoint).Port;
    probe.Stop();

    using var receiver = new TcpResultReceiver();
    var received = new TaskCompletionSource<ResultMessage>(
        TaskCreationOptions.RunContinuationsAsynchronously);
    receiver.ResultReceived += message => received.TrySetResult(message);
    await receiver.StartAsync("127.0.0.1", port, preferListen: true);

    TcpClient? connectedClient = null;
    for (var attempt = 0; attempt < 30 && connectedClient is null; attempt++)
    {
        var candidate = new TcpClient();
        try
        {
            await candidate.ConnectAsync(IPAddress.Loopback, port);
            connectedClient = candidate;
        }
        catch (SocketException)
        {
            candidate.Dispose();
            await Task.Delay(50);
        }
    }

    using (var client = connectedClient
        ?? throw new InvalidOperationException("TCP 测试客户端无法连接"))
    {
        var payload = Encoding.UTF8.GetBytes("20261124153000123|OK");
        await client.GetStream().WriteAsync(payload);
    }

    var actual = await received.Task.WaitAsync(TimeSpan.FromSeconds(3));
    Assert(actual.Verdict == "OK", "TCP 接收判定错误");
    Assert(actual.SourceTimestamp == parsed.SourceTimestamp, "TCP 接收时间错误");
}

static async Task TestTimeMatchingAndHistoryAsync(string testRoot)
{
    var imagesFolder = Path.Combine(testRoot, "matching-images");
    var historyFolder = Path.Combine(testRoot, "matching-history");
    Directory.CreateDirectory(imagesFolder);

    var original = Path.Combine(imagesFolder, "original_20261124151438471.bmp");
    var caliper = Path.Combine(imagesFolder, "caliper_20261124151438470.bmp");
    await File.WriteAllBytesAsync(original, new byte[64]);
    await File.WriteAllBytesAsync(caliper, new byte[64]);

    var snapshot = ImagePairService.FindNearest(
        imagesFolder,
        DateTime.Parse("2026-11-24 15:14:38.471"),
        TimeSpan.FromSeconds(2),
        fallbackToLatest: false);
    Assert(snapshot.OriginalPath == original, "最近原图匹配错误");
    Assert(snapshot.CaliperPath == caliper, "最近卡尺图匹配错误");
    Assert(snapshot.OriginalTimestamp == DateTime.Parse("2026-11-24 15:14:38.471"),
        "原图时间错误");
    Assert(snapshot.CaliperTimestamp == DateTime.Parse("2026-11-24 15:14:38.470"),
        "卡尺图时间错误");
    Assert(await ImagePairService.AreFilesStableAsync(snapshot, TimeSpan.FromMilliseconds(80)),
        "已写完图片未通过稳定性检查");

    var store = new HistoryStore(historyFolder);
    var record = await store.SaveAsync(
        new ResultMessage("NG", null, null, null, null, "NG"),
        snapshot);
    Assert(Regex.IsMatch(record.Sequence!, @"^\d{17}$"), "记录号不是 17 位时间格式");
    Assert(record.OriginalTimestamp == snapshot.OriginalTimestamp, "历史原图时间未保存");
    Assert(record.CaliperTimestamp == snapshot.CaliperTimestamp, "历史卡尺图时间未保存");

    var missing = ImagePairService.FindNearest(
        imagesFolder,
        DateTime.Parse("2026-11-24 16:00:00"),
        TimeSpan.FromSeconds(2),
        fallbackToLatest: false);
    Assert(missing.OriginalPath is null && missing.CaliperPath is null,
        "时间窗口外错误回退到旧图");

    var writableOriginal = Path.Combine(imagesFolder, "original_20261124160000000.bmp");
    var writableCaliper = Path.Combine(imagesFolder, "caliper_20261124160000000.bmp");
    await File.WriteAllBytesAsync(writableOriginal, new byte[32]);
    await File.WriteAllBytesAsync(writableCaliper, new byte[32]);
    var writingSnapshot = new ImageSnapshot(
        writableOriginal,
        writableCaliper,
        null,
        null,
        null,
        null,
        DateTime.Now,
        2);

    await using var writer = new FileStream(
        writableOriginal,
        FileMode.Append,
        FileAccess.Write,
        FileShare.ReadWrite);
    var appendTask = AppendAfterDelayAsync(writer);
    var stableWhileWriting = await ImagePairService.AreFilesStableAsync(
        writingSnapshot,
        TimeSpan.FromMilliseconds(150));
    Assert(!stableWhileWriting, "正在写入的图片被错误判定为稳定");
    await appendTask;
}

static async Task TestClearScopeAsync(string testRoot)
{
    var imagesFolder = Path.Combine(testRoot, "clear-images");
    var historyFolder = Path.Combine(testRoot, "clear-history");
    var nestedFolder = Path.Combine(imagesFolder, "nested");
    Directory.CreateDirectory(nestedFolder);

    var original = Path.Combine(imagesFolder, "original_20261124170000000.bmp");
    var caliper = Path.Combine(imagesFolder, "caliper_20261124170000000.bmp");
    var keep = Path.Combine(imagesFolder, "keep.txt");
    var nestedImage = Path.Combine(nestedFolder, "keep.bmp");
    await File.WriteAllBytesAsync(original, [1, 2, 3]);
    await File.WriteAllBytesAsync(caliper, [4, 5, 6]);
    await File.WriteAllTextAsync(keep, "keep");
    await File.WriteAllBytesAsync(nestedImage, [7, 8, 9]);

    var snapshot = new ImageSnapshot(
        original,
        caliper,
        null,
        null,
        DateTime.Parse("2026-11-24 17:00:00"),
        DateTime.Parse("2026-11-24 17:00:00"),
        DateTime.Now,
        2);
    var store = new HistoryStore(historyFolder);
    await store.SaveAsync(
        new ResultMessage("OK", null, null, null, null, "OK"),
        snapshot);

    var historyCleanup = await store.ClearAllAsync();
    var imageCleanup = await ImagePairService.ClearOutputImagesAsync(imagesFolder);
    Assert(historyCleanup.DeletedCount > 0 && !Directory.EnumerateFileSystemEntries(historyFolder).Any(),
        "历史目录未清空");
    Assert(imageCleanup.DeletedCount == 2, "目标目录图片删除数量错误");
    Assert(!File.Exists(original) && !File.Exists(caliper), "目标目录图片未删除");
    Assert(File.Exists(keep), "非图片文件被误删");
    Assert(File.Exists(nestedImage), "子目录图片被越界删除");
}

static async Task AppendAfterDelayAsync(FileStream writer)
{
    await Task.Delay(60);
    await writer.WriteAsync(new byte[64]);
    await writer.FlushAsync();
}

static void Assert(bool condition, string message)
{
    if (!condition)
        throw new InvalidOperationException(message);
}
