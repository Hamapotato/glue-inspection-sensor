using System;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using GlueInspect.VmViewer.Models;
using GlueInspect.VmViewer.Services;

var testRoot = Path.Combine(Path.GetTempPath(), $"GlueVmViewerTests-{Guid.NewGuid():N}");
try
{
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
