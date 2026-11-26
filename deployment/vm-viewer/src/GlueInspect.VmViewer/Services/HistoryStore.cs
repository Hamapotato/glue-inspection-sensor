using System.IO;
using System.Globalization;
using System.Text.Json;
using System.Windows.Media.Imaging;
using GlueInspect.VmViewer.Models;

namespace GlueInspect.VmViewer.Services;

/// <summary>把每次 VM 结果和对应图片复制到本地历史目录。</summary>
public sealed class HistoryStore(string rootDirectory)
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,
    };

    public string RootDirectory { get; } = rootDirectory;

    public async Task<HistoryRecord> SaveAsync(
        ResultMessage message,
        ImageSnapshot snapshot,
        CancellationToken cancellationToken = default)
    {
        var receivedAt = DateTime.Now;
        var dayDirectory = Path.Combine(
            RootDirectory,
            receivedAt.ToString("yyyyMMdd", CultureInfo.InvariantCulture));
        Directory.CreateDirectory(dayDirectory);

        // 记录号统一使用 VM 时间；未发送时间时使用结果到达时间。
        var sequence = (message.SourceTimestamp ?? receivedAt).ToString(
            "yyyyMMddHHmmssfff",
            CultureInfo.InvariantCulture);
        var temporaryDirectory = Path.Combine(dayDirectory, $".{sequence}.{Guid.NewGuid():N}.tmp");
        Directory.CreateDirectory(temporaryDirectory);

        string? originalImageFile = null;
        string? caliperImageFile = null;
        var degradations = new List<string>();
        if (snapshot.OriginalPath is not null)
        {
            originalImageFile = $"original{Path.GetExtension(snapshot.OriginalPath)}";
            if (!await CopyImageAsync(
                snapshot.OriginalPath,
                Path.Combine(temporaryDirectory, originalImageFile),
                cancellationToken))
            {
                originalImageFile = null;
                degradations.Add("原图复制失败");
            }
        }
        else
        {
            degradations.Add("时间窗口内未找到原图");
        }
        if (snapshot.CaliperPath is not null)
        {
            caliperImageFile = $"caliper{Path.GetExtension(snapshot.CaliperPath)}";
            if (!await CopyImageAsync(
                snapshot.CaliperPath,
                Path.Combine(temporaryDirectory, caliperImageFile),
                cancellationToken))
            {
                caliperImageFile = null;
                degradations.Add("卡尺图复制失败");
            }
        }
        else
        {
            degradations.Add("时间窗口内未找到卡尺图");
        }

        var finalDirectory = CreateUniqueDirectory(dayDirectory, sequence, receivedAt);
        var record = new HistoryRecord
        {
            Id = Path.GetRelativePath(RootDirectory, finalDirectory),
            Sequence = sequence,
            OriginalSequence = message.OriginalSequence ?? snapshot.OriginalSequence,
            CaliperSequence = message.CaliperSequence ?? snapshot.CaliperSequence,
            OriginalTimestamp = snapshot.OriginalTimestamp,
            CaliperTimestamp = snapshot.CaliperTimestamp,
            Verdict = message.Verdict,
            ReceivedAt = receivedAt,
            RawMessage = message.Raw,
            OriginalSourceName = snapshot.OriginalPath is null ? null : Path.GetFileName(snapshot.OriginalPath),
            CaliperSourceName = snapshot.CaliperPath is null ? null : Path.GetFileName(snapshot.CaliperPath),
            OriginalImageFile = originalImageFile,
            CaliperImageFile = caliperImageFile,
            Degradations = degradations.ToArray(),
            RecordDirectory = finalDirectory,
        };

        var recordJson = JsonSerializer.Serialize(record, JsonOptions);
        await File.WriteAllTextAsync(
            Path.Combine(temporaryDirectory, "record.json"),
            recordJson,
            cancellationToken);

        Directory.Move(temporaryDirectory, finalDirectory);
        return record;
    }

    /// <summary>
    /// 只保存内存帧中的 NG；OK/UNKNOWN 不应调用此方法。
    /// 图片和 record.json 先写入临时目录，全部完成后原子改名为正式记录目录。
    /// </summary>
    public async Task<HistoryRecord> SaveNgMemoryFrameAsync(
        MemoryFrame frame,
        CancellationToken cancellationToken = default)
    {
        if (frame.VerdictCode != 1)
        {
            throw new InvalidOperationException(
                "仅允许持久化内存帧中的 NG 结果");
        }

        var receivedAt = DateTime.Now;
        var frameLocalTime = frame.TimestampUtc.ToLocalTime().DateTime;
        var dayDirectory = Path.Combine(
            RootDirectory,
            frameLocalTime.ToString("yyyyMMdd", CultureInfo.InvariantCulture));
        Directory.CreateDirectory(dayDirectory);

        var sequence = frameLocalTime.ToString(
            "yyyyMMddHHmmssfff",
            CultureInfo.InvariantCulture);
        var temporaryDirectory = Path.Combine(
            dayDirectory,
            $".{sequence}.{Guid.NewGuid():N}.tmp");
        Directory.CreateDirectory(temporaryDirectory);

        try
        {
            var originalBitmap = MemoryImageConverter.ToBitmapSource(frame.Original);
            var caliperBitmap = MemoryImageConverter.ToBitmapSource(frame.Caliper);
            await Task.Run(
                () =>
                {
                    SavePng(originalBitmap, Path.Combine(temporaryDirectory, "original.png"));
                    SavePng(caliperBitmap, Path.Combine(temporaryDirectory, "caliper.png"));
                },
                cancellationToken);

            var finalDirectory = CreateUniqueDirectory(dayDirectory, sequence, receivedAt);
            var record = new HistoryRecord
            {
                Id = Path.GetRelativePath(RootDirectory, finalDirectory),
                Sequence = sequence,
                Verdict = frame.Verdict,
                ReceivedAt = receivedAt,
                Transport = "memory_frame_v1",
                FrameId = frame.FrameId,
                VmTimestampUnixMs = frame.TimestampUnixMs,
                FlawCount = frame.FlawCount,
                DefectTypes = frame.DefectTypes.ToArray(),
                OriginalTimestamp = frameLocalTime,
                CaliperTimestamp = frameLocalTime,
                RawMessage =
                    $"GIFR frame_id={frame.FrameId}; verdict={frame.Verdict}; "
                    + $"flaw_count={frame.FlawCount}",
                OriginalSourceName = $"memory-original-{frame.FrameId}",
                CaliperSourceName = $"memory-caliper-{frame.FrameId}",
                OriginalImageFile = "original.png",
                CaliperImageFile = "caliper.png",
                RecordDirectory = finalDirectory,
            };

            var recordJson = JsonSerializer.Serialize(record, JsonOptions);
            await File.WriteAllTextAsync(
                Path.Combine(temporaryDirectory, "record.json"),
                recordJson,
                cancellationToken);
            Directory.Move(temporaryDirectory, finalDirectory);
            return record;
        }
        catch
        {
            try
            {
                if (Directory.Exists(temporaryDirectory))
                    Directory.Delete(temporaryDirectory, recursive: true);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
            }
            throw;
        }
    }

    /// <summary>删除历史根目录下的全部记录目录和文件。</summary>
    public async Task<CleanupResult> ClearAllAsync(CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(RootDirectory) || !Directory.Exists(RootDirectory))
            return new CleanupResult(0, []);

        var fullRoot = Path.GetFullPath(RootDirectory);
        var pathRoot = Path.GetPathRoot(fullRoot);
        if (pathRoot is not null
            && string.Equals(
                fullRoot.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar),
                pathRoot.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar),
                StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidOperationException("历史目录不能是磁盘根目录。");
        }

        var entries = Directory.EnumerateFileSystemEntries(fullRoot)
            .Select(Path.GetFullPath)
            .ToList();
        var deleted = 0;
        var failed = new List<string>();

        foreach (var entry in entries)
        {
            cancellationToken.ThrowIfCancellationRequested();
            try
            {
                if (Directory.Exists(entry))
                    Directory.Delete(entry, recursive: true);
                else
                    File.Delete(entry);
                deleted++;
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                failed.Add(entry);
            }
        }

        await Task.CompletedTask;
        return new CleanupResult(deleted, failed);
    }

    public IReadOnlyList<HistoryRecord> LoadAll()
    {
        if (!Directory.Exists(RootDirectory))
            return [];

        var records = new List<HistoryRecord>();
        foreach (var file in Directory.EnumerateFiles(RootDirectory, "record.json", SearchOption.AllDirectories))
        {
            try
            {
                var record = JsonSerializer.Deserialize<HistoryRecord>(File.ReadAllText(file));
                if (record is null || string.IsNullOrWhiteSpace(record.Verdict))
                    continue;
                record.RecordDirectory = Path.GetDirectoryName(file)!;
                records.Add(record);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or JsonException)
            {
                // 单条损坏记录不阻塞其他历史记录加载。
            }
        }

        return records
            .OrderByDescending(record => record.ReceivedAt)
            .ToList();
    }

    private static string CreateUniqueDirectory(string dayDirectory, string sequence, DateTime receivedAt)
    {
        var stem = $"{sequence}_{receivedAt:yyyyMMdd_HHmmss_fff}";
        var candidate = Path.Combine(dayDirectory, stem);
        var suffix = 1;
        while (Directory.Exists(candidate))
            candidate = Path.Combine(dayDirectory, $"{stem}_{suffix++}");
        return candidate;
    }

    private static void SavePng(BitmapSource bitmap, string path)
    {
        var encoder = new PngBitmapEncoder();
        encoder.Frames.Add(BitmapFrame.Create(bitmap));
        using var stream = new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.None);
        encoder.Save(stream);
    }

    private static async Task<bool> CopyImageAsync(
        string source,
        string destination,
        CancellationToken cancellationToken)
    {
        for (var attempt = 0; attempt < 4; attempt++)
        {
            try
            {
                await using var input = new FileStream(
                    source,
                    FileMode.Open,
                    FileAccess.Read,
                    FileShare.ReadWrite | FileShare.Delete,
                    bufferSize: 81920,
                    useAsync: true);
                await using var output = new FileStream(
                    destination,
                    FileMode.Create,
                    FileAccess.Write,
                    FileShare.None,
                    bufferSize: 81920,
                    useAsync: true);
                await input.CopyToAsync(output, cancellationToken);
                return true;
            }
            catch (IOException) when (attempt < 3)
            {
                await Task.Delay(150, cancellationToken);
            }
            catch (IOException)
            {
                try
                {
                    File.Delete(destination);
                }
                catch (IOException)
                {
                }
                return false;
            }
        }
        return false;
    }
}
