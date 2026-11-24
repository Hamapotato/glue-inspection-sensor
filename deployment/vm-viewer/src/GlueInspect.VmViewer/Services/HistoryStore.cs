using System.IO;
using System.Globalization;
using System.Text.Json;
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
