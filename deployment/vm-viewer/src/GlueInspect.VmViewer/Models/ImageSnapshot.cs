using System.IO;
using System.Globalization;
using System.Text.RegularExpressions;

namespace GlueInspect.VmViewer.Models;

/// <summary>当前最新的一对 VM 输出图片。</summary>
public sealed record ImageSnapshot(
    string? OriginalPath,
    string? CaliperPath,
    int? OriginalSequence,
    int? CaliperSequence,
    DateTime? OriginalTimestamp,
    DateTime? CaliperTimestamp,
    DateTime? UpdatedAt,
    int ImageCount)
{
    public static ImageSnapshot Empty { get; } = new(null, null, null, null, null, null, null, 0);
}

/// <summary>从 VM 输出目录选择最新原图和最新卡尺图。</summary>
public static partial class ImagePairService
{
    private static readonly HashSet<string> Extensions = new(StringComparer.OrdinalIgnoreCase)
    {
        ".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff",
    };

    public static ImageSnapshot FindLatest(
        string folder,
        int? expectedOriginalSequence = null,
        int? expectedCaliperSequence = null)
    {
        if (string.IsNullOrWhiteSpace(folder) || !Directory.Exists(folder))
            return ImageSnapshot.Empty;

        var files = Directory.EnumerateFiles(folder)
            .Where(path => Extensions.Contains(Path.GetExtension(path)))
            .Select(path => new FileInfo(path))
            .ToList();

        var original = files
            .Where(file => IsOriginal(file.Name))
            .OrderByDescending(file => SequenceOf(file.Name) == expectedOriginalSequence)
            .ThenByDescending(file => file.LastWriteTimeUtc)
            .FirstOrDefault();
        var caliper = files
            .Where(file => IsCaliper(file.Name))
            .OrderByDescending(file => SequenceOf(file.Name) == expectedCaliperSequence)
            .ThenByDescending(file => file.LastWriteTimeUtc)
            .FirstOrDefault();

        var latest = new[] { original?.LastWriteTime, caliper?.LastWriteTime }.Where(t => t.HasValue).Max();
        return new ImageSnapshot(
            original?.FullName,
            caliper?.FullName,
            SequenceOf(original?.Name),
            SequenceOf(caliper?.Name),
            TimestampOf(original?.Name),
            TimestampOf(caliper?.Name),
            latest,
            files.Count);
    }

    /// <summary>按 VM 消息时间查找最接近的原图和卡尺图。</summary>
    public static ImageSnapshot FindNearest(
        string folder,
        DateTime referenceTime,
        TimeSpan tolerance,
        bool fallbackToLatest = true)
    {
        if (string.IsNullOrWhiteSpace(folder) || !Directory.Exists(folder))
            return ImageSnapshot.Empty;

        var files = Directory.EnumerateFiles(folder)
            .Where(path => Extensions.Contains(Path.GetExtension(path)))
            .Select(path => new FileInfo(path))
            .ToList();

        var original = FindNearestFile(
            files.Where(file => IsOriginal(file.Name)),
            referenceTime,
            tolerance,
            fallbackToLatest);
        var caliper = FindNearestFile(
            files.Where(file => IsCaliper(file.Name)),
            referenceTime,
            tolerance,
            fallbackToLatest);
        var latest = new[] { original?.LastWriteTime, caliper?.LastWriteTime }.Where(t => t.HasValue).Max();
        return new ImageSnapshot(
            original?.FullName,
            caliper?.FullName,
            SequenceOf(original?.Name),
            SequenceOf(caliper?.Name),
            TimestampOf(original?.Name),
            TimestampOf(caliper?.Name),
            latest,
            files.Count);
    }

    public static bool IsOriginal(string fileName) =>
        fileName.StartsWith("original", StringComparison.OrdinalIgnoreCase)
        || fileName.StartsWith("原图", StringComparison.Ordinal);

    public static bool IsCaliper(string fileName) =>
        fileName.StartsWith("caliper", StringComparison.OrdinalIgnoreCase)
        || fileName.StartsWith("卡尺", StringComparison.Ordinal);

    /// <summary>等待原图和卡尺图完成写入，避免把正在写入的文件登记到历史。</summary>
    public static async Task<bool> AreFilesStableAsync(
        ImageSnapshot snapshot,
        TimeSpan stableFor,
        CancellationToken cancellationToken = default)
    {
        if (snapshot.OriginalPath is null || snapshot.CaliperPath is null)
            return false;

        var paths = new[] { snapshot.OriginalPath, snapshot.CaliperPath };
        try
        {
            var before = paths.Select(ReadStamp).ToArray();
            if (before.Any(stamp => stamp is null || stamp.Length == 0))
                return false;

            await Task.Delay(stableFor, cancellationToken);
            var after = paths.Select(ReadStamp).ToArray();
            return before.Zip(after).All(pair =>
                pair.First is { } first
                && pair.Second is { } second
                && first.Length > 0
                && first.Length == second.Length
                && first.LastWriteTimeUtc == second.LastWriteTimeUtc);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            return false;
        }
    }

    /// <summary>删除 VM 输出目录顶层的全部支持图片，不删除子目录或其他文件。</summary>
    public static async Task<CleanupResult> ClearOutputImagesAsync(
        string folder,
        CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(folder) || !Directory.Exists(folder))
            return new CleanupResult(0, []);

        var fullFolder = Path.GetFullPath(folder);
        var pathRoot = Path.GetPathRoot(fullFolder);
        if (pathRoot is not null
            && string.Equals(
                fullFolder.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar),
                pathRoot.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar),
                StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidOperationException("不能直接清理磁盘根目录。");
        }

        var files = Directory.EnumerateFiles(fullFolder, "*", SearchOption.TopDirectoryOnly)
            .Where(path => Extensions.Contains(Path.GetExtension(path)))
            .ToList();
        var deleted = 0;
        var failed = new List<string>();

        foreach (var file in files)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (await TryDeleteFileAsync(file, cancellationToken))
            {
                deleted++;
            }
            else
            {
                failed.Add(file);
            }
        }

        return new CleanupResult(deleted, failed);
    }

    public static int? SequenceOf(string? fileName)
    {
        if (string.IsNullOrWhiteSpace(fileName))
            return null;
        var match = TrailingSequenceRegex().Match(Path.GetFileNameWithoutExtension(fileName));
        return match.Success && int.TryParse(match.Value, out var value) ? value : null;
    }

    public static DateTime? TimestampOf(string? fileName)
    {
        if (string.IsNullOrWhiteSpace(fileName))
            return null;
        var match = TimestampRegex().Match(Path.GetFileNameWithoutExtension(fileName));
        if (!match.Success)
            return null;
        return DateTime.TryParseExact(
            match.Value,
            "yyyyMMddHHmmssfff",
            CultureInfo.InvariantCulture,
            DateTimeStyles.AssumeLocal,
            out var timestamp)
            ? timestamp
            : null;
    }

    private static FileInfo? FindNearestFile(
        IEnumerable<FileInfo> candidates,
        DateTime referenceTime,
        TimeSpan tolerance,
        bool fallbackToLatest)
    {
        var files = candidates.ToList();
        var nearest = files
            .Select(file => new { File = file, Timestamp = TimestampOf(file.Name) })
            .Where(item => item.Timestamp.HasValue)
            .OrderBy(item => Math.Abs((item.Timestamp!.Value - referenceTime).Ticks))
            .FirstOrDefault();

        if (nearest is not null && (nearest.Timestamp!.Value - referenceTime).Duration() <= tolerance)
            return nearest.File;

        return fallbackToLatest
            ? files.OrderByDescending(file => file.LastWriteTimeUtc).FirstOrDefault()
            : null;
    }

    private static async Task<bool> TryDeleteFileAsync(
        string path,
        CancellationToken cancellationToken)
    {
        for (var attempt = 0; attempt < 4; attempt++)
        {
            try
            {
                File.Delete(path);
                return true;
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                if (attempt == 3)
                    return false;
                await Task.Delay(150, cancellationToken);
            }
        }
        return false;
    }

    private static FileStamp? ReadStamp(string path)
    {
        var file = new FileInfo(path);
        return file.Exists ? new FileStamp(file.Length, file.LastWriteTimeUtc) : null;
    }

    private sealed record FileStamp(long Length, DateTime LastWriteTimeUtc);

    [GeneratedRegex(@"(\d+)(?!.*\d)")]
    private static partial Regex TrailingSequenceRegex();

    [GeneratedRegex(@"(?<!\d)(20\d{15})(?!\d)")]
    private static partial Regex TimestampRegex();
}
