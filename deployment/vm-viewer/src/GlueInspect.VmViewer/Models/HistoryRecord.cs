using System.IO;
using System.Globalization;
using System.Text.Json.Serialization;

namespace GlueInspect.VmViewer.Models;

/// <summary>一次检测结果及其绑定图片的本地历史记录。</summary>
public sealed class HistoryRecord
{
    public string Id { get; set; } = "";
    public string? Sequence { get; init; }
    public int? OriginalSequence { get; init; }
    public int? CaliperSequence { get; init; }
    public DateTime? OriginalTimestamp { get; init; }
    public DateTime? CaliperTimestamp { get; init; }
    public string Verdict { get; init; } = "";
    public DateTime ReceivedAt { get; init; }
    public string Transport { get; init; } = "legacy_directory";
    public ulong? FrameId { get; init; }
    public long? VmTimestampUnixMs { get; init; }
    public int? FlawCount { get; init; }
    public int[] DefectTypes { get; init; } = [];
    public string? RawMessage { get; init; }
    public string? OriginalSourceName { get; init; }
    public string? CaliperSourceName { get; init; }
    public string? OriginalImageFile { get; init; }
    public string? CaliperImageFile { get; init; }
    public string[] Degradations { get; init; } = [];

    [JsonIgnore]
    public string RecordDirectory { get; set; } = "";

    [JsonIgnore]
    public string? OriginalPath => ImagePath(OriginalImageFile);

    [JsonIgnore]
    public string? CaliperPath => ImagePath(CaliperImageFile);

    [JsonIgnore]
    public string TimeText => ReceivedAt.ToLocalTime().ToString(
        "yyyy-MM-dd HH:mm:ss",
        CultureInfo.CurrentCulture);

    [JsonIgnore]
    public string SequenceText
    {
        get
        {
            if (string.IsNullOrWhiteSpace(Sequence))
                return "无记录号";
            return DateTime.TryParseExact(
                Sequence,
                "yyyyMMddHHmmssfff",
                CultureInfo.InvariantCulture,
                DateTimeStyles.AssumeLocal,
                out var timestamp)
                ? $"时间 {timestamp:HH:mm:ss.fff}"
                : $"#{Sequence}";
        }
    }

    [JsonIgnore]
    public string ImageSequenceText
    {
        get
        {
            var original = OriginalTimestamp is { } originalTime
                ? originalTime.ToString("HH:mm:ss.fff", CultureInfo.CurrentCulture)
                : OriginalSequence?.ToString(CultureInfo.CurrentCulture) ?? "?";
            var caliper = CaliperTimestamp is { } caliperTime
                ? caliperTime.ToString("HH:mm:ss.fff", CultureInfo.CurrentCulture)
                : CaliperSequence?.ToString(CultureInfo.CurrentCulture) ?? "?";
            return $"原图 {original} · 卡尺 {caliper}";
        }
    }

    [JsonIgnore]
    public bool HasDegradation => Degradations.Length > 0;

    [JsonIgnore]
    public string DefectTypeText => DefectTypeCatalog.Format(DefectTypes);

    [JsonIgnore]
    public string DefectSummaryText
    {
        get
        {
            if (FrameId is null)
                return "未记录缺陷明细";
            return (FlawCount ?? 0) > 0 && DefectTypes.Length == 0
                ? $"缺陷 {FlawCount ?? 0} · 类型未提供"
                : $"缺陷 {FlawCount ?? 0} · {DefectTypeText}";
        }
    }

    [JsonIgnore]
    public string FrameDetail
    {
        get
        {
            if (FrameId is null)
                return "";
            return $"内存帧 {FrameId} · {DefectSummaryText}";
        }
    }

    private string? ImagePath(string? fileName) =>
        string.IsNullOrWhiteSpace(fileName) ? null : Path.Combine(RecordDirectory, fileName);
}
