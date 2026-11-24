namespace GlueInspect.VmViewer.Models;

/// <summary>一次本地文件清理的结果。</summary>
public sealed record CleanupResult(int DeletedCount, IReadOnlyList<string> FailedPaths)
{
    public bool HasFailures => FailedPaths.Count > 0;
}
