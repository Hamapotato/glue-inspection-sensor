namespace GlueInspect.VmViewer.Models;

/// <summary>
/// VisionMaster 边缘对模型缺陷类型代号与中文名称的显示映射。
/// [假设] 代号按 docs/algo/visionmaster-dragdrop-end-to-end.md 记录的 VM 4.4 资料；
/// 未知代号保留原值显示，便于现场发现映射变化。
/// </summary>
public static class DefectTypeCatalog
{
    public static string DisplayName(int code) => code switch
    {
        1 => "边缘对宽度缺陷",
        2 => "边缘对位置偏移",
        3 => "边缘对阶梯缺陷",
        4 => "边缘对断裂缺陷",
        5 => "边缘对气泡缺陷",
        _ => "未知缺陷",
    };

    public static string Format(IEnumerable<int> defectTypes)
    {
        var groups = defectTypes
            .Where(code => code > 0)
            .GroupBy(code => code)
            .Select(group =>
            {
                var label = DisplayName(group.Key);
                return group.Count() > 1 ? $"{label} ×{group.Count()}" : label;
            })
            .ToArray();
        return groups.Length == 0 ? "无" : string.Join("、", groups);
    }
}
