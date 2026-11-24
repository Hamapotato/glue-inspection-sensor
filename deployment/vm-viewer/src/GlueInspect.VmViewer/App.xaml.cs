using System.IO;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Threading;
using GlueInspect.VmViewer.ViewModels;

namespace GlueInspect.VmViewer;

/// <summary>VM 结果查看器启动入口。只负责显示 VM 输出，不包含检测算法。</summary>
public partial class App : Application
{
    private MainViewModel? _viewModel;

    protected override async void OnStartup(StartupEventArgs e)
    {
        base.OnStartup(e);
        var options = ViewerOptions.Parse(e.Args);
        _viewModel = new MainViewModel(options);
        var window = new MainWindow { DataContext = _viewModel };

        await _viewModel.StartAsync();

        if (options.ScreenshotPath is null)
        {
            window.Show();
            return;
        }

        window.WindowStartupLocation = WindowStartupLocation.Manual;
        window.Left = -30000;
        window.Top = 0;
        window.Width = 1640;
        window.Height = 980;
        window.Show();
        if (options.PreviewVerdict is { } verdict)
            _viewModel.ApplyResult(verdict);
        await Dispatcher.InvokeAsync(() => { }, DispatcherPriority.ApplicationIdle);
        await Task.Delay(options.ScreenshotDelayMs);
        SaveScreenshot(window, options.ScreenshotPath);
        _viewModel.Dispose();
        Shutdown();
    }

    private static void SaveScreenshot(Window window, string path)
    {
        var content = (FrameworkElement)window.Content;
        var dpi = VisualTreeHelper.GetDpi(content);
        var width = Math.Max(1, (int)(content.ActualWidth * dpi.DpiScaleX));
        var height = Math.Max(1, (int)(content.ActualHeight * dpi.DpiScaleY));
        var bitmap = new RenderTargetBitmap(width, height, 96 * dpi.DpiScaleX, 96 * dpi.DpiScaleY, PixelFormats.Pbgra32);
        var background = new DrawingVisual();
        using (var dc = background.RenderOpen())
            dc.DrawRectangle(new SolidColorBrush(Color.FromRgb(0x0B, 0x0F, 0x14)), null, new Rect(0, 0, content.ActualWidth, content.ActualHeight));
        bitmap.Render(background);
        bitmap.Render(content);
        var encoder = new PngBitmapEncoder();
        encoder.Frames.Add(BitmapFrame.Create(bitmap));
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        using var stream = File.Create(path);
        encoder.Save(stream);
    }

    protected override void OnExit(ExitEventArgs e)
    {
        _viewModel?.Dispose();
        base.OnExit(e);
    }
}

/// <summary>命令行设置。默认监听 VM 客户端连接的 127.0.0.1:5000。</summary>
public sealed record ViewerOptions
{
    public string ImagesFolder { get; init; } = @"D:\GlueVM\results";
    public string HistoryFolder { get; init; } = @"D:\GlueVM\history";
    public string Host { get; init; } = "127.0.0.1";
    public int Port { get; init; } = 5000;
    public bool PreferListen { get; init; } = true;
    public int MatchWindowMs { get; init; } = 2000;
    public string? ScreenshotPath { get; init; }
    public string? PreviewVerdict { get; init; }
    public int ScreenshotDelayMs { get; init; } = 500;

    public static ViewerOptions Parse(string[] args)
    {
        var map = new Dictionary<string, string?>(StringComparer.OrdinalIgnoreCase);
        for (var i = 0; i < args.Length; i++)
        {
            if (!args[i].StartsWith("--", StringComparison.Ordinal))
                continue;
            map[args[i][2..]] = i + 1 < args.Length && !args[i + 1].StartsWith("--", StringComparison.Ordinal)
                ? args[++i]
                : null;
        }

        var port = 5000;
        if (map.TryGetValue("port", out var portText) && int.TryParse(portText, out var parsedPort))
            port = parsedPort;
        var screenshotDelay = 500;
        if (map.TryGetValue("screenshot-delay-ms", out var delayText) && int.TryParse(delayText, out var parsedDelay))
            screenshotDelay = Math.Clamp(parsedDelay, 200, 30000);
        var matchWindow = 2000;
        if (map.TryGetValue("match-window-ms", out var matchText) && int.TryParse(matchText, out var parsedMatchWindow))
            matchWindow = Math.Clamp(parsedMatchWindow, 100, 30000);

        return new ViewerOptions
        {
            ImagesFolder = map.GetValueOrDefault("images") ?? @"D:\GlueVM\results",
            HistoryFolder = map.GetValueOrDefault("history") ?? @"D:\GlueVM\history",
            Host = map.GetValueOrDefault("host") ?? "127.0.0.1",
            Port = port,
            PreferListen = !map.ContainsKey("tcp-client"),
            MatchWindowMs = matchWindow,
            ScreenshotPath = map.GetValueOrDefault("screenshot"),
            PreviewVerdict = map.GetValueOrDefault("verdict"),
            ScreenshotDelayMs = screenshotDelay,
        };
    }
}
