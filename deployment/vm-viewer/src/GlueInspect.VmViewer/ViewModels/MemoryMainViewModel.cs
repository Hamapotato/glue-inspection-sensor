using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using GlueInspect.VmViewer.Models;
using GlueInspect.VmViewer.Services;

namespace GlueInspect.VmViewer.ViewModels;

/// <summary>
/// 查看 VisionMaster 内存图像帧。
/// 每帧同时显示两张图，OK 不写盘，仅 NG 原子保存到历史目录。
/// </summary>
public sealed class MemoryMainViewModel : ObservableObject, IDisposable
{
    private readonly ViewerOptions _options;
    private readonly MemoryFrameReceiver _receiver = new();
    private readonly HistoryStore _historyStore;
    private readonly SemaphoreSlim _operationGate = new(1, 1);
    private BitmapSource? _originalImage;
    private BitmapSource? _caliperImage;
    private BitmapSource? _lastLiveOriginalImage;
    private BitmapSource? _lastLiveCaliperImage;
    private string _originalName = "等待 VisionMaster 内存原图";
    private string _caliperName = "等待 VisionMaster 内存卡尺图";
    private string _lastLiveOriginalName = "等待 VisionMaster 内存原图";
    private string _lastLiveCaliperName = "等待 VisionMaster 内存卡尺图";
    private string _verdictText = "待判定";
    private string _verdictSubText = "等待 VM 发送内存帧";
    private string _defectSummaryText = "等待缺陷明细";
    private Brush _verdictBrush = new SolidColorBrush(Color.FromRgb(0x6B, 0x77, 0x87));
    private string _tcpState = "正在启动";
    private Brush _tcpBrush = new SolidColorBrush(Color.FromRgb(0xF4, 0xB7, 0x42));
    private string _lastUpdate = "—";
    private int _frameCount;
    private string _historyCount = "0";
    private string _status = "等待 VisionMaster 内存帧";
    private string _lastLiveVerdict = "待判定";
    private string _lastLiveVerdictSubText = "等待 VM 发送内存帧";
    private string _lastLiveDefectSummaryText = "等待缺陷明细";
    private Brush _lastLiveVerdictBrush =
        new SolidColorBrush(Color.FromRgb(0x6B, 0x77, 0x87));
    private bool _historyMode;
    private HistoryRecord? _selectedHistory;
    private bool _clearInProgress;
    private bool _disposed;

    public MemoryMainViewModel(ViewerOptions options)
    {
        _options = options;
        _historyStore = new HistoryStore(options.HistoryFolder);
        OpenHistoryFolderCommand = new RelayCommand(OpenHistoryFolder);
        RefreshHistoryCommand = new RelayCommand(RefreshHistory);
        ClearHistoryCommand = new RelayCommand(
            () => _ = ClearHistoryAsync(),
            () => !_clearInProgress);
        ReturnLiveCommand = new RelayCommand(ReturnToLive);
        FitCommand = new RelayCommand(() => FitRequested?.Invoke());

        _receiver.FrameReceived += ProcessFrameOnUiThreadAsync;
        _receiver.StateChanged += state => Application.Current.Dispatcher.BeginInvoke(() =>
        {
            TcpState = state.Text;
            TcpBrush = new SolidColorBrush(state.Error
                ? Color.FromRgb(0xFF, 0x64, 0x6E)
                : state.Connected
                    ? Color.FromRgb(0x48, 0xD5, 0x97)
                    : Color.FromRgb(0xF4, 0xB7, 0x42));
        });
    }

    public ObservableCollection<HistoryRecord> HistoryRecords { get; } = [];

    public RelayCommand OpenHistoryFolderCommand { get; }
    public RelayCommand RefreshHistoryCommand { get; }
    public RelayCommand ClearHistoryCommand { get; }
    public RelayCommand ReturnLiveCommand { get; }
    public RelayCommand FitCommand { get; }
    public event Action? FitRequested;

    public BitmapSource? OriginalImage
    {
        get => _originalImage;
        private set => SetProperty(ref _originalImage, value);
    }

    public BitmapSource? CaliperImage
    {
        get => _caliperImage;
        private set => SetProperty(ref _caliperImage, value);
    }

    public string OriginalName
    {
        get => _originalName;
        private set => SetProperty(ref _originalName, value);
    }

    public string CaliperName
    {
        get => _caliperName;
        private set => SetProperty(ref _caliperName, value);
    }

    public string VerdictText
    {
        get => _verdictText;
        private set => SetProperty(ref _verdictText, value);
    }

    public string VerdictSubText
    {
        get => _verdictSubText;
        private set => SetProperty(ref _verdictSubText, value);
    }

    public string DefectSummaryText
    {
        get => _defectSummaryText;
        private set => SetProperty(ref _defectSummaryText, value);
    }

    public Brush VerdictBrush
    {
        get => _verdictBrush;
        private set => SetProperty(ref _verdictBrush, value);
    }

    public string TcpState
    {
        get => _tcpState;
        private set => SetProperty(ref _tcpState, value);
    }

    public Brush TcpBrush
    {
        get => _tcpBrush;
        private set => SetProperty(ref _tcpBrush, value);
    }

    public string LastUpdate
    {
        get => _lastUpdate;
        private set => SetProperty(ref _lastUpdate, value);
    }

    public int FrameCount
    {
        get => _frameCount;
        private set => SetProperty(ref _frameCount, value);
    }

    public string HistoryCount
    {
        get => _historyCount;
        private set => SetProperty(ref _historyCount, value);
    }

    public string Status
    {
        get => _status;
        private set => SetProperty(ref _status, value);
    }

    public string HistoryFolder => _historyStore.RootDirectory;

    public HistoryRecord? SelectedHistory
    {
        get => _selectedHistory;
        set
        {
            if (!SetProperty(ref _selectedHistory, value) || value is null)
                return;
            ShowHistory(value);
        }
    }

    public bool IsHistoryMode => _historyMode;

    public string Endpoint => $"{_options.Host}:{_options.Port}";

    public Task StartAsync()
    {
        RefreshHistory();
        return _receiver.StartAsync(_options.Host, _options.Port);
    }

    /// <summary>仅用于界面截图预览，不创建检测历史。</summary>
    public void ApplyResult(string verdict)
    {
        var normalized = verdict.Trim().ToUpperInvariant();
        _lastLiveVerdict = normalized is "OK" or "NG" ? normalized : "UNKNOWN";
        _lastLiveVerdictSubText = "界面预览，不是 VM 实时帧";
        _lastLiveDefectSummaryText = "界面预览，未提供缺陷明细";
        _lastLiveVerdictBrush = VerdictBrushFor(_lastLiveVerdict);
        SetDisplayedVerdict(
            _lastLiveVerdict,
            _lastLiveVerdictSubText,
            _lastLiveDefectSummaryText,
            _lastLiveVerdictBrush);
    }

    private Task ProcessFrameOnUiThreadAsync(MemoryFrame frame)
    {
        var dispatcher = Application.Current.Dispatcher;
        if (dispatcher.CheckAccess())
            return ApplyFrameAsync(frame);
        return dispatcher
            .InvokeAsync(new Func<Task>(() => ApplyFrameAsync(frame)))
            .Task
            .Unwrap();
    }

    private async Task ApplyFrameAsync(MemoryFrame frame)
    {
        await _operationGate.WaitAsync();
        try
        {
            var original = MemoryImageConverter.ToBitmapSource(frame.Original);
            var caliper = MemoryImageConverter.ToBitmapSource(frame.Caliper);
            var localTime = frame.TimestampUtc.ToLocalTime();
            var brush = VerdictBrushFor(frame.Verdict);
            var subText =
                $"内存帧 {frame.FrameId} · {localTime:HH:mm:ss.fff} · "
                + $"缺陷 {frame.FlawCount}";

            _lastLiveOriginalImage = original;
            _lastLiveCaliperImage = caliper;
            _lastLiveOriginalName =
                $"内存原图 · {frame.Original.Width}×{frame.Original.Height} "
                + $"{frame.Original.PixelFormatText}";
            _lastLiveCaliperName =
                $"内存卡尺图 · {frame.Caliper.Width}×{frame.Caliper.Height} "
                + $"{frame.Caliper.PixelFormatText}";
            _lastLiveVerdict = frame.Verdict;
            _lastLiveVerdictSubText = subText;
            _lastLiveDefectSummaryText = frame.DefectSummaryText;
            _lastLiveVerdictBrush = brush;

            FrameCount++;
            LastUpdate = localTime.ToString(
                "yyyy-MM-dd HH:mm:ss",
                CultureInfo.CurrentCulture);
            if (!_historyMode)
            {
                OriginalImage = original;
                CaliperImage = caliper;
                OriginalName = _lastLiveOriginalName;
                CaliperName = _lastLiveCaliperName;
                SetDisplayedVerdict(
                    frame.Verdict,
                    subText,
                    frame.DefectSummaryText,
                    brush);
            }

            if (frame.VerdictCode == 0)
            {
                Status = $"收到 {frame.Verdict} 内存帧，未写硬盘 · {frame.FrameSummary}";
                return;
            }
            if (frame.VerdictCode == 2)
            {
                Status = $"收到未知判定内存帧，未写硬盘 · {frame.FrameSummary}";
                return;
            }

            try
            {
                Status = $"收到 NG 内存帧，正在保存缺陷图片 · {frame.FrameSummary}";
                var record = await _historyStore.SaveNgMemoryFrameAsync(frame);
                HistoryRecords.Insert(0, record);
                HistoryCount = HistoryRecords.Count.ToString(CultureInfo.CurrentCulture);
                Status =
                    $"NG 已保存 · 帧 {frame.FrameId} · {record.SequenceText}";
            }
            catch (Exception ex) when (ex is not OutOfMemoryException)
            {
                Status = $"NG 已显示，但缺陷图片保存失败：{ex.Message}";
            }
        }
        catch (Exception ex) when (ex is not OutOfMemoryException)
        {
            Status = $"内存帧解码失败：{ex.Message}";
        }
        finally
        {
            _operationGate.Release();
        }
    }

    private async Task ClearHistoryAsync()
    {
        if (_clearInProgress)
            return;

        _clearInProgress = true;
        ClearHistoryCommand.Refresh();
        try
        {
            var confirmation = MessageBox.Show(
                "将永久删除以下目录中的全部 NG 历史记录和缺陷图片：\n\n"
                + $"{_historyStore.RootDirectory}\n\n"
                + "当前实时画面不会被清空。此操作无法撤销，是否继续？",
                "清除 NG 历史记录",
                MessageBoxButton.YesNo,
                MessageBoxImage.Warning,
                MessageBoxResult.No);
            if (confirmation != MessageBoxResult.Yes)
                return;

            await _operationGate.WaitAsync();
            try
            {
                Status = "正在清除 NG 历史记录…";
                var result = await _historyStore.ClearAllAsync();
                HistoryRecords.Clear();
                HistoryCount = "0";
                _historyMode = false;
                _selectedHistory = null;
                RaisePropertyChanged(nameof(IsHistoryMode));
                RaisePropertyChanged(nameof(SelectedHistory));
                ReturnToLive();
                Status = result.HasFailures
                    ? $"清理完成，但有 {result.FailedPaths.Count} 个文件无法删除"
                    : $"已清除 {result.DeletedCount} 个历史项目";
            }
            catch (Exception ex) when (ex is not OutOfMemoryException)
            {
                Status = $"清除历史失败：{ex.Message}";
            }
            finally
            {
                _operationGate.Release();
            }
        }
        finally
        {
            _clearInProgress = false;
            ClearHistoryCommand.Refresh();
        }
    }

    private void ShowHistory(HistoryRecord record)
    {
        _historyMode = true;
        RaisePropertyChanged(nameof(IsHistoryMode));
        OriginalImage = TryLoadImage(record.OriginalPath);
        CaliperImage = TryLoadImage(record.CaliperPath);
        OriginalName = record.OriginalSourceName ?? "该记录缺少原图";
        CaliperName = record.CaliperSourceName ?? "该记录缺少卡尺图";
        SetDisplayedVerdict(
            record.Verdict,
            $"NG 历史记录 · {record.TimeText}",
            record.DefectSummaryText,
            VerdictBrushFor(record.Verdict));
        Status =
            $"正在查看 NG 历史记录 {record.SequenceText}"
            + (string.IsNullOrWhiteSpace(record.FrameDetail)
                ? ""
                : $" · {record.FrameDetail}");
    }

    private void ReturnToLive()
    {
        _historyMode = false;
        _selectedHistory = null;
        RaisePropertyChanged(nameof(SelectedHistory));
        RaisePropertyChanged(nameof(IsHistoryMode));
        OriginalImage = _lastLiveOriginalImage;
        CaliperImage = _lastLiveCaliperImage;
        OriginalName = _lastLiveOriginalName;
        CaliperName = _lastLiveCaliperName;
        SetDisplayedVerdict(
            _lastLiveVerdict,
            _lastLiveVerdictSubText,
            _lastLiveDefectSummaryText,
            _lastLiveVerdictBrush);
    }

    private void RefreshHistory()
    {
        try
        {
            var records = _historyStore.LoadAll();
            HistoryRecords.Clear();
            foreach (var record in records)
                HistoryRecords.Add(record);
            HistoryCount = HistoryRecords.Count.ToString(CultureInfo.CurrentCulture);
        }
        catch (Exception ex) when (ex is not OutOfMemoryException)
        {
            Status = $"读取历史记录失败：{ex.Message}";
        }
    }

    private void SetDisplayedVerdict(
        string text,
        string subText,
        string defectSummaryText,
        Brush brush)
    {
        VerdictText = text;
        VerdictSubText = subText;
        DefectSummaryText = defectSummaryText;
        VerdictBrush = brush;
    }

    private static SolidColorBrush VerdictBrushFor(string verdict) =>
        new(verdict switch
        {
            "OK" => Color.FromRgb(0x2E, 0xC4, 0x8C),
            "NG" => Color.FromRgb(0xEF, 0x44, 0x59),
            _ => Color.FromRgb(0xF4, 0xB7, 0x42),
        });

    private static BitmapImage? TryLoadImage(string? path)
    {
        if (path is null || !File.Exists(path))
            return null;
        try
        {
            using var file = new FileStream(
                path,
                FileMode.Open,
                FileAccess.Read,
                FileShare.ReadWrite | FileShare.Delete);
            using var stream = new MemoryStream();
            file.CopyTo(stream);
            stream.Position = 0;
            var image = new BitmapImage();
            image.BeginInit();
            image.CacheOption = BitmapCacheOption.OnLoad;
            image.StreamSource = stream;
            image.EndInit();
            image.Freeze();
            return image;
        }
        catch (Exception ex) when (ex is not OutOfMemoryException)
        {
            return null;
        }
    }

    private void OpenHistoryFolder()
    {
        if (!Directory.Exists(HistoryFolder))
            Directory.CreateDirectory(HistoryFolder);
        Process.Start(new ProcessStartInfo("explorer.exe", HistoryFolder)
        {
            UseShellExecute = true,
        });
    }

    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
        _receiver.Dispose();
        _operationGate.Dispose();
    }
}
