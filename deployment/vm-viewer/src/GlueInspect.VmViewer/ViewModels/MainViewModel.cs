using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Runtime.CompilerServices;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Threading;
using GlueInspect.VmViewer.Models;
using GlueInspect.VmViewer.Services;
using Microsoft.Win32;

namespace GlueInspect.VmViewer.ViewModels;

public abstract class ObservableObject : INotifyPropertyChanged
{
    public event PropertyChangedEventHandler? PropertyChanged;

    protected bool SetProperty<T>(ref T field, T value, [CallerMemberName] string? propertyName = null)
    {
        if (EqualityComparer<T>.Default.Equals(field, value))
            return false;
        field = value;
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        return true;
    }

    protected void RaisePropertyChanged([CallerMemberName] string? propertyName = null) =>
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
}

public sealed class RelayCommand(Action execute, Func<bool>? canExecute = null) : ICommand
{
    public event EventHandler? CanExecuteChanged;
    public bool CanExecute(object? parameter) => canExecute?.Invoke() ?? true;
    public void Execute(object? parameter) => execute();
    public void Refresh() => CanExecuteChanged?.Invoke(this, EventArgs.Empty);
}

/// <summary>VM 结果查看器主视图模型。</summary>
public sealed class MainViewModel : ObservableObject, IDisposable
{
    private readonly ViewerOptions _options;
    private readonly TcpResultReceiver _tcp = new();
    private readonly HistoryStore _historyStore;
    private readonly DispatcherTimer _refreshTimer;
    private readonly SemaphoreSlim _operationGate = new(1, 1);
    private FileSystemWatcher? _watcher;
    private string _folder;
    private BitmapSource? _originalImage;
    private BitmapSource? _caliperImage;
    private string _originalName = "等待 VM 导出原图";
    private string _caliperName = "等待 VM 导出卡尺图";
    private string _verdictText = "待判定";
    private string _verdictSubText = "等待 VM 发送 OK/NG";
    private Brush _verdictBrush = new SolidColorBrush(Color.FromRgb(0x6B, 0x77, 0x87));
    private string _tcpState = "正在启动";
    private Brush _tcpBrush = new SolidColorBrush(Color.FromRgb(0xF4, 0xB7, 0x42));
    private string _lastUpdate = "—";
    private string _imageCount = "0";
    private string _historyCount = "0";
    private string _status = "等待检测数据";
    private DateTime _lastVerdictAt = DateTime.MinValue;
    private string? _lastOriginal;
    private string? _lastCaliper;
    private string _lastLiveVerdict = "待判定";
    private string _lastLiveVerdictSubText = "等待 VM 发送 OK/NG";
    private Brush _lastLiveVerdictBrush = new SolidColorBrush(Color.FromRgb(0x6B, 0x77, 0x87));
    private bool _historyMode;
    private HistoryRecord? _selectedHistory;
    private bool _clearInProgress;
    private bool _disposed;

    public MainViewModel(ViewerOptions options)
    {
        _options = options;
        _folder = options.ImagesFolder;
        _historyStore = new HistoryStore(options.HistoryFolder);
        _refreshTimer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(220) };
        _refreshTimer.Tick += (_, _) =>
        {
            _refreshTimer.Stop();
            RefreshImages();
        };

        BrowseFolderCommand = new RelayCommand(BrowseFolder);
        RefreshCommand = new RelayCommand(RefreshImages);
        OpenFolderCommand = new RelayCommand(OpenImagesFolder);
        OpenHistoryFolderCommand = new RelayCommand(OpenHistoryFolder);
        RefreshHistoryCommand = new RelayCommand(RefreshHistory);
        ClearHistoryCommand = new RelayCommand(
            () => _ = ClearHistoryAsync(),
            () => !_clearInProgress);
        ReturnLiveCommand = new RelayCommand(ReturnToLive);
        FitCommand = new RelayCommand(() => FitRequested?.Invoke());

        _tcp.ResultReceived += message => Application.Current.Dispatcher.BeginInvoke(
            new Action(async () => await ApplyResultAsync(message)));
        _tcp.StateChanged += state => Application.Current.Dispatcher.BeginInvoke(() =>
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

    public RelayCommand BrowseFolderCommand { get; }
    public RelayCommand RefreshCommand { get; }
    public RelayCommand OpenFolderCommand { get; }
    public RelayCommand OpenHistoryFolderCommand { get; }
    public RelayCommand RefreshHistoryCommand { get; }
    public RelayCommand ClearHistoryCommand { get; }
    public RelayCommand ReturnLiveCommand { get; }
    public RelayCommand FitCommand { get; }
    public event Action? FitRequested;

    public string Folder
    {
        get => _folder;
        set
        {
            if (!SetProperty(ref _folder, value))
                return;
            ReturnToLive();
            StartWatcher();
            RefreshImages();
        }
    }

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

    public string ImageCount
    {
        get => _imageCount;
        private set => SetProperty(ref _imageCount, value);
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

    public async Task StartAsync()
    {
        if (!Directory.Exists(Folder))
            Directory.CreateDirectory(Folder);
        RefreshHistory();
        StartWatcher();
        RefreshImages();
        await _tcp.StartAsync(_options.Host, _options.Port, _options.PreferListen);
    }

    public void ApplyResult(string verdict)
    {
        var message = TcpResultReceiver.ParseMessage(verdict)
            ?? new ResultMessage(verdict, null, null, null, null, verdict);
        _ = ApplyResultAsync(message);
    }

    private async Task ApplyResultAsync(ResultMessage message)
    {
        await _operationGate.WaitAsync();
        try
        {
            var receivedAt = DateTime.Now;
            _lastVerdictAt = receivedAt;
            var lastText = $"最后一次 VM 通信 · {receivedAt:HH:mm:ss}";
            var brush = VerdictBrushFor(message.Verdict);
            _lastLiveVerdict = message.Verdict;
            _lastLiveVerdictSubText = lastText;
            _lastLiveVerdictBrush = brush;
            if (!_historyMode)
                SetDisplayedVerdict(message.Verdict, lastText, brush);

            Status = string.IsNullOrWhiteSpace(message.RecordSequence)
                ? $"收到 VM 结果：{message.Verdict}（未带记录号）"
                : $"收到 VM 结果：{message.Verdict} · 记录号 {message.RecordSequence}";

            try
            {
                var referenceTime = message.SourceTimestamp ?? receivedAt;
                var snapshot = await FindMatchingImagesAsync(referenceTime);

                if (!_historyMode)
                    ApplyLiveSnapshot(snapshot);

                var record = await _historyStore.SaveAsync(message, snapshot);
                HistoryRecords.Insert(0, record);
                HistoryCount = HistoryRecords.Count.ToString(CultureInfo.CurrentCulture);
                Status = record.HasDegradation
                    ? $"{message.Verdict} 已保存，但图片不完整：{string.Join("、", record.Degradations)}"
                    : $"{message.Verdict} 已保存到历史记录 · {record.SequenceText}";
            }
            catch (Exception ex) when (ex is not OutOfMemoryException)
            {
                Status = $"结果已显示，但历史保存失败：{ex.Message}";
            }
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
        string? finalStatus = null;
        try
        {
            var confirmation = MessageBox.Show(
                $"将永久删除以下内容：\n\n"
                + $"1. 历史记录和绑定结果：\n{_historyStore.RootDirectory}\n\n"
                + $"2. VM 输出目录中的图片：\n{Folder}\n\n"
                + "记录号之后继续按时间生成。此操作无法撤销，是否继续？",
                "清除历史记录",
                MessageBoxButton.YesNo,
                MessageBoxImage.Warning,
                MessageBoxResult.No);
            if (confirmation != MessageBoxResult.Yes)
                return;

            await _operationGate.WaitAsync();
            try
            {
                Status = "正在清除历史记录和 VM 输出图片…";
                _refreshTimer.Stop();
                var historyResult = await _historyStore.ClearAllAsync();
                var imageResult = await ImagePairService.ClearOutputImagesAsync(Folder);

                HistoryRecords.Clear();
                HistoryCount = "0";
                ImageCount = "0";
                LastUpdate = "—";
                OriginalImage = null;
                CaliperImage = null;
                OriginalName = "等待 VM 导出原图";
                CaliperName = "等待 VM 导出卡尺图";
                _lastOriginal = null;
                _lastCaliper = null;
                _lastLiveVerdict = "待判定";
                _lastLiveVerdictSubText = "等待 VM 发送 OK/NG";
                _lastLiveVerdictBrush = new SolidColorBrush(Color.FromRgb(0x6B, 0x77, 0x87));
                SetDisplayedVerdict(
                    _lastLiveVerdict,
                    _lastLiveVerdictSubText,
                    _lastLiveVerdictBrush);
                _historyMode = false;
                _selectedHistory = null;
                RaisePropertyChanged(nameof(IsHistoryMode));
                RaisePropertyChanged(nameof(SelectedHistory));

                var failedCount = historyResult.FailedPaths.Count + imageResult.FailedPaths.Count;
                finalStatus = failedCount == 0
                    ? $"已清除 {historyResult.DeletedCount} 个历史项目、{imageResult.DeletedCount} 张 VM 图片"
                    : $"清理完成，但有 {failedCount} 个文件被占用或无法删除";
            }
            catch (Exception ex) when (ex is not OutOfMemoryException)
            {
                finalStatus = $"清除历史失败：{ex.Message}";
            }
            finally
            {
                _operationGate.Release();
                RefreshImages();
                if (finalStatus is not null)
                    Status = finalStatus;
            }
        }
        finally
        {
            _clearInProgress = false;
            ClearHistoryCommand.Refresh();
        }
    }

    private async Task<ImageSnapshot> FindMatchingImagesAsync(DateTime referenceTime)
    {
        var tolerance = TimeSpan.FromMilliseconds(_options.MatchWindowMs);
        var deadline = DateTime.UtcNow.Add(tolerance).AddMilliseconds(500);
        ImageSnapshot snapshot;
        do
        {
            snapshot = ImagePairService.FindNearest(
                Folder,
                referenceTime,
                tolerance,
                fallbackToLatest: false);
            if (snapshot.OriginalPath is not null && snapshot.CaliperPath is not null)
            {
                var stable = await ImagePairService.AreFilesStableAsync(
                    snapshot,
                    TimeSpan.FromMilliseconds(120));
                if (stable && TryLoadImage(snapshot.OriginalPath) is not null
                    && TryLoadImage(snapshot.CaliperPath) is not null)
                {
                    return snapshot;
                }
            }

            await Task.Delay(100);
        }
        while (DateTime.UtcNow < deadline);

        // 等不到同一时间窗口内的完整图片时保留缺失状态，不能静默绑定旧图。
        return snapshot;
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
            $"历史记录 · {record.TimeText} · {record.ImageSequenceText}",
            VerdictBrushFor(record.Verdict));
        Status = $"正在查看历史记录 {record.SequenceText} · {record.TimeText}";
    }

    private void ReturnToLive()
    {
        _historyMode = false;
        _selectedHistory = null;
        RaisePropertyChanged(nameof(SelectedHistory));
        RaisePropertyChanged(nameof(IsHistoryMode));
        _lastOriginal = null;
        _lastCaliper = null;
        SetDisplayedVerdict(_lastLiveVerdict, _lastLiveVerdictSubText, _lastLiveVerdictBrush);
        RefreshImages();
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

    private void RefreshImages()
    {
        try
        {
            var snapshot = ImagePairService.FindLatest(Folder);
            ImageCount = snapshot.ImageCount.ToString(CultureInfo.CurrentCulture);
            LastUpdate = snapshot.UpdatedAt?.ToLocalTime().ToString(
                "yyyy-MM-dd HH:mm:ss",
                CultureInfo.CurrentCulture) ?? "—";
            if (_historyMode)
                return;

            var pairChanged = !string.Equals(_lastOriginal, snapshot.OriginalPath, StringComparison.OrdinalIgnoreCase)
                || !string.Equals(_lastCaliper, snapshot.CaliperPath, StringComparison.OrdinalIgnoreCase);
            ApplyLiveSnapshot(snapshot);

            if (pairChanged && snapshot.UpdatedAt is { } updated && updated > _lastVerdictAt.AddSeconds(2))
            {
                _lastLiveVerdict = "待判定";
                _lastLiveVerdictSubText = "检测图已更新，等待 VM 发送 OK/NG";
                _lastLiveVerdictBrush = new SolidColorBrush(Color.FromRgb(0x6B, 0x77, 0x87));
                SetDisplayedVerdict(
                    _lastLiveVerdict,
                    _lastLiveVerdictSubText,
                    _lastLiveVerdictBrush);
            }

            OriginalName = snapshot.OriginalPath is null ? "未找到 original 图片" : Path.GetFileName(snapshot.OriginalPath);
            CaliperName = snapshot.CaliperPath is null ? "未找到 caliper 图片" : Path.GetFileName(snapshot.CaliperPath);
            Status = snapshot.UpdatedAt is null ? "目录中没有可显示的 VM 图片" : "显示最新一组 VM 输出";
        }
        catch (Exception ex) when (ex is not OutOfMemoryException)
        {
            Status = $"读取图片失败：{ex.Message}";
        }
    }

    private void ApplyLiveSnapshot(ImageSnapshot snapshot)
    {
        if (snapshot.OriginalPath is not null
            && !string.Equals(_lastOriginal, snapshot.OriginalPath, StringComparison.OrdinalIgnoreCase))
        {
            OriginalImage = LoadImage(snapshot.OriginalPath);
            OriginalName = Path.GetFileName(snapshot.OriginalPath);
            _lastOriginal = snapshot.OriginalPath;
        }
        if (snapshot.CaliperPath is not null
            && !string.Equals(_lastCaliper, snapshot.CaliperPath, StringComparison.OrdinalIgnoreCase))
        {
            CaliperImage = LoadImage(snapshot.CaliperPath);
            CaliperName = Path.GetFileName(snapshot.CaliperPath);
            _lastCaliper = snapshot.CaliperPath;
        }
    }

    private void SetDisplayedVerdict(string text, string subText, Brush brush)
    {
        VerdictText = text;
        VerdictSubText = subText;
        VerdictBrush = brush;
    }

    private static SolidColorBrush VerdictBrushFor(string verdict) =>
        new SolidColorBrush(verdict == "OK"
            ? Color.FromRgb(0x2E, 0xC4, 0x8C)
            : Color.FromRgb(0xEF, 0x44, 0x59));

    private static BitmapImage LoadImage(string path)
    {
        using var file = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);
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

    private static BitmapImage? TryLoadImage(string? path)
    {
        if (path is null || !File.Exists(path))
            return null;
        try
        {
            return LoadImage(path);
        }
        catch (Exception ex) when (ex is not OutOfMemoryException)
        {
            return null;
        }
    }

    private void StartWatcher()
    {
        _watcher?.Dispose();
        _watcher = null;
        if (!Directory.Exists(Folder))
            return;

        _watcher = new FileSystemWatcher(Folder)
        {
            IncludeSubdirectories = false,
            NotifyFilter = NotifyFilters.FileName | NotifyFilters.LastWrite | NotifyFilters.Size,
            Filter = "*.*",
            EnableRaisingEvents = true,
        };
        _watcher.Created += (_, _) => QueueRefresh();
        _watcher.Changed += (_, _) => QueueRefresh();
        _watcher.Renamed += (_, _) => QueueRefresh();
    }

    private void QueueRefresh()
    {
        Application.Current.Dispatcher.BeginInvoke(() =>
        {
            _refreshTimer.Stop();
            _refreshTimer.Start();
        });
    }

    private void BrowseFolder()
    {
        var dialog = new OpenFolderDialog
        {
            Title = "选择 VM 图片输出目录",
            InitialDirectory = Directory.Exists(Folder) ? Folder : @"D:\GlueVM",
        };
        if (dialog.ShowDialog() == true)
            Folder = dialog.FolderName;
    }

    private void OpenImagesFolder() => OpenFolder(Folder);

    private void OpenHistoryFolder() => OpenFolder(_historyStore.RootDirectory);

    private static void OpenFolder(string folder)
    {
        if (!Directory.Exists(folder))
            Directory.CreateDirectory(folder);
        Process.Start(new ProcessStartInfo("explorer.exe", folder) { UseShellExecute = true });
    }

    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
        _watcher?.Dispose();
        _refreshTimer.Stop();
        _tcp.Dispose();
    }
}
