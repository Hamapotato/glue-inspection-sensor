using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace GlueInspect.VmViewer.Controls;

/// <summary>支持滚轮缩放、右键拖动和平移的图片查看控件。</summary>
public sealed class ImageViewer : Grid
{
    public static readonly DependencyProperty SourceProperty = DependencyProperty.Register(
        nameof(Source),
        typeof(ImageSource),
        typeof(ImageViewer),
        new PropertyMetadata(null, (d, e) => ((ImageViewer)d).OnSourceChanged(e)));

    public ImageSource? Source
    {
        get => (ImageSource?)GetValue(SourceProperty);
        set => SetValue(SourceProperty, value);
    }

    private readonly Canvas _content = new();
    private readonly Image _image = new() { Stretch = Stretch.None };
    private readonly ScaleTransform _scale = new(1, 1);
    private readonly TranslateTransform _translate = new();
    private Point? _dragStart;
    private Point _dragOrigin;
    private bool _userZoomed;

    public ImageViewer()
    {
        ClipToBounds = true;
        Background = new SolidColorBrush(Color.FromRgb(0x08, 0x0C, 0x11));
        RenderOptions.SetBitmapScalingMode(_image, BitmapScalingMode.HighQuality);
        _content.RenderTransform = new TransformGroup { Children = { _scale, _translate } };
        _content.Children.Add(_image);
        Children.Add(_content);
        SizeChanged += (_, _) => { if (!_userZoomed) Fit(); };
    }

    public void Fit()
    {
        if (Source is null || ActualWidth <= 0 || ActualHeight <= 0)
            return;

        var scale = Math.Min(ActualWidth / Source.Width, ActualHeight / Source.Height) * 0.97;
        _scale.ScaleX = _scale.ScaleY = scale;
        _translate.X = (ActualWidth - Source.Width * scale) / 2;
        _translate.Y = (ActualHeight - Source.Height * scale) / 2;
        _userZoomed = false;
    }

    private void OnSourceChanged(DependencyPropertyChangedEventArgs e)
    {
        _image.Source = Source;
        _image.Width = Source?.Width ?? 0;
        _image.Height = Source?.Height ?? 0;
        _userZoomed = false;
        Dispatcher.BeginInvoke(Fit, System.Windows.Threading.DispatcherPriority.Loaded);
    }

    protected override void OnMouseWheel(MouseWheelEventArgs e)
    {
        base.OnMouseWheel(e);
        if (Source is null)
            return;

        var point = e.GetPosition(this);
        var before = ToImage(point);
        var factor = e.Delta > 0 ? 1.18 : 1 / 1.18;
        var scale = Math.Clamp(_scale.ScaleX * factor, 0.02, 40);
        _scale.ScaleX = _scale.ScaleY = scale;
        _translate.X = point.X - before.X * scale;
        _translate.Y = point.Y - before.Y * scale;
        _userZoomed = true;
        e.Handled = true;
    }

    protected override void OnMouseDown(MouseButtonEventArgs e)
    {
        base.OnMouseDown(e);
        if (Source is null)
            return;

        if (e.ChangedButton == MouseButton.Right && e.ClickCount == 2)
        {
            Fit();
            return;
        }

        if (e.ChangedButton is MouseButton.Left or MouseButton.Middle or MouseButton.Right)
        {
            _dragStart = e.GetPosition(this);
            _dragOrigin = new Point(_translate.X, _translate.Y);
            CaptureMouse();
        }
    }

    protected override void OnMouseMove(MouseEventArgs e)
    {
        base.OnMouseMove(e);
        if (_dragStart is not { } start)
            return;

        var point = e.GetPosition(this);
        _translate.X = _dragOrigin.X + point.X - start.X;
        _translate.Y = _dragOrigin.Y + point.Y - start.Y;
        _userZoomed = true;
    }

    protected override void OnMouseUp(MouseButtonEventArgs e)
    {
        base.OnMouseUp(e);
        if (_dragStart is null)
            return;
        _dragStart = null;
        ReleaseMouseCapture();
    }

    private Point ToImage(Point point) => new(
        (point.X - _translate.X) / _scale.ScaleX,
        (point.Y - _translate.Y) / _scale.ScaleY);
}
