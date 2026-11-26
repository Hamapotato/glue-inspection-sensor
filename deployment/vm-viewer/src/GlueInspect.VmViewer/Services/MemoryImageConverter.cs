using System.IO;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using GlueInspect.VmViewer.Models;

namespace GlueInspect.VmViewer.Services;

/// <summary>把协议中的 Mono8/RGB24 原始像素转换为可冻结的 WPF 位图。</summary>
public static class MemoryImageConverter
{
    public static BitmapSource ToBitmapSource(MemoryImage image)
    {
        var format = image.PixelFormatCode switch
        {
            1 => PixelFormats.Gray8,
            2 => PixelFormats.Rgb24,
            _ => throw new InvalidDataException(
                $"不支持的像素格式编码：{image.PixelFormatCode}"),
        };
        var expectedLength = checked(image.Stride * image.Height);
        if (image.Pixels.Length != expectedLength)
        {
            throw new InvalidDataException(
                $"像素缓冲长度 {image.Pixels.Length} 与行跨度乘高度 {expectedLength} 不一致");
        }

        var bitmap = BitmapSource.Create(
            image.Width,
            image.Height,
            96,
            96,
            format,
            null,
            image.Pixels,
            image.Stride);
        bitmap.Freeze();
        return bitmap;
    }
}
