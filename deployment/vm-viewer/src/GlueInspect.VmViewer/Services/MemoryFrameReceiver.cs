using System.IO;
using System.Net;
using System.Net.Sockets;
using GlueInspect.VmViewer.Models;

namespace GlueInspect.VmViewer.Services;

/// <summary>
/// 监听 VisionMaster TCP 客户端，并按 GIFR v1 协议连续拆出内存图像帧。
/// 协议见 docs/algo/visionmaster-memory-frame.md。
/// </summary>
public sealed class MemoryFrameReceiver : IDisposable
{
    private readonly CancellationTokenSource _cts = new();
    private TcpListener? _listener;
    private Task? _backgroundTask;
    private ulong _lastFrameId;
    private bool _disposed;

    /// <summary>处理帧的回调会在读取下一帧前完成，用 TCP 背压避免静默丢帧。</summary>
    public event Func<MemoryFrame, Task>? FrameReceived;

    public event Action<TcpState>? StateChanged;

    public Task StartAsync(string host, int port)
    {
        _backgroundTask = RunListenerAsync(host, port, _cts.Token);
        return Task.CompletedTask;
    }

    private async Task RunListenerAsync(string host, int port, CancellationToken cancellationToken)
    {
        try
        {
            var address = IPAddress.Parse(host);
            _listener = new TcpListener(address, port);
            _listener.Start();
            StateChanged?.Invoke(new TcpState($"监听中 · {host}:{port}", false, false));

            while (!cancellationToken.IsCancellationRequested)
            {
                using var client = await _listener.AcceptTcpClientAsync(cancellationToken);
                client.NoDelay = true;
                var remote = client.Client.RemoteEndPoint?.ToString() ?? "VisionMaster";
                StateChanged?.Invoke(new TcpState($"已连接 · {remote}", true, false));
                try
                {
                    await ReadClientAsync(client, cancellationToken);
                }
                catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
                {
                    return;
                }
                catch (Exception ex) when (ex is InvalidDataException or EndOfStreamException)
                {
                    StateChanged?.Invoke(new TcpState($"帧格式错误：{ex.Message}", false, true));
                }
                catch (Exception ex) when (ex is IOException or SocketException)
                {
                    StateChanged?.Invoke(new TcpState($"连接中断：{ex.Message}", false, true));
                }

                StateChanged?.Invoke(new TcpState(
                    $"监听中 · {host}:{port}",
                    false,
                    false));
            }
        }
        catch (OperationCanceledException)
        {
        }
        catch (Exception ex)
        {
            StateChanged?.Invoke(new TcpState($"接收失败：{ex.Message}", false, true));
        }
    }

    private async Task ReadClientAsync(TcpClient client, CancellationToken cancellationToken)
    {
        await using var stream = client.GetStream();
        while (!cancellationToken.IsCancellationRequested)
        {
            var frame = await MemoryFrameProtocol.ReadAsync(
                stream,
                MemoryFrameProtocol.DefaultMaxFrameBytes,
                cancellationToken);
            if (frame is null)
                return;
            if (frame.FrameId <= _lastFrameId)
            {
                throw new InvalidDataException(
                    $"帧号未递增：收到 {frame.FrameId}，上一帧 {_lastFrameId}");
            }

            _lastFrameId = frame.FrameId;
            var handler = FrameReceived;
            if (handler is not null)
                await handler(frame);
        }
    }

    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
        _cts.Cancel();
        _listener?.Stop();
        try
        {
            _backgroundTask?.Wait(TimeSpan.FromMilliseconds(500));
        }
        catch (AggregateException)
        {
        }
        _cts.Dispose();
    }
}
