using System.IO;
using System.Globalization;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Text.RegularExpressions;

namespace GlueInspect.VmViewer.Services;

public sealed record TcpState(string Text, bool Connected, bool Error);

/// <summary>
/// VM 发送的一条检测结果。支持旧格式 OK/NG，以及带记录/图片序号的新格式。
/// </summary>
public sealed record ResultMessage(
    string Verdict,
    string? RecordSequence,
    int? OriginalSequence,
    int? CaliperSequence,
    DateTime? SourceTimestamp,
    string Raw);

/// <summary>
/// 接收 VM 发送的 OK/NG。默认由前端监听，VM 使用 TCP 客户端连接；
/// 端口已被其他进程监听时自动改为客户端模式，兼容 VM 作为 TCP 服务端的配置。
/// </summary>
public sealed partial class TcpResultReceiver : IDisposable
{
    private readonly CancellationTokenSource _cts = new();
    private TcpListener? _listener;
    private Task? _backgroundTask;
    private bool _disposed;

    public event Action<ResultMessage>? ResultReceived;
    public event Action<TcpState>? StateChanged;

    public Task StartAsync(string host, int port, bool preferListen)
    {
        _backgroundTask = preferListen
            ? RunListenerAsync(host, port, _cts.Token)
            : RunClientAsync(host, port, _cts.Token);
        return Task.CompletedTask;
    }

    public static ResultMessage? ParseMessage(string text)
    {
        var value = text.Trim().Trim('\0', '\uFEFF');
        var matches = StatusRegex().Matches(value);
        if (matches.Count == 0)
            return null;

        var statusMatch = matches[^1];
        var verdict = statusMatch.Groups[1].Value.ToUpperInvariant();
        var prefix = value[..statusMatch.Index].Trim(' ', '\t', '\r', '\n', '|', ',', ';', ':');
        var parts = prefix.Split(
            ['|', ',', ';'],
            StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

        string? recordSequence = null;
        int? originalSequence = null;
        int? caliperSequence = null;
        DateTime? sourceTimestamp = null;
        if (parts.Length == 1)
        {
            recordSequence = CleanToken(parts[0]);
            sourceTimestamp = ParseTimestamp(recordSequence);
        }
        else if (parts.Length == 2)
        {
            originalSequence = ParseSequence(parts[0]);
            caliperSequence = ParseSequence(parts[1]);
            recordSequence = $"{CleanToken(parts[0])}-{CleanToken(parts[1])}";
        }
        else if (parts.Length >= 3)
        {
            recordSequence = CleanToken(parts[0]);
            originalSequence = ParseSequence(parts[1]);
            caliperSequence = ParseSequence(parts[2]);
        }

        return new ResultMessage(
            verdict,
            recordSequence,
            originalSequence,
            caliperSequence,
            sourceTimestamp,
            value);
    }

    private async Task RunListenerAsync(string host, int port, CancellationToken ct)
    {
        try
        {
            var address = IPAddress.Parse(host);
            _listener = new TcpListener(address, port);
            _listener.Start();
            StateChanged?.Invoke(new TcpState($"监听中 · {host}:{port}", false, false));

            while (!ct.IsCancellationRequested)
            {
                var client = await _listener.AcceptTcpClientAsync(ct);
                var remote = client.Client.RemoteEndPoint?.ToString() ?? "VM";
                StateChanged?.Invoke(new TcpState($"已连接 · {remote}", true, false));
                await ReadClientAsync(client, ct);
                StateChanged?.Invoke(new TcpState($"监听中 · {host}:{port}", false, false));
            }
        }
        catch (SocketException ex) when (ex.SocketErrorCode == SocketError.AddressAlreadyInUse)
        {
            // 兼容 VM 已配置为 TCP 服务端的情况。
            await RunClientAsync(host, port, ct);
        }
        catch (OperationCanceledException)
        {
        }
        catch (Exception ex)
        {
            StateChanged?.Invoke(new TcpState($"接收失败：{ex.Message}", false, true));
        }
    }

    private async Task RunClientAsync(string host, int port, CancellationToken ct)
    {
        while (!ct.IsCancellationRequested)
        {
            try
            {
                using var client = new TcpClient { NoDelay = true };
                await client.ConnectAsync(host, port, ct);
                StateChanged?.Invoke(new TcpState($"已连接 · {host}:{port}", true, false));
                await ReadClientAsync(client, ct);
            }
            catch (OperationCanceledException)
            {
                return;
            }
            catch (Exception ex) when (ex is SocketException or IOException)
            {
                StateChanged?.Invoke(new TcpState($"等待 VM 服务 · {host}:{port}", false, false));
                try
                {
                    await Task.Delay(1000, ct);
                }
                catch (OperationCanceledException)
                {
                    return;
                }
            }
        }
    }

    private async Task ReadClientAsync(TcpClient client, CancellationToken ct)
    {
        using (client)
        {
            client.NoDelay = true;
            await using var stream = client.GetStream();
            var buffer = new byte[2048];
            var pending = new StringBuilder();
            while (!ct.IsCancellationRequested)
            {
                var count = await stream.ReadAsync(buffer, ct);
                if (count == 0)
                    break;
                pending.Append(Encoding.UTF8.GetString(buffer, 0, count));
                EmitCompleteMessages(pending);
            }
            if (pending.Length > 0 && ParseMessage(pending.ToString()) is { } finalMessage)
                ResultReceived?.Invoke(finalMessage);
        }
    }

    private void EmitCompleteMessages(StringBuilder pending)
    {
        while (true)
        {
            var text = pending.ToString();
            var separatorIndex = text.IndexOfAny(['\r', '\n']);
            if (separatorIndex >= 0)
            {
                var line = text[..separatorIndex];
                pending.Remove(0, separatorIndex + 1);
                if (ParseMessage(line) is { } lineMessage)
                    ResultReceived?.Invoke(lineMessage);
                continue;
            }

            // VM 可以只发送 OK/NG 而不带换行；收到完整状态后立即处理。
            if (ParseMessage(text) is { } immediateMessage)
            {
                pending.Clear();
                ResultReceived?.Invoke(immediateMessage);
            }
            return;
        }
    }

    private static string? CleanToken(string value)
    {
        var token = value.Trim();
        var labelSeparator = token.LastIndexOfAny(['=', ':']);
        if (labelSeparator >= 0 && labelSeparator < token.Length - 1)
            token = token[(labelSeparator + 1)..].Trim();
        return string.IsNullOrWhiteSpace(token) ? null : token;
    }

    private static int? ParseSequence(string value)
    {
        var token = CleanToken(value);
        return int.TryParse(token, out var sequence) ? sequence : null;
    }

    private static DateTime? ParseTimestamp(string? value)
    {
        if (value is null)
            return null;
        return DateTime.TryParseExact(
            value,
            "yyyyMMddHHmmssfff",
            CultureInfo.InvariantCulture,
            DateTimeStyles.AssumeLocal,
            out var timestamp)
            ? timestamp
            : null;
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
            _backgroundTask?.Wait(TimeSpan.FromMilliseconds(300));
        }
        catch (AggregateException)
        {
        }
        _cts.Dispose();
    }

    [GeneratedRegex(@"(?i)\b(OK|NG)\b")]
    private static partial Regex StatusRegex();
}
