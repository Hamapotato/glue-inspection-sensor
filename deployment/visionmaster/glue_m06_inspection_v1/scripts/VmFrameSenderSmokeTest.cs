using System;
using System.IO;
using System.Reflection;
using System.Text;

/// <summary>
/// 不启动 VisionMaster，仅验证 VmFrameSender 的二进制组帧结果。
/// </summary>
internal static class VmFrameSenderSmokeTest
{
    private static int Main()
    {
        try
        {
            Type scriptType = typeof(UserScript);
            MethodInfo buildPacket = scriptType.GetMethod(
                "BuildPacket",
                BindingFlags.NonPublic | BindingFlags.Static);
            MethodInfo countValidDefects = scriptType.GetMethod(
                "CountValidDefects",
                BindingFlags.NonPublic | BindingFlags.Static);
            MethodInfo validateFlawCount = scriptType.GetMethod(
                "ValidateFlawCount",
                BindingFlags.NonPublic | BindingFlags.Static);
            MethodInfo acceptOptionalDefectResult = scriptType.GetMethod(
                "AcceptOptionalDefectResult",
                BindingFlags.NonPublic | BindingFlags.Static);
            Type imageInfoType = scriptType.GetNestedType("ImageInfo", BindingFlags.NonPublic);
            if (buildPacket == null || countValidDefects == null ||
                validateFlawCount == null || acceptOptionalDefectResult == null ||
                imageInfoType == null)
                throw new InvalidOperationException("找不到组帧函数或图像信息类型");

            Assert(
                !(bool)acceptOptionalDefectResult.Invoke(
                    null,
                    new object[] { 0, (byte)0, "flawCount" }),
                "读取成功时不应标记缺陷明细缺失");
            Assert(
                (bool)acceptOptionalDefectResult.Invoke(
                    null,
                    new object[] { 7, (byte)0, "flawCount" }),
                "OK 帧缺少缺陷明细时应继续发送");
            AssertThrows(
                () => acceptOptionalDefectResult.Invoke(
                    null,
                    new object[] { 7, (byte)1, "flawCount" }),
                "NG 帧缺少缺陷明细时应失败");

            Assert(
                (int)countValidDefects.Invoke(null, new object[] { new int[] { 4, 0, 5 }, 3 }) == 2,
                "有效缺陷类型计数错误");
            AssertThrows(
                () => countValidDefects.Invoke(null, new object[] { new int[] { 4 }, 2 }),
                "缺陷类型数量超过数组容量时应失败");
            AssertThrows(
                () => validateFlawCount.Invoke(null, new object[] { -1 }),
                "缺陷个数为负数时应失败");
            AssertThrows(
                () => validateFlawCount.Invoke(null, new object[] { 65536 }),
                "缺陷个数超过 uint16 范围时应失败");

            ConstructorInfo imageInfoConstructor = imageInfoType.GetConstructor(
                BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance,
                null,
                new Type[] { typeof(byte[]), typeof(int), typeof(int), typeof(int), typeof(ushort) },
                null);
            if (imageInfoConstructor == null)
                throw new InvalidOperationException("找不到图像信息构造函数");

            object original = imageInfoConstructor.Invoke(
                new object[] { new byte[] { 0x11, 0x22 }, 2, 1, 2, (ushort)1 });
            object caliper = imageInfoConstructor.Invoke(
                new object[] { new byte[] { 0x33 }, 1, 1, 1, (ushort)2 });

            byte[] packet = (byte[])buildPacket.Invoke(
                null,
                new object[]
                {
                    123456789L,
                    1760000000000L,
                    (byte)1,
                    2,
                    new int[] { 4, 5, 0 },
                    3,
                    original,
                    caliper,
                });

            Assert(packet.Length == 79, "整帧长度错误");
            using (MemoryStream stream = new MemoryStream(packet))
            using (BinaryReader reader = new BinaryReader(stream, Encoding.UTF8))
            {
                byte[] magic = reader.ReadBytes(4);
                Assert(Encoding.ASCII.GetString(magic) == "GIFR", "magic 错误");
                Assert(reader.ReadUInt16() == 1, "协议版本错误");
                Assert(reader.ReadUInt16() == 72, "帧头长度错误");
                Assert(reader.ReadUInt32() == 79, "帧长度字段错误");
                Assert(reader.ReadInt64() == 123456789L, "frame_id 错误");
                Assert(reader.ReadInt64() == 1760000000000L, "时间戳错误");
                Assert(reader.ReadByte() == 1, "verdict 错误");
                Assert(reader.ReadByte() == 0, "保留字节错误");
                Assert(reader.ReadUInt16() == 2, "缺陷个数错误");
                Assert(reader.ReadUInt16() == 2, "缺陷类型数量错误");
                Assert(reader.ReadUInt16() == 1, "原图像素格式错误");
                Assert(reader.ReadUInt16() == 2, "卡尺图像素格式错误");
                Assert(reader.ReadUInt32() == 2, "原图宽度错误");
                Assert(reader.ReadUInt32() == 1, "原图高度错误");
                Assert(reader.ReadUInt32() == 2, "原图行跨度错误");
                Assert(reader.ReadUInt32() == 2, "原图长度错误");
                Assert(reader.ReadUInt32() == 1, "卡尺图宽度错误");
                Assert(reader.ReadUInt32() == 1, "卡尺图高度错误");
                Assert(reader.ReadUInt32() == 1, "卡尺图行跨度错误");
                Assert(reader.ReadUInt32() == 1, "卡尺图长度错误");
                Assert(reader.ReadUInt16() == 0, "帧尾保留字段错误");
                Assert(reader.ReadUInt16() == 4, "首个缺陷类型错误");
                Assert(reader.ReadUInt16() == 5, "第二个缺陷类型错误");
                Assert(reader.ReadByte() == 0x11 && reader.ReadByte() == 0x22, "原图载荷错误");
                Assert(reader.ReadByte() == 0x33, "卡尺图载荷错误");
                Assert(stream.Position == stream.Length, "帧内有未读取数据");
            }

            Console.WriteLine("VmFrameSender protocol smoke test passed.");
            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine("VmFrameSender smoke test failed: " + ex);
            return 1;
        }
    }

    private static void Assert(bool condition, string message)
    {
        if (!condition)
            throw new InvalidOperationException(message);
    }

    private static void AssertThrows(Action action, string message)
    {
        try
        {
            action();
        }
        catch (TargetInvocationException)
        {
            return;
        }

        throw new InvalidOperationException(message);
    }
}
