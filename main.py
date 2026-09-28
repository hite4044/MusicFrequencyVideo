""" 音乐频谱生成器 python main.py "Special Ability - Empty.flac" -o result.mp4 """
import argparse
import os
import subprocess
import sys
from io import BytesIO

import numpy as np
import soundfile as sf
from PIL import Image
from scipy.fft import rfft


def generate_spectrum_video(
    audio_path,
    output_path,
    bar_count=100,
    bar_width=10,
    bar_gap=2,
    color=(0, 255, 255),        # RGB: Cyan
    video_height=400,
    fps=60,
    bitrate='8M',
    codec='h264_nvenc',          # [新增] 视频编码器
    scale=1,                     # [新增] 输出视频缩放倍数 (仅整数)
    use_jpeg=True,               # [新增] 是否使用 jpeg 帧序列传输给 ffmpeg
    gain=1.0,
    smooth_factor=0.3,           # 条形高度平滑系数
    range_smooth_factor=0.05,    # 动态范围平滑系数 (越小，范围变化越慢，对比度越强，延迟越高)
    audio=False
):
    """生成横向长条频谱视频，支持动态范围调整、音量调整和平滑效果"""

    # [新增] 缩放仅允许整数
    if not isinstance(scale, int) or scale < 1:
        print("错误：视频缩放倍数必须是不小于 1 的整数。")
        sys.exit(1)

    # 1. 读取音频数据
    print(f"正在读取音频文件：{audio_path} ...")
    data, sample_rate = sf.read(audio_path)

    if len(data.shape) > 1:
        data = data.mean(axis=1)

    if gain != 1.0:
        print(f"应用音量增益系数：{gain}")
        data = data * gain

    total_samples = len(data)
    duration = total_samples / sample_rate
    total_frames = int(duration * fps)
    print(f"音频时长：{duration:.2f}s, 总帧数：{total_frames}")

    # 2. 计算视频宽度
    frame_width = bar_count * (bar_width + bar_gap) - bar_gap

    # [新增] 输出尺寸 = 渲染尺寸 * 整数缩放
    out_width = frame_width * scale
    out_height = video_height * scale
    if scale != 1:
        print(f"输出视频缩放：{scale}x (渲染 {frame_width}x{video_height} -> 输出 {out_width}x{out_height})")

    # 3. 初始化 FFmpeg 进程
    # [新增] 输入端根据传输模式选择：rawvideo 或 jpeg 帧序列 (image2pipe)
    if use_jpeg:
        input_args = [
            '-f', 'image2pipe',
            '-vcodec', 'mjpeg',
            '-r', str(fps),
            '-i', '-'
        ]
        print("帧传输模式：JPEG 帧序列 (image2pipe)")
    else:
        input_args = [
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{frame_width}x{video_height}',
            '-pix_fmt', 'rgb24',
            '-r', str(fps),
            '-i', '-'
        ]
        print("帧传输模式：RGB rawvideo")

    ffmpeg_cmd = [
        'ffmpeg',
        '-hide_banner',
        '-y',
        *input_args,
        '-vf', f'scale={out_width}:{out_height}',   # [新增] 整数缩放输出
        '-c:v', codec,                              # [修改] 编码器可指定
        '-b:v', bitrate,
        '-pix_fmt', 'yuv420p',
        output_path
    ]

    print(f"正在启动 FFmpeg 编码进程 (编码器: {codec}) ...")
    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    # 4. 预分配帧缓冲区
    frame_buffer = np.zeros((video_height, frame_width, 3), dtype=np.uint8)

    bar_indices = []
    for i in range(bar_count):
        start_x = i * (bar_width + bar_gap)
        end_x = start_x + bar_width
        bar_indices.append((start_x, end_x))

    color_array = np.array(color, dtype=np.uint8)
    samples_per_frame = sample_rate // fps
    fft_size = samples_per_frame

    # 初始化平滑所需的历史高度数组
    prev_heights = np.zeros(bar_count, dtype=float)

    # [新增] 初始化动态范围的平滑追踪变量
    # 初始值设为一个较宽的范围，避免第一帧异常
    running_min_db = -100.0
    running_max_db = -10.0

    sample_positions = np.linspace(0, 1, bar_count)

    print("开始处理帧并推送到 FFmpeg (动态范围模式)...")
    try:
        for frame_idx in range(total_frames):
            start_sample = frame_idx * samples_per_frame
            end_sample = start_sample + fft_size

            if end_sample > total_samples:
                break

            chunk = data[start_sample:end_sample]

            # 汉宁窗
            window = np.hanning(len(chunk))
            chunk_windowed = chunk * window

            # FFT
            spectrum = rfft(chunk_windowed)
            magnitude = np.abs(spectrum)

            # 对数转换
            magnitude_db = 20 * np.log10(magnitude + 1e-6)

            # --- [核心修改] 动态计算当前帧的统计范围 ---
            valid_len = len(magnitude_db) // 2
            valid_magnitude_db = magnitude_db[:valid_len]

            current_max_db = np.percentile(valid_magnitude_db, 95)
            current_min_db = np.percentile(valid_magnitude_db, 5)

            if current_max_db - current_min_db < 1.0:
                current_max_db = current_min_db + 10.0

            # --- 应用范围平滑 (延迟效果) ---
            running_max_db = running_max_db * (1 - range_smooth_factor) + current_max_db * range_smooth_factor
            running_min_db = running_min_db * (1 - range_smooth_factor) + current_min_db * range_smooth_factor

            if running_max_db <= running_min_db:
                running_max_db = running_min_db + 10.0

            # 动态归一化
            magnitude_norm = np.clip(
                (valid_magnitude_db - running_min_db) / (running_max_db - running_min_db),
                0, 1
            )

            # 线性采样获取目标高度比例
            bar_heights_ratio = np.interp(
                sample_positions * (valid_len - 1),
                np.arange(valid_len),
                magnitude_norm
            )

            # 计算目标像素高度 (预留底部 10px)
            target_heights = bar_heights_ratio * (video_height - 10)

            # --- 条形高度平滑处理 ---
            current_heights = prev_heights * (1 - smooth_factor) + target_heights * smooth_factor
            current_heights = np.clip(current_heights.astype(int), 0, video_height)

            prev_heights = current_heights

            # --- 绘图 ---
            frame_buffer[:] = 0
            for i, (start_x, end_x) in enumerate(bar_indices):
                h = current_heights[i]
                if h > 0:
                    y_start = video_height - h
                    y_end = video_height
                    if y_start < 0:
                        y_start = 0
                    frame_buffer[y_start:y_end, start_x:end_x] = color_array

            # --- 写入管道 ---
            if use_jpeg:
                # [新增] 使用PIL编码为 jpeg 后写入
                img_buf = BytesIO()
                Image.fromarray(frame_buffer).save(img_buf, format='jpeg', quality=100)
                process.stdin.write(img_buf.getvalue())
            else:
                raw_bytes = np.ascontiguousarray(frame_buffer).tobytes()
                process.stdin.write(raw_bytes)

        print("\n处理完成，等待 FFmpeg 封装...")

    except Exception as e:
        print(f"发生错误：{e}")
        import traceback
        traceback.print_exc()
    finally:
        if process.stdin:
            process.stdin.close()
        process.wait()

    if not audio:
        return

    # 混流音频
    ffmpeg_cmd = [
        'ffmpeg',
        '-y',
        '-i', f'{output_path}',
        '-i', f'{audio_path}',
        '-map', '0:v',
        '-map', '1:a',
        '-c:a', 'copy',
        '-c:v', 'copy',
        output_path + ".tmp.mp4"
    ]
    print("正在混流...")
    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    process.wait()

    if os.path.exists(output_path + ".tmp.mp4"):
        if os.path.exists(output_path):
            os.remove(output_path)
        os.rename(output_path + ".tmp.mp4", output_path)
        print(f"视频已保存至：{output_path}")
    else:
        print("混流失败，检查输出文件。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="生成音频频谱视频 (动态范围 + 平滑 + 音量控制 + 编码器/缩放/jpeg帧序列)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ============ 可视化设置 ============
    visual_group = parser.add_argument_group('可视化设置')
    visual_group.add_argument("-b", "--bars", type=int, default=20, help="频谱条数量")
    visual_group.add_argument("-w", "--width", type=int, default=40, help="单个条形宽度 (像素)")
    visual_group.add_argument("-g", "--gap", type=int, default=10, help="条形间距 (像素)")
    visual_group.add_argument("--smooth", type=float, default=0.2,
                              help="条形高度平滑系数 (0.1-0.9, 频谱条本身的高度缩放)")
    visual_group.add_argument("--range-smooth", type=float, default=0.1,
                              help="动态缩放范围平滑系数 (0.01-0.2, 频谱条高度根据时间的缩放, 越小对比度越强且延迟越高)")

    # ============ 外观设置 ============
    appearance_group = parser.add_argument_group('外观设置')
    appearance_group.add_argument("-c", "--color", type=str, default="255,255,255", help="条形颜色 (R,G,B)")
    appearance_group.add_argument("-H", "--height", type=int, default=400, help="视频高度 (像素)")

    # ============ 编码设置 ============
    encoding_group = parser.add_argument_group('编码设置')
    encoding_group.add_argument("-e", "--codec", type=str, default="h264_nvenc",
                                help="输出视频编码器名称 (手动指定，如 h264_nvenc / hevc_nvenc / libx264)")
    encoding_group.add_argument("-r", "--bitrate", type=str, default="8M", help="编码码率")
    encoding_group.add_argument("-f", "--fps", type=int, default=30, help="视频帧率")
    encoding_group.add_argument("-s", "--scale", type=int, default=1, help="输出视频缩放倍数 (仅整数，>=1)")
    encoding_group.add_argument("-j", "--no-jpeg", dest="use_jpeg", action="store_false",
                                help="禁用 jpeg 帧序列传输 (默认开启，使用 rawvideo)")
    encoding_group.add_argument("-G", "--gain", type=float, default=0.1, help="音量增益系数")
    encoding_group.add_argument("-a", "--audio", action="store_true", help="输出包含音频")

    parser.add_argument("input", help="输入音频文件路径")
    parser.add_argument("-o", "--output", default="output_spectrum.mp4", help="输出视频路径")

    args = parser.parse_args()

    try:
        r, g, b = map(int, args.color.split(','))
        color_tuple = (r, g, b)
    except ValueError:
        print("颜色格式错误，请使用 R,G,B 格式 (例如：0,255,255)")
        sys.exit(1)

    generate_spectrum_video(
        audio_path=args.input,
        output_path=args.output,
        bar_count=args.bars,
        bar_width=args.width,
        bar_gap=args.gap,
        color=color_tuple,
        video_height=args.height,
        fps=args.fps,
        bitrate=args.bitrate,
        codec=args.codec,
        scale=args.scale,
        use_jpeg=args.use_jpeg,
        gain=args.gain,
        smooth_factor=args.smooth,
        range_smooth_factor=args.range_smooth,
        audio=args.audio
    )
