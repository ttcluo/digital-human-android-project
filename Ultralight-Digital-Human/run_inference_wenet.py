import os
import subprocess
import argparse


ROOT = "/data/luochuan/digital-human-android-project"
ULTRA = os.path.join(ROOT, "Ultralight-Digital-Human")


def run(cmd: str, cwd: str | None = None) -> None:
    print(f">> {cmd}")
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="一键运行 Ultralight WeNet 推理（特征提取 + 视频生成 + 合成音频）"
    )
    parser.add_argument(
        "--audio",
        required=True,
        help="原始音频路径（mp3 或 wav，任意采样率，推荐 16kHz 单声道）",
    )
    parser.add_argument(
        "--dataset",
        default=os.path.join(ROOT, "data/raw"),
        help="训练时使用的数据目录，需包含 full_body_img/ 与 landmarks/",
    )
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(ULTRA, "checkpoint_wenet/195.pth"),
        help="U-Net 训练好的权重文件路径",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(ROOT, "data/result_preview_with_audio.mp4"),
        help="最终带音频输出视频路径（.mp4）",
    )
    args = parser.parse_args()

    audio_path = os.path.abspath(args.audio)
    dataset_dir = os.path.abspath(args.dataset)
    checkpoint = os.path.abspath(args.checkpoint)
    out_video = os.path.abspath(args.out)

    # 1. 转 16kHz 单声道 wav
    base, _ = os.path.splitext(audio_path)
    wav_path = base + ".wav"
    run(
        f'ffmpeg -y -i "{audio_path}" -ar 16000 -ac 1 "{wav_path}"'
    )

    # 2. 提取 WeNet 特征（会生成 *_wenet.npy）
    audio_npy = base + "_wenet.npy"
    data_utils_dir = os.path.join(ULTRA, "data_utils")
    run(
        f'python wenet_infer.py "{wav_path}"',
        cwd=data_utils_dir,
    )

    # 3. 使用 inference.py 生成无声视频
    tmp_noaudio = os.path.splitext(out_video)[0] + "_noaudio.mp4"
    run(
        "python inference.py "
        f'--asr wenet --dataset "{dataset_dir}" '
        f'--audio_feat "{audio_npy}" '
        f'--save_path "{tmp_noaudio}" '
        f'--checkpoint "{checkpoint}"',
        cwd=ULTRA,
    )

    # 4. 使用 ffmpeg 合成音频
    run(
        f'ffmpeg -y -i "{tmp_noaudio}" -i "{wav_path}" '
        f'-c:v libx264 -c:a aac "{out_video}"'
    )


if __name__ == "__main__":
    main()

