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
        description="一键生成 Ultralight vs OnDevice U-Net WeNet 推理左右对比视频"
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
        "--ultra_ckpt",
        default=os.path.join(ULTRA, "checkpoint_wenet/195.pth"),
        help="原始 Ultralight U-Net 权重路径",
    )
    parser.add_argument(
        "--ondevice_ckpt",
        default=os.path.join(ULTRA, "checkpoint_ondevice/105.pth"),
        help="OnDevice U-Net 权重路径",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(ROOT, "data/result_compare_ultra_vs_ondevice.mp4"),
        help="最终左右对比输出视频路径（.mp4）",
    )
    args = parser.parse_args()

    audio_path = os.path.abspath(args.audio)
    dataset_dir = os.path.abspath(args.dataset)
    ultra_ckpt = os.path.abspath(args.ultra_ckpt)
    ondevice_ckpt = os.path.abspath(args.ondevice_ckpt)
    out_compare = os.path.abspath(args.out)

    # 1. 转 16kHz 单声道 wav
    base, _ = os.path.splitext(audio_path)
    wav_path = base + ".wav"
    run(f'ffmpeg -y -i "{audio_path}" -ar 16000 -ac 1 "{wav_path}"')

    # 2. 提取 WeNet 特征
    audio_npy = base + "_wenet.npy"
    data_utils_dir = os.path.join(ULTRA, "data_utils")
    run(f'python wenet_infer.py "{wav_path}"', cwd=data_utils_dir)

    # 3. 生成原始 Ultralight U-Net 版本（无声+合成）
    ultra_video = os.path.join(ROOT, "data/result_ultralight_195.mp4")
    run(
        "python inference.py "
        f'--asr wenet --dataset "{dataset_dir}" '
        f'--audio_feat "{audio_npy}" '
        f'--save_path "{ultra_video}" '
        f'--checkpoint "{ultra_ckpt}" '
        f'--unet ultralight',
        cwd=ULTRA,
    )

    # 4. 生成 OnDevice U-Net 版本
    ondevice_video = os.path.join(ROOT, "data/result_ondevice_105.mp4")
    run(
        "python inference.py "
        f'--asr wenet --dataset "{dataset_dir}" '
        f'--audio_feat "{audio_npy}" '
        f'--save_path "{ondevice_video}" '
        f'--checkpoint "{ondevice_ckpt}" '
        f'--unet ondevice',
        cwd=ULTRA,
    )

    # 5. 左右拼接对比（共用原始音频）
    # 注意：假设两段视频分辨率一致（由 inference.py 保证）
    run(
        'ffmpeg -y '
        f'-i "{ultra_video}" '
        f'-i "{ondevice_video}" '
        '-filter_complex "[0:v][1:v]hstack=inputs=2[v]" '
        f'-map "[v]" -map 0:a "{out_compare}"'
    )

    print(f"[INFO] 对比视频已生成: {out_compare}")


if __name__ == "__main__":
    main()

