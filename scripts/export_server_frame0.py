#!/usr/bin/env python3
"""
从服务器推理生成的视频中导出第 0 帧 PNG，用于与 Android 对比。
用法: python scripts/export_server_frame0.py data/result_server.mp4 -o data/server_frame0.png
"""

import argparse
import sys

import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="服务器推理输出的 mp4 路径")
    parser.add_argument("-o", "--out", default="server_frame0.png", help="输出 PNG 路径")
    parser.add_argument("-f", "--frame", type=int, default=0, help="帧索引，默认 0")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"错误: 无法打开视频 {args.video}")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"错误: 无法读取第 {args.frame} 帧")
        sys.exit(1)
    cv2.imwrite(args.out, frame)
    print(f"已保存: {args.out}")


if __name__ == "__main__":
    main()
