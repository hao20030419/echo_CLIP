import argparse
import csv
import random
from pathlib import Path

import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg"}


def collect_videos(folder: Path):
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in VIDEO_EXTS])


def safe_stem(video_path: Path):
    return video_path.stem.replace(" ", "_")


def sample_frame_indices(total_frames, frame_stride=1, max_frames=None, target_fps=None, src_fps=None):
    if total_frames <= 0:
        return []

    stride = max(1, int(frame_stride))
    if target_fps is not None and src_fps is not None and src_fps > 0:
        stride = max(stride, int(round(src_fps / float(target_fps))))

    indices = list(range(0, total_frames, stride))
    if max_frames is not None and max_frames > 0:
        indices = indices[: max_frames]
    return indices


def extract_frames_for_video(
    video_path: Path,
    label_name: str,
    split_name: str,
    output_root: Path,
    frame_stride: int,
    max_frames_per_video,
    target_fps,
    resize,
    jpeg_quality,
):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], f"Cannot open video: {video_path}"

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_indices = sample_frame_indices(
        total_frames,
        frame_stride=frame_stride,
        max_frames=max_frames_per_video,
        target_fps=target_fps,
        src_fps=src_fps,
    )

    split_label_dir = output_root / split_name / label_name
    split_label_dir.mkdir(parents=True, exist_ok=True)

    records = []
    video_key = safe_stem(video_path)

    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue

        if resize is not None:
            frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)

        out_name = f"{video_key}_f{frame_idx:06d}.jpg"
        out_path = split_label_dir / out_name
        cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])

        records.append(
            {
                "split": split_name,
                "label": label_name,
                "video_path": str(video_path),
                "frame_index": int(frame_idx),
                "frame_path": str(out_path),
            }
        )

    cap.release()
    return records, None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract frame dataset from positive/negative echo videos for ResNet training."
    )
    parser.add_argument("--data-root", type=str, default="data", help="Folder containing positive/negative.")
    parser.add_argument("--positive-dir", type=str, default="positive", help="Subfolder name for positive videos.")
    parser.add_argument("--negative-dir", type=str, default="negative", help="Subfolder name for negative videos.")
    parser.add_argument("--out-dir", type=str, default="data_frames", help="Output frame dataset root.")

    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--random-seed", type=int, default=42)

    parser.add_argument("--frame-stride", type=int, default=2, help="Take every Nth frame.")
    parser.add_argument("--max-frames-per-video", type=int, default=0, help="0 means unlimited.")
    parser.add_argument("--target-fps", type=float, default=0.0, help="0 means disabled.")
    parser.add_argument("--resize-width", type=int, default=224)
    parser.add_argument("--resize-height", type=int, default=224)
    parser.add_argument("--jpeg-quality", type=int, default=95)

    return parser.parse_args()


def stratified_video_split(pos_videos, neg_videos, train_ratio, val_ratio, test_ratio, random_seed):
    ratios_sum = train_ratio + val_ratio + test_ratio
    if abs(ratios_sum - 1.0) > 1e-6:
        raise ValueError("train/val/test ratio must sum to 1.0")

    all_videos = pos_videos + neg_videos
    all_labels = [1] * len(pos_videos) + [0] * len(neg_videos)

    if len(all_videos) == 0:
        return [], [], [], [], [], []

    if val_ratio == 0 and test_ratio == 0:
        return all_videos, [], [], all_labels, [], []

    temp_ratio = val_ratio + test_ratio
    train_videos, temp_videos, train_labels, temp_labels = train_test_split(
        all_videos,
        all_labels,
        test_size=temp_ratio,
        random_state=random_seed,
        stratify=all_labels if len(set(all_labels)) > 1 else None,
    )

    if len(temp_videos) == 0:
        return train_videos, [], [], train_labels, [], []

    if val_ratio == 0:
        return train_videos, [], temp_videos, train_labels, [], temp_labels
    if test_ratio == 0:
        return train_videos, temp_videos, [], train_labels, temp_labels, []

    val_fraction_in_temp = val_ratio / (val_ratio + test_ratio)
    val_videos, test_videos, val_labels, test_labels = train_test_split(
        temp_videos,
        temp_labels,
        test_size=(1 - val_fraction_in_temp),
        random_state=random_seed,
        stratify=temp_labels if len(set(temp_labels)) > 1 else None,
    )

    return train_videos, val_videos, test_videos, train_labels, val_labels, test_labels


def write_metadata_csv(rows, csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "label", "video_path", "frame_index", "frame_path"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    random.seed(args.random_seed)

    data_root = Path(args.data_root)
    pos_root = data_root / args.positive_dir
    neg_root = data_root / args.negative_dir
    out_root = Path(args.out_dir)

    if not pos_root.exists() or not neg_root.exists():
        raise FileNotFoundError(
            f"Expecting class folders at: {pos_root} and {neg_root}"
        )

    pos_videos = collect_videos(pos_root)
    neg_videos = collect_videos(neg_root)

    if len(pos_videos) == 0 and len(neg_videos) == 0:
        raise RuntimeError("No video files found under positive/negative folders")

    print(f"Found videos: positive={len(pos_videos)} negative={len(neg_videos)}")

    max_frames_per_video = args.max_frames_per_video if args.max_frames_per_video > 0 else None
    target_fps = args.target_fps if args.target_fps > 0 else None
    resize = (args.resize_width, args.resize_height) if args.resize_width > 0 and args.resize_height > 0 else None

    train_videos, val_videos, test_videos, _, _, _ = stratified_video_split(
        pos_videos,
        neg_videos,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.random_seed,
    )

    split_to_videos = {
        "train": train_videos,
        "val": val_videos,
        "test": test_videos,
    }

    all_rows = []
    errors = []

    for split_name, videos in split_to_videos.items():
        if len(videos) == 0:
            continue

        print(f"Extracting split={split_name}, videos={len(videos)}")
        for video_path in tqdm(videos, desc=f"{split_name} videos"):
            label_name = "positive" if "positive" in str(video_path).replace("\\", "/") else "negative"
            rows, err = extract_frames_for_video(
                video_path=video_path,
                label_name=label_name,
                split_name=split_name,
                output_root=out_root,
                frame_stride=args.frame_stride,
                max_frames_per_video=max_frames_per_video,
                target_fps=target_fps,
                resize=resize,
                jpeg_quality=args.jpeg_quality,
            )

            if err is not None:
                errors.append(err)
                continue
            all_rows.extend(rows)

    metadata_path = out_root / "frames_metadata.csv"
    write_metadata_csv(all_rows, metadata_path)

    print(f"Done. Total frames: {len(all_rows)}")
    print(f"Metadata CSV: {metadata_path}")
    print("Dataset layout example:")
    print(f"  {out_root}/train/positive/*.jpg")
    print(f"  {out_root}/train/negative/*.jpg")
    print(f"  {out_root}/val/positive/*.jpg")
    print(f"  {out_root}/test/negative/*.jpg")

    if errors:
        print(f"Warnings: {len(errors)} videos failed")
        for msg in errors[:20]:
            print(f" - {msg}")


if __name__ == "__main__":
    main()
