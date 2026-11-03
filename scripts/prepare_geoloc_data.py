"""复用并检查 Geoloc 预处理数据；不完整时停止，避免覆盖已有产物。"""

import argparse
import csv
from pathlib import Path
import subprocess
import sys


def validate_splits(data_root):
    """只检查图片配对和文件完整性，保留 CSV 原有顺序及重复行。"""
    split_ids = []
    for name in ("train_subarea5.csv", "val_subarea5.csv"):
        rows = 0
        ids = set()
        with (data_root / name).open(encoding="utf-8", newline="") as handle:
            for number, pair in enumerate(csv.reader(handle), start=1):
                if len(pair) != 2:
                    raise ValueError(f"{name}:{number} 应包含两列图片路径")
                paths = []
                for relative, directory in zip(pair, ("01_Satellite", "00_SVI_aligned")):
                    rel = Path(relative)
                    path = (data_root / rel).resolve()
                    if (rel.is_absolute() or not rel.parts or rel.parts[0] != directory
                            or not path.is_relative_to(data_root)):
                        raise ValueError(f"{name}:{number} 图片路径不符合预期: {relative}")
                    if not path.is_file() or path.stat().st_size == 0:
                        raise ValueError(f"{name}:{number} 图片缺失或为空: {path}")
                    paths.append(path)
                if (not paths[0].stem.isdecimal() or paths[0].suffix.lower() != ".png"
                        or paths[0].name != paths[1].name
                        or paths[0].parent.name != paths[1].parent.name):
                    raise ValueError(f"{name}:{number} 图片 ID、类别或格式不匹配")
                ids.add(int(paths[0].stem))
                rows += 1
        if rows == 0:
            raise ValueError(f"{name} 没有样本")
        print(f"GEOLOC_SPLIT_OK {name} rows={rows} unique_ids={len(ids)}", flush=True)
        split_ids.append(ids)
    if split_ids[0] & split_ids[1]:
        raise ValueError("训练集与验证集包含相同图片 ID，请检查划分")


def prepare(data_root, preprocess_script):
    data_root = Path(data_root).resolve()
    artifacts = [data_root / name for name in
                 ("train_subarea5.csv", "val_subarea5.csv", "00_SVI_aligned")]
    if not data_root.is_dir():
        raise ValueError(f"数据目录不存在: {data_root}")
    if not any(path.exists() for path in artifacts):
        script = Path(preprocess_script).resolve()
        print("GEOLOC_PREPROCESS_START", flush=True)
        subprocess.run([sys.executable, "-u", str(script)], cwd=script.parent, check=True)
    elif not all(path.exists() for path in artifacts):
        raise ValueError("已有预处理产物不完整；停止运行，不自动覆盖。请检查两份 CSV 和 00_SVI_aligned")
    else:
        print("GEOLOC_REUSE_PREPROCESSED_DATA：复用现有划分及对齐图片", flush=True)
    validate_splits(data_root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--preprocess-script", required=True)
    args = parser.parse_args()
    prepare(args.data_root, args.preprocess_script)
