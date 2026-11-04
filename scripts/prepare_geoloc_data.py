"""复用并检查 Geoloc 预处理数据；不完整时停止，避免覆盖已有产物。"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def validate_splits(data_root):
    """核对随机 6:4 的实际文件、唯一 ID、划分算法和顺序，不自动修复任何差异。"""
    import pandas as pd
    from sklearn.model_selection import train_test_split

    # 作者对两种视角的相对路径分别排序，再 zip，之后才调用 sklearn 随机划分。
    # 只重建期望列表用于核验，不覆盖服务器上已有 CSV，也不重新进行图像对齐。
    svi_files = sorted(path.relative_to(data_root).as_posix()
                       for path in (data_root / '00_SVI_aligned').rglob('*.png'))
    sat_files = sorted(path.relative_to(data_root).as_posix()
                       for path in (data_root / '01_Satellite').rglob('*.png'))
    if len(svi_files) != 4121 or len(sat_files) != 4121:
        raise ValueError('新版论文要求街景、卫星各 4121 张，请检查预处理数据完整性')
    expected = train_test_split(
        pd.DataFrame(list(zip(sat_files, svi_files)), columns=['sat', 'ground']),
        train_size=0.6, random_state=42)
    split_ids = []
    manifest = {'protocol': '2025-paper-geoloc-random-60', 'random_state': 42,
                'train_size': 0.6, 'splits': {}}
    for name, expected_frame in zip(("train_60.csv", "val_60.csv"), expected):
        rows = 0
        ids = set()
        actual_pairs = []
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
                image_id = int(paths[0].stem)
                if image_id in ids:
                    raise ValueError(f'{name}:{number} 随机比例划分不应出现重复 ID: {image_id}')
                ids.add(image_id)
                actual_pairs.append(pair)
                rows += 1
        if rows == 0:
            raise ValueError(f"{name} 没有样本")
        if actual_pairs != expected_frame.values.tolist():
            raise ValueError(f'{name} 与作者 random_state=42 的样本或顺序不一致')
        print(f"GEOLOC_SPLIT_OK {name} rows={rows} unique_ids={len(ids)}", flush=True)
        split_ids.append(ids)
        manifest['splits'][name] = {
            'rows': rows, 'unique_ids': len(ids),
            'sha256': hashlib.sha256((data_root / name).read_bytes()).hexdigest(),
        }
    if split_ids[0] & split_ids[1]:
        raise ValueError("训练集与测试集包含相同图片 ID，请检查划分")
    return manifest


def prepare(data_root, preprocess_script):
    data_root = Path(data_root).resolve()
    artifacts = [data_root / name for name in
                 ("train_60.csv", "val_60.csv", "00_SVI_aligned")]
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
    manifest = validate_splits(data_root)
    # 将实际使用的 CSV 快照和哈希保存在本次独立运行目录，便于结果追溯。
    # 共享数据目录只读；历史实验目录及已有划分文件不被改写。
    for name in manifest['splits']:
        shutil.copyfile(data_root / name, Path.cwd() / name)
    Path('geoloc_split_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--preprocess-script", required=True)
    args = parser.parse_args()
    prepare(args.data_root, args.preprocess_script)
