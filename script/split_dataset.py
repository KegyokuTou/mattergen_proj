import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import os
from pathlib import Path
import numpy as np

def split_dataset(
    input_file: str,
    output_dir: str,
    stratify_by: str = "dft_mag_density",
    val_size: float = 0.2,
    random_seed: int = 42,
    max_num_sites: int = None,
    energy_above_hull_max: float = None,
):
    """
    使用分层抽样将数据集划分为训练集和验证集。
    支持类别型和连续型数据分层。

    Args:
        input_file (str): 包含完整数据集的输入文件路径 (例如 .jsonl 或 .csv).
        output_dir (str): 保存划分后文件的输出目录。
        stratify_by (str): 用于分层的列名。
        val_size (float): 验证集所占的比例。
        random_seed (int): 用于复现的随机种子。
        max_num_sites (int, optional): 如果提供，则移除 num_sites 大于此值的行。
        energy_above_hull_max (float, optional): 如果提供，则移除 energy_above_hull 大于此值的行。
    """
    print(f"正在加载数据集: {input_file}")
    
    # 根据文件扩展名加载数据
    file_ext = Path(input_file).suffix.lower()
    if file_ext == ".jsonl":
        df = pd.read_json(input_file, lines=True)
    elif file_ext == ".csv":
        df = pd.read_csv(input_file)
    else:
        raise ValueError(f"不支持的文件格式: {file_ext}。请使用 .jsonl 或 .csv。")
    
    if max_num_sites is not None:
        if 'num_sites' in df.columns:
            initial_count = len(df)
            df = df[df['num_sites'] <= max_num_sites].copy()
            removed_count = initial_count - len(df)
            print(f"\n已根据 num_sites <= {max_num_sites} 进行过滤，移除了 {removed_count} 行。")
        else:
            print(f"\n警告: 请求按 'num_sites' 进行过滤，但数据集中不存在该列。")
    
    if energy_above_hull_max is not None:
        if 'energy_above_hull' in df.columns:
            initial_count = len(df)
            df = df[df['energy_above_hull'] <= energy_above_hull_max].copy()
            removed_count = initial_count - len(df)
            print(f"\n已根据 energy_above_hull <= {energy_above_hull_max} 进行过滤，移除了 {removed_count} 行。")
        else:
            print(f"\n警告: 请求按 'energy_above_hull' 进行过滤，但数据集中不存在该列。")

    print(f"数据集加载完成，总计 {len(df)} 条记录。")
    print(f"将根据 '{stratify_by}' 列进行分层抽样。")

    if stratify_by not in df.columns:
        raise ValueError(f"错误: 用于分层的列 '{stratify_by}' 不在数据集中。")

    stratify_column = df[stratify_by]

    # 判断分层列是连续型还是类别型
    # 如果是数值类型且唯一值数量 > 100，则视为连续数据进行分桶
    if np.issubdtype(stratify_column.dtype, np.number) and stratify_column.nunique() > 100:
        print(f"检测到 '{stratify_by}' 为连续值，将进行分桶处理。")
        # 创建分桶后的新列用于分层
        num_bins = int(np.floor(1 + 3.322 * np.log(len(df)))) # Sturges' rule
        num_bins = min(num_bins, 50) # 限制最大桶数
        print(f"将数据分为 {num_bins} 个桶。")
        stratify_key = pd.cut(stratify_column, bins=num_bins, labels=False)
    else:
        print(f"检测到 '{stratify_by}' 为类别值。")
        stratify_key = stratify_column

    # 检查分层键中是否有样本过少的类别
    value_counts = stratify_key.value_counts()
    if (value_counts < 2).any():
        print("警告: 存在样本数量小于2的类别，这些类别将不会出现在验证集中。")
        # 筛选出可以进行划分的数据
        valid_categories = value_counts[value_counts >= 2].index
        stratify_mask = stratify_key.isin(valid_categories)
        stratify_df = df[stratify_mask]
        stratify_key_filtered = stratify_key[stratify_mask]
        single_sample_df = df[~stratify_mask]
    else:
        stratify_df = df
        stratify_key_filtered = stratify_key
        single_sample_df = pd.DataFrame()

    # 执行分层抽样
    train_df, val_df = train_test_split(
        stratify_df,
        test_size=val_size,
        stratify=stratify_key_filtered,
        random_state=random_seed,
    )
    
    # 将无法划分的单一样本数据添加回训练集
    if not single_sample_df.empty:
        train_df = pd.concat([train_df, single_sample_df], ignore_index=True)


    print("\n划分结果:")
    print(f"  训练集大小: {len(train_df)}")
    print(f"  验证集大小: {len(val_df)}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 根据输入文件类型决定输出文件类型
    if file_ext == ".jsonl":
        train_output_path = os.path.join(output_dir, "train.jsonl")
        val_output_path = os.path.join(output_dir, "val.jsonl")
        train_df.to_json(train_output_path, orient="records", lines=True)
        val_df.to_json(val_output_path, orient="records", lines=True)
    else: # .csv
        train_output_path = os.path.join(output_dir, "train.csv")
        val_output_path = os.path.join(output_dir, "val.csv")
        train_df.to_csv(train_output_path, index=False)
        val_df.to_csv(val_output_path, index=False)


    print(f"\n训练集已保存至: {train_output_path}")
    print(f"验证集已保存至: {val_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用分层抽样划分晶体数据集。")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="包含完整数据集的输入文件路径 (例如: data/all_materials.jsonl)。",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="保存划分后文件的输出目录 (例如: ../datasets/cache/alex_mp_20)。",
    )
    parser.add_argument(
        "--stratify_by",
        type=str,
        default="chemical_system",
        help="用于分层的列名 (例如: 'chemical_system', 'space_group')。",
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.2,
        help="验证集所占的比例 (例如: 0.2 表示 20%%)。",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="用于复现的随机种子。"
    )
    parser.add_argument(
        "--max_num_sites",
        type=int,
        default=None,
        help="可选：过滤掉 num_sites 大于此值的行 (例如: 20)。"
    )
    parser.add_argument(
        "--energy_above_hull_max",
        type=float,
        default=None,
        help="可选：过滤掉 energy_above_hull 大于此值的行 (例如: 0.1)。"
    )

    args = parser.parse_args()

    split_dataset(
        input_file=args.input,
        output_dir=args.output_dir,
        stratify_by=args.stratify_by,
        val_size=args.val_size,
        random_seed=args.seed,
        max_num_sites=args.max_num_sites,
        energy_above_hull_max=args.energy_above_hull_max,
    )