#!/usr/bin/env python3
"""
聚合comprehensive_benchmark中的所有实验结果
提取ic_auc_additive和threshold_point_additive数据
"""

import os
import pandas as pd
from pathlib import Path

def aggregate_benchmark_results(benchmark_dir, output_file):
    """
    聚合所有实验结果到单个CSV文件

    Args:
        benchmark_dir: benchmark结果目录
        output_file: 输出CSV文件路径
    """

    # 查找所有ic_auc_comparison.tsv文件
    result_files = list(Path(benchmark_dir).rglob("ic_auc_comparison.tsv"))

    print(f"找到 {len(result_files)} 个结果文件")

    all_results = []

    for file_path in result_files:
        try:
            # 读取TSV文件
            df = pd.read_csv(file_path, sep='\t')

            # 检查必需的列是否存在
            required_cols = ['model', 'ic_auc_additive', 'threshold_point_additive',
                           'dataset', 'task_type', 'downstream_model']

            if not all(col in df.columns for col in required_cols):
                print(f"警告: {file_path} 缺少必需的列，跳过")
                continue

            # 提取需要的列
            subset = df[['model', 'category', 'description',
                        'ic_auc_additive', 'threshold_point_additive',
                        'dataset', 'task_type', 'downstream_model']].copy()

            # 添加来源文件信息
            subset['source_file'] = str(file_path.relative_to(benchmark_dir))

            all_results.append(subset)

        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            continue

    if not all_results:
        print("错误: 没有成功读取任何结果文件")
        return

    # 合并所有结果
    combined_df = pd.concat(all_results, ignore_index=True)

    # 按数据集、任务类型、模型排序
    combined_df = combined_df.sort_values(
        by=['dataset', 'task_type', 'downstream_model', 'ic_auc_additive'],
        ascending=[True, True, True, False]
    )

    # 保存为CSV
    combined_df.to_csv(output_file, index=False)

    print(f"\n聚合完成!")
    print(f"总共 {len(combined_df)} 条记录")
    print(f"数据集数量: {combined_df['dataset'].nunique()}")
    print(f"任务类型数量: {combined_df['task_type'].nunique()}")
    print(f"模型数量: {combined_df['model'].nunique()}")
    print(f"结果已保存到: {output_file}")

    # 打印统计信息
    print("\n=== 数据集统计 ===")
    print(combined_df['dataset'].value_counts().sort_index())

    print("\n=== 任务类型统计 ===")
    print(combined_df['task_type'].value_counts().sort_index())

    print("\n=== 模型统计 ===")
    print(combined_df['model'].value_counts().sort_index())

    # 生成数据透视表：模型 x (数据集_任务)
    print("\n=== 生成数据透视表 ===")

    # 创建组合键
    combined_df['experiment'] = (combined_df['dataset'] + '_' +
                                  combined_df['task_type'] + '_' +
                                  combined_df['downstream_model'])

    # IC-AUC透视表
    pivot_ic_auc = combined_df.pivot_table(
        values='ic_auc_additive',
        index='model',
        columns='experiment',
        aggfunc='first'
    )

    ic_auc_file = output_file.replace('.csv', '_ic_auc_pivot.csv')
    pivot_ic_auc.to_csv(ic_auc_file)
    print(f"IC-AUC透视表已保存到: {ic_auc_file}")

    # Threshold Point透视表
    pivot_threshold = combined_df.pivot_table(
        values='threshold_point_additive',
        index='model',
        columns='experiment',
        aggfunc='first'
    )

    threshold_file = output_file.replace('.csv', '_threshold_pivot.csv')
    pivot_threshold.to_csv(threshold_file)
    print(f"Threshold Point透视表已保存到: {threshold_file}")

    return combined_df

if __name__ == "__main__":
    benchmark_dir = "/home/huyut/Projects/GS/results/comprehensive_benchmark"
    output_file = "/home/huyut/Projects/GS/results/aggregated_benchmark_results.csv"

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 执行聚合
    df = aggregate_benchmark_results(benchmark_dir, output_file)

    print("\n完成!")
