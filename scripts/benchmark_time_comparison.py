#!/usr/bin/env python3
"""
时间对比测试脚本
测试 gradient_based 和 pri_graphs 在 Cora 和 CiteSeer 数据集上的运行时间
"""

import sys
import os
from pathlib import Path
import time
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from GS.benchmark.unified import UnifiedBenchmark


def benchmark_time_comparison():
    """运行时间对比测试"""

    # 配置
    models = ['gradient_based', 'pri_graphs']
    datasets = ['Cora', 'CiteSeer']
    task_type = 'original'
    downstream_model = 'gcn'
    num_steps = 10
    epochs = 30

    results_dir = './results/time_benchmark'
    benchmark = UnifiedBenchmark(results_dir=results_dir, device='cpu')

    print("=" * 100)
    print("时间对比测试: gradient_based vs PRI-Graphs")
    print("数据集: Cora, CiteSeer (original task)")
    print("=" * 100)

    # 存储所有时间结果
    time_results = []

    # 对每个数据集和模型组合进行测试
    for dataset_name in datasets:
        for model_name in models:
            print(f"\n{'='*80}")
            print(f"测试: {model_name} on {dataset_name}")
            print(f"{'='*80}")

            # 记录开始时间
            start_time = time.time()

            try:
                # 运行benchmark测试 - 使用 compare_models 方法
                results = benchmark.compare_models(
                    dataset_name=dataset_name,
                    task_type=task_type,
                    downstream_model=downstream_model,
                    model_names=[model_name],
                    num_steps=num_steps,
                    epochs=epochs
                )

                # 记录结束时间
                end_time = time.time()
                elapsed_time = end_time - start_time

                # 获取数据集信息
                success = False
                if results and len(results) > 0:
                    model_result = results[0]
                    success = model_result.get('success', False)

                # 记录结果
                time_results.append({
                    'model': model_name,
                    'dataset': dataset_name,
                    'success': success,
                    'total_time_seconds': elapsed_time,
                    'total_time_minutes': elapsed_time / 60,
                    'time_per_step': elapsed_time / num_steps if num_steps > 0 else 0,
                })

                print(f"✅ 完成 - 用时: {elapsed_time:.2f}秒 ({elapsed_time/60:.2f}分钟)")

            except Exception as e:
                end_time = time.time()
                elapsed_time = end_time - start_time

                time_results.append({
                    'model': model_name,
                    'dataset': dataset_name,
                    'success': False,
                    'total_time_seconds': elapsed_time,
                    'total_time_minutes': elapsed_time / 60,
                    'time_per_step': 0,
                    'error': str(e)
                })

                print(f"❌ 失败 - 用时: {elapsed_time:.2f}秒, 错误: {e}")

    # 生成对比表格
    print("\n" + "=" * 100)
    print("时间对比结果汇总")
    print("=" * 100)

    df = pd.DataFrame(time_results)

    # 按数据集分组显示
    for dataset in datasets:
        print(f"\n{dataset} 数据集:")
        print("-" * 80)
        dataset_df = df[df['dataset'] == dataset]

        for _, row in dataset_df.iterrows():
            status = "✅" if row['success'] else "❌"
            print(f"  {status} {row['model']:<25} {row['total_time_seconds']:>10.2f}秒 "
                  f"({row['total_time_minutes']:>8.2f}分钟) "
                  f"[平均每步: {row['time_per_step']:>6.2f}秒]")

        # 计算加速比
        gradient_time = dataset_df[dataset_df['model'] == 'gradient_based']['total_time_seconds'].values
        pri_time = dataset_df[dataset_df['model'] == 'pri_graphs']['total_time_seconds'].values

        if len(gradient_time) > 0 and len(pri_time) > 0 and gradient_time[0] > 0 and pri_time[0] > 0:
            if gradient_time[0] > pri_time[0]:
                speedup = gradient_time[0] / pri_time[0]
                faster = 'pri_graphs'
            else:
                speedup = pri_time[0] / gradient_time[0]
                faster = 'gradient_based'

            print(f"\n  📊 {faster} 快 {speedup:.2f}x")

    # 保存结果到文件
    output_file = Path(results_dir) / 'time_comparison_summary.tsv'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, sep='\t', index=False)
    print(f"\n💾 结果已保存到: {output_file}")

    # 总体统计
    print("\n" + "=" * 100)
    print("总体统计")
    print("=" * 100)

    for model in models:
        model_df = df[df['model'] == model]
        total_time = model_df['total_time_seconds'].sum()
        avg_time = model_df['total_time_seconds'].mean()
        success_rate = (model_df['success'].sum() / len(model_df) * 100) if len(model_df) > 0 else 0

        print(f"\n{model}:")
        print(f"  总运行时间: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
        print(f"  平均时间: {avg_time:.2f}秒 ({avg_time/60:.2f}分钟)")
        print(f"  成功率: {success_rate:.1f}%")

    print("\n" + "=" * 100)
    print("✅ 时间对比测试完成!")
    print("=" * 100)

    return df


if __name__ == '__main__':
    benchmark_time_comparison()
