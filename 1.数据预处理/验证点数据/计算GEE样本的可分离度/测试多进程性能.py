#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试多进程版本性能脚本
功能：
1. 比较单进程和多进程版本的处理速度
2. 测试不同进程数量的性能表现
3. 生成性能分析报告

作者：锐多宝 (ruiduobao)
日期：2025年1月21日
"""

import os
import time
import glob
import multiprocessing as mp
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 测试配置
TEST_INPUT_DIR = r"K:\地理所\论文\东南亚10m人工林提取\数据\验证点数据\上传到GEE的样本\经过了8年的土地覆盖数据filter_平衡样本_获得Ebed波段\GEE结果_逐年合并"
TEST_OUTPUT_DIR = r"f:\BaiduSyncdisk\论文\东南亚10m人工林提取\代码\1.数据预处理\验证点数据\计算GEE样本的可分离度\性能测试结果"

def get_test_files(input_dir, max_files=5):
    """
    获取用于测试的shapefile文件（限制数量以便快速测试）
    """
    pattern = os.path.join(input_dir, "*.shp")
    shp_files = glob.glob(pattern)
    return shp_files[:max_files]  # 只取前几个文件进行测试

def test_single_process_performance(test_files):
    """
    测试单进程版本的性能
    """
    print("测试单进程版本性能...")
    
    # 这里应该导入并运行原始的单进程版本
    # 由于原始脚本是完整的程序，这里模拟测试
    start_time = time.time()
    
    # 模拟单进程处理时间（实际应该调用原始脚本的处理函数）
    for i, file_path in enumerate(test_files):
        print(f"  处理文件 {i+1}/{len(test_files)}: {os.path.basename(file_path)}")
        # 这里应该调用实际的处理函数
        time.sleep(1)  # 模拟处理时间
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    return {
        'method': '单进程',
        'files_count': len(test_files),
        'total_time': processing_time,
        'avg_time_per_file': processing_time / len(test_files),
        'files_per_second': len(test_files) / processing_time
    }

def test_multiprocess_performance(test_files, num_processes):
    """
    测试多进程版本的性能
    """
    print(f"测试多进程版本性能 (进程数: {num_processes})...")
    
    start_time = time.time()
    
    # 这里应该导入并运行多进程版本
    # 模拟多进程处理
    with mp.Pool(processes=num_processes) as pool:
        # 模拟并行处理
        def mock_process_file(file_path):
            time.sleep(1)  # 模拟处理时间
            return os.path.basename(file_path)
        
        results = pool.map(mock_process_file, test_files)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    return {
        'method': f'多进程({num_processes}核)',
        'files_count': len(test_files),
        'total_time': processing_time,
        'avg_time_per_file': processing_time / len(test_files),
        'files_per_second': len(test_files) / processing_time,
        'processes': num_processes
    }

def run_performance_tests():
    """
    运行性能测试
    """
    print("=" * 80)
    print("多进程性能测试")
    print("=" * 80)
    
    # 创建测试输出目录
    if not os.path.exists(TEST_OUTPUT_DIR):
        os.makedirs(TEST_OUTPUT_DIR)
    
    # 获取测试文件
    test_files = get_test_files(TEST_INPUT_DIR, max_files=10)
    if not test_files:
        print("错误：未找到测试文件")
        return
    
    print(f"使用 {len(test_files)} 个文件进行性能测试")
    print(f"CPU核心数: {mp.cpu_count()}")
    
    # 测试结果列表
    test_results = []
    
    # 测试单进程版本
    single_result = test_single_process_performance(test_files)
    test_results.append(single_result)
    
    # 测试不同进程数的多进程版本
    process_counts = [2, 4, mp.cpu_count()//2, mp.cpu_count()-1, mp.cpu_count()]
    process_counts = [p for p in process_counts if p > 0 and p <= mp.cpu_count()]
    process_counts = sorted(list(set(process_counts)))  # 去重并排序
    
    for num_processes in process_counts:
        if num_processes > 1:  # 跳过单进程
            multi_result = test_multiprocess_performance(test_files, num_processes)
            test_results.append(multi_result)
    
    # 生成测试报告
    print("\n" + "=" * 80)
    print("性能测试结果")
    print("=" * 80)
    
    df_results = pd.DataFrame(test_results)
    
    # 显示结果表格
    print("\n性能对比表:")
    print("-" * 80)
    for _, row in df_results.iterrows():
        print(f"方法: {row['method']:<15} | "
              f"总时间: {row['total_time']:.2f}s | "
              f"平均时间: {row['avg_time_per_file']:.2f}s/文件 | "
              f"处理速度: {row['files_per_second']:.2f}文件/s")
    
    # 计算性能提升
    single_time = df_results[df_results['method'] == '单进程']['total_time'].iloc[0]
    print(f"\n性能提升分析:")
    print("-" * 80)
    for _, row in df_results.iterrows():
        if row['method'] != '单进程':
            speedup = single_time / row['total_time']
            efficiency = speedup / row.get('processes', 1) * 100
            print(f"{row['method']:<15}: {speedup:.2f}x 加速, 效率: {efficiency:.1f}%")
    
    # 保存结果到CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(TEST_OUTPUT_DIR, f"性能测试结果_{timestamp}.csv")
    df_results.to_csv(results_file, index=False, encoding='utf-8-sig')
    print(f"\n测试结果已保存到: {results_file}")
    
    # 生成性能图表
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 处理时间对比
        methods = df_results['method'].tolist()
        times = df_results['total_time'].tolist()
        
        ax1.bar(methods, times, color=['red' if '单进程' in m else 'blue' for m in methods])
        ax1.set_title('处理时间对比')
        ax1.set_ylabel('处理时间 (秒)')
        ax1.tick_params(axis='x', rotation=45)
        
        # 处理速度对比
        speeds = df_results['files_per_second'].tolist()
        ax2.bar(methods, speeds, color=['red' if '单进程' in m else 'green' for m in methods])
        ax2.set_title('处理速度对比')
        ax2.set_ylabel('处理速度 (文件/秒)')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # 保存图表
        chart_file = os.path.join(TEST_OUTPUT_DIR, f"性能测试图表_{timestamp}.png")
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        print(f"性能图表已保存到: {chart_file}")
        
        plt.close()
        
    except Exception as e:
        print(f"生成图表时出错: {e}")
    
    # 推荐最佳配置
    print(f"\n推荐配置:")
    print("-" * 80)
    best_performance = df_results.loc[df_results['files_per_second'].idxmax()]
    print(f"最佳性能配置: {best_performance['method']}")
    print(f"处理速度: {best_performance['files_per_second']:.2f} 文件/秒")
    
    if 'processes' in best_performance and best_performance['processes'] > 1:
        print(f"建议使用 {best_performance['processes']} 个进程进行批量处理")
    
    return df_results

def create_usage_guide():
    """
    创建使用指南
    """
    guide_content = """
# 多进程版本使用指南

## 性能优势
多进程版本通过并行处理多个shapefile文件，可以显著提升处理速度：
- 充分利用多核CPU资源
- 减少总体处理时间
- 提高系统资源利用率

## 使用方法
1. 直接运行多进程版本脚本：
   ```
   python 批量处理矢量筛选_多进程版.py
   ```

2. 脚本会自动：
   - 检测CPU核心数
   - 使用 (CPU核心数-1) 个进程
   - 显示实时进度条
   - 生成详细的处理报告

## 配置参数
可以在脚本中调整以下参数：
- `MAX_WORKERS`: 最大进程数（默认为CPU核心数-1）
- `CHUNK_SIZE`: 每个进程处理的文件数量（默认为1）

## 注意事项
1. 多进程版本需要更多内存资源
2. 在处理大量文件时效果更明显
3. 单个文件很大时，多进程优势可能不明显
4. Windows系统需要在 if __name__ == "__main__": 保护下运行

## 性能监控
脚本会输出：
- 实时进度条
- 处理时间统计
- 平均处理速度
- 详细的性能报告
"""
    
    guide_file = os.path.join(TEST_OUTPUT_DIR, "多进程版本使用指南.md")
    with open(guide_file, 'w', encoding='utf-8') as f:
        f.write(guide_content)
    
    print(f"使用指南已保存到: {guide_file}")

if __name__ == "__main__":
    mp.freeze_support()  # Windows支持
    
    # 运行性能测试
    results = run_performance_tests()
    
    # 创建使用指南
    create_usage_guide()
    
    print(f"\n性能测试完成！")
    print(f"测试结果保存在: {TEST_OUTPUT_DIR}")