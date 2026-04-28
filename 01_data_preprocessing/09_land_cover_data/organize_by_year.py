#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ESRI土地覆盖数据按年份归纳整理工具
功能：根据文件名中的年份信息，将文件按年份归纳到对应的文件夹中
作者：锐多宝 (ruiduobao)
日期：2024年
"""

import os
import shutil
import re
from datetime import datetime
import logging
from pathlib import Path

def setup_logging():
    """
    设置日志记录系统
    """
    # 创建输出文件夹
    output_dir = "输出文件夹"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 设置日志文件
    log_filename = os.path.join(output_dir, f"文件归纳日志_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return log_filename

def extract_year_from_filename(filename):
    """
    从文件名中提取年份信息
    
    参数:
    filename: 文件名，例如 "51P_20210101-20220101.tif"
    
    返回:
    年份字符串，例如 "2021"
    """
    # 使用正则表达式匹配年份模式：YYYYMMDD-YYYYMMDD
    pattern = r'(\d{4})\d{4}-\d{8}'
    match = re.search(pattern, filename)
    
    if match:
        return match.group(1)  # 返回第一个年份
    else:
        # 如果没有匹配到标准格式，尝试其他可能的年份格式
        year_pattern = r'(20\d{2})'
        year_match = re.search(year_pattern, filename)
        if year_match:
            return year_match.group(1)
    
    return None

def organize_files_by_year(source_dir, dry_run=False):
    """
    按年份组织文件
    
    参数:
    source_dir: 源文件夹路径
    dry_run: 是否为试运行模式（不实际移动文件）
    """
    
    log_filename = setup_logging()
    logging.info("开始按年份归纳ESRI土地覆盖数据文件")
    logging.info(f"源文件夹: {source_dir}")
    logging.info(f"试运行模式: {dry_run}")
    
    # 检查源文件夹是否存在
    if not os.path.exists(source_dir):
        logging.error(f"源文件夹不存在: {source_dir}")
        return False
    
    # 统计变量
    total_files = 0
    processed_files = 0
    error_files = 0
    year_stats = {}
    
    # 遍历源文件夹中的所有文件
    for filename in os.listdir(source_dir):
        file_path = os.path.join(source_dir, filename)
        
        # 跳过文件夹和非.tif文件
        if not os.path.isfile(file_path) or not filename.lower().endswith('.tif'):
            continue
        
        total_files += 1
        
        # 从文件名中提取年份
        year = extract_year_from_filename(filename)
        
        if year is None:
            logging.warning(f"无法从文件名中提取年份: {filename}")
            error_files += 1
            continue
        
        # 创建年份文件夹路径
        year_folder = os.path.join(source_dir, f"{year}年")
        
        # 统计年份信息
        if year not in year_stats:
            year_stats[year] = 0
        year_stats[year] += 1
        
        try:
            # 创建年份文件夹（如果不存在）
            if not dry_run and not os.path.exists(year_folder):
                os.makedirs(year_folder)
                logging.info(f"创建年份文件夹: {year_folder}")
            
            # 目标文件路径
            target_path = os.path.join(year_folder, filename)
            
            # 检查目标文件是否已存在
            if os.path.exists(target_path):
                logging.warning(f"目标文件已存在，跳过: {target_path}")
                continue
            
            # 移动文件
            if not dry_run:
                shutil.move(file_path, target_path)
                logging.info(f"移动文件: {filename} -> {year}年/")
            else:
                logging.info(f"[试运行] 将移动文件: {filename} -> {year}年/")
            
            processed_files += 1
            
        except Exception as e:
            logging.error(f"处理文件 {filename} 时出错: {str(e)}")
            error_files += 1
    
    # 输出统计信息
    logging.info("=" * 60)
    logging.info("文件归纳统计信息:")
    logging.info(f"总文件数: {total_files}")
    logging.info(f"成功处理: {processed_files}")
    logging.info(f"错误文件: {error_files}")
    logging.info("=" * 60)
    
    logging.info("各年份文件统计:")
    for year, count in sorted(year_stats.items()):
        logging.info(f"  {year}年: {count} 个文件")
    
    logging.info("=" * 60)
    logging.info(f"日志文件位置: {log_filename}")
    
    return True

def main():
    """
    主函数：执行文件归纳操作
    """
    
    print("=" * 80)
    print("ESRI土地覆盖数据按年份归纳整理工具")
    print("作者：锐多宝 (ruiduobao)")
    print("=" * 80)
    
    # 源文件夹路径
    source_directory = r"K:\地理所\论文\东南亚10m人工林提取\数据\ESRI\ESRI_2017_2023"
    
    print(f"源文件夹: {source_directory}")
    print()
    
    # 检查源文件夹是否存在
    if not os.path.exists(source_directory):
        print(f"❌ 错误：源文件夹不存在")
        print(f"请检查路径: {source_directory}")
        return
    
    try:
        # 自动先进行试运行
        print("🔍 开始试运行（预览文件归纳操作）...")
        print("=" * 60)
        organize_files_by_year(source_directory, dry_run=True)
        print("=" * 60)
        print("✅ 试运行完成！请查看上方日志了解详细信息。")
        print()
        
        # 询问是否继续正式运行
        print("是否继续正式运行文件归纳操作？")
        print("注意：正式运行将实际移动文件到对应年份文件夹中！")
        confirm = input("请输入 (y/n): ").strip().lower()
        
        if confirm in ['y', 'yes', '是', 'Y']:
            print("\n📁 开始正式归纳文件...")
            print("=" * 60)
            organize_files_by_year(source_directory, dry_run=False)
            print("=" * 60)
            print("✅ 文件归纳完成！")
        else:
            print("\n⏹️ 操作已取消，文件未被移动")
            
    except KeyboardInterrupt:
        print("\n\n⏹️ 用户中断操作")
    except Exception as e:
        print(f"\n❌ 程序执行出错: {str(e)}")

if __name__ == "__main__":
    main()