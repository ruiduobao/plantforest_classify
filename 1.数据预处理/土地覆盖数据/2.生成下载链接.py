#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
土地覆盖数据下载链接生成器
功能：生成ESRI土地覆盖时间序列数据的下载链接
作者：锐多宝 (ruiduobao)
日期：2024年
"""

import os
import itertools
from datetime import datetime
import logging

def setup_logging():
    """
    设置日志记录
    """
    # 创建输出文件夹
    output_dir = "输出文件夹"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 设置日志文件
    log_filename = os.path.join(output_dir, f"下载链接生成日志_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return log_filename

def generate_download_links():
    """
    生成土地覆盖数据的下载链接
    
    参数说明：
    - 根据用户指定的UTM区域组合生成下载链接
    - 年份范围：2017-2023
    
    链接格式：lulctimeseries.blob.core.windows.net/lulctimeseriesv003/lc{year}/{zone}{letter}_{year}0101-{next_year}0101.tif
    """
    
    # 设置日志
    log_filename = setup_logging()
    logging.info("开始生成土地覆盖数据下载链接（基于用户指定的UTM区域）")
    
    # 定义用户指定的UTM区域组合
    utm_zones = [
        # 46区域
        (46, 'N'), (46, 'P'), (46, 'Q'), (46, 'R'),
        # 47区域
        (47, 'M'), (47, 'N'), (47, 'P'), (47, 'Q'), (47, 'R'),
        # 48区域
        (48, 'M'), (48, 'N'), (48, 'P'), (48, 'Q'),
        # 49区域
        (49, 'L'), (49, 'M'), (49, 'N'), (49, 'P'), (49, 'Q'),
        # 50区域
        (50, 'L'), (50, 'M'), (50, 'N'), (50, 'P'), (50, 'Q'),
        # 51区域
        (51, 'L'), (51, 'M'), (51, 'N'), (51, 'P'), (51, 'Q'),
        # 52区域
        (52, 'L'), (52, 'M'), (52, 'N'), (52, 'P'),
        # 53区域
        (53, 'L'), (53, 'M'),
        # 54区域
        (54, 'L'), (54, 'M')
    ]
    
    years = list(range(2017, 2024))  # 2017到2023，包含2023
    
    logging.info(f"指定的UTM区域组合数量: {len(utm_zones)}")
    logging.info(f"UTM区域组合: {utm_zones}")
    logging.info(f"年份范围: {years}")
    
    # 基础URL
    base_url = "lulctimeseries.blob.core.windows.net/lulctimeseriesv003"
    
    # 存储所有链接
    all_links = []
    
    # 生成所有组合的下载链接
    total_combinations = len(utm_zones) * len(years)
    logging.info(f"总共需要生成 {total_combinations} 个下载链接")
    
    count = 0
    for year in years:
        next_year = year + 1
        for zone, letter in utm_zones:
            # 构建文件名：例如 46N_20170101-20180101.tif
            filename = f"{zone}{letter}_{year}0101-{next_year}0101.tif"
            
            # 构建完整的下载链接
            download_link = f"{base_url}/lc{year}/{filename}"
            
            all_links.append(download_link)
            count += 1
            
            # 每处理50个链接输出一次进度
            if count % 50 == 0:
                logging.info(f"已生成 {count}/{total_combinations} 个链接")
    
    logging.info(f"链接生成完成，总共生成了 {len(all_links)} 个下载链接")
    
    return all_links, log_filename

def save_links_to_file(links, log_filename):
    """
    将生成的链接保存到文件中
    
    参数:
    links: 下载链接列表
    log_filename: 日志文件名
    """
    
    # 创建输出文件夹
    output_dir = "输出文件夹"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 生成输出文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = os.path.join(output_dir, f"土地覆盖数据下载链接_{timestamp}.txt")
    
    # 保存链接到文件
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write("# 土地覆盖数据下载链接（筛选后的UTM区域）\n")
            f.write(f"# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# 总链接数: {len(links)}\n")
            f.write("# 筛选的UTM区域: 46N,46P,46Q,46R,47M,47N,47P,47Q,47R,48M,48N,48P,48Q,49L,49M,49N,49P,49Q,50L,50M,50N,50P,50Q,51L,51M,51N,51P,51Q,52L,52M,52N,52P,53L,53M,54L,54M\n")
            f.write("# 年份范围: 2017-2023\n")
            f.write("# " + "="*60 + "\n\n")
            
            for i, link in enumerate(links, 1):
                f.write(f"{link}\n")
        
        logging.info(f"下载链接已保存到文件: {output_filename}")
        logging.info(f"日志文件位置: {log_filename}")
        
        return output_filename
        
    except Exception as e:
        logging.error(f"保存文件时出错: {str(e)}")
        return None

def print_sample_links(links, sample_size=10):
    """
    打印部分示例链接
    
    参数:
    links: 下载链接列表
    sample_size: 要显示的示例数量
    """
    
    logging.info(f"\n显示前 {sample_size} 个下载链接示例:")
    logging.info("-" * 80)
    
    for i, link in enumerate(links[:sample_size], 1):
        logging.info(f"{i:2d}. {link}")
    
    if len(links) > sample_size:
        logging.info(f"... (还有 {len(links) - sample_size} 个链接)")
    
    logging.info("-" * 80)

def main():
    """
    主函数：执行下载链接生成流程
    """
    
    print("=" * 80)
    print("土地覆盖数据下载链接生成器")
    print("作者：锐多宝 (ruiduobao)")
    print("=" * 80)
    
    try:
        # 生成下载链接
        links, log_filename = generate_download_links()
        
        # 显示部分示例链接
        print_sample_links(links, 15)
        
        # 保存链接到文件
        output_file = save_links_to_file(links, log_filename)
        
        if output_file:
            print(f"\n✅ 任务完成!")
            print(f"📁 输出文件: {output_file}")
            print(f"📋 日志文件: {log_filename}")
            print(f"🔗 总链接数: {len(links)}")
        else:
            print("\n❌ 保存文件时出现错误，请查看日志")
            
    except Exception as e:
        print(f"\n❌ 程序执行出错: {str(e)}")
        logging.error(f"程序执行出错: {str(e)}")

if __name__ == "__main__":
    main()