#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3.find_dataset.py - 查找Global Forest Watch数据集脚本

功能说明：
1. 获取所有可用的GFW数据集列表
2. 查找特定的SDPT (Spatial Database of Planted Trees) 数据集
3. 获取数据集的详细信息和元数据
4. 保存结果到日志文件

作者：锐多宝 (ruiduobao)
创建时间：2025年1月19日
"""

import requests
import json
import logging
from datetime import datetime
import os

# 配置参数
API_KEY = "1c86f6f3-c84f-4de2-8f9f-e0dfc20dc50e"  # 用户提供的API密钥
BASE_URL = "https://data-api.globalforestwatch.org"  # GFW API基础URL
SDPT_DATASET_ID = "gfw_planted_forests"  # SDPT数据集标识符

# 配置日志记录
def setup_logging():
    """
    设置日志记录配置
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"dataset_search_{timestamp}.log"
    
    # 创建日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"日志文件已创建: {log_filename}")
    return logger

def make_api_request(url, headers=None):
    """
    发送API请求的通用函数
    
    参数:
        url (str): 请求的URL
        headers (dict): 请求头信息
    
    返回:
        dict: API响应的JSON数据，如果失败返回None
    """
    try:
        # 设置默认请求头
        if headers is None:
            headers = {
                'Authorization': f'Bearer {API_KEY}',
                'Content-Type': 'application/json'
            }
        
        logger.info(f"发送GET请求到: {url}")
        
        # 发送GET请求
        response = requests.get(url, headers=headers, timeout=30)
        
        # 记录响应状态
        logger.info(f"响应状态码: {response.status_code}")
        
        # 检查响应状态
        if response.status_code == 200:
            data = response.json()
            logger.info("请求成功，数据获取完成")
            return data
        else:
            logger.error(f"请求失败，状态码: {response.status_code}")
            logger.error(f"错误信息: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"请求异常: {str(e)}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"JSON解析错误: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"未知错误: {str(e)}")
        return None

def get_all_datasets(page_number=1, page_size=50):
    """
    获取所有可用的数据集列表
    
    参数:
        page_number (int): 页码，默认为1
        page_size (int): 每页数据量，默认为50
    
    返回:
        dict: 包含数据集列表的响应数据
    """
    logger.info(f"获取数据集列表 - 页码: {page_number}, 每页数量: {page_size}")
    
    # 构建请求URL
    url = f"{BASE_URL}/datasets?page[number]={page_number}&page[size]={page_size}"
    
    # 发送请求
    data = make_api_request(url)
    
    if data and data.get('status') == 'success':
        datasets = data.get('data', [])
        logger.info(f"成功获取 {len(datasets)} 个数据集")
        
        # 打印数据集基本信息
        for i, dataset in enumerate(datasets, 1):
            dataset_id = dataset.get('dataset', 'Unknown')
            is_downloadable = dataset.get('is_downloadable', False)
            created_on = dataset.get('created_on', 'Unknown')
            logger.info(f"{i}. 数据集ID: {dataset_id}")
            logger.info(f"   可下载: {is_downloadable}")
            logger.info(f"   创建时间: {created_on}")
            logger.info("   " + "-" * 50)
        
        return data
    else:
        logger.error("获取数据集列表失败")
        return None

def get_specific_dataset(dataset_id):
    """
    获取特定数据集的详细信息
    
    参数:
        dataset_id (str): 数据集标识符
    
    返回:
        dict: 数据集详细信息
    """
    logger.info(f"获取数据集详细信息: {dataset_id}")
    
    # 构建请求URL
    url = f"{BASE_URL}/dataset/{dataset_id}"
    
    # 发送请求
    data = make_api_request(url)
    
    if data and data.get('status') == 'success':
        dataset_info = data.get('data', {})
        logger.info(f"成功获取数据集 '{dataset_id}' 的详细信息")
        
        # 打印详细信息
        logger.info("=" * 60)
        logger.info(f"数据集标识符: {dataset_info.get('dataset', 'Unknown')}")
        logger.info(f"是否可下载: {dataset_info.get('is_downloadable', False)}")
        logger.info(f"创建时间: {dataset_info.get('created_on', 'Unknown')}")
        logger.info(f"更新时间: {dataset_info.get('updated_on', 'Unknown')}")
        
        # 显示版本信息
        versions = dataset_info.get('versions', [])
        if versions:
            logger.info(f"可用版本: {', '.join(versions)}")
        
        # 显示元数据信息
        metadata = dataset_info.get('metadata', {})
        if metadata:
            logger.info("元数据信息:")
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    logger.info(f"  {key}: {value}")
                elif isinstance(value, (list, dict)):
                    logger.info(f"  {key}: {type(value).__name__} (长度: {len(value)})")
        
        logger.info("=" * 60)
        return data
    else:
        logger.error(f"获取数据集 '{dataset_id}' 详细信息失败")
        return None

def get_dataset_assets(dataset_id=None, asset_type=None, page_number=1, page_size=50):
    """
    获取数据集的资产格式信息
    
    参数:
        dataset_id (str): 特定数据集ID，如果为None则获取所有数据集的资产
        asset_type (str): 资产类型过滤器，可选值包括:
                         - "Raster tile set"
                         - "Database table" 
                         - "Geo database table"
                         - "ESRI Shapefile"
                         - "Geopackage"
                         - "SH"
        page_number (int): 页码，默认为1
        page_size (int): 每页数据量，默认为50
    
    返回:
        dict: 包含资产信息的响应数据
    """
    logger.info(f"获取数据集资产格式信息")
    if dataset_id:
        logger.info(f"  数据集ID: {dataset_id}")
    if asset_type:
        logger.info(f"  资产类型过滤: {asset_type}")
    logger.info(f"  页码: {page_number}, 每页数量: {page_size}")
    
    # 构建请求URL
    url = f"{BASE_URL}/assets"
    
    # 构建查询参数
    params = {
        'page[number]': page_number,
        'page[size]': page_size
    }
    
    # 添加可选参数
    if dataset_id:
        params['dataset'] = dataset_id
    if asset_type:
        params['asset_type'] = asset_type
    
    # 构建完整URL
    param_string = '&'.join([f"{k}={v}" for k, v in params.items()])
    full_url = f"{url}?{param_string}"
    
    # 发送请求
    data = make_api_request(full_url)
    
    if data and data.get('status') == 'success':
        assets = data.get('data', [])
        logger.info(f"成功获取 {len(assets)} 个资产记录")
        
        # 统计资产类型
        asset_types = {}
        datasets_with_assets = set()
        
        for asset in assets:
            asset_type_name = asset.get('asset_type', 'Unknown')
            dataset_name = asset.get('dataset', 'Unknown')
            
            # 统计资产类型
            if asset_type_name not in asset_types:
                asset_types[asset_type_name] = 0
            asset_types[asset_type_name] += 1
            
            # 记录有资产的数据集
            datasets_with_assets.add(dataset_name)
        
        # 打印统计信息
        logger.info("资产类型统计:")
        for asset_type_name, count in asset_types.items():
            logger.info(f"  {asset_type_name}: {count} 个")
        
        if dataset_id:
            logger.info(f"数据集 '{dataset_id}' 的可用格式:")
            dataset_assets = [asset for asset in assets if asset.get('dataset') == dataset_id]
            for asset in dataset_assets:
                logger.info(f"  - {asset.get('asset_type', 'Unknown')}")
                logger.info(f"    创建时间: {asset.get('created_on', 'Unknown')}")
                logger.info(f"    更新时间: {asset.get('updated_on', 'Unknown')}")
        else:
            logger.info(f"共有 {len(datasets_with_assets)} 个数据集包含资产")
        
        return data
    else:
        logger.error("获取数据集资产信息失败")
        return None

def get_specific_dataset_formats(dataset_id):
    """
    获取特定数据集支持的所有数据格式
    
    参数:
        dataset_id (str): 数据集标识符
    
    返回:
        list: 支持的数据格式列表
    """
    logger.info(f"查询数据集 '{dataset_id}' 支持的数据格式")
    
    # 使用专门的数据集资产端点获取格式信息
    assets_data = get_specific_dataset_assets(dataset_id)
    
    if assets_data and assets_data.get('status') == 'success':
        assets = assets_data.get('data', [])
        
        # 提取所有格式类型
        formats = []
        for asset in assets:
            asset_type = asset.get('asset_type', 'Unknown')
            if asset_type not in formats:
                formats.append(asset_type)
        
        if formats:
            logger.info(f"数据集 '{dataset_id}' 支持以下格式:")
            for i, format_type in enumerate(formats, 1):
                logger.info(f"  {i}. {format_type}")
        else:
            logger.warning(f"数据集 '{dataset_id}' 未找到可用的数据格式")
        
        return formats
    else:
        logger.error(f"无法获取数据集 '{dataset_id}' 的格式信息")
        return []

def get_all_assets(page_number=1, page_size=50, asset_type=None):
    """
    获取所有数据集的资产格式信息
    
    参数:
        page_number (int): 页码，默认为1
        page_size (int): 每页数据量，默认为50
        asset_type (str): 资产类型过滤器，可选值包括：
                         'raster_tile_set', 'database_table', 'geo_database_table',
                         'esri_shapefile', 'geopackage', 'sh'
    
    返回:
        dict: 包含资产信息的响应数据
    """
    logger.info(f"获取数据集资产格式信息 - 页码: {page_number}, 每页数量: {page_size}")
    if asset_type:
        logger.info(f"过滤资产类型: {asset_type}")
    
    # 构建请求URL
    url = f"{BASE_URL}/assets?page[number]={page_number}&page[size]={page_size}"
    if asset_type:
        url += f"&asset_type={asset_type}"
    
    # 发送请求
    data = make_api_request(url)
    
    if data and data.get('status') == 'success':
        assets = data.get('data', [])
        logger.info(f"成功获取 {len(assets)} 个数据集资产")
        
        # 统计资产类型
        asset_types = {}
        for asset in assets:
            asset_type_name = asset.get('asset_type', 'Unknown')
            dataset_id = asset.get('dataset', 'Unknown')
            
            if asset_type_name not in asset_types:
                asset_types[asset_type_name] = []
            asset_types[asset_type_name].append(dataset_id)
        
        # 打印资产类型统计
        logger.info("资产类型统计:")
        for asset_type, datasets in asset_types.items():
            logger.info(f"  {asset_type}: {len(datasets)} 个数据集")
        
        return data
    else:
        logger.error("获取数据集资产信息失败")
        return None

def get_specific_dataset_assets(dataset_id, version=None):
    """
    获取特定数据集的资产格式信息（通过dataset/{id}/assets端点）
    
    参数:
        dataset_id (str): 数据集标识符
        version (str): 数据集版本，可选
    
    返回:
        dict: 数据集资产信息
    """
    logger.info(f"获取数据集资产格式: {dataset_id}")
    if version:
        logger.info(f"指定版本: {version}")
    
    # 构建请求URL
    if version:
        url = f"{BASE_URL}/dataset/{dataset_id}/{version}/assets"
    else:
        url = f"{BASE_URL}/dataset/{dataset_id}/assets"
    
    # 发送请求
    data = make_api_request(url)
    
    if data and data.get('status') == 'success':
        assets = data.get('data', [])
        logger.info(f"数据集 '{dataset_id}' 支持 {len(assets)} 种资产格式")
        
        # 打印支持的格式详细信息
        logger.info("=" * 60)
        logger.info(f"数据集 '{dataset_id}' 支持的数据格式:")
        logger.info("=" * 60)
        
        format_summary = {}
        for i, asset in enumerate(assets, 1):
            asset_type = asset.get('asset_type', 'Unknown')
            asset_uri = asset.get('asset_uri', 'N/A')
            creation_options = asset.get('creation_options', {})
            
            logger.info(f"{i}. 格式类型: {asset_type}")
            logger.info(f"   资产URI: {asset_uri}")
            
            # 显示创建选项（如果有）
            if creation_options:
                logger.info("   创建选项:")
                for key, value in creation_options.items():
                    logger.info(f"     {key}: {value}")
            
            logger.info("   " + "-" * 50)
            
            # 统计格式类型
            if asset_type not in format_summary:
                format_summary[asset_type] = 0
            format_summary[asset_type] += 1
        
        # 打印格式摘要
        logger.info("支持的格式摘要:")
        for format_type, count in format_summary.items():
            logger.info(f"  - {format_type}: {count} 个资产")
        
        logger.info("=" * 60)
        return data
    else:
        logger.error(f"获取数据集 '{dataset_id}' 资产信息失败")
        return None

def search_sdpt_dataset():
    """
    专门搜索SDPT数据集的函数
    
    返回:
        dict: SDPT数据集信息
    """
    logger.info("开始搜索SDPT (Spatial Database of Planted Trees) 数据集")
    
    # 首先获取所有数据集，查找SDPT
    all_datasets = get_all_datasets(page_size=100)  # 增加页面大小以获取更多数据集
    
    if all_datasets:
        datasets = all_datasets.get('data', [])
        
        # 查找SDPT相关数据集
        sdpt_datasets = []
        for dataset in datasets:
            dataset_id = dataset.get('dataset', '')
            if 'planted' in dataset_id.lower() or 'sdpt' in dataset_id.lower():
                sdpt_datasets.append(dataset)
        
        if sdpt_datasets:
            logger.info(f"找到 {len(sdpt_datasets)} 个与SDPT相关的数据集:")
            for dataset in sdpt_datasets:
                logger.info(f"  - {dataset.get('dataset', 'Unknown')}")
        else:
            logger.warning("在数据集列表中未找到SDPT相关数据集")
    
    # 直接获取已知的SDPT数据集详细信息
    sdpt_info = get_specific_dataset(SDPT_DATASET_ID)
    
    return sdpt_info

def save_results_to_file(data, filename_prefix="dataset_results"):
    """
    将结果保存到JSON文件
    
    参数:
        data (dict): 要保存的数据
        filename_prefix (str): 文件名前缀
    """
    if data:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"结果已保存到文件: {filename}")
        except Exception as e:
            logger.error(f"保存文件失败: {str(e)}")

def main():
    """
    主函数 - 执行数据集查找流程
    """
    global logger
    logger = setup_logging()
    
    logger.info("开始执行GFW数据集查找脚本")
    logger.info(f"使用API密钥: {API_KEY[:10]}...")
    
    try:
        # 1. 获取所有数据集列表（第一页）
        logger.info("\n" + "="*50)
        logger.info("步骤1: 获取数据集列表")
        logger.info("="*50)
        
        all_datasets = get_all_datasets(page_number=1, page_size=20)
        if all_datasets:
            save_results_to_file(all_datasets, "all_datasets")
        
        # 2. 专门查找SDPT数据集
        logger.info("\n" + "="*50)
        logger.info("步骤2: 查找SDPT数据集")
        logger.info("="*50)
        
        sdpt_info = search_sdpt_dataset()
        if sdpt_info:
            save_results_to_file(sdpt_info, "sdpt_dataset_info")
            logger.info("SDPT数据集信息获取成功！")
            
            # 查询SDPT数据集支持的格式
            logger.info("\n查询SDPT数据集支持的数据格式:")
            sdpt_formats = get_specific_dataset_formats(SDPT_DATASET_ID)
            if sdpt_formats:
                logger.info(f"SDPT数据集支持 {len(sdpt_formats)} 种数据格式")
        else:
            logger.error("SDPT数据集信息获取失败")
        
        # 3. 获取其他推荐数据集的信息和格式
        logger.info("\n" + "="*50)
        logger.info("步骤3: 获取其他推荐数据集信息和支持格式")
        logger.info("="*50)
        
        recommended_datasets = [
            "gfw_forest_carbon_net_flux",
            "nasa_viirs_fire_alerts", 
            "umd_tree_cover_loss_from_fires",
            "gfw_integrated_alerts"
        ]
        
        for dataset_id in recommended_datasets:
            logger.info(f"\n获取数据集: {dataset_id}")
            dataset_info = get_specific_dataset(dataset_id)
            if dataset_info:
                save_results_to_file(dataset_info, f"dataset_{dataset_id}")
                
                # 查询该数据集支持的格式
                logger.info(f"查询数据集 '{dataset_id}' 支持的格式:")
                formats = get_specific_dataset_formats(dataset_id)
                if formats:
                    logger.info(f"数据集 '{dataset_id}' 支持 {len(formats)} 种格式")
        
        # 4. 获取所有数据集的资产格式概览
        logger.info("\n" + "="*50)
        logger.info("步骤4: 获取数据集资产格式概览")
        logger.info("="*50)
        
        # 获取前50个资产记录作为概览
        assets_overview = get_all_assets(page_size=50)
        if assets_overview:
            save_results_to_file(assets_overview, "dataset_assets_overview")
            logger.info("数据集资产格式概览获取成功！")
        
        logger.info("\n" + "="*50)
        logger.info("数据集查找脚本执行完成！")
        logger.info("="*50)
        
    except Exception as e:
        logger.error(f"脚本执行过程中发生错误: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    # 执行主函数
    success = main()
    
    if success:
        print("\n✅ 脚本执行成功！请查看日志文件获取详细信息。")
    else:
        print("\n❌ 脚本执行失败！请查看日志文件了解错误详情。")