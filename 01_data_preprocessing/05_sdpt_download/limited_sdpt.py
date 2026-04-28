#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4.limited_SDPT2.py - 限制SDPT数据到指定区域的脚本

功能说明：
1. 读取GeoJSON文件，获取研究区域的几何信息
2. 使用POST请求方法限制SDPT数据到指定区域
3. 使用Geostore方法限制数据区域（可选）
4. 保存筛选后的数据结果到文件

作者：锐多宝 (ruiduobao)
创建时间：2025年1月19日
"""

import requests
import json
import logging
from datetime import datetime
import os
import time

# 配置参数
API_KEY = "1c86f6f3-c84f-4de2-8f9f-e0dfc20dc50e"  # API密钥
BASE_URL = "https://data-api.globalforestwatch.org"  # GFW API基础URL
DATASET_ID = "gfw_planted_forests"  # SDPT数据集标识符
DATASET_VERSION = "v20239998"  # 使用最新版本
GEOJSON_FILE = "map.geojson"  # GeoJSON文件路径

# 配置日志记录
def setup_logging():
    """
    设置日志记录配置
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"sdpt_limited_query_{timestamp}.log"
    
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

def load_geojson(file_path):
    """
    读取并解析GeoJSON文件
    
    参数:
        file_path (str): GeoJSON文件路径
    
    返回:
        dict: GeoJSON数据，如果失败返回None
    """
    try:
        logger.info(f"正在读取GeoJSON文件: {file_path}")
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            logger.error(f"GeoJSON文件不存在: {file_path}")
            return None
        
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            geojson_data = json.load(f)
        
        logger.info("GeoJSON文件读取成功")
        
        # 验证GeoJSON格式
        if geojson_data.get('type') != 'FeatureCollection':
            logger.error("GeoJSON格式错误：不是FeatureCollection类型")
            return None
        
        features = geojson_data.get('features', [])
        if not features:
            logger.error("GeoJSON中没有找到features")
            return None
        
        logger.info(f"找到 {len(features)} 个地理要素")
        
        # 获取第一个要素的几何信息
        first_feature = features[0]
        geometry = first_feature.get('geometry')
        
        if not geometry:
            logger.error("第一个要素中没有找到几何信息")
            return None
        
        geometry_type = geometry.get('type')
        coordinates = geometry.get('coordinates')
        
        logger.info(f"几何类型: {geometry_type}")
        logger.info(f"坐标数量: {len(coordinates[0]) if coordinates else 0}")
        
        return geojson_data
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON解析错误: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"读取GeoJSON文件时发生错误: {str(e)}")
        return None

def make_api_request(url, method='GET', headers=None, data=None):
    """
    发送API请求的通用函数
    
    参数:
        url (str): 请求的URL
        method (str): 请求方法 ('GET' 或 'POST')
        headers (dict): 请求头信息
        data (dict): POST请求的数据
    
    返回:
        dict: API响应的JSON数据，如果失败返回None
    """
    try:
        # 设置默认请求头
        if headers is None:
            headers = {
                'x-api-key': API_KEY,
                'Content-Type': 'application/json'
            }
        
        logger.info(f"发送{method}请求到: {url}")
        
        # 发送请求
        if method.upper() == 'POST':
            response = requests.post(url, headers=headers, json=data, timeout=60)
        else:
            response = requests.get(url, headers=headers, timeout=60)
        
        # 记录响应状态
        logger.info(f"响应状态码: {response.status_code}")
        
        # 检查响应状态
        if response.status_code in [200, 201]:
            response_data = response.json()
            logger.info("请求成功，数据获取完成")
            return response_data
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

def query_sdpt_with_post(geometry, sql_query=None):
    """
    使用POST请求方法查询指定区域的SDPT数据
    
    参数:
        geometry (dict): GeoJSON几何对象
        sql_query (str): SQL查询语句，如果为None则使用默认查询
    
    返回:
        dict: 查询结果数据
    """
    logger.info("使用POST请求方法查询SDPT数据")
    
    # 构建请求URL
    url = f"{BASE_URL}/dataset/{DATASET_ID}/{DATASET_VERSION}/query/json"
    
    # 设置默认SQL查询
    if sql_query is None:
        sql_query = "SELECT * FROM data"
    
    # 构建POST请求数据
    post_data = {
        "sql": sql_query,
        "geometry": geometry
    }
    
    logger.info(f"SQL查询语句: {sql_query}")
    logger.info(f"几何类型: {geometry.get('type')}")
    
    # 发送POST请求
    result = make_api_request(url, method='POST', data=post_data)
    
    if result:
        # 检查返回的数据
        data_rows = result.get('data', [])
        logger.info(f"查询成功，返回 {len(data_rows)} 条数据记录")
        
        # 显示前几条数据的基本信息
        if data_rows:
            logger.info("前5条数据记录预览:")
            for i, row in enumerate(data_rows[:5], 1):
                logger.info(f"  记录{i}: {dict(list(row.items())[:3])}...")  # 只显示前3个字段
    else:
        logger.error("POST请求查询失败")
    
    return result

def create_geostore(geojson_data):
    """
    创建Geostore用于重复使用几何对象
    
    参数:
        geojson_data (dict): 完整的GeoJSON数据
    
    返回:
        str: Geostore ID，如果失败返回None
    """
    logger.info("创建Geostore")
    
    # 构建请求URL
    url = f"{BASE_URL}/geostore"
    
    # 构建POST请求数据
    post_data = {
        "geojson": geojson_data
    }
    
    # 发送POST请求创建Geostore
    result = make_api_request(url, method='POST', data=post_data)
    
    if result:
        geostore_id = result.get('data', {}).get('id')
        if geostore_id:
            logger.info(f"Geostore创建成功，ID: {geostore_id}")
            return geostore_id
        else:
            logger.error("Geostore创建失败：未返回ID")
            return None
    else:
        logger.error("Geostore创建失败")
        return None

def query_sdpt_with_geostore(geostore_id, sql_query=None):
    """
    使用Geostore方法查询指定区域的SDPT数据
    
    参数:
        geostore_id (str): Geostore ID
        sql_query (str): SQL查询语句，如果为None则使用默认查询
    
    返回:
        dict: 查询结果数据
    """
    logger.info(f"使用Geostore方法查询SDPT数据，Geostore ID: {geostore_id}")
    
    # 设置默认SQL查询
    if sql_query is None:
        sql_query = "SELECT * FROM data"
    
    # 构建请求URL
    url = f"{BASE_URL}/dataset/{DATASET_ID}/{DATASET_VERSION}/query/json"
    url += f"?sql={sql_query}&geostore_id={geostore_id}"
    
    logger.info(f"SQL查询语句: {sql_query}")
    
    # 发送GET请求
    result = make_api_request(url, method='GET')
    
    if result:
        # 检查返回的数据
        data_rows = result.get('data', [])
        logger.info(f"查询成功，返回 {len(data_rows)} 条数据记录")
        
        # 显示前几条数据的基本信息
        if data_rows:
            logger.info("前5条数据记录预览:")
            for i, row in enumerate(data_rows[:5], 1):
                logger.info(f"  记录{i}: {dict(list(row.items())[:3])}...")  # 只显示前3个字段
    else:
        logger.error("Geostore请求查询失败")
    
    return result

def save_results_to_file(data, filename_prefix="sdpt_limited_results"):
    """
    将查询结果保存到JSON文件
    
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
            logger.info(f"查询结果已保存到文件: {filename}")
            
            # 显示文件大小
            file_size = os.path.getsize(filename)
            logger.info(f"文件大小: {file_size / 1024:.2f} KB")
            
        except Exception as e:
            logger.error(f"保存文件失败: {str(e)}")

def analyze_query_results(data):
    """
    分析查询结果的基本统计信息
    
    参数:
        data (dict): 查询结果数据
    """
    if not data:
        logger.warning("没有数据可供分析")
        return
    
    logger.info("=" * 50)
    logger.info("查询结果分析")
    logger.info("=" * 50)
    
    # 获取数据记录
    data_rows = data.get('data', [])
    
    if not data_rows:
        logger.info("查询结果为空")
        return
    
    logger.info(f"总记录数: {len(data_rows)}")
    
    # 分析字段信息
    if data_rows:
        first_row = data_rows[0]
        fields = list(first_row.keys())
        logger.info(f"字段数量: {len(fields)}")
        logger.info(f"字段列表: {', '.join(fields)}")
        
        # 分析数值字段的统计信息
        numeric_fields = []
        for field in fields:
            try:
                # 检查是否为数值字段
                values = [row.get(field) for row in data_rows[:100]]  # 只检查前100条记录
                numeric_values = [v for v in values if isinstance(v, (int, float)) and v is not None]
                
                if len(numeric_values) > len(values) * 0.5:  # 如果超过50%是数值
                    numeric_fields.append(field)
            except:
                continue
        
        if numeric_fields:
            logger.info(f"数值字段: {', '.join(numeric_fields)}")
    
    logger.info("=" * 50)

def main():
    """
    主函数 - 执行SDPT数据区域限制查询流程
    """
    global logger
    logger = setup_logging()
    
    logger.info("开始执行SDPT数据区域限制查询脚本")
    logger.info(f"使用API密钥: {API_KEY[:10]}...")
    logger.info(f"数据集: {DATASET_ID}")
    logger.info(f"版本: {DATASET_VERSION}")
    
    try:
        # 步骤1: 读取GeoJSON文件
        logger.info("\n" + "="*50)
        logger.info("步骤1: 读取研究区域GeoJSON文件")
        logger.info("="*50)
        
        geojson_data = load_geojson(GEOJSON_FILE)
        if not geojson_data:
            logger.error("无法读取GeoJSON文件，程序终止")
            return False
        
        # 获取第一个要素的几何信息
        first_feature = geojson_data['features'][0]
        geometry = first_feature['geometry']
        
        # 步骤2: 使用POST请求方法查询数据
        logger.info("\n" + "="*50)
        logger.info("步骤2: 使用POST请求方法查询SDPT数据")
        logger.info("="*50)
        
        # 定义SQL查询语句 - 查询所有数据并添加地理位置信息
        sql_query = "SELECT *, ST_X(ST_Centroid(the_geom)) as longitude, ST_Y(ST_Centroid(the_geom)) as latitude FROM data"  # 查询所有数据记录并获取经纬度
        
        post_result = query_sdpt_with_post(geometry, sql_query)
        if post_result:
            save_results_to_file(post_result, "sdpt_post_query")
            analyze_query_results(post_result)
        
        # 等待一段时间避免API限制
        logger.info("等待5秒钟...")
        time.sleep(5)
        
        # 步骤3: 使用Geostore方法查询数据
        logger.info("\n" + "="*50)
        logger.info("步骤3: 使用Geostore方法查询SDPT数据")
        logger.info("="*50)
        
        # 创建Geostore
        geostore_id = create_geostore(geojson_data)
        
        if geostore_id:
            # 等待一段时间确保Geostore创建完成
            logger.info("等待3秒钟确保Geostore创建完成...")
            time.sleep(3)
            
            # 使用Geostore查询数据
            geostore_result = query_sdpt_with_geostore(geostore_id, sql_query)
            if geostore_result:
                save_results_to_file(geostore_result, "sdpt_geostore_query")
                analyze_query_results(geostore_result)
        else:
            logger.warning("Geostore创建失败，跳过Geostore方法查询")
        
        logger.info("\n" + "="*50)
        logger.info("SDPT数据区域限制查询脚本执行完成！")
        logger.info("="*50)
        
        return True
        
    except Exception as e:
        logger.error(f"脚本执行过程中发生错误: {str(e)}")
        return False

if __name__ == "__main__":
    # 执行主函数
    success = main()
    
    if success:
        print("\n✅ 脚本执行成功！请查看日志文件和结果文件获取详细信息。")
    else:
        print("\n❌ 脚本执行失败！请查看日志文件了解错误详情。")