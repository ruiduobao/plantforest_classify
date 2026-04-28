#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GFW人工林数据集查询脚本
目的：使用指定的几何范围查询GFW人工林数据集
作者：锐多宝 (ruiduobao)
"""

import requests
import json
from datetime import datetime

def query_gfw_planted_forests():
    """
    查询GFW人工林数据集
    """
    # API配置
    api_key = "81a61ed9-254d-4974-8097-385a346f721b"  # API密钥
    url = "https://data-api.globalforestwatch.org/dataset/gfw_planted_forests/latest/query/json"  # 查询端点
    
    # 请求头
    headers = {
        'x-api-key': api_key,  # API密钥认证
        'Content-Type': 'application/json'  # 内容类型
    }
    
    # 查询参数
    payload = {
        "sql": "SELECT * FROM data LIMIT 1",  # SQL查询语句，使用data表，限制返回1条记录
        "geometry": {  # 几何范围定义
            "type": "Polygon",  # 几何类型为多边形
            "coordinates": [[  # 多边形坐标点
                [-57.39, -20.27],  # 左下角
                [-57.39, 23.22],   # 左上角
                [-54.44, -23.22],  # 右下角
                [-54.44, 20.27],   # 右上角
                [-57.39, -20.27]   # 闭合到起始点
            ]]
        }
    }
    
    try:
        print(f"开始查询GFW人工林数据集...")
        print(f"查询时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"使用API密钥: {api_key[:10]}...")  # 只显示前10位用于调试
        print(f"请求URL: {url}")
        
        # 发送POST请求
        response = requests.post(url, headers=headers, json=payload)
        
        # 检查响应状态
        if response.status_code == 200:
            result = response.json()  # 解析JSON响应
            print(f"查询成功！返回数据条数: {len(result.get('data', []))}")
            
            # 显示数据结构
            if result.get('data'):
                print(f"数据字段: {list(result['data'][0].keys())}")
            
            # 保存结果到文件
            output_file = f"gfw_planted_forests_query_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"结果已保存到: {output_file}")
            
        else:
            print(f"查询失败！状态码: {response.status_code}")
            print(f"错误信息: {response.text}")
            
    except Exception as e:
        print(f"查询过程中发生错误: {str(e)}")

if __name__ == "__main__":
    query_gfw_planted_forests()  # 执行查询函数