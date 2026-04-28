#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GFW人工林数据下载脚本
目的：基于查询结果下载指定区域的GFW人工林数据，自动判断并下载可用格式
作者：锐多宝 (ruiduobao)
"""

import requests
import json
import os
from datetime import datetime

def load_query_results(json_file):
    """读取查询结果JSON文件"""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ 成功读取查询结果: {json_file}")
        return data
    except Exception as e:
        print(f"❌ 读取文件失败: {str(e)}")
        return None

def get_dataset_assets(api_key, dataset_id, version="latest"):
    """获取数据集支持的资产格式"""
    url = f"https://data-api.globalforestwatch.org/dataset/{dataset_id}/{version}/assets"
    headers = {'x-api-key': api_key}
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            assets = data.get('data', [])
            downloadable_assets = [asset for asset in assets if asset.get('is_downloadable', False)]
            print(f"📋 数据集支持 {len(downloadable_assets)} 种可下载格式:")
            
            for asset in downloadable_assets:
                asset_type = asset.get('asset_type', 'Unknown')
                print(f"   - {asset_type}")
                
                # 如果是栅格瓦片集，提取grid和pixel_meaning信息
                if asset_type == 'Raster tile set':
                    asset_uri = asset.get('asset_uri', '')
                    asset_id = asset.get('asset_id', '')
                    print(f"     资产ID: {asset_id}")
                    
                    # 从asset_uri中解析grid和pixel_meaning
                    # 例如: s3://.../espy-3857/zoom_0/coverage_gradient/geotiff/....tif
                    if asset_uri:
                        uri_parts = asset_uri.split('/')
                        for i, part in enumerate(uri_parts):
                            if part.startswith('zoom_'):
                                grid = part
                                if i + 1 < len(uri_parts):
                                    pixel_meaning = uri_parts[i + 1]
                                    print(f"     网格: {grid}, 像素含义: {pixel_meaning}")
                                    # 将这些信息存储到asset中以便后续使用
                                    asset['grid'] = grid
                                    asset['pixel_meaning'] = pixel_meaning
                                break
            
            return downloadable_assets
        else:
            print(f"⚠️  获取资产信息失败: {response.status_code}")
            return []
    except Exception as e:
        print(f"❌ 获取资产信息异常: {str(e)}")
        return []

def get_tiles_info(api_key, asset_id):
    """获取指定资产的瓦片信息"""
    url = f"https://data-api.globalforestwatch.org/asset/{asset_id}/tiles_info"
    headers = {'x-api-key': api_key}
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            tiles = data.get('data', [])
            print(f"🗂️  资产 {asset_id} 包含 {len(tiles)} 个瓦片")
            
            # 显示前几个瓦片的信息
            for i, tile in enumerate(tiles[:3]):  # 只显示前3个
                tile_id = tile.get('tile_id', 'Unknown')
                print(f"   瓦片 {i+1}: {tile_id}")
            
            if len(tiles) > 3:
                print(f"   ... 还有 {len(tiles) - 3} 个瓦片")
            
            return tiles
        else:
            print(f"⚠️  获取瓦片信息失败: {response.status_code}")
            return []
    except Exception as e:
        print(f"❌ 获取瓦片信息异常: {str(e)}")
        return []

def download_data(api_key, dataset_id, version, format_type, geostore_id, sql_query, asset_info=None):
    """下载指定格式的数据"""
    base_url = f"https://data-api.globalforestwatch.org/dataset/{dataset_id}/{version}/download"
    headers = {'x-api-key': api_key}
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    try:
        if format_type in ['json', 'csv']:
            # JSON/CSV格式使用POST请求
            url = f"{base_url}/{format_type}"
            payload = {"sql": sql_query}
            if geostore_id:
                payload["geostore_id"] = geostore_id
            
            print(f"📥 下载{format_type.upper()}格式...")
            print(f"🔗 下载URL: {url}")
            print(f"📋 请求头: {headers}")
            print(f"📦 POST数据: {json.dumps(payload, indent=2, ensure_ascii=False)}")
            
            response = requests.post(url, headers=headers, json=payload, stream=True)
            filename = f"gfw_planted_forests_{timestamp}.{format_type}"
            
        elif format_type in ['shp', 'gpkg']:
            # Shapefile/Geopackage格式使用GET请求
            url = f"{base_url}/{format_type}"
            params = {}
            if geostore_id:
                params["geostore_id"] = geostore_id
                url += f"?geostore_id={geostore_id}"
            
            print(f"📥 下载{format_type.upper()}格式...")
            print(f"🔗 下载URL: {url}")
            print(f"📋 请求头: {headers}")
            if params:
                print(f"📦 GET参数: {json.dumps(params, indent=2, ensure_ascii=False)}")
            
            # 生成curl命令供第三方工具使用
            curl_cmd = f'curl -X GET "{url}"'
            for key, value in headers.items():
                curl_cmd += f' -H "{key}: {value}"'
            print(f"💻 等效curl命令:")
            print(f"   {curl_cmd}")
            print(f"   输出文件: gfw_planted_forests_{timestamp}.{format_type}")
            
            response = requests.get(url, headers=headers, stream=True)
            ext = 'zip' if format_type == 'shp' else format_type
            filename = f"gfw_planted_forests_{timestamp}.{ext}"
            
        elif format_type == 'geotiff':
            # Geotiff格式需要特殊处理
            if not asset_info:
                print("❌ Geotiff格式需要资产信息")
                return False
            
            print(f"📥 下载Geotiff格式...")
            print(f"🗂️  资产信息: {asset_info}")
            
            # 获取瓦片信息
            tiles_info = get_tiles_info(api_key, asset_info['asset_id'])
            if not tiles_info or not tiles_info.get('tiles'):
                print("❌ 无法获取瓦片信息")
                return False
            
            # 选择第一个瓦片进行下载
            tile_id = tiles_info['tiles'][0]['tile_id']
            grid = asset_info.get('grid', 'zoom_0')
            pixel_meaning = asset_info.get('pixel_meaning', 'coverage_gradient')
            
            url = f"https://data-api.globalforestwatch.org/dataset/{dataset_id}/{version}/assets/{asset_info['asset_id']}/tiles/{tile_id}"
            
            print(f"🔗 下载URL: {url}")
            print(f"📋 请求头: {headers}")
            print(f"🗂️  瓦片ID: {tile_id}")
            print(f"🌐 网格: {grid}")
            print(f"🎨 像素含义: {pixel_meaning}")
            
            # 生成curl命令供第三方工具使用
            curl_cmd = f'curl -X GET "{url}"'
            for key, value in headers.items():
                curl_cmd += f' -H "{key}: {value}"'
            print(f"💻 等效curl命令:")
            print(f"   {curl_cmd}")
            print(f"   输出文件: gfw_planted_forests_{timestamp}.tif")
            
            response = requests.get(url, headers=headers, stream=True)
            filename = f"gfw_planted_forests_{timestamp}.tif"
            
        else:
            print(f"⚠️  不支持的格式: {format_type}")
            return False
        
        # 检查响应状态
        if response.status_code == 200:
            print(f"✅ 开始下载文件: {filename}")
            
            # 获取文件大小
            total_size = int(response.headers.get('content-length', 0))
            if total_size > 0:
                print(f"📊 文件大小: {total_size / (1024*1024):.2f} MB")
            
            # 下载文件并显示进度
            downloaded = 0
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"\r📥 下载进度: {progress:.1f}% ({downloaded / (1024*1024):.2f}/{total_size / (1024*1024):.2f} MB)", end='', flush=True)
            
            print(f"\n✅ {format_type.upper()}格式下载完成: {filename}")
            return True
        else:
            print(f"❌ {format_type.upper()}下载失败: {response.status_code} - {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"❌ {format_type.upper()}下载异常: {str(e)}")
        return False

def main():
    """主函数"""
    # 配置参数
    api_key = "81a61ed9-254d-4974-8097-385a346f721b"
    dataset_id = "gfw_planted_forests"
    version = "latest"
    query_file = r"F:\BaiduSyncdisk\论文\东南亚10m人工林提取\代码\1.数据预处理\SDPT下载\gfw_planted_forests_query_20250919_165906.json"
    
    print("🌲 GFW人工林数据下载器")
    print("=" * 50)
    
    # 读取查询结果
    query_data = load_query_results(query_file)
    if not query_data:
        return
    
    # 提取geostore_id和构建SQL查询
    geostore_id = None
    if query_data.get('data') and len(query_data['data']) > 0:
        geostore_id = query_data['data'][0].get('gfw_geostore_id')
        print(f"🎯 使用Geostore ID: {geostore_id}")
    
    sql_query = f"SELECT * FROM data WHERE gfw_geostore_id = '{geostore_id}'" if geostore_id else "SELECT * FROM data LIMIT 100"
    
    # 获取可用的下载格式
    assets = get_dataset_assets(api_key, dataset_id, version)
    if not assets:
        print("⚠️  未找到可下载的资产格式，尝试默认格式...")
        download_formats = ['json', 'csv']  # 默认尝试这些格式
        asset_mapping = {}
    else:
        # 根据资产类型映射到下载格式
        format_mapping = {
            'Geo database table': ['json', 'csv'],
            'Database table': ['json', 'csv'],
            'ESRI Shapefile': ['shp'],
            'Geopackage': ['gpkg'],
            'Raster tile set': ['geotiff']  # 栅格瓦片集支持geotiff格式
        }
        
        download_formats = []
        asset_mapping = {}  # 存储格式到资产信息的映射
        
        for asset in assets:
            asset_type = asset.get('asset_type', '')
            formats = format_mapping.get(asset_type, [])
            download_formats.extend(formats)
            
            # 为geotiff格式存储资产信息
            if asset_type == 'Raster tile set' and 'geotiff' in formats:
                asset_mapping['geotiff'] = asset
        
        # 去重并优先选择常用格式
        download_formats = list(dict.fromkeys(download_formats))  # 保持顺序去重
        if not download_formats:
            download_formats = ['json', 'csv']  # 默认格式
    
    print(f"📦 准备下载格式: {', '.join(download_formats)}")
    
    # 下载数据
    success_count = 0
    for fmt in download_formats:
        print(f"\n{'='*30}")
        # 获取对应格式的资产信息（如果有的话）
        asset_info = asset_mapping.get(fmt, None)
        success = download_data(api_key, dataset_id, version, fmt, geostore_id, sql_query, asset_info)
        if success:
            success_count += 1
    
    print(f"\n{'='*50}")
    print(f"🎉 下载完成! 成功下载 {success_count}/{len(download_formats)} 种格式")
    print("=" * 50)

if __name__ == "__main__":
    main()