#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GFW数据快速下载器 - 支持断点续传、VPN代理、多线程下载
目的：针对大文件提供高效的下载解决方案，支持断点续传和代理设置
作者：锐多宝 (ruiduobao)
"""

import requests
import os
import time
import threading
from datetime import datetime
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

class FastDownloader:
    def __init__(self, use_proxy=True, proxy_port=7890, max_workers=8, chunk_size=1024*1024):
        """
        初始化快速下载器
        
        Args:
            use_proxy (bool): 是否使用VPN代理
            proxy_port (int): 代理端口号
            max_workers (int): 最大线程数
            chunk_size (int): 每个线程下载的块大小(字节)
        """
        self.use_proxy = use_proxy
        self.proxy_port = proxy_port
        self.max_workers = max_workers
        self.chunk_size = chunk_size
        self.session = requests.Session()
        
        # 设置代理
        if self.use_proxy:
            proxies = {
                'http': f'http://127.0.0.1:{proxy_port}',
                'https': f'http://127.0.0.1:{proxy_port}'
            }
            self.session.proxies.update(proxies)
            print(f"🌐 已配置VPN代理: 127.0.0.1:{proxy_port}")
        
        # 设置请求头
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # 设置日志
        self.setup_logging()
    
    def setup_logging(self):
        """设置日志记录"""
        log_filename = f"download_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        print(f"📝 日志文件: {log_filename}")
    
    def get_file_size(self, url, headers=None):
        """获取远程文件大小"""
        try:
            response = self.session.head(url, headers=headers or {})
            if response.status_code == 200:
                return int(response.headers.get('content-length', 0))
            else:
                # 如果HEAD请求失败，尝试GET请求获取部分内容
                response = self.session.get(url, headers=headers or {}, stream=True)
                size = int(response.headers.get('content-length', 0))
                response.close()
                return size
        except Exception as e:
            self.logger.error(f"获取文件大小失败: {str(e)}")
            return 0
    
    def download_chunk(self, url, headers, start, end, filename, chunk_id):
        """下载文件的一个块"""
        chunk_headers = headers.copy() if headers else {}
        chunk_headers['Range'] = f'bytes={start}-{end}'
        
        temp_filename = f"{filename}.part{chunk_id}"
        
        try:
            response = self.session.get(url, headers=chunk_headers, stream=True)
            
            if response.status_code in [206, 200]:  # 206: Partial Content, 200: OK
                with open(temp_filename, 'wb') as f:
                    downloaded = 0
                    for data in response.iter_content(chunk_size=8192):
                        if data:
                            f.write(data)
                            downloaded += len(data)
                
                self.logger.info(f"块 {chunk_id} 下载完成: {downloaded} 字节")
                return True, chunk_id, downloaded
            else:
                self.logger.error(f"块 {chunk_id} 下载失败: HTTP {response.status_code}")
                return False, chunk_id, 0
                
        except Exception as e:
            self.logger.error(f"块 {chunk_id} 下载异常: {str(e)}")
            return False, chunk_id, 0
    
    def merge_chunks(self, filename, total_chunks):
        """合并所有下载的块"""
        print("🔄 正在合并文件块...")
        
        with open(filename, 'wb') as output_file:
            for i in range(total_chunks):
                chunk_filename = f"{filename}.part{i}"
                if os.path.exists(chunk_filename):
                    with open(chunk_filename, 'rb') as chunk_file:
                        output_file.write(chunk_file.read())
                    os.remove(chunk_filename)  # 删除临时文件
                else:
                    self.logger.error(f"缺少文件块: {chunk_filename}")
                    return False
        
        print("✅ 文件合并完成!")
        return True
    
    def resume_download(self, filename):
        """检查是否可以断点续传"""
        if os.path.exists(filename):
            existing_size = os.path.getsize(filename)
            print(f"🔄 发现已存在文件，大小: {existing_size / (1024*1024):.2f} MB")
            return existing_size
        return 0
    
    def download_file(self, url, filename, headers=None, resume=True):
        """
        多线程下载文件，支持断点续传
        
        Args:
            url (str): 下载链接
            filename (str): 保存文件名
            headers (dict): 请求头
            resume (bool): 是否支持断点续传
        """
        print(f"🚀 开始下载: {filename}")
        print(f"🔗 下载链接: {url}")
        
        # 检查断点续传
        existing_size = 0
        if resume:
            existing_size = self.resume_download(filename)
        
        # 获取文件总大小
        total_size = self.get_file_size(url, headers)
        if total_size == 0:
            print("⚠️  无法获取文件大小，尝试单线程下载...")
            return self.single_thread_download(url, filename, headers, existing_size)
        
        print(f"📊 文件总大小: {total_size / (1024*1024):.2f} MB")
        
        # 如果文件已完整下载
        if existing_size >= total_size:
            print("✅ 文件已完整下载!")
            return True
        
        # 计算需要下载的范围
        remaining_size = total_size - existing_size
        
        # 计算线程数和每个线程的下载范围
        if remaining_size < self.chunk_size:
            # 文件太小，使用单线程
            num_threads = 1
        else:
            num_threads = min(self.max_workers, remaining_size // self.chunk_size + 1)
        
        chunk_size = remaining_size // num_threads
        
        print(f"🧵 使用 {num_threads} 个线程下载")
        
        # 创建下载任务
        download_tasks = []
        for i in range(num_threads):
            start = existing_size + i * chunk_size
            if i == num_threads - 1:
                end = total_size - 1  # 最后一个线程下载到文件末尾
            else:
                end = start + chunk_size - 1
            
            download_tasks.append((start, end, i))
        
        # 执行多线程下载
        start_time = time.time()
        completed_chunks = 0
        total_downloaded = existing_size
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_to_chunk = {
                executor.submit(self.download_chunk, url, headers, start, end, filename, chunk_id): chunk_id
                for start, end, chunk_id in download_tasks
            }
            
            for future in as_completed(future_to_chunk):
                success, chunk_id, downloaded = future.result()
                if success:
                    completed_chunks += 1
                    total_downloaded += downloaded
                    
                    # 显示进度
                    progress = (total_downloaded / total_size) * 100
                    elapsed_time = time.time() - start_time
                    speed = total_downloaded / elapsed_time / (1024*1024) if elapsed_time > 0 else 0
                    
                    print(f"\r📥 下载进度: {progress:.1f}% "
                          f"({total_downloaded/(1024*1024):.2f}/{total_size/(1024*1024):.2f} MB) "
                          f"速度: {speed:.2f} MB/s", end='', flush=True)
                else:
                    self.logger.error(f"块 {chunk_id} 下载失败")
        
        print()  # 换行
        
        # 检查是否所有块都下载成功
        if completed_chunks == num_threads:
            # 合并文件块
            if self.merge_chunks(filename, num_threads):
                elapsed_time = time.time() - start_time
                avg_speed = total_size / elapsed_time / (1024*1024) if elapsed_time > 0 else 0
                print(f"✅ 下载完成! 用时: {elapsed_time:.1f}秒, 平均速度: {avg_speed:.2f} MB/s")
                return True
            else:
                print("❌ 文件合并失败!")
                return False
        else:
            print(f"❌ 下载失败! 只完成了 {completed_chunks}/{num_threads} 个块")
            return False
    
    def single_thread_download(self, url, filename, headers=None, resume_size=0):
        """单线程下载（备用方案）"""
        print("📥 使用单线程下载...")
        
        download_headers = headers.copy() if headers else {}
        if resume_size > 0:
            download_headers['Range'] = f'bytes={resume_size}-'
            mode = 'ab'  # 追加模式
        else:
            mode = 'wb'   # 覆写模式
        
        try:
            response = self.session.get(url, headers=download_headers, stream=True)
            
            if response.status_code in [200, 206]:
                total_size = int(response.headers.get('content-length', 0)) + resume_size
                downloaded = resume_size
                
                start_time = time.time()
                
                with open(filename, mode) as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            if total_size > 0:
                                progress = (downloaded / total_size) * 100
                                elapsed_time = time.time() - start_time
                                speed = (downloaded - resume_size) / elapsed_time / (1024*1024) if elapsed_time > 0 else 0
                                
                                print(f"\r📥 下载进度: {progress:.1f}% "
                                      f"({downloaded/(1024*1024):.2f}/{total_size/(1024*1024):.2f} MB) "
                                      f"速度: {speed:.2f} MB/s", end='', flush=True)
                
                print()
                print("✅ 单线程下载完成!")
                return True
            else:
                print(f"❌ 下载失败: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"单线程下载异常: {str(e)}")
            return False

def main():
    """主函数 - 下载GFW数据"""
    # 从之前的脚本输出中提取下载信息
    download_url = "https://data-api.globalforestwatch.org/dataset/gfw_planted_forests/latest/download/gpkg?geostore_id=0c6aea80-150d-866a-030f-c892d7f76757"
    api_key = "81a61ed9-254d-4974-8097-385a346f721b"
    filename = f"gfw_planted_forests_{datetime.now().strftime('%Y%m%d_%H%M%S')}.gpkg"
    
    # 设置请求头
    headers = {
        'x-api-key': api_key,
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    print("🌲 GFW数据快速下载器")
    print("=" * 60)
    print(f"📁 目标文件: {filename}")
    print(f"🔗 下载链接: {download_url}")
    print("=" * 60)
    
    # 创建下载器实例
    downloader = FastDownloader(
        use_proxy=True,      # 使用VPN代理
        proxy_port=7890,     # VPN端口
        max_workers=8,       # 8个线程
        chunk_size=10*1024*1024  # 每个线程10MB
    )
    
    # 开始下载
    success = downloader.download_file(download_url, filename, headers, resume=True)
    
    if success:
        file_size = os.path.getsize(filename) / (1024*1024)
        print(f"🎉 下载成功! 文件大小: {file_size:.2f} MB")
        print(f"📁 保存位置: {os.path.abspath(filename)}")
    else:
        print("❌ 下载失败!")
    
    print("=" * 60)

if __name__ == "__main__":
    main()