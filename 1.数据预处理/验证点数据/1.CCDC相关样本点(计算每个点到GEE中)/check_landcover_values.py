# 检查土地覆盖数据的像素值分布
import rasterio
import numpy as np
from rasterio.windows import Window
import os

def check_landcover_values():
    """检查土地覆盖数据的像素值分布"""
    landcover_file = r"F:\BaiduSyncdisk\论文\东南亚10m人工林提取\数据\土地覆盖数据\southeast_asia_landcover_2024_mosaic.tif"
    
    print("正在读取土地覆盖数据...")
    
    # 首先检查文件基本信息
    print(f"文件路径: {landcover_file}")
    print(f"文件存在: {os.path.exists(landcover_file)}")
    if os.path.exists(landcover_file):
        file_size = os.path.getsize(landcover_file)
        print(f"文件大小: {file_size / (1024*1024):.2f} MB")
    
    try:
        with rasterio.open(landcover_file) as src:
            print(f"\n=== 栅格基本信息 ===")
            print(f"栅格大小: {src.width}x{src.height}")
            print(f"波段数: {src.count}")
            print(f"NoData值: {src.nodata}")
            print(f"数据类型: {src.dtypes[0]}")
            print(f"坐标系: {src.crs}")
            print(f"范围: {src.bounds}")
            print(f"像素分辨率: {src.res}")
            print(f"压缩方式: {src.compression}")
            print(f"内部瓦片: {src.is_tiled}")
            if src.is_tiled:
                print(f"瓦片大小: {src.block_shapes}")
            
            # 检查栅格统计信息
            try:
                print(f"\n=== 栅格统计信息 ===")
                stats = src.statistics(1)
                print(f"最小值: {stats.min}")
                print(f"最大值: {stats.max}")
                print(f"平均值: {stats.mean}")
                print(f"标准差: {stats.std}")
            except Exception as e:
                print(f"无法获取统计信息: {e}")
            
            # 使用块读取方式
            print(f"\n=== 块读取检查 ===")
            
            # 获取第一个块的信息
            block_windows = list(src.block_windows(1))
            print(f"总块数: {len(block_windows)}")
            
            all_values = []
            blocks_with_data = 0
            
            # 检查前10个块
            for i, (block_id, window) in enumerate(block_windows[:10]):
                try:
                    print(f"\n检查块 {i+1}/{min(10, len(block_windows))}: {window}")
                    data = src.read(1, window=window)
                    
                    print(f"  块大小: {data.shape}")
                    unique_vals = np.unique(data)
                    print(f"  唯一值: {unique_vals}")
                    
                    # 统计每个值的数量
                    vals, counts = np.unique(data, return_counts=True)
                    for val, count in zip(vals, counts):
                        print(f"    值 {val}: {count} 个像素")
                    
                    # 检查是否有非零值
                    nonzero_data = data[data != 0]
                    if len(nonzero_data) > 0:
                        print(f"  *** 找到 {len(nonzero_data)} 个非零像素! ***")
                        all_values.extend(nonzero_data.flatten())
                        blocks_with_data += 1
                        
                        # 如果找到足够的数据就停止
                        if len(all_values) > 1000:
                            break
                            
                except Exception as e:
                    print(f"  读取块失败: {e}")
            
            print(f"\n有数据的块数: {blocks_with_data}")
            
            # 如果前10个块没有数据，尝试随机采样更多块
            if not all_values and len(block_windows) > 10:
                print(f"\n=== 随机采样更多块 ===")
                import random
                random_blocks = random.sample(block_windows, min(50, len(block_windows)))
                
                for i, (block_id, window) in enumerate(random_blocks):
                    try:
                        data = src.read(1, window=window)
                        nonzero_data = data[data != 0]
                        if len(nonzero_data) > 0:
                            print(f"块 {i+1}: 找到 {len(nonzero_data)} 个非零像素")
                            unique_vals = np.unique(nonzero_data)
                            print(f"  唯一非零值: {unique_vals}")
                            all_values.extend(nonzero_data.flatten())
                            blocks_with_data += 1
                            
                            if len(all_values) > 1000:
                                break
                    except Exception as e:
                        continue
            
            # 总体统计
            if all_values:
                print(f"\n=== 总体统计 ===")
                all_values = np.array(all_values)
                unique_vals, counts = np.unique(all_values, return_counts=True)
                print(f"总非零像素数: {len(all_values)}")
                print(f"唯一值数量: {len(unique_vals)}")
                print(f"\n完整像素值分布:")
                for val, count in zip(unique_vals, counts):
                    print(f"  值 {val}: {count} 个像素 ({count/len(all_values)*100:.2f}%)")
                
                # 检查是否有值为2的像素（林地）
                if 2 in unique_vals:
                    forest_count = counts[unique_vals == 2][0]
                    print(f"\n*** 发现林地像素 (值=2): {forest_count} 个 ({forest_count/len(all_values)*100:.2f}%) ***")
                else:
                    print(f"\n*** 警告: 没有发现值为2的林地像素! ***")
                    print(f"实际发现的像素值:")
                    for val in unique_vals:
                        count = counts[unique_vals == val][0]
                        print(f"  值 {val}: {count} 个像素")
                        
                    # 建议可能的林地值
                    if len(unique_vals) > 0:
                        print(f"\n可能的林地值候选 (建议修改FOREST_VALUE):")
                        for val in sorted(unique_vals):
                            count = counts[unique_vals == val][0]
                            print(f"  候选值 {val}: {count} 个像素 ({count/len(all_values)*100:.2f}%)")
            else:
                print(f"\n*** 问题: 在检查的 {blocks_with_data} 个块中没有找到非零像素! ***")
                print(f"可能的原因:")
                print(f"1. 数据确实全部为0或NoData")
                print(f"2. 数据压缩方式特殊，需要特殊处理")
                print(f"3. 数据存储格式问题")
                
                # 尝试读取整个数据集的一个小样本
                print(f"\n=== 尝试直接读取小样本 ===")
                try:
                    # 读取中心的一个小区域
                    center_x = src.width // 2
                    center_y = src.height // 2
                    sample_data = src.read(1, window=Window(center_x-50, center_y-50, 100, 100))
                    print(f"中心样本唯一值: {np.unique(sample_data)}")
                    
                    # 尝试读取不同位置的样本
                    positions = [(0, 0), (src.width//4, src.height//4), 
                               (src.width*3//4, src.height*3//4)]
                    for i, (x, y) in enumerate(positions):
                        try:
                            sample = src.read(1, window=Window(x, y, 100, 100))
                            unique_sample = np.unique(sample)
                            print(f"位置 {i+1} ({x},{y}) 唯一值: {unique_sample}")
                        except:
                            continue
                            
                except Exception as e:
                    print(f"直接读取失败: {e}")
                    
    except Exception as e:
        print(f"打开文件失败: {e}")
        print(f"可能的原因:")
        print(f"1. 文件损坏")
        print(f"2. 文件格式不支持")
        print(f"3. 权限问题")

if __name__ == "__main__":
    check_landcover_values()