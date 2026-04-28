# -*- coding: utf-8 -*-
"""
本脚本旨在分析2017年至2024年东南亚地区的土地利用变化，特别关注人工林和自然林的动态。
脚本会执行以下操作：
1. 读取2017年和2024年的土地利用分类栅格数据（TIF格式）。
2. 比较两年的数据，识别出以下四种变化类型：
   - 类型1: 自然林转为人工林
   - 类型2: 自然林消失（转为非林地）
3. 将分析结果保存为一个新的TIF文件，其中不同的变化类型用不同的像素值表示。
4. 使用多进程并行处理来加速计算，特别适合处理大尺寸的栅格数据。
5. 将详细的执行过程和错误信息记录到日志文件中。
"""

# ================================= 导入库 =================================
import rasterio  # 用于栅格数据读写的核心库
from rasterio.windows import Window  # 用于窗口化读写，节省内存
import numpy as np  # 用于高效的数组计算
import multiprocessing  # 用于实现多进程并行计算
import logging  # 用于记录程序运行日志
import time  # 用于计算程序总耗时
from tqdm import tqdm  # 用于在终端显示美观的进度条
import os  # 用于文件和路径操作

# ================================= 全局参数配置 =================================

# --- 输入文件路径 ---
# 2017年的分类数据，其中像素值1代表自然林
FILE_2017 = r"F:\人工林增长和自然林消失\SE_Asia_2017_PL_NL_classification.tif"
# 2024年的分类数据，其中像素值1代表人工林，0代表其他
FILE_2024 = r"F:\人工林增长和自然林消失\SE_Asia_2024_PL_NL_classification.tif"
# # 2017年的分类数据，其中像素值1代表自然林
# FILE_2017 = r"F:\人工林增长和自然林消失\zone1_2017.tif"
# # 2024年的分类数据，其中像素值1代表人工林，0代表其他
# FILE_2024 = r"F:\人工林增长和自然林消失\zone1_2024.tif"

output_dir=r"F:\人工林增长和自然林消失\output"
# --- 输出文件路径 ---

# 如果输出目录不存在，则创建它
os.makedirs(output_dir, exist_ok=True)
# 定义输出的TIF文件名
output_file = os.path.join(output_dir, "plantation_growth_and_natural_forest_disappearance_zone1.tif")
# 定义日志文件名
log_file = os.path.join(output_dir, "processing_log.log")

# --- 多进程配置 ---
# 设置并行处理的进程数量。为了最大化利用CPU，我们设置为30，为系统和主进程保留2个核心
NUM_PROCESSES = 30

# ================================= 日志配置 =================================
# 配置日志记录器
logging.basicConfig(
    level=logging.INFO,  # 记录INFO级别及以上的信息
    format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式
    handlers=[
        logging.FileHandler(log_file, mode='w'),  # 将日志写入文件，每次运行覆盖旧日志
        logging.StreamHandler()  # 同时将日志输出到终端
    ]
)

# ================================= 核心计算函数 =================================

def process_data_chunk(args):
    """
    子进程执行的函数，只负责计算，不接触文件I/O。
    """
    # 从元组中解包出数据和窗口的行列号元组
    data_2017, data_2024, window_slices = args
    try:
        # 初始化一个与输入块相同大小的结果数组，数据类型为8位无符号整型
        result = np.zeros_like(data_2017, dtype=np.uint8)
        
        # 条件1: 2017年其他(值不为1) 且 2024年是人工林(值为1) --> 其他转为人工林 (输出值为1)
        result[(data_2017 != 1) & (data_2024 == 1)] = 1
        
        # 条件2: 2017年是自然林(值为2) 且 2024年是其他(值为0) --> 自然林消失 (输出值为2)
        result[(data_2017 == 2) & (data_2024 != 2)] = 2
        
        # 返回计算结果和对应的窗口行列号元组
        return result, window_slices
    except Exception:
        # 在子进程中捕获任何异常，并返回None，避免整个池崩溃
        return None, None

# ================================= 主函数 =================================

def main():
    """
    主函数，负责调度整个处理流程：
    1. 打开输入和输出文件。
    2. 创建一个“主进程读写，子进程计算”的架构。
    3. 主进程负责从TIF文件中读取数据块，并将其分发给进程池。
    4. 子进程接收数据块，执行Numpy计算，然后返回结果。
    5. 主进程接收计算结果，并将其写回输出TIF文件的正确位置。
    """
    try:
        # 记录开始时间
        start_time = time.time()
        logging.info("开始处理...")

        # 使用rasterio打开两个输入文件
        with rasterio.open(FILE_2017) as src_2017, rasterio.open(FILE_2024) as src_2024:
            # 检查两个栅格的尺寸和地理变换是否一致，这是进行像素级比较的前提
            if src_2017.profile != src_2024.profile:
                logging.error("错误：两个输入TIF文件的空间参考或尺寸不匹配。")
                return

            # 复制源文件的元数据（profile），并为输出文件进行修改
            profile = src_2017.profile.copy()
            profile.update(
                dtype=rasterio.uint8,  # 将输出数据类型设为8位无符号整型
                count=1,  # 输出文件只有一个波段
                compress='lzw'  # 使用LZW无损压缩，减小文件体积
            )

            # 创建输出文件
            with rasterio.open(output_file, 'w', **profile) as dst:
                # 获取所有处理窗口，同时获取窗口对象和其对应的行列号元组
                windows_with_slices = [(win, win.toranges()) for _, win in src_2017.block_windows(1)]
                logging.info(f"启动 {NUM_PROCESSES} 个进程来处理 {len(windows_with_slices)} 个数据块...")

                def data_chunk_generator(windows_list):
                    """
                    一个生成器，负责从TIF文件中读取数据块并打包成元组。
                    主进程使用Window对象读取，但传递给子进程的是更安全的行列号元组。
                    """
                    for window_obj, window_slices in windows_list:
                        # 主进程读取数据
                        data_2017 = src_2017.read(1, window=window_obj)
                        data_2024 = src_2024.read(1, window=window_obj)
                        # yield一个元组，包含计算所需的所有数据，传递slices而不是window对象
                        yield (data_2017, data_2024, window_slices)
                
                # 创建多进程池
                # maxtasksperchild=10: 让每个子进程处理10个任务后重启，强制释放内存，增加系统稳定性
                with multiprocessing.Pool(processes=NUM_PROCESSES, maxtasksperchild=10) as pool:
                    # 使用tqdm显示进度条
                    with tqdm(total=len(windows_with_slices), desc="多进程计算中") as pbar:
                        # 使用imap_unordered来高效处理，它会按完成顺序返回结果，而不是提交顺序
                        for result_data, result_window_slices in pool.imap_unordered(process_data_chunk, data_chunk_generator(windows_with_slices)):
                            # 确保返回的结果是有效的
                            if result_data is not None and result_window_slices is not None:
                                # 主进程使用行列号元组写入结果
                                dst.write(result_data.astype(profile['dtype']), 1, window=Window.from_slices(*result_window_slices))
                            # 更新进度条
                            pbar.update(1)

    except Exception as e:
        # 捕获并记录主程序中的任何严重错误
        logging.error(f"主程序发生严重错误: {e}")
        import traceback
        logging.error(traceback.format_exc()) # 记录完整的错误堆栈
    finally:
        # 记录结束时间并计算总耗时
        end_time = time.time()
        logging.info(f"全过程处理完成。总耗时: {end_time - start_time:.2f} 秒。")

# ================================= 程序入口 =================================

if __name__ == '__main__':
    # 在Windows上，建议使用'spawn'启动方法以避免多进程相关的问题
    multiprocessing.set_start_method('spawn', force=True)
    main()

