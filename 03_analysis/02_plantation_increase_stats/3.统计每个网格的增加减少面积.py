"""
# 脚本目的：
# 统计鱼网格矢量中每个网格范围内的栅格像素数量：
# - NF_count：人工林增加（像素值=1）的像素个数
# - RF_count：自然林减少（像素值=2）的像素个数
# 技术要点：分块(window)读取栅格、统一坐标系、可选多进程并行、日志记录与备份。
"""

# 导入标准库
import os  # 文件路径处理
import sys  # 系统交互（如异常退出）
import shutil  # 备份文件
import datetime  # 时间戳生成
import logging  # 日志记录
from typing import Tuple  # 类型标注

# 导入科学与GIS库
import numpy as np  # 数值计算
import geopandas as gpd  # 读取与写出矢量数据
from shapely.geometry import shape  # 几何构造
from shapely.wkb import loads as wkb_loads  # WKB反序列化（用于多进程安全传递几何）
import rasterio  # 栅格读取
from rasterio.windows import from_bounds  # 根据几何外接矩形生成窗口
from rasterio.features import geometry_mask  # 将几何转为掩膜

# 可选：多进程并行
from concurrent.futures import ProcessPoolExecutor, as_completed  # 多进程并行

# ------------------------ 配置区 ------------------------
# 输入矢量与栅格路径（使用原始字符串避免反斜杠转义）
FISHNET_SHP = r"Z:\Mywork\论文\东南亚10m人工林提取\制图\3.逐年变化图\人工林增长和自然林减少图\fishnet.shp"  # 鱼网格矢量
RASTER_TIF = r"Z:\Mywork\论文\东南亚10m人工林提取\数据\正式分类_10.29\5.GEE导出结果_结果合并_马尔可夫模型_逐年合并_裁剪东南亚_人工林增长和自然林消失\plantation_growth_and_natural_forest_disappearance_等面积投影.tif"  # 分类栅格

# 输出设置：在矢量同目录写入新文件与日志
OUTPUT_DIR = os.path.dirname(FISHNET_SHP)  # 输出目录选择为矢量所在文件夹
OUTPUT_SHP = os.path.join(OUTPUT_DIR, "fishnet_with_counts.shp")  # 输出矢量文件名
LOG_PATH = os.path.join(OUTPUT_DIR, "fishnet_count_log.txt")  # 日志文件路径

# 统计目标像素值
VALUE_NF = 1  # 人工林增加像素值
VALUE_RF = 2  # 自然林减少像素值

# 并行参数
ENABLE_MULTIPROCESS = True  # 是否启用多进程并行
MAX_WORKERS = max(1, os.cpu_count() - 25)  # 进程数（保留1个CPU给系统）

# ------------------------------------------------------

def setup_logger() -> logging.Logger:
    """配置日志记录器（同时写文件与终端）。"""
    logger = logging.getLogger("fishnet_count")  # 获取日志器
    logger.setLevel(logging.INFO)  # 设置日志级别
    # 文件处理器
    fh = logging.FileHandler(LOG_PATH, encoding="utf-8")  # 写入日志文件
    fh.setLevel(logging.INFO)  # 文件日志级别
    # 终端处理器
    ch = logging.StreamHandler(sys.stdout)  # 输出到终端
    ch.setLevel(logging.INFO)  # 终端日志级别
    # 日志格式
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")  # 格式化器
    fh.setFormatter(formatter)  # 文件使用格式化器
    ch.setFormatter(formatter)  # 终端使用格式化器
    # 避免重复添加处理器
    if not logger.handlers:  # 若尚未添加处理器
        logger.addHandler(fh)  # 添加文件处理器
        logger.addHandler(ch)  # 添加终端处理器
    return logger  # 返回日志器


def backup_shapefile(shp_path: str, logger: logging.Logger) -> None:
    """备份矢量shapefile（含配套文件）。"""
    base = os.path.splitext(shp_path)[0]  # 基础文件名（不含扩展名）
    exts = [".shp", ".shx", ".dbf", ".prj", ".cpg", ".qmd"]  # 可能存在的配套扩展
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # 时间戳
    backup_dir = os.path.join(os.path.dirname(shp_path), f"backup_{ts}")  # 备份目录
    os.makedirs(backup_dir, exist_ok=True)  # 创建备份目录
    copied = []  # 记录已复制文件
    for ext in exts:  # 遍历扩展名
        src = base + ext  # 源文件路径
        if os.path.exists(src):  # 若文件存在
            dst = os.path.join(backup_dir, os.path.basename(src))  # 目标路径
            shutil.copy2(src, dst)  # 复制文件（保留元数据）
            copied.append(src)  # 记录复制
    logger.info(f"已备份矢量文件组件到: {backup_dir}; 备份文件数: {len(copied)}")  # 记录备份情况


def count_values_for_polygon(raster_path: str, polygon_wkb: bytes, value_nf: int, value_rf: int) -> Tuple[int, int]:
    """为单个多边形统计两类像素值的数量。"""
    poly = wkb_loads(polygon_wkb)  # 反序列化得到几何
    with rasterio.open(raster_path) as src:  # 打开栅格数据集
        window = from_bounds(*poly.bounds, transform=src.transform)  # 根据外接矩形生成窗口
        if window.width <= 0 or window.height <= 0:  # 若窗口无有效像素
            return 0, 0  # 返回零计数
        data = src.read(1, window=window, boundless=True, masked=True)  # 读取窗口内第一波段，允许越界并保留nodata掩膜
        if data.size == 0:  # 若无像素
            return 0, 0  # 返回零计数
        w_transform = src.window_transform(window)  # 获取窗口仿射变换，用于生成掩膜
        poly_mask = geometry_mask([poly.__geo_interface__], out_shape=data.shape, transform=w_transform, invert=True)  # 生成多边形内部为True的掩膜
        combined_mask = np.logical_or(data.mask, ~poly_mask)  # 合并掩膜：排除nodata与多边形外部
        arr = np.ma.array(data.data, mask=combined_mask)  # 构造最终掩膜数组
        valid = ~arr.mask  # 有效像素布尔掩膜
        nf_count = int(np.sum((arr.data == value_nf) & valid))  # 统计像素值为1的数量
        rf_count = int(np.sum((arr.data == value_rf) & valid))  # 统计像素值为2的数量
        return nf_count, rf_count  # 返回两个计数


def main():
    """主流程：读取数据、统一坐标、并行统计、写出结果与日志。"""
    logger = setup_logger()  # 初始化日志器
    logger.info("任务开始：统计每个鱼网格的NF_count与RF_count")  # 开始日志
    logger.info(f"输入矢量: {FISHNET_SHP}")  # 记录矢量路径
    logger.info(f"输入栅格: {RASTER_TIF}")  # 记录栅格路径

    # 备份原始矢量
    try:
        backup_shapefile(FISHNET_SHP, logger)  # 执行备份
    except Exception as e:
        logger.warning(f"备份矢量文件失败：{e}")  # 记录备份失败但不中断

    # 读取数据
    fishnet = gpd.read_file(FISHNET_SHP)  # 读取鱼网格矢量
    logger.info(f"鱼网格要素数: {len(fishnet)}")  # 记录要素数量
    with rasterio.open(RASTER_TIF) as src:  # 打开栅格
        raster_crs = src.crs  # 栅格坐标系
        raster_transform = src.transform  # 仿射变换（未直接使用，仅校验）
        raster_nodata = src.nodata  # nodata值（供参考）
        logger.info(f"栅格CRS: {raster_crs}; nodata: {raster_nodata}")  # 记录栅格信息

    # 坐标统一
    if fishnet.crs != raster_crs:  # 若坐标系不一致
        logger.info(f"坐标系不一致，开始将鱼网格从 {fishnet.crs} 重投影到 {raster_crs}")  # 记录重投影
        fishnet = fishnet.to_crs(raster_crs)  # 重投影到栅格CRS
        logger.info("重投影完成")  # 完成提示

    # 准备并行输入（使用WKB减少跨进程序列化开销）
    polygons_wkb = [geom.wkb for geom in fishnet.geometry]  # 将几何转WKB列表

    # 执行统计
    nf_list = []  # 保存NF_count结果
    rf_list = []  # 保存RF_count结果
    if ENABLE_MULTIPROCESS and len(polygons_wkb) > 0:  # 条件允许并行
        logger.info(f"启用多进程并行，进程数: {MAX_WORKERS}")  # 记录并行参数
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:  # 创建进程池
            futures = {ex.submit(count_values_for_polygon, RASTER_TIF, wkb, VALUE_NF, VALUE_RF): i for i, wkb in enumerate(polygons_wkb)}  # 提交任务
            for fut in as_completed(futures):  # 迭代已完成任务
                idx = futures[fut]  # 获取索引
                try:
                    nf, rf = fut.result()  # 获取结果
                except Exception as e:
                    logger.error(f"要素索引 {idx} 统计失败：{e}")  # 记录错误
                    nf, rf = 0, 0  # 失败则记零
                nf_list.append(nf)  # 记录NF_count
                rf_list.append(rf)  # 记录RF_count
                if (idx + 1) % 1000 == 0:  # 每1000个要素提示进度
                    logger.info(f"已完成 {idx + 1} / {len(polygons_wkb)} 个网格")  # 记录进度
    else:
        logger.info("不启用并行，改为串行统计")  # 记录串行模式
        for i, wkb in enumerate(polygons_wkb):  # 逐个处理
            try:
                nf, rf = count_values_for_polygon(RASTER_TIF, wkb, VALUE_NF, VALUE_RF)  # 统计当前网格
            except Exception as e:
                logger.error(f"要素索引 {i} 统计失败：{e}")  # 记录错误
                nf, rf = 0, 0  # 失败则记零
            nf_list.append(nf)  # 保存结果
            rf_list.append(rf)  # 保存结果
            if (i + 1) % 1000 == 0:  # 每1000要素提示
                logger.info(f"已完成 {i + 1} / {len(polygons_wkb)} 个网格")  # 进度日志

    # 写入属性表
    fishnet["NF_count"] = np.array(nf_list, dtype=np.int32)  # 新增NF_count列
    fishnet["RF_count"] = np.array(rf_list, dtype=np.int32)  # 新增RF_count列
    logger.info("已将统计结果写入GeoDataFrame列：NF_count, RF_count")  # 记录写入

    # 输出新矢量文件（避免破坏原始数据）
    fishnet.to_file(OUTPUT_SHP)  # 写出到新的shp
    logger.info(f"已写出结果到: {OUTPUT_SHP}")  # 记录输出路径

    # 简单验证：汇总统计输出
    total_nf = int(np.sum(fishnet["NF_count"]))  # 汇总NF
    total_rf = int(np.sum(fishnet["RF_count"]))  # 汇总RF
    logger.info(f"总NF_count: {total_nf}; 总RF_count: {total_rf}")  # 记录汇总
    logger.info("任务完成")  # 完成日志


if __name__ == "__main__":  # 脚本入口
    main()  # 执行主流程
