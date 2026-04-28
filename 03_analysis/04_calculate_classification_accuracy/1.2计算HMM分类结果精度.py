# 脚本目的：对10个zone的逐年分类结果进行点位采样并计算精度指标（3类：1-人工林，2-自然林，3-其他），输出每个zone每年的混淆矩阵、用户精度、生产者精度、总体精度及年度汇总结果，采用高效采样与日志机制，确保在Windows环境下稳定运行。

# 说明：本脚本将按zone与year读取验证点（shapefile），依据year匹配对应的分类结果tif，使用rasterio的按点采样方式（无需整图读入，避免内存溢出），并把每个阶段的结果持久化为CSV，同时记录详细日志。

# 依赖库：geopandas、rasterio、numpy、pandas、scikit-learn、tqdm、logging
# 注意：请在运行前确保上述库已安装；采样采用rasterio.sample按点读取，满足“分块读取/避免整图进内存”的性能要求；并在每次文件打开后及时关闭以释放内存。

import os  # 导入os用于路径拼接和文件操作
import sys  # 导入sys用于异常捕获时的额外信息以及退出控制
import time  # 导入time用于记录耗时与时间戳
import gc  # 导入gc用于手动触发垃圾回收释放内存
from datetime import datetime  # 导入datetime用于生成日志文件名时间戳

import numpy as np  # 导入numpy用于数组与数学计算
import pandas as pd  # 导入pandas用于表格数据处理与CSV导出
import geopandas as gpd  # 导入geopandas用于读取shapefile和坐标系转换
import rasterio  # 导入rasterio用于读取栅格tif并进行按点采样
from rasterio.windows import Window  # 导入Window用于安全的像素窗口读取
from sklearn.metrics import confusion_matrix  # 导入confusion_matrix用于混淆矩阵计算
import logging  # 导入logging用于统一日志记录
from concurrent.futures import ProcessPoolExecutor, as_completed  # 导入并行执行器与结果收集
try:
    from tqdm import tqdm  # 尝试导入tqdm用于进度条显示
except ImportError:
    tqdm = None  # 若未安装tqdm，则设置为None以便降级为仅日志


# ============================ 路径与常量配置 ============================
# 设置验证点根目录（输出也写到此目录下），使用原始字符串避免反斜杠转义
VALIDATION_ROOT = r"D:\地理所\论文\东南亚10m人工林提取\数据\测试集"  # 验证点根目录

# 设置分类结果tif的根目录
CLASSIFICATION_ROOT = r"D:\地理所\论文\东南亚10m人工林提取\数据\正式分类_10.29\3.GEE导出结果_结果合并_马尔可夫模型"  # 分类结果根目录

# 定义zone范围与年份范围
ZONES = list(range(1, 11))  # 定义zone编号列表1-10
YEARS = list(range(2017, 2024 + 1))  # 定义年份列表2017-2024

# 定义类别标签（按题意：1-人工林，2-自然林，3-其他）
CLASS_LABELS = [1, 2, 3]  # 定义类别列表用于稳定的矩阵维度与指标计算

# 输出目录结构定义（在验证点根目录下创建results与logs）
OUTPUT_DIR = os.path.join(VALIDATION_ROOT, "precision_results_HMM")  # 每个zone每年结果与年度汇总的输出目录
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")  # 日志目录
SAMPLES_DIR = os.path.join(OUTPUT_DIR, "samples")  # 每个zone每年的采样对结果目录
MATRICES_DIR = os.path.join(OUTPUT_DIR, "matrices")  # 每个zone每年的混淆矩阵目录
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")  # 每个zone每年的指标目录
SUMMARY_DIR = os.path.join(OUTPUT_DIR, "summary")  # 年度汇总输出目录

# 并行配置（可按需要调整或关闭）
USE_MULTIPROCESS = True  # 是否启用多进程并行处理
MAX_WORKERS = min(8, max(1, (os.cpu_count() or 4) - 1))  # 并行进程数限制为最多8，至少1


# ============================ 日志初始化 ============================
def setup_logging():
    """初始化日志记录到文件与控制台。"""  # 函数注释说明功能
    # 确保输出目录存在
    os.makedirs(LOG_DIR, exist_ok=True)  # 创建日志目录（若不存在）
    # 生成日志文件名（包含时间戳）
    log_filename = f"1_计算分类结果精度_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"  # 定义日志文件名
    log_path = os.path.join(LOG_DIR, log_filename)  # 拼接完整日志路径
    # 配置日志格式与处理器（文件与控制台）
    logging.basicConfig(  # 设置日志基础配置
        level=logging.INFO,  # 设置日志级别为INFO
        format="%(asctime)s - %(levelname)s - %(message)s",  # 设置日志格式
        handlers=[  # 设置日志输出处理器列表
            logging.FileHandler(log_path, encoding="utf-8"),  # 文件处理器输出到日志文件
            logging.StreamHandler(sys.stdout),  # 控制台处理器输出到终端
        ],
    )
    # 返回日志文件路径以便在终端打印提示
    return log_path  # 返回日志路径


# ============================ 路径构造工具 ============================
def zone_shp_path(zone: int) -> str:
    """给定zone编号返回验证点shapefile路径。"""  # 函数注释说明输入输出
    # 依据题意生成文件名（Zone{n}_Merged_AllBand_Sample_filtered_test.shp）
    shp_name = f"Zone{zone}_Merged_AllBand_Sample_filtered_test.shp"  # 生成shp文件名
    # 返回完整路径
    return os.path.join(VALIDATION_ROOT, shp_name)  # 拼接并返回shp路径


def zone_year_tif_path(zone: int, year: int) -> str:
    """给定zone与年份返回分类结果tif路径。"""  # 函数注释说明功能
    # 依据题意分类结果目录下各zone为子目录（如...\zone1），文件为zone{n}_{year}.tif
    zone_dir = os.path.join(CLASSIFICATION_ROOT, f"zone{zone}")  # 拼接zone目录路径
    tif_name = f"optimized_zone{zone}_{year}_添加颜色映射表.tif"  # 生成tif文件名
    # 返回完整路径
    return os.path.join(zone_dir, tif_name)  # 拼接并返回tif路径


# ============================ 指标计算工具 ============================
def compute_metrics_from_cm(cm: np.ndarray) -> pd.DataFrame:
    """从混淆矩阵计算每类用户精度、生产者精度及总体精度。"""  # 函数注释说明功能
    # 将矩阵转换为numpy数组并确保是二维
    cm = np.asarray(cm)  # 转换为numpy数组
    # 计算每行（真实类别）的总数，用于生产者精度
    row_sums = cm.sum(axis=1)  # 计算行和（真实类别样本数）
    # 计算每列（预测类别）的总数，用于用户精度
    col_sums = cm.sum(axis=0)  # 计算列和（预测为该类别的样本数）
    # 计算总体精度（对角线之和除以总样本数）
    overall = np.trace(cm) / cm.sum() if cm.sum() > 0 else 0.0  # 计算总体精度
    # 为每个类别计算用户精度（Precision）与生产者精度（Recall）
    records = []  # 初始化结果记录列表
    for i, cls in enumerate(CLASS_LABELS):  # 遍历类别索引与标签
        # 计算该类别的生产者精度（TP/真实为该类的总数）
        producer = (cm[i, i] / row_sums[i]) if row_sums[i] > 0 else 0.0  # 生产者精度计算
        # 计算该类别的用户精度（TP/预测为该类的总数）
        user = (cm[i, i] / col_sums[i]) if col_sums[i] > 0 else 0.0  # 用户精度计算
        # 支持度（真实为该类的样本数）
        support = int(row_sums[i])  # 将行和转为整数作为支持度
        # 追加记录到列表
        records.append({"class": cls, "user_accuracy": user, "producer_accuracy": producer, "support": support, "overall_accuracy": overall})  # 追加该类别的指标
    # 转为DataFrame并返回
    return pd.DataFrame(records)  # 返回指标数据框


# ============================ 采样与精度计算核心 ============================
def sample_points_and_evaluate(zone: int, year: int) -> dict:
    """对指定zone与year进行采样与精度计算，返回简要统计字典。"""  # 函数注释说明功能
    # 记录起始时间
    t0 = time.time()  # 记录开始时间
    # 构造shp与tif路径
    shp_path = zone_shp_path(zone)  # 获取当前zone的shp路径
    tif_path = zone_year_tif_path(zone, year)  # 获取当前zone与年份的tif路径
    # 准备输出文件路径（采样结果、矩阵与指标）
    sample_out = os.path.join(SAMPLES_DIR, f"zone{zone}_{year}_samples.csv")  # 采样结果输出文件路径
    matrix_out = os.path.join(MATRICES_DIR, f"zone{zone}_{year}_confusion_matrix.csv")  # 混淆矩阵输出文件路径
    metrics_out = os.path.join(METRICS_DIR, f"zone{zone}_{year}_metrics.csv")  # 指标输出文件路径

    try:
        # 检查文件是否存在，不存在则记录警告并跳过
        if not os.path.exists(shp_path):  # 判断shp文件是否存在
            logging.warning(f"缺少验证点文件：{shp_path}")  # 记录警告日志
            return {"zone": zone, "year": year, "samples": 0, "skipped": True}  # 返回跳过信息
        if not os.path.exists(tif_path):  # 判断tif文件是否存在
            logging.warning(f"缺少分类结果文件：{tif_path}")  # 记录警告日志
            return {"zone": zone, "year": year, "samples": 0, "skipped": True}  # 返回跳过信息

        # 读取验证点为GeoDataFrame
        gdf = gpd.read_file(shp_path)  # 读取shapefile为GeoDataFrame
        # 若没有geometry或必要字段，直接跳过
        if gdf.empty or "geometry" not in gdf.columns:  # 判断数据是否为空且包含geometry
            logging.warning(f"验证点为空或缺少geometry：{shp_path}")  # 记录警告日志
            return {"zone": zone, "year": year, "samples": 0, "skipped": True}  # 返回跳过信息
        if "landcover" not in gdf.columns or "year" not in gdf.columns:  # 判断是否存在必要属性字段
            logging.warning(f"验证点缺少字段landcover或year：{shp_path}")  # 记录警告日志
            return {"zone": zone, "year": year, "samples": 0, "skipped": True}  # 返回跳过信息

        # 过滤指定年份（year为double，需转换为int比较）
        gdf["year_int"] = gdf["year"].astype(int)  # 将year转换为整数新列year_int
        gdf_year = gdf[gdf["year_int"] == int(year)].copy()  # 按目标年份过滤并复制子集
        # 若该年份在该zone没有样本，记录并返回
        if gdf_year.empty:  # 判断过滤后的数据是否为空
            logging.info(f"zone{zone}年{year}无测试样本，跳过")  # 记录信息日志
            return {"zone": zone, "year": year, "samples": 0, "skipped": True}  # 返回跳过信息

        # 打开栅格进行采样
        with rasterio.open(tif_path) as src:  # 使用with语句打开tif确保及时关闭
            gdf_year = gdf_year.to_crs(src.crs)  # 将GeoDataFrame重投影到栅格CRS
            coords = [(geom.x, geom.y) for geom in gdf_year.geometry]  # 生成按点的坐标列表
            nodata = src.nodata  # 读取栅格nodata设定
            preds_list = []  # 逐点安全读取的预测值列表
            for x, y in coords:  # 逐点遍历坐标
                try:
                    row, col = src.index(x, y)  # 将坐标转换为行列索引
                    if 0 <= row < src.height and 0 <= col < src.width:  # 边界检查
                        arr = src.read(1, window=Window(col, row, 1, 1), masked=False)  # 读取1×1像素
                        val = int(arr[0, 0])  # 取像素值并转为整数
                        if nodata is not None and val == nodata:  # nodata处理
                            preds_list.append(-1)
                        else:
                            preds_list.append(val)
                    else:
                        preds_list.append(-1)  # 边界外标记为无效
                except Exception:
                    preds_list.append(-1)  # 读取失败标记为无效
            preds = np.array(preds_list)  # 转为数组

        # 释放文件句柄后手动触发垃圾回收
        gc.collect()  # 触发垃圾回收，减少内存占用

        # 构造结果DataFrame（包含真实与预测）
        df = pd.DataFrame({  # 创建结果数据框
            "zone": zone,  # 记录zone编号列
            "year": year,  # 记录年份列
            "x": [geom.x for geom in gdf_year.geometry],  # 提取x坐标列
            "y": [geom.y for geom in gdf_year.geometry],  # 提取y坐标列
            "landcover": gdf_year["landcover"].astype(int).values,  # 真实类别列（转为int）
            "pred": preds,  # 预测类别列
        })

        # 过滤无效预测（-1）与越界类别（不在1,2,3）
        df_valid = df[df["pred"].isin(CLASS_LABELS)]  # 仅保留预测在类别标签中的记录
        # 若全部为无效预测，记录并返回
        if df_valid.empty:  # 判断有效数据是否为空
            logging.warning(f"zone{zone}年{year}采样无有效预测（可能全部nodata），跳过")  # 记录警告日志
            # 仍保存原始采样结果以供检查
            os.makedirs(SAMPLES_DIR, exist_ok=True)  # 确保采样输出目录存在
            df.to_csv(sample_out, index=False, encoding="utf-8-sig")  # 保存采样结果CSV
            return {"zone": zone, "year": year, "samples": int(len(df)), "valid": 0, "skipped": True}  # 返回跳过信息

        # 保存有效采样结果CSV
        os.makedirs(SAMPLES_DIR, exist_ok=True)  # 确保采样输出目录存在
        df_valid.to_csv(sample_out, index=False, encoding="utf-8-sig")  # 保存有效采样结果到CSV

        # 计算混淆矩阵（按固定标签顺序）
        cm = confusion_matrix(df_valid["landcover"].values, df_valid["pred"].values, labels=CLASS_LABELS)  # 计算混淆矩阵
        # 保存混淆矩阵为CSV（行是真实类别，列为预测类别）
        os.makedirs(MATRICES_DIR, exist_ok=True)  # 确保矩阵输出目录存在
        cm_df = pd.DataFrame(cm, index=[f"actual_{c}" for c in CLASS_LABELS], columns=[f"pred_{c}" for c in CLASS_LABELS])  # 构造矩阵DataFrame
        cm_df.to_csv(matrix_out, encoding="utf-8-sig")  # 输出混淆矩阵到CSV

        # 计算并保存精度指标
        metrics_df = compute_metrics_from_cm(cm)  # 计算每类指标与总体精度
        os.makedirs(METRICS_DIR, exist_ok=True)  # 确保指标输出目录存在
        metrics_df.to_csv(metrics_out, index=False, encoding="utf-8-sig")  # 保存指标CSV

        # 记录耗时与样本数，返回简要统计
        dt = time.time() - t0  # 计算处理耗时
        logging.info(f"完成 zone{zone} 年{year}：样本{len(df_valid)}，耗时{dt:.2f}s")  # 记录完成日志
        return {"zone": zone, "year": year, "samples": int(len(df_valid)), "skipped": False}  # 返回成功统计

    except Exception as e:  # 捕获所有异常
        # 记录异常详细信息
        logging.error(f"处理 zone{zone} 年{year} 发生异常：{e}")  # 输出错误日志
        return {"zone": zone, "year": year, "samples": 0, "error": str(e), "skipped": True}  # 返回异常统计


# ============================ 年度汇总计算 ============================
def aggregate_yearly_summary(year: int) -> dict:
    """读取该年的所有zone采样结果进行年度汇总，输出年度矩阵与指标CSV。"""  # 函数注释说明功能
    # 记录起始时间
    t0 = time.time()  # 记录开始时间
    # 为该年份收集所有zone的有效采样文件路径
    sample_files = [os.path.join(SAMPLES_DIR, f"zone{z}_{year}_samples.csv") for z in ZONES]  # 构造该年所有zone的采样文件列表
    # 读取并合并有效数据
    dfs = []  # 初始化数据框列表
    for fp in sample_files:  # 遍历采样文件路径
        if os.path.exists(fp):  # 判断采样文件是否存在
            try:
                df = pd.read_csv(fp)  # 读取采样CSV
                df_valid = df[df["pred"].isin(CLASS_LABELS)].copy()  # 过滤有效预测
                if not df_valid.empty:  # 若不为空则加入列表
                    dfs.append(df_valid)  # 追加到列表
            except Exception as e:  # 捕获读取异常
                logging.warning(f"读取采样文件失败 {fp}：{e}")  # 记录警告日志
        else:  # 若采样文件不存在
            logging.debug(f"缺少采样文件：{fp}")  # 记录调试信息

    # 若没有任何有效数据则跳过汇总
    if len(dfs) == 0:  # 判断是否有有效采样数据
        logging.info(f"年份{year}无有效采样数据用于汇总，跳过")  # 记录信息日志
        return {"year": year, "samples": 0, "skipped": True}  # 返回跳过信息

    # 合并所有zone的有效数据
    all_df = pd.concat(dfs, ignore_index=True)  # 合并为一个数据框
    # 计算年度混淆矩阵
    cm = confusion_matrix(all_df["landcover"].values, all_df["pred"].values, labels=CLASS_LABELS)  # 计算年度混淆矩阵
    # 保存年度混淆矩阵CSV
    os.makedirs(SUMMARY_DIR, exist_ok=True)  # 确保汇总输出目录存在
    cm_out = os.path.join(SUMMARY_DIR, f"year_{year}_confusion_matrix.csv")  # 年度矩阵输出文件路径
    cm_df = pd.DataFrame(cm, index=[f"actual_{c}" for c in CLASS_LABELS], columns=[f"pred_{c}" for c in CLASS_LABELS])  # 构造矩阵DataFrame
    cm_df.to_csv(cm_out, encoding="utf-8-sig")  # 输出年度矩阵到CSV

    # 计算并保存年度精度指标
    metrics_df = compute_metrics_from_cm(cm)  # 计算年度指标
    metrics_out = os.path.join(SUMMARY_DIR, f"year_{year}_metrics.csv")  # 年度指标输出文件路径
    metrics_df.to_csv(metrics_out, index=False, encoding="utf-8-sig")  # 输出年度指标到CSV

    # 记录耗时并返回统计
    dt = time.time() - t0  # 计算耗时
    logging.info(f"完成 年度汇总 {year}：样本{len(all_df)}，耗时{dt:.2f}s")  # 记录完成日志
    return {"year": year, "samples": int(len(all_df)), "skipped": False}  # 返回成功统计


# ============================ 主流程 ============================
def main():
    """主入口：按zone与year执行采样与评估，并进行年度汇总。"""  # 函数注释说明功能
    # 初始化日志并输出日志文件位置
    log_path = setup_logging()  # 调用日志初始化，获取日志文件路径
    logging.info(f"日志文件：{log_path}")  # 打印日志文件位置到控制台与日志

    # 创建输出目录树
    for d in [OUTPUT_DIR, SAMPLES_DIR, MATRICES_DIR, METRICS_DIR, SUMMARY_DIR]:  # 遍历需要的输出目录
        os.makedirs(d, exist_ok=True)  # 创建目录（若不存在）

    # 记录总体开始时间
    all_t0 = time.time()  # 记录总开始时间

    # 逐zone逐年执行采样与评估（支持并行或顺序）
    summary_rows = []  # 初始化简要统计列表用于最终总览CSV
    total_tasks = len(ZONES) * len(YEARS)  # 计算总任务数量用于进度条
    if USE_MULTIPROCESS:  # 判断是否使用多进程
        logging.info(f"并行处理启用，进程数：{MAX_WORKERS}")  # 记录并行配置到日志
        tasks = []  # 初始化任务列表
        bar = tqdm(total=total_tasks, desc="采样评估", unit="任务") if tqdm else None  # 创建进度条（若可用）
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:  # 创建进程池执行器
            for zone in ZONES:  # 遍历所有zone编号
                for year in YEARS:  # 遍历所有年份
                    future = executor.submit(sample_points_and_evaluate, zone, year)  # 提交并行任务
                    tasks.append(future)  # 记录future对象
            for fut in as_completed(tasks):  # 收集任务结果（按完成顺序）
                try:
                    stat = fut.result()  # 获取任务返回结果
                    summary_rows.append(stat)  # 追加结果到列表
                except Exception as e:  # 捕获任务执行异常
                    logging.error(f"并行任务异常：{e}")  # 记录错误日志
                finally:
                    if bar:  # 若进度条存在
                        bar.update(1)  # 更新进度条完成一个任务
        if bar:  # 处理结束后
            bar.close()  # 关闭进度条
    else:  # 若不使用并行
        bar = tqdm(total=total_tasks, desc="采样评估", unit="任务") if tqdm else None  # 创建顺序进度条（若可用）
        for zone in ZONES:  # 遍历所有zone编号
            for year in YEARS:  # 遍历所有年份
                stat = sample_points_and_evaluate(zone, year)  # 调用核心函数进行采样与评估
                summary_rows.append(stat)  # 追加简要统计结果
                if bar:  # 若进度条存在
                    bar.update(1)  # 更新进度条
        if bar:  # 顺序处理完成后
            bar.close()  # 关闭进度条

    # 输出每个zone-年处理的简要总览
    overview_out = os.path.join(OUTPUT_DIR, "zone_year_overview.csv")  # 定义总览输出CSV路径
    pd.DataFrame(summary_rows).to_csv(overview_out, index=False, encoding="utf-8-sig")  # 保存总览CSV
    logging.info(f"已保存处理总览：{overview_out}")  # 记录信息日志

    # 年度汇总计算（按年读取前面保存的采样结果合并）
    year_overview = []  # 初始化年度汇总统计列表
    bar_year = tqdm(total=len(YEARS), desc="年度汇总", unit="年") if tqdm else None  # 创建年度汇总进度条（若可用）
    for year in YEARS:  # 遍历所有年份
        stat = aggregate_yearly_summary(year)  # 调用年度汇总函数
        year_overview.append(stat)  # 追加年度汇总统计
        if bar_year:  # 若年度进度条存在
            bar_year.update(1)  # 更新年度进度条
    if bar_year:  # 年度汇总完成后
        bar_year.close()  # 关闭年度进度条

    # 输出年度汇总总览
    year_overview_out = os.path.join(SUMMARY_DIR, "year_overview.csv")  # 定义年度总览输出路径
    pd.DataFrame(year_overview).to_csv(year_overview_out, index=False, encoding="utf-8-sig")  # 保存年度总览CSV
    logging.info(f"已保存年度汇总总览：{year_overview_out}")  # 记录信息日志

    # 总体耗时记录
    all_dt = time.time() - all_t0  # 计算总体耗时
    logging.info(f"全部完成，总耗时 {all_dt:.2f}s")  # 记录完成日志


# ============================ 脚本执行入口 ============================
if __name__ == "__main__":  # 判断是否作为主脚本运行
    main()  # 调用主函数执行流程
