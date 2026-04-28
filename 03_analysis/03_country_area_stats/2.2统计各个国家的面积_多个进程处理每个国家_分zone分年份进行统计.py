# 目的：读取每个国家在每一年的人工林和自然林的像素个数，请注意内存溢出以及投影问题，还有尽量快一点处理
max_workers = 8
RASTER_YEAR = 2017
# 统计每个国家从2017年到2024年的人工林（值为1）、自然林（值为2）像素的个数
country_SHP=r"F:\地理所\论文\东南亚10m人工林提取\数据\正式分类_10.29\6.GEE导出结果_结果合并_马尔可夫模型_逐年合并_裁剪东南亚_转为等面积投影并裁剪每个国家\矢量\southeast_asia_ADM_0_aea.shp"
# 国家的名字的唯一属性为NAME_0
attri_shp="NAME_0"
# 栅格的位置
TIFS_PATH=r"D:\地理所\论文\东南亚10m人工林提取\数据\正式分类_10.29\4.GEE导出结果_结果合并_马尔可夫模型_前后向推导_转为等面积投影"
TEST_TIF_PATH=r"D:\地理所\论文\东南亚10m人工林提取\数据\正式分类_10.29\4.GEE导出结果_结果合并_马尔可夫模型_前后向推导_转为等面积投影2\zone1\optimized_zone1_2017_等面积投影.tif"
OUTPUT_DIR = r"D:\地理所\论文\东南亚10m人工林提取\数据\正式分类_10.29\4.GEE导出结果_结果合并_马尔可夫模型_前后向推导_转为等面积投影2\面积统计结果"
# 统计输出位置：F:\地理所\论文\东南亚10m人工林提取\数据\正式分类_10.29\6.GEE导出结果_结果合并_马尔可夫模型_逐年合并_裁剪东南亚_转为等面积投影并裁剪每个国家\面积统计结果
import os
import time
import csv
from datetime import datetime
import re
# 解释：导入GIS与并行处理所需库
import fiona  # 读取矢量
from shapely.geometry import shape as shp_shape  # 将GeoJSON几何转为Shapely对象
import rasterio  # 读取栅格
from rasterio.features import geometry_mask  # 几何掩膜生成
import numpy as np  # 数值计算
import multiprocessing
# 解释：日志设置（输出到结果文件夹）
LOG_PATH = os.path.join(OUTPUT_DIR, "processing.log")
def _ensure_dirs():
    # 解释：创建输出目录与日志文件父目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
def _log(msg: str):
    # 解释：统一日志打印到控制台与文件
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass
def _mem_usage(prefix: str = ""):
    return
def load_countries(shp_path: str, attr_name: str):
    # 解释：读取国家矢量，返回GeoJSON几何与属性名称列表
    countries = []
    with fiona.open(shp_path, 'r') as src:
        for feat in src:
            props = feat.get('properties', {})
            name = props.get(attr_name)
            if not name:
                continue
            geom = feat.get('geometry')
            if not geom:
                continue
            countries.append({'name': str(name), 'geometry': geom})
    if not countries:
        raise RuntimeError("矢量数据为空或未找到有效要素")
    return countries
def _country_intersects_raster(country_geojson: dict, raster_bounds: tuple):
    geom = shp_shape(country_geojson)
    minx, miny, maxx, maxy = raster_bounds
    raster_poly = shp_shape({"type":"Polygon","coordinates":[[
        [minx,miny],[minx,maxy],[maxx,maxy],[maxx,miny],[minx,miny]
    ]]})
    return geom.intersects(raster_poly)
RASTER_ARR = None
RASTER_TRANSFORM = None
RASTER_NODATA = None

RASTER_ZONE = "zone"
def _init_full_array_worker(arr, transform, nodata, year, zone):
    global RASTER_ARR, RASTER_TRANSFORM, RASTER_NODATA, RASTER_YEAR, RASTER_ZONE
    RASTER_ARR = arr
    RASTER_TRANSFORM = transform
    RASTER_NODATA = nodata
    RASTER_YEAR = year
    RASTER_ZONE = zone
def _country_worker_full(country: dict):
    name = country['name']
    geojson = country['geometry']
    mask_inside = geometry_mask([geojson], out_shape=RASTER_ARR.shape, transform=RASTER_TRANSFORM, invert=True)
    valid_mask = (RASTER_ARR != RASTER_NODATA)
    inside = mask_inside & valid_mask
    if not inside.any():
        return None
    sub = RASTER_ARR[inside]
    pf = np.int64(np.count_nonzero(sub == 1))
    nf = np.int64(np.count_nonzero(sub == 2))
    ot = np.int64(np.count_nonzero(sub == 3))
    total = np.int64(pf + nf + ot)
    return {
        'country': name,
        'year': RASTER_YEAR,
        'zone': RASTER_ZONE,
        'pf': pf,
        'nf': nf,
        'others': ot,
        'total': total,
        'process_time': datetime.now().isoformat(timespec='seconds')
    }
def write_csv_zone(rows: list, out_path: str):
    cols = ['国家名称','zone','年份','PF像素数量','NF像素数量','OTHERS像素数量','总像素数量']
    rows_sorted = sorted(rows, key=lambda r: (r['country'], r['year']))
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows_sorted:
            w.writerow([r['country'], r['zone'], r['year'], int(r['pf']), int(r['nf']), int(r['others']), int(r['total'])])
def main():
    _ensure_dirs()
    _log("单文件本地测试：按国家边界统计PF/NF/OTHERS像素数量")
    if not os.path.exists(TEST_TIF_PATH):
        _log(f"缺失测试栅格：{TEST_TIF_PATH}")
        return
    fname = os.path.basename(TEST_TIF_PATH)
    zm = re.search(r"(zone\d+)", fname, re.IGNORECASE)
    ym = re.search(r"(20\d{2})", fname)
    zone = (zm.group(1).lower() if zm else "zone_unknown")
    year = (int(ym.group(1)) if ym else 2017)
    _log(f"解析文件名：zone={zone}，year={year}")
    countries = load_countries(country_SHP, attri_shp)
    _log(f"总国家数：{len(countries)}")
    with rasterio.open(TEST_TIF_PATH, 'r', sharing=False) as ds:
        raster_crs = ds.crs
        try:
            with fiona.open(country_SHP, 'r') as shp:
                shp_crs = shp.crs
        except Exception:
            shp_crs = None
        if raster_crs and shp_crs and (raster_crs != shp_crs):
            _log("投影不一致警告：栅格CRS与矢量CRS不一致")
        arr = ds.read(1)
        if arr.dtype != np.uint8:
            _log(f"类型警告：读取到{arr.dtype}，将转换为uint8")
            arr = arr.astype(np.uint8)
        nodata = ds.nodata if ds.nodata is not None else 0
        bounds = ds.bounds
        filtered = [c for c in countries if _country_intersects_raster(c['geometry'], (bounds.left, bounds.bottom, bounds.right, bounds.top))]
        _log(f"与栅格范围相交国家数：{len(filtered)}，不相交：{len(countries)-len(filtered)}")
        _log(f"并行进程数：{max_workers}")
        results = []
        with multiprocessing.Pool(processes=max_workers, initializer=_init_full_array_worker, initargs=(arr, ds.transform, nodata, year, zone), maxtasksperchild=50) as pool:
            for row in pool.imap_unordered(_country_worker_full, filtered):
                if row is not None:
                    results.append(row)
        csv_path = os.path.join(OUTPUT_DIR, f"{zone}_{year}.csv")
        write_csv_zone(results, csv_path)
        _log(f"已输出CSV：{csv_path}，记录数：{len(results)}")
if __name__ == "__main__":
    main()