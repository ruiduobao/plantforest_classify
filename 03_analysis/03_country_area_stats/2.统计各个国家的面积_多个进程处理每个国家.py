
# 目的：读取每个国家在每一年的人工林和自然林的像素个数，请注意内存溢出以及投影问题，还有尽量快一点处理
# 统计每个国家从2017年到2024年的人工林（值为1）、自然林（值为2）像素的个数
country_SHP=r"F:\地理所\论文\东南亚10m人工林提取\数据\正式分类_10.29\6.GEE导出结果_结果合并_马尔可夫模型_逐年合并_裁剪东南亚_转为等面积投影并裁剪每个国家\矢量\southeast_asia_ADM_0_aea.shp"
# 国家的名字的唯一属性为NAME_0
attri_shp="NAME_0"
# 栅格的位置
TIFS_PATH=r"F:\地理所\论文\东南亚10m人工林提取\数据\正式分类_10.29\6.GEE导出结果_结果合并_马尔可夫模型_逐年合并_裁剪东南亚_转为等面积投影并裁剪每个国家\东南亚等面积投影"
# 栅格的名字为：人工林提取结果_每个区域16000个样本点_2017年_变化概率_筛选样本_逐年分类.tif，到人工林提取结果_每个区域16000个样本点_2024年_变化概率_筛选样本_逐年分类.tif

# 统计输出位置：F:\地理所\论文\东南亚10m人工林提取\数据\正式分类_10.29\6.GEE导出结果_结果合并_马尔可夫模型_逐年合并_裁剪东南亚_转为等面积投影并裁剪每个国家\面积统计结果
import os
import sys
import time
import csv
import random
import traceback
from datetime import datetime

# 解释：导入GIS与并行处理所需库

import fiona  # 读取矢量
from shapely.geometry import shape as shp_shape  # 将GeoJSON几何转为Shapely对象
import rasterio  # 读取栅格
from rasterio.windows import Window  # 定义分块窗口
from rasterio.features import geometry_mask  # 几何掩膜生成
import numpy as np  # 数值计算


# 解释：内存监控（可选）
psutil = None

# 解释：并行处理
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# 解释：日志设置（输出到结果文件夹）
OUTPUT_DIR = r"F:\地理所\论文\东南亚10m人工林提取\数据\正式分类_10.29\6.GEE导出结果_结果合并_马尔可夫模型_逐年合并_裁剪东南亚_转为等面积投影并裁剪每个国家\面积统计结果"
LOG_PATH = os.path.join(OUTPUT_DIR, "processing.log")
ENABLE_VALIDATION = True
ENABLE_FAST_MULTI_YEAR = False
ENABLE_PARALLEL_BY_COUNTRY = False

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

def _open_year_datasets(tifs_base: str, years: list):
    # 解释：打开所有年份影像并检查是否同一网格
    ds_list = []
    transforms = []
    sizes = []
    nodatas = []
    for y in years:
        tif_name = f"人工林提取结果_每个区域16000个样本点_{y}年_变化概率_筛选样本_逐年分类.tif"
        p = os.path.join(tifs_base, tif_name)
        if not os.path.exists(p):
            _log(f"缺失文件：{p}")
            ds_list.append(None)
            transforms.append(None)
            sizes.append(None)
            nodatas.append(0)
            continue
        d = rasterio.open(p, 'r', sharing=False)
        ds_list.append(d)
        transforms.append(d.transform)
        sizes.append((d.width, d.height))
        nodatas.append(d.nodata if d.nodata is not None else 0)
    base_t = next((t for t in transforms if t is not None), None)
    base_s = next((s for s in sizes if s is not None), None)
    same_grid = all((t is None or t == base_t) and (s is None or s == base_s) for t, s in zip(transforms, sizes))
    if not same_grid:
        _log("警告：多年份栅格网格不一致，无法启用fast-multi-year")
    return ds_list, nodatas, same_grid

def count_pixels_for_country_multi_year(ds_list: list, nodatas: list, years: list, country_geojson: dict):
    # 解释：在相同网格前提下，对单个国家一次遍历分块，统计所有年份的像素数量
    first = next((d for d in ds_list if d is not None), None)
    if first is None:
        return []
    geom_shp = shp_shape(country_geojson)
    minx, miny, maxx, maxy = geom_shp.bounds
    bounds_window = rasterio.windows.from_bounds(minx, miny, maxx, maxy, first.transform)
    if first.block_shapes and len(first.block_shapes) > 0:
        tile_h, tile_w = first.block_shapes[0]
    else:
        tile_h, tile_w = 1024, 1024
    stats = {y: {'plantation': 0, 'natural': 0, 'others': 0, 'invalid': 0} for y in years}
    for tile in _tile_windows(bounds_window, tile_h, tile_w):
        tile_transform = rasterio.windows.transform(tile, first.transform)
        geom_mask = geometry_mask([country_geojson], out_shape=(int(tile.height), int(tile.width)), transform=tile_transform, invert=True)
        for y, ds in zip(years, ds_list):
            if ds is None:
                continue
            arr = ds.read(1, window=tile)
            valid_mask = (arr != nodatas[years.index(y)])
            inside = geom_mask & valid_mask
            if not inside.any():
                continue
            sub = arr[inside]
            stats[y]['plantation'] += int(np.count_nonzero(sub == 1))
            stats[y]['natural'] += int(np.count_nonzero(sub == 2))
            stats[y]['others'] += int(np.count_nonzero(sub == 3))
            inv_mask = (sub != 1) & (sub != 2) & (sub != 3)
            stats[y]['invalid'] += int(np.count_nonzero(inv_mask))
    return [
        {
            'year': y,
            'plantation': stats[y]['plantation'],
            'natural': stats[y]['natural'],
            'others': stats[y]['others'],
            'invalid': stats[y]['invalid']
        } for y in years
    ]

def _tile_windows(bounds_window: Window, tile_h: int, tile_w: int):
    # 解释：将大窗口切分为若干小窗口以分块读取
    row_start = int(bounds_window.row_off)
    col_start = int(bounds_window.col_off)
    rows = int(bounds_window.height)
    cols = int(bounds_window.width)
    for r in range(row_start, row_start + rows, tile_h):
        h = min(tile_h, row_start + rows - r)
        for c in range(col_start, col_start + cols, tile_w):
            w = min(tile_w, col_start + cols - c)
            yield Window(c, r, w, h)

def count_pixels_for_country(ds_path: str, country_geojson: dict, country_name: str):
    # 解释：对单个国家在一个栅格年份文件中统计像素数量（值1为人工林，值2为自然林）
    with rasterio.open(ds_path, 'r', sharing=False) as ds:
        nodata = ds.nodata if ds.nodata is not None else 0
        if ds.count != 1:
            _log(f"警告：{os.path.basename(ds_path)}波段数量为{ds.count}，预期为1")
        # 解释：计算国家几何与栅格的边界窗口，避免读全图
        geom_shp = shp_shape(country_geojson)
        minx, miny, maxx, maxy = geom_shp.bounds
        bounds_window = rasterio.windows.from_bounds(minx, miny, maxx, maxy, ds.transform)
        # 解释：分块大小，采用栅格本身的block大小（如有），否则4096
        if ds.block_shapes and len(ds.block_shapes) > 0:
            tile_h, tile_w = ds.block_shapes[0]
        else:
            tile_h, tile_w = 1024, 1024
        plantation = 0
        natural = 0
        others = 0
        invalid = 0
        # 解释：遍历窗口内的每个小块
        for tile in _tile_windows(bounds_window, tile_h, tile_w):
            tile_transform = rasterio.windows.transform(tile, ds.transform)
            # 解释：生成几何掩膜（True=在几何外，False=在几何内；invert=True得到几何内为True）
            geom_mask = geometry_mask([country_geojson], out_shape=(int(tile.height), int(tile.width)), transform=tile_transform, invert=True)
            # 解释：读取当前块像素
            arr = ds.read(1, window=tile)
            valid_mask = (arr != nodata)
            # 解释：仅统计几何内且有效像素
            inside = geom_mask & valid_mask
            if not inside.any():
                continue
            sub = arr[inside]
            # 解释：统计1与2的数量
            plantation += int(np.count_nonzero(sub == 1))
            natural += int(np.count_nonzero(sub == 2))
            others += int(np.count_nonzero(sub == 3))
            # 解释：记录无效值（既不是1也不是2且非nodata）
            inv_mask = (sub != 1) & (sub != 2) & (sub != 3)
            invalid += int(np.count_nonzero(inv_mask))
        return {
            'country': country_name,
            'plantation': plantation,
            'natural': natural,
            'others': others,
            'invalid': invalid
        }

def compute_country_on_year(year: int, country: dict, tifs_base: str, data_version: str):
    # 解释：子任务——按国家并行统计指定年份像素
    name = country['name']
    geojson = country['geometry']
    tif_name = f"人工林提取结果_每个区域16000个样本点_{year}年_变化概率_筛选样本_逐年分类.tif"
    ds_path = os.path.join(tifs_base, tif_name)
    if not os.path.exists(ds_path):
        return None
    try:
        res = count_pixels_for_country(ds_path, geojson, name)
        row = {
            'country': name,
            'year': year,
            'plantation': res['plantation'],
            'natural': res['natural'],
            'others': res.get('others', 0),
            'invalid': res['invalid'],
            'data_version': data_version,
            'process_time': datetime.now().isoformat(timespec='seconds')
        }
        return row
    except Exception:
        return None

def year_total_check(ds_path: str):
    # 解释：遍历整幅影像分块统计总的1与2的数量，用于校验
    with rasterio.open(ds_path, 'r', sharing=False) as ds:
        nodata = ds.nodata if ds.nodata is not None else 0
        if ds.block_shapes and len(ds.block_shapes) > 0:
            tile_h, tile_w = ds.block_shapes[0]
        else:
            tile_h, tile_w = 1024, 1024
        total_1 = 0
        total_2 = 0
        total_3 = 0
        # 解释：全图窗口
        full = Window(0, 0, ds.width, ds.height)
        for tile in _tile_windows(full, tile_h, tile_w):
            arr = ds.read(1, window=tile)
            valid = (arr != nodata)
            if not valid.any():
                continue
            sub = arr[valid]
            total_1 += int(np.count_nonzero(sub == 1))
            total_2 += int(np.count_nonzero(sub == 2))
            total_3 += int(np.count_nonzero(sub == 3))
        return total_1, total_2, total_3

def process_one_year(year: int, countries: list, attr_name: str, tifs_base: str, shp_crs, data_version: str, enable_validation: bool):
    # 解释：处理单一年份，返回每个国家结果与校验信息
    start = time.time()
    tif_name = f"人工林提取结果_每个区域16000个样本点_{year}年_变化概率_筛选样本_逐年分类.tif"
    ds_path = os.path.join(tifs_base, tif_name)
    if not os.path.exists(ds_path):
        _log(f"缺失文件：{ds_path}")
        return {
            'year': year,
            'rows': [],
            'missing': True,
            'summary': {}
        }
    # 解释：打开栅格并检查投影一致性
    with rasterio.open(ds_path, 'r', sharing=False) as ds:
        raster_crs = ds.crs
        if raster_crs and shp_crs and (raster_crs != shp_crs):
            _log(f"投影不一致警告：年份{year} 栅格CRS与矢量CRS不一致")
        _mem_usage(prefix=f"年份{year} 开始：")
    rows = []
    # 解释：准备年份内的进度信息（总国家数与进程ID），并按固定步长输出进度日志
    total_countries = len(countries)
    pid = os.getpid()
    step = max(1, total_countries // 2)  # 解释：每约10%输出一次进度
    _log(f"年份{year} 进度：0/{total_countries} (0.0%) PID={pid}")
    # 解释：逐国家统计
    for idx, c in enumerate(countries, start=1):
        name = c['name']
        geojson = c['geometry']
        try:
            res = count_pixels_for_country(ds_path, geojson, name, shp_crs)
            rows.append({
                'country': name,
                'year': year,
                'plantation': res['plantation'],
                'natural': res['natural'],
                'others': res.get('others', 0),
                'invalid': res['invalid'],
                'data_version': data_version,
                'process_time': datetime.now().isoformat(timespec='seconds')
            })
            # 解释：分阶段输出进度（避免日志过多），包含百分比与PID
            if (idx % step == 0) or (idx == total_countries):
                pct = (idx / total_countries) * 100.0
                _log(f"年份{year} 进度：{idx}/{total_countries} ({pct:.1f}%) PID={pid}")
        except Exception as e:
            _log(f"国家{name} 年份{year}统计异常：{e}")
            _log(traceback.format_exc())
    # 解释：校验与抽样（可选）
    if enable_validation:
        total_1, total_2, total_3 = year_total_check(ds_path)
        sum_1 = sum(r['plantation'] for r in rows)
        sum_2 = sum(r['natural'] for r in rows)
        sum_3 = sum(r.get('others', 0) for r in rows)
        diff_1 = total_1 - sum_1
        diff_2 = total_2 - sum_2
        diff_3 = total_3 - sum_3
        _log(f"年份{year} 校验：影像总(1)={total_1}，国家汇总(1)={sum_1}，差值={diff_1}")
        _log(f"年份{year} 校验：影像总(2)={total_2}，国家汇总(2)={sum_2}，差值={diff_2}")
        _log(f"年份{year} 校验：影像总(3)={total_3}，国家汇总(3)={sum_3}，差值={diff_3}")
        sample_notes = []
        sample_countries = random.sample(rows, k=min(3, len(rows))) if rows else []
        for sr in sample_countries:
            try:
                recheck = count_pixels_for_country(ds_path, next(c['geometry'] for c in countries if c['name']==sr['country']), sr['country'], shp_crs)
                ok = (recheck['plantation']==sr['plantation']) and (recheck['natural']==sr['natural']) and (recheck.get('others',0)==sr.get('others',0))
                note = f"{sr['country']} 重算一致={ok}"
                sample_notes.append(note)
            except Exception as e:
                sample_notes.append(f"{sr['country']} 重算异常：{e}")
    took = time.time() - start
    _mem_usage(prefix=f"年份{year} 结束：")
    if enable_validation:
        return {
            'year': year,
            'rows': rows,
            'missing': False,
            'summary': {
                    'validation': True,
                    'total_1': total_1,
                    'total_2': total_2,
                    'total_3': total_3,
                    'sum_1': sum_1,
                    'sum_2': sum_2,
                    'sum_3': sum_3,
                    'diff_1': diff_1,
                    'diff_2': diff_2,
                    'diff_3': diff_3,
                    'samples': sample_notes,
                    'seconds': took
            }
        }

def process_all_years_fast(countries: list, tifs_base: str, data_version: str, years: list):
    # 解释：快速模式——同网格条件下，一次遍历国家分块，统计所有年份
    start = time.time()
    ds_list, nodatas, same_grid = _open_year_datasets(tifs_base, years)
    if not same_grid:
        _log("fast-multi-year不可用：降级为普通模式")
        # 解释：关闭快速路径，交给普通模式处理
        for d in ds_list:
            try:
                if d: d.close()
            except Exception:
                pass
        return None
    rows = []
    total_countries = len(countries)
    pid = os.getpid()
    step = max(1, total_countries // 10)
    _log(f"快速模式：同网格年份={len([d for d in ds_list if d is not None])} PID={pid}")
    _log(f"进度：0/{total_countries} (0.0%)")
    for idx, c in enumerate(countries, start=1):
        name = c['name']
        geojson = c['geometry']
        try:
            stats_years = count_pixels_for_country_multi_year(ds_list, nodatas, years, geojson)
            for st in stats_years:
                rows.append({
                    'country': name,
                    'year': st['year'],
                    'plantation': st['plantation'],
                    'natural': st['natural'],
                    'others': st['others'],
                    'invalid': st['invalid'],
                    'data_version': data_version,
                    'process_time': datetime.now().isoformat(timespec='seconds')
                })
        except Exception as e:
            _log(f"国家{name} 快速统计异常：{e}")
            _log(traceback.format_exc())
        if (idx % step == 0) or (idx == total_countries):
            pct = (idx / total_countries) * 100.0
            _log(f"进度：{idx}/{total_countries} ({pct:.1f}%)")
    # 解释：关闭数据集
    for d in ds_list:
        try:
            if d: d.close()
        except Exception:
            pass
    took = time.time() - start
    _log(f"快速模式完成，用时{took:.2f}s")
    summary = {'seconds': took}
    return rows, summary

def write_csv(rows: list, out_path: str):
    # 解释：写出CSV（按国家与年份排序）
    cols = ['国家名称','年份','人工林像素数','自然林像素数','其他像素数','无效像素数','处理时间','数据版本']
    # 解释：排序
    rows_sorted = sorted(rows, key=lambda r: (r['country'], r['year']))
    # 解释：写文件
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows_sorted:
            w.writerow([
                r['country'], r['year'], r['plantation'], r['natural'], r.get('others',0), r.get('invalid',0), r['process_time'], r['data_version']
            ])

def _year_paths(tifs_base: str, years: list):
    paths = []
    for y in years:
        tif_name = f"人工林提取结果_每个区域16000个样本点_{y}年_变化概率_筛选样本_逐年分类.tif"
        paths.append(os.path.join(tifs_base, tif_name))
    return paths

def _generate_windows_for_country(first_ds, country_geojson):
    geom_shp = shp_shape(country_geojson)
    minx, miny, maxx, maxy = geom_shp.bounds
    bounds_window = rasterio.windows.from_bounds(minx, miny, maxx, maxy, first_ds.transform)
    if first_ds.block_shapes and len(first_ds.block_shapes) > 0:
        tile_h, tile_w = first_ds.block_shapes[0]
    else:
            tile_h, tile_w = 1024, 1024
    return list(_tile_windows(bounds_window, tile_h, tile_w))

def _tile_worker(args):
    year_paths, tile, first_transform, country_geojson = args
    try:
        tile_transform = rasterio.windows.transform(tile, first_transform)
        geom_mask = geometry_mask([country_geojson], out_shape=(int(tile.height), int(tile.width)), transform=tile_transform, invert=True)
        counts = {}
        for y, p in year_paths:
            if not os.path.exists(p):
                continue
            with rasterio.open(p, 'r', sharing=False) as ds:
                arr = ds.read(1, window=tile)
                nodata = ds.nodata if ds.nodata is not None else 0
                valid_mask = (arr != nodata)
                inside = geom_mask & valid_mask
                if not inside.any():
                    counts[y] = (0,0,0,0)
                    continue
                sub = arr[inside]
                pl = np.int64(np.count_nonzero(sub == 1))
                nl = np.int64(np.count_nonzero(sub == 2))
                ot = np.int64(np.count_nonzero(sub == 3))
                inv = np.int64(np.count_nonzero((sub != 1) & (sub != 2) & (sub != 3)))
                counts[y] = (pl, nl, ot, inv)
        return counts
    except Exception:
        return {}

WORKER_YEAR_PAIRS = None
WORKER_YEAR_DS = None
WORKER_NODATAS = None
WORKER_FIRST_TRANSFORM = None
WORKER_COUNTRY_GEOJSON = None

def _init_worker(year_pairs, first_transform, country_geojson):
    global WORKER_YEAR_PAIRS, WORKER_YEAR_DS, WORKER_NODATAS, WORKER_FIRST_TRANSFORM, WORKER_COUNTRY_GEOJSON
    WORKER_YEAR_PAIRS = year_pairs
    WORKER_FIRST_TRANSFORM = first_transform
    WORKER_COUNTRY_GEOJSON = country_geojson
    WORKER_YEAR_DS = []
    WORKER_NODATAS = []
    for y, p in year_pairs:
        if not os.path.exists(p):
            WORKER_YEAR_DS.append(None)
            WORKER_NODATAS.append(0)
        else:
            with rasterio.Env(GDAL_CACHEMAX=64, GDAL_NUM_THREADS='1'):
                ds = rasterio.open(p, 'r', sharing=False)
            WORKER_YEAR_DS.append(ds)
            WORKER_NODATAS.append(ds.nodata if ds.nodata is not None else 0)

def _tile_worker_idx(tile):
    counts = {}
    tile_transform = rasterio.windows.transform(tile, WORKER_FIRST_TRANSFORM)
    geom_mask = geometry_mask([WORKER_COUNTRY_GEOJSON], out_shape=(int(tile.height), int(tile.width)), transform=tile_transform, invert=True)
    for (y, _), ds, nod in zip(WORKER_YEAR_PAIRS, WORKER_YEAR_DS, WORKER_NODATAS):
        if ds is None:
            continue
        try:
            arr = ds.read(1, window=tile)
        except Exception:
            counts[y] = (np.int64(0), np.int64(0), np.int64(0), np.int64(0))
            continue
        valid_mask = (arr != nod)
        inside = geom_mask & valid_mask
        if not inside.any():
            counts[y] = (np.int64(0), np.int64(0), np.int64(0), np.int64(0))
            continue
        sub = arr[inside]
        pl = np.int64(np.count_nonzero(sub == 1))
        nl = np.int64(np.count_nonzero(sub == 2))
        ot = np.int64(np.count_nonzero(sub == 3))
        inv = np.int64(np.count_nonzero((sub != 1) & (sub != 2) & (sub != 3)))
        counts[y] = (pl, nl, ot, inv)
    return counts

def process_one_country(country: dict, tifs_base: str, years: list, data_version: str, max_workers: int):
    name = country['name']
    geojson = country['geometry']
    year_paths = _year_paths(tifs_base, years)
    pairs = list(zip(years, year_paths))
    first_existing = next((p for p in year_paths if os.path.exists(p)), None)
    if first_existing is None:
        return []
    with rasterio.open(first_existing, 'r', sharing=False) as first_ds:
        windows = _generate_windows_for_country(first_ds, geojson)
        args_list = [(pairs, w, first_ds.transform, geojson) for w in windows]
    totals = {y: np.zeros(4, dtype=np.int64) for y in years}
    with multiprocessing.Pool(processes=max_workers, initializer=_init_worker, initargs=(pairs, first_ds.transform, geojson), maxtasksperchild=50) as pool:
        total_windows = len(windows)
        completed = 0
        step = max(1, total_windows // 10)
        for cdict in pool.imap_unordered(_tile_worker_idx, windows):
            for y, vals in cdict.items():
                totals[y] += np.array(vals, dtype=np.int64)
            completed += 1
            if (completed % step == 0) or (completed == total_windows):
                pct = (completed / total_windows) * 100.0
                _log(f"国家{name} 进度：{completed}/{total_windows} ({pct:.1f}%)")
    rows = []
    now = datetime.now().isoformat(timespec='seconds')
    for y in years:
        rows.append({
            'country': name,
            'year': y,
            'plantation': totals[y][0],
            'natural': totals[y][1],
            'others': totals[y][2],
            'invalid': totals[y][3],
            'data_version': data_version,
            'process_time': now
        })
    return rows

def main():
    # 解释：按国家逐个处理，国家内多进程分块统计，完成即保存
    _ensure_dirs()
    data_version = "正式分类_10.29"
    _log("开始统计东南亚各国2017-2024年人工林/自然林像素数量")
    countries = load_countries(country_SHP, attri_shp)
    _log(f"国家数量：{len(countries)}")
    years = list(range(2017, 2025))
    csv_path = os.path.join(OUTPUT_DIR, "southeast_asia_forest_pixels_2017_2024.csv")
    if not os.path.exists(csv_path):
        write_csv([], csv_path)
    per_country_workers = min(12, max(1, (os.cpu_count() or 4) - 2))
    for idx, c in enumerate(countries, start=1):
        _log(f"开始国家：{c['name']} ({idx}/{len(countries)})")
        rows = process_one_country(c, TIFS_PATH, years, data_version, per_country_workers)
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            for r in sorted(rows, key=lambda x: x['year']):
                w.writerow([r['country'], r['year'], int(r['plantation']), int(r['natural']), int(r.get('others',0)), int(r.get('invalid',0)), r['process_time'], r['data_version']])
        _log(f"国家{c['name']} 已保存 {len(rows)} 条记录")
    _log(f"统计完成，CSV输出：{csv_path}")

if __name__ == "__main__":
    main()
