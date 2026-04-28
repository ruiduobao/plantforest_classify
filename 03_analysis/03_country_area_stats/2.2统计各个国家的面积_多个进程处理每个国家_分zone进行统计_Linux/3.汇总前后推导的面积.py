CSVS_PATH=r"D:\地理所\论文\东南亚10m人工林提取\数据\正式分类_10.29\4.GEE导出结果_结果合并_马尔可夫模型_前后向推导_转为等面积投影\面积统计结果\面积统计结果"
CSVS_PATH_ALT=r"D:\地理所\论文\东南亚10m人工林提取\数据\正式分类_10.29\4.GEE导出结果_结果合并_马尔可夫模型_前后向推导_转 为等面积投影\面积统计结果\面积统计结果"
OUT_PUT=r"Z:\Mywork\论文\东南亚10m人工林提取\制图\5.各个区域的像素个数\表格_前后推导"
OUT_PUT_ZONE_SUM=r"Z:\Mywork\论文\东南亚10m人工林提取\制图\5.各个区域的像素个数\表格_前后推导_zone汇总"
import os
import csv
from collections import defaultdict
import glob

LOG_PATH=os.path.join(OUT_PUT,"summary.log")

def _log(msg):
    ts=[]
    try:
        os.makedirs(OUT_PUT,exist_ok=True)
        with open(LOG_PATH,'a',encoding='utf-8') as f:
            f.write(str(msg)+"\n")
    except Exception:
        pass

def read_zone_year(path):
    if not os.path.exists(path):
        _log(f"缺失文件: {path}")
        return []
    rows=[]
    with open(path,'r',encoding='utf-8') as f:
        r=csv.reader(f)
        header=next(r,None)
        for row in r:
            if not row or len(row)<7:
                continue
            rows.append({
                'country':row[0],
                'zone':row[1],
                'year':int(row[2]),
                'pf':int(row[3]),
                'nf':int(row[4]),
                'others':int(row[5]),
                'total':int(row[6])
            })
    return rows

def write_country_csv(country,rows,out_dir):
    os.makedirs(out_dir,exist_ok=True)
    out_path=os.path.join(out_dir,f"{country}_zone1_zone10_2017_2024.csv")
    rows_sorted=sorted(rows,key=lambda x:(x['year'],x['zone']))
    with open(out_path,'w',newline='',encoding='utf-8') as f:
        w=csv.writer(f)
        w.writerow(['国家名称','zone','年份','PF像素数量','NF像素数量','OTHERS像素数量','总像素数量'])
        for r in rows_sorted:
            w.writerow([r['country'],r['zone'],r['year'],r['pf'],r['nf'],r['others'],r['total']])

def write_all_csv(summary,out_dir):
    os.makedirs(out_dir,exist_ok=True)
    out_path=os.path.join(out_dir,"ALL_zone1_zone10_2017_2024.csv")
    keys_sorted=sorted(summary.items(),key=lambda kv:(kv[0][1],kv[0][0]))
    with open(out_path,'w',newline='',encoding='utf-8') as f:
        w=csv.writer(f)
        w.writerow(['国家名称','zone','年份','PF像素数量','NF像素数量','OTHERS像素数量','总像素数量'])
        for (zone,year),vals in keys_sorted:
            w.writerow(['ALL',f"zone{zone}",year,vals['pf'],vals['nf'],vals['others'],vals['total']])

def write_country_year_sum(country,year_sums,out_dir):
    os.makedirs(out_dir,exist_ok=True)
    out_path=os.path.join(out_dir,f"{country}_year_sum_2017_2024.csv")
    rows=[]
    for y in sorted(year_sums.keys()):
        vals=year_sums[y]
        pf=vals['pf']; nf=vals['nf']; ot=vals['others']; tot=vals['total']
        rows.append([country,y,pf,nf,ot,tot,pf/10000.0,nf/10000.0,ot/10000.0,tot/10000.0])
    with open(out_path,'w',newline='',encoding='utf-8') as f:
        w=csv.writer(f)
        w.writerow(['国家名称','年份','PF像素数量','NF像素数量','OTHERS像素数量','总像素数量','PF面积(km2)','NF面积(km2)','OTHERS面积(km2)','总面积(km2)'])
        w.writerows(rows)

def write_all_year_sum(all_year_sums,out_dir):
    os.makedirs(out_dir,exist_ok=True)
    out_path=os.path.join(out_dir,"ALL_year_sum_2017_2024.csv")
    rows=[]
    for y in sorted(all_year_sums.keys()):
        vals=all_year_sums[y]
        pf=vals['pf']; nf=vals['nf']; ot=vals['others']; tot=vals['total']
        rows.append(['ALL',y,pf,nf,ot,tot,pf/10000.0,nf/10000.0,ot/10000.0,tot/10000.0])
    with open(out_path,'w',newline='',encoding='utf-8') as f:
        w=csv.writer(f)
        w.writerow(['国家名称','年份','PF像素数量','NF像素数量','OTHERS像素数量','总像素数量','PF面积(km2)','NF面积(km2)','OTHERS面积(km2)','总面积(km2)'])
        w.writerows(rows)

def main():
    base_dir = CSVS_PATH if os.path.isdir(CSVS_PATH) else (CSVS_PATH_ALT if os.path.isdir(CSVS_PATH_ALT) else CSVS_PATH)
    zones=list(range(1,10+1))
    years=list(range(2017,2024+1))
    countries=defaultdict(list)
    country_year_sum=defaultdict(lambda: defaultdict(lambda:{'pf':0,'nf':0,'others':0,'total':0}))
    summary=defaultdict(lambda:{'pf':0,'nf':0,'others':0,'total':0})
    for z in zones:
        for y in years:
            p=os.path.join(base_dir,f"zone{z}_{y}.csv")
            if not os.path.exists(p):
                candidates=glob.glob(os.path.join(base_dir,f"zone{z}_{y}.csv"))
                p=candidates[0] if candidates else p
            rows=read_zone_year(p)
            _log(f"读取: zone{z}_{y}.csv 行数={len(rows)} 路径={p}")
            for r in rows:
                countries[r['country']].append(r)
                cy=country_year_sum[r['country']][r['year']]
                cy['pf']+=r['pf']; cy['nf']+=r['nf']; cy['others']+=r['others']; cy['total']+=r['total']
                k=(z,y)
                summary[k]['pf']+=r['pf']
                summary[k]['nf']+=r['nf']
                summary[k]['others']+=r['others']
                summary[k]['total']+=r['total']
    _log(f"国家数量={len(countries)}")
    for c,rows in countries.items():
        write_country_csv(c,rows,OUT_PUT)
    write_all_csv(summary,OUT_PUT)
    all_year_sum=defaultdict(lambda:{'pf':0,'nf':0,'others':0,'total':0})
    for c,ys in country_year_sum.items():
        write_country_year_sum(c,ys,OUT_PUT_ZONE_SUM)
        for y,vals in ys.items():
            all_year_sum[y]['pf']+=vals['pf']
            all_year_sum[y]['nf']+=vals['nf']
            all_year_sum[y]['others']+=vals['others']
            all_year_sum[y]['total']+=vals['total']
    write_all_year_sum(all_year_sum,OUT_PUT_ZONE_SUM)

if __name__=="__main__":
    main()
