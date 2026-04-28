// ================= PyCCD数据提取配置 =================
var studyPoints = TimorLeste;
var beginYear = 1985;
var finishYear = 2024;
var outputFolder = 'TimorLeste_PyCCD_RAW';
var outputBands = ['B','G','R','NIR','SWIR1','SWIR2','THERMAL']; // 包含热红外
var maxCloudCover = 100; // 不过滤云量，让PyCCD处理
var spatialScale = 30;


// 执行第一批（修改这个变量来处理不同批次）
var batchSize = 20;
var currentBatch = 1;

print('开始提取Landsat原始数据用于PyCCD处理...');
print('时间范围: ' + beginYear + ' 至 ' + finishYear);

// 载入点数据
var samplePoints = ee.FeatureCollection(studyPoints);
var aoiBounds = samplePoints.geometry().bounds();

print('从资产中加载采样点');

// 完全重写的波段统一函数（分别处理每个传感器）
function standardizeBandsForPyCCD(inputImg) {
  // 获取影像的传感器信息
  var platform = ee.String(inputImg.get('SPACECRAFT_ID'));
  
  // 根据传感器平台选择正确的波段
  var standardizedImg = ee.Algorithms.If(
    platform.equals('LANDSAT_8').or(platform.equals('LANDSAT_9')),
    // L8/L9 波段选择
    inputImg.select(
      ['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7','ST_B10','QA_PIXEL'], 
      ['B','G','R','NIR','SWIR1','SWIR2','THERMAL','QA']
    ),
    // L5/L7 波段选择  
    inputImg.select(
      ['SR_B1','SR_B2','SR_B3','SR_B4','SR_B5','SR_B7','ST_B6','QA_PIXEL'], 
      ['B','G','R','NIR','SWIR1','SWIR2','THERMAL','QA']
    )
  );
  
  return ee.Image(standardizedImg).copyProperties(inputImg, [
    'system:time_start','CLOUD_COVER','LANDSAT_SCENE_ID','SPACECRAFT_ID'
  ]);
}

// 单影像点值采样函数（无云掩膜）
function sampleRawValues(inputImg) {
  var imgTime = ee.Date(inputImg.get('system:time_start'));
  var dateString = imgTime.format('YYYY-MM-dd');
  var daysSince1970 = imgTime.difference(ee.Date('1970-01-01'), 'day');
  
  var sceneID = inputImg.get('LANDSAT_SCENE_ID');
  var cloudPercent = inputImg.get('CLOUD_COVER');
  
  var sampledValues = inputImg.reduceRegions({
    collection: samplePoints,
    reducer: ee.Reducer.first(),
    scale: spatialScale,
    tileScale: 4
  });
  
  var enrichedSamples = sampledValues.map(function(feat) {
    return feat.set({
      'date_str': dateString,
      'days_since_1970': daysSince1970,
      'scene_id': sceneID,
      'cloud_cover': cloudPercent
    });
  });
  
  return enrichedSamples;
}

// 年度数据处理函数（分别处理每个传感器避免波段混淆）
function extractRawDataForYear(targetYear) {
  print('提取年份: ' + targetYear + ' 的原始数据');
  
  var yearStart = ee.Date.fromYMD(targetYear, 1, 1);
  var yearEnd = ee.Date.fromYMD(targetYear, 12, 31);

  // 分别处理每个传感器，避免波段混淆
  var l5Collection = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2')
    .filterBounds(aoiBounds)
    .filterDate(yearStart, yearEnd)
    .filter(ee.Filter.lt('CLOUD_COVER', maxCloudCover))
    .map(function(img) {
      return img.select(
        ['SR_B1','SR_B2','SR_B3','SR_B4','SR_B5','SR_B7','ST_B6','QA_PIXEL'], 
        ['B','G','R','NIR','SWIR1','SWIR2','THERMAL','QA']
      ).copyProperties(img, ['system:time_start','CLOUD_COVER','LANDSAT_SCENE_ID']);
    });
    
  var l7Collection = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2')
    .filterBounds(aoiBounds)
    .filterDate(yearStart, yearEnd)
    .filter(ee.Filter.lt('CLOUD_COVER', maxCloudCover))
    .map(function(img) {
      return img.select(
        ['SR_B1','SR_B2','SR_B3','SR_B4','SR_B5','SR_B7','ST_B6','QA_PIXEL'], 
        ['B','G','R','NIR','SWIR1','SWIR2','THERMAL','QA']
      ).copyProperties(img, ['system:time_start','CLOUD_COVER','LANDSAT_SCENE_ID']);
    });
    
  var l8Collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    .filterBounds(aoiBounds)
    .filterDate(yearStart, yearEnd)
    .filter(ee.Filter.lt('CLOUD_COVER', maxCloudCover))
    .map(function(img) {
      return img.select(
        ['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7','ST_B10','QA_PIXEL'], 
        ['B','G','R','NIR','SWIR1','SWIR2','THERMAL','QA']
      ).copyProperties(img, ['system:time_start','CLOUD_COVER','LANDSAT_SCENE_ID']);
    });
    
  var l9Collection = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
    .filterBounds(aoiBounds)
    .filterDate(yearStart, yearEnd)
    .filter(ee.Filter.lt('CLOUD_COVER', maxCloudCover))
    .map(function(img) {
      return img.select(
        ['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7','ST_B10','QA_PIXEL'], 
        ['B','G','R','NIR','SWIR1','SWIR2','THERMAL','QA']
      ).copyProperties(img, ['system:time_start','CLOUD_COVER','LANDSAT_SCENE_ID']);
    });

  // 合并所有已处理的数据
  var annualCollection = l5Collection.merge(l7Collection).merge(l8Collection).merge(l9Collection);

  var availableCount = annualCollection.size();
  
  // 添加调试信息
  // print('年份 ' + targetYear + ' 总影像数: ' + availableCount);
  print('  L5影像数: ' + l5Collection.size());
  print('  L7影像数: ' + l7Collection.size());
  print('  L8影像数: ' + l8Collection.size());
  print('  L9影像数: ' + l9Collection.size());
  
  var processedData = ee.Algorithms.If(
    availableCount.gt(0),
    annualCollection.map(sampleRawValues).flatten(),
    ee.FeatureCollection([]) // 空集合
  );

  var columnOrder = [
    'id', 'date_str', 'days_since_1970', 'scene_id', 'cloud_cover',
    'B', 'G', 'R', 'NIR', 'SWIR1', 'SWIR2', 'THERMAL', 'QA'
  ];

  Export.table.toDrive({
    collection: ee.FeatureCollection(processedData),
    description: outputFolder+'_raw_data_' + targetYear,
    folder: outputFolder,
    fileNamePrefix: 'raw_data_' + targetYear,
    fileFormat: 'CSV',
    selectors: columnOrder
  });

  print('年份 ' + targetYear + ' 原始数据导出任务已创建');
  
  return {
    year: targetYear,
    status: 'processed'
  };
}

// 分批处理（每批3年，减少内存使用）

var processingYears = [];
for (var y = beginYear; y <= finishYear; y++) {
  processingYears.push(y);
}

print('待处理年份列表: [' + processingYears.join(', ') + ']');
print('采用分批处理模式，每批 ' + batchSize + ' 年');

// 分批处理函数
function processBatch(startIdx, endIdx) {
  var batchYears = processingYears.slice(startIdx, endIdx);
  print('处理批次年份: [' + batchYears.join(', ') + ']');
  
  var batchResults = batchYears.map(extractRawDataForYear);
  
  batchResults.forEach(function(res) {
    print('年份 ' + res.year + ' - 状态: ' + res.status);
  });
  
  return batchResults;
}


var startIndex = currentBatch * batchSize;
var endIndex = Math.min(startIndex + batchSize, processingYears.length);

print('=== 当前处理批次 ' + (currentBatch + 1) + ' ===');
print('年份索引范围: ' + startIndex + ' 到 ' + (endIndex - 1));

var currentResults = processBatch(startIndex, endIndex);

print('当前批次的导出任务已创建');
print('完成后，请修改 currentBatch 变量继续处理下一批');
print('总共需要处理 ' + Math.ceil(processingYears.length / batchSize) + ' 个批次');

// ================= 使用说明 =================
print('');
print('=== PyCCD数据处理流程 ===');
print('1. GEE提取原始光谱值和QA标志');
print('2. 下载CSV文件到本地');
print('3. 使用Python和PyCCD进行变化检测:');
print('   import ccd');
print('   results = ccd.detect(dates, blues, greens, reds, nirs, swir1s, swir2s, thermals, qas)');
print('4. PyCCD会自动处理云掩膜和时间序列分析');

