/************************************************************
 单年分类脚本 — SingleYear_Classifier
 说明：
 - 使用 SAMPLES 中 year == targetYear 的样本，仅做单年分类（训练样本即所有样本）
 - landcover 编码：1=人工林，2=自然林，3=others
*************************************************************/
var SAMPLES_A = ee.FeatureCollection("projects/ruiduobaonb/assets/Z7_train_SAMPLE");

// 配置（单年）
var areaCodeA = 7;
var targetYearA = 2023;     // 单年分类年份（可改）
var rfTreesA = 100;
var splitA = 1;             // 保持你当前的 split=1 行为（全部作为训练）

var boundaryA = ee.FeatureCollection('projects/ruiduobaonb/assets/BOUNDARY_ZONE_' + areaCodeA);
var EMB_COLL_A = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL');

Map.centerObject(boundaryA, 8);

// 工具函数（A）
function buildYearMosaicA(year) {
  return EMB_COLL_A.filterBounds(boundaryA)
                   .filter(ee.Filter.calendarRange(year, year, 'year'))
                   .mosaic()
                   .clip(boundaryA);
}

function prepareSamplesForYearA(year) {
  return SAMPLES_A.filter(ee.Filter.eq('year', year))
                  .filterBounds(boundaryA)
                  .map(function(f){
                    var lc = ee.Number(f.get('landcover'));
                    return ee.Feature(f.geometry(), f.toDictionary()).set('landcover', lc.toInt());
                  });
}

// 执行单年分类（A）
print('=== SingleYear_Classifier: year =', targetYearA);
var mosaicTargetA = buildYearMosaicA(targetYearA);
var mosaicTrainingA = buildYearMosaicA(targetYearA);

var yearSamplesA = prepareSamplesForYearA(targetYearA);
var sampleCountA = yearSamplesA.size().getInfo();
print('样本总数（点） for', targetYearA, ':', sampleCountA);

if (sampleCountA > 0) {
  var samplesWithBandsA = mosaicTrainingA.sampleRegions({
    collection: yearSamplesA,
    properties: ['landcover'],
    scale: 10,
    tileScale: 2
  });

  var trainingSamplesA = samplesWithBandsA; // 全部作为训练
  print('训练样本数:', trainingSamplesA.size());

  var bandNamesA = mosaicTrainingA.bandNames();
  var classifierA = ee.Classifier.smileRandomForest(rfTreesA)
    .train({
      features: trainingSamplesA,
      classProperty: 'landcover',
      inputProperties: bandNamesA
    });

  var classifiedA = mosaicTargetA.classify(classifierA).rename('classification_' + targetYearA);
  var finalOutputA = classifiedA.toInt8();

  Map.addLayer(finalOutputA, {min: 1, max: 3}, 'classification_' + targetYearA + '_SingleYear');

  // Export.image.toDrive({
  //   image: finalOutputA,
  //   description: 'zone' + areaCodeA + '_singleYear_classification_' + targetYearA + '_rf' + rfTreesA,
  //   folder: 'GEE_Change_NODetection_SINGLE',
  //   scale: 10,
  //   region: boundaryA.geometry(),
  //   maxPixels: 1e13,
  //   crs: 'EPSG:4326'
  // });

  print('SingleYear 导出任务已提交（年份：' + targetYearA + '）');
} else {
  print('SingleYear 脚本：没有找到样本，跳过。');
}

/************************************************************
 逐年分类脚本 — MultiYear_Classifier
 说明：
 - 2017-2024 年逐年处理，每年用当年样本训练并分类（训练样本即该年全部样本）
*************************************************************/
var SAMPLES_B = ee.FeatureCollection("projects/ruiduobaonb/assets/Z7_train_SAMPLE");

// 配置（逐年）
var areaCodeB = 7;
var yearsB = [2017,2018,2019,2020,2021,2022,2023,2024];
var rfTreesB = 100;
var boundaryB = ee.FeatureCollection('projects/ruiduobaonb/assets/BOUNDARY_ZONE_' + areaCodeB);
var EMB_COLL_B = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL');

Map.centerObject(boundaryB, 8);

// 工具函数（B）
function buildYearMosaicB(year) {
  return EMB_COLL_B.filterBounds(boundaryB)
                   .filter(ee.Filter.calendarRange(year, year, 'year'))
                   .mosaic()
                   .clip(boundaryB);
}

function prepareSamplesForYearB(year) {
  return SAMPLES_B.filter(ee.Filter.eq('year', year))
                  .filterBounds(boundaryB)
                  .map(function(f){
                    var lc = ee.Number(f.get('landcover'));
                    return ee.Feature(f.geometry(), f.toDictionary()).set('landcover', lc.toInt());
                  });
}

// 逐年循环（B）
for (var idx = 0; idx < yearsB.length; idx++) {
  var y = yearsB[idx];
  print('-------------------------');
  print('MultiYear_Classifier 处理年份：', y);

  var mosaicTargetB = buildYearMosaicB(y);
  var mosaicTrainingB = buildYearMosaicB(y);

  print('目标/训练年份(' + y + ') 波段数:', mosaicTargetB.bandNames().size());

  var yearSamplesB = prepareSamplesForYearB(y);
  var sampleCountB = yearSamplesB.size().getInfo();
  print('样本总数（点） for', y, ':', sampleCountB);

  if (sampleCountB === 0) {
    print('年份 ' + y + ' 没有样本，已跳过。');
    continue;
  }

  var samplesWithBandsB = mosaicTrainingB.sampleRegions({
    collection: yearSamplesB,
    properties: ['landcover'],
    scale: 10,
    tileScale: 2
  });

  var trainingSamplesB = samplesWithBandsB; // 全部作为训练（与 SingleYear 一致）
  print('训练样本数:', trainingSamplesB.size());

  var bandNamesB = mosaicTrainingB.bandNames();
  var classifierB = ee.Classifier.smileRandomForest(rfTreesB)
    .train({
      features: trainingSamplesB,
      classProperty: 'landcover',
      inputProperties: bandNamesB
    });

  var classifiedB = mosaicTargetB.classify(classifierB).rename('classification_' + y);
  var finalOutputB = classifiedB.toInt8();

  Map.addLayer(finalOutputB, {min: 1, max: 3}, 'classification_' + y + '_MultiYear');

  Export.image.toDrive({
    image: finalOutputB,
    description: 'zone' + areaCodeB + '_classification_' + y + '_rf' + rfTreesB,
    folder: 'GEE_Change_NODetection_ALLSAMPLE',
    scale: 10,
    region: boundaryB.geometry(),
    maxPixels: 1e13,
    crs: 'EPSG:4326'
  });

  print('MultiYear 导出任务已提交（年份：' + y + '）');
}
