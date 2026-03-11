/**
 * ═══════════════════════════════════════════════════════════════
 * Google Earth Engine Script — Sentinel-2 Monthly NDVI Extraction
 * Indian Forest Phenology Predictor
 * ═══════════════════════════════════════════════════════════════
 *
 * USAGE:
 *   1. Open https://code.earthengine.google.com
 *   2. Paste this script
 *   3. Edit CONFIGURATION section
 *   4. Click Run → Tasks → Run export
 *
 * OUTPUT: CSV with monthly NDVI + NDSI + NDWI + n_scenes
 * SENSOR: Sentinel-2 SR Harmonized (10m)
 * CLOUD MASK: SCL-based (removes cloud, shadow, snow classes)
 *
 * RECOMMENDED FOR:
 *   Alpine / Subalpine (Spiti, Valley of Flowers)
 *   Tropical Thorn Scrub (Jaisalmer)
 *   Subtropical Hill Forest
 * ═══════════════════════════════════════════════════════════════
 */

// ── CONFIGURATION — edit these values ─────────────────────────
var SITE_NAME  = 'Spiti';           // used in output filename
var LATITUDE   = 32.24;             // decimal degrees North
var LONGITUDE  = 78.07;             // decimal degrees East
var BUFFER_M   = 300;               // buffer radius in metres
var START_DATE = '2016-01-01';
var END_DATE   = '2025-12-31';
var SEASON     = 'May-Oct';         // label only (informational)
// ───────────────────────────────────────────────────────────────

var point = ee.Geometry.Point([LONGITUDE, LATITUDE]);
var roi   = point.buffer(BUFFER_M);

// SCL cloud mask: remove cloud shadow(3), medium/high cloud(8,9),
// cirrus(10), snow/ice(11)
function maskS2clouds(img) {
  var scl = img.select('SCL');
  var mask = scl.neq(3).and(scl.neq(8)).and(scl.neq(9))
                .and(scl.neq(10)).and(scl.neq(11));
  return img.updateMask(mask);
}

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterDate(START_DATE, END_DATE)
  .filterBounds(roi)
  .map(maskS2clouds);

// Get distinct year-months to build monthly composites
var months = ee.List.sequence(0, ee.Date(END_DATE).difference(ee.Date(START_DATE), 'month').round());

var timeSeries = months.map(function(offset) {
  var start  = ee.Date(START_DATE).advance(offset, 'month');
  var end    = start.advance(1, 'month');
  var subset = s2.filterDate(start, end);
  var n      = subset.size();

  var composite = subset.median();

  // Band calculations
  var ndvi = composite.normalizedDifference(['B8', 'B4']).rename('NDVI');
  var ndsi = composite.normalizedDifference(['B3', 'B11']).rename('NDSI');
  var ndwi = composite.normalizedDifference(['B3', 'B8']).rename('NDWI');

  var vals = ndvi.addBands(ndsi).addBands(ndwi).reduceRegion({
    reducer:   ee.Reducer.mean(),
    geometry:  roi,
    scale:     10,
    maxPixels: 1e9
  });

  return ee.Feature(null, {
    'date':     start.format('YYYY-MM-dd'),
    'year':     start.get('year'),
    'month':    start.get('month'),
    'NDVI':     vals.get('NDVI'),
    'NDSI':     vals.get('NDSI'),
    'NDWI':     vals.get('NDWI'),
    'n_scenes': n,
    'season':   SEASON,
    'sensor':   'Sentinel2_SR_Harmonized_10m',
    'site_key': SITE_NAME
  });
});

// Remove months with no valid pixels
var clean = ee.FeatureCollection(timeSeries)
  .filter(ee.Filter.notNull(['NDVI']))
  .filter(ee.Filter.gt('n_scenes', 0));

Export.table.toDrive({
  collection:  clean,
  description: 'S2_monthly_' + SITE_NAME + '_' + START_DATE.slice(0,4) + '_' + END_DATE.slice(0,4),
  fileFormat:  'CSV'
});

print('Total monthly composites:', clean.size());

// ── IMPORTANT POST-PROCESSING NOTE ────────────────────────────
// After downloading:
//  - For Alpine sites (Spiti, Valley of Flowers):
//    Remove rows where NDSI > 0.40 OR NDVI < 0.01 (snow-covered)
//    Remove 2016-2017 data if Jun-Sep observations are missing
//  - For Thorn Scrub (Jaisalmer):
//    Do NOT remove low NDVI rows — sparse desert pixels are real
//    Set SOS threshold to 10-12% in the app
// ──────────────────────────────────────────────────────────────
