/**
 * ═══════════════════════════════════════════════════════════════
 * Google Earth Engine Script — MODIS MOD13Q1 NDVI Extraction
 * Indian Forest Phenology Predictor
 * ═══════════════════════════════════════════════════════════════
 *
 * USAGE:
 *   1. Open https://code.earthengine.google.com
 *   2. Paste this script
 *   3. Edit the CONFIGURATION section below
 *   4. Click Run → Tasks → Run export
 *   5. Download CSV from Google Drive
 *
 * OUTPUT: CSV with columns: date, NDVI
 * SENSOR: MODIS MOD13Q1 (16-day composite, 250m)
 * QUALITY: pixel_reliability ≤ 1 (good + marginal pixels only)
 *
 * RECOMMENDED FOR:
 *   All monsoon forest types (Tropical Dry/Moist Deciduous,
 *   Wet Evergreen, NE India, Shola, Mangrove, Thorn Scrub)
 *   where monsoon cloud cover prevents Sentinel-2 usage.
 * ═══════════════════════════════════════════════════════════════
 */

// ── CONFIGURATION — edit these values ─────────────────────────
var SITE_NAME  = 'Tirupati';        // used in output filename
var LATITUDE   = 13.63;             // decimal degrees North
var LONGITUDE  = 79.42;             // decimal degrees East
var BUFFER_M   = 500;               // buffer radius in metres
var START_DATE = '2016-01-01';      // inclusive start
var END_DATE   = '2025-12-31';      // inclusive end
// ───────────────────────────────────────────────────────────────

var point = ee.Geometry.Point([LONGITUDE, LATITUDE]);
var roi   = point.buffer(BUFFER_M);

var modis = ee.ImageCollection('MODIS/061/MOD13Q1')
  .filterDate(START_DATE, END_DATE)
  .filterBounds(roi)
  .select(['NDVI', 'pixel_reliability']);

// Quality filter: keep only good (0) and marginal (1) pixels
var filtered = modis.map(function(img) {
  var qa = img.select('pixel_reliability');
  return img.updateMask(qa.lte(1));
});

// Extract mean NDVI per composite, apply scale factor 0.0001
var timeSeries = filtered.map(function(img) {
  var ndvi = img.select('NDVI').multiply(0.0001);
  var val  = ndvi.reduceRegion({
    reducer:   ee.Reducer.mean(),
    geometry:  roi,
    scale:     250,
    maxPixels: 1e9
  });
  return ee.Feature(null, {
    'date': img.date().format('YYYY-MM-dd'),
    'NDVI': val.get('NDVI')
  });
});

// Remove observations with no valid NDVI (complete cloud cover)
var clean = ee.FeatureCollection(timeSeries)
  .filter(ee.Filter.notNull(['NDVI']));

// Export to Google Drive
Export.table.toDrive({
  collection:  clean,
  description: 'MODIS_NDVI_' + SITE_NAME + '_' + START_DATE.slice(0,4) + '_' + END_DATE.slice(0,4),
  fileFormat:  'CSV',
  selectors:   ['date', 'NDVI']
});

print('Total observations:', clean.size());
print('Sample (first 5):', clean.limit(5));
