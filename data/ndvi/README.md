# NDVI Data Files — All Sites

All NDVI time-series files used by the Indian Forest Phenology Predictor.  
**32 files | 3 sensors | 20+ sites | 2016–2025**

---

## File Types

| Prefix | Sensor | Resolution | How to use in app |
|--------|--------|------------|-------------------|
| `MODIS_NDVI_*` | MODIS MOD13Q1 | 250m, 16-day composite | Upload directly — no filtering needed |
| `S2_monthly_*` | Sentinel-2 SR | 10m, monthly median | Upload directly (alpine sites need NDVI filter first — see below) |
| `FUSION_fused_*` | MODIS + S2 merged | Monthly | Upload directly — best quality for mangrove and montane sites |

---

## MODIS Files (15 sites)

| File | Site | Forest Type | App Setting | Seasons | Status |
|------|------|-------------|-------------|---------|--------|
| `MODIS_NDVI_TDD_Tirupati_2016_2025.csv` | IIT Tirupati, AP | Tropical Dry Deciduous | Jun–May | 10/10 | ✅ Ready |
| `MODIS_NDVI_TDD_Mudumalai_2016_2025.csv` | Mudumalai TR, TN | Tropical Dry Deciduous | Jun–May | 10/10 | ✅ Ready |
| `MODIS_NDVI_TMD_Simlipal_2016_2025.csv` | Simlipal BR, Odisha | Tropical Moist Deciduous | Jun–May | 10/10 | ✅ Ready |
| `MODIS_NDVI_TMD_Bastar_2016_2025.csv` | Bastar, CG | Tropical Moist Deciduous | Jun–May | 10/10 | ✅ Ready |
| `MODIS_NDVI_TWE_Agumbe_2016_2025.csv` | Agumbe, Karnataka | Tropical Wet Evergreen | Jan–Dec | 10/10 | ✅ Ready |
| `MODIS_NDVI_TWE_SilentValley_2016_2025.csv` | Silent Valley NP, Kerala | Tropical Wet Evergreen | Jan–Dec | 10/10 | ✅ Ready |
| `MODIS_NDVI_SHO_Mukurthi_2016_2025.csv` | Mukurthi NP, TN | Shola Forest | Jan–Dec | 10/10 | ✅ Ready |
| `MODIS_NDVI_SHO_Eravikulam_2016_2025.csv` | Eravikulam NP, Kerala | Shola Forest | Jan–Dec | 10/10 | ✅ Ready |
| `MODIS_NDVI_MNG_Bhitarkanika_2016_2025.csv` | Bhitarkanika NP, Odisha | Mangrove Forest | Jan–Dec | 10/10 | ✅ Ready |
| `MODIS_NDVI_MNG_Sundarbans_2016_2025.csv` | Sundarbans, WB | Mangrove Forest | Jan–Dec | 10/10 | ✅ Ready |
| `MODIS_NDVI_NEE_Kaziranga_2016_2025.csv` | Kaziranga NP, Assam | NE India Moist Evergreen | Jun–May | 10/10 | ✅ Ready |
| `MODIS_NDVI_NEE_Cherrapunji_2016_2025.csv` | Cherrapunji, Meghalaya | NE India Moist Evergreen | Jan–Dec | 10/10 | ✅ Ready |
| `MODIS_NDVI_KHF_Warangal_2016_2025.csv` | Warangal, Telangana | Kharif / Summer Crop | Jun–Oct | 10/10 | ✅ Ready |
| `MODIS_NDVI_KHF_Cuttack_2016_2025.csv` | Cuttack, Odisha | Kharif / Summer Crop | Jun–Oct | 10/10 | ✅ Ready |
| `MODIS_NDVI_ALL_SITES_2016_2025.csv` | All MODIS sites combined | — | — | — | 📊 Overview only |

---

## Sentinel-2 Monthly Files (10 sites)

| File | Site | Forest Type | App Setting | Notes |
|------|------|-------------|-------------|-------|
| `S2_monthly_ALP_Spiti_2016_2025.csv` | Spiti Valley, HP | Alpine / Subalpine | May–Oct | ⚠️ Remove NDVI < 0.01 before loading |
| `S2_monthly_ALP_ValleyFlowers_2016_2025.csv` | Valley of Flowers, UK | Alpine / Subalpine | May–Oct | ⚠️ Remove NDVI < 0.05 before loading |
| `S2_monthly_TTF_Jaisalmer_2016_2025.csv` | Desert NP, Rajasthan | Tropical Thorn Scrub | Jun–May | ⚠️ Use SOS threshold 10–12% |
| `S2_monthly_TTF_Ranthambore_2016_2025.csv` | Ranthambore NP, Rajasthan | Tropical Thorn Scrub | Jun–May | ✅ Ready |
| `S2_monthly_TDE_Pichavaram_2016_2025.csv` | Pichavaram, TN | Tropical Dry Evergreen | Jan–Dec | ✅ Ready |
| `S2_monthly_TDE_PointCalimere_2016_2025.csv` | Point Calimere, TN | Tropical Dry Evergreen | Jan–Dec | ✅ Ready |
| `S2_monthly_SBH_Rajaji_2016_2025.csv` | Rajaji NP, Uttarakhand | Subtropical Hill Forest | Apr–Mar | ✅ Ready |
| `S2_monthly_SBH_Manas_2016_2025.csv` | Manas NP, Assam | Subtropical Hill Forest | Apr–Mar | ✅ Ready |
| `S2_monthly_RBC_Hisar_2016_2025.csv` | Hisar, Haryana | Rabi / Winter Crop | Nov–Apr | ✅ Ready |
| `S2_monthly_RBC_Ludhiana_2016_2025.csv` | Ludhiana, Punjab | Rabi / Winter Crop | Nov–Apr | ✅ Ready |
| `S2_monthly_ALL10SITES_2016_2025.csv` | All S2 sites combined | — | — | 📊 Overview only |

---

## Fusion Files — MODIS + Sentinel-2 Combined (4 sites)

Best quality for cloud-prone sites. MODIS fills gaps where S2 is cloud-blocked.

| File | Site | Forest Type | App Setting | Notes |
|------|------|-------------|-------------|-------|
| `FUSION_fused_MNG_Bhitarkanika_2016_2025.csv` | Bhitarkanika NP, Odisha | Mangrove Forest | Jan–Dec | ✅ Better than MODIS-only |
| `FUSION_fused_MNG_Sundarbans_2016_2025.csv` | Sundarbans, WB | Mangrove Forest | Jan–Dec | ✅ Better than MODIS-only |
| `FUSION_fused_MTF_Kedarnath_2016_2025.csv` | Kedarnath WS, UK | Montane Temperate Forest | Apr–Nov | ✅ Ready |
| `FUSION_fused_MTF_GreatHimal_2016_2025.csv` | Great Himalayan NP, HP | Montane Temperate Forest | Apr–Nov | ✅ Ready |
| `FUSION_fused_ALL_SITES_2016_2025.csv` | All fusion sites combined | — | — | 📊 Overview only |

---

## Pre-filtering Instructions (Alpine Sites Only)

### Spiti Valley
Remove rows where `NDVI < 0.01` (snow-covered months) and `month` outside 5–10.

### Valley of Flowers  
Remove rows where `NDVI < 0.05` and `month` outside 5–10. Use years 2018–2025 only.

### Jaisalmer Thorn Scrub
Do **NOT** remove low NDVI values — they are real desert bare-soil signal.  
Set SOS threshold to **10–12%** in the app sidebar instead.

---

## Column Format

All files share this common structure (extra columns are ignored by the app):

| Column | Required? | Description |
|--------|-----------|-------------|
| `date` | ✅ Yes | Observation date — YYYY-MM-DD |
| `NDVI` | ✅ Yes | NDVI value 0–1 |
| `NDSI` | S2 only | Snow index — use to filter alpine sites |
| `n_scenes` | S2 only | Number of scenes in monthly composite |
| `source` | Fusion only | Which sensor contributed this observation |

The app auto-detects `date` and `NDVI` columns — all other columns are ignored.
