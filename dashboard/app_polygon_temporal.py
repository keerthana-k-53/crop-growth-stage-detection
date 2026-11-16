import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Intelligent Crop Growth Monitoring",
    page_icon="🌱",
    layout="wide"
)


# ============================================================
# CONSTANTS
# ============================================================

STAGE_NAMES = {
    0: "Early Growth",
    1: "Mid Growth",
    2: "Late Growth"
}


# ============================================================
# LOAD DATA
# ============================================================

    
@st.cache_data
def load_data():
    # Project-defined NDVI category rule (not field-validated crop phenology).
    def classify_stage(ndvi):
        if pd.isna(ndvi):
            return None
        if ndvi < 0.45:
            return "Early Growth"
        if ndvi < 0.60:
            return "Mid Growth"
        return "Late Growth"

    # Load the existing three-location dataset
    df = pd.read_csv(DATA_DIR / "crop_growth_stage_dataset.csv")
    df["Date"] = pd.to_datetime(df["Date"])

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["NDVI"] = pd.to_numeric(df["NDVI"], errors="coerce")

    # Ensure the existing dataset has the stage column used by the dashboard.
    if "Growth_Stage" not in df.columns:
        if "Stage_Label" in df.columns:
            df["Growth_Stage"] = df["Stage_Label"].map({
                0: "Early Growth", 1: "Mid Growth", 2: "Late Growth",
                "0": "Early Growth", "1": "Mid Growth", "2": "Late Growth"
            })
        else:
            df["Growth_Stage"] = df["NDVI"].apply(classify_stage)

    # Load and add Shravanabelagola only if its CSV is present.
    new_path = DATA_DIR / "shravanabelagola_ndvi_2025.csv"
    if new_path.exists():
        new_df = pd.read_csv(new_path)
        required = {"Location", "Date", "NDVI"}
        missing = required - set(new_df.columns)
        if missing:
            raise ValueError(
                f"{new_path.name} is missing required columns: {', '.join(sorted(missing))}"
            )

        new_df["Date"] = pd.to_datetime(new_df["Date"], errors="coerce")
        new_df["NDVI"] = pd.to_numeric(new_df["NDVI"], errors="coerce")

        # Create project-defined NDVI categories; leave missing NDVI unlabelled.
        new_df["Growth_Stage"] = new_df["NDVI"].apply(classify_stage)
        df = pd.concat([df, new_df], ignore_index=True)

    # Keep rows with valid dates, including months where NDVI is missing.
    # Missing NDVI must remain missing, not be treated as zero.
    df = df.dropna(subset=["Date", "Location"]).copy()
    return df



@st.cache_data
def load_temporal_data():

    df = pd.read_csv(
        DATA_DIR / "crop_growth_temporal_features.csv"
    )

    df["Date"] = pd.to_datetime(df["Date"])

    return df


@st.cache_resource
def load_baseline_model():

    return joblib.load(
        str(MODELS_DIR / "crop_growth_stage_model.pkl")
    )


@st.cache_resource
def load_temporal_model():

    return joblib.load(
        str(MODELS_DIR / "crop_growth_temporal_model.pkl")
    )


df = load_data()
temporal_df = load_temporal_data()

baseline_model = load_baseline_model()
temporal_model = load_temporal_model()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🌾 Crop Growth Monitoring")

st.sidebar.markdown("""
### Project Information

**Satellite:** Sentinel-2

**Platform:** Google Earth Engine

**Machine Learning:** Random Forest

**Study Locations:**  
See the location selector in the dashboard.

**Year:** 2025
""")

st.sidebar.markdown("---")

st.sidebar.success(
    "Project Status\n\n"
    "✅ Satellite Data Processed\n\n"
    "✅ NDVI Time Series Created\n\n"
    "✅ ML Models Trained"
)


# ============================================================
# TITLE
# ============================================================

st.title(
    "🌱 Intelligent Crop Growth Stage Monitoring"
)

st.markdown(
    """
    ### Satellite-Based Crop Monitoring Using Sentinel-2 NDVI Time Series

    This dashboard uses Sentinel-2 satellite imagery processed through
    Google Earth Engine to analyze vegetation conditions across
    all locations available in the loaded dataset.

    Monthly NDVI values are used to monitor vegetation variation
    and support crop growth-stage classification.
    """
)


# ============================================================
# DRAW A NEW AREA AND ANALYZE LIVE SENTINEL-2 NDVI
# ============================================================

st.divider()
st.header("🗺️ Analyze a New Area (Draw a Polygon)")
st.markdown(
    "Draw a polygon anywhere on the map, choose a year, and calculate "
    "monthly Sentinel-2 NDVI for the selected area. This uses live GEE data "
    "and is separate from the saved study-location dataset."
)

try:
    import folium
    from folium.plugins import Draw
    from streamlit_folium import st_folium
except ImportError:
    folium = None
    Draw = None
    st_folium = None

if folium is None or st_folium is None:
    st.warning(
        "The map feature needs extra packages. In your project terminal run: "
        "`pip install folium streamlit-folium earthengine-api`, then restart Streamlit."
    )
else:
    map_col, control_col = st.columns([2, 1])
    with control_col:
        analysis_year = st.selectbox(
            "Analysis year", list(range(2020, 2027)), index=5, key="polygon_analysis_year"
        )
        st.caption("Use the map's polygon tool to draw an area. You can zoom and pan to other countries/regions.")
        run_polygon_analysis = st.button(
            "Analyze Drawn Polygon", type="primary", key="run_polygon_analysis"
        )
        st.caption("A polygon may include non-crop land. NDVI alone cannot confirm crop type or biological growth stage.")

    draw_map = folium.Map(location=[20.5937, 78.9629], zoom_start=5, control_scale=True)
    Draw(
        export=False,
        draw_options={
            "polyline": False, "rectangle": True, "circle": False,
            "circlemarker": False, "marker": False, "polygon": True
        },
        edit_options={"edit": True, "remove": True},
    ).add_to(draw_map)
    with map_col:
        map_result = st_folium(
            draw_map, height=480, width=None, key="new_area_draw_map",
            returned_objects=["all_drawings"]
        )

    drawings = (map_result or {}).get("all_drawings") or []
    polygon_feature = None
    for feature in drawings:
        if feature.get("geometry", {}).get("type") in ("Polygon", "MultiPolygon"):
            polygon_feature = feature
            break

    
    polygon = None
    if polygon_feature:
        try:
            import ee
            ee.Initialize(project="crop-growth-stage-detection")

            coords = polygon_feature["geometry"]["coordinates"]
            if polygon_feature["geometry"]["type"] == "Polygon":
                polygon = ee.Geometry.Polygon(coords)
            else:
                polygon = ee.Geometry.MultiPolygon(coords)

            area_km2 = polygon.area(maxError=1).divide(1_000_000).getInfo()
            st.info(f"Polygon detected · approximate area: {area_km2:.3f} km²")

        except Exception as exc:
            st.warning(f"Could not calculate polygon area: {exc}")

        if run_polygon_analysis:
            try:
                import ee
                try:
                    ee.Initialize(project="crop-growth-stage-detection")
                except Exception:
                    raise RuntimeError(
                        "Google Earth Engine is not authenticated. Run `earthengine authenticate` "
                        "in the same Python environment, then restart the app. If your account "
                        "requires a Cloud project, configure ee.Initialize(project='YOUR_CLOUD_PROJECT_ID')."
                    )

                geometry = polygon
                start_date = f"{analysis_year}-01-01"
                end_date = f"{analysis_year + 1}-01-01"

                def mask_sentinel2(image):
                    scl = image.select("SCL")
                    clear = (
                        scl.neq(1).And(scl.neq(3)).And(scl.neq(8))
                        .And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
                    )
                    return image.updateMask(clear).copyProperties(image, ["system:time_start"])

                collection = (
                    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                    .filterBounds(geometry)
                    .filterDate(start_date, end_date)
                    .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", 80))
                    .map(mask_sentinel2)
                )
                scene_count = collection.size().getInfo()
                if scene_count == 0:
                    st.warning("No Sentinel-2 scenes matched this polygon and year. Try another year or area.")
                else:
                    records = []
                    progress = st.progress(0, text="Calculating monthly NDVI…")
                    for month in range(1, 13):
                        month_start = ee.Date.fromYMD(analysis_year, month, 1)
                        month_end = month_start.advance(1, "month")
                        monthly = collection.filterDate(month_start, month_end)
                        count = monthly.size().getInfo()
                        if count == 0:
                            records.append({"Month": f"{analysis_year}-{month:02d}", "Image_Count": 0, "NDVI": None, "Growth_Stage": "No data"})
                        else:
                            composite = monthly.median()
                            ndvi = composite.normalizedDifference(["B8", "B4"]).rename("NDVI")
                            result = ndvi.reduceRegion(
                                reducer=ee.Reducer.mean(), geometry=geometry,
                                scale=10, maxPixels=1e9, bestEffort=True
                            ).get("NDVI").getInfo()
                            if result is None:
                                stage = "No valid pixels"
                            elif result < 0.45:
                                stage = "Early Growth (NDVI rule)"
                            elif result < 0.60:
                                stage = "Mid Growth (NDVI rule)"
                            else:
                                stage = "Late Growth (NDVI rule)"
                            records.append({
                                "Month": f"{analysis_year}-{month:02d}",
                                "Image_Count": count,
                                "NDVI": result,
                                "Growth_Stage": stage,
                            })
                        progress.progress(month / 12, text=f"Processed {month}/12 months")
                    result_df = pd.DataFrame(records)

                    # Generate temporal features from this newly drawn polygon's own
                    # valid monthly NDVI history, then run the existing trained model.
                    # This uses the same feature order as the existing dashboard:
                    # Previous NDVI, Month Gap, NDVI Change, NDVI Trend.
                    result_df["Temporal_Predicted_Stage"] = None
                    valid_rows = result_df.dropna(subset=["NDVI"]).copy()
                    valid_rows["Parsed_Month"] = pd.to_datetime(valid_rows["Month"], format="%Y-%m")
                    valid_rows = valid_rows.sort_values("Parsed_Month")
                    temporal_errors = []

                    for row_position in range(1, len(valid_rows)):
                        previous = valid_rows.iloc[row_position - 1]
                        current = valid_rows.iloc[row_position]
                        month_gap = (
                            (current["Parsed_Month"].year - previous["Parsed_Month"].year) * 12
                            + current["Parsed_Month"].month - previous["Parsed_Month"].month
                        )
                        ndvi_change = float(current["NDVI"] - previous["NDVI"])
                        ndvi_trend = ndvi_change / month_gap if month_gap > 0 else 0.0
                        features = [[
                            float(previous["NDVI"]),
                            float(month_gap),
                            ndvi_change,
                            ndvi_trend,
                        ]]
                        try:
                            prediction = temporal_model.predict(features)[0]
                            predicted_label = STAGE_NAMES.get(int(prediction), f"Unknown class ({prediction})")
                            result_df.loc[current.name, "Temporal_Predicted_Stage"] = predicted_label
                        except Exception as exc:
                            temporal_errors.append(str(exc))
                            break

                    st.subheader(f"Monthly NDVI results — {analysis_year}")
                    st.caption(f"Sentinel-2 scenes matching the year and polygon before monthly grouping: {scene_count}. Image counts are not the same as valid-pixel counts.")
                    st.dataframe(result_df, use_container_width=True, hide_index=True)
                    if temporal_errors:
                        st.warning(
                            "Temporal model predictions could not be generated. Check that the saved model "
                            "expects [Previous NDVI, Month Gap, NDVI Change, NDVI Trend]. Details: "
                            + temporal_errors[0]
                        )
                    elif len(valid_rows) < 2:
                        st.info("At least two valid monthly NDVI observations are required for a temporal prediction.")
                    else:
                        st.caption(
                            "Temporal predictions use the previous valid observation for this polygon. "
                            "The first valid month has no prediction because no earlier observation is available. "
                            "These are model estimates, not field-validated crop stages."
                        )
                    chart_df = result_df.dropna(subset=["NDVI"]).copy()
                    if not chart_df.empty:
                        st.line_chart(chart_df.set_index("Month")["NDVI"])
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Valid months", int(chart_df["NDVI"].notna().sum()))
                        m2.metric("Mean NDVI", f"{chart_df['NDVI'].mean():.3f}")
                        max_row = chart_df.loc[chart_df["NDVI"].idxmax()]
                        m3.metric("Highest NDVI", f"{max_row['NDVI']:.3f}", str(max_row["Month"]))
                    else:
                        st.warning("No valid monthly NDVI values were returned for this polygon and year.")
                    st.download_button(
                        "Download monthly NDVI CSV",
                        data=result_df.to_csv(index=False).encode("utf-8"),
                        file_name=f"polygon_monthly_ndvi_{analysis_year}.csv",
                        mime="text/csv",
                    )
                    st.warning(
                        "These growth-stage names are project-defined NDVI threshold categories, "
                        "not independently validated crop phenology. The polygon has not been "
                        "automatically verified as farmland."
                    )
            except Exception as exc:
                st.error(f"Polygon analysis failed: {exc}")
                st.caption("Check your internet connection, Earth Engine access, and package installation. The rest of the dashboard should remain available.")
    else:
        st.caption("Draw a polygon or rectangle on the map first. Click 'Analyze Drawn Polygon' after drawing.")

st.divider()


# ============================================================
# TOP METRICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "🛰 Satellite",
        "Sentinel-2"
    )

with col2:
    st.metric(
        "📍 Locations",
        str(df["Location"].nunique())
    )

with col3:
    st.metric(
        "📊 Valid NDVI Observations",
        int(df["NDVI"].notna().sum())
    )

with col4:
    st.metric(
        "🤖 ML Model",
        "Random Forest"
    )


st.divider()


# ============================================================
# LOCATION SELECTION
# ============================================================

st.header("📍 Study Area")

locations = sorted(
    df["Location"].unique()
)

selected_location = st.selectbox(
    "Select study location",
    locations
)


location_df = df[
    df["Location"] == selected_location
].copy()

location_df = location_df.sort_values(
    "Date"
)
valid_location_df = location_df.dropna(subset=["NDVI"]).copy()


# ============================================================
# LOCATION SUMMARY
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Observations",
        int(location_df["NDVI"].notna().sum())
    )

with col2:

    st.metric(
        "Average NDVI",
        round(
            valid_location_df["NDVI"].mean(),
            3
        )
    )

with col3:

    st.metric(
        "Maximum NDVI",
        round(
            valid_location_df["NDVI"].max(),
            3
        )
    )


# ============================================================
# NDVI TIME SERIES
# ============================================================

st.header("📈 Monthly NDVI Time Series")

chart_data = location_df[
    ["Date", "NDVI"]
].copy()

chart_data = chart_data.set_index(
    "Date"
)

st.line_chart(
    chart_data,
    y="NDVI"
)

st.caption(
    "Monthly mean NDVI derived from Sentinel-2 imagery."
)


# ============================================================
# NDVI DATA TABLE
# ============================================================

st.header("📋 NDVI Observations")

display_df = location_df[
    [
        "Date",
        "Location",
        "NDVI",
        "Growth_Stage"
    ]
].copy()

display_df["Date"] = display_df[
    "Date"
].dt.strftime("%Y-%m")

display_df["NDVI"] = display_df[
    "NDVI"
].round(4)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# GROWTH STAGE DISTRIBUTION
# ============================================================

st.header("🌿 Growth Stage Distribution")

stage_counts = (
    valid_location_df["Growth_Stage"]
    .dropna()
    .value_counts()
)

st.bar_chart(
    stage_counts
)


# ============================================================
# SELECT MONTH
# ============================================================

st.header("🎯 Growth Stage Analysis")

available_dates = location_df.loc[location_df["NDVI"].notna(),
    "Date"
].sort_values()

selected_date = st.selectbox(
    "Select month",
    available_dates,
    format_func=lambda x: x.strftime("%B %Y")
)


selected_row = location_df[
    location_df["Date"] == selected_date
].iloc[0]

selected_ndvi = selected_row["NDVI"]

actual_stage = selected_row[
    "Growth_Stage"
]


# ============================================================
# CURRENT NDVI INFORMATION
# ============================================================

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Selected NDVI",
        f"{selected_ndvi:.3f}"
    )

with col2:

    st.metric(
        "NDVI-Based Stage",
        actual_stage
    )


# ============================================================
# STAGE DISPLAY
# ============================================================

if actual_stage == "Early Growth":

    st.warning(
        "🌱 Early Growth"
    )

elif actual_stage == "Mid Growth":

    st.info(
        "🌿 Mid Growth"
    )

else:

    st.success(
        "🌳 Late Growth"
    )


st.caption(
    "Growth stages are defined using project-specific NDVI thresholds."
)


# ============================================================
# TEMPORAL ML PREDICTION
# ============================================================

st.header("🤖 Temporal Random Forest Prediction")

# Build temporal features from the selected location's own valid NDVI history.
# This allows the workflow to run for newly added locations, not only those
# present in the original temporal-features CSV.
valid_history = location_df.dropna(subset=["NDVI"]).sort_values("Date").copy()
current_history = valid_history[valid_history["Date"] == selected_date]

if current_history.empty:
    st.info("No valid NDVI value is available for the selected month.")
elif len(valid_history[valid_history["Date"] < selected_date]) == 0:
    st.info(
        "Temporal ML prediction needs a previous valid NDVI observation "
        "for this location. Select a later month with available history."
    )
else:
    previous_row = valid_history[valid_history["Date"] < selected_date].iloc[-1]
    current_row = current_history.iloc[0]

    month_gap = (
        (current_row["Date"].year - previous_row["Date"].year) * 12
        + current_row["Date"].month - previous_row["Date"].month
    )
    ndvi_change = float(current_row["NDVI"] - previous_row["NDVI"])
    ndvi_trend = ndvi_change / month_gap if month_gap > 0 else 0.0
    temporal_features = [[
        float(previous_row["NDVI"]),
        float(month_gap),
        ndvi_change,
        ndvi_trend
    ]]

    try:
        temporal_prediction = temporal_model.predict(temporal_features)[0]
        predicted_stage = STAGE_NAMES[int(temporal_prediction)]
        st.metric("Temporal ML Predicted Stage", predicted_stage)
        st.caption(
            "Prediction uses the selected location's previous valid NDVI, "
            "month gap, NDVI change, and NDVI trend. This is a model estimate, "
            "not field-validated crop-stage ground truth."
        )
        feature_table = pd.DataFrame({
            "Feature": ["Previous NDVI", "Month Gap", "NDVI Change", "NDVI Trend"],
            "Value": [previous_row["NDVI"], month_gap, ndvi_change, ndvi_trend]
        })
        feature_table["Value"] = feature_table["Value"].round(4)
        st.write("Temporal features used:")
        st.dataframe(feature_table, use_container_width=True, hide_index=True)
    except Exception as exc:
        st.warning(f"Prediction could not be generated: {exc}")


# ============================================================
# ALL LOCATION COMPARISON
# ============================================================


st.header("📊 Location Comparison")

# The main dataframe already contains the new location, if its CSV exists.
comparison_df = df[["Location", "NDVI"]].copy()
comparison_df["NDVI"] = pd.to_numeric(comparison_df["NDVI"], errors="coerce")
comparison_df = comparison_df.dropna(subset=["Location", "NDVI"])

comparison = (
    comparison_df.groupby("Location")["NDVI"]
    .mean()
    .sort_values(ascending=False)
    .round(3)
)

st.bar_chart(comparison)

st.caption(
    "Bars show the mean of available monthly NDVI observations "
    "for each location in the loaded datasets."
)

st.subheader("Monthly NDVI Comparison")
monthly_comparison = df.dropna(subset=["NDVI"]).copy()
monthly_comparison["Month"] = monthly_comparison["Date"].dt.strftime("%Y-%m")
monthly_comparison = monthly_comparison.pivot_table(
    index="Month", columns="Location", values="NDVI", aggfunc="mean"
).sort_index()
st.line_chart(monthly_comparison)
st.caption(
    "Compare locations for months where data is available. Blank chart values "
    "mean no valid NDVI observation was available for that location and month."
)



# ============================================================
# MODEL INFORMATION
# ============================================================

st.header("🧠 Machine Learning Models")

col1, col2 = st.columns(2)

with col1:

    st.subheader(
        "Baseline Random Forest"
    )

    st.write(
        """
        Uses current NDVI as the input feature.
        Growth-stage labels were generated using
        predefined NDVI thresholds.
        """
    )

    st.info(
        "Test accuracy: 100%"
    )

    st.caption(
        "This accuracy should not be interpreted as "
        "independent field validation because the labels "
        "were derived from NDVI."
    )


with col2:

    st.subheader(
        "Temporal Random Forest"
    )

    st.write(
        """
        Uses previous NDVI, month gap, NDVI change,
        and NDVI trend to classify growth stages.
        """
    )

    st.info(
        "Test accuracy: 57.14%"
    )

    st.caption(
        "This is not independent field validation. The target labels "
        "are NDVI-threshold categories, and the dataset is small."
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.header(
    "📌 Temporal Model Feature Importance"
)

feature_importance = pd.DataFrame({
    "Feature": [
        "Previous NDVI",
        "NDVI Change",
        "NDVI Trend",
        "Month Gap"
    ],
    "Importance": [
        0.349921,
        0.310822,
        0.242695,
        0.096561
    ]
})

feature_importance = feature_importance.set_index(
    "Feature"
)

st.bar_chart(
    feature_importance
)





# ============================================================
# TEAM
# ============================================================

st.divider()

st.header("👨‍💻 Development Team")

team = [
    "Suhas Harave",
    "Keerthana K",
    "Ashika J B",
    "Mahadev Prasad T P"
]

for member in team:
    st.write(f"• {member}")


st.info("""
**Bachelor of Engineering**

Information Science and Engineering

**Academic Year:** 2025–2026
""")


# ============================================================
# TECHNOLOGIES
# ============================================================

st.header("🛠 Technologies Used")

st.markdown("""
- 🛰 Sentinel-2 Satellite Imagery
- 🌍 Google Earth Engine
- 🌿 NDVI Time Series
- 🐍 Python
- 🤖 Random Forest
- 📊 Pandas
- 📈 Streamlit
""")