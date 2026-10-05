import streamlit as st

import tensorflow as tf

from tensorflow.keras import layers

import numpy as np

import pandas as pd

import joblib

from io import BytesIO





# ============================================================

# LOAD WEATHER DATASET ONLY ONCE PER UPLOADED FILE

# ============================================================



def load_weather_dataset(uploaded_file):



    file_signature = (

        uploaded_file.name,

        uploaded_file.size

    )



    # Reuse the already loaded dataframe when the same

    # file is used during Streamlit reruns.

    if (

        "weather_df" not in st.session_state

        or st.session_state.get("weather_file_signature")

        != file_signature

    ):



        file_bytes = uploaded_file.getvalue()



        if uploaded_file.name.lower().endswith(".csv"):



            df = pd.read_csv(

                BytesIO(file_bytes)

            )



        else:



            df = pd.read_excel(

                BytesIO(file_bytes)

            )



        st.session_state.weather_df = df

        st.session_state.weather_file_signature = file_signature



    return st.session_state.weather_df







# ============================================================

# PAGE CONFIGURATION

# ============================================================



st.set_page_config(

    page_title="Rainfall Prediction System",

    page_icon="🌧️",

    layout="wide"

)





# ============================================================

# CUSTOM TEMPORAL ATTENTION

# ============================================================



class TemporalAttention(layers.Layer):



    def __init__(self, **kwargs):

        super().__init__(**kwargs)



    def build(self, input_shape):



        feature_dim = input_shape[-1]

        time_steps = input_shape[1]



        self.W = self.add_weight(

            name="attention_weight",

            shape=(feature_dim, 1),

            initializer="glorot_uniform",

            trainable=True

        )



        self.b = self.add_weight(

            name="attention_bias",

            shape=(time_steps, 1),

            initializer="zeros",

            trainable=True

        )



        super().build(input_shape)



    def call(self, inputs):



        scores = tf.tanh(

            tf.matmul(inputs, self.W) + self.b

        )



        attention_weights = tf.nn.softmax(

            scores,

            axis=1

        )



        context = tf.reduce_sum(

            inputs * attention_weights,

            axis=1

        )



        return context





# ============================================================

# LOAD MODEL AND SCALERS

# ============================================================



@st.cache_resource

def load_model_and_scalers():



    model = tf.keras.models.load_model(

        "attention_mqr_model.keras",

        custom_objects={

            "TemporalAttention": TemporalAttention

        },

        compile=False

    )



    scaler_X = joblib.load(

        "scaler_X.joblib"

    )



    scaler_y = joblib.load(

        "scaler_y.joblib"

    )



    return model, scaler_X, scaler_y





model, scaler_X, scaler_y = load_model_and_scalers()





# ============================================================

# TITLE

# ============================================================



st.title(

    "🌧️ Rainfall Prediction System"

)



st.write(

    "Attention-based CNN-LSTM with "

    "Multi-Quantile Regression"

)



st.success(

    "Model loaded successfully!"

)





# ============================================================

# MODEL INFORMATION

# ============================================================



st.subheader("📊 Model Information")



col1, col2 = st.columns(2)



with col1:



    st.write(

        "**Input shape:**",

        model.input_shape

    )



with col2:



    st.write(

        "**Output shape:**",

        model.output_shape

    )



st.info(

    "The model predicts three rainfall quantiles: "

    "Q50, Q95 and Q99."

)





# ============================================================

# FEATURE ORDER

# ============================================================



FEATURE_COLS = [



    "avg_temp",

    "min_temp",

    "max_temp",

    "wind_speed",

    "air_pressure",

    "rainfall",

    "month_sin",

    "month_cos",

    "day_of_year_sin",

    "day_of_year_cos"



]





# ============================================================

# REQUIRED FEATURES

# ============================================================



st.subheader("🔢 Required Features")



for feature in FEATURE_COLS:



    st.write(

        "•",

        feature

    )





# ============================================================

# UPLOAD DATASET

# ============================================================



st.header(

    "📂 Upload Weather Dataset"

)



uploaded_file = st.file_uploader(

    "Upload your weather dataset",

    type=["csv", "xlsx"]

)





# ============================================================

# PROCESS DATASET

# ============================================================



if uploaded_file is not None:



    # --------------------------------------------------------

    # READ FILE

    # --------------------------------------------------------



    try:



        df = load_weather_dataset(

            uploaded_file

        )



        st.success(

            "Dataset uploaded successfully!"

        )



    except Exception as e:



        st.error(

            f"Error reading dataset: {e}"

        )



        st.stop()





    # --------------------------------------------------------

    # REQUIRED RAW COLUMNS

    # --------------------------------------------------------



    required_raw_columns = [



        "date_of_record",

        "station_name",

        "avg_temp",

        "min_temp",

        "max_temp",

        "wind_speed",

        "air_pressure",

        "rainfall"



    ]





    missing_columns = [



        column

        for column in required_raw_columns

        if column not in df.columns



    ]





    if missing_columns:



        st.error(

            "The following required columns are missing:"

        )



        st.write(

            missing_columns

        )



        st.stop()





    # --------------------------------------------------------

    # CONVERT DATE

    # --------------------------------------------------------



    df["date_of_record"] = pd.to_datetime(

        df["date_of_record"],

        errors="coerce"

    )





    # --------------------------------------------------------

    # REMOVE INVALID DATES

    # --------------------------------------------------------



    df = df.dropna(

        subset=["date_of_record"]

    )





    # --------------------------------------------------------

    # SORT DATA

    # --------------------------------------------------------



    df = df.sort_values(

        [

            "station_name",

            "date_of_record"

        ]

    )





    # --------------------------------------------------------

    # DATASET INFORMATION

    # --------------------------------------------------------



    st.subheader(

        "📋 Dataset Information"

    )





    col1, col2 = st.columns(2)





    with col1:



        st.metric(

            "Number of Records",

            f"{len(df):,}"

        )





    with col2:



        st.metric(

            "Number of Stations",

            df["station_name"].nunique()

        )





    # ========================================================

    # STATION SELECTION

    # ========================================================



    st.subheader(

        "📍 Select Weather Station"

    )





    stations = sorted(

        df["station_name"]

        .dropna()

        .unique()

    )





    selected_station = st.selectbox(

        "Choose a station",

        stations

    )





    # --------------------------------------------------------

    # FILTER STATION

    # --------------------------------------------------------



    station_df = df[

        df["station_name"] == selected_station

    ].copy()





    # --------------------------------------------------------

    # SORT BY DATE

    # --------------------------------------------------------



    station_df = station_df.sort_values(

        "date_of_record"

    )





    st.write(

        f"### Selected Station: {selected_station}"

    )





    st.write(

        f"Available records: "

        f"**{len(station_df):,}**"

    )





    # ========================================================

    # REMOVE DUPLICATE DATES

    # ========================================================



    duplicate_count = (

        station_df["date_of_record"]

        .duplicated()

        .sum()

    )





    if duplicate_count > 0:



        st.warning(

            f"{duplicate_count} duplicate date records "

            f"were found for {selected_station}. "

            f"Duplicate dates will be combined into one "

            f"daily record."

        )





        numeric_columns = [



            "avg_temp",

            "min_temp",

            "max_temp",

            "wind_speed",

            "air_pressure",

            "rainfall"



        ]





        station_df = (

            station_df

            .groupby(

                "date_of_record",

                as_index=False

            )[numeric_columns]

            .mean()

        )





    # ========================================================

    # CHECK NUMBER OF DISTINCT DAYS

    # ========================================================



    if len(station_df) < 30:



        st.error(

            "This station does not have at least "

            "30 distinct days of data."

        )



        st.stop()





    # ========================================================

    # SORT AGAIN

    # ========================================================



    station_df = station_df.sort_values(

        "date_of_record"

    )





    # ========================================================

    # CHECK FOR MISSING DAYS

    # ========================================================



    date_difference = (

        station_df["date_of_record"]

        .diff()

        .dt.days

        .dropna()

    )





    missing_day_gaps = (

        date_difference > 1

    ).sum()





    if missing_day_gaps > 0:



        st.warning(

            "There are gaps in this station's daily records. "

            "The app will use the 30 available records immediately "

            "before the selected date."

        )





    # ========================================================

    # SELECT TARGET DATE

    # ========================================================



    st.subheader(

        "📅 Select Prediction Date"

    )



    min_target_date = (

        station_df["date_of_record"].min()

        + pd.Timedelta(days=30)

    )



    max_target_date = (

        station_df["date_of_record"].max()

    )



    target_date = st.date_input(

        "Choose the day you want to predict",

        value=max_target_date.date(),

        min_value=min_target_date.date(),

        max_value=max_target_date.date()

    )



    target_date = pd.Timestamp(target_date)



    st.info(

        f"The model will use the **30 records immediately before "

        f"{target_date.strftime('%Y-%m-%d')}** and predict rainfall "

        f"for **{target_date.strftime('%Y-%m-%d')}**."

    )





    # ========================================================

    # GET PREVIOUS 30 RECORDS

    # ========================================================



    previous_data = station_df[

        station_df["date_of_record"] < target_date

    ].copy()



    if len(previous_data) < 30:



        st.error(

            "There are not enough records before the selected "

            "date to create a 30-step input sequence."

        )



        st.stop()



    # Take the 30 records immediately before the target date.

    prediction_df = previous_data.tail(30).copy()





    # ========================================================

    # DISPLAY PREVIOUS 30 DAYS

    # ========================================================



    st.subheader(

        "📅 Previous 30 Days Used for Prediction"

    )





    display_columns = [



        "date_of_record",

        "avg_temp",

        "min_temp",

        "max_temp",

        "wind_speed",

        "air_pressure",

        "rainfall"



    ]





    st.dataframe(

        prediction_df[display_columns],

        width="stretch"

    )





    first_date = prediction_df[

        "date_of_record"

    ].min()



    last_input_date = prediction_df[

        "date_of_record"

    ].max()





    st.info(

        f"Input window: "

        f"**{first_date.strftime('%Y-%m-%d')}** "

        f"to "

        f"**{last_input_date.strftime('%Y-%m-%d')}** "

        f" → Predicting **{target_date.strftime('%Y-%m-%d')}**."

    )





    # ========================================================

    # FEATURE ENGINEERING

    # ========================================================



    st.subheader(

        "⚙️ Feature Engineering"

    )





    # --------------------------------------------------------

    # MONTH CYCLICAL FEATURES

    # --------------------------------------------------------



    prediction_df["month_sin"] = np.sin(

        2 * np.pi *

        prediction_df["date_of_record"].dt.month / 12

    )





    prediction_df["month_cos"] = np.cos(

        2 * np.pi *

        prediction_df["date_of_record"].dt.month / 12

    )





    # --------------------------------------------------------

    # DAY OF YEAR CYCLICAL FEATURES

    # --------------------------------------------------------



    day_of_year = (

        prediction_df["date_of_record"]

        .dt.dayofyear

    )



    prediction_df["day_of_year_sin"] = np.sin(

        2 * np.pi *

        day_of_year / 365.25

    )



    prediction_df["day_of_year_cos"] = np.cos(

        2 * np.pi *

        day_of_year / 365.25

    )



    # ========================================================

    # CREATE 10-FEATURE MATRIX

    # ========================================================



    X_input = prediction_df[

        FEATURE_COLS

    ].copy()





    # --------------------------------------------------------

    # CHECK NUMERIC VALUES

    # --------------------------------------------------------



    X_input = X_input.apply(

        pd.to_numeric,

        errors="coerce"

    )





    if X_input.isnull().any().any():



        st.error(

            "Missing or invalid values were found "

            "in the prediction features."

        )



        st.dataframe(

            X_input[

                X_input.isnull().any(axis=1)

            ]

        )



        st.stop()





    # ========================================================

    # DISPLAY ENGINEERED FEATURES

    # ========================================================



    with st.expander(

        "View 30 × 10 Model Input Features"

    ):

        st.dataframe(

            X_input,

            width="stretch"

        )



        st.write("### Last 5 input rows")

        st.dataframe(

            X_input.tail(5),

            width="stretch"

        )



        st.write("### Input shape")

        st.write(X_input.shape)



        st.write("### Raw input values before scaling")

        st.dataframe(

            prediction_df[

                ["date_of_record"] + FEATURE_COLS

            ].tail(5),

            width="stretch"

        )





    # ========================================================

    # CONVERT TO NUMPY

    # ========================================================



    X_array = X_input.values.astype(

        np.float32

    )





    # ========================================================

    # CHECK SHAPE

    # ========================================================



    if X_array.shape != (30, 10):



        st.error(

            f"Incorrect input shape: "

            f"{X_array.shape}. "

            f"Expected (30, 10)."

        )



        st.stop()





    # ========================================================

    # SCALE INPUT

    # ========================================================



    try:



        X_scaled = scaler_X.transform(

            X_array

        )



    except Exception as e:



        st.error(

            f"Error while scaling input data: {e}"

        )



        st.stop()





    # ========================================================

    # CREATE MODEL INPUT

    # ========================================================



    X_model = np.expand_dims(

        X_scaled,

        axis=0

    )





    # Shape should be:

    # (1, 30, 10)





    # ========================================================

    # PREDICT

    # ========================================================



    st.subheader(

        "🔮 Rainfall Prediction"

    )





    if st.button(

        "🌧️ Predict Selected Date",

        type="primary"

    ):



        with st.spinner(

            "Running rainfall prediction..."

        ):



            try:



                prediction_scaled = model.predict(

                    X_model,

                    verbose=0

                )



            except Exception as e:



                st.error(

                    f"Prediction failed: {e}"

                )



                st.stop()





        # ====================================================

        # CHECK MODEL OUTPUT

        # ====================================================



        prediction_scaled = np.asarray(

            prediction_scaled

        )





        if prediction_scaled.ndim == 1:



            prediction_scaled = (

                prediction_scaled.reshape(

                    1, -1

                )

            )





        if prediction_scaled.shape[1] != 3:



            st.error(

                f"Unexpected model output shape: "

                f"{prediction_scaled.shape}. "

                f"Expected (1, 3)."

            )



            st.stop()





        # ====================================================

        # INVERSE TRANSFORM EACH QUANTILE

        # ====================================================



        q50 = scaler_y.inverse_transform(

            prediction_scaled[:, 0].reshape(

                -1, 1

            )

        )[0, 0]





        q95 = scaler_y.inverse_transform(

            prediction_scaled[:, 1].reshape(

                -1, 1

            )

        )[0, 0]





        q99 = scaler_y.inverse_transform(

            prediction_scaled[:, 2].reshape(

                -1, 1

            )

        )[0, 0]





        # ----------------------------------------------------

        # RAINFALL CANNOT BE NEGATIVE

        # ----------------------------------------------------



        q50 = max(

            0.0,

            float(q50)

        )



        q95 = max(

            0.0,

            float(q95)

        )



        q99 = max(

            0.0,

            float(q99)

        )





        # ====================================================

        # PRESERVE MODEL QUANTILE OUTPUTS

        # ====================================================



        # Keep the original Q50, Q95 and Q99 predictions.

        # Do not sort them, because sorting would change the

        # identity of the predicted quantiles.







        # ====================================================

        # PREDICTION DATE

        # ====================================================



        prediction_date = target_date





        # ====================================================

        # DISPLAY RESULTS

        # ====================================================



        st.success(

            "Prediction completed successfully!"

        )





        st.write(

            f"### 📅 Predicted Date: "

            f"{prediction_date.strftime('%Y-%m-%d')}"

        )





        col1, col2, col3 = st.columns(3)





        with col1:



            st.metric(

                "Q50 — Median",

                f"{q50:.2f} mm"

            )





        with col2:



            st.metric(

                "Q95 — High Quantile",

                f"{q95:.2f} mm"

            )





        with col3:



            st.metric(

                "Q99 — Very High Quantile",

                f"{q99:.2f} mm"

            )





        # ====================================================

        # INTERPRETATION

        # ====================================================



        st.subheader(

            "📌 Prediction Interpretation"

        )





        st.write(

            f"**Q50:** {q50:.2f} mm — "

            "central/median rainfall estimate."

        )





        st.write(

            f"**Q95:** {q95:.2f} mm — "

            "higher rainfall estimate representing "

            "a high-quantile scenario."

        )





        st.write(

            f"**Q99:** {q99:.2f} mm — "

            "very-high upper-quantile rainfall estimate."

        )





        # ====================================================

        # SUMMARY TABLE

        # ====================================================



        result_df = pd.DataFrame({



            "Quantile": [

                "Q50",

                "Q95",

                "Q99"

            ],



            "Predicted Rainfall (mm)": [

                q50,

                q95,

                q99

            ]



        })





        st.subheader(

            "📊 Prediction Summary"

        )





        st.dataframe(

            result_df,

            width="stretch",

            hide_index=True

        )





        # ====================================================

        # ACTUAL RAINFALL IF AVAILABLE
        # ====================================================

        actual_target = station_df[
            station_df["date_of_record"] == target_date
        ]

        if not actual_target.empty:

            actual_rainfall = float(
                actual_target["rainfall"].iloc[0]
            )

            st.subheader("🎯 Actual Rainfall on Selected Date")

            st.metric(
                "Observed Rainfall",
                f"{actual_rainfall:.2f} mm"
            )

            # Calculate absolute error for each quantile
            q50_error = abs(actual_rainfall - q50)
            q95_error = abs(actual_rainfall - q95)
            q99_error = abs(actual_rainfall - q99)

            quantile_predictions = {
                "Q50": q50,
                "Q95": q95,
                "Q99": q99
            }

            quantile_errors = {
                "Q50": q50_error,
                "Q95": q95_error,
                "Q99": q99_error
            }

            # Find which quantile is closest to the actual value
            closest_quantile = min(
                quantile_errors,
                key=quantile_errors.get
            )

            closest_prediction = quantile_predictions[closest_quantile]
            closest_error = quantile_errors[closest_quantile]

            # ----------------------------------------------------
            # QUANTILE COMPARISON
            # ----------------------------------------------------

            st.subheader("📊 Quantile Comparison with Actual Rainfall")

            comparison_df = pd.DataFrame({
                "Quantile": ["Q50", "Q95", "Q99"],
                "Predicted Rainfall (mm)": [q50, q95, q99],
                "Absolute Error (mm)": [
                    q50_error, q95_error, q99_error
                ]
            })

            st.dataframe(
                comparison_df,
                width="stretch",
                hide_index=True
            )

            # ----------------------------------------------------
            # CLOSEST QUANTILE
            # ----------------------------------------------------

            st.subheader("🏆 Closest Quantile Prediction")

            st.success(
                f"**{closest_quantile}** is closest to the observed rainfall.\n\n"
                f"Actual Rainfall: **{actual_rainfall:.2f} mm**  \n"
                f"{closest_quantile} Prediction: **{closest_prediction:.2f} mm**  \n"
                f"Absolute Error: **{closest_error:.2f} mm**"
            )

            # ----------------------------------------------------
            # INDIVIDUAL ERROR INTERPRETATION
            # ----------------------------------------------------

            st.subheader("🔍 Prediction Accuracy Interpretation")

            st.write(
                f"**Q50:** {q50:.2f} mm — "
                f"Absolute error: {q50_error:.2f} mm"
            )

            st.write(
                f"**Q95:** {q95:.2f} mm — "
                f"Absolute error: {q95_error:.2f} mm"
            )

            st.write(
                f"**Q99:** {q99:.2f} mm — "
                f"Absolute error: {q99_error:.2f} mm"
            )

            st.info(
                f"For this particular observed case, **{closest_quantile}** "
                f"provides the closest prediction with an absolute error "
                f"of **{closest_error:.2f} mm**."
            )

        else:

            st.info(
                "No observed rainfall value is available for "
                "the selected date. Only the model prediction "
                "is shown."
            )


        # SIMPLE BAR CHART

        # ====================================================



        st.subheader(

            "📈 Predicted Rainfall Quantiles"

        )





        chart_df = result_df.set_index(

            "Quantile"

        )





        st.bar_chart(

            chart_df[

                "Predicted Rainfall (mm)"

            ]

        )