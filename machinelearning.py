from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("Waste Collection Capacity Prediction Dashboard")

uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=["date"])
else:
    df = pd.read_csv("sustainable_waste_management_dataset_2024.csv", parse_dates=["date"])

st.sidebar.header("Controls")

show_data = st.sidebar.checkbox("Show dataset preview", False)
test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.2, 0.05)

area_filter = st.sidebar.multiselect(
    "Filter by area",
    options=df["area"].unique(),
    default=df["area"].unique()
)

df = df[df["area"].isin(area_filter)]

features = [
    "area", "population", "waste_kg", "recyclable_kg", "organic_kg",
    "overflow", "is_weekend", "is_holiday", "recycling_campaign",
    "temp_c", "rain_mm"
]

X = pd.get_dummies(df[features], columns=["area"], drop_first=True)
y = df["collection_capacity_kg"]

data = pd.concat([X, y], axis=1).dropna()
X = data.drop(columns=["collection_capacity_kg"])
y = data["collection_capacity_kg"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

if show_data:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(50))

col1, col2 = st.columns(2)

with col1:
    show_mse = st.checkbox("Show MSE", value=True)
with col2:
    show_r2 = st.checkbox("Show R²", value=True)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=y_test,
    y=y_pred,
    mode="markers",
    opacity=0.7,
    name="Predictions"
))

min_v, max_v = y.min(), y.max()
fig.add_trace(go.Scatter(
    x=[min_v, max_v],
    y=[min_v, max_v],
    mode="lines",
    line=dict(dash="dash"),
    name="Perfect Prediction"
))

fig.update_layout(
    title="Predicted vs Actual Collection Capacity",
    xaxis_title="Actual",
    yaxis_title="Predicted"
)

st.plotly_chart(fig, use_container_width=True)

metrics = []
if show_mse:
    metrics.append(f"MSE = {mse:.2f}")
if show_r2:
    metrics.append(f"R² = {r2:.3f}")

if metrics:
    st.markdown(
        f"<div style='text-align:center; margin-top:10px; font-size:14px;'>"
        f"{' | '.join(metrics)}"
        f"</div>",
        unsafe_allow_html=True
    )

st.divider()

if st.checkbox("Show residuals plot"):
    residuals = y_test - y_pred
    fig_res = go.Figure()
    fig_res.add_trace(go.Scatter(
        x=y_pred,
        y=residuals,
        mode="markers",
        name="Residuals"
    ))
    fig_res.add_hline(y=0)
    fig_res.update_layout(
        title="Residuals vs Predicted",
        xaxis_title="Predicted",
        yaxis_title="Residuals"
    )
    st.plotly_chart(fig_res, use_container_width=True)

if st.checkbox("Download predictions"):
    result_df = pd.DataFrame({
        "Actual": y_test,
        "Predicted": y_pred
    })
    st.download_button(
        "Download CSV",
        result_df.to_csv(index=False),
        file_name="predictions.csv",
        mime="text/csv"
    )
