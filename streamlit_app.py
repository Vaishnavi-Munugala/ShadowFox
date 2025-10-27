import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
st.set_page_config(page_title="Delhi AQI Research & Analysis", layout="wide")
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            min-width: 500px;   /* wider sidebar */
            max-width: 520px;
        }
    </style>
""", unsafe_allow_html=True)
st.title("In-Depth Analysis of Delhi’s Air Quality Index (AQI)")
st.markdown("""
Delhi faces some of the **highest air pollution levels** in the world, driven by vehicle emissions, 
industrial activity, crop residue burning, and meteorological conditions.  
This dashboard conducts a **comprehensive research analysis** of AQI patterns in Delhi to uncover:
- Seasonal and pollutant-specific trends  
- Interrelationships between pollutants  
- Statistical and visual insights for informed policy and health interventions  
""")
st.markdown("---")
uploaded_file = st.file_uploader("Upload your Delhi AQI dataset (CSV)", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding="utf-8")
    df = df.loc[:, ~df.columns.duplicated()]  
    st.success(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")
    date_col = None
    for c in df.columns:
        if "date" in c.lower() or "time" in c.lower():
            date_col = c
            break
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        df = df.sort_values(by=date_col)
        df = df.set_index(date_col)
    else:
        st.warning("No date column detected — time-series plots may not function correctly.")
    st.sidebar.header("Filter Options")
    pollutant_cols = [c for c in df.columns if any(x in c.lower() for x in ["pm2", "pm10", "no2", "so2", "co", "o3"])]
    selected_pollutants = st.sidebar.multiselect(
        "Select pollutants to analyze:",
        pollutant_cols,
        default=pollutant_cols[:3]
    )
    if not selected_pollutants:
        st.warning("Please select at least one pollutant to continue.")
        st.stop()
    start_date = st.sidebar.date_input("Start Date", df.index.min().date())
    end_date = st.sidebar.date_input("End Date", df.index.max().date())
    df = df.loc[(df.index.date >= start_date) & (df.index.date <= end_date)].copy()
    if df.empty:
        st.error("No data available for the selected date range.")
        st.stop()
    st.markdown("---")
    df[selected_pollutants] = df[selected_pollutants].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=selected_pollutants, how="all")
    df_daily = df.resample("D").mean()
    if len(df_daily) < 7:
        st.warning("The selected range has less than 7 days — rolling and monthly charts may appear flat.")
    st.header("Statistical Overview of Air Quality")
    col1, col2, col3 = st.columns(3)
    avg_aqi = round(df[selected_pollutants].mean().mean(), 2)
    max_aqi = round(df[selected_pollutants].max().max(), 2)
    min_aqi = round(df[selected_pollutants].min().min(), 2)
    col1.metric("Average AQI (Overall)", avg_aqi)
    col2.metric("Maximum AQI Observed", max_aqi)
    col3.metric("Minimum AQI Observed", min_aqi)
    st.write("### Descriptive Statistics by Pollutant")
    st.dataframe(df[selected_pollutants].describe().T.style.background_gradient(cmap="YlOrRd"))
    st.markdown("---")
    st.header("Time-Series Analysis (All Selected Pollutants)")
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    for col in selected_pollutants:
        ax1.plot(df_daily.index, df_daily[col], linewidth=1.5, label=col)
    ax1.set_title("Time Series of Selected Pollutants (Daily Mean)")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Concentration")
    ax1.legend()
    st.pyplot(fig1)
    st.info("💡 **Insight:** Daily mean values help identify short-term pollution spikes and patterns.")
    if len(df_daily) >= 7:
        rolling_avg = df_daily.rolling(window=7).mean()
    else:
        rolling_avg = df_daily
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    for col in selected_pollutants:
        ax2.plot(rolling_avg.index, rolling_avg[col], linewidth=2, label=col)
    ax2.set_title("7-Day Rolling Average of Selected Pollutants")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Concentration")
    ax2.legend()
    st.pyplot(fig2)
    st.info("💡 **Insight:** The 7-day rolling average smooths daily fluctuations and highlights weekly trends.")
    st.markdown("---")
    st.header("Monthly Analysis")
    df_daily["Month"] = df_daily.index.month.astype(int)
    df_daily["Year"] = df_daily.index.year
    monthly_avg = df_daily.groupby("Month")[selected_pollutants].mean()
    if monthly_avg.empty:
        st.error("No valid monthly data to display.")
    else:
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        if monthly_avg.shape[0] <= 2:
            monthly_avg.plot(kind="bar", ax=ax3)
            ax3.set_title("Average Pollutant Levels by Month (Bar View)")
        else:
            monthly_avg.plot(ax=ax3)
            ax3.set_title("Average Pollutant Levels by Month (1=Jan)")
        ax3.set_xlabel("Month")
        ax3.set_ylabel("Average Concentration")
        ax3.legend(title="Pollutant")
        st.pyplot(fig3)
        st.write("### Monthly Distribution by Pollutant")
        melted = df_daily.reset_index().melt(
            id_vars=["Month"], 
            value_vars=selected_pollutants, 
            var_name="Pollutant", 
            value_name="Value"
        )
        fig4, ax4 = plt.subplots(figsize=(8, 4))
        sns.boxplot(x="Month", y="Value", hue="Pollutant", data=melted, ax=ax4)
        ax4.set_title("Monthly Distribution of Selected Pollutants (Boxplot)")
        plt.xticks(rotation=0)
        st.pyplot(fig4)
    st.markdown("---")
    st.header("Correlation Matrix of Pollutants")
    corr = df_daily[selected_pollutants].corr()
    fig_heat, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(corr, annot=True, cmap="viridis", fmt=".2f")
    plt.title("Correlation Matrix of Pollutants")
    st.pyplot(fig_heat)
    st.header("Monthly Mean Heatmaps")
    monthly_means_all = df_daily.groupby(["Year", "Month"])[selected_pollutants].mean()
    for col in selected_pollutants:
        st.subheader(f"Monthly Mean Heatmap – {col}")
        fig, ax = plt.subplots(figsize=(8, 4))
        pivot_data = monthly_means_all[col].unstack()
        sns.heatmap(pivot_data, cmap="viridis", cbar_kws={"label": f"{col} mean"})
        ax.set_title(f"Monthly Means of {col} by Year")
        st.pyplot(fig)
    st.markdown("---")
    st.header("Relationship Between Pollutants")
    col_x = st.selectbox("Select X-axis pollutant:", selected_pollutants)
    col_y = st.selectbox("Select Y-axis pollutant:", selected_pollutants, index=1 if len(selected_pollutants) > 1 else 0)
    fig_scatter = px.scatter(
        df_daily, x=col_x, y=col_y,
        trendline="ols",
        color_discrete_sequence=["#FF6347"],
        opacity=0.7,
        title=f"Scatter and Regression between {col_x.upper()} and {col_y.upper()}"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.markdown("---")
    st.header("Contribution of Pollutants to Overall AQI")

    mean_values = df_daily[selected_pollutants].mean()
    fig_pie = px.pie(
        values=mean_values.values,
        names=mean_values.index,
        title="Average Pollutant Contribution to Delhi AQI",
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    st.success("Analysis complete. All major AQI trends, relationships, and temporal patterns visualized.")

else:
    st.info("Upload your Delhi AQI dataset to begin your research analysis.")
