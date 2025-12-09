# streamlit app overviewing cognitive state classification
# to run program: streamlit run brain_cognitive_state_app.py

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# --------------------------
# Load Data
# --------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("../Data/processed_features_final/all_participants_features.csv")
    return df

df = load_data()

# --------------------------
# Sidebar - Select Participant
# --------------------------
st.sidebar.title("Participant Selection")
participants = df['participant'].unique()
selected_participant = st.sidebar.selectbox("Choose Participant", participants)

# Filter data for selected participant
df_participant = df[df['participant'] == selected_participant]

# --------------------------
# Page Navigation
# --------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Participant Overview", "Cognitive State Classification", "Summary & Insights"])

# Define Seaborn palette
palette = sns.color_palette("husl", 8)

# --------------------------
# PAGE 1: Participant Overview
# --------------------------
if page == "Participant Overview":
    st.title(f"Participant Overview: {selected_participant}")

    st.subheader("Summary Statistics")
    st.dataframe(df_participant.describe())

    st.subheader("Raw Time-Series Plots")
    metrics = [
        'mean_RT', 'prop_correct',
        'left_eda_mean', 'right_eda_mean',
        'left_bvp_hr_mean', 'right_bvp_hr_mean',
        'left_ibi_ibi_rmssd', 'right_ibi_ibi_rmssd',
        'left_acc_mag_mean', 'right_acc_mag_mean'
    ]

    for idx, metric in enumerate(metrics):
        if metric in df_participant.columns:
            fig, ax = plt.subplots(figsize=(10,4))
            sns.lineplot(
                x='window_center', y=metric,
                data=df_participant, color=palette[idx % len(palette)]
            )
            ax.set_title(metric.replace('_', ' ').title())
            ax.set_xlabel("Time (window center)")
            ax.set_ylabel(metric.replace('_', ' ').title())
            st.pyplot(fig)

# --------------------------
# PAGE 2: Cognitive State Classification
# --------------------------
elif page == "Cognitive State Classification":
    st.title(f"Cognitive State Classification: {selected_participant}")

    # --------------------------
    # High vs Low Performance
    # --------------------------
    st.subheader("High vs Low Performance")
    median_RT = df_participant['mean_RT'].median()
    median_accuracy = df_participant['prop_correct'].median()

    def classify_performance(row):
        if row['prop_correct'] > median_accuracy and row['mean_RT'] < median_RT:
            return 'High'
        else:
            return 'Low'

    df_participant['performance_state'] = df_participant.apply(classify_performance, axis=1)

    fig, ax = plt.subplots(figsize=(10,4))
    sns.scatterplot(
        x='window_center', y='mean_RT', hue='performance_state',
        palette=["#2ca02c", "#d62728"],  # green=High, red=Low
        data=df_participant, s=50
    )
    ax.set_title("Performance Timeline")
    ax.set_xlabel("Time (window center)")
    ax.set_ylabel("Mean RT")
    st.pyplot(fig)

    # Show % time in each state
    perf_counts = df_participant['performance_state'].value_counts(normalize=True) * 100
    st.write("Percentage of time in each performance state:")
    st.dataframe(perf_counts)

    # --------------------------
    # Stressed vs Calm
    # --------------------------
    st.subheader("Stressed vs Calm")
    median_eda = (df_participant['left_eda_mean'].median() + df_participant['right_eda_mean'].median()) / 2
    median_hr = (df_participant['left_bvp_hr_mean'].median() + df_participant['right_bvp_hr_mean'].median()) / 2
    median_rmssd = (df_participant['left_ibi_ibi_rmssd'].median() + df_participant['right_ibi_ibi_rmssd'].median()) / 2

    def classify_stress(row):
        eda = (row['left_eda_mean'] + row['right_eda_mean']) / 2
        hr = (row['left_bvp_hr_mean'] + row['right_bvp_hr_mean']) / 2
        rmssd = (row['left_ibi_ibi_rmssd'] + row['right_ibi_ibi_rmssd']) / 2

        if eda > median_eda or hr > median_hr or rmssd < median_rmssd:
            return 'Stressed'
        else:
            return 'Calm'

    df_participant['stress_state'] = df_participant.apply(classify_stress, axis=1)

    fig, ax = plt.subplots(figsize=(10,4))
    sns.scatterplot(
        x='window_center', y='left_eda_mean', hue='stress_state',
        palette=["#1f77b4", "#ff7f0e"],  # blue=Calm, orange=Stressed
        data=df_participant, s=50
    )
    ax.set_title("Stress Timeline (EDA)")
    ax.set_xlabel("Time (window center)")
    ax.set_ylabel("EDA Mean")
    st.pyplot(fig)

    # Show % time stressed
    stress_counts = df_participant['stress_state'].value_counts(normalize=True) * 100
    st.write("Percentage of time in each stress state:")
    st.dataframe(stress_counts)

# --------------------------
# PAGE 3: Summary & Insights
# --------------------------
elif page == "Summary & Insights":
    st.title(f"Summary & Insights: {selected_participant}")

    st.subheader("Correlation Heatmap")
    metrics = [
        'mean_RT', 'prop_correct',
        'left_eda_mean', 'right_eda_mean',
        'left_bvp_hr_mean', 'right_bvp_hr_mean',
        'left_ibi_ibi_rmssd', 'right_ibi_ibi_rmssd'
    ]
    corr = df_participant[metrics].corr()
    fig, ax = plt.subplots(figsize=(8,6))
    palette = sns.color_palette("husl", 8)
    cmap = ListedColormap(palette)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, cbar=True, ax=ax)
    st.pyplot(fig)

    st.subheader("Scatter Plots")
    fig, ax = plt.subplots(figsize=(10,4))
    sns.scatterplot(
        x='left_ibi_ibi_rmssd', y='prop_correct', hue='left_eda_mean',
        palette=palette, data=df_participant, s=50
    )
    ax.set_title("HRV vs Accuracy (colored by EDA)")
    ax.set_xlabel("Left RMSSD")
    ax.set_ylabel("Prop Correct")
    st.pyplot(fig)

    st.subheader("Download Summary CSV")
    csv = df_participant.to_csv(index=False)
    st.download_button(
        label="Download Participant Data",
        data=csv,
        file_name=f"{selected_participant}_summary.csv",
        mime="text/csv"
    )
