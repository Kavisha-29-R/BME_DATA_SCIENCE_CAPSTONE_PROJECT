# brain_cognitive_state_app.py
# Streamlit app: cognitive state classification with interactive plots
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# --------------------------
# Load Data
# --------------------------
@st.cache_data
def load_data_full():
    df = pd.read_csv("../Data/processed_features_final/all_participants_features.csv")
    df['window_center'] = pd.to_datetime(df['window_center'], unit='s', utc=True)
    return df

@st.cache_data
def load_data_clusters():
    df = pd.read_csv("../Data/processed_features_final/all_participants_clusters.csv")
    df['window_center'] = pd.to_datetime(df['window_center'], unit='s', utc=True)
    return df

full_df = load_data_full()
df_cluster = load_data_clusters()

# --------------------------
# Sidebar - Participant Selection
# --------------------------

# --------------------------
# Sidebar - Experiment and Participant Selection
# --------------------------

# Identify participants per experiment
exp1_subjects = [p for p in full_df['participant'].unique() if str(p).startswith("A")]
exp2_subjects = [p for p in full_df['participant'].unique() if str(p).startswith("B")]

st.sidebar.title("Participant Selection")

# Select experiment
experiment = st.sidebar.radio("Choose Experiment", ["Experiment 1 (A-group)", "Experiment 2 (B-group)"])

# Update participant list based on experiment
if experiment.startswith("Experiment 1"):
    participants = exp1_subjects
else:
    participants = exp2_subjects

# Select participant
selected_participant = st.sidebar.selectbox("Choose Participant", participants)

# Filter data for selected participant (make a copy to avoid warnings)
df_participant = full_df[full_df['participant'] == selected_participant].copy()


# Filter data for selected participant and make a copy to avoid SettingWithCopyWarning
df_participant = full_df[full_df['participant'] == selected_participant].copy()

# --------------------------
# Page Navigation
# --------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Participant Overview", 
                                  "Cognitive State Classification", 
                                  "Summary & Insights", 
                                  "Experiment Group Comparison"])

# --------------------------
# Participant Overview 
# --------------------------
if page == "Participant Overview":
    st.title(f"Participant Overview: {selected_participant}")

    st.subheader("Summary Statistics")
    st.dataframe(df_participant.describe())

    st.subheader("Raw Time-Series Plots (Left vs Right)")

    metrics_pairs = [
        ('mean_RT', None),
        ('prop_correct', None),
        ('eda_mean', ('left_eda_mean','right_eda_mean')),
        ('bvp_hr_mean', ('left_bvp_hr_mean','right_bvp_hr_mean')),
        ('acc_mag_mean', ('left_acc_mag_mean','right_acc_mag_mean')),
        ('temp_mean', ('left_temp_mean','right_temp_mean')),
        ('ibi_ibi_rmssd', ('left_ibi_ibi_rmssd','right_ibi_ibi_rmssd')),
        ('ibi_ibi_sdnn', ('left_ibi_ibi_sdnn','right_ibi_ibi_sdnn'))
    ]

    for metric, pair in metrics_pairs:
        if pair:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_participant['window_center'],
                y=df_participant[pair[0]],
                mode='lines+markers',
                name=pair[0]
            ))
            fig.add_trace(go.Scatter(
                x=df_participant['window_center'],
                y=df_participant[pair[1]],
                mode='lines+markers',
                name=pair[1]
            ))
            fig.update_layout(
                title=f"{metric.replace('_',' ').title()} (Left vs Right)",
                xaxis_title="Time (UTC)",
                yaxis_title=metric.replace('_',' ').title(),
                hovermode='x unified'
            )
        else:
            fig = px.line(
                df_participant, x='window_center', y=metric,
                title=metric.replace('_',' ').title(),
                labels={"window_center":"Time (UTC)", metric: metric.replace('_',' ').title()}
            )
            fig.update_traces(mode='lines+markers')
        st.plotly_chart(fig, width='stretch')


# --------------------------
# Cognitive State Classification
# --------------------------
elif page == "Cognitive State Classification":
    st.title(f"Cognitive State Classification: {selected_participant}")

    # --- Performance ---
    st.subheader("High vs Low Performance by Session Type")
    median_RT = df_participant['mean_RT'].median()
    median_accuracy = df_participant['prop_correct'].median()

    def classify_performance(row):
        if row['prop_correct'] > median_accuracy and row['mean_RT'] < median_RT:
            return 'High'
        else:
            return 'Low'

    df_participant['performance_state'] = df_participant.apply(classify_performance, axis=1)

    fig = px.scatter(
        df_participant,
        x='window_center',
        y='mean_RT',
        color='performance_state',
        symbol='session_type',
        hover_data=['session_type', 'prop_correct', 'mean_RT'],
        color_discrete_map={"High":"green","Low":"red"},
        title="Performance Timeline (by Session Type)",
        labels={"window_center":"Time (UTC)", "mean_RT":"Mean RT"}
    )
    st.plotly_chart(fig, width='stretch')

    st.write("Percentage of time in each performance state:")
    st.dataframe(df_participant['performance_state'].value_counts(normalize=True)*100)

    # --- Stress ---
    st.subheader("Stressed vs Calm by Session Type")
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

    fig = px.scatter(
        df_participant,
        x='window_center',
        y='left_eda_mean',
        color='stress_state',
        symbol='session_type',
        hover_data=['session_type', 'left_eda_mean', 'right_eda_mean', 'left_bvp_hr_mean', 'right_bvp_hr_mean'],
        color_discrete_map={"Calm":"blue","Stressed":"orange"},
        title="Stress Timeline (EDA Left) by Session Type",
        labels={"window_center":"Time (UTC)", "left_eda_mean":"EDA Left Mean"}
    )
    st.plotly_chart(fig, width='stretch')


    st.write("Percentage of time in each stress state:")
    st.dataframe(df_participant['stress_state'].value_counts(normalize=True)*100)

# --------------------------
# Summary & Insights
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
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale='RdBu_r')
    st.plotly_chart(fig, width='stretch')


    st.subheader("Download Summary CSV")
    csv = df_participant.to_csv(index=False)
    st.download_button(
        label="Download Participant Data",
        data=csv,
        file_name=f"{selected_participant}_summary.csv",
        mime="text/csv"
    )

# --------------------------
# Experiment Group Comparison
# --------------------------
elif page == "Experiment Group Comparison":
    st.title("Experiment Group Comparison")

    # Identify participants per experiment
    exp1_participants = [p for p in full_df['participant'].unique() if str(p).startswith("A")]
    exp2_participants = [p for p in full_df['participant'].unique() if str(p).startswith("B")]

    exp1_df = full_df[full_df['participant'].isin(exp1_participants)]
    exp2_df = full_df[full_df['participant'].isin(exp2_participants)]

    # Compute mean accuracy per session type
    exp1_grouped = exp1_df.groupby(["participant","session_type"])["prop_correct"].mean().unstack()
    exp2_grouped = exp2_df.groupby(["participant","session_type"])["prop_correct"].mean().unstack()

    st.subheader("Experiment 1 (A-group)")
    fig = px.bar(exp1_grouped, barmode="group", title="Behavioral Accuracy by Session Type")
    st.plotly_chart(fig, width='stretch')


    st.subheader("Experiment 2 (B-group)")
    fig = px.bar(exp2_grouped, barmode="group", title="Behavioral Accuracy by Session Type")
    st.plotly_chart(fig, width='stretch')
