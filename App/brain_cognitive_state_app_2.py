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
    # Convert Unix UTC timestamps to datetime
    df['window_center'] = pd.to_datetime(df['window_center'], unit='s', utc=True)
    return df

def load_data_clusters():
    df = pd.read_csv("../Data/processed_features_final/all_participants_clusters.csv")
    # Convert Unix UTC timestamps to datetime
    df['window_center'] = pd.to_datetime(df['window_center'], unit='s', utc=True)
    return df

full_df = load_data_full()
df_cluster = load_data_clusters()

# --- df_cluster is the processed data with features, clusters, and performance metrics ---
# --- full_df contains all participants processed data ---

# Example feature pairs for plotting left vs right
metrics_pairs = [
    ('left_eda_mean','right_eda_mean'),
    ('left_bvp_hr_mean','right_bvp_hr_mean'),
    ('left_acc_mag_mean','right_acc_mag_mean'),
    None  # for single metrics
]

# Sidebar for page selection
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Home",
    "Subject Overview",
    "Cognitive State Classification",
    "Experiment Group Comparison"
])

# --------------------------
# Page 1: Home
# --------------------------
if page == "Home":
    st.title("Physiological Data Dashboard")
    st.markdown("""
    **Purpose:** View physiological features of subjects performing different tasks to study brain cognitive states.

    **Instructions:**
    - Use the sidebar to navigate between pages.
    - Filter by participant on Subject Overview and Cognitive State Classification pages.
    - Explore clusters, performance, and stress states.
    - Download subject data as CSV if needed.
    """)

# --------------------------
# Page 2: Subject Overview
# --------------------------
elif page == "Subject Overview":
    st.title("Subject Overview")
    
    participants = df_cluster['participant'].unique()
    selected_participant = st.selectbox("Select Participant", participants)
    df_participant = df_cluster[df_cluster['participant'] == selected_participant]
    
    st.subheader("Summary Statistics")
    st.dataframe(df_participant.describe().T[['mean','std']])
    
    st.subheader("Physiological Metrics Over Time")
    for metric, pair in zip(df_participant.columns, metrics_pairs):
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
        st.plotly_chart(fig, use_container_width=True)
    
    # Option to download participant data
    csv = df_participant.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Participant Data as CSV",
        data=csv,
        file_name=f"{selected_participant}_data.csv",
        mime='text/csv'
    )

# --------------------------
# Page 3: Cognitive State Classification
# --------------------------
elif page == "Cognitive State Classification":
    st.title(f"Cognitive State Classification: {selected_participant}")
    
    # --- Performance State ---
    st.subheader("High vs Low Performance by Session Type")
    median_RT = df_participant['mean_RT'].median()
    median_accuracy = df_participant['prop_correct'].median()
    
    def classify_performance(row):
        return 'High' if row['prop_correct'] > median_accuracy and row['mean_RT'] < median_RT else 'Low'
    
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
    st.plotly_chart(fig, use_container_width=True)
    st.write("Percentage of time in each performance state:")
    st.dataframe(df_participant['performance_state'].value_counts(normalize=True)*100)
    
    # --- Stress State ---
    st.subheader("Stressed vs Calm by Session Type")
    median_eda = (df_participant['left_eda_mean'].median() + df_participant['right_eda_mean'].median()) / 2
    median_hr = (df_participant['left_bvp_hr_mean'].median() + df_participant['right_bvp_hr_mean'].median()) / 2
    median_rmssd = (df_participant['left_ibi_ibi_rmssd'].median() + df_participant['right_ibi_ibi_rmssd'].median()) / 2
    
    def classify_stress(row):
        eda = (row['left_eda_mean'] + row['right_eda_mean']) / 2
        hr = (row['left_bvp_hr_mean'] + row['right_bvp_hr_mean']) / 2
        rmssd = (row['left_ibi_ibi_rmssd'] + row['right_ibi_ibi_rmssd']) / 2
        return 'Stressed' if eda > median_eda or hr > median_hr or rmssd < median_rmssd else 'Calm'
    
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
    st.plotly_chart(fig, use_container_width=True)
    st.write("Percentage of time in each stress state:")
    st.dataframe(df_participant['stress_state'].value_counts(normalize=True)*100)
    
    # --- Cluster Visualization ---
    st.subheader("Physiological Cluster States Over Time")
    fig = px.scatter(
        df_participant,
        x='window_center',
        y='physio_cluster',
        color='physio_state',
        symbol='session_type',
        hover_data=['session_type', 'physio_cluster', 'physio_state'],
        title="Physiological Cluster Timeline"
    )
    st.plotly_chart(fig, use_container_width=True)

# --------------------------
# Page 4: Experiment Group Comparison
# --------------------------
elif page == "Experiment Group Comparison":
    st.title("Experiment Group Comparison")
    
    exp1_participants = [p for p in full_df['participant'].unique() if str(p).startswith("A")]
    exp2_participants = [p for p in full_df['participant'].unique() if str(p).startswith("B")]
    
    exp1_df = full_df[full_df['participant'].isin(exp1_participants)]
    exp2_df = full_df[full_df['participant'].isin(exp2_participants)]
    
    # Behavioral Accuracy
    exp1_grouped = exp1_df.groupby(["participant","session_type"])["prop_correct"].mean().unstack()
    exp2_grouped = exp2_df.groupby(["participant","session_type"])["prop_correct"].mean().unstack()
    
    st.subheader("Experiment 1 (A-group)")
    fig = px.bar(exp1_grouped, barmode="group", title="Behavioral Accuracy by Session Type")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Experiment 2 (B-group)")
    fig = px.bar(exp2_grouped, barmode="group", title="Behavioral Accuracy by Session Type")
    st.plotly_chart(fig, use_container_width=True)
    
    # Optional: Cluster Comparison by Experiment
    st.subheader("Cluster Distribution by Experiment")
    exp1_cluster = exp1_df.groupby("physio_state").size() / len(exp1_df)
    exp2_cluster = exp2_df.groupby("physio_state").size() / len(exp2_df)
    
    cluster_df = pd.DataFrame({
        "Experiment 1": exp1_cluster,
        "Experiment 2": exp2_cluster
    }).fillna(0)
    
    fig = px.bar(cluster_df, barmode="group", title="Cluster Distribution by Experiment")
    st.plotly_chart(fig, use_container_width=True)
