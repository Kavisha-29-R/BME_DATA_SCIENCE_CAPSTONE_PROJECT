# brain_cognitive_state_app.py
# Streamlit app: cognitive state classification with interactive plots
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import os

base_dir = os.path.dirname(__file__)  
features_file = os.path.join(base_dir, "data", "all_participants_features.csv")
clusters_file = os.path.join(base_dir, "data", "all_participants_clusters.csv")


# --------------------------
# Load Data
# --------------------------
@st.cache_data
def load_data_full():
    #df = pd.read_csv("data/all_participants_features.csv")
    df = pd.read_csv(features_file)
    df['window_center'] = pd.to_datetime(df['window_center'], unit='s', utc=True)
    return df

@st.cache_data
def load_data_clusters():
    #df = pd.read_csv("data/all_participants_clusters.csv")
    df = pd.read_csv(clusters_file)
    df['window_center'] = pd.to_datetime(df['window_center'], unit='s', utc=True)
    return df

full_df = load_data_full()
df_cluster = load_data_clusters()

data_copy = full_df.copy()

data_copy = full_df.merge(
    df_cluster[['participant', 'window_center', 'physio_cluster', 'physio_state']],
    on=['participant', 'window_center'],
    how='left'
)

cluster_features = [
    # EDA (sympathetic arousal)
    'left_eda_mean',
    'right_eda_mean',
    'left_eda_n_peaks',
    'right_eda_n_peaks',

    # Heart rate
    'left_bvp_hr_mean',
    'right_bvp_hr_mean',

    # Movement context
    'left_acc_mag_mean',
    'right_acc_mag_mean',

    # Skin temperature
    'left_temp_mean',
    'right_temp_mean'
]

# Create cluster description mapping
cluster_descriptions = {
    0: "High tonic arousal (EDA mean), low movement; likely focused/alert states.",
    1: "Low arousal, minimal movement; relaxed or disengaged states.",
    2: "Low arousal, low movement, high temp; relaxed and comfortable states."
}

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

# Filter data for selected participant 
df_participant = full_df[full_df['participant'] == selected_participant].copy()


# Filter data for selected participant 
df_participant = full_df[full_df['participant'] == selected_participant].copy()

# --------------------------
# Page Navigation
# --------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "Participant Overview",
        "Cognitive & Physiological Classification",
        "Physiological Cluster Explorer"
    ]
)

# --------------------------
# Home Page
# --------------------------
if page == "Home":
    st.title("Physiological Monitoring of Cognitive States")
    st.markdown("### Wearable-based Cognitive State Dashboard")

    st.markdown(
        """
        **Purpose of this app**

        This Streamlit dashboard provides an interactive visualization and analysis
        of **physiological signals collected from wearable devices** during two cognitive
        experiments. The app focuses exclusively on **peripheral physiological features**
        (e.g. electrodermal activity, heart rate, heart rate variability, movement, and skin temperature)
        and **does not use EEG signals**.

        The goals of this dashboard are to:
        - Explore physiological dynamics over time for individual participants
        - Compare responses across experimental session types
        - Identify **unsupervised physiological states** using clustering
        - Support interpretation of wearable signals in cognitive monitoring research
        """
    )

    st.markdown(
        """
        **Data source**

        The data used in this project comes from the following PhysioNet dataset:

        [Brain Wearable Monitoring Dataset (PhysioNet)](https://physionet.org/content/brain-wearable-monitoring/1.0.0/)
        """
    )

    st.markdown(
        """
        **Clustering approach**

        Cognitive states are explored using **K-Means clustering applied only to
        physiological features**, including:
        - Electrodermal activity (EDA)
        - Heart rate (HR) and heart rate variability (HRV), e.g., RMSSD and SDNN
        - Accelerometer-derived movement
        - Skin temperature

        The clustering is **unsupervised** and intended to reveal **physiologically
        interpretable states** (e.g., baseline, arousal, movement-related activity),
        rather than definitive cognitive labels.
        """
    )

    st.markdown("---")

    # link to research paper
    st.subheader("Related Research Paper")
    
    st.markdown(
        """
        **References**

        The logic and research used in this project comes from the following:

        [Regulation of brain cognitive states through auditory, gustatory, and olfactory stimulation with wearable monitoring](https://www.nature.com/articles/s41598-023-37829-z )
        """
    )

# --------------------------
# Participant Overview 
# --------------------------
if page == "Participant Overview":
    st.title(f"Participant Overview: {selected_participant}")

    st.subheader("Summary Statistics")
    st.caption(
        "Descriptive statistics of behavioural and physiological features computed over sliding windows. "
        "These features are used to characterise cognitive state changes during sensory stimulation."
    )
    st.dataframe(df_participant.describe())

    st.subheader("Behavioural and Physiological Time-Series")

    # ------------------------------------------------------------------
    # Feature metadata 
    # ------------------------------------------------------------------
    feature_info = {
        'mean_RT': {
            'title': "Mean Reaction Time (Cognitive Processing Speed)",
            'description': (
                "Mean reaction time reflects cognitive processing speed and attentional engagement. "
                "Lower reaction times are typically associated with higher alertness and improved cognitive performance, "
                "while increased reaction times may indicate fatigue or reduced cognitive efficiency."
            )
        },
        'prop_correct': {
            'title': "Proportion of Correct Responses (Task Accuracy)",
            'description': (
                "Proportion correct represents behavioural accuracy during task execution. "
                "This metric captures sustained attention and decision-making quality, which are sensitive "
                "to changes in cognitive state induced by sensory stimulation."
            )
        },
        'eda_mean': {
            'title': "Electrodermal Activity - Mean Level (Sympathetic Arousal)",
            'description': (
                "Mean electrodermal activity (EDA) reflects sympathetic nervous system activation. "
                "Increased EDA is associated with heightened arousal and emotional or cognitive engagement, "
                "which the study links to modulation of brain cognitive states."
            )
        },
        'bvp_hr_mean': {
            'title': "Heart Rate from BVP (Autonomic Regulation)",
            'description': (
                "Heart rate derived from blood volume pulse (BVP) provides insight into autonomic nervous system balance. "
                "Variations in heart rate are linked to cognitive load, stress, and sensory-induced regulation effects."
            )
        },
        'acc_mag_mean': {
            'title': "Acceleration Magnitude (Movement Context)",
            'description': (
                "Mean acceleration magnitude captures physical movement during the experiment. "
                "This feature helps contextualise physiological changes by distinguishing cognitive effects "
                "from motion-related artefacts."
            )
        },
        'temp_mean': {
            'title': "Skin Temperature (Peripheral Physiological Response)",
            'description': (
                "Skin temperature reflects peripheral vascular changes influenced by autonomic regulation. "
                "Gradual changes in temperature have been associated with stress, relaxation, and cognitive state shifts."
            )
        },
        'ibi_ibi_rmssd': {
            'title': "Heart Rate Variability - RMSSD (Parasympathetic Activity)",
            'description': (
                "RMSSD is a heart rate variability metric sensitive to parasympathetic (vagal) activity. "
                "It is computed as the root mean square of successive differences between inter-beat intervals (IBIs) "
                "within each sliding window. Higher RMSSD values are associated with relaxation and adaptive cognitive regulation, "
                "while lower values may indicate cognitive load or sympathetic dominance. "
                "Windows with fewer than two IBIs yield undefined RMSSD."
            )
        },
        'ibi_ibi_sdnn': {
            'title': "Heart Rate Variability - SDNN (Overall Autonomic Variability)",
            'description': (
            "SDNN represents overall heart rate variability within a sliding window, capturing both sympathetic and parasympathetic influences. "
            "It is calculated as the sample standard deviation of IBIs (ddof=1). "
            "Higher SDNN values reflect greater autonomic flexibility, while lower values indicate reduced variability or sustained stress."
            )
        }
    }

    # ------------------------------------------------------------------
    # Metric plotting configuration
    # ------------------------------------------------------------------
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
    
    session_types = ['All'] + sorted(df_participant['session_type'].dropna().unique().tolist())

    # ------------------------------------------------------------------
    # Plot loop
    # ------------------------------------------------------------------
    for metric, pair in metrics_pairs:

        # Section header + description
        st.markdown(f"### {feature_info[metric]['title']}")
        st.caption(feature_info[metric]['description'])
        
        # Session type filter 
        selected_session = st.selectbox(
            "Filter by session type",
            session_types,
            key=f"{metric}_session_filter"
        )
        
        if selected_session == 'All':
            plot_df = df_participant.copy()
        else:
            plot_df = df_participant[df_participant['session_type'] == selected_session]

        if plot_df.empty:
            st.warning("No data available for the selected session type.")
            continue

        if pair:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=plot_df['window_center'],
                y=plot_df[pair[0]],
                mode='lines+markers',
                name='Left Sensor'
            ))
            fig.add_trace(go.Scatter(
                x=plot_df['window_center'],
                y=plot_df[pair[1]],
                mode='lines+markers',
                name='Right Sensor'
            ))
            fig.update_layout(
                title=f"{feature_info[metric]['title']} (Left vs Right)",
                xaxis_title="Time (UTC)",
                yaxis_title=feature_info[metric]['title'],
                hovermode='x unified'
            )
        else:
            fig = px.line(
                plot_df,
                x='window_center',
                y=metric,
                title=feature_info[metric]['title'],
                labels={
                    "window_center": "Time (UTC)",
                    metric: feature_info[metric]['title']
                }
            )
            fig.update_traces(mode='lines+markers')

        st.plotly_chart(fig, width='stretch')
        
    # ------------------------------------------------------------------
    # HRV Feature Definitions Table (Compact)
    # ------------------------------------------------------------------
    with st.expander("Heart Rate Variability (HRV) Feature Definitions", expanded=False):

        st.markdown("**IBI Count** (`ibi_count`)") 
        st.markdown(r"""
        - **Definition:** Number of valid IBIs in the sliding window  
        - **Meaning:** Data sufficiency indicator  
        - **Cognitive Interpretation:** Low values may indicate poor signal quality or insufficient beats
        """)

        st.markdown("**SDNN** (`ibi_sdnn`)") 
        st.markdown(r"""
        - **Definition:** Sample standard deviation of IBIs  
        - **Meaning:** Reflects overall HRV (sympathetic + parasympathetic)  
        - **Cognitive Interpretation:** Higher = flexible autonomic regulation; lower = stress/fatigue
        """)

        st.markdown("**RMSSD** (`ibi_rmssd`)") 
        st.markdown(r"""
        - **Definition:** Root mean square of successive differences  
        - **Meaning:** Short-term HRV dominated by parasympathetic activity  
        - **Cognitive Interpretation:** Higher = relaxation and adaptive regulation; lower = cognitive load or sympathetic dominance
        """)
        
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
# Cognitive State Classification
# --------------------------
elif page == "Cognitive & Physiological Classification":
    st.title(f"Cognitive & Physiological Classification: {selected_participant}")

    # Work on a copy to avoid SettingWithCopyWarning
    df_participant_copy = data_copy[data_copy['participant'] == selected_participant].copy()

    # --- Session type filter ---
    session_types = ['All'] + sorted(df_participant_copy['session_type'].dropna().unique())
    selected_session = st.selectbox("Filter by Session Type", session_types, key="session_filter")

    if selected_session != 'All':
        df_participant_copy = df_participant_copy[df_participant_copy['session_type'] == selected_session]

    # --- Performance Classification ---
    st.subheader("High vs Low Performance")

    median_RT = df_participant_copy['mean_RT'].median()
    median_accuracy = df_participant_copy['prop_correct'].median()

    df_participant_copy['performance_state'] = np.where(
        (df_participant_copy['prop_correct'] > median_accuracy) &
        (df_participant_copy['mean_RT'] < median_RT),
        'High',
        'Low'
    )

    fig_perf = px.scatter(
        df_participant_copy,
        x='window_center',
        y='mean_RT',
        color='performance_state',
        symbol='session_type',
        hover_data=['session_type', 'prop_correct', 'mean_RT'],
        color_discrete_map={"High":"green","Low":"red"},
        title="Performance Timeline",
        labels={"window_center":"Time (UTC)", "mean_RT":"Mean RT"}
    )
    st.plotly_chart(fig_perf, width='stretch')

    st.write("Percentage of time in each performance state:")
    st.dataframe((df_participant_copy['performance_state'].value_counts(normalize=True)*100).round(2))

    # --- Stress Classification ---
    st.subheader("Stressed vs Calm")

    median_eda = (df_participant_copy['left_eda_mean'].median() + df_participant_copy['right_eda_mean'].median()) / 2
    median_hr = (df_participant_copy['left_bvp_hr_mean'].median() + df_participant_copy['right_bvp_hr_mean'].median()) / 2
    median_rmssd = (df_participant_copy['left_ibi_ibi_rmssd'].median() + df_participant_copy['right_ibi_ibi_rmssd'].median()) / 2

    df_participant_copy['stress_state'] = np.where(
        ((df_participant_copy['left_eda_mean'] + df_participant_copy['right_eda_mean'])/2 > median_eda) |
        ((df_participant_copy['left_bvp_hr_mean'] + df_participant_copy['right_bvp_hr_mean'])/2 > median_hr) |
        ((df_participant_copy['left_ibi_ibi_rmssd'] + df_participant_copy['right_ibi_ibi_rmssd'])/2 < median_rmssd),
        'Stressed',
        'Calm'
    )

    fig_stress = px.scatter(
        df_participant_copy,
        x='window_center',
        y='left_eda_mean',
        color='stress_state',
        symbol='session_type',
        hover_data=['session_type', 'left_eda_mean', 'right_eda_mean', 'left_bvp_hr_mean', 'right_bvp_hr_mean'],
        color_discrete_map={"Calm":"blue","Stressed":"orange"},
        title="Stress Timeline (EDA Left)",
        labels={"window_center":"Time (UTC)", "left_eda_mean":"EDA Left Mean"}
    )
    st.plotly_chart(fig_stress, width='stretch')

    st.write("Percentage of time in each stress state:")
    st.dataframe((df_participant_copy['stress_state'].value_counts(normalize=True)*100).round(2))

    # --- Physiological Cluster Distribution ---
    
    st.subheader("Physiological Cluster Profiles with Interpretation")

    cluster_profiles = (
        df_cluster
        .groupby('physio_cluster')[cluster_features]
        .mean()
        .round(4)
    )
    
    # Add description column
    cluster_profiles['Description'] = cluster_profiles.index.map(cluster_descriptions)
    
    # Display in the app
    st.dataframe(cluster_profiles)

    cluster_session_dist = (
        df_participant_copy.groupby(['physio_cluster','session_type'])
        .size()
        .reset_index(name='count')
    )

    cluster_session_dist['percentage'] = cluster_session_dist.groupby('session_type')['count'].transform(lambda x: 100 * x / x.sum())

    clusters = cluster_session_dist['physio_cluster'].unique()
    bar_data = []

    for cluster in clusters:
        df_cluster_only = cluster_session_dist[cluster_session_dist['physio_cluster'] == cluster]
        bar_data.append(go.Bar(
            x=df_cluster_only['session_type'],
            y=df_cluster_only['percentage'],
            name=f'Cluster {cluster}'
        ))

    stacked_bar = go.Figure(data=bar_data)
    stacked_bar.update_layout(
        barmode='stack',
        title='Physiological Cluster Distribution',
        xaxis_title='Session Type',
        yaxis_title='Percentage of Time (%)',
        legend_title='Cluster',
        template='plotly_white'
    )
    st.plotly_chart(stacked_bar, width='stretch')

    # --- Mean RT and Proportion Correct by Session / Cluster ---
    st.subheader("Mean RT and Accuracy by Session / Physiological Cluster")

    fig_box = go.Figure()
    for cluster in df_participant_copy['physio_cluster'].unique():
        df_c = df_participant_copy[df_participant_copy['physio_cluster'] == cluster]
        fig_box.add_trace(go.Box(
            y=df_c['mean_RT'],
            x=df_c['session_type'],
            name=f'Cluster {cluster}',
            boxmean='sd'
        ))

    fig_box.update_layout(
        title="Mean RT Distribution by Session and Cluster",
        xaxis_title="Session Type",
        yaxis_title="Mean Reaction Time (ms)",
        boxmode='group'
    )
    st.plotly_chart(fig_box, width='stretch')

    fig_box_acc = go.Figure()
    for cluster in df_participant_copy['physio_cluster'].unique():
        df_c = df_participant_copy[df_participant_copy['physio_cluster'] == cluster]
        fig_box_acc.add_trace(go.Box(
            y=df_c['prop_correct'],
            x=df_c['session_type'],
            name=f'Cluster {cluster}',
            boxmean='sd'
        ))
    fig_box_acc.update_layout(
        title="Proportion Correct by Session and Cluster",
        xaxis_title="Session Type",
        yaxis_title="Proportion Correct",
        boxmode='group'
    )
    st.plotly_chart(fig_box_acc, width='stretch')
    
    # Define the threshold table
    physio_thresholds = pd.DataFrame([
        {
            "Label": "Movement / Artifact",
            "Condition": "ACC > 0.83",
            "Meaning": "High movement; data may be contaminated by motion artifacts."
        },
        {
            "Label": "High Arousal",
            "Condition": "EDA mean > 0.51 OR EDA peaks > 0.67 AND HR > 0.63",
            "Meaning": "Sympathetic activation; participant likely stressed, alert, or engaged."
        },
        {
            "Label": "Low Arousal",
            "Condition": "ACC <= 0.83 AND EDA mean <= 0.51 AND EDA peaks <= 0.67 AND HR <= 0.63",
            "Meaning": "Calm or resting state; baseline physiological activity."
        }
    ])

    # Display in Streamlit
    st.subheader("Physiological States Thresholds & Interpretations")
    st.dataframe(physio_thresholds)
    
    threshold_summary = {}

    for feature in cluster_features:
        threshold_summary[feature] = {
            '50th_percentile': df_cluster[feature].quantile(0.50),
            '75th_percentile': df_cluster[feature].quantile(0.75),
            '84th_percentile (~+1z)': df_cluster[feature].quantile(0.84),
            '90th_percentile': df_cluster[feature].quantile(0.90)
        }

    summary_threshold_df = pd.DataFrame(threshold_summary)
    
    # Display in the app
    st.dataframe(summary_threshold_df)
    # --- Mean RT and Proportion Correct by Session / Physiological State ---
    st.subheader("Mean RT and Accuracy by Session / Physiological State")

    fig_box = go.Figure()
    for state in df_participant_copy['physio_state'].unique():
        df_c = df_participant_copy[df_participant_copy['physio_state'] == state]
        fig_box.add_trace(go.Box(
            y=df_c['mean_RT'],
            x=df_c['session_type'],
            name=f'State {state}',
            boxmean='sd'
        ))

    fig_box.update_layout(
        title="Mean RT Distribution by Session and State",
        xaxis_title="Session Type",
        yaxis_title="Mean Reaction Time (ms)",
        boxmode='group'
    )
    st.plotly_chart(fig_box, width='stretch')

    fig_box_acc = go.Figure()
    for state in df_participant_copy['physio_state'].unique():
        df_c = df_participant_copy[df_participant_copy['physio_state'] == state]
        fig_box_acc.add_trace(go.Box(
            y=df_c['prop_correct'],
            x=df_c['session_type'],
            name=f'State {state}',
            boxmean='sd'
        ))
    fig_box_acc.update_layout(
        title="Proportion Correct by Session and State",
        xaxis_title="Session Type",
        yaxis_title="Proportion Correct",
        boxmode='group'
    )
    st.plotly_chart(fig_box_acc, width='stretch')
    
# --------------------------
# Physiological Cluster Explorer
# --------------------------
elif page == "Physiological Cluster Explorer":
    st.title("Physiological Cluster Explorer")
    
    st.markdown("""
        Adjust the sliders for physiological features to see which cluster this profile would belong to,
        along with its interpretation.
    """)
    
    st.markdown("""
    ### Physiological Feature Sliders
    Adjust the sliders to simulate a participant’s physiological state. Each feature represents a peripheral signal measured from wearable sensors:

    - **EDA Mean (Left/Right):** Sympathetic arousal; higher = stressed/alert, lower = calm.
    - **EDA Peaks (Left/Right):** Number of rapid EDA changes; more peaks = more physiological responses.
    - **Heart Rate (Left/Right):** Beats per minute; higher = stress or activity, lower = relaxed.
    - **Acceleration Magnitude (Left/Right):** Physical movement; higher = moving, lower = still.
    - **Skin Temperature (Left/Right):** Peripheral temperature; reflects stress vs relaxation.
    """)

    # Filter by experiment/session 
    # Select experiment
    experiment = st.radio("Select Experiment", ["Experiment 1 (A-group)", "Experiment 2 (B-group)"])

    # Filter participants by experiment
    if experiment.startswith("Experiment 1"):
        participants_exp = [p for p in full_df['participant'].unique() if str(p).startswith("A")]
    else:
        participants_exp = [p for p in full_df['participant'].unique() if str(p).startswith("B")]

    # Filter sessions based on experiment participants
    sessions_exp = full_df[full_df['participant'].isin(participants_exp)]['session_type'].dropna().unique()
    session_types = ['All'] + sorted(sessions_exp)

    # Select session
    session = st.selectbox("Select Session Type", session_types)

    # Filter df_cluster to selected experiment/session if needed
    filtered_df = df_cluster.copy()
    if experiment.startswith("Experiment 1"):
        participants = [p for p in full_df['participant'].unique() if str(p).startswith("A")]
    else:
        participants = [p for p in full_df['participant'].unique() if str(p).startswith("B")]
    filtered_df = filtered_df[filtered_df['participant'].isin(participants)]
    if session != 'All':
        filtered_df = filtered_df[filtered_df['session_type'] == session]

    # Get ranges for sliders
    slider_values = {}
    for feature in cluster_features:
        min_val = float(filtered_df[feature].min())
        max_val = float(filtered_df[feature].max())
        default_val = float(filtered_df[feature].median())
        slider_values[feature] = st.slider(
            label=feature,
            min_value=min_val,
            max_value=max_val,
            value=default_val
        )
        
    # Convert input into array
    user_input = np.array([slider_values[f] for f in cluster_features]).reshape(1, -1)
    
    # Choose number of clusters
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    def prepare_physio_data_for_clustering(df, features):
        df_clean = df.copy()

        df_clean = df_clean[['participant', 'session_type', 'window_center'] + features]

        # Cast to float
        df_clean[features] = df_clean[features].astype(float)

        # ---- Impute missing values per participant (median) ----
        for p in df_clean['participant'].unique():
            mask = df_clean['participant'] == p
            df_p = df_clean.loc[mask, features]

            medians = df_p.median()
            df_clean.loc[mask, features] = df_p.fillna(medians)

        # ---- Z-score normalization per participant ----
        df_norm = []
        for p in df_clean['participant'].unique():
            df_p = df_clean[df_clean['participant'] == p].copy()
            scaler = StandardScaler()
            df_p[features] = scaler.fit_transform(df_p[features])
            df_norm.append(df_p)

        df_norm = pd.concat(df_norm, ignore_index=True)
        return df_norm
    
    
    
    df_model = prepare_physio_data_for_clustering(full_df, cluster_features)
    
    # Choose number of clusters
    k = 3  # physiologically interpretable (low / high arousal / movement)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
    df_model['physio_cluster'] = kmeans.fit_predict(df_model[cluster_features])

    # Predict nearest cluster (using fitted KMeans)
    user_input_df = pd.DataFrame([slider_values], columns=cluster_features)
    cluster_label = kmeans.predict(user_input_df)[0]

    #cluster_label = kmeans.predict(user_input)[0]
    
    st.subheader(f"Predicted Cluster: {cluster_label}")
    st.write(f"Description: {cluster_descriptions[cluster_label]}")
    
    #distances = np.linalg.norm(kmeans.cluster_centers_ - user_input, axis=1)
    distances = np.linalg.norm(kmeans.cluster_centers_ - user_input_df.values, axis=1)

    st.subheader("Distance to Cluster Centroids")
    for i, dist in enumerate(distances):
        st.write(f"Cluster {i}: {dist:.4f}")
