# Cognitive Physiological State Explorer

**Author:** Kavisha Reddy

**Project Link:** [Cognitive Physio Explorer App](https://cognitive-physio-explorer.streamlit.app/)

**Project Repositroy:** [Capstone Project Repository](https://github.com/Kavisha-29-R/BME_DATA_SCIENCE_CAPSTONE_PROJECT)

---

## Project Description

The **Cognitive Physiological State Explorer** is an interactive web application built using **Streamlit** that allows researchers and enthusiasts to explore peripheral physiological signals collected from wearable devices during cognitive experiments.

The app focuses on **behavioral and physiological features** such as:

- Electrodermal Activity (EDA) – left/right sensors
- Heart Rate (HR) and Heart Rate Variability (HRV) metrics
- Accelerometer-derived movement
- Skin temperature

The app provides a comprehensive visualization and analysis platform for:

- Tracking cognitive performance (reaction time, accuracy)
- Monitoring stress and arousal levels
- Exploring unsupervised physiological clusters
- Comparing experimental sessions for individual participants

**Purpose & Use Cases:**

- **Research:** Supports cognitive monitoring studies using wearable devices without EEG data.
- **Education:** Demonstrates physiological signal analysis and clustering techniques.
- **Experiment Exploration:** Enables users to interactively explore participant-specific responses.
- **Feature Interpretation:** Provides explanations for HRV, EDA, and movement features in cognitive and stress contexts.

---

## Features

1. **Home Page**
   - Overview of the project purpose and data source.
   - Explanation of the clustering approach and physiological features.
   - Links to related research papers.

2. **Participant Overview**
   - Summary statistics of behavioral and physiological features.
   - Interactive time-series plots for metrics like reaction time, EDA, HR, movement, and skin temperature.
   - HRV feature definitions and a correlation heatmap.

3. **Cognitive & Physiological Classification**
   - Classifies participants into high vs low performance and stressed vs calm states.
   - Visualizes cluster distributions and physiological state interpretations.
   - Boxplots of mean RT and proportion correct by session and cluster/state.
   - Threshold tables for physiological feature interpretation.

4. **Physiological Cluster Explorer**
   - Sliders to simulate a participant’s physiological profile.
   - Predicts cluster group and shows distance to cluster centroids.
   - Provides detailed physiological cluster interpretations.

---

## Running the App

**Requirements:**

- Python 3.12+ (recommended)
- Required packages listed in `requirements.txt`

**Steps:**

1. Clone or download the repository.
2. Create a Python virtual environment:
```bash
   python -m venv venv
````

3. Create a new conda environment:
```bash
conda create -n cognitive_env python=3.12
```

3. Activate the environment:

   * Windows:

     ```bash
     venv\Scripts\activate
     ```
      ```bash
     conda activate cognitive_env
     ```
   * macOS/Linux:

     ```bash
     source venv/bin/activate
     ```
4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
5. Navigate to the `App` folder:

   ```bash
   cd App
   ```
6. Run the Streamlit app:

   ```bash
   streamlit run brain_cognitive_state_app.py
   ```

---

## Video Demonstration

A **5-minute video demonstration** of the app can be found here in this repository: /Video/Demonstration_Video_App.mp4

The video showcases all key functionalities including participant selection, cognitive and stress classification, physiological cluster exploration, and interactive visualizations.

---

## Folder Structure

```
App/
├── brain_cognitive_state_app.py   # Main Streamlit app
├── data/
│   ├── all_participants_features.csv
│   └── all_participants_clusters.csv
├── requirements.txt
└── README.md
```

---

## References

* **Dataset:** [Brain Wearable Monitoring Dataset (PhysioNet)](https://physionet.org/content/brain-wearable-monitoring/1.0.0/)
* **Research Paper:** Regulation of brain cognitive states through auditory, gustatory, and olfactory stimulation with wearable monitoring: [Nature Scientific Reports](https://www.nature.com/articles/s41598-023-37829-z)

---

## Notes

* All analysis uses **peripheral physiological features**; EEG signals are not used.
* Clustering is **unsupervised** and provides physiologically interpretable states rather than definitive cognitive labels.
* The app was designed for **individual participant exploration** and **session-based comparisons**.


