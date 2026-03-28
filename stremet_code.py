# =============================
# SMART FACTORY ML SYSTEM
# End-to-End Pipeline with Root Cause Analysis
# =============================

import pandas as pd
import numpy as np
import sqlite3
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# =============================
# 1. DATA LOADER
# =============================
class DataLoader:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)

    def load_all(self):
        events = pd.read_sql("SELECT * FROM production_process_events", self.conn)
        machines = pd.read_sql("SELECT * FROM machine_parameters", self.conn)
        materials = pd.read_sql("SELECT * FROM material_information", self.conn)
        quality = pd.read_sql("SELECT * FROM quality_inspection", self.conn)
        queue = pd.read_sql("SELECT * FROM queue_lengths", self.conn)

        df = events.merge(machines, on="machine_id") \
                   .merge(materials, on="unit_id") \
                   .merge(quality, on="unit_id")

        return df, queue

# =============================
# 2. PREPROCESSOR
# =============================
class Preprocessor:
    def transform(self, df):
        df = df.copy()

        # Scrap label
        df['is_scrap'] = df['severity'].apply(
            lambda x: 1 if str(x).lower() in ['high', 'critical'] else 0
        )

        # Encode categorical (simple)
        df = pd.get_dummies(df, drop_first=True)

        return df

# =============================
# 3. ML MODEL
# =============================
class DefectModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100)

    def train(self, df):
        target = 'is_scrap'

        X = df.drop(columns=[target])
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        self.model.fit(X_train, y_train)

        preds = self.model.predict(X_test)
        print(classification_report(y_test, preds))

        self.features = X.columns

    def predict(self, input_data):
        return self.model.predict(input_data)

    def feature_importance(self):
        return pd.DataFrame({
            'feature': self.features,
            'importance': self.model.feature_importances_
        }).sort_values(by='importance', ascending=False)

# =============================
# 4. ROOT CAUSE ANALYZER
# =============================
class RootCauseAnalyzer:
    def __init__(self, model):
        self.model = model

    def analyze(self, df):
        importance = self.model.feature_importance()

        top_factors = importance.head(5)

        insights = []
        for _, row in top_factors.iterrows():
            insights.append(f"High impact factor: {row['feature']} (importance={row['importance']:.2f})")

        return insights, top_factors

# =============================
# 5. BOTTLENECK ANALYZER
# =============================
class BottleneckAnalyzer:
    def detect(self, queue_df):
        bottlenecks = queue_df.sort_values(by='average_wait_time', ascending=False)
        return bottlenecks

# =============================
# 6. MAIN PIPELINE
# =============================
@st.cache_data
def load_and_train():
    loader = DataLoader("stremet_database.db")
    df, queue = loader.load_all()

    pre = Preprocessor()
    df_clean = pre.transform(df)

    model = DefectModel()
    model.train(df_clean)

    return df_clean, queue, model

# =============================
# 7. STREAMLIT DASHBOARD
# =============================

def main():
    st.title("🏭 Smart Factory AI Dashboard")

    df, queue, model = load_and_train()

    st.header("📊 Bottleneck Detection")
    bottleneck = BottleneckAnalyzer().detect(queue)
    st.dataframe(bottleneck.head(10))

    st.header("🤖 Root Cause Analysis")
    analyzer = RootCauseAnalyzer(model)
    insights, factors = analyzer.analyze(df)

    for i in insights:
        st.write(i)

    st.dataframe(factors)

    st.header("⚠️ Scrap Risk Prediction")

    sample = df.drop(columns=['is_scrap']).iloc[0:1]

    if st.button("Predict Sample Risk"):
        prediction = model.predict(sample)
        st.write("Scrap Risk:", "HIGH" if prediction[0] == 1 else "LOW")

# =============================
# RUN
# =============================
if __name__ == "__main__":
    main()


