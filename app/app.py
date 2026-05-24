import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import re
from pathlib import Path

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"

st.set_page_config(
    page_title="Issue Salience Electoral Predictor",
    page_icon="\U0001F5F3️",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_data
def load_artifacts():
    data = {}

    parquet_files = {
        "df_sample": "df_sample.parquet",
        "df_model": "df_model.parquet",
        "feat_imp": "feat_imp.parquet",
    }
    for key, fname in parquet_files.items():
        path = ARTIFACTS_DIR / fname
        if path.exists():
            data[key] = pd.read_parquet(path)

    json_files = {
        "topic_labels": "topic_labels.json",
        "political_topic_ids": "political_topic_ids.json",
        "model_metrics": "model_metrics.json",
        "state_metadata": "state_metadata.json",
    }
    for key, fname in json_files.items():
        path = ARTIFACTS_DIR / fname
        if path.exists():
            with open(path) as f:
                data[key] = json.load(f)

    return data


@st.cache_resource
def load_topic_model():
    model_path = ARTIFACTS_DIR / "topic_model"
    if model_path.exists():
        from bertopic import BERTopic
        return BERTopic.load(str(model_path))
    return None


@st.cache_resource
def load_vader():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()


@st.cache_resource
def load_zero_shot():
    from transformers import pipeline
    return pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=-1
    )


def clean_tweet(text):
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def artifacts_available():
    return (ARTIFACTS_DIR / "df_model.parquet").exists()


# ── Sidebar ──────────────────────────────────────────────────

st.sidebar.title("\U0001F5F3️ Issue Salience Predictor")
st.sidebar.markdown("Predicting U.S. election outcomes from political tweet discourse.")

if not artifacts_available():
    st.sidebar.warning(
        "Artifacts not found. Run the notebook first to generate "
        "`app/artifacts/`. The app will show demo placeholders."
    )

tab = st.sidebar.radio(
    "Navigate",
    ["\U0001F50D Live Tweet Analysis",
     "\U0001F5FA️ Interactive State Maps",
     "\U0001F4CA Topic Explorer",
     "\U0001F4C8 Model Performance"]
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Research Question:** To what extent do state-level issue salience "
    "profiles predict partisan electoral outcomes across U.S. states?"
)
st.sidebar.markdown("*Petrocik (1996) — Issue Ownership Theory*")


# ── Tab 1: Live Tweet Analysis ───────────────────────────────

if tab == "\U0001F50D Live Tweet Analysis":
    st.header("Live Tweet Analysis")
    st.markdown(
        "Enter a tweet below to analyze its **political topic**, "
        "**sentiment**, and **partisan stance** in real-time."
    )

    tweet_input = st.text_area(
        "Enter a tweet:",
        placeholder="e.g., The economy is in terrible shape, prices are through the roof and nobody in Washington cares",
        height=100
    )

    if st.button("Analyze", type="primary") and tweet_input.strip():
        cleaned = clean_tweet(tweet_input)

        col1, col2, col3 = st.columns(3)

        # VADER Sentiment
        with col2:
            st.subheader("Sentiment (VADER)")
            analyzer = load_vader()
            scores = analyzer.polarity_scores(cleaned)
            compound = scores["compound"]

            if compound >= 0.05:
                label = "Positive"
                color = "green"
            elif compound <= -0.05:
                label = "Negative"
                color = "red"
            else:
                label = "Neutral"
                color = "gray"

            st.metric("Compound Score", f"{compound:.3f}", delta=label)

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=compound,
                domain={"x": [0, 1], "y": [0, 1]},
                gauge={
                    "axis": {"range": [-1, 1]},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [-1, -0.05], "color": "#ffcccc"},
                        {"range": [-0.05, 0.05], "color": "#f0f0f0"},
                        {"range": [0.05, 1], "color": "#ccffcc"},
                    ],
                },
                title={"text": "Sentiment"}
            ))
            fig.update_layout(height=250, margin=dict(t=50, b=0, l=30, r=30))
            st.plotly_chart(fig, use_container_width=True)

            st.caption(f"Pos: {scores['pos']:.3f} | Neu: {scores['neu']:.3f} | Neg: {scores['neg']:.3f}")

        # BERTopic
        with col1:
            st.subheader("Topic (BERTopic)")
            topic_model = load_topic_model()
            if topic_model is not None:
                topics_pred, probs_pred = topic_model.transform([cleaned])
                topic_id = topics_pred[0]

                data = load_artifacts()
                topic_labels = data.get("topic_labels", {})
                label = topic_labels.get(str(topic_id), f"Topic {topic_id}")

                if topic_id == -1:
                    st.warning("No clear topic detected (noise cluster)")
                else:
                    st.success(f"**{label}**")

                st.metric("Topic ID", topic_id)

                if probs_pred is not None and len(probs_pred) > 0:
                    prob_arr = probs_pred[0] if hasattr(probs_pred[0], '__len__') else probs_pred
                    if hasattr(prob_arr, '__len__') and len(prob_arr) > 1:
                        top_indices = np.argsort(prob_arr)[-5:][::-1]
                        st.markdown("**Top 5 topic probabilities:**")
                        for idx in top_indices:
                            tl = topic_labels.get(str(idx), f"Topic {idx}")
                            st.caption(f"{tl}: {prob_arr[idx]:.3f}")
            else:
                st.info("BERTopic model not loaded. Run the notebook to export artifacts.")

        # Zero-Shot Stance
        with col3:
            st.subheader("Partisan Stance (LLM)")
            with st.spinner("Running zero-shot classification..."):
                try:
                    classifier = load_zero_shot()
                    labels = [
                        "supports Republican party",
                        "supports Democratic party",
                        "neutral or nonpartisan"
                    ]
                    result = classifier(cleaned, candidate_labels=labels)

                    top_label = result["labels"][0]
                    top_score = result["scores"][0]

                    if "Republican" in top_label:
                        st.error(f"**{top_label}**")
                    elif "Democratic" in top_label:
                        st.info(f"**{top_label}**")
                    else:
                        st.warning(f"**{top_label}**")

                    st.metric("Confidence", f"{top_score:.1%}")

                    fig = px.bar(
                        x=result["scores"],
                        y=result["labels"],
                        orientation="h",
                        labels={"x": "Confidence", "y": ""},
                        color=result["labels"],
                        color_discrete_map={
                            "supports Republican party": "#e74c3c",
                            "supports Democratic party": "#3498db",
                            "neutral or nonpartisan": "#95a5a6"
                        }
                    )
                    fig.update_layout(
                        height=200,
                        showlegend=False,
                        margin=dict(t=10, b=10, l=10, r=10),
                        xaxis=dict(range=[0, 1])
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Zero-shot model unavailable: {e}")


# ── Tab 2: Interactive State Maps ────────────────────────────

elif tab == "\U0001F5FA️ Interactive State Maps":
    st.header("Interactive State Maps")

    data = load_artifacts()
    df_model = data.get("df_model")

    if df_model is None:
        st.warning("Model data not found. Run the notebook to export artifacts.")
        st.stop()

    map_choice = st.selectbox("Select map:", [
        "Predicted vs Actual 2024 Outcome",
        "Average Sentiment by State",
        "LLM Partisan Lean by State",
        "Tweet Volume by State"
    ])

    if map_choice == "Predicted vs Actual 2024 Outcome":
        if "predicted" in df_model.columns and "winner" in df_model.columns:
            df_model["result"] = df_model.apply(
                lambda r: f"Correct ({r['winner']})" if r.get("correct", 0) == 1
                else f"Wrong (Actual: {r['winner']}, Pred: {r.get('pred_label', '?')})",
                axis=1
            )
            color_val = df_model.apply(
                lambda r: 2 if r.get("correct", 0) == 1 and r["winner"] == "R"
                else (0 if r.get("correct", 0) == 1 and r["winner"] == "D" else 1),
                axis=1
            )
            fig = px.choropleth(
                df_model,
                locations="state_code",
                locationmode="USA-states",
                color=color_val,
                color_continuous_scale=["blue", "yellow", "red"],
                scope="usa",
                hover_data=["state_code", "winner", "pred_label", "correct"],
                title="Predicted vs Actual 2024 Election Outcomes"
            )
            fig.update_layout(height=550, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                correct = df_model.get("correct", pd.Series())
                st.metric("Correctly Predicted", f"{correct.sum()}/{len(correct)}")
            with col2:
                st.metric("Accuracy", f"{correct.mean():.1%}" if len(correct) > 0 else "N/A")

            misclassified = df_model[df_model.get("correct", 0) == 0]
            if len(misclassified) > 0:
                st.markdown("**Misclassified states:**")
                for _, row in misclassified.iterrows():
                    st.caption(
                        f"{row['state_code']}: Actual={row['winner']}, "
                        f"Predicted={row.get('pred_label', '?')}"
                    )
        else:
            st.info("Prediction data not available in artifacts.")

    elif map_choice == "Average Sentiment by State":
        if "sentiment_mean" in df_model.columns:
            fig = px.choropleth(
                df_model,
                locations="state_code",
                locationmode="USA-states",
                color="sentiment_mean",
                color_continuous_scale="RdYlGn",
                color_continuous_midpoint=0,
                scope="usa",
                title="Average Tweet Sentiment by State"
            )
            fig.update_layout(height=550)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Sentiment data not available.")

    elif map_choice == "LLM Partisan Lean by State":
        if "partisan_lean" in df_model.columns:
            fig = px.choropleth(
                df_model,
                locations="state_code",
                locationmode="USA-states",
                color="partisan_lean",
                color_continuous_scale="RdBu",
                color_continuous_midpoint=0,
                scope="usa",
                title="LLM-Derived Partisan Lean (Blue=Dem, Red=Rep)",
                labels={"partisan_lean": "Partisan Lean"}
            )
            fig.update_layout(height=550)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Partisan lean data not available.")

    elif map_choice == "Tweet Volume by State":
        if "tweet_count" in df_model.columns:
            fig = px.choropleth(
                df_model,
                locations="state_code",
                locationmode="USA-states",
                color="tweet_count",
                color_continuous_scale="YlOrRd",
                scope="usa",
                title="Tweet Volume by State"
            )
            fig.update_layout(height=550)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Tweet count data not available.")


# ── Tab 3: Topic Explorer ────────────────────────────────────

elif tab == "\U0001F4CA Topic Explorer":
    st.header("Topic Explorer")

    data = load_artifacts()
    df_sample = data.get("df_sample")
    topic_labels = data.get("topic_labels", {})
    political_ids = data.get("political_topic_ids", [])

    if df_sample is None:
        st.warning("Sample data not found. Run the notebook to export artifacts.")
        st.stop()

    political_labels = {str(k): v for k, v in topic_labels.items() if k in political_ids or int(k) in political_ids}
    label_list = sorted(set(political_labels.values()))

    col1, col2 = st.columns(2)

    with col1:
        selected_state = st.selectbox(
            "Select a state:",
            ["All States"] + sorted(df_sample["state_code"].dropna().unique().tolist())
        )

    with col2:
        selected_topic = st.selectbox(
            "Select a topic:",
            ["All Topics"] + label_list
        )

    df_filtered = df_sample[df_sample["topic"].isin(political_ids)].copy()
    if "topic_label" not in df_filtered.columns:
        df_filtered["topic_label"] = df_filtered["topic"].map(
            {int(k): v for k, v in topic_labels.items()}
        )

    if selected_state != "All States":
        df_filtered = df_filtered[df_filtered["state_code"] == selected_state]

    if selected_topic != "All Topics":
        df_filtered = df_filtered[df_filtered["topic_label"] == selected_topic]

    st.markdown(f"**{len(df_filtered):,} political tweets** matching filters")

    # Topic distribution chart
    if selected_state != "All States" and selected_topic == "All Topics":
        topic_counts = df_filtered["topic_label"].value_counts()
        fig = px.bar(
            x=topic_counts.index, y=topic_counts.values,
            title=f"Topic Distribution in {selected_state}",
            labels={"x": "Topic", "y": "Tweet Count"},
            color=topic_counts.index,
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig.update_layout(showlegend=False, xaxis_tickangle=45, height=400)
        st.plotly_chart(fig, use_container_width=True)

    elif selected_topic != "All Topics" and selected_state == "All States":
        state_counts = df_filtered["state_code"].value_counts().head(20)
        fig = px.bar(
            x=state_counts.index, y=state_counts.values,
            title=f"Top 20 States Discussing: {selected_topic}",
            labels={"x": "State", "y": "Tweet Count"}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    else:
        topic_counts = df_filtered["topic_label"].value_counts()
        fig = px.bar(
            x=topic_counts.index, y=topic_counts.values,
            title="Political Topic Distribution",
            labels={"x": "Topic", "y": "Tweet Count"},
            color=topic_counts.index,
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig.update_layout(showlegend=False, xaxis_tickangle=45, height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Sentiment breakdown
    if "vader_compound" in df_filtered.columns and len(df_filtered) > 0:
        st.subheader("Sentiment Breakdown")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean Sentiment", f"{df_filtered['vader_compound'].mean():.3f}")
        with col2:
            pos_pct = (df_filtered["vader_compound"] >= 0.05).mean() * 100
            st.metric("Positive %", f"{pos_pct:.1f}%")
        with col3:
            neg_pct = (df_filtered["vader_compound"] <= -0.05).mean() * 100
            st.metric("Negative %", f"{neg_pct:.1f}%")

    # Sample tweets
    st.subheader("Sample Tweets")
    if len(df_filtered) > 0:
        sample_size = min(10, len(df_filtered))
        samples = df_filtered.sample(n=sample_size, random_state=42)
        display_cols = ["tweet_clean", "topic_label", "state_code"]
        if "vader_compound" in samples.columns:
            display_cols.append("vader_compound")
        st.dataframe(
            samples[display_cols].reset_index(drop=True),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No tweets match the current filters.")


# ── Tab 4: Model Performance Dashboard ───────────────────────

elif tab == "\U0001F4C8 Model Performance":
    st.header("Model Performance Dashboard")

    data = load_artifacts()
    metrics = data.get("model_metrics", {})
    df_model = data.get("df_model")
    feat_imp = data.get("feat_imp")

    # Overview metrics
    st.subheader("Pipeline Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Coherence (c_v)", f"{metrics.get('coherence', 'N/A')}")
    with col2:
        st.metric("Outlier Rate", f"{metrics.get('outlier_pct', 'N/A')}%")
    with col3:
        st.metric("Best Accuracy", f"{metrics.get('best_accuracy', 'N/A')}")
    with col4:
        st.metric("Best Model", metrics.get("best_model", "N/A"))

    # Model comparison
    if "model_results" in metrics:
        st.subheader("Model Comparison (LOOCV)")
        results = metrics["model_results"]
        df_results = pd.DataFrame(results).T
        st.dataframe(df_results, use_container_width=True)

    # Feature importance
    if feat_imp is not None:
        st.subheader("Feature Importance")
        feat_imp_sorted = feat_imp.sort_values("importance", ascending=True).tail(15)
        fig = px.bar(
            feat_imp_sorted,
            x="importance", y="feature",
            orientation="h",
            title=f"Top 15 Features ({metrics.get('best_model', 'Best Model')})",
            labels={"importance": "Importance", "feature": "Feature"}
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    # Bias audit summary
    if df_model is not None:
        st.subheader("Bias Audit Summary")

        # Geographic representation
        state_pop = {
            'CA': 39538, 'TX': 29146, 'FL': 21538, 'NY': 20202, 'PA': 13002,
            'IL': 12812, 'OH': 11800, 'GA': 10712, 'NC': 10439, 'MI': 10078,
            'NJ': 9289, 'VA': 8632, 'WA': 7615, 'AZ': 7151, 'MA': 7030,
            'TN': 6910, 'IN': 6786, 'MD': 6178, 'MO': 6155, 'WI': 5894,
            'CO': 5774, 'MN': 5707, 'SC': 5119, 'AL': 5024, 'LA': 4658,
            'KY': 4506, 'OR': 4238, 'OK': 3960, 'CT': 3606, 'UT': 3272,
            'IA': 3190, 'NV': 3105, 'AR': 3012, 'MS': 2962, 'KS': 2937,
            'NM': 2118, 'NE': 1962, 'ID': 1901, 'WV': 1794, 'HI': 1456,
            'NH': 1378, 'ME': 1362, 'MT': 1085, 'RI': 1098, 'DE': 990,
            'SD': 887, 'ND': 779, 'AK': 733, 'DC': 690, 'VT': 643, 'WY': 577
        }
        total_pop = sum(state_pop.values())

        if "tweet_count" in df_model.columns:
            total_tweets = df_model["tweet_count"].sum()
            df_model["pop"] = df_model["state_code"].map(state_pop)
            df_model["rep_ratio"] = (
                (df_model["tweet_count"] / total_tweets) /
                (df_model["pop"] / total_pop)
            )

            fig = px.choropleth(
                df_model.dropna(subset=["rep_ratio"]),
                locations="state_code",
                locationmode="USA-states",
                color="rep_ratio",
                color_continuous_scale="RdYlGn",
                color_continuous_midpoint=1.0,
                scope="usa",
                title="Geographic Representation Ratio (1.0 = proportional)",
                labels={"rep_ratio": "Ratio"}
            )
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)

        # Swing vs Safe accuracy
        if "correct" in df_model.columns:
            swing_states = ["AZ", "GA", "MI", "NV", "NC", "PA", "WI"]
            swing = df_model[df_model["state_code"].isin(swing_states)]
            safe = df_model[~df_model["state_code"].isin(swing_states)]

            col1, col2 = st.columns(2)
            with col1:
                if len(swing) > 0:
                    st.metric("Swing State Accuracy",
                              f"{swing['correct'].mean():.1%} (n={len(swing)})")
            with col2:
                if len(safe) > 0:
                    st.metric("Safe State Accuracy",
                              f"{safe['correct'].mean():.1%} (n={len(safe)})")

    if not metrics and df_model is None and feat_imp is None:
        st.info("No model performance data found. Run the notebook to export artifacts.")
