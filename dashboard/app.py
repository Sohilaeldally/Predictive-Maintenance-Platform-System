"""PMPS dashboard (Streamlit).

Talks to the FastAPI service only - it never touches the database directly.

Run from the project root:
    streamlit run dashboard/app.py

The API address can be changed with the API_URL environment variable
(default: http://localhost:8000).
"""
import os

import altair as alt
import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

STATUS_STYLE = {
    "CRITICAL": "background-color: #fecaca; color: #7f1d1d",
    "WARNING": "background-color: #fde68a; color: #78350f",
    "NORMAL": "background-color: #bbf7d0; color: #14532d",
}

st.set_page_config(page_title="PMPS Dashboard", layout="wide")


# ---------- API helpers ----------
def api_get(path: str):
    response = requests.get(f"{API_URL}{path}", timeout=30)
    response.raise_for_status()
    return response.json()


def api_post(path: str):
    # A full simulation can take several seconds, so allow a long timeout.
    response = requests.post(f"{API_URL}{path}", timeout=300)
    response.raise_for_status()
    return response.json()


def show_api_error(exc: Exception) -> None:
    st.error(
        f"Could not complete the request to the API at {API_URL}.\n\n"
        f"Is the API running? Details: {exc}"
    )


def first_cycle_below(df: pd.DataFrame, threshold: float):
    hits = df.loc[df["predicted_rul"] <= threshold, "cycle"]
    return "-" if hits.empty else int(hits.min())


def style_status(df: pd.DataFrame):
    styler = df.style
    apply = styler.map if hasattr(styler, "map") else styler.applymap
    return apply(lambda value: STATUS_STYLE.get(value, ""), subset=["maintenance_status"])


# ---------- Sidebar ----------
st.title("PMPS - Predictive Maintenance Dashboard")

with st.sidebar:
    st.header("Settings")
    st.caption(f"API: {API_URL}")
    critical_rul = st.number_input("CRITICAL when RUL <=", min_value=1, value=15)
    warning_rul = st.number_input("WARNING when RUL <=", min_value=1, value=30)
    st.caption(
        "These lines only affect the chart. Keep them equal to the thresholds "
        "in src/api/services/alerts.py."
    )

if warning_rul <= critical_rul:
    st.warning("The WARNING threshold should be higher than the CRITICAL threshold.")

simulation_tab, latest_tab = st.tabs(["Simulation", "Latest predictions"])


# ---------- Simulation tab ----------
with simulation_tab:
    left, right = st.columns([1, 3])
    engine_id = left.number_input("Engine ID", min_value=1, value=1, step=1)

    if left.button("Run simulation", type="primary"):
        with st.spinner("Replaying the engine's life cycle by cycle..."):
            try:
                created = api_post(f"/simulations/{int(engine_id)}")
                rows = api_get(f"/simulations/{created['simulation_id']}")
            except requests.RequestException as exc:
                show_api_error(exc)
            else:
                st.session_state["simulation"] = {
                    "id": created["simulation_id"],
                    "engine_id": int(engine_id),
                    "rows": rows,
                }

    simulation = st.session_state.get("simulation")

    if simulation is None:
        right.info("Choose an engine and press **Run simulation**.")
    else:
        df = pd.DataFrame(simulation["rows"]).sort_values("cycle")
        has_actual = df["actual_rul"].notna().any()

        st.subheader(f"Engine {simulation['engine_id']}")
        st.caption(f"Simulation ID: {simulation['id']}")

        mae = (df["predicted_rul"] - df["actual_rul"]).abs().mean() if has_actual else None
        last = df.iloc[-1]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Cycles simulated", len(df))
        c2.metric("MAE (cycles)", "-" if mae is None else f"{mae:.1f}")
        c3.metric("First WARNING cycle", first_cycle_below(df, warning_rul))
        c4.metric("First CRITICAL cycle", first_cycle_below(df, critical_rul))

        plot_df = df.melt(
            id_vars="cycle",
            value_vars=["predicted_rul", "actual_rul"],
            var_name="series",
            value_name="rul",
        ).dropna(subset=["rul"])
        plot_df["series"] = plot_df["series"].map(
            {"predicted_rul": "Predicted RUL", "actual_rul": "Actual RUL"}
        )

        lines = (
            alt.Chart(plot_df)
            .mark_line(strokeWidth=2.5)
            .encode(
                x=alt.X("cycle:Q", title="Cycle"),
                y=alt.Y("rul:Q", title="RUL (cycles)"),
                color=alt.Color(
                    "series:N",
                    title=None,
                    scale=alt.Scale(
                        domain=["Predicted RUL", "Actual RUL"],
                        range=["#2563eb", "#6b7280"],
                    ),
                ),
                strokeDash=alt.condition(
                    alt.datum.series == "Actual RUL",
                    alt.value([6, 4]),
                    alt.value([1, 0]),
                ),
                tooltip=["cycle:Q", "series:N", alt.Tooltip("rul:Q", format=".1f")],
            )
        )

        def threshold_line(value: float, color: str, label: str):
            rule = (
                alt.Chart(pd.DataFrame({"y": [value]}))
                .mark_rule(strokeDash=[5, 5], color=color)
                .encode(y="y:Q")
            )
            text = (
                alt.Chart(pd.DataFrame({"y": [value], "text": [label]}))
                .mark_text(align="left", dx=4, dy=-6, color=color)
                .encode(y="y:Q", text="text:N", x=alt.value(0))
            )
            return rule + text

        layers = [lines]
        has_band = {"rul_lower", "rul_upper"} <= set(df.columns) and df["rul_lower"].notna().any()
        if has_band:
            band = (
                alt.Chart(df)
                .mark_area(opacity=0.15, color="#2563eb")
                .encode(
                    x="cycle:Q",
                    y=alt.Y("rul_lower:Q", title="RUL (cycles)"),
                    y2="rul_upper:Q",
                )
            )
            layers = [band, lines]

        chart = alt.layer(
            *layers,
            threshold_line(warning_rul, "#d97706", f"WARNING <= {warning_rul}"),
            threshold_line(critical_rul, "#dc2626", f"CRITICAL <= {critical_rul}"),
        ).properties(height=400, width="container")

        st.altair_chart(chart)

        if has_band:
            st.caption(
                "The blue band is the 80% prediction interval: about 8 times in 10 the "
                "true RUL falls inside it. It is narrow close to failure and wide when "
                "the engine is still healthy."
            )

        if not has_actual:
            st.caption("This simulation has no actual RUL values saved, so only the prediction is shown.")
        else:
            st.caption(
                f"At the last observed cycle the model predicted {last['predicted_rul']:.1f} "
                f"and the true RUL was {last['actual_rul']:.0f}."
            )

        with st.expander("Show data"):
            st.dataframe(df[["cycle", "predicted_rul", "actual_rul"]], hide_index=True)


# ---------- Latest predictions tab ----------
with latest_tab:
    if st.button("Run predictions for all engines"):
        with st.spinner("Predicting every engine..."):
            try:
                summary = api_post("/predictions/run-all")
            except requests.RequestException as exc:
                show_api_error(exc)
            else:
                st.success(
                    f"{summary['engines_predicted']} engines predicted: "
                    f"{summary['critical']} critical, {summary['warning']} warning, "
                    f"{summary['normal']} normal."
                )

    try:
        latest = api_get("/predictions/latest")
    except requests.RequestException as exc:
        show_api_error(exc)
        latest = []

    if not latest:
        st.info("No predictions yet. Press the button above to create them.")
    else:
        latest_df = pd.DataFrame(latest)
        table = latest_df[
            ["engine_id", "predicted_rul", "maintenance_status", "prediction_time"]
        ].copy()
        table["predicted_rul"] = table["predicted_rul"].round(1)
        if "rul_lower" in latest_df.columns and latest_df["rul_lower"].notna().any():
            table["likely_range (80%)"] = (
                latest_df["rul_lower"].round(0).astype("Int64").astype(str)
                + " - "
                + latest_df["rul_upper"].round(0).astype("Int64").astype(str)
            )
        table = table.sort_values("predicted_rul").reset_index(drop=True)

        counts = table["maintenance_status"].value_counts()
        c1, c2, c3 = st.columns(3)
        c1.metric("CRITICAL", int(counts.get("CRITICAL", 0)))
        c2.metric("WARNING", int(counts.get("WARNING", 0)))
        c3.metric("NORMAL", int(counts.get("NORMAL", 0)))

        st.dataframe(style_status(table), hide_index=True)
