"""Plotly Visualizations styled with the Turku Energia Infographic Palette (Page 14).

Supports both Control-Room Dark Mode (`dark_mode=True`, default) and Clean Light Mode (`dark_mode=False`):
- Dark Mode Canvas: #0D1E2E card surface / #091622 plot area with #162E45 gridlines
- Light Mode Canvas: #FFFFFF with subtle #EAF7FB gridlines
- P50 Median Forecast: Luminous Ice-White #F0FAFC (Dark Mode) / Deep Navy #0F4B73 (Light Mode)
- P10–P90 80% Uncertainty Fan: Turquoise / Ice Blue (#46BEDC / rgba(70, 190, 220, 0.22))
- Day-Ahead Spot Price Anchor: Turku Energia Orange #F59B00
- Realized History (Usable <= t): Fresh Lime Green #AAC800
- Previous Origin Forecast (t - 1h): Muted Slate / Gray
- High Price Spike (> Spot + 50): Berry Pink #E6508C
- Low Price Spike (< Spot - 50): Infographic Plum #C85AB2 / #A03C8C
- mFRR Up / Down / None: #F59B00 (Up), #46BEDC (Down), #35526E / #E6E1DC (None)
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from apps.theme import TE_COLORS, format_driver_name

FONT_FAMILY = "'Fira Sans', 'Franklin Gothic Book', sans-serif"
HEADING_FONT = "'Fira Sans Extra Condensed', sans-serif"


def _apply_te_layout(
    fig: go.Figure,
    height: int = 420,
    title: str | None = None,
    dark_mode: bool = True,
) -> go.Figure:
    """Apply consistent Turku Energia Dark or Light styling to a Plotly figure."""
    paper_bg = "#0D1E2E" if dark_mode else TE_COLORS["white"]
    plot_bg = "#091622" if dark_mode else TE_COLORS["white"]
    text_col = "#E6F3F8" if dark_mode else TE_COLORS["navy"]
    grid_col = "#152C42" if dark_mode else "#EAF7FB"
    line_col = "#244766" if dark_mode else TE_COLORS["ice_blue"]
    zero_col = "#2E577B" if dark_mode else TE_COLORS["stone"]
    legend_bg = "rgba(13, 30, 46, 0.88)" if dark_mode else "rgba(255, 255, 255, 0.92)"

    layout_kwargs: dict = dict(
        height=height,
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        font=dict(family=FONT_FAMILY, color=text_col, size=12),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#0B1D2C" if dark_mode else TE_COLORS["navy"],
            font_size=12,
            font_family=FONT_FAMILY,
            font_color="#FFFFFF",
            bordercolor=TE_COLORS["orange"],
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="left",
            x=0,
            bgcolor=legend_bg,
            font=dict(size=11.5, color=text_col),
        ),
        margin=dict(l=48, r=24, t=52 if title else 38, b=38),
    )
    if title:
        layout_kwargs["title"] = dict(
            text=title,
            font=dict(family=HEADING_FONT, size=16, color=text_col),
            x=0.0,
            xanchor="left",
        )
    fig.update_layout(**layout_kwargs)
    fig.update_xaxes(
        showgrid=True,
        gridcolor=grid_col,
        linecolor=line_col,
        tickfont=dict(color=text_col, size=11),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=grid_col,
        zeroline=True,
        zerolinecolor=zero_col,
        linecolor=line_col,
        tickfont=dict(color=text_col, size=11),
    )
    return fig


def build_price_fan_chart(
    origin_df: pd.DataFrame,
    prev_origin_df: pd.DataFrame | None,
    hist_idx: pd.DatetimeIndex,
    hist_imb: pd.Series,
    hist_spot: pd.Series,
    selected_origin: pd.Timestamp,
    show_hindsight: bool = False,
    dark_mode: bool = True,
) -> go.Figure:
    """Build the primary 10-Hour Probabilistic Price Forecast Fan Chart (P10-P50-P90 vs Spot)."""
    fig = go.Figure()

    p50_color = "#F4FAFC" if dark_mode else TE_COLORS["navy"]
    p50_marker_border = TE_COLORS["orange"] if dark_mode else "#FFFFFF"
    prev_color = "#89A8C0" if dark_mode else TE_COLORS["gray"]
    band_fill = "rgba(70, 190, 220, 0.22)" if dark_mode else "rgba(70, 190, 220, 0.20)"
    band_edge = "rgba(70, 190, 220, 0.55)" if dark_mode else "rgba(70, 190, 220, 0.45)"

    # 1. P10-P90 Uncertainty Band (80% Confidence Interval)
    fig.add_trace(
        go.Scatter(
            x=origin_df["delivery_start_utc"],
            y=origin_df["p90"],
            mode="lines",
            line=dict(width=1.1, color=band_edge),
            name="P90 (Upper 90th Percentile)",
            showlegend=False,
            hovertemplate="P90 (Upper Bound): <b>%{y:.2f} €/MWh</b><extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=origin_df["delivery_start_utc"],
            y=origin_df["p10"],
            mode="lines",
            line=dict(width=1.1, color=band_edge),
            fill="tonexty",
            fillcolor=band_fill,
            name="80% Confidence Interval (P10–P90)",
            hovertemplate="P10 (Lower Bound): <b>%{y:.2f} €/MWh</b><extra></extra>",
        )
    )

    # 2. Known Day-Ahead Spot Price (Past 6h + Future 10h)
    spot_full_x = list(hist_idx) + list(origin_df["delivery_start_utc"])
    spot_full_y = list(hist_spot.values) + list(origin_df["spot_price"].values)
    fig.add_trace(
        go.Scatter(
            x=spot_full_x,
            y=spot_full_y,
            mode="lines",
            line=dict(color=TE_COLORS["orange"], width=2.5, dash="dash"),
            name="Day-Ahead Spot Price (Known since D−1 14:00)",
            hovertemplate="Spot Price: <b>%{y:.2f} €/MWh</b><extra></extra>",
        )
    )

    # 3. Realized Imbalance Price before Origin (Usable <= t)
    fig.add_trace(
        go.Scatter(
            x=hist_idx,
            y=hist_imb,
            mode="lines+markers",
            line=dict(color=TE_COLORS["lime"], width=2.6),
            marker=dict(size=5, color=TE_COLORS["lime"]),
            name="Realized Imbalance Price (Usable ≤ t)",
            hovertemplate="Realized (≤ t): <b>%{y:.2f} €/MWh</b><extra></extra>",
        )
    )

    # 4. Previous Origin P50 Forecast (t - 1h)
    if prev_origin_df is not None and len(prev_origin_df):
        fig.add_trace(
            go.Scatter(
                x=prev_origin_df["delivery_start_utc"],
                y=prev_origin_df["p50"],
                mode="lines",
                line=dict(color=prev_color, width=1.9, dash="dot"),
                name="Previous Origin P50 (t − 1h)",
                hovertemplate="P50 (t − 1h): <b>%{y:.2f} €/MWh</b><extra></extra>",
            )
        )

    # 5. Current Median Forecast (P50)
    fig.add_trace(
        go.Scatter(
            x=origin_df["delivery_start_utc"],
            y=origin_df["p50"],
            mode="lines+markers",
            line=dict(color=p50_color, width=3.4),
            marker=dict(size=6, color=p50_color, line=dict(width=1.6, color=p50_marker_border)),
            name="AI Median Forecast (P50)",
            hovertemplate="<b>P50 Forecast: %{y:.2f} €/MWh</b><extra></extra>",
        )
    )

    # 6. Optional Hindsight Actuals
    if show_hindsight and "actual_imbalance_price" in origin_df.columns and origin_df["actual_imbalance_price"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=origin_df["delivery_start_utc"],
                y=origin_df["actual_imbalance_price"],
                mode="lines+markers",
                line=dict(color=TE_COLORS["berry"], width=2.3),
                marker=dict(size=5, symbol="diamond", color=TE_COLORS["berry"]),
                name="Actual Imbalance Price (Hindsight)",
                hovertemplate="Actual Price (Hindsight): <b>%{y:.2f} €/MWh</b><extra></extra>",
            )
        )

    # Vertical Origin Marker (NOW = t)
    origin_dt = selected_origin.to_pydatetime()
    fig.add_vline(
        x=origin_dt,
        line_width=2.2,
        line_dash="solid",
        line_color=TE_COLORS["orange"] if dark_mode else TE_COLORS["navy"],
    )
    fig.add_annotation(
        x=origin_dt,
        y=1.0,
        yref="paper",
        text=f"  NOW (Origin t = {selected_origin.strftime('%H:%M UTC')})",
        showarrow=False,
        xanchor="left",
        yanchor="top",
        font=dict(
            family=HEADING_FONT,
            size=12,
            color="#FFFFFF" if dark_mode else TE_COLORS["navy"],
        ),
        bgcolor="#132A40" if dark_mode else "#EAF7FB",
        bordercolor=TE_COLORS["orange"] if dark_mode else TE_COLORS["ice_blue"],
        borderpad=4,
    )

    _apply_te_layout(fig, height=430, dark_mode=dark_mode)
    fig.update_yaxes(title_text="Price (€/MWh)")
    return fig


def build_spike_probability_chart(
    origin_df: pd.DataFrame,
    alert_threshold: float = 0.30,
    dark_mode: bool = True,
) -> go.Figure:
    """Build the Exceptional Spike Probability bar chart (>Spot+50 and <Spot-50)."""
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=origin_df["delivery_start_utc"],
            y=origin_df["p_spike_high"],
            name="High-Price Spike (> Spot + 50 €)",
            marker_color=TE_COLORS["berry"],
            opacity=0.92,
            hovertemplate="P(Spike High): <b>%{y:.1%}</b><extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=origin_df["delivery_start_utc"],
            y=origin_df["p_spike_low"],
            name="Low / Negative Spike (< Spot − 50 €)",
            marker_color="#C85AB2" if dark_mode else TE_COLORS["plum"],
            opacity=0.92,
            hovertemplate="P(Spike Low): <b>%{y:.1%}</b><extra></extra>",
        )
    )

    fig.add_hline(
        y=alert_threshold,
        line_dash="dash",
        line_width=2,
        line_color=TE_COLORS["orange"],
        annotation_text="30% Alert Threshold",
        annotation_position="top right",
        annotation_font=dict(
            color="#FFFFFF" if dark_mode else TE_COLORS["navy"],
            size=11,
            family=HEADING_FONT,
        ),
    )

    _apply_te_layout(fig, height=300, dark_mode=dark_mode)
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Probability", range=[0, 1.02], tickformat=".0%")
    return fig


def build_direction_probability_chart(
    origin_df: pd.DataFrame,
    dark_mode: bool = True,
) -> go.Figure:
    """Build the mFRR Balancing Direction Probability chart (Up / Down / None)."""
    fig = go.Figure()

    none_color = "#29455E" if dark_mode else TE_COLORS["stone"]

    fig.add_trace(
        go.Bar(
            x=origin_df["delivery_start_utc"],
            y=origin_df["p_up"],
            name="Up-Regulation (Deficit / +Price)",
            marker_color=TE_COLORS["orange"],
            hovertemplate="P(mFRR Up): <b>%{y:.1%}</b><extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=origin_df["delivery_start_utc"],
            y=origin_df["p_down"],
            name="Down-Regulation (Surplus / −Price)",
            marker_color=TE_COLORS["turquoise"],
            hovertemplate="P(mFRR Down): <b>%{y:.1%}</b><extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=origin_df["delivery_start_utc"],
            y=origin_df["p_none"],
            name="No Dominance (Price ≈ Spot)",
            marker_color=none_color,
            hovertemplate="P(mFRR None): <b>%{y:.1%}</b><extra></extra>",
        )
    )

    _apply_te_layout(fig, height=300, dark_mode=dark_mode)
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title_text="Share (Sum = 100%)", range=[0, 1.02], tickformat=".0%")
    return fig


def build_physical_drivers_chart(
    origin_df: pd.DataFrame,
    dark_mode: bool = True,
) -> go.Figure:
    """Build Day-Ahead Consumption, Wind & Residual Load trajectory chart."""
    fig = go.Figure()
    cons_color = TE_COLORS["turquoise"] if dark_mode else TE_COLORS["navy"]

    fig.add_trace(
        go.Scatter(
            x=origin_df["delivery_start_utc"],
            y=origin_df["cons_da"],
            mode="lines",
            name="Consumption Forecast DA (MW)",
            line=dict(color=cons_color, width=2.6),
            hovertemplate="Consumption DA: <b>%{y:,.0f} MW</b><extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=origin_df["delivery_start_utc"],
            y=origin_df["wind_da"],
            mode="lines",
            name="Wind Generation Forecast DA (MW)",
            line=dict(color=TE_COLORS["lime"], width=2.6),
            hovertemplate="Wind DA: <b>%{y:,.0f} MW</b><extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=origin_df["delivery_start_utc"],
            y=origin_df["residual_load_da"],
            mode="lines",
            name="Residual Load (Load − Wind − Solar, MW)",
            line=dict(color=TE_COLORS["orange"], width=2.8, dash="dot"),
            hovertemplate="Residual Load: <b>%{y:,.0f} MW</b><extra></extra>",
        )
    )
    _apply_te_layout(fig, height=360, dark_mode=dark_mode)
    fig.update_yaxes(title_text="Power (MW)")
    return fig


def build_shap_drivers_bar_chart(
    origin_df: pd.DataFrame,
    dark_mode: bool = True,
) -> go.Figure | None:
    """Build horizontal bar chart of weighted TreeSHAP Top Drivers across the 40 steps."""
    drv_records = []
    for s in origin_df["top_drivers"].dropna().astype(str):
        for rank, d in enumerate(s.split(";")[:3], start=1):
            clean = d.strip()
            if clean:
                drv_records.append({
                    "driver": format_driver_name(clean),
                    "weight": 4 - rank,
                })
    if not drv_records:
        return None

    drv_df = (
        pd.DataFrame(drv_records)
        .groupby("driver", as_index=False)["weight"]
        .sum()
        .sort_values("weight", ascending=True)
    )
    colorscale = (
        [[0.0, "#1E4B6E"], [0.5, TE_COLORS["turquoise"]], [1.0, TE_COLORS["orange"]]]
        if dark_mode
        else [[0.0, TE_COLORS["ice_blue"]], [0.5, TE_COLORS["turquoise"]], [1.0, TE_COLORS["navy"]]]
    )
    fig = go.Figure(
        go.Bar(
            x=drv_df["weight"],
            y=drv_df["driver"],
            orientation="h",
            marker=dict(
                color=drv_df["weight"],
                colorscale=colorscale,
            ),
            hovertemplate="<b>%{y}</b><br>Weighted SHAP Score: %{x}<extra></extra>",
        )
    )
    _apply_te_layout(fig, height=360, dark_mode=dark_mode)
    fig.update_xaxes(title_text="Weighted TreeSHAP Dominance Across 40 Steps")
    return fig


def build_horizon_benchmark_chart(
    val_h: pd.DataFrame,
    base_h: pd.DataFrame,
    dark_mode: bool = True,
) -> go.Figure:
    """Build side-by-side MAE and Pinball loss horizon comparison vs. official baseline."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "MAE on P50 by Forecast Horizon Hour (€/MWh)",
            "Pinball Loss (P10/P90 Calibration) by Hour",
        ),
        horizontal_spacing=0.1,
    )
    base_color = "#89A8C0" if dark_mode else TE_COLORS["gray"]
    suite_mae_color = "#F4FAFC" if dark_mode else TE_COLORS["navy"]

    fig.add_trace(
        go.Scatter(
            x=base_h["horizon_hour"],
            y=base_h["mae_p50"],
            mode="lines+markers",
            name="Official Baseline (MAE)",
            line=dict(color=base_color, width=2, dash="dash"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=val_h["horizon_hour"],
            y=val_h["mae_p50"],
            mode="lines+markers",
            name="Our LightGBM Suite (MAE)",
            line=dict(color=suite_mae_color, width=3.2),
            marker=dict(size=7, color=TE_COLORS["orange"]),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=base_h["horizon_hour"],
            y=base_h["calibration_pinball_mean"],
            mode="lines+markers",
            name="Official Baseline (Pinball)",
            line=dict(color=base_color, width=2, dash="dash"),
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=val_h["horizon_hour"],
            y=val_h["calibration_pinball_mean"],
            mode="lines+markers",
            name="Our LightGBM Suite (Pinball)",
            line=dict(color=TE_COLORS["turquoise"], width=3.2),
            marker=dict(size=7, color=TE_COLORS["orange"]),
        ),
        row=1,
        col=2,
    )
    _apply_te_layout(fig, height=370, dark_mode=dark_mode)
    fig.update_xaxes(title_text="Forecast Horizon (Hour 1 to 10)", dtick=1)
    fig.update_yaxes(title_text="€/MWh (lower = better)", row=1, col=1)
    fig.update_yaxes(title_text="Pinball Loss (lower = better)", row=1, col=2)
    return fig
