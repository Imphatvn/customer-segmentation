import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def calculate_iqr_bounds(df, columns, multiplier=10):
    """calcualte IQR bounds for the selected spending features."""
    q1 = df[columns].quantile(0.25)
    q3 = df[columns].quantile(0.75)
    iqr = q3 - q1

    bounds = pd.DataFrame({
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_bound": q1 - multiplier * iqr,
        "upper_bound": q3 + multiplier * iqr})

    return bounds

def flag_iqr_outliers(df, bounds):
    """Return an outlier flag for each customer and selected feature."""
    columns = bounds.index.tolist()

    flags = (df[columns].lt(bounds["lower_bound"])
             | df[columns].gt(bounds["upper_bound"]))

    return flags

def plot_spending_distribution(df, column, bounds):
    """Plot spending and the IQR bounds used to flag observations."""
    data = df[column].dropna()

    lower = bounds.loc[column, "lower_bound"]
    upper = bounds.loc[column, "upper_bound"]

    title = (
        column.replace("spend_per_year_", "")
        .replace("_", " ")
        .title())

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.25, 0.75],
        vertical_spacing=0.12,
        subplot_titles=(
            "Distribution summary",
            "Spending distribution and flagging thresholds"))

    fig.add_trace(
        go.Box(
            x=data,
            name=title,
            boxpoints=False,
            marker_color="#4C78A8",
            showlegend=False,),
        row=1,
        col=1)
    
    fig.add_trace(
        go.Histogram(
            x=data,
            nbinsx=60,
            marker_color="#4C78A8",
            showlegend=False,
        ),
        row=2,
        col=1)

    fig.add_vline(
        x=upper,
        line_color="#E45756",
        line_dash="dash",
        annotation_text="Upper IQR bound",
        row=2,
        col=1)

    if lower >= data.min():
        fig.add_vline(
            x=lower,
            line_color="#E45756",
            line_dash="dash",
            annotation_text="Lower IQR bound",
            row=2,
            col=1)

    fig.update_layout(
        title=f"Annual Spending — {title}",
        template="plotly_white",
        height=550,
        autosize=True,
        margin=dict(l=70, r=40, t=100, b=60),
        bargap=0.05,
    )

    fig.update_xaxes(
        title_text="Annual spending",
        tickformat=",.0f",
        row=2,
        col=1)

    fig.update_yaxes(
        title_text="Customers",
        row=2,
        col=1,
    )

    return fig
