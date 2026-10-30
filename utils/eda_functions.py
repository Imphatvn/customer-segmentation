import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_missing_values(df, width=1000, height=550):
    """Show the number and percentage of missing values for each column."""
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    missing_pct = (missing / len(df) * 100).round(1)

    missing_df = pd.DataFrame({
        "missing_values": missing,
        "missing_percentage": missing_pct,
    }).reset_index(names="column")

    fig = px.bar(
        missing_df,
        x="missing_percentage",
        y="column",
        orientation="h",
        text="missing_values",
        color="missing_percentage",
        color_continuous_scale="Reds",
        width=width,
        height=height,
    )

    fig.update_traces(textposition="outside")
    fig.update_layout(
        title="Missing Values by Feature",
        xaxis_title="Missing Values (%)",
        yaxis_title=None,
        coloraxis_showscale=False,
        yaxis=dict(categoryorder="total ascending"),
    )

    return fig


def plot_numeric_distributions(df, columns, width=1200, height=900):
    """Show the distribution of numerical features using histograms."""
    columns = [column for column in columns if column in df.columns]
    rows = int(np.ceil(len(columns) / 3))

    fig = make_subplots(
        rows=rows,
        cols=3,
        subplot_titles=[column.replace("_", " ").title() for column in columns],
    )

    for i, column in enumerate(columns):
        fig.add_trace(
            go.Histogram(x=df[column], marker_color="#4C78A8", showlegend=False),
            row=i // 3 + 1,
            col=i % 3 + 1,
        )

    fig.update_layout(
        title="Distribution of Numerical Features",
        width=width,
        height=height,
        bargap=0.05,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    return fig


def plot_age_by_gender(df, gender_column="customer_gender", width=900, height=550):
    """Compare the age distribution of male and female customers."""
    df_plot = df.copy()
    df_plot[gender_column] = df_plot[gender_column].str.title()

    fig = px.violin(
        df_plot,
        x=gender_column,
        y="age",
        color=gender_column,
        box=True,
        points="outliers",
        color_discrete_map={"Male": "#4C78A8", "Female": "#D356E4"},
        width=width,
        height=height,
    )

    fig.update_layout(
        title="Age Distribution by Gender",
        xaxis_title="Gender",
        yaxis_title="Age",
        legend_title="Gender",
    )

    return fig


def plot_category_counts(df, column, width=1000, height=500):
    """Show the number of customers in each category of a feature."""
    counts = (
        df[column]
        .value_counts(dropna=False)
        .rename_axis(column)
        .reset_index(name="customers")
    )
    counts[column] = counts[column].astype(str)

    fig = px.bar(
        counts,
        x=column,
        y="customers",
        text="customers",
        color="customers",
        color_continuous_scale="Blues",
        width=width,
        height=height,
    )

    fig.update_traces(textposition="outside")
    fig.update_layout(
        title=f"Customers by {column.replace('_', ' ').title()}",
        xaxis_title=column.replace("_", " ").title(),
        yaxis_title="Number of Customers",
        coloraxis_showscale=False,
    )

    return fig


def plot_complaints_and_loyalty_over_time(df, width=1000, height=550):
    """Compare complaint and loyalty-card rates across first-transaction years."""
    df_plot = df.copy()

    grouped = (
        df_plot.groupby("year_first_transaction")
        .agg(
            complaint_rate=("complained", "mean"),
            loyalty_rate=("loyalty_card", "mean"),
            customers=("customer_id", "count"),
        )
        .reset_index()
    )

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_bar(
        x=grouped["year_first_transaction"],
        y=grouped["customers"],
        name="Customers",
        marker_color="#D9D9D9",
        opacity=0.5,
    )

    fig.add_scatter(
        x=grouped["year_first_transaction"],
        y=grouped["complaint_rate"],
        name="Complaint Rate",
        mode="lines+markers",
        line=dict(color="#E45756", width=3),
        secondary_y=True,
    )

    fig.add_scatter(
        x=grouped["year_first_transaction"],
        y=grouped["loyalty_rate"],
        name="Loyalty-Card Rate",
        mode="lines+markers",
        line=dict(color="#4C78A8", width=3),
        secondary_y=True,
    )

    fig.update_layout(
        title="Complaints, Loyalty Cards and Customer Volume Over Time",
        xaxis_title="Year of First Transaction",
        width=width,
        height=height,
        legend=dict(orientation="h", y=1.12, x=0),
    )
    fig.update_yaxes(title_text="Number of Customers", secondary_y=False)
    fig.update_yaxes(title_text="Rate", tickformat=".0%", secondary_y=True)

    return fig


def plot_promotion_by_degree(df, width=900, height=550):
    """Compare promotion purchasing between customers with and without a degree."""
    df_plot = df.copy()
    df_plot["degree_label"] = df_plot["degree"].map({0: "No Degree", 1: "Has Degree"})

    fig = px.box(
        df_plot,
        x="degree_label",
        y="pct_prod_in_promotion",
        color="degree_label",
        points="outliers",
        color_discrete_map={"No Degree": "#E45756", "Has Degree": "#4C78A8"},
        width=width,
        height=height,
    )

    fig.update_layout(
        title="Promotion Purchases by Academic Degree",
        xaxis_title=None,
        yaxis_title="Percentage of Products Bought on Promotion",
        showlegend=False,
    )
    fig.update_yaxes(tickformat=".0%")

    return fig


def plot_spend_correlation(df, spend_columns, width=850, height=750):
    """Show correlations between annual spending categories."""
    correlation = df[spend_columns].corr()

    labels = [column.replace("spend_per_year_", "").replace("_", " ").title() for column in spend_columns]
    correlation.index = labels
    correlation.columns = labels

    fig = px.imshow(
        correlation,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
        width=width,
        height=height,
    )

    fig.update_layout(
        title="Correlation Between Annual Spending Categories",
        xaxis_title=None,
        yaxis_title=None,
    )

    return fig


def plot_spend_by_age_group(df, spend_columns, width=1100, height=550):
    """Compare average annual spending across age groups."""
    df_plot = df.copy()
    df_plot["age_group"] = pd.cut(
        df_plot["age"],
        bins=[22, 35, 50, 65, np.inf],
        labels=["23-35", "36-50", "51-65", "66+"],
    )

    grouped = (
        df_plot.groupby("age_group", observed=True)[spend_columns]
        .mean()
        .reset_index()
    )

    fig = go.Figure()

    for column in spend_columns:
        fig.add_trace(go.Bar(
            x=grouped["age_group"],
            y=grouped[column],
            name=column.replace("spend_per_year_", "").replace("_", " ").title(),
        ))

    fig.update_layout(
        title="Average Annual Spending by Age Group",
        xaxis_title="Age Group",
        yaxis_title="Average Annual Spending",
        barmode="group",
        width=width,
        height=height,
        legend_title="Spending Category",
    )

    return fig
