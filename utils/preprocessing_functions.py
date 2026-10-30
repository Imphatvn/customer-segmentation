import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor


# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

REFERENCE_YEAR = 2025

SPEND_COLUMNS = [
    "lifetime_spend_groceries",
    "lifetime_spend_electronics",
    "lifetime_spend_vegetables",
    "lifetime_spend_nonalcohol_drinks",
    "lifetime_spend_alcohol_drinks",
    "lifetime_spend_meat",
    "lifetime_spend_fish",
    "lifetime_spend_hygiene",
    "lifetime_spend_petfood",
    "lifetime_spend_videogames",
]

DEGREE_TITLES = ["Phd.", "Msc.", "Mst.", "Bsc."]


# ------------------------------------------------------------------------------
# Cleaning
# ------------------------------------------------------------------------------

def fix_promotion_percentage(df, column="percentage_of_products_bought_promotion"):
    """Correct promotion percentages outside the expected 0-1 range.

    Negative values are assumed to have the wrong sign. Values between 1 and 10
    are assumed to have a misplaced decimal point. Values above 10 cannot be
    interpreted reliably and are set to missing.
    """
    df = df.copy()

    df[column] = df[column].abs()

    df.loc[df[column].between(1, 10, inclusive="right"), column] = (
        df.loc[df[column].between(1, 10, inclusive="right"), column] / 10
    )
    df.loc[df[column] > 1, column] = np.nan

    df = df.rename(columns={column: "pct_prod_in_promotion"})

    return df


def correct_first_transaction_year(df, year_column="year_first_transaction"):
    """Correct impossible first-transaction years.

    A first transaction after the reference year is replaced with the reference
    year minus one. This avoids negative customer lifetimes.
    """
    df = df.copy()

    invalid_year = df[year_column] > REFERENCE_YEAR
    df.loc[invalid_year, year_column] = REFERENCE_YEAR - 1

    return df


# ------------------------------------------------------------------------------
# Feature engineering
# ------------------------------------------------------------------------------

def create_age_feature(df, birthdate_column="customer_birthdate", reference_year=REFERENCE_YEAR):
    """Create an age feature from the customer birth date.

    Missing birth dates are imputed with the median age.
    """
    df = df.copy()
    df[birthdate_column] = pd.to_datetime(df[birthdate_column], format="%m/%d/%Y %I:%M %p", errors="coerce")

    age = reference_year - df[birthdate_column].dt.year
    df["age"] = age.fillna(age.median()).astype(int)

    return df.drop(columns=[birthdate_column])


def extract_degree_feature(df, name_column="customer_name"):
    """Create a binary feature indicating whether the customer has a degree."""
    df = df.copy()

    df["degree"] = (
        df[name_column]
        .str.contains("|".join(DEGREE_TITLES), case=False, na=False)
        .astype(int)
    )

    return df.drop(columns=[name_column])


def create_loyalty_card_feature(df, loyalty_column="loyalty_card_number"):
    """Create a binary feature indicating whether the customer has a loyalty card.

    A missing card number is interpreted as the customer not having a card.
    """
    df = df.copy()
    df["loyalty_card"] = df[loyalty_column].notna().astype(int)

    return df.drop(columns=[loyalty_column])


def create_complaint_feature(df, complaints_column="number_complaints"):
    """Create a binary feature indicating whether the customer has complained.

    Missing complaint values are interpreted as no registered complaint.
    """
    df = df.copy()
    df[complaints_column] = df[complaints_column].fillna(0)
    df["complained"] = (df[complaints_column] > 0).astype(int)

    return df.drop(columns=[complaints_column])


def create_children_feature(df):
    """Combine children and teenagers at home into one dependants feature.

    Missing values are interpreted as no children or teenagers at home.
    """
    df = df.copy()

    df["kids_home"] = df["kids_home"].fillna(0)
    df["teens_home"] = df["teens_home"].fillna(0)
    df["child_home"] = df["kids_home"] + df["teens_home"]

    return df.drop(columns=["kids_home", "teens_home"])


def encode_gender(df, gender_column="customer_gender"):
    """Encode gender as a binary male indicator.

    Unknown or missing gender values are encoded as zero.
    """
    df = df.copy()
    df["male"] = df[gender_column].str.lower().eq("male").astype(int)

    return df.drop(columns=[gender_column])


def create_time_category_feature(df, hour_column="typical_hour"):
    """Create a categorical feature for the customer's typical shopping period.

    Missing hours are imputed with the most frequent value. The periods are
    morning (0), afternoon (1) and night (2).
    """
    df = df.copy()
    df[hour_column] = df[hour_column].fillna(df[hour_column].mode()[0])

    df["typical_time"] = pd.cut(
        df[hour_column],
        bins=[5, 11, 17, 24],
        labels=[0, 1, 2],
        include_lowest=True,
    ).astype(int)

    return df.drop(columns=[hour_column])


def create_customer_lifetime_feature(df, year_column="year_first_transaction", reference_year=REFERENCE_YEAR):
    """Create the number of years since the customer's first transaction."""
    df = df.copy()
    df["customer_lifetime"] = reference_year - df[year_column]

    return df


def create_annual_spend_features(df):
    """Convert lifetime spending features into average annual spending features.

    Customers with zero or negative lifetime are set to missing because their
    annual spending cannot be calculated reliably.
    """
    df = df.copy()

    valid_lifetime = df["customer_lifetime"] > 0

    for column in SPEND_COLUMNS:
        if column in df.columns:
            new_column = column.replace("lifetime_spend", "spend_per_year")
            df[new_column] = np.where(
                valid_lifetime,
                df[column] / df["customer_lifetime"],
                np.nan,
            )

    return df


# ------------------------------------------------------------------------------
# Imputation
# ------------------------------------------------------------------------------

def impute_spend_with_knn(df, target_columns, n_neighbors=5):
    """Impute missing annual spending values using K-Nearest Neighbours.

    All numeric spending features are used as predictors. The values are scaled
    before applying KNN so that features with larger ranges do not dominate the
    distance calculation.
    """
    df = df.copy()

    spend_columns = [
        column for column in df.columns
        if column.startswith("spend_per_year_")
    ]
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(df[spend_columns])

    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_values = imputer.fit_transform(scaled_values)

    imputed_values = scaler.inverse_transform(imputed_values)
    df[spend_columns] = imputed_values

    return df


def evaluate_vegetable_imputer(df, target_column="spend_per_year_vegetables",leakage_column="lifetime_spend_vegetables", random_state=42):
    """Evaluate the gradient-boosting imputation model on known vegetable values."""
    df_model = df[df[target_column].notna()].copy()

    features = pd.get_dummies(
        df_model.drop(columns=[target_column, leakage_column]),
        drop_first=True,
    )
    target = df_model[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=random_state,
    )

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mse = np.mean((y_test - predictions) ** 2)
    r2 = 1 - mse / np.var(y_test)

    return {"mse": mse, "r2": r2}


def impute_vegetables_with_boosting(df, target_column="spend_per_year_vegetables", leakage_column="lifetime_spend_vegetables", random_state=42):
    """Impute missing vegetable spending with a gradient-boosting regression model.

    The model is trained on all customers with known vegetable spending before
    predicting the missing values.
    """
    df = df.copy()

    features = pd.get_dummies(
        df.drop(columns=[target_column, leakage_column]),
        drop_first=True,
    )
    known = df[target_column].notna()

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=random_state,
    )
    model.fit(features[known], df.loc[known, target_column])

    df.loc[~known, target_column] = model.predict(features[~known])

    return df

def impute_distinct_stores(df,column="distinct_stores_visited"):
    """Fill missing store counts with the most frequent observed count.

    Missing values are treated as unknown, not as one store visited.
    """
    df = df.copy()

    observed = df[column].dropna()

    if observed.empty:
        raise ValueError(
            f"Cannot impute '{column}': all values are missing."
        )

    if (observed < 1).any() or (observed % 1 != 0).any():
        raise ValueError(
            f"'{column}' must contain positive whole-number counts."
        )

    fill_value = observed.mode().iloc[0]
    df[column] = df[column].fillna(fill_value).astype(int)

    return df

# ------------------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------------------

def validate_customer_data(df):
    """Run basic checks before using the data for clustering."""
    checks = {
        "unique_customer_id": df["customer_id"].is_unique,
        "promotion_within_range": df["pct_prod_in_promotion"].between(0, 1).all(),
        "positive_customer_lifetime": (df["customer_lifetime"] >= 0).all(),
        "no_missing_annual_spend": df.filter(like="spend_per_year_").isna().sum().sum() == 0,
    }

    return pd.Series(checks, name="passed")


# ------------------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------------------

def preprocess_customer_data(df):
    """Apply the complete preprocessing pipeline to the customer information data."""
    df = fix_promotion_percentage(df)
    df = correct_first_transaction_year(df)
    df = create_age_feature(df)
    df = extract_degree_feature(df)
    df = create_loyalty_card_feature(df)
    df = create_complaint_feature(df)
    df = create_children_feature(df)
    df = encode_gender(df)
    df = create_time_category_feature(df)
    df = create_customer_lifetime_feature(df)
    df = create_annual_spend_features(df)
    df = impute_distinct_stores(df)
    df = impute_vegetables_with_boosting(df)
    df = impute_spend_with_knn(df, target_columns=None)

    return df
