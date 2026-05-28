import numpy as np
import polars as pl
from ucimlrepo import fetch_ucirepo


def get_Census_Income_dataset():
    """
    https://archive.ics.uci.edu/dataset/20/census+income
    Take only "sex, education" d= 17
    Flip the predictors.
    """
    # fetch dataset
    census_income = fetch_ucirepo(id=20)

    # data (as pandas dataframes)
    X = census_income.data.features
    y = census_income.data.targets

    # variable information
    # print(census_income.variables)

    def encode(df: pl.DataFrame, columns) -> pl.DataFrame:
        df_encoded = []

        for col in columns:
            dummies = df.select(col).to_dummies()
            if dummies.shape[1] > 1:
                dummies = dummies[:, 1:]
            df_encoded.append(dummies)
        for col in df.columns:
            if col not in columns:
                df_encoded.append(df.select(col))

        df_encoded = pl.concat(df_encoded, how="horizontal")
        return df_encoded

    # Define the predictors
    X.loc[:, "occupation"] = X[["occupation"]].astype(str)
    df = pl.from_pandas(
        X.filter(["sex", "age", "education", "race", "hours-per-week", "relationship", "marital-status", "occupation"]))
    encoded_df = encode(df, ["sex", "education", "race", "relationship", "marital-status", "occupation"])

    # There are 4 distinct values for y, but two of them can be merged
    y_pl = pl.from_pandas(y['income'].replace("<=50K", "<=50K.").replace(">50K", ">50K."))
    y_pl = y_pl.to_dummies()[:, 0].to_frame()  # 0, 1

    y_pl_np = np.array(y_pl, dtype=bool) * 2 - 1
    encoded_df = y_pl_np * encoded_df  # flipping
    flipped_predictors = np.array(encoded_df)
    return flipped_predictors
