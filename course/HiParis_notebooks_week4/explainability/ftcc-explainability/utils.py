import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings("ignore")


def prepared_housing(housing):
    X = housing.drop("median_house_value", axis=1)
    y = housing["median_house_value"]

    rng = np.random.RandomState(seed=110)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )
    numerical = X_train.drop("ocean_proximity", axis=1)

    # column index
    rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

    class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
        def __init__(self, add_bedrooms_per_room=True):  # no *args or **kargs
            self.add_bedrooms_per_room = add_bedrooms_per_room

        def fit(self, X, y=None):
            return self  # nothing else to do

        def transform(self, X):
            rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
            population_per_household = (
                X[:, population_ix] / X[:, households_ix]
            )
            if self.add_bedrooms_per_room:
                bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
                return np.c_[
                    X,
                    rooms_per_household,
                    population_per_household,
                    bedrooms_per_room,
                ]
            else:
                return np.c_[X, rooms_per_household, population_per_household]

    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("attribs_adder", CombinedAttributesAdder()),
        ]
    )

    X_train_num_tr = num_pipeline.fit_transform(numerical)

    num_attribs = list(numerical)
    cat_attribs = ["ocean_proximity"]

    full_pipeline = ColumnTransformer(
        [
            ("num", num_pipeline, num_attribs),
            ("cat", OneHotEncoder(), cat_attribs),
        ]
    )

    X_train_prepared = full_pipeline.fit_transform(X_train)
    X_test_prepared = full_pipeline.transform(X_test)

    cols_added = [
        "rooms_per_household",
        "population_per_household",
        "bedrooms_per_room",
    ]
    cols_one_hot = list(
        full_pipeline.transformers_[1][1].get_feature_names_out()
    )

    X_train_prepared = pd.DataFrame(
        X_train_prepared, columns=num_attribs + cols_added + cols_one_hot
    )

    X_test_prepared = pd.DataFrame(
        X_test_prepared, columns=num_attribs + cols_added + cols_one_hot
    )

    return X_train_prepared, X_test_prepared, y_train, y_test
