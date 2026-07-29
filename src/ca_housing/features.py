import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

SF_LAT, SF_LON = 37.7749, -122.4194
LA_LAT, LA_LON = 34.0522, -118.2437


class FeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X):
        X = X.copy()

        X["bedrooms_ratio"] = X["AveBedrms"] / X["AveRooms"]

        lat_gap_sf = X["Latitude"] - SF_LAT
        lon_gap_sf = X["Longitude"] - SF_LON
        distance_to_sf = np.sqrt(lat_gap_sf**2 + lon_gap_sf**2)

        lat_gap_la = X["Latitude"] - LA_LAT
        lon_gap_la = X["Longitude"] - LA_LON
        distance_to_la = np.sqrt(lat_gap_la**2 + lon_gap_la**2)

        X["dist_city"] = np.minimum(distance_to_sf, distance_to_la)

        return X


class QuantileClipper(BaseEstimator, TransformerMixin):

    def __init__(self, columns, lower=0.001, upper=0.999):
        self.columns = columns
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        self.bounds_ = {}
        for col in self.columns:
            low_value = X[col].quantile(self.lower)
            high_value = X[col].quantile(self.upper)
            self.bounds_[col] = (low_value, high_value)
        return self

    def transform(self, X):
        X = X.copy()
        for col, (low_value, high_value) in self.bounds_.items():
            X[col] = X[col].clip(low_value, high_value)
        return X