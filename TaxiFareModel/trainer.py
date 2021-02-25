# imports
from TaxiFareModel.data import get_data, clean_data
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression


class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""

        distance_pipeline = Pipeline([('distance_features', DistanceTransformer()),
                                      ('scaler', StandardScaler())])

        time_features_pipeline = Pipeline([('time_features', TimeFeaturesEncoder(time_column = 'pickup_date')),
                                           ('ohe', OneHotEncoder())])

        dist_cols = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
        time_cols = ['pickup_datetime']
        prep_pipeline = ColumnTransformer([('dist_pipeline', distance_pipeline, dist_cols),
                                          ('time_pipeline', time_features_pipeline, time_cols)])

        final_pipeline = Pipeline([('preprocessor', prep_pipeline),
                                   ('model', LinearRegression())])

        self.pipeline = final_pipeline

    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        rmse = compute_rmse(self.pipeline.predict(X_test), y_test)
        print(f"RMSE: {rmse}")
        return rmse


if __name__ == "__main__":
    # get data
    df = get_data()

    # clean data
    cleaned_df = clean_data(df)

    # set X and y
    X = cleaned_df.drop(columns = 'fare_amount')
    y = cleaned_df['fare_amount']

    # hold out
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)

    # train
    trainer = Trainer(X_train, y_train)
    trainer.run()

    # evaluate
    trainer.evaluate(X_test, y_test)
