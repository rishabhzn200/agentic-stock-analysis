from sklearn.ensemble import RandomForestClassifier
import joblib

MODEL_PATH = "models/stock_model.pkl"


def train_model(df, features, model_path=MODEL_PATH):
    """
    Train a RandomForestClassifier on the given dataframe and feature columns.
    """
    X = df[features]
    y = df["Target"]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, model_path)
    return model


def load_model(model_path=MODEL_PATH):
    """
    Load a trained model from disk.
    """
    return joblib.load(model_path)


def predict(model, X_latest):
    """
    Predict class for the latest feature row (or rows).
    """
    return model.predict(X_latest)
