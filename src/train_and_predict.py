import pandas as pd
from xgboost import XGBClassifier

def main():
    # Load Data
    train = pd.read_csv("data/TRAIN.csv")
    test = pd.read_csv("data/TEST.csv")

    # Separate features and target
    X = train.drop("Class", axis=1)
    y = train["Class"]

    # Define tuned XGBoost model
    model = XGBClassifier(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=7,
        subsample=0.9,
        colsample_bytree=0.9,
        gamma=0.1,
        reg_alpha=0.5,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1
    )

    # Train model on full training data
    model.fit(X, y)

    # Prepare test data
    X_test = test.drop("ID", axis=1)

    # Predict probabilities
    probs = model.predict_proba(X_test)[:, 1]

    # Apply optimized threshold
    threshold = 0.44
    predictions = (probs > threshold).astype(int)

    # Create submission file
    submission = pd.DataFrame({
        "ID": test["ID"],
        "Class": predictions
    })

    submission.to_csv("FINAL.csv", index=False)
    print("FINAL.csv generated successfully.")

if __name__ == "__main__":
    main()