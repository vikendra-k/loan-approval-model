import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(csv_path):
    df = pd.read_csv(csv_path)

    # handle missing values (example)
    df.ffill(inplace=True)


    # encode categorical columns
    label_encoder = LabelEncoder()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = label_encoder.fit_transform(df[col])

    # separate features and target
    X = df.drop("Loan_Approved", axis=1)
    y = df["Loan_Approved"]

    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    return X_train, X_test, y_train, y_test
