from sklearn.preprocessing import StandardScaler

def scale_features(X_train, X_test):
    """
    Scales features using StandardScaler.
    Fit on training data, transform both train and test.
    """
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler

'''from sklearn.linear_model import LogisticRegression

def train_logistic_regression(X_train, y_train):
    """
    Trains a Logistic Regression classifier.
    """
    model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs"
    )
    
    model.fit(X_train, y_train)
    return model'''

from sklearn.linear_model import LogisticRegression

def train_logistic_regression(X_train, y_train):
    """
    Trains a cost-sensitive Logistic Regression classifier.
    """
    model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        class_weight={0: 1, 1: 2}  # penalize wrong approvals
    )
    
    model.fit(X_train, y_train)
    return model


