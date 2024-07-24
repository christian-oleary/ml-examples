"""Testing model performance using Holdout and Cross-Validation (CV)."""

from sklearn.model_selection import cross_validate, train_test_split
from sklearn.tree import DecisionTreeClassifier

from ml.e1_create_dataset import create_classification_dataset
from ml.e3_metrics import classification_scores

# Recommended reading:
# https://scikit-learn.org/stable/modules/cross_validation.html

# For time series data, see:
# https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-of-time-series-data


def holdout():
    """Train a model using a training set and a test set."""
    # Create a dataset and a model
    _, X, y = create_classification_dataset()
    model = DecisionTreeClassifier()

    # Train using a holdout methodology:
    # 1. Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # 2. Train the model using the training set
    model.fit(X_train, y_train)

    # 3. Make predictions using the trained model on the test set
    preds = model.predict(X_test)

    # 4. Evaluate the model
    print('\nTraining score:', classification_scores(y_train, model.predict(X_train))['accuracy'])
    print('Testing score:', classification_scores(y_test, preds)['accuracy'])


def cross_validation():
    """Train a model using Cross Validation."""
    # Create a dataset and a model
    _, X, y = create_classification_dataset()

    # Train and evaluate the model
    # See other scoring metrics here: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    cv_results = cross_validate(
        DecisionTreeClassifier(),
        X,
        y,
        cv=5,  # Use 5 folds
        scoring='accuracy',  # Measure model performance using accuracy
        return_train_score=True,  # Record results
    )
    print('\nAverage validation score:', cv_results['train_score'].mean())
    print('Testing scores:', cv_results['test_score'])
    print('Testing scores (averaged):', cv_results['test_score'].mean())


def run():
    """Run this exercise."""
    holdout()
    cross_validation()

    # Which scores are better, holdout or CV?
    # Why do the cross-validation scores vary, i.e. the 5 scores of the different 'folds'?
    # How do these scores compare to the previous exercise?


if __name__ == '__main__':
    run()
