from sklearn.model_selection import cross_validate, train_test_split
from sklearn.tree import DecisionTreeClassifier

from e1_create_dataset import create_classification_dataset
from e3_metrics import classification_scores

# Recommended reading: https://scikit-learn.org/stable/modules/cross_validation.html

def holdout():
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
    # Create a dataset and a model
    _, X, y = create_classification_dataset()

    # Train and evaluate the model
    # See other scoring metrics here: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    cv_results = cross_validate(DecisionTreeClassifier(), X, y, cv=5, scoring='accuracy', return_train_score=True)
    print('\nTraining score:', cv_results['train_score'].mean())
    print('Testing scores:', cv_results['test_score'])
    print('Testing scores (averaged):', cv_results['test_score'].mean())


if __name__ == '__main__':
    holdout()
    cross_validation()

    # Which scores are better?
    # Why do the cross-validation scores vary?
    # How do these scores compare to the previous exercise?
