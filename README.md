# ml-examples

Machine Learning Examples

## Concepts

Machine Learning:

- Training, validation and test sets
- Hyperparameters
- Holdout, Cross-Validation, Nested Cross-Validation
- Leakage
- Underfitting, overfitting
- Curse of dimensionality
- Regularization
- Data augmentation

Programming:

- Object Oriented Programming
- Git

## Tips

Machine Learning:

- Collect any metrics that you think may be useful. You don't have  to use them all, but it is easier than repeating experiments because you forgot to record something important.
- Nested Cross-Validation is better than Cross-Validation, Cross-Validation is better than Holdout
- With neural networks, try different architectures/optimizers before tuning subtler hyperparameters like regularization
- If you have good performance on the validation set, but bad performance on test set, then use a bigger validation set
- If you get poor real-world performance, you may need a different/bigger test set or cost function
- Apply early stopping after you have some models performing somewhat well to save time
- Scaling/normalization should come before PCA
- If the train/test curves are not converging:
  - Increase the complexity - if you think you are underfitting due to high bias
    - Move to a more complex model, (e.g. polynomial of higher degree, or a different kind of model, such as a nonparametric one that makes fewer assumptions about the target function)
    - Move to a more complex variant of your current model
  - Add more useful features
  - Remove useless features
- If the curves are converging too slowly:
  - Decrease the complexity - if you think you are overfitting due to high variance
    - Move to a less complex model (e.g. polynomial of a lower degree or a different kind of model that makes more assumptions about the target function)
    - Move to a less complex variant of your current model (e.g. regularization, smoothing)
  - Add more instances
  - Remove noisy samples

Programming:

- Try to limit code repetition as much as possible
- Back up code, preferably in a repository using a website such as GitHub
- Use informative function/class/variable names
