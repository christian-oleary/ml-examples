"""Analyse features and provide feature scores."""

import os

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

from src.e1_create_dataset import create_regression_dataset


# For an example of using feature selection in a pipeline, see example e9_pipelines.py


def feature_scoring_regression(output_dir='feature_scores'):
    """Feature scoring for a regression target."""
    df, X, y = create_regression_dataset()

    os.makedirs(output_dir, exist_ok=True)  # Make a directory called feature_scores

    def plot_scores(scoring_function, filename):
        # Get feature scores
        if scoring_function is None:
            results = []
            for col in X.columns:
                # This handles y having multiple columns
                try:
                    weight = abs(stats.pearsonr(X[col], y.iloc[:, 0])[0])
                except pd.errors.IndexingError:
                    weight = abs(stats.pearsonr(X[col], y)[0])

                results.append({'feature': col, 'score': weight})
            weights = pd.DataFrame(results)
        else:
            selector = SelectKBest(scoring_function, k=3)
            selector.fit_transform(X, y)
            weights = pd.DataFrame({'feature': X.columns, 'score': selector.scores_})

        # Sort by score and save to file
        weights = weights.sort_values('score', ascending=False)
        weights.to_csv(os.path.join(output_dir, f'{filename}.csv'))

        # Create plot
        plot = weights.plot(x='feature', y='score', title=filename, kind='bar')
        fig = plot.get_figure()
        plot.set_xlabel('Features')
        plot.set_ylabel('Score')
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f'{filename}.png'))

    plot_scores(f_regression, 'f_regression')
    plot_scores(mutual_info_regression, 'mutual_info_regression')
    plot_scores(None, 'pearsonr')

    # Finally, create a correlation heatmap
    plt.clf()
    fig, _ = plt.subplots(figsize=(10, 10))
    corr = df.corr()
    heatmap = sns.heatmap(corr, cmap="Blues", annot=True)
    fig = heatmap.get_figure()
    fig.savefig(os.path.join(output_dir, 'heatmap.png'), dpi=400)


def run():
    """Run this exercise."""
    feature_scoring_regression()


if __name__ == '__main__':
    feature_scoring_regression()

    # As an exercise, how would you adapt the code to work with a classification target?
    # Clue: consider the scoring functions
