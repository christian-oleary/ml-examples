import os

import pandas as pd
from scipy import stats
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

from e1_create_dataset import create_regression_dataset


# For an example of using feature selection in a pipeline, see example e9_pipelines.py


def feature_scoring_example(output_dir='feature_scores'):
    _, X, y = create_regression_dataset()

    os.makedirs(output_dir, exist_ok=True) # Make a directory called feature_scores

    def plot_scores(scoring_function, filename):
        if scoring_function == None:
            results = []
            for col in X.columns:
                try:
                    weight = abs(stats.pearsonr(X[col], y.iloc[:,0])[0])
                except:
                    weight = abs(stats.pearsonr(X[col], y)[0])
                results.append({ 'feature': col, 'score': weight })
            weights = pd.DataFrame(results)
        else:
            selector = SelectKBest(scoring_function, k=3)
            _ = selector.fit_transform(X, y)
            weights = pd.DataFrame({'feature': X.columns, 'score': selector.scores_})

        weights.to_csv(os.path.join(output_dir, f'{filename}.csv'))
        plot = weights.plot(title=filename)
        fig = plot.get_figure()
        plot.set_xlabel('Features')
        plot.set_ylabel('Score')
        fig.savefig(os.path.join(output_dir, f'{filename}.png'))

    plot_scores(f_regression, 'f_regression')
    plot_scores(mutual_info_regression, 'mutual_info_regression')
    plot_scores(None, 'pearsonr')

if __name__ == '__main__':
    feature_scoring_example()
