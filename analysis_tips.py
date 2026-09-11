"""
End-to-end example analysis using seaborn's `tips` dataset.
Performs EDA, feature correlation analysis, and a simple prediction (regression)
with Linear Regression and RandomForest. Outputs a Markdown report and figures.
"""
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sns.set(style="whitegrid")


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def save_fig(fig, outdir, name):
    path = os.path.join(outdir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(outdir='eda_output_tips'):
    ensure_dir(outdir)
    df = sns.load_dataset('tips')

    # Basic info
    df.to_csv(os.path.join(outdir, 'tips_full.csv'), index=False)
    with open(os.path.join(outdir, 'dataset_info.txt'), 'w') as f:
        f.write(f'Rows: {df.shape[0]}, Columns: {df.shape[1]}\n')
        f.write('\nColumns and types:\n')
        f.write(str(df.dtypes))

    # Descriptive stats
    desc = df.describe(include='all').transpose()
    desc.to_csv(os.path.join(outdir, 'descriptive_stats.csv'))

    # Missing
    missing = df.isnull().sum()
    missing.to_csv(os.path.join(outdir, 'missing_values.csv'))

    # Numeric EDA
    numeric = df.select_dtypes(include=[np.number])
    hist_dir = os.path.join(outdir, 'histograms')
    ensure_dir(hist_dir)
    for col in numeric.columns:
        fig = plt.figure(figsize=(6,4))
        sns.histplot(numeric[col].dropna(), kde=True)
        plt.title(f'Histogram: {col}')
        save_fig(fig, hist_dir, f'hist_{col}.png')

    box_dir = os.path.join(outdir, 'boxplots')
    ensure_dir(box_dir)
    for col in numeric.columns:
        fig = plt.figure(figsize=(6,4))
        sns.boxplot(x=numeric[col].dropna())
        plt.title(f'Boxplot: {col}')
        save_fig(fig, box_dir, f'box_{col}.png')

    # Correlation
    corr = numeric.corr()
    corr.to_csv(os.path.join(outdir, 'correlation_matrix.csv'))
    fig = plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True)
    save_fig(fig, outdir, 'correlation_heatmap.png')

    # Identify top correlated features with target 'tip'
    target = 'tip'
    corr_target = corr[target].drop(target).abs().sort_values(ascending=False)
    corr_target.to_csv(os.path.join(outdir, 'correlation_with_tip.csv'), header=['abs_correlation'])

    # Prepare features: encode categorical variables
    df2 = df.copy()
    df2 = pd.get_dummies(df2, drop_first=True)

    X = df2.drop(columns=[target])
    y = df2[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    models = {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42)
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        results.append({'model': name, 'rmse': rmse, 'mae': mae, 'r2': r2})

        # Save predictions
        pred_df = pd.DataFrame({'y_true': y_test, 'y_pred': preds})
        pred_df.to_csv(os.path.join(outdir, f'predictions_{name}.csv'))

        # Pred vs True plot
        fig = plt.figure(figsize=(6,5))
        sns.scatterplot(x=y_test, y=preds, alpha=0.7)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('True tip')
        plt.ylabel('Predicted tip')
        plt.title(f'{name}: True vs Predicted')
        save_fig(fig, outdir, f'pred_vs_true_{name}.png')

        # Feature importances
        if hasattr(model, 'feature_importances_'):
            fi = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
            fi.to_csv(os.path.join(outdir, f'feature_importances_{name}.csv'))
            fig = plt.figure(figsize=(6,6))
            fi.head(10).plot(kind='bar')
            plt.title(f'Feature importances: {name}')
            save_fig(fig, outdir, f'feature_importances_{name}.png')

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(outdir, 'model_results.csv'), index=False)

    # Generate report
    report = []
    report.append(f'# EDA + Prediction Report: tips dataset')
    report.append('\n## Dataset')
    report.append(f'- Rows: {df.shape[0]}, Columns: {df.shape[1]}')
    report.append('\n## Key Numeric Summaries')
    report.append('- See `descriptive_stats.csv`')
    report.append('\n## Missing Data')
    report.append('- See `missing_values.csv`')
    report.append('\n## Correlations')
    report.append('- Correlation heatmap: `correlation_heatmap.png`')
    report.append('- Correlations with target saved: `correlation_with_tip.csv`')
    report.append('\n## Modeling')
    report.append('- Models trained: LinearRegression, RandomForest')
    report.append('- Results saved: `model_results.csv`')
    report.append('\n### Model performance (lower RMSE/MAE better, higher R2 better)')
    report.append(results_df.to_markdown(index=False))

    # Key influencing factors from RandomForest (if present)
    fi_path = os.path.join(outdir, f'feature_importances_RandomForest.csv')
    if os.path.exists(fi_path):
        fi = pd.read_csv(fi_path, index_col=0, header=None).squeeze()
        top_factors = fi.head(5).to_dict()
        report.append('\n## Key influencing factors (RandomForest)')
        for k, v in top_factors.items():
            report.append(f'- {k}: importance {v:.3f}')

    report_path = os.path.join(outdir, 'REPORT.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(report))

    print('Analysis complete. Outputs in:', outdir)


if __name__ == '__main__':
    main()
