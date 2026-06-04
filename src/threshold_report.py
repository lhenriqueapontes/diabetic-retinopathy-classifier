from pathlib import Path
import argparse
import pandas as pd


def build_report(predictions='reports/predictions.csv', output_dir='reports'):
    df = pd.read_csv(predictions)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    binary_true = (df['y_true'] > 0).astype(int)
    for threshold in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
        binary_pred = ((df['y_pred'] > 0) & (df['confidence'] >= threshold)).astype(int)
        tp = int(((binary_true == 1) & (binary_pred == 1)).sum())
        tn = int(((binary_true == 0) & (binary_pred == 0)).sum())
        fp = int(((binary_true == 0) & (binary_pred == 1)).sum())
        fn = int(((binary_true == 1) & (binary_pred == 0)).sum())
        sensitivity = tp / (tp + fn) if (tp + fn) else 0
        specificity = tn / (tn + fp) if (tn + fp) else 0
        rows.append({'threshold': threshold, 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 'sensitivity': sensitivity, 'specificity': specificity})
    pd.DataFrame(rows).round(4).to_csv(out / 'threshold_report.csv', index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', default='reports/predictions.csv')
    parser.add_argument('--output-dir', default='reports')
    args = parser.parse_args()
    build_report(args.predictions, args.output_dir)
    print(args.output_dir)


if __name__ == '__main__':
    main()
