from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def features(path):
    img = Image.open(path).convert('RGB').resize((64, 64))
    arr = np.asarray(img) / 255.0
    return np.r_[arr.mean(axis=(0, 1)), arr.std(axis=(0, 1)), arr.max(axis=(0, 1))]


def train(labels='data/demo_images/labels.csv', output_dir='reports'):
    df = pd.read_csv(labels)
    x = np.vstack([features(p) for p in df['image_path']])
    y = df['label'].values
    x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(
        x, y, df.index, test_size=0.25, random_state=42, stratify=y
    )
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, multi_class='auto'))
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    proba = model.predict_proba(x_test).max(axis=1)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(classification_report(y_test, pred, output_dict=True)).T.to_csv(out / 'classification_report.csv')
    pd.DataFrame(confusion_matrix(y_test, pred)).to_csv(out / 'confusion_matrix.csv', index=False)
    pd.DataFrame({'row_id': idx_test, 'y_true': y_test, 'y_pred': pred, 'confidence': proba}).to_csv(out / 'predictions.csv', index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', default='data/demo_images/labels.csv')
    parser.add_argument('--output-dir', default='reports')
    args = parser.parse_args()
    train(args.labels, args.output_dir)
    print(args.output_dir)


if __name__ == '__main__':
    main()
