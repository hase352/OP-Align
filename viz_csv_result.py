import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def scatter_confidence(csv_file_path, shape_id):
    data = pd.read_csv(csv_file_path)
    shape_id_data = data[data['shape_id'] == shape_id]
    iou = shape_id_data["iou_sum"].values
    confidence = shape_id_data["confidence"].values
    # 相関係数の計算
    correlation_coefficient = np.corrcoef(iou, confidence)[0, 1]

    # 散布図の作成
    plt.figure(figsize=(8, 6))
    plt.scatter(confidence, iou, color='blue', alpha=0.7, label="Data Points")
    plt.title(f"Scatter Plot of Confidence vs IoU shape_id: {shape_id}")
    plt.ylabel("IoU")
    plt.xlabel("Confidence")
    plt.grid(True)

    # 相関係数をプロット上に表示
    plt.text(0.55, 0.85, f"Correlation Coefficient: {correlation_coefficient:.2f}",
            fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    # 凡例の追加
    plt.legend()

    # プロットの表示
    plt.show()


if __name__  == "__main__":
    scatter_confidence("log/safe-hsaur-101603-30_test/model_20250123_130609/csv/safe-hsaur-101603-30_eval.csv", "101603")