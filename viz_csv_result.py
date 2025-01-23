import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

def scatter_confidence(csv_file_path, shape_id):
    data = pd.read_csv(csv_file_path)
    data['shape_id'] = data['shape_id'].astype(int)
    if not isinstance(shape_id, int):
        raise TypeError(f"Expected an integer, but got {type(shape_id).__name__}")
    shape_id_filtered_data = data[data['shape_id'] == shape_id]
    iou = shape_id_filtered_data["iou_sum"].values
    confidence = shape_id_filtered_data["confidence"].values
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


def eval_result_by_percentage(csv_file_path, open_percentage):
    data = pd.read_csv(csv_file_path)
    data['idx'] = data['idx'].astype(int)
    if not isinstance(open_percentage, int):
        raise TypeError(f"Expected an integer, but got {type(open_percentage).__name__}")
    open_percentage_filtered_data = data[data['idx'] == open_percentage]
    
    mean_seg = [open_percentage_filtered_data['seg_0'].mean() *100, open_percentage_filtered_data['seg_1'].mean() * 100]
    mean_joint = open_percentage_filtered_data['joint_0'].mean()
    mean_drct = open_percentage_filtered_data['direction_0'].mean()
    
    joint_5degree = torch.tensor(open_percentage_filtered_data['direction_0'].values) < 5
    joint_10degree = torch.tensor(open_percentage_filtered_data['direction_0'].values) < 10
    joint_5cm = torch.tensor(open_percentage_filtered_data['joint_0'].values) < 0.5
    joint_10cm = torch.tensor(open_percentage_filtered_data['joint_0'].values) < 1
    joint_5d5c = torch.logical_and(joint_5degree, joint_5cm).sum() / open_percentage_filtered_data['idx'].shape[0]
    joint_5d10c = torch.logical_and(joint_5degree, joint_10cm).sum() / open_percentage_filtered_data['idx'].shape[0]
    joint_10d5c = torch.logical_and(joint_10degree, joint_5cm).sum() / open_percentage_filtered_data['idx'].shape[0]
    joint_10d10c = torch.logical_and(joint_10degree, joint_10cm).sum() / open_percentage_filtered_data['idx'].shape[0]
    seg_50 = torch.logical_and(torch.tensor(open_percentage_filtered_data['seg_0'].values) > 0.50,
                               torch.tensor(open_percentage_filtered_data['seg_1'].values)> 0.50)
    seg_75 = torch.logical_and(torch.tensor(open_percentage_filtered_data['seg_0'].values) > 0.75, 
                               torch.tensor(open_percentage_filtered_data['seg_1'].values) > 0.75)
    seg_50 = seg_50.sum() / open_percentage_filtered_data['idx'].shape[0]
    seg_75 = seg_75.sum() / open_percentage_filtered_data['idx'].shape[0]
    
    print('Testing', 'Average Segmentation IoU: ' + str(mean_seg))
    print('Testing', 'Average Joint Position Error: ' + str(mean_joint))
    print('Testing', 'Average Joint Direction Error: ' + str(mean_drct))
    print('Testing', 'Joint 5degree5cm: ' + str(100 * joint_5d5c))
    print('Testing', 'Joint 5degree10cm: ' + str(100 * joint_5d10c))
    print('Testing', 'Joint 10degree5cm: ' + str(100 * joint_10d5c))
    print('Testing', 'Joint 10degree10cm: ' + str(100 * joint_10d10c))
    print('Testing', 'Segmentation 50: ' + str(100 * seg_50))
    print('Testing', 'Segmentation 75: ' + str(100 * seg_75))

if __name__  == "__main__":
    #scatter_confidence("log/safe_test/model_20250123_163921/csv/safe_eval.csv", 101363)
    eval_result_by_percentage("log/safe-30_test/model_20250123_180116/csv/safe-30_eval.csv", 30)