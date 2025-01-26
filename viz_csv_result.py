import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import re

def scatter_confidence(csv_file_path, shape_ids, confidence_name):
    data = pd.read_csv(csv_file_path)
    data['shape_id'] = data['shape_id'].astype(int)
    avg_correlation_coefficient, n_len_1 = 0, 0
    outputs_opalign, outputs_pm= [], []
    for shape_id in shape_ids:
        shape_id_filtered_data = data[data['shape_id'] == int(shape_id)]
        iou = shape_id_filtered_data["iou_sum"].values
        confidence = shape_id_filtered_data[confidence_name].values
        # 相関係数の計算
        if len(iou) == 1:
            n_len_1 += 1
            correlation_coefficient = 0
        else:
            correlation_coefficient = np.corrcoef(iou, confidence)[0, 1]
            avg_correlation_coefficient += correlation_coefficient
        
            outputs_opalign.append(iou[0])
            confidence_list = confidence.tolist()
            max_index = confidence_list.index(max(confidence_list))
            output_pm = iou[max_index]
            outputs_pm.append(output_pm)

        # 散布図の作成
        plt.figure(figsize=(8, 6))
        plt.scatter(confidence, iou, color='blue', alpha=0.7, label="Data Points")
        plt.suptitle(f"Scatter Plot of Confidence vs IoU shape id: {shape_id}", fontsize=12, y=0.95)
        plt.title(f"Correlation Coefficient: {correlation_coefficient:.2f}")
        plt.ylabel("IoU")
        plt.xlabel("Confidence")
        plt.grid(True)

        # 凡例の追加
        plt.legend()

        # プロットの表示
        #plt.show()
        
    avg_correlation_coefficient /= (len(shape_ids) - n_len_1)
    avg_outputs_opalign = sum(outputs_opalign) / len(outputs_opalign)
    avg_outputs_pm = sum(outputs_pm) / len(outputs_pm)
    
    print(csv_file_path, "相関係数", avg_correlation_coefficient, "op-align iou", avg_outputs_opalign, "proposed method iou",avg_outputs_pm)


def eval_test_result(csv_file_path, test_shape_ids):
    data = pd.read_csv(csv_file_path)
    data['shape_id'] = data['shape_id'].astype(int)
    data['idx'] = data['idx'].astype(int)
    test_shape_ids = list(map(int, test_shape_ids))
    
    shape_id_filtered_data = data[data['shape_id'] == test_shape_ids[0]]
    max_confidence_row = shape_id_filtered_data.loc[shape_id_filtered_data['confidence_seg_1'].idxmax()]
    opalign_result, ours_result = shape_id_filtered_data[shape_id_filtered_data['idx'] == 0], max_confidence_row.to_frame().T
    for shape_id in test_shape_ids[1:]:
        shape_id_filtered_data = data[data['shape_id'] == shape_id]
        max_confidence_row = shape_id_filtered_data.loc[shape_id_filtered_data['confidence_seg_1'].idxmax()]
        opalign_result = pd.concat([opalign_result, shape_id_filtered_data[shape_id_filtered_data['idx'] == 0]], axis=0)
        ours_result = pd.concat([ours_result, max_confidence_row.to_frame().T], axis=0)
        
    opalign_result = opalign_result.astype({'shape_id': 'int', 'idx': 'str'})
    ours_result = ours_result.astype({'shape_id': 'int', 'idx': 'int'})
    print("op-align result")
    op_table = print_eval(opalign_result)
    print("\nours result")
    ours_table = print_eval(ours_result)
    
    latex_op, latex_ours = "", ""
    if op_table[0] > ours_table[0]:
        latex_op += r'& \textbf{' +f"{round(op_table[0], 2):.2f}" + "}  "
        latex_ours += f"& {round(ours_table[0], 2):.2f}  "
    elif op_table[0] < ours_table[0]:
        latex_op += f"& {round(op_table[0], 2):.2f}  "
        latex_ours += r"& \textbf{" +f"{round(ours_table[0], 2):.2f}" + "}  "
    else:
        latex_op += r"& \textbf{" +f"{round(op_table[0], 2):.2f}" + "}  "
        latex_ours += r"& \textbf{" +f"{round(ours_table[0], 2):.2f}" + "}  "
    
    for i in range(1,3):
        ndigits = 4 - i
        if op_table[i] < ours_table[i]:
            latex_op += r'& \textbf{' +f"{round(op_table[i], ndigits):.{ndigits}f}" + "}  "
            latex_ours += f"& {round(ours_table[i], ndigits):.{ndigits}f}  "
        elif op_table[i] > ours_table[i]:
            latex_op += f"& {round(op_table[i], ndigits):.{ndigits}f}  "
            latex_ours += r"& \textbf{" +f"{round(ours_table[i], ndigits):.{ndigits}f}" + "}  "
        else:
            latex_op += r"& \textbf{" +f"{round(op_table[i], ndigits):.{ndigits}f}" + "}  "
            latex_ours += r"& \textbf{" +f"{round(ours_table[i], ndigits):.{ndigits}f}" + "}  "
       
    
    latex = "& OP-Align  " + latex_op + r"\\    & Ours  " + latex_ours + r"\\  \hline"

    
    for shape_id in test_shape_ids:
        shape_id_filtered_data = data[data['shape_id'] == shape_id]
        if len(shape_id_filtered_data) == 1:
            opalign_result = opalign_result[opalign_result["shape_id"] != shape_id]
            ours_result = ours_result[ours_result['shape_id'] != shape_id]
    
    print("\n\nop-align result after remove manipulation miss data")
    op_table = print_eval(opalign_result)
    print("\nours result after remove manipulation miss data")
    ours_table = print_eval(ours_result)
    
    latex_op, latex_ours = "", ""
    if op_table[0] > ours_table[0]:
        latex_op += r'& \textbf{' +f"{round(op_table[0], 2):.2f}" + "}  "
        latex_ours += f"& {round(ours_table[0], 2):.2f}  "
    elif op_table[0] < ours_table[0]:
        latex_op += f"& {round(op_table[0], 2):.2f}  "
        latex_ours += r"& \textbf{" +f"{round(ours_table[0], 2):.2f}" + "}  "
    else:
        latex_op += r"& \textbf{" +f"{round(op_table[0], 2):.2f}" + "}  "
        latex_ours += r"& \textbf{" +f"{round(ours_table[0], 2):.2f}" + "}  "
    
    for i in range(1,3):
        ndigits = 4 - i
        if op_table[i] < ours_table[i]:
            latex_op += r'& \textbf{' +f"{round(op_table[i], ndigits):.{ndigits}f}" + "}  "
            latex_ours += f"& {round(ours_table[i], ndigits):.{ndigits}f}  "
        elif op_table[i] > ours_table[i]:
            latex_op += f"& {round(op_table[i], ndigits):.{ndigits}f}  "
            latex_ours += r"& \textbf{" +f"{round(ours_table[i], ndigits):.{ndigits}f}" + "}  "
        else:
            latex_op += r"& \textbf{" +f"{round(op_table[i], ndigits):.{ndigits}f}" + "}  "
            latex_ours += r"& \textbf{" +f"{round(ours_table[i], ndigits):.{ndigits}f}" + "}  "
       
    
    latex_rm = "& OP-Align  " + latex_op + r"\\    & Ours  " + latex_ours + r"\\  \hline"
    
    file_name = csv_file_path.split('/')[-1]
    percentage = ''.join(re.findall(r'\d+', file_name))
    
    return "\multirow{2}{*}{" + str(percentage) + "}  " + latex, "\multirow{2}{*}{" + str(percentage) + "}  " + latex_rm
    
def print_eval(data):
    mean_seg = [data['seg_0'].mean() *100, data['seg_1'].mean() * 100, data['iou_sum'].mean() * 100]
    mean_joint = data['joint_0'].mean()
    mean_drct = data['direction_0'].mean()
    
    joint_5degree = torch.tensor(data['direction_0'].values) < 5
    joint_10degree = torch.tensor(data['direction_0'].values) < 10
    joint_15degree = torch.tensor(data['direction_0'].values) < 15
    joint_5cm = torch.tensor(data['joint_0'].values) < 0.1
    joint_10cm = torch.tensor(data['joint_0'].values) < 0.2
    joint_15cm = torch.tensor(data['joint_0'].values) < 0.3
    joint_5d = joint_5degree.sum() / data['idx'].shape[0]
    joint_5d10c = torch.logical_and(joint_5degree, joint_10cm).sum() / data['idx'].shape[0]
    joint_10d5c = torch.logical_and(joint_10degree, joint_5cm).sum() / data['idx'].shape[0]
    joint_10d = joint_10degree.sum() / data['idx'].shape[0]
    joint_15d = joint_15degree.sum() / data['idx'].shape[0]
    seg_50 = torch.logical_and(torch.tensor(data['seg_0'].values) > 0.50,
                               torch.tensor(data['seg_1'].values)> 0.50)
    seg_75 = torch.logical_and(torch.tensor(data['seg_0'].values) > 0.75, 
                               torch.tensor(data['seg_1'].values) > 0.75)
    seg_50 = seg_50.sum() / data['idx'].shape[0]
    seg_75 = seg_75.sum() / data['idx'].shape[0]
    
    print('Testing', 'Average Segmentation IoU: ' + str(mean_seg))
    print('Testing', 'Average Joint Position Error: ' + str(mean_joint))
    print('Testing', 'Average Joint Direction Error: ' + str(mean_drct))
    print('Testing', 'Joint 5degree: ' + str(100 * joint_5d))
    #print('Testing', 'Joint 5degree10cm: ' + str(100 * joint_5d10c))
    #print('Testing', 'Joint 10degree5cm: ' + str(100 * joint_10d5c))
    print('Testing', 'Joint 10degree: ' + str(100 * joint_10d))
    print('Testing', 'Joint 15degree: ' + str(100 * joint_15d))
    print('Testing', 'Segmentation 50: ' + str(100 * seg_50))
    print('Testing', 'Segmentation 75: ' + str(100 * seg_75))
    
    return [mean_seg[2], mean_joint, mean_drct]
    #return f"& {str(round(mean_seg[2], 2))}  & {str(round((mean_joint), 3))}  & {str(round(mean_drct, 2))}  \\      "


if __name__  == "__main__":
    train_shape_ids = ["101604", "101605", "101611", "101612", "101613", "101619", "101623", "102316", "102318"]
    test_shape_ids = ["101363", "101564", "101579", "101584", "101591", "101593", "101594", 
                       "101599", "101603", "102380", "102381", "102384", "102387", "102389", 
                       "102418", "102423", "102278", "102301",  "102309", "102311"]
    csv_path_0 = "log/safe-hsaur-opalign-0_test/model_20250124_165237/csv/safe-hsaur-opalign-0_eval.csv"
    csv_path_10 = "log/safe-hsaur-opalign-10_test/model_20250124_170426/csv/safe-hsaur-opalign-10_eval.csv"
    csv_path_20 = "log/safe-hsaur-opalign-20_test/model_20250124_170950/csv/safe-hsaur-opalign-20_eval.csv"
    csv_path_30 = "log/safe-hsaur-opalign-30_test/model_20250125_010343/csv/safe-hsaur-opalign-30_eval.csv"#
    csv_path_40 = "log/safe-hsaur-opalign-40_test/model_20250125_014626/csv/safe-hsaur-opalign-40_eval.csv"
    csv_path_50 = "log/safe-hsaur-opalign-50_test/model_20250125_010514/csv/safe-hsaur-opalign-50_eval.csv"
    one_row, one_row_rm = eval_test_result(csv_path_0, test_shape_ids)
    
    print(one_row)
    print(one_row_rm)
    #scatter_confidence(csv_path_30, test_shape_ids, "confidence_seg_1")