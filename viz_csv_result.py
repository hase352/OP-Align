import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import re
import japanize_matplotlib
import statistics

def scatter_confidence(csv_file_path, shape_ids, confidence_name):
    data = pd.read_csv(csv_file_path)
    data['shape_id'] = data['shape_id'].astype(int)
    data['idx'] = data['idx'].astype(int)
    avg_correlation_coefficient, n_len_1 = 0, 0
    list_correlation_coefficient, outputs_opalign, outputs_pm= [], [], []
    for shape_id in shape_ids:
        shape_id_filtered_data = data[data['shape_id'] == int(shape_id)]
        iou = shape_id_filtered_data["iou_sum"].values
        confidence = shape_id_filtered_data[confidence_name].values
        # 相関係数の計算
        if len(iou) == 1:
            n_len_1 += 1
            correlation_coefficient = 0
            max_index = 0
        else:
            correlation_coefficient = np.corrcoef(iou, confidence)[0, 1]
            avg_correlation_coefficient += correlation_coefficient
            list_correlation_coefficient.append(correlation_coefficient)
        
            outputs_opalign.append(iou[0])
            confidence_list = confidence.tolist()
            max_index = confidence_list.index(max(confidence_list))
            output_pm = iou[max_index]
            outputs_pm.append(output_pm)

        # 散布図の作成
        plt.figure(figsize=(8, 6))
        plt.scatter(confidence, iou, color='blue', alpha=0.7)
        #plt.scatter(confidence[max_index], iou[max_index], color='red', label="Ours Output")
        #plt.suptitle(f"$w^1$ と IoU の散布図", fontsize=20, y=0.99)
        plt.title(f"相関係数: {correlation_coefficient:.2f}", fontsize=80)
        plt.xlabel(r"$w_{\mathrm{all}}$", fontsize=80)
        plt.ylabel("mIoU", fontsize=80)
        plt.grid(True)

        # 凡例の追加
        #plt.legend()

        # プロットの表示
        plt.show()
        
    avg_correlation_coefficient /= (len(shape_ids) - n_len_1)
    avg_outputs_opalign = sum(outputs_opalign) / len(outputs_opalign)
    avg_outputs_pm = sum(outputs_pm) / len(outputs_pm)
    print(statistics.pvariance(list_correlation_coefficient))
    
    print(csv_file_path, "相関係数", avg_correlation_coefficient, "op-align iou", avg_outputs_opalign, "proposed method iou",avg_outputs_pm)


def create_table_1_latex(csv_paths_all, test_shape_ids):
    all_row = R"""
\begin{table}[ht]
\centering
\begin{tabular}{ccccccc}
\toprule
\multirow{2}{*}{Joint Angle(\%)} & \multirow{2}{*}{Method} & \multicolumn{3}{c}{Segmentation IoU} & \multicolumn{2}{c}{Joint} \\ 
                     &  &  segment 0 $\uparrow$ & segment 1 $\uparrow$ & mean $\uparrow$ & Position $\downarrow$ & Direction $\downarrow$   \\ \midrule"""
    all_row_rm = R"""
\begin{table}[ht]
\centering
\begin{tabular}{ccccccc}
\toprule
\multirow{2}{*}{Joint Angle(\%)} & \multirow{2}{*}{Method} & \multicolumn{3}{c}{Segmentation IoU↑} & \multicolumn{2}{c}{Joint} \\ 
                     &   &   segment 0 $\uparrow$ & segment 1 $\uparrow$ & mean $\uparrow$ & Position $\downarrow$     & Direction $\downarrow$   \\ \midrule"""
    
    for csv_paths in csv_paths_all:
        one_row, one_row_rm= process_csv_and_create_latex_one_row(csv_paths, test_shape_ids)
        all_row += f"""
{one_row}"""
        all_row_rm += f"""
{one_row_rm}"""

    all_row += R"""
\end{tabular}
\caption{提案手法と既存手法の結果}
\label{tab:result-score}
\end{table}"""
    all_row_rm += R"""
\end{tabular}
\caption{提案手法と既存手法の結果(ただし、ロボットマニピュレーションによって物体に変化が起きなかった時を除く)}
\label{tab:result-score-without-miss}
\end{table}"""
    
    print(all_row)
    print(all_row_rm)
    

def create_table_2_latex(csv_path_all, test_shape_ids):
    all_row = R"""
\begin{table}[ht]
\centering
\begin{tabular}{ccccccc}
\toprule
\multirow{2}{*}{Joint Angle(\%)} & \multirow{2}{*}{Method} & \multicolumn{2}{c}{Segmentation↑} & \multicolumn{3}{c}{Joint↑} \\ 
                     && IoU75\%以上 & IoU50\%以上& $5\tcdegree$未満& $10\tcdegree$未満& $15\tcdegree$未満  \\ \midrule
""" 
    for csv_path in csv_path_all:
        one_row = ""
    

def process_csv_and_create_latex_one_row(csv_file_paths, test_shape_ids):
    opalign_result_pandas, ours_result_pandas, opalign_result_rm_pandas, ours_result_rm_pandas = [], [], [], []
    test_shape_ids = list(map(int, test_shape_ids))
    file_name = csv_file_paths[0].split('/')[-1]
    percentage = ''.join(re.findall(r'\d+', file_name))
    for csv_file_path in csv_file_paths:
        data = pd.read_csv(csv_file_path)
        data['shape_id'] = data['shape_id'].astype(int)
        data['idx'] = data['idx'].astype(int)
        
        shape_id_filtered_data = data[data['shape_id'] == test_shape_ids[0]]
        max_confidence_row = shape_id_filtered_data.loc[shape_id_filtered_data['confidence_seg_1'].idxmax()]
        opalign_result_one_csv, ours_result_one_csv = shape_id_filtered_data[shape_id_filtered_data['idx'] == 0], max_confidence_row.to_frame().T
        for shape_id in test_shape_ids[1:]:
            shape_id_filtered_data = data[data['shape_id'] == shape_id]
            max_confidence_row = shape_id_filtered_data.loc[shape_id_filtered_data['confidence_seg_1'].idxmax()]
            opalign_result_one_csv = pd.concat([opalign_result_one_csv, shape_id_filtered_data[shape_id_filtered_data['idx'] == 0]], axis=0)
            ours_result_one_csv = pd.concat([ours_result_one_csv, max_confidence_row.to_frame().T], axis=0)
        opalign_result_pandas.append(opalign_result_one_csv)
        ours_result_pandas.append(ours_result_one_csv)
    
    opalign_result = pd.concat(opalign_result_pandas, axis=0)
    ours_result = pd.concat(ours_result_pandas, axis=0)

    opalign_result = opalign_result.astype({'shape_id': 'int', 'idx': 'str'})
    ours_result = ours_result.astype({'shape_id': 'int', 'idx': 'int'})
    op_table = eval(opalign_result)
    ours_table = eval(ours_result)
    latex = create_latex_one_row(op_table, ours_table, percentage)
    
    #提案手法によって物体に変化が起きていないものを除く
    for csv_file_path in csv_file_paths:
        data = pd.read_csv(csv_file_path)
        data['shape_id'] = data['shape_id'].astype(int)
        data['idx'] = data['idx'].astype(int)
    
        opalign_result_one_csv, ours_result_one_csv = [], []
        for shape_id in test_shape_ids:
            shape_id_filtered_data = data[data['shape_id'] == shape_id]
            if len(shape_id_filtered_data) != 1:
                max_confidence_row = shape_id_filtered_data.loc[shape_id_filtered_data['confidence_seg_1'].idxmax()]
                opalign_result_one_csv.append(shape_id_filtered_data[shape_id_filtered_data['idx'] == 0])
                ours_result_one_csv.append(max_confidence_row.to_frame().T)
        opalign_result_one_csv = pd.concat(opalign_result_one_csv, axis=0)
        ours_result_one_csv = pd.concat(ours_result_one_csv, axis=0)
        opalign_result_rm_pandas.append(opalign_result_one_csv)
        ours_result_rm_pandas.append(ours_result_one_csv)
    
    opalign_result_rm = pd.concat(opalign_result_rm_pandas, axis=0)
    ours_result_rm = pd.concat(ours_result_rm_pandas,axis=0)
        
    op_table = eval(opalign_result_rm)
    ours_table = eval(ours_result_rm)
    latex_rm = create_latex_one_row(op_table, ours_table, percentage)

    
    return latex, latex_rm
    
def eval(data):
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
    """
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
    """
    return [mean_seg[0], mean_seg[1], (mean_seg[0] + mean_seg[1])/2, mean_joint, mean_drct]
    #return f"& {str(round(mean_seg[2], 2))}  & {str(round((mean_joint), 3))}  & {str(round(mean_drct, 2))}  \\      "
    
def create_latex_one_row(op_table, ours_table, percentage):
    latex_op, latex_ours = "", ""
    for i in range(3):
        if op_table[i] > ours_table[i]:
            latex_op += r'& \textbf{' +f"{round(op_table[i], 2):.2f}" + "}  "
            latex_ours += f"& {round(ours_table[i], 2):.2f}  "
        elif op_table[i] < ours_table[i]:
            latex_op += f"& {round(op_table[i], 2):.2f}  "
            latex_ours += r"& \textbf{" +f"{round(ours_table[i], 2):.2f}" + "}  "
        else:
            latex_op += r"& \textbf{" +f"{round(op_table[i], 2):.2f}" + "}  "
            latex_ours += r"& \textbf{" +f"{round(ours_table[i], 2):.2f}" + "}  "
    
    for i in range(3,5):
        ndigits = 6 - i
        if op_table[i] < ours_table[i]:
            latex_op += r'& \textbf{' +f"{round(op_table[i], ndigits):.{ndigits}f}" + "}  "
            latex_ours += f"& {round(ours_table[i], ndigits):.{ndigits}f}  "
        elif op_table[i] > ours_table[i]:
            latex_op += f"& {round(op_table[i], ndigits):.{ndigits}f}  "
            latex_ours += r"& \textbf{" +f"{round(ours_table[i], ndigits):.{ndigits}f}" + "}  "
        else:
            latex_op += r"& \textbf{" +f"{round(op_table[i], ndigits):.{ndigits}f}" + "}  "
            latex_ours += r"& \textbf{" +f"{round(ours_table[i], ndigits):.{ndigits}f}" + "}  "
    
    if int(percentage) == 100:
        latex = "\multirow{2}{*}{" + str(percentage) + "}  & OP-Align  " + latex_op + r"""\\    
                    & Ours  """ + latex_ours + r"\\  \bottomrule"
    else:
        latex = "\multirow{2}{*}{" + str(percentage) + "}  & OP-Align  " + latex_op + r"""\\    
                    & Ours  """ + latex_ours + r"\\  \hline"
    
    return latex




def Line_plot(csv_path, shape_id):
    data = pd.read_csv(csv_path)
    data['shape_id'] = data['shape_id'].astype(int)
    data['idx'] = data['idx'].astype(int)
    shape_id_filtered_data = data[data['shape_id'] == int(shape_id)]
    shape_id_filtered_data = shape_id_filtered_data.sort_values(by='idx')
    
    df = shape_id_filtered_data.set_index('idx')  # 'itr'をインデックスにする
    df = df.reindex(range(df.index.min(), df.index.max() + 1))  # 欠損しているitrを追加
    df = df.ffill()  # 前の値で埋める

    # インデックスをリセット（元の形に戻す）
    shape_id_filtered_data = df.reset_index()
    
    itr = np.linspace(0, 10, 11)
    iou  = (shape_id_filtered_data['seg_1'] + shape_id_filtered_data['seg_0']) / 2
    confidence = shape_id_filtered_data['confidence_seg_1']

    fig, ax1 = plt.subplots()

    import matplotlib as mpl
    mpl.rcParams['font.family'] = 'cmr10'  # Computer Modernフォント
    mpl.rcParams['mathtext.fontset'] = 'cm'  # 数式もLaTeX風にする

    # 1つ目の縦軸
    ax1.plot(itr, iou, color='blue', linestyle='-', label='mIoU')
    ax1.set_xlabel('Iteration', fontsize=20)
    ax1.set_ylabel('mIoU', color='blue', fontsize=20)
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True)

    # 2つ目の縦軸を作成
    ax2 = ax1.twinx()
    ax2.plot(itr, confidence, color='red', linestyle='--', label=rf'$w$')
    ax2.set_ylabel(rf'$w$', color='red', fontsize=20)
    ax2.tick_params(axis='y', labelcolor='red')

    # 凡例の追加
    #ax1.legend(loc='upper left', bbox_to_anchor=(0.1, 1.0))
    #ax2.legend(loc='upper left', bbox_to_anchor=(0.9, 1.0))
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=13)

    plt.show()

    
        


if __name__  == "__main__":
    train_shape_ids = ["101564", "101593", "101599",  "101604", "101605", "101611", 
                       "101612", "101613", "101619", "101623", "102301", "102316", 
                       "102318", "102380", "102381", "102423"]
    test_shape_ids = ["101363", "101579", "101584", "101591", "101594", "101603", 
                      "102384", "102387", "102389", "102418", "102278",  "102309", "102311"]
    csv_path_1_0 = "log/safe-hsaur-opalign-0_test/model_20250124_165237/csv/safe-hsaur-opalign-0_eval.csv"
    csv_path_2_0 = "log/safe-ours-2-0_test/model_20250128_102412/csv/safe-ours-2-0_eval.csv"
    csv_path_3_0 = "log/safe-ours-3-0_test/model_20250128_005133/csv/safe-ours-3-0_eval.csv"
    
    csv_path_1_10 = "log/safe-hsaur-opalign-10_test/model_20250124_170426/csv/safe-hsaur-opalign-10_eval.csv"
    csv_path_2_10 = "log/safe-ours-2-10_test/model_20250128_102514/csv/safe-ours-2-10_eval.csv"
    csv_path_3_10 = "log/safe-ours-3-10_test/model_20250128_005229/csv/safe-ours-3-10_eval.csv"
    
    csv_path_1_20 = "log/safe-hsaur-opalign-20_test/model_20250124_170950/csv/safe-hsaur-opalign-20_eval.csv"
    csv_path_2_20 = "log/safe-ours-2-20_test/model_20250128_102534/csv/safe-ours-2-20_eval.csv"
    csv_path_3_20 = "log/safe-ours-3-20_test/model_20250128_005252/csv/safe-ours-3-20_eval.csv"
    
    csv_path_1_30 = "log/safe-hsaur-opalign-30_test/model_20250125_010343/csv/safe-hsaur-opalign-30_eval.csv"#
    csv_path_2_30 = "log/safe-ours-2-30_test/model_20250128_102557/csv/safe-ours-2-30_eval.csv"
    csv_path_3_30 = "log/safe-ours-3-30_test/model_20250128_005310/csv/safe-ours-3-30_eval.csv"
    
    csv_path_1_40 = "log/safe-hsaur-opalign-40_test/model_20250125_014626/csv/safe-hsaur-opalign-40_eval.csv"
    csv_path_2_40 = "log/safe-ours-2-40_test/model_20250128_102616/csv/safe-ours-2-40_eval.csv"
    csv_path_3_40 = "log/safe-ours-3-40_test/model_20250128_005329/csv/safe-ours-3-40_eval.csv"
    
    csv_path_1_50 = "log/safe-hsaur-opalign-50_test/model_20250125_010514/csv/safe-hsaur-opalign-50_eval.csv"
    csv_path_2_50 = "log/safe-ours-2-50_test/model_20250128_102639/csv/safe-ours-2-50_eval.csv"
    csv_path_3_50 = "log/safe-ours-3-50_test/model_20250128_005358/csv/safe-ours-3-50_eval.csv"
    
    csv_path_1_80 = "log/safe-hsaur-opalign-80_test/model_20250126_122256/csv/safe-hsaur-opalign-75_eval.csv"
    csv_path_2_80 = "log/safe-ours-2-80_test/model_20250128_102702/csv/safe-ours-2-80_eval.csv"
    csv_path_3_80 = "log/safe-ours-3-80_test/model_20250128_005427/csv/safe-ours-3-80_eval.csv"
    
    csv_path_1_100 = "log/safe-hsaur-opalign-100_test/model_20250126_122316/csv/safe-hsaur-opalign-100_eval.csv"
    csv_path_2_100 = "log/safe-ours-2-100_test/model_20250128_102730/csv/safe-ours-2-100_eval.csv"
    csv_path_3_100 = "log/safe-ours-3-100_test/model_20250128_005448/csv/safe-ours-3-100_eval.csv"
    #csv_path_all = [csv_path_1_0, csv_path_10, csv_path_20, csv_path_1_30, csv_path_40, csv_path_50, csv_path_80, csv_path_100]
    csv_paths_all = [[csv_path_1_0, csv_path_2_0, csv_path_3_0], [csv_path_1_10, csv_path_2_10, csv_path_3_10], [csv_path_1_20, csv_path_2_20, csv_path_3_20],
                     [csv_path_1_30, csv_path_2_30, csv_path_3_30], [csv_path_1_40, csv_path_2_40, csv_path_3_40], [csv_path_1_50, csv_path_2_50, csv_path_3_50], 
                     [csv_path_1_80, csv_path_2_80, csv_path_3_80], [csv_path_1_100, csv_path_2_100, csv_path_3_100]]
    
    create_table_1_latex(csv_paths_all=csv_paths_all, test_shape_ids=test_shape_ids)
    scatter_confidence("log/safe-obj-train_test/model_20250126_181417/csv/safe-obj-train_eval.csv", [102318], "confidence_all")
    #Line_plot("log/safe-ours-gomi-101594-10_test/model_20250130_183832/csv/safe-ours-gomi-101594-10_eval.csv", "101594")