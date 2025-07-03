import sys
import os
from glob import glob 
import torch

sys.path.append(os.path.join(os.path.dirname(__file__),'vgtk') )
from SPConvNets.trainer_art import Trainer
from SPConvNets.options import opt


#以下のコードでop-alignを動かせる。ここでは、テストデータも設定できる点に注意！！（light　datasetのみ）
opt.model.miath = "model_weight/syn/safe.pth"  # モデルのパスを設定
opt.experiment_id = "safe"  # 実験IDを設定
opt.mode = "test"  # 実行モードを設定
opt.equi_settings.shape_type = "safe"  # オブジェクトの種類を設定
opt.equi_settings.nmasks = 2
opt.equi_settings.njoints = 1
opt.model.rotation_range = 120  # 回転範囲を設定
opt.model.joint_type = "r"  # ジョイントタイプを設定
opt.equi_settings.dataset_type = "Light"
opt.instance_list_path = glob(os.path.join("/home/hasegawa/research/efficient_manip/OP_Align/dataset/pc", 'partial/safe-ours', "4", '*', '*', '*.pt'))
#opt.instance_list_path = glob(os.path.join("/home/hasegawa/research/efficient_manip/OP_Align/dataset/pc", 'partial/laptop_h', '*', '*.pt'))


with torch.autograd.set_detect_anomaly(False):
  # トレーナーを実行
  trainer = Trainer(opt)
  trainer.test()
