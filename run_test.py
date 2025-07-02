import sys
import os
import torch

sys.path.append(os.path.join(os.path.dirname(__file__),'vgtk') )
from SPConvNets.trainer_art import Trainer
from SPConvNets.options import opt



opt.model.model = "art_so3net_pn"
opt.resume_path = "model_weight/syn/safe.pth"  # モデルのパスを設定
opt.experiment_id = "safe"  # 実験IDを設定
opt.mode = "test"  # 実行モードを設定
opt.equi_settings.shape_type = "safe"  # オブジェクトの種類を設定
opt.equi_settings.nmasks = 2
opt.equi_settings.njoints = 1
opt.model.rotation_range = 120  # 回転範囲を設定
opt.model.joint_type = "r"  # ジョイントタイプを設定
opt.equi_settings.dataset_type = "Light"


with torch.autograd.set_detect_anomaly(False):
  # トレーナーを実行
  trainer = Trainer(opt)
  trainer.test()
