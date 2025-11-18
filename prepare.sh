#!/bin/bash

# ========== LINEMOD数据集验证数据准备 ==========
# LINEMOD数据集包含6个类别，所有类别使用 linemod_val split

python3 prepare.py --action gen_val_set --estimator_cfg configs/cas6d_train.yaml --que_database linemod/cat --que_split linemod_val --ref_database linemod/cat --ref_split linemod_val

python3 prepare.py --action gen_val_set --estimator_cfg configs/cas6d_train.yaml --que_database linemod/benchvise --que_split linemod_val --ref_database linemod/benchvise --ref_split linemod_val

python3 prepare.py --action gen_val_set --estimator_cfg configs/cas6d_train.yaml --que_database linemod/cam --que_split linemod_val --ref_database linemod/cam --ref_split linemod_val

python3 prepare.py --action gen_val_set --estimator_cfg configs/cas6d_train.yaml --que_database linemod/driller --que_split linemod_val --ref_database linemod/driller --ref_split linemod_val

python3 prepare.py --action gen_val_set --estimator_cfg configs/cas6d_train.yaml --que_database linemod/duck --que_split linemod_val --ref_database linemod/duck --ref_split linemod_val

python3 prepare.py --action gen_val_set --estimator_cfg configs/cas6d_train.yaml --que_database linemod/lamp --que_split linemod_val --ref_database linemod/lamp --ref_split linemod_val

# ========== GENMOP数据集验证数据准备 ==========
# GENMOP数据集包含5个类别，所有类别使用 all split

python3 prepare.py --action gen_val_set --estimator_cfg configs/cas6d_train.yaml --que_database genmop/chair-test --que_split all --ref_database genmop/chair-ref --ref_split all

python3 prepare.py --action gen_val_set --estimator_cfg configs/cas6d_train.yaml --que_database genmop/plug_en-test --que_split all --ref_database genmop/plug_en-ref --ref_split all

python3 prepare.py --action gen_val_set --estimator_cfg configs/cas6d_train.yaml --que_database genmop/piggy-test --que_split all --ref_database genmop/piggy-ref --ref_split all

python3 prepare.py --action gen_val_set --estimator_cfg configs/cas6d_train.yaml --que_database genmop/scissors-test --que_split all --ref_database genmop/scissors-ref --ref_split all

python3 prepare.py --action gen_val_set --estimator_cfg configs/cas6d_train.yaml --que_database genmop/tformer-test --que_split all --ref_database genmop/tformer-ref --ref_split all

