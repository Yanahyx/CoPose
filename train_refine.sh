# python3 prepare.py --action gen_val_set \
#                   --estimator_cfg configs/cas6d_train.yaml \
#                   --que_database linemod/cat \
#                   --que_split linemod_val \
#                   --ref_database linemod/cat \
#                   --ref_split linemod_val

# python3 prepare.py --action gen_val_set \
#                   --estimator_cfg configs/cas6d_train.yaml \
#                   --que_database genmop/tformer-test \
#                   --que_split all \
#                   --ref_database genmop/tformer-ref \
#                   --ref_split all 

# python3 train_model.py --cfg configs/detector/detector_train.yaml
# python3 train_model.py --cfg configs/selector/selector_train.yaml
nohup python3 train_model.py --cfg configs/refiner/refiner_train.yaml > refiner_train.log 2>&1 &
nohup python3 eval.py --cfg configs/cas6d_train.yaml --dataset LINEMOD > eval1.log 2>&1 &
nohup python3 eval.py --cfg configs/cas6d_train.yaml --dataset GENMOP > eval2.log 2>&1 &