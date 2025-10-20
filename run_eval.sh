python3 eval.py --cfg configs/cas6d_pretrain.yaml --object_name linemod/cat
python3 eval.py --cfg configs/cas6d_pretrain.yaml --object_name linemod/benchvise
python3 eval.py --cfg configs/cas6d_pretrain.yaml --object_name linemod/cam
python3 eval.py --cfg configs/cas6d_pretrain.yaml --object_name linemod/driller
python3 eval.py --cfg configs/cas6d_pretrain.yaml --object_name linemod/duck
python3 eval.py --cfg configs/cas6d_pretrain.yaml --object_name linemod/lamp


#Locposenet detector eval 
python ./eval_detector.py


# Evaluate on the object TFormer from the casMOP/LINEMOD dataset
python3 eval.py --cfg configs/cas6d_train.yaml 
