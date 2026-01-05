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
nohup python3 train_model.py --cfg configs/refiner/mvs2d_refiner.yaml > mvs2d_refiner_train.log 2>&1 &

nohup python3 eval.py --cfg configs/cas6d_train.yaml --dataset LINEMOD > eval1.log 2>&1 &
nohup python3 eval.py --cfg configs/cas6d_train.yaml --dataset GENMOP > eval2.log 2>&1 &

nohup python ./train_model.py --cfg configs/detector/detector_train.yaml > detector_train.log 2>&1 &
nohup python ./train_model.py --cfg configs/detector/detector_train_dino.yaml > detector_train_dino.log 2>&1 &

#Locposenet detector eval 
nohup python ./eval_detector.py > eval_detector1.log 2>&1 &


# Evaluate on the object TFormer from the casMOP/LINEMOD dataset
python3 eval.py --cfg configs/cas6d_train.yaml 


# we save one image every 10 frames and maximum image side length is 960
python prepare.py --action video2image \
                  --input ../data/custom/video/mouse-ref.mp4 \
                  --output ../data/custom/mouse/images \
                  --frame_inter 10 \
                  --image_size 960 \
                  --transpose