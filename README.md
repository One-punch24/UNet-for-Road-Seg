# Project Road Segmentation

For training (default, cuda),

python train.py --train True --epochs 10 --record_interval 5

For testing, (default, cpu)

python train.py --load_path ckpt/model_ep9.pth

For mask to submission, 

python mask_to_submission.py epoch epoch_number
