export PYTHONUNBUFFERED="True"

LOG="Outputs/triplet_pair_sub/train-`date +'%Y-%m-%d-%H-%M-%S'`.log"

CUDA_VISIBLE_DEVICES=1 python tools/train_val.py \
    --root_folder /home/gqwang/Spoof_Croped \
    --train_list new_train_list.txt \
    --test_list new_val_list.txt \
    --pretrained_model /home/gqwang/PyTorch_ResNet80/caffe2pytorch/ResNet80_pytorch_model_lastest.pth.tar \
    --cfg /home/gqwang/PyTorch_ResNet80/config/config.yml \
	--num_epochs 20 \
    --output /home/gqwang/PyTorch_ResNet80/Outputs/spoof_train_noprint | tee $LOG
