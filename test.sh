CUDA_VISIBLE_DEVICES=0,2 python tools/run_test.py \
    --root_folder /home/gqwang/Spoof_Croped \
    --test_list 1test_list.txt \
    --pretrained_model /home/gqwang/PyTorch_ResNet80/Outputs/spoof_train_noprint/20_latest.path.tar \
    --cfg /home/gqwang/PyTorch_ResNet80/config/config.yml \
    --output /home/gqwang/PyTorch_ResNet80/Outputs/triplet_pair_spoof_test
