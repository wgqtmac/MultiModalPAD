

root_folder = "/home/hanhu/CASIA-SURF"
preroot_folder = "/home/hanhu/Spoof_Croped"
pretrain_train_list = "train_list.txt"
pretrain_val_list = "val_list.txt"
src_train_list = "train_list.txt"
src_val_list = "val_private_list.txt"
src_test_list = "test_public_list.txt"
# src_val_list = "val_public_list_new.txt"
#src_val_list = "val_public_list_with_label.txt"
# src_val_list = "test_list_divide_new.txt"
# src_val_list = "val_public_list_cxlan.txt"
# root_folder = "/home/gqwang/Spoof_Croped"
# src_train_list = "2train_list.txt"
# src_val_list = "2test_list.txt"

# pretrained_model = "/home/hanhu/code/PADChallenge/MultiModalPAD/caffe2pytorch/ResNet80_pytorch_model_lastest.pth.tar"
# pretrained_model = "/home/hanhu/code/PADChallenge/MultiModalPAD/resnet18.pth"
pretrained_model = "/home/hanhu/code/PADChallenge/MultiModalPAD/resnet18.pth"
# pretrained_model = "/home/hanhu/code/PADChallenge/MultiModalPAD/20_latest.path.tar"

cfg = "/home/hanhu/code/PADChallenge/MultiModalPAD/config/config.yml"
num_epochs = 50
output = "/home/hanhu/code/PADChallenge/Outputs/spoof_train"
workers = 6
weight_decay = 0.0005
momentum = 0.9
display = 50

max_iter = 2000

batch_size = 512
test_batch_size = 1
test_interval = 200
samples_per = 50
train_subs = 3229
test_subs = 807
topk = 3

base_lr = 0.0001
start_epoch = 0
start_iters = 0
best_model= 12345678.9
#-------------lr_policy--------------------#
# step
lr_policy = 'step'
#policy_parameter:
gamma = 0.333
step_size = 800

d_learning_rate = 1e-3
c_learning_rate = 1e-6
beta1 = 0.5
beta2 = 0.9
adapt_num_epochs = 100
save_step_pre = 5
log_step = 50
save_step = 1
model_root = "snapshots"



fusion_encoder_restore = "snapshots/MultiNet-fusion-final.pt"

manual_seed = None
save_step_pre = 1
log_step_pre = 10
eval_step_pre = 1


