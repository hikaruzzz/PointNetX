# paras
n_workers = 1
batch_size = 1
max_epoch = 51

lr = 0.0001
decay_rate = 1e-4
optimizer_name = 'adam'
scannet_file_root = r'C:\Users\PC\Desktop\scannet'
# scannet_file_root = r'/home/downloads/scannet'
max_point = 4096*2
n_class = 20
momentum = 9
is_load_checkpoints = False
load_ckpt_name = 'eph_32_iou_13.18%.ckpt.pth'  # ./checkpoint + load_ckpt_name

n_trian = 4 # train set = n-1/n_train ,test set = 1/n_train