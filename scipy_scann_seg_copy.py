import os
import shutil


origin_root = '/mnt/dataset/scannet/'  # root of ScanNet dir
target_root = './dataset'

assert os.path.isdir(os.path.join(origin_root,"scans")), "not found scans"

file_list = os.walk(os.path.join(origin_root,"scans"))
n_count = 0
len_filenames = len(os.listdir(os.path.join(origin_root,"scans")))
for dirpath, _dirnames, filenames in file_list:
    for f_name in filenames:
        n_count += 1
        if f_name[-18:] == "clean_2.labels.ply" or f_name[-11:] == "clean_2.ply":
            try:
                os.makedirs(os.path.join(target_root,dirpath.split("/")[-3],dirpath.split("/")[-2],dirpath.split("/")[-1]))
            except:
                pass
            origin_file_path = os.path.join(dirpath, f_name)
            #print("cpy file from \n{} to:\n".format(origin_file_path))
            target_file_path = os.path.join(target_root,dirpath.split("/")[-3],dirpath.split("/")[-2],dirpath.split("/")[-1],f_name)
            #print(target_file_path)
            shutil.copy(origin_file_path,target_file_path)
            print("\r dir={} copying : {}/{}".format('scans',n_count, len_filenames),end='',flush=True)