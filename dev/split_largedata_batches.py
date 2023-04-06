import os
import shutil
from glob import glob

def load_args():
    parser = argparse.ArgumentParser(description='cppn-pytorch')
    parser.add_argument('-i', '--input',  type=str, help='name of input dir')
    parser.add_argument('-o', '--output', default='all_batch', type=str, help='name of output dir')
    parser.add_argument('-n', '--n_per_batch', default=5000, type=int, help='number of files per folder')
    args = parser.parse_args()
    return args

args = load_args()

os.makedirs(args.output, exist_ok=True)
assert os.path.isdir(args.input)
files = glob(f'{args.input}/*.jpg')

ind = 0
for i, fn in enumerate(files):
    other_fn = fn.replace('.jpg', '.tif')
    assert os.path.isfile(other_fn)

    save_dir = f'{args.output}/batch{ind}'
    os.makedirs(save_dir, exist_ok=True)
    new_fn_pair1 = fn.split('/')[-1]
    new_fn_pair2 = other_fn.split('/')[-1]
    shutil.copyfile(fn, f'{save_dir}/{new_fn_pair1}')
    shutil.copyfile(other_fn, f'{save_dir}/{new_fn_pair2}')
    if i % args.n == 0:
        ind += 1



