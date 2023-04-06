import torch
import numpy as np
import tqdm
starts = range(0, 3000, 1500)
for start in starts:
    print ('START', start)
    frame = np.empty((1500, 3000, 3000, 3), dtype='u1')
    frame[0] = (torch.load(f'f_temp_gen_{start}.pt').numpy()*255.).astype('u1')
    for k in tqdm.tqdm(range(1, 1500)):
        idx = start + k
        frame[k] = (torch.load(f'f_temp_gen_{idx}.pt').numpy()*255.).astype('u1')
    print (frame.shape)
    np.save(f'frames_{start}-{k}.npy', frame)
