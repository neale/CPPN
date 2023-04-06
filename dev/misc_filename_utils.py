import os
import shutil
from glob import glob

dirs = ['act_flow_trial1',
        'act_flow_trial2',
        'act_flow_trial3_lrg_search',
        'act_flow_trial3_lrg_search2',
        'actflow_trial1']

total = 0
tif_files = glob('keep/act_findtif/*.tif')
print (f'Need to find/replace {len(tif_files)} TIF files')
for tf in tif_files:
    tf = tf.split('/')[-1]
    print (f'Finding {tf}')
    result = []
    for d in dirs:
        result.extend([y for x in os.walk(d) for y in glob(os.path.join(x[0], tf))])
        #print (result)
    src = result[0]
    dst = os.path.join('keep/act_findtif', result[0].split('/')[-1])
    print (f'{src} -> {dst}')
    shutil.copyfile(src, dst)

"""
total = 0
for d in dirs:
    print (d)
    for ext in ['jpg', 'tif']:
        result = [y for x in os.walk(d) for y in glob(os.path.join(x[0], f'*{ext}'))]
        total += len(result)
        fns = [s.split('/')[-1] for s in result]
        for f, new_fn in zip(result, fns):
            print (f'{f} to all_act/{new_fn}')
            shutil.copyfile(f, f'all_act/{new_fn}')

print (total, 'files')
"""

dirs = ['graph_flow_debug',
        'graph_flow_debugw',
        'graph_flow_trial1',
        'graph_flow_trial2',
        'graph_flow_trial3']

total = 0
tif_files = glob('keep/graph_findtif/*.tif')
print (f'Need to find/replace {len(tif_files)} TIF files')
for tf in tif_files:
    tf = tf.split('/')[-1]
    print (f'Finding {tf}')
    result = []
    for d in dirs:
        result.extend([y for x in os.walk(d) for y in glob(os.path.join(x[0], tf))])
        #print (result)
    src = result[0]
    dst = os.path.join('keep/graph_findtif', result[0].split('/')[-1])
    print (f'{src} -> {dst}')
    shutil.copyfile(src, dst)


"""
total = 0
for d in dirs:
    print (d)
    for ext in ['jpg', 'tif']:
        result = [y for x in os.walk(d) for y in glob(os.path.join(x[0], f'*{ext}'))]
        total += len(result)
        fns = [s.split('/')[-1] for s in result]
        for f, new_fn in zip(result, fns):
            shutil.copyfile(f, f'all_graph/{new_fn}')
"""
print (total, 'files')
