import os
import re
import zlib
import json
import glob
import base64
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, required=True)
parser.add_argument('--metric', type=str, required=True)
args = parser.parse_args()

results = {}
log_files = glob.glob(os.path.join(args.dir, '*.txt'))
for file in log_files:
    regex = re.match(r'^.*?args-(.*)__seed-(.*)__code-(.*).txt$', file)
    assert(regex)
    args_hash = regex.group(1)
    seed = int(regex.group(2))
    code_hash = regex.group(3)
    key = 'args-%s__code-%s' % (args_hash, code_hash)
    with open(file, 'r') as f:
        contents = f.read()
    test_code = re.findall(r'Test: .*\n', contents)[-1].split()[-1]
    test_result = zlib.decompress(base64.b64decode(test_code.encode())).decode().split('\n')[0]
    if args.metric not in test_result:
        continue
    if key not in results:
        results[key] = {'args': re.search(r'args = {[\s\S]*}', contents).group(0), 'test': {}}
    test_value = float(re.search('Test %s:.*?([0-9.]+)' % args.metric, test_result).group(1))
    results[key]['test'][seed] = test_value

for k, v in results.items():
    print(k)
    print(v['args'])
    mean = np.mean(list(v['test'].values()))
    std = np.std(list(v['test'].values()))
    test_list = sorted(list(v['test'].items()), key=lambda x: x[0])
    for x in test_list:
        print('seed %05d: %s' % x)
    print('mean ± std: %g ± %g' % (mean, std))
    print('--------------------------')

# out_str = json.dumps(results)
# print(out_str.replace('\\n', '\n'))
