from easydict import EasyDict as edict
import yaml
import numpy as np

with open('config.yml', 'r') as f:
    parser = edict(yaml.load(f))
for x in parser:
    print '{} : {}'.format(x, parser[x])
