import os
import sys
import apecosm

config = apecosm.read_config(os.path.join('doc', 'data', 'config', 'oope.conf'))
print(config.keys())
print(config['grid.mask.var.e2v'])
