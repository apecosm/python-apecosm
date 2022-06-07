import os
import sys
sys.path.insert(0, os.path.abspath('../python/'))
import apecosm

config = apecosm.read_config('_static/example/data/config/oope.conf')

wstep, lstep = apecosm.extract_weight_grid(config)

print(wstep)
