# Standard Modules
import argparse
import glob
from time import time
TIME=str(time)

# Custom Modules
import pdspy.interferometry as uv

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input',dest='inp')
parser.add_argument('-o', '--output',dest='out',default='data_{}.hdf5'.format(TIME))
args = parser.parse_args()

print('Using Directory: {}'.format(args.inp))
files=glob.glob('{}'.format(args.inp))

data = []
for file in files:
    data.append(uv.readvis(file))

data[-1].weights[data[-1].uvdist < 50000] = 0.
data[-1].weights[data[-1].weights < 0.0] = 0.
vis = uv.concatenate(data)

print('Writing to file: {}'.format(args.out))
vis.write(args.out)
