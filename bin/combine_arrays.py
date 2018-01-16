import pdspy.interferometry as uv
import glob

files=glob.glob('*.vis')
print(files)

data = []
for file in files:
    print(file)
    data.append(uv.readvis(file))


data[-1].weights[data[-1].uvdist < 50000] = 0.
data[-1].weights[data[-1].weights < 0.0] = 0.
vis = uv.concatenate(data)

vis.write("data.hdf5")
