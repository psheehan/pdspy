#!/usr/bin/env python3

import pdspy.interferometry as uv
import matplotlib.pyplot as plt

data = uv.Visibilities()
data.read("testdata.hdf5")

image = uv.invert(data, imsize=512, pixel_size=0.5, convolution="expsinc")

plt.imshow(image.image[:,:,0], origin="lower", interpolation="none")
plt.colorbar()
plt.savefig("test.pdf")
