from matplotlib.widgets import Slider, Button, RadioButtons
import matplotlib.pyplot as plt

"""
Display a 3d ndarray with a slider to move along the third dimension.

Extra keyword arguments are passed to imshow

Original from here: http://nbarbey.github.io/2011/07/08/matplotlib-slider.html
"""

def cubeshow(cube, axis=2, **kwargs):
    # check dim
    if not cube.ndim == 3:
        raise ValueError("cube should be an ndarray with ndim == 3")

    # Generate a figure with a set of axes.

    fig = plt.figure()

    ax = plt.subplot(111)

    fig.subplots_adjust(bottom=0.17, top=0.98)

    # Pick out just the first image to show.

    s = [slice(0, 1) if i == axis else slice(None) for i in range(3)]
    im = cube[tuple(s)].squeeze()

    # Show the image.

    l = ax.imshow(im, **kwargs)

    # Create the slider

    axslide = fig.add_axes([0.3, 0.02, 0.4, 0.075])

    slider = Slider(axslide, '', 0, cube.shape[axis] - 1, valinit=0, \
            valfmt=' %i', valstep=1)

    def update(val):
        ind = int(slider.val)
        s = [slice(ind, ind + 1) if i == axis else slice(None)
                 for i in range(3)]
        im = cube[tuple(s)].squeeze()
        l.set_data(im)
        fig.canvas.draw()

    slider.on_changed(update)

    # Create a button to go to the next frame.

    axnext = fig.add_axes([0.81, 0.02, 0.1, 0.075])

    bnext = Button(axnext, 'Next')

    def next_image(event):
        if slider.val < cube.shape[axis]-1:
            slider.set_val(slider.val+1)
        elif slider.val == cube.shape[axis]-1:
            slider.set_val(0)

    bnext.on_clicked(next_image)

    # Create a button to go to the previous frame.

    axprev = fig.add_axes([0.09, 0.02, 0.1, 0.075])

    bprev = Button(axprev, 'Prev.')

    def prev_image(event):
        if slider.val > 0:
            slider.set_val(slider.val-1)
        elif slider.val == 0:
            slider.set_val(cube.shape[axis]-1)

    bprev.on_clicked(prev_image)

    # Show the plot.

    plt.show(block=True)
