import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.patches import Ellipse
from numpy import nanmax, nanmin, pi
import pywcsgrid2

def implot(image,ext=0,contour=False,spectral=False,griddim=None,save=None, \
        zoom=1,figsize=(20,30),beam=False):

    im = image.image
    if image.header != None:
        hdr = image.header[0]
    xsize=20
    ysize=30
    nx = im.shape[1]
    ny = im.shape[0]
    if zoom > 1:
        x1 = (zoom-1)*nx/(2*zoom)
        x2 = (zoom+1)*nx/(2*zoom)
        y1 = (zoom-1)*ny/(2*zoom)
        y2 = (zoom+1)*ny/(2*zoom)

        im = im[y1:y2,x1:x2,:]
        if image.header != None:
            hdr["CRPIX1"] = (x2-x1)/2+1
            hdr["NAXIS1"] = nx/zoom
            hdr["CRPIX2"] = (y2-y1)/2+1
            hdr["NAXIS2"] = ny/zoom

    if image.header != None:
        gh = pywcsgrid2.GridHelper(wcs=hdr)
        gh.update_delta_trans()
    
    if spectral:
        fig = plt.figure(figsize=figsize)
        if image.header != None:
            grid = ImageGrid(fig, 111, nrows_ncols=griddim,share_all=True, \
                    axes_class=(pywcsgrid2.Axes, dict(grid_helper=gh)))

        for i in range(griddim[0]*griddim[1]):
            norm = mcolors.Normalize()
            norm.vmin = abs(nanmin(im))
            norm.vmax = nanmax(im)
            if contour:
                grid[i].contour(im[:,:,i],origin="lower",colors='k')
            else:
                grid[i].imshow(im[:,:,i],origin="lower", \
                        interpolation='nearest',norm=norm)
            if image.header != None:
                grid[i].set_ticklabel_type("delta")
            grid[i].annotate(r"$v=%5.2f$ km s$^{-1}$" % (image.velocity[i]), \
                    xy=(0.1,0.8), xycoords='axes fraction')
            if beam:
                pixel_size = hdr["CDELT2"]
                bmaj = hdr["BMAJ"]/pixel_size
                bmin = hdr["BMIN"]/pixel_size
                pa = hdr["BPA"]
                x0 = 3*bmaj/2
                y0 = 3*bmaj/2
                ellipse = Ellipse(xy=(x0,y0), width=bmin, height=bmaj, \
                        anble=pa, color='k')
                grid.add_artist(ellipse)
    else:
        if image.header != None:
            plt1 = pywcsgrid2.subplot(111, grid_helper=gh, aspect=float(ny)/nx)
        else:
            plt1 = plt.subplot(111, aspect='equal')
        if contour:
            plt1.contour(im[:,:,ext],origin="lower",colors='k')
        else:
            plt1.imshow(im[:,:,ext],origin="lower",interpolation='nearest')
        if image.header != None:
            plt1.set_ticklabel_type("delta")
        if beam:
            pixel_size = hdr["CDELT2"]
            bmaj = hdr["BMAJ"]/pixel_size
            bmin = hdr["BMIN"]/pixel_size
            pa = hdr["BPA"]
            x0 = 3*bmaj/2
            y0 = 3*bmaj/2
            ellipse = Ellipse(xy=(x0,y0), width=bmin, height=bmaj, angle=pa, \
                    facecolor='k')
            plt1.add_artist(ellipse)
    
    if save != None:
        plt.savefig(save)
    else:
        plt.show()
