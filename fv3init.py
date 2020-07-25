#!/usr/bin/env python3
#
import matplotlib
import pylab as plt
import numpy as np
import sys
import netCDF4
from optparse import OptionParser
import os
import datetime as DT
from matplotlib.offsetbox import AnchoredText
import matplotlib.ticker as ticker
import time as timeit
from cbook2 import *
from metpy.plots import ctables
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Other plotting stuff....
#_ref_norm, _ref_cmap = ctables.registry.get_with_steps('NWSReflectivity', 5, 5)
_ref_cmap = ctables.registry.get_colortable('NWSReflectivity')
_ref_norm = Normalize(5,75)

_wz_clevels  = np.arange(-150.,175.,25.)
_w_clevels   = np.arange(-15.,16.,1.)
_dbz_clevels = [20., 45]
_vector_scale = 3.0

interactive   = True
output_format = "pdf"
_cbar_orien   = 'vertical'

# default time and z-level for plotting
_time    = 90.0
_min_dbz = 10.
_min_w   = 0.01
_vec_w   = 0.005

_iwidth = 30
_jwidth = 30
_dx     = 1.000

figsize = (24,5)

_height = 1000.
_time   = 90

# Other plotting stuff....

#matplotlib.rcParams['contour.negative_linestyle'] = 'solid'

#===============================================================================
def plot_W_DBZ_T_WZ(w, dbz, t, pp, xx, yy, height, time, member, \
                    glat=None, glon=None, sfc=False, \
                    out_prefix=None, vector = False, noshow=False, zoom=None):

    filename = "%s_%2.2dmin_%2.2dkm" % (out_prefix, int(time), int(height/1000.))
               
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True, figsize = figsize)


    at = AnchoredText("Max W: %4.1f \n Min W: %4.1f" % (w.max(),w.min()), \
                      loc=4, prop=dict(size=6), frameon=True,)
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax2.add_artist(at)

#--

    clevels = N.arange(-20.,21.,1.)
    plot    = ax3.contourf(xx, yy, pp, clevels, cmap='bwr')

    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(plot, cax=cax, orientation=_cbar_orien, label='pertP (mb)')

    plot    = ax3.contour(xx, yy, pp, clevels[::2], colors='k', linewidths=0.5)

    title = ("Pert. Pressure in mb)")
    ax3.set_title(title, fontsize=10)

    if zoom:
        ax3.set_xlim(1000*zoom[0],1000*zoom[1])
        ax3.set_ylim(1000*zoom[2],1000*zoom[3])
 
    at = AnchoredText("Max pertP: %4.1f \n Min pertP: %4.1f" % (pp.max(),pp.min()), \
                      loc=4, prop=dict(size=6), frameon=True,)
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax3.add_artist(at)

#---

    clevels = N.arange(-12.,13.,1.)
    plot    = ax4.contourf(xx, yy, t, clevels, cmap='bwr')

    divider = make_axes_locatable(ax4)
    cax     = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(plot, cax=cax, orientation=_cbar_orien, label='pertT (K)')

    plot    = ax4.contour(xx, yy, t, clevels[::2], colors='k', linewidths=0.5)
    title   = ("Pert. Pot. Temperature")
    ax4.set_title(title, fontsize=10)

    if zoom:
        ax4.set_xlim(1000*zoom[0],1000*zoom[1])
        ax4.set_ylim(1000*zoom[2],1000*zoom[3])
 
    at       = AnchoredText("Max pertT: %4.1f \n Min pertT: %4.1f" % (t.max(),t.min()), 
                            loc=4, prop=dict(size=6), frameon=True,)

    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax4.add_artist(at)

#------- finish

    title = ("\n Time:  %s  min      Height:  %4.2f km" % (time,height/1000.))
    fig.suptitle(title, fontsize=12)

    if output_format != None:
        new_filename = "%s.%s" % (filename, output_format)
        print("\n Saving file %s" % (new_filename))
        fig.savefig(new_filename, format=output_format, dpi=300)

    if interactive and not noshow:
        print(filename)
        os.system("open %s" % new_filename)

    return filename

#---------------------------------------------------------------------------------------------------
# Main function defined to return correct sys.exit() calls
#
parser = OptionParser()
parser.add_option("-f", "--file", dest="file", type="string", default= None, \
                                  help="Name of netCDF file from 2d run")
parser.add_option("-t", "--time", dest="time", type="float", default=_time, \
                                  help="Time plot, default is T = %4.0f min" % _time)
parser.add_option("-z", "--height", dest="height", type="float", default=_height, \
                                  help="Height in the file, default is z = %4.0f m" % _height)
parser.add_option("-s", "--sfc",  dest="sfc", action="store_true", help="plot surface temperature")

parser.add_option(      "--vec",  dest="vector", action="store_true", help="plot u/v vectors with updraft")

parser.add_option(      "--dx",   dest="dx", type="float", default=_dx, \
                                  help="Grid spacing for run in km")

(options, args) = parser.parse_args()

if options.file == None:
    print()
    parser.print_help()
    print()
    sys.exit(1)
else:
    if not os.path.exists(options.file):
        print("\nError!  netCDF file does not seem to exist?")
        print("Filename:  %s" % (options.file))
        sys.exit(1)

if options.dx:
    dx = options.dx
    iwidth = _iwidth / dx
else:
    dx = _dx

# iwidth/jwith are plotting window size

iwidth = int(_iwidth / dx)
jwidth = int(_jwidth / dx)
        
time = 0
step = 0
height = options.height
    
plot_prefix = 'fv3_init'
    
f  = netCDF4.Dataset(options.file, "r")

# get the model times
zstag = f.variables['zh'][::-1,:,:]
z     = (zstag[:-1]+zstag[1:])/2

nz, ny, nx = z.shape
print(z.shape)

t  = f.variables['t'][::-1,:,:]
t0 = f.variables['t'][::-1,:,:]
for k in np.arange(nz):
    t0[k,:,:] = t0[k,0,0]

tp = t - t0

print(t.shape)

#tp = t0 * (tp/t0 - 0.285*pp/p0) / pi0

print("Temperature max/min:  ", t.max(), t.min())
print("Pert Theta max/min:  ", tp.max(), tp.min())

wloc = np.array([ny/2, nx/2], dtype='int')
print(wloc)

x = (wloc[1] - iwidth)*dx + dx*np.arange(2*iwidth)
y = (wloc[0] - jwidth)*dx + dx*np.arange(2*jwidth)
xx, yy = np.meshgrid(x, y)

wplot = np.zeros((ny,nx))
dplot = np.zeros((ny,nx))
pplot = np.zeros((ny,nx))

if( options.vector ):
    tplot = np.zeros((2,ny,nx))
else:
    tplot = np.zeros((ny,nx))

for i in np.arange(nx):
    for j in np.arange(ny):
        tplot[j,i] = np.interp(height,z[:,j,i], tp [:,j,i])

wplot = wplot[wloc[0]-jwidth:wloc[0]+jwidth,wloc[1]-iwidth:wloc[1]+iwidth]
dplot = dplot[wloc[0]-jwidth:wloc[0]+jwidth,wloc[1]-iwidth:wloc[1]+iwidth]
pplot = pplot[wloc[0]-jwidth:wloc[0]+jwidth,wloc[1]-iwidth:wloc[1]+iwidth]
tplot  = tplot[wloc[0]-jwidth:wloc[0]+jwidth,wloc[1]-iwidth:wloc[1]+iwidth]

plot_W_DBZ_T_WZ(wplot, dplot, tplot, pplot, xx, yy, height, time, \
                member=2, glat=0.0, glon=-100., sfc=False, out_prefix=plot_prefix, \
                noshow=None, vector=options.vector, zoom=None)

# End of file
