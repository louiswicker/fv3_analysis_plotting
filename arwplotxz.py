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

_wz_clevels   = np.arange(-150.,175.,25.)
_w_clevels    = np.arange(-15.,16.,1.)
_dbz_clevels  = [20., 45]
_vector_scale = 3.0

interactive   = True
output_format = "pdf"
_cbar_orien   = 'vertical'

# default time and z-level for plotting
_time    = 90.0
_min_dbz = 10.
_min_w   = 0.01
_vec_w   = 0.005

iwidth = 20
jwidth = 20
dx     = 3.000

figsize = (24,5)

_height = 1000.
_time   = 90

# Other plotting stuff....

#matplotlib.rcParams['contour.negative_linestyle'] = 'solid'

#===============================================================================
def plot_W_DBZ_T_WZ(w, dbz, t, pp, xx, yy, height, time, member, \
                    glat=None, glon=None, sfc=False, \
                    out_prefix=None, noshow=False, zoom=None):

    filename = "%s_%3.3imin_%3.3ikm" % (out_prefix, time, np.floor(height))
               
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True, figsize = figsize)

#---- 

    clevels = N.arange(5.,80., 2.5)
    plot    = ax1.contourf(xx, yy, N.ma.masked_less_equal(dbz,_min_dbz), \
              clevels, cmap=_ref_cmap)

    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(plot, cax=cax, orientation='vertical', label='dBZ')

    plot    = ax1.contour(xx, yy,  dbz, _dbz_clevels[::2], colors='k', linewidths=0.5)
    title   = ("Reflectivity")
    ax1.set_title(title, fontsize=10)
    if zoom:
        ax1.set_xlim(1000*zoom[0],1000*zoom[1])
        ax1.set_ylim(1000*zoom[2],1000*zoom[3])

    at = AnchoredText("Max dBZ: %4.1f" % (dbz.max()), loc=4, prop=dict(size=6), frameon=True,)
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax1.add_artist(at)

#---- 

    scale_w_clevels = 6.0
    clevels = scale_w_clevels*N.arange(-10.,11.,1.)
    wmask   = np.ma.masked_array(w, mask = [N.abs(w) <= scale_w_clevels*_min_w])
    plot    = ax2.contourf(xx, yy, wmask, clevels, cmap='bwr')

    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(plot, cax=cax, orientation=_cbar_orien, label=('W %s' % ("$m s^{-1}$")))

    plot    = ax2.contour(xx, yy, wmask, clevels[::2], colors='k', linewidths=0.5)

    title = ("Vertical Velocity")
    ax2.set_title(title, fontsize=10)
    if zoom:
        ax2.set_xlim(1000*zoom[0],1000*zoom[1])
        ax2.set_ylim(1000*zoom[2],1000*zoom[3])

    at = AnchoredText("Max W: %4.1f \n Min W: %4.1f" % (w.max(),w.min()), \
                      loc=4, prop=dict(size=6), frameon=True,)
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax2.add_artist(at)

#--

    clevels = N.arange(-20.,21.,1.0)
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

    clevels = N.arange(-20.,21.,1.)
    plot    = ax4.contourf(xx, yy, t, clevels, cmap='bwr')

    divider = make_axes_locatable(ax4)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(plot, cax=cax, orientation=_cbar_orien, label='pertT (K)')

    plot    = ax4.contour(xx, yy, t, clevels[::2], colors='k', linewidths=0.5)
    title = ("Pert. Temperature")
    ax4.set_title(title, fontsize=10)

    if zoom:
        ax4.set_xlim(1000*zoom[0],1000*zoom[1])
        ax4.set_ylim(1000*zoom[2],1000*zoom[3])
 
    at = AnchoredText("Max TH: %4.1f \n Min TH: %4.1f" % (t.max(),t.min()), 
                           loc=4, prop=dict(size=6), frameon=True,)
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax4.add_artist(at)

#------- finish

    title = ("\n Time:  %s  min      Y-Plane:  %4.2f km" % (time,height))
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
parser.add_option("-s", "--slide", dest="slide", type="int", default=0, \
                                  help="Move the XZ plot north or south by slide points")

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
        
time = options.time
step = options.time // 5
    
plot_prefix = 'arw_xz'
    
f  = netCDF4.Dataset(options.file, "r")

zstag = (f.variables['PH'][step,:,:]+f.variables['PHB'][step,:,:]) / 9.806
z     = (zstag[:-1]+zstag[1:])/2
print(z.max(), z.min(), z.shape)

nz, ny, nx = z.shape
print(nx,ny,nz)

pbar = f.variables['PB'][0,:,:,:]
pi0  = (1.0e5/pbar)**0.285
p    = f.variables['P'][step,:,:,:] + pbar
pi   = (1.0e5/p)**0.285
pp   = (p - pbar) / 100.

print("P: ",p.max(), p.min())
print("pertP: ",pp.max(), pp.min())
print("PI: ",pi.max(), pi.min())
print("PI0: ",pi.max(), pi.min())

t  = f.variables['T'][step,:,:,:]/pi
t0 = f.variables['T'][0,:,:,:]/pi0
t  = f.variables['T'][step,:,:,:]
t0 = f.variables['T'][0,:,:,:]
tp = t - t0

print(tp.max(), tp.min(),t.shape)

wstag  = f.variables['W'][step,:,:,:]
w      = (wstag[:-1]+wstag[1:])/2
print(w.max(), w.min(),w.shape)

wloc = np.unravel_index(np.argmax(w.max(axis=0), axis=None), w.shape[1:])

wloc = np.array(wloc)

if( options.slide != 0 ):
    print("sliding grid %f km" % (options.slide*dx))
    wloc[0] = wloc[0] + options.slide

print('Wmax locations:  ',wloc)

ustag = f.variables['U'][step,:,:,:]
u      = (ustag[:,:,:-1]+ustag[:,:,1:])/2
print(u.max(), u.min(),u.shape)
vstag = f.variables['V'][step,:,:,:]
v      = (vstag[:,:-1,:]+vstag[:,1:,:])/2
print(v.max(), v.min(),v.shape)

dbz = f.variables['REFL_10CM'][step,:,:,:]

x = (wloc[1] + 0.5)*dx + dx*np.arange(2*iwidth)
y = (wloc[0] + 0.5)*dx + dx*np.arange(2*jwidth)
xx = np.tile(x, nz).reshape(nz,x.size)
zz = z[:,wloc[0],wloc[1]-iwidth:wloc[1]+iwidth] / 1000.

wplot = np.zeros((nz,nx))
dplot = np.zeros((nz,nx))
pplot = np.zeros((nz,nx))
tplot = np.zeros((nz,nx))

wplot  = w[:,wloc[0],wloc[1]-iwidth:wloc[1]+iwidth]
dplot  = dbz[:,wloc[0],wloc[1]-iwidth:wloc[1]+iwidth]
pplot  = pp[:,wloc[0],wloc[1]-iwidth:wloc[1]+iwidth]
tplot  = tp[:,wloc[0],wloc[1]-iwidth:wloc[1]+iwidth]

plot_W_DBZ_T_WZ(wplot, dplot, tplot, pplot, xx, zz, y[jwidth+options.slide], time, \
                member=2, glat=0.0, glon=-100., sfc=False, out_prefix=plot_prefix, \
                noshow=None, zoom=False)
# End of file
