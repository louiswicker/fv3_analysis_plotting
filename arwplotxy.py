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

    filename = "%s_%3.3imin_%4.2f" % (out_prefix, time, height)
               
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
    ax1.set_aspect('equal', 'datalim')
    ax1.set_title(title, fontsize=10)
    if zoom:
        ax1.set_xlim(1000*zoom[0],1000*zoom[1])
        ax1.set_ylim(1000*zoom[2],1000*zoom[3])

    at = AnchoredText("Max dBZ: %4.1f" % (dbz.max()), loc=4, prop=dict(size=6), frameon=True,)
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax1.add_artist(at)

#---- 

    scale_w_clevels = min(max(N.int(height/1000.), 1.0), 6.0)
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

    if( height <= 1001. ):  # Plot surface pertT
        clevels = N.arange(-12.,13.,1.)
        plot    = ax4.contourf(xx, yy, t, clevels, cmap='bwr')

        divider = make_axes_locatable(ax4)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(plot, cax=cax, orientation=_cbar_orien, label='pertT (K)')

        plot    = ax4.contour(xx, yy, t, clevels[::2], colors='k', linewidths=0.5)
        title = ("SFC Pert. Temperature")
        ax4.set_title(title, fontsize=10)

        if zoom:
            ax4.set_xlim(1000*zoom[0],1000*zoom[1])
            ax4.set_ylim(1000*zoom[2],1000*zoom[3])
 
        at = AnchoredText("Max pertT: %4.1f \n Min pertT: %4.1f" % (t.max(),t.min()), 
                           loc=4, prop=dict(size=6), frameon=True,)
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax4.add_artist(at)

    else:  # Plot wind vectors
        ax4.set_title("Vert Velocity & Horizontal Wind", fontsize=10)
        scale_w_clevels = min(max(N.int(height/1000.), 1.0), 6.0)
        clevels = scale_w_clevels*N.arange(-10.,11.,1.)
        wmask   = np.ma.masked_array(w, mask = [N.abs(w) <= scale_w_clevels*_min_w])
     #  ax4.contourf(xx, yy, wmask, clevels, cmap='bwr')
        ax4.contour(xx, yy, w, clevels[::2], colors='k', linewidths=1.0)

        spd = np.sqrt(t[0,:,:]**2 + t[1,:,:]**2)

        plot = ax4.quiver(xx[::2,::2], yy[::2,::2], t[0,::2,::2], t[1,::2,::2], spd[::2,::2], 
                   pivot='middle', color='black', width=_vec_w, \
                   cmap=plt.get_cmap('YlOrRd'), norm=plt.Normalize(vmin=5.0, vmax=60.), \
                   angles='xy', scale_units='xy', scale=_vector_scale)

        divider = make_axes_locatable(ax4)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(plot, cax=cax, orientation=_cbar_orien, label=('W %s' % ("$m s^{-1}$")))

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
height = options.height
    
plot_prefix = 'arw'
    
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

t  = f.variables['T'][step,:,:,:] + 300.
print("Temp?:  ", t.max(), t.min())
t  = f.variables['T'][step,0,:,:] + 300.
t0 = f.variables['T'][0,0,:,:] + 300.
tp = t/pi[0,:,:] - t0/pi0[0,:,:]

print(tp.max(), tp.min(),t.shape)

wstag  = f.variables['W'][step,:,:,:]
w      = (wstag[:-1]+wstag[1:])/2
print(w.max(), w.min(),w.shape)

wloc = np.unravel_index(np.argmax(w.max(axis=0), axis=None), w.shape[1:])
print(wloc)

ustag = f.variables['U'][step,:,:,:]
u      = (ustag[:,:,:-1]+ustag[:,:,1:])/2
print(u.max(), u.min(),u.shape)
vstag = f.variables['V'][step,:,:,:]
v      = (vstag[:,:-1,:]+vstag[:,1:,:])/2
print(v.max(), v.min(),v.shape)

dbz = f.variables['REFL_10CM'][step,:,:,:]

x = (wloc[1] + 0.5)*dx + dx*np.arange(2*iwidth)
y = (wloc[0] + 0.5)*dx + dx*np.arange(2*jwidth)
xx, yy = np.meshgrid(x, y)

wplot = np.zeros((ny,nx))
dplot = np.zeros((ny,nx))
pplot = np.zeros((ny,nx))


if( height > 1001. ):
    tplot = np.zeros((2,ny,nx))
else:
    tplot = tp

for i in np.arange(nx):
    for j in np.arange(ny):

        wplot[j,i] = np.interp(height,z[:,j,i], w  [:,j,i])
        dplot[j,i] = np.interp(height,z[:,j,i], dbz[:,j,i])
        pplot[j,i] = np.interp(height,z[:,j,i], pp [:,j,i])
 
        if( height > 1001. ):
            tplot[0,j,i] = np.interp(height,z[:,j,i], u [:,j,i])
            tplot[1,j,i] = np.interp(height,z[:,j,i], v [:,j,i])

wplot  = wplot[wloc[0]-jwidth:wloc[0]+jwidth,wloc[1]-iwidth:wloc[1]+iwidth]
dplot  = dplot[wloc[0]-jwidth:wloc[0]+jwidth,wloc[1]-iwidth:wloc[1]+iwidth]
pplot  = pplot[wloc[0]-jwidth:wloc[0]+jwidth,wloc[1]-iwidth:wloc[1]+iwidth]
if( height <= 1001. ):
    tplot  = tplot[wloc[0]-jwidth:wloc[0]+jwidth,wloc[1]-iwidth:wloc[1]+iwidth]
else:
    tplot  = tplot[:,wloc[0]-jwidth:wloc[0]+jwidth,wloc[1]-iwidth:wloc[1]+iwidth]

plot_W_DBZ_T_WZ(wplot, dplot, tplot, pplot, xx, yy, height, time, \
                member=2, glat=0.0, glon=-100., sfc=False, out_prefix=plot_prefix, \
                noshow=None, zoom=None)

# End of file
