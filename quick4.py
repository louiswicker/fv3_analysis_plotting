#!/usr/bin/env python
#
import matplotlib
import pylab as P
import numpy as N
import sys
import netCDF4
from optparse import OptionParser
import os
import datetime as DT
from matplotlib.offsetbox import AnchoredText
import matplotlib.ticker as ticker
import time as timeit
from metpy.plots import colortables

interactive   = True
output_format = "pdf"

# default time and z-level for plotting
_time  = 900.
_min_w = 0.1
t00    = 00.0

# Other plotting stuff....

matplotlib.rcParams['contour.negative_linestyle'] = 'solid'

#===============================================================================
def plot_U_W_P_T(u, w, p, t, time, x, y, plot_prefix="workshop"):

    plot_filename = "%s.%s" % (plot_prefix, output_format)

    fig, ((ax1, ax2, ax3, ax4)) = P.subplots(4, 1, sharey=True, sharex=True, figsize=(6,8))

    yy, xx = N.meshgrid(y,x)
    yy     = yy/1000.
    xx     = xx/1000.
               
    clevels = N.arange(-35.,40.,5.)
    norm, cmap = colortables.get_with_steps('viridis', clevels.shape[0],5.)
    plot    = ax1.contourf(xx, yy, u, clevels, cmap=cmap)
#   cbar    = ax1.colorbar(plot,location='right',pad="5%")
#   cbar.set_label("U")
    plot    = ax1.contour(xx, yy, u, clevels[::2], colors='k', linewidths=0.5)
    title   = ("U-Wind (m/s)")
    ax1.set_title(title, loc='left', fontsize=8)
    start, end = ax1.get_xlim()
#   ax1.xaxis.set_ticks(N.arange(start, end+6, 6))
    ax1.xaxis.set_ticks(N.arange(start, end+11, 10))
    ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    start, end = ax1.get_ylim()
    ax1.yaxis.set_ticks(N.arange(start, end, 2))
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))

    at = AnchoredText("Max U: %4.1f \n Min U: %4.1f" % (u.max(),u.min()), loc=1, prop=dict(size=10), frameon=True,)
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax1.add_artist(at)

    if  N.abs(t).max() > 5.0:
        clevels = N.arange(-30.,32.,2.)
    else:
        clevels = N.arange(-1.,1.1,0.1) / 10.
    norm, cmap = colortables.get_with_steps('viridis', clevels.shape[0], clevels[1]-clevels[0])
    wmask   = N.ma.masked_array(w, mask = [N.abs(w) <= _min_w])
    plot    = ax2.contourf(xx, yy, wmask, clevels, cmap=cmap)
#   cbar    = plot.colorbar(plot,location='right',pad="5%")
    plot    = ax2.contour(xx, yy, wmask, clevels[::2], colors='k', linewidths=0.5)
#   cbar.set_label('%s' % ("$m s^{-1}$"))
    title = ("Vertical Velocity (m/s)")
    ax2.set_title(title, loc='left', fontsize=8)

    at = AnchoredText("Max W: %4.1f \n Min W: %4.1f" % (w.max(),w.min()), loc=1, prop=dict(size=10), frameon=True,)
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax2.add_artist(at)

    if  N.abs(t).max() > 5.0:
        clevels = N.arange(-16.,10.,1.)
    else:
        clevels = N.arange(-100,105,5)
        clevels = 0.01*clevels[clevels !=0]
    norm, cmap = colortables.get_with_steps('viridis', clevels.shape[0], clevels[1]-clevels[0])
    tmask   = N.ma.masked_array(t, mask = [N.abs(t) <= 0.00001])
    plot    = ax3.contourf(xx, yy, tmask, clevels, cmap=cmap)
#   cbar    = ax3.colorbar(plot,location='right',pad="5%")
    plot    = ax3.contour(xx, yy, tmask, clevels, colors='k', linewidths=0.5)
#   cbar.set_label('%s' % ("K"))
    title = ("Pert. Potential Temperature (K)")
    ax3.set_title(title, loc='left', fontsize=8)

    at = AnchoredText("Max TH: %5.3f \n Min TH: %5.3f" % (t.max(),t.min()), loc=1, prop=dict(size=10), frameon=True,)
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax3.add_artist(at)

    if  N.abs(t).max() > 5.0:
        clevels = N.arange(-1000.,1000.,100.0)
#       clevels = N.arange(-3.,3.,0.1)
    else:
        clevels = N.arange(-15.,15.5,0.5)

    norm, cmap = colortables.get_with_steps('viridis', clevels.shape[0],  clevels[1]-clevels[0])
    plot    = ax4.contourf(xx, yy, p, clevels, cmap=cmap)
#   cbar    = ax4.colorbar(plot,location='right',pad="5%")
    plot    = ax4.contour(xx, yy, p, clevels[::2], colors='k', linewidths=0.5)
#   cbar.set_label('%s' % ("mb"))
    title = ("Pressure (mb)")
    ax4.set_title(title, fontsize=8)
    ax4.set_xlabel('X km', fontsize=8)

    at = AnchoredText("Max P: %4.1f \n Min P: %4.1f" % (p.max(),p.min()), loc=1, prop=dict(size=10), frameon=True,)
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax4.add_artist(at)

    if int(time) < 1000:
       title = ("Time = %3d sec" % (int(time)))
    else:
       title = ("Time = %4d sec" % (int(time)))

    fig.suptitle(title, y=1.0, fontsize=10)

    P.tight_layout(h_pad=0.5)

    if output_format == "pdf": 
        print("\n Saving file %s" % (plot_filename))
        fig.savefig(plot_filename, format="pdf", dpi=300)

    if output_format == "png": 
        print("\n Saving file %s" % (plot_filename))
        fig.savefig(plot_filename, format="png", dpi=300)

    if interactive:
        print(plot_filename)
        os.system("open %s" % plot_filename)

    return plot_filename

#---------------------------------------------------------------------------------------------------
# Main function defined to return correct sys.exit() calls
#
parser = OptionParser()
parser.add_option("-f", "--file", dest="file", type="string", default= None, \
                                  help="Name of netCDF file from 2d run")
parser.add_option("-t", "--step", dest="step", type="int", default=-1, \
                                  help="Time step to plot, default is last time in file")

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
    
plot_prefix = 'test'
    
f  = netCDF4.Dataset(options.file, "r")

mut_init   = f.variables['MU'][0,1,:] + f.variables['MUB'][0,1,:]
tpert_init = t00 + f.variables['T'][0,:,1,:]
dnw_init   = f.variables['DNW'][0,:]

nx = tpert_init.shape[1]
nz = tpert_init.shape[0]
dx = f.DX
dz = 100.
print(nx, nz)

for k in N.arange(tpert_init.shape[0]):
    tpert_init[k,:] = -dnw_init[k]*mut_init[:]*tpert_init[k,:]

sum_init = tpert_init.sum()

print(("sum of tinit:  %f" % sum_init))

dnw= f.variables['DNW'][options.step,:]
mut= f.variables['MU'][options.step,1,:] + f.variables['MUB'][options.step,1,:]
t  = f.variables['T'][options.step,:,1,:]
p  = f.variables['P'][options.step,:,1,:]
u  = f.variables['U'][options.step,:,1,:]
u  = 0.5*(u[:,0:-1] + u[:,1:])
w  = f.variables['W'][-1,:,1,:]
w  =  0.5*(w[0:-1,:] + w[1:,:])
xc = 0.5*dx + dx*N.arange(nx)
zc = 0.5*dz + dz*N.arange(nz)
yc = 0.5*dy + dy*N.arange(ny)

for k in N.arange(tpert_init.shape[0]):
    tpert_init[k,:] = -dnw[k]*mut[:]*(t00+t[k,:]) - tpert_init[k,:]

sum = tpert_init.sum()
print(("sum of T:  %f" % sum))
print(("Percentage error:  %f" % (100.*(sum)/sum_init)))


if options.step > -1:
    time = options.step * 60
else:
    time = 900.

plot_U_W_P_T(u.transpose(), w.transpose(), p.transpose(), t.transpose(), time, xc, zc,
            plot_prefix=plot_prefix)

# End of file
