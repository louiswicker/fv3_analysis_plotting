import numpy as np
import netCDF4 as ncdf
import matplotlib.pylab as plt
from matplotlib.ticker import ScalarFormatter
import os
import matplotlib.ticker as ticker
from scipy import interpolate
from sklearn.neighbors import KernelDensity
from types import SimpleNamespace

w_thresh = 10.0
_bins    = 2.5*np.arange(21)
color    = ['black','blue','red','green']


records = [  
           [["R12_MYNN/wrfout_d01_2011-04-27_01_00_00", [850, 350, 150, 150]],
            ["R12_MYNN/wrfout_d01_2011-04-27_02_00_00", [850, 350, 150, 150]],
            ["R12_MYNN/wrfout_d01_2011-04-27_03_00_00", [850, 350, 150, 150]],
            ["R12_MYNN/wrfout_d01_2011-04-27_20_00_00", [1200,425, 150, 150]],
            ["R12_MYNN/wrfout_d01_2011-04-27_21_00_00", [1200,425, 150, 150]],
            ["R12_MYNN/wrfout_d01_2011-04-27_22_00_00", [1200,425, 150, 150]]
           ],
           [["R20_hrrr_MYNN/wrfout_d01_2011-04-27_01_00_00", [850, 350, 150, 150]],
            ["R20_hrrr_MYNN/wrfout_d01_2011-04-27_02_00_00", [850, 350, 150, 150]],
            ["R20_hrrr_MYNN/wrfout_d01_2011-04-27_03_00_00", [850, 350, 150, 150]],
            ["R20_hrrr_MYNN/wrfout_d01_2011-04-27_20_00_00", [1200,425, 150, 150]],
            ["R20_hrrr_MYNN/wrfout_d01_2011-04-27_21_00_00", [1200,425, 150, 150]],
            ["R20_hrrr_MYNN/wrfout_d01_2011-04-27_22_00_00", [1200,425, 150, 150]]
           ],
           [["R24_ieva_MYNN/wrfout_d01_2011-04-27_01_00_00", [850, 350, 150, 150]],
            ["R24_ieva_MYNN/wrfout_d01_2011-04-27_02_00_00", [850, 350, 150, 150]],
            ["R24_ieva_MYNN/wrfout_d01_2011-04-27_03_00_00", [850, 350, 150, 150]],
            ["R24_ieva_MYNN/wrfout_d01_2011-04-27_20_00_00", [1200,425, 150, 150]],
            ["R24_ieva_MYNN/wrfout_d01_2011-04-27_21_00_00", [1200,425, 150, 150]],
            ["R24_ieva_MYNN/wrfout_d01_2011-04-27_22_00_00", [1200,425, 150, 150]]
           ]
          ]
def read_profile(record):

    for n, rec in enumerate(record):

        f = rec[0]
        coords = rec[1]
        i0 = coords[0] - coords[2]
        j0 = coords[1] - coords[3]
        i1 = coords[0] + coords[2]
        j1 = coords[1] + coords[3]

        print(f, i0, i1, j0, j1)
    
        name = os.path.dirname(f).split("/")[-1]
        if( n == 1 ): date_time = os.path.basename(f)[-14:-6]
        f1 = ncdf.Dataset(f)

        wloc      = (f1.variables['W'][0,:,j0:j1,i0:i1].max(axis=0)).flatten()
        index_pos = np.where(wloc > w_thresh)

        if n > 0:
            wraw      = np.concatenate((wraw, wloc[wloc>w_thresh]))
        else:
            wraw      = wloc[wloc>w_thresh]

        f1.close()

    date_time = date_time.replace("-", "_")
    print("Run name:  %s  Time:  %s  Number of storms:  %d" % (name, date_time, wraw.shape[0]))

    return wraw, name, date_time

# Plotting stuff now...

n = 0
wprofile = []
name     = []
for rec in records:
    wp, fname, date_time = read_profile(rec)
    name.append(fname)
    wprofile.append(wp)

fig, ax0 = plt.subplots(ncols=3, sharey=True, figsize=(10,8))

for n, ax in enumerate(ax0):
    m, bins, patches = ax.hist(wprofile[n], bins=_bins, color=color[n], label=name[n]) # histogram
    ax.set_xlabel("W (m/s)")
    ax.set_title(name[n])
    ax.set_xlim(10.0, 50.)

fig.suptitle('Vertical Velocity for %sZ' % date_time)
ax0[0].set_ylabel('Number of Updrafts' )

#plt.legend(loc='upper right')

plt.savefig("hist_%sZ.pdf" % date_time)
