import numpy as np
import netCDF4 as ncdf
import matplotlib.pylab as plt
from matplotlib.ticker import ScalarFormatter
import os
import matplotlib.ticker as ticker
from scipy import interpolate
from sklearn.neighbors import KernelDensity

w_thresh = 10.0
s_thresh = 25.0
color    = ['black','blue','red','green']
marker   = ['.', 'x', '+']

pintrp = - (900. - 25.*np.arange(33))

if True:
    files = [
             ["R12_MYNN/wrfout_d01_2011-04-27_01_00_00",
              "R12_MYNN/wrfout_d01_2011-04-27_02_00_00",
              "R12_MYNN/wrfout_d01_2011-04-27_03_00_00"],
             ["R20_hrrr_MYNN/wrfout_d01_2011-04-27_01_00_00",
              "R20_hrrr_MYNN/wrfout_d01_2011-04-27_02_00_00",
              "R20_hrrr_MYNN/wrfout_d01_2011-04-27_03_00_00"],
             ["R24_ieva_MYNN/wrfout_d01_2011-04-27_01_00_00",
              "R24_ieva_MYNN/wrfout_d01_2011-04-27_02_00_00",
              "R24_ieva_MYNN/wrfout_d01_2011-04-27_03_00_00"],
            ]
    ic       = 850
    jc       = 350
    ihwidth  = 150
    jhwidth  = 150
else:
    files = [
             ["R12_MYNN/wrfout_d01_2011-04-27_20_00_00",
              "R12_MYNN/wrfout_d01_2011-04-27_21_00_00",
              "R12_MYNN/wrfout_d01_2011-04-27_22_00_00"],
             ["R20_hrrr_MYNN/wrfout_d01_2011-04-27_20_00_00",
              "R20_hrrr_MYNN/wrfout_d01_2011-04-27_21_00_00",
              "R20_hrrr_MYNN/wrfout_d01_2011-04-27_22_00_00"],
             ["R24_ieva_MYNN/wrfout_d01_2011-04-27_20_00_00",
              "R24_ieva_MYNN/wrfout_d01_2011-04-27_21_00_00",
              "R24_ieva_MYNN/wrfout_d01_2011-04-27_22_00_00"],
            ]
    ic       = 1200
    jc       = 425
    ihwidth  = 150
    jhwidth  = 150

def read_profile(file, i0, i1, j0, j1):

    for n, f in enumerate(file):
        name = os.path.dirname(f).split("/")[-1]
        if( n == 1 ): date_time = os.path.basename(f)[-14:-6]
        f1 = ncdf.Dataset(f)
        if n > 0:
            wt = f1.variables['W'][0,:,j0:j1,i0:i1]
            pt = (f1.variables['P'][0,:,j0:j1,i0:i1] + f1.variables['PB'][0,:,j0:j1,i0:i1]) / 100.
            index_pos = np.where(wt.max(axis=0) > w_thresh)
            w1 = 0.5*(wt[1:,index_pos[0],index_pos[1]]+wt[:-1,index_pos[0],index_pos[1]])
            p1 = pt[:,index_pos[0],index_pos[1]]
            wraw = np.concatenate((wraw,w1), axis = 1)
            praw = np.concatenate((praw,p1), axis = 1)
        else:
            wt = f1.variables['W'][0,:,j0:j1,i0:i1]
            pt = (f1.variables['P'][0,:,j0:j1,i0:i1] + f1.variables['PB'][0,:,j0:j1,i0:i1]) / 100.
            index_pos = np.where(wt.max(axis=0) > w_thresh)
            wraw = 0.5*(wt[1:,index_pos[0],index_pos[1]]+wt[:-1,index_pos[0],index_pos[1]])
            praw = pt[:,index_pos[0],index_pos[1]]
        f1.close()

    date_time = date_time.replace("-", "_")
    print("Run name:  %s  Time:  %s  Number of storms:  %d" % (name, date_time, wraw.shape[1]))

    index_pos = np.where(wraw > s_thresh)

    return praw[index_pos], wraw[index_pos], name, date_time

# Plotting stuff now...

n = 0
wprofile = []
pprofile = []
name     = []
for file in files:
    p, wp, fname, date_time = read_profile(file, ic-ihwidth, ic+ihwidth, jc-jhwidth, jc+jhwidth)
    name.append(fname)
    wprofile.append(wp)
    pprofile.append(p)

print(pprofile[0])
fig, ax0 = plt.subplots(figsize=(6,9))

#X_plot = np.linspace(0.0, 15.0, 1000)[:, np.newaxis]

for n in range(len(wprofile)):
#   kde = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(wprofile)
#   log_dens = kde.score_samples(X_plot)
    ax0.scatter(wprofile[n], pprofile[n], marker=marker[n], c=color[n], label=name[n], alpha=0.3, s=5, edgecolors='none')
#   ax0.fill(X_plot[:,0], np.exp(log_dens), c=color[n], label=name[n],alpha=0.1)

ax0.set_title('Vertical Velocity Profiles for %sZ' % date_time)
ax0.set_yticks([1000,900,800,700,600,500,400,300,200,100])
ax0.set_xlabel("W (" + r"${m s}^{-1}$ )" )
ax0.set_ylabel('Pressure (hPa)' )


plt.ylim(1000., 100.)
plt.yscale('log')
for axis in [ax0.yaxis]:
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    axis.set_major_formatter(formatter)
    axis.set_minor_formatter(formatter)

plt.legend(loc='upper right')

plt.savefig("scatter_%sZ.pdf" % date_time)
