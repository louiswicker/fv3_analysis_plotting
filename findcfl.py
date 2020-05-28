import numpy as np
import netCDF4 as ncdf
import pylab as P

filename = "/scratch/wicker/WRFV3/run/rk3_15s_nofilter/wrfout_d01_2011-04-27_15_00_00"
filename = "/scratch/wicker/WRFV3_aia/run/rk3_25s_fpPrecise/wrfout_d01_2011-04-27_16_00_00"
filename = "/scratch/wicker/WRFV3_aia/run/rk3_20s_aia/wrfrst_d01_2011-04-27_15_00_00"
filename = "/scratch/wicker/WRFV3_aia/run/rk3_25s_debug6/wrfrst_d01_2011-04-27_15_30_00"
filename = "/scratch/wicker/WRFV3_aia/run/rk3_24s_aia/wrfrst_d01_2011-04-27_15_00_00"
filename = "/scratch/wicker/WRFV3_aia/run/rk3_25s_aia/wrfrst_d01_2011-04-27_15_00_00"
filename = "/scratch/wicker/WRFV3_aia/run/rk3_25s_debug7/wrfrst_d01_2011-04-27_15_35_00"
max_locs = 200

def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)


print(" Reading file....  %s " % filename)

f   = ncdf.Dataset(filename)
lon = f.variables['XLONG'][0]
lat = f.variables['XLAT'][0]

dt = f.DT

print(" Time step is: %4.1f " % dt)


#ph  = f.variables['PH'][0]
#phb = f.variables['PHB'][0]
#z   = (ph+phb) / 9.81
#dz  = z[1:] - z[0:-1]

#U = f.variables['U'][0]
#V = f.variables['V'][0]
#W = f.variables['W'][0]

#wcfl = dt * (W[1:] + W[0:-1]) / (2.0*dz)

#----------------------------------------------------------
# These variables are only available in restart files
#   but they give about the same answers regarding CFL, etc.
#   when compared to the wrfout files.

ph  = f.variables['PH_2'][0]
phb = f.variables['PHB'][0]
z   = (ph+phb) / 9.81
mu  = f.variables['MU_2'][0] + f.variables['MUB'][0] 
rdn = f.variables['RDN'][0]
dnw = f.variables['DNW'][0]
W   = f.variables['WW'][0]
den = f.variables['RHO'][0]
U   = f.variables['U_2'][0]
V   = f.variables['V_2'][0]

nz   = rdn.shape[0]
wcfl = 0.5 * dt * (W[1:] + W[0:-1]) * rdn.reshape(rdn.shape[0], 1, 1)
dz   = 0.0*wcfl.copy()

for k in np.arange(nz):
    wcfl[k,:,:] = wcfl[k,:,:] / mu[:,:]
    dz[k,:,:]   = mu[:,:] * dnw[k] / (9.81*den[k,:,:])
    
# get the actual W (m/s) for printing
W = f.variables['W_2'][0]

#----------------------------------------------------------

idx = largest_indices(wcfl,max_locs)

for n in np.arange(max_locs):
   k = idx[0][n]
   j = idx[1][n]
   i = idx[2][n]
   print(" N=%3.3d, K=%3.3d, LAT=%7.3f, LON=%7.3f, OMEGA_CFL=%4.2f, W=%4.2f, W+1=%4.2f, DZ=%4.1f, U=%4.1f  V=%4.1f " % \
         (n, k, lat[j,i], lon[j,i], wcfl[k,j,i], W[k,j,i], W[k+1,j,i], dz[k,j,i], U[k,j,i], V[k,j,i]))


k = idx[0][0]
_w_min = 1.0
ctable = 'viridis'

scale_w_clevels = min(max(np.int(z[k].mean()/1000.), 1.0), 7.0)
clevels = scale_w_clevels*np.arange(-10.,11.,1.)
wmask   = np.ma.masked_array(W[k], mask = [np.abs(W[k]) <= scale_w_clevels*_w_min])
plot    = P.contourf(lon, lat, wmask, clevels, cmap=P.get_cmap(ctable))
#cbar    = P.colorbar(plot,location='right',pad="5%")
plot    = P.contour(lon,lat, wmask, clevels[::2], colors='k', linewidths=0.5)
#cbar.set_label('%s' % ("$m s^{-1}$"))
title = ("Vertical Velocity  K = %d " % (k))
P.title(title, fontsize=10)
P.show()

f.close()

#at = AnchoredText("Max W: %4.1f \n Min W: %4.1f" % (w.max(),w.min()), loc=4, prop=dict(size=6), frameon=True,)
#at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
#ax2.add_artist(at)
