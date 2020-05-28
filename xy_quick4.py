#
import matplotlib
import pylab as P
import numpy as N
import sys
import netCDF4
from optparse import OptionParser
from netcdftime import utime
from netCDF4 import num2date
import os
import ctables
import datetime as DT
from mpl_toolkits.basemap import Basemap
from matplotlib.offsetbox import AnchoredText
import time as timeit
from cbook2 import *

interactive = True
output_format = "pdf"

# default time and z-level for plotting
_height  = 1000.
_time    = 3600

_min_dbz = 10.
_min_w   = 0.1

# Other plotting stuff....
_ref_ctable = ctables.REF_default
_wz_clevels = np.arange(-150.,175.,25.)
_w_clevels  = np.arange(-15.,16.,1.)

#===============================================================================
def mymap(x, y, glat, glon, scale = 1.0, ax = None, ticks = True, resolution='c',\
          area_thresh = 10., shape_env = False, counties=False, pickle = False):

    tt = timeit.clock()

    xmax = max(x) / scale
    xmin = min(x) / scale
    ymax = max(y) / scale
    ymin = min(y) / scale

    sw_lat, sw_lon = dxy_2_dll(xmin, ymin, glat, glon, degrees=True)
    ne_lat, ne_lon = dxy_2_dll(xmax, ymax, glat, glon, degrees=True)

    map = Basemap(llcrnrlon=sw_lon, llcrnrlat=sw_lat, \
                  urcrnrlon=ne_lon, urcrnrlat=ne_lat, \
                  lat_1=0.5*(ne_lat+sw_lat), lon_0=0.5*(ne_lon+sw_lon), \
                  projection = 'lcc',      \
                  resolution=resolution,   \
                  area_thresh=area_thresh, \
                  suppress_ticks=ticks, \
                  ax=ax)

    if counties:
        map.drawcounties()

# Shape file stuff

    if shape_env:

        try:
            shapelist = os.getenv("PYESVIEWER_SHAPEFILES").split(":")

            if len(shapelist) > 0:

                for item in shapelist:
                    items = item.split(",")
                    shapefile  = items[0]
                    color      = items[1]
                    linewidth  = items[2]

                    s = map.readshapefile(shapefile,'counties',drawbounds=False)

                    for shape in map.counties:
                        xx, yy = list(zip(*shape))
                        map.plot(xx,yy,color=color,linewidth=linewidth)

        except OSError:
            print("GIS_PLOT:  NO SHAPEFILE ENV VARIABLE FOUND ")

# pickle the class instance.

    print((timeit.clock()-tt,' secs to create original Basemap instance'))

    if pickle:
        pickle.dump(map,open('mymap.pickle','wb'),-1)
        print((timeit.clock()-tt,' secs to create original Basemap instance and pickle it'))

    return map
#===============================================================================
def plot_W_DBZ_T_WZ(w, dbz, t, wz, x, y, height, time, member, glat=None, glon=None, sfc=False, \
                    filename=None, noshow=False, zoom=None):

    if filename == None:
        time = time.replace(year=2000) 
        filename = "%s_%s_%4.2f" % ("wrf", time.strftime('%H:%M:%S'), height)
    else:
        filename = filename
               
    fig = P.figure(figsize = (6,16))

#   fig, ((ax1, ax2), (ax3, ax4)) = P.subplots(2, 2, sharex=True, sharey=True)
    fig, (ax1, ax2) = P.subplots(2, 1, sharex=True)

    map = mymap(x, y, glat, glon, ax = ax1, shape_env=False)

# get coordinates for contour plots

    lon2d, lat2d, xx, yy = map.makegrid(x.size, y.size, returnxy=True)

    clevels = N.arange(0.,75.,5.)
    plot    = map.contourf(xx, yy, N.ma.masked_less_equal(dbz,_min_dbz), clevels, cmap=_ref_ctable)
    cbar    = map.colorbar(plot,location='right',pad="5%")
    cbar.set_label("dBZ")
    plot    = map.contour(xx, yy,  dbz, clevels[::2], colors='k', linewidths=0.5)
    title   = ("Reflectivity")
    ax1.set_title(title, fontsize=10)
    if zoom:
      ax1.set_xlim(1000*zoom[0],1000*zoom[1])
      ax1.set_ylim(1000*zoom[2],1000*zoom[3])

    at = AnchoredText("Max dBZ: %4.1f" % (dbz.max()), loc=4, prop=dict(size=6), frameon=True,)
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax1.add_artist(at)

    map = mymap(x, y, glat, glon, scale = hscale, ax = ax2, shape_env=False)

    scale_w_clevels = min(max(N.int(height/1000.), 1.0), 4.0)
    clevels = scale_w_clevels*N.arange(-15.,16.,1.)
    wmask   = N.ma.masked_array(w, mask = [N.abs(w) <= scale_w_clevels*_min_w])
    plot    = map.contourf(xx, yy, wmask, clevels, cmap=ctables.Not_PosDef_Default)
    cbar    = map.colorbar(plot,location='right',pad="5%")
    plot    = map.contour(xx, yy, wmask, clevels[::2], colors='k', linewidths=0.5)
    cbar.set_label('%s' % ("$m s^{-1}$"))
    title = ("Vertical Velocity")
    ax2.set_title(title, fontsize=10)
    if zoom:
      ax2.set_xlim(1000*zoom[0],1000*zoom[1])
      ax2.set_ylim(1000*zoom[2],1000*zoom[3])

    at = AnchoredText("Max W: %4.1f \n Min W: %4.1f" % (w.max(),w.min()), loc=4, prop=dict(size=6), frameon=True,)
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax2.add_artist(at)

#   map = mymap(x, y, glat, glon, scale = hscale, ax = ax3, shape_env=False)
#
#   clevels = N.arange(-10.,11.,1.)
#   plot    = map.contourf(xx, yy, t, clevels, cmap=ctables.Not_PosDef_Default)
#   cbar    = map.colorbar(plot,location='right',pad="5%")
#   plot    = map.contour(xx, yy, t, clevels[::2], colors='k', linewidths=0.5)
#   cbar.set_label('%s' % ("K"))
#   if sfc:
#       title = ("SFC Pert. Potential Temperature")
#   else:
#       title = ("Pert. Potential Temperature")
#   ax3.set_title(title, fontsize=10)
#   if zoom:
#     ax3.set_xlim(1000*zoom[0],1000*zoom[1])
#     ax3.set_ylim(1000*zoom[2],1000*zoom[3])
#
#   at = AnchoredText("Max TH: %4.1f \n Min TH: %4.1f" % (t.max(),t.min()), loc=4, prop=dict(size=6), frameon=True,)
#   at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
#   ax3.add_artist(at)
#
#   map = mymap(x, y, glat, glon, scale = hscale, ax = ax4, shape_env=False)
#
#   s_wz    = wz*10000.
#   plot    = map.contourf(xx, yy, s_wz, _wz_clevels, cmap=ctables.Not_PosDef_Default)
#   cbar    = map.colorbar(plot,location='right',pad="5%")
#   plot    = map.contour(xx, yy, s_wz, _wz_clevels[::2], colors='k', linewidths=0.5)
#   cbar.set_label('%s' % ("x $ 10^{4}s^{-1}$"))
#   if sfc:
#       title = ("SFC Vert. Vorticity")
#   else:
#       title = ("Vert. Vorticity")
#   ax4.set_title(title, fontsize=10)
#   if zoom:
#     ax4.set_xlim(1000*zoom[0],1000*zoom[1])
#     ax4.set_ylim(1000*zoom[2],1000*zoom[3])
#
#   at = AnchoredText("Max Wz: %4.1f \n Min Wz: %4.1f" % (s_wz.max(),s_wz.min()), loc=4, prop=dict(size=6), frameon=True,)
#   at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
#   ax4.add_artist(at)

#   title = ("\n Time:  %s       Height:  %4.2f km" % (time.strftime('%H:%M:%S'),height/1000.))
#   fig.suptitle(title, fontsize=8)

    if output_format != None:
        new_filename = "%s.%s" % (filename, output_format)
        print("\n Saving file %s" % (new_filename))
        fig.savefig(new_filename, format=output_format, dpi=300)

    if interactive and not noshow:
        print(filename)
        os.system("display %s" % new_filename)

    return filename

#---------------------------------------------------------------------------------------------------
# Main function defined to return correct sys.exit() calls
#
parser = OptionParser()
parser.add_option("-f", "--file", dest="file", type="string", default= None, \
                                  help="Name of netCDF file created from oban analysis")
parser.add_option("-o", "--output", dest="outfile", type="string", default= None, \
                                    help="Name of PNG file to be written out")
parser.add_option("-t", "--time", dest="time",   type="int",   help = "Model time seconds from initial time")
parser.add_option("-z", "--height", dest="height", type="float", help = "Model height in meters...1000., 2000., ")
parser.add_option("-s", "--sfc",    dest="sfc",    action="store_true", help="plot surface vorticity and pert pot temp")
parser.add_option("--noshow", dest="noshow", action="store_true", help="dont show plot interactively")
parser.add_option(       "--zoom",     dest="zoom", type="int", nargs = 4, help="bounds (km) of plot - 4 args required: xmin xmax ymin ymax)")


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
    
if options.time == None:
    print()
    print(("\nNo time supplied, using default time: %4.2f\n" % _time))
    print()
    time = _time
else:
    time = options.time
    
if options.height == None:
    print()
    print(("\nNo height supplied, using default height: %4.2f\n" % _height))
    print()
    height = _height
else:
    height = options.height

file_obj  = netCDF4.Dataset(options.file, "r")
glat      = 35.
glon      = 99.
tindex    = time
member    = 0
if member == "analysis":  member = 0

tindex, valid_DT    = GetTimeIndex(file_obj, time, closest=False)
print(tindex, valid_DT)
coords    = GetCoords(file_obj, tindex=tindex)
xc        = coords[0][0]
yc        = coords[0][1]
zc        = coords[0][2]
ze        = coords[1][2]

print(zc)

zb, zt, dzb, dzt, dz = interp_weights(height, ze)
wplot     = (file_obj.variables['W'][tindex,zb,:,:]*dzb + file_obj.variables['W'][tindex,zt,:,:]*dzt) / dz
dplot     = (file_obj.variables['REFL_10CM'][tindex,zb,:,:]*dzb + file_obj.variables['REFL_10CM'][tindex,zt,:,:]*dzt) / dz

if options.sfc:
    tplot     = file_obj.variables['T'][tindex,0,:,:] - file_obj.variables['T'][0,0,0,0]
    wzplot    = ComputeWZ(xc, yc, file_obj.variables['U'][tindex,0,:,:], file_obj.variables['V'][tindex,0,:,:])
else:
    tplot     = ((file_obj.variables['T'][tindex,zb,:,:]*dzb + file_obj.variables['T'][tindex,zt,:,:]*dzt) / dz) \
              - ((file_obj.variables['T'][0,zb,0,0]*dzb + file_obj.variables['T'][0,zt,0,0]*dzt) / dz)
    wzplot    = (ComputeWZ(xc, yc, file_obj.variables['U'][tindex,zb,:,:], file_obj.variables['V'][tindex,zb,:,:])*dzb
                +ComputeWZ(xc, yc, file_obj.variables['U'][tindex,zt,:,:], file_obj.variables['V'][tindex,zt,:,:])*dzt) / dz

plot_W_DBZ_T_WZ(wplot, dplot, tplot, wzplot, xc, yc, height, valid_DT, \
                member=member, glat=glat, glon=glon, sfc=options.sfc, filename=options.outfile, \
                noshow=options.noshow, zoom=options.zoom)

# End of file
