import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.cm import get_cmap
from netCDF4 import Dataset
import ctables
import getopt, sys
from optparse import OptionParser
import warnings 

warnings.filterwarnings("ignore") 

from wrf import getvar, to_np, vertcross, smooth2d, CoordPair, get_basemap, latlon_coords

### NWS Reflectivity Colors (courtesy MetPy library):

c5 =  (0.0,                 0.9254901960784314, 0.9254901960784314)
c10 = (0.00392156862745098, 0.6274509803921569, 0.9647058823529412)
c15 = (0.0,                 0.0,                0.9647058823529412)
c20 = (0.0,                 1.0,                0.0)
c25 = (0.0,                 0.7843137254901961, 0.0)
c30 = (0.0,                 0.5647058823529412, 0.0)
c35 = (1.0,                 1.0,                0.0)
c40 = (0.9058823529411765,  0.7529411764705882, 0.0)
c45 = (1.0,                 0.5647058823529412, 0.0)
c50 = (1.0,                 0.0,                0.0)
c55 = (0.8392156862745098,  0.0,                0.0)
c60 = (0.7529411764705882,  0.0,                0.0)
c65 = (1.0,                 0.0,                1.0)
c70 = (0.6,                 0.3333333333333333, 0.788235294117647)
c75 = (0.0,                 0.0,                0.0)

# Create the figure  specs
_plot_type  = "png"
fig         = plt.figure(figsize=(10,7))
ax_plt       = fig.add_subplot(1,1,1)

datetime  = "2019-05-22_18:00:00"
run       = ""
ddir      = "/scratch/wof/realtime/20190522/ic1"
zoom      = [None, None, None, None]

# this is used to make pretty plots when variable >> clevels.max
_clip_variable = False

parser = OptionParser()

parser.add_option("-d",  "--dir", dest="dir",      type="string", default=ddir, 
                                  help = "Path to WRF run directory")
parser.add_option("-e",  "--exp", dest="exp",      type="string", default=run, 
                                  help = "Experiment directory inside WRF run directory")
parser.add_option("-t",  "--time",dest="datetime", type = "string", default=datetime, 
                                  help = "Usage:  --time 2011-04-27_23_00_00")
parser.add_option("-v",  "--var", dest="variable", type = "string", default=None, 
                                  help = "Supported WRF Variables:  W (max in column), UH (1-6 km max), REF_1KM, CREF, [default=REF_1KM]") 
parser.add_option(       "--zoom",dest="zoom",     type="int", nargs = 4, default=zoom,
                                  help="bounds (lat/lon) of plot - 4 args required: lat_min lat_max lon_min lon_max)")

parser.add_option(       "--display", dest="display", action="store_true", help="Show plot interactively")

(options, args) = parser.parse_args()

if( options.variable == "W" ):

    _ctable   = ctables.Positive_Definite
    _ctable   = get_cmap("YlOrRd")
    _var_min  = 5.0
    _clevels  = np.arange(_var_min,45.,5.)
    _llevels  = _clevels[::2]
    _variable = "W_UP_MAX"
    _pf_label = "WMAX"
    _cbar_label = "($ m s^{-1}$)"

elif( options.variable == "UH16" ):

    _ctable   = ctables.Positive_Definite
    _var_min  = 20.0
    _ctable   = get_cmap("YlOrRd")
    _clevels  = np.arange(_var_min,350.,50.)
    _llevels  = [100,200,300]
    _variable = "UP_HELI_MAX16"
    _pf_label = "UHMAX16"
    _cbar_label = "($ m^{2}s^{-2}$)"

elif( options.variable == "UH" ):

    _ctable   = ctables.Positive_Definite
    _var_min  = 20.0
    _ctable   = get_cmap("YlOrRd")
    _clevels  = np.arange(_var_min,350.,50.)
    _llevels  = [100,200,300]
    _variable = "UP_HELI_MAX"
    _pf_label = "UHMAX"
    _cbar_label = "($ m^{2}s^{-2}$)"

elif( options.variable == "T2" ):

    _ctable   = ctables.Not_PosDef_Default
    _clevels  = np.arange(252,312,2.0)
    _ctable   = get_cmap("RdBu_r")
    _llevels  = [273.16]
    _variable = "T2"
    _pf_label = "TEMP_SFC"
    _var_min  = 250.0
    _cbar_label = "(K)"

elif( options.variable == "CREF" ):

    _ctable   = ctables.REF_default
    _clevels  = np.arange(0.,75.,5.)
    _llevels  = [25,45]
    _variable = "COMPOSITE_REFL_10CM"
    _pf_label = "CREF"
    _var_min  = 20.0
    _cbar_label = ""

elif( options.variable == "SMOIS" ):

    _ctable   = ctables.Positive_Definite
    _clevels  = 0.01 * np.arange(0.,50.,1.)
    _llevels  = [0.1,0.5]
    _variable = "SMOIS"
    _pf_label = "SOIL_MOISTURE_TOP_LYR"
    _var_min  = 0.01
    _cbar_label = ""

else:

    _ctable   = ctables.REF_default
    _ctable   = matplotlib.colors.ListedColormap([c20, c25, c30, c35, c40, c45, c50, c55, c60, c65, c70])
    _clevels  = np.arange(20.,75.,5.)
    _llevels  = [25,45]
    _variable = "REFL_10CM"
    _pf_label = "REF1KM_K=5"
    _var_min  = 19.9
    _cbar_label = ""

# Start code....

print("\n=======================WRF PLOT=====================\n")
print("WRF directory path:   %s"   % options.dir)
print("Run to plot:          %s"   % options.exp)
print("WRF Variable to plot: %s"   % _variable)
print("Date & time to plot:  %s"   % options.datetime)
print("NAME of plot:         %s"   % _pf_label)
print("Min value to plot:    %3.1f"% _var_min)
if options.zoom[0] != None:
    print("Lat Min of plot:      %4.1f"% options.zoom[0])
    print("Lat Max of plot:      %4.1f"% options.zoom[1])
    print("Lon Min of plot:     %4.1f" % options.zoom[2])
    print("Lon Max of plot:     %4.1f" % options.zoom[3])
print("\n=======================WRF PLOT=====================\n")

# Open the NetCDF file
try:
    ncfile = Dataset(os.path.join(options.dir, options.exp, "wrfwof_d01_%s" % (options.datetime)))
    suffix = ""
except:
    pass
try:
    ncfile = Dataset(os.path.join(options.dir, options.exp, "wrfout_d01_%s" % (options.datetime)))
    suffix = ""
except:
    pass
try:
    ncfile = Dataset(os.path.join(options.dir, "wrfinput_d01_ic" ))
    suffix = os.path.split(options.dir)[-1]
except:
    print("\n NOPE --- cannot find a wrf netcdf file..... %s" % os.path.join(options.dir, "wrfinput_d01_ic" ))
    sys.exit(-1)


# Get the WRF variables
var = getvar(ncfile, _variable)[0]

# Get the latitude and longitude points
lats, lons = latlon_coords(var)

# Get the basemap object
bm = get_basemap(var)

# Convert the lat/lon points in to x/y points in the projection space
x, y = bm(to_np(lons), to_np(lats))

# Make the contour plot for var, mask off min values

mask_var = np.ma.masked_less_equal(to_np(var),_var_min)
# If Clip, 
if _clip_variable:
    mask_var = np.clip(mask_var, _clevels.min(), _clevels.max())

c1_contour = bm.contour(x, y, mask_var, levels=_llevels, colors="black", zorder=3, linewidths=0.25, ax=ax_plt)

c1_contourf = bm.contourf(x, y, mask_var, _clevels, cmap=_ctable, zorder=2, ax=ax_plt)
cbar        = bm.colorbar(c1_contourf,location='right',pad="5%")
cbar.set_label('%s   %s' % (_pf_label,_cbar_label))

# Draw the oceans, land, and states
bm.drawcoastlines(linewidth=0.25, ax=ax_plt)
bm.drawstates(linewidth=0.25, ax=ax_plt)
bm.drawcountries(linewidth=0.25, ax=ax_plt)
bm.fillcontinents(color=np.array([ 0.9375 , 0.9375 , 0.859375]), ax=ax_plt, lake_color=np.array([ 0.59375 , 0.71484375, 0.8828125 ]))
bm.drawmapboundary(fill_color=np.array([ 0.59375 , 0.71484375, 0.8828125 ]), ax=ax_plt)

# Draw Parallels
parallels = np.arange(np.amin(lats), 30., 2.5)
#bm.drawparallels(parallels, ax=ax_plt, color="white")

merids = np.arange(-100.0, -65.0, 2.5)
#bm.drawmeridians(merids, ax=ax_plt, color="white")

if options.zoom[0] != None:
    x_start, y_start = bm(options.zoom[2], options.zoom[0])
    x_end,   y_end   = bm(options.zoom[3], options.zoom[1])

    ax_plt.set_xlim([x_start, x_end])
    ax_plt.set_ylim([y_start, y_end])

#coord_pairs = to_np(dbz_cross.coords["xy_loc"])
#x_ticks = np.arange(coord_pairs.shape[0])
#x_labels = [pair.latlon_str() for pair in to_np(coord_pairs)]
#ax_plt.set_xticks(x_ticks[::20])
#ax_plt.set_xticklabels(x_labels[::20], rotation=45, fontsize=4)

# Set the x-axis and  y-axis labels
ax_plt.set_xlabel("Latitude", fontsize=5)
ax_plt.set_ylabel("Longitude", fontsize=5)

# Add titles
ax_plt.set_title("%s\nRUN:  %s     %s      Max: %4.1f  Min: %4.1f" % (options.datetime, options.exp, 
                         _variable, mask_var.max(), mask_var.min()), {"fontsize" : 12})

plt.savefig("%s_%s_%s.%s" % (_pf_label, options.datetime, suffix, _plot_type))

if(options.display):
    plt.show()
