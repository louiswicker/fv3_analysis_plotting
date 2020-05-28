import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from netCDF4 import Dataset
import ctables
import getopt, sys
from optparse import OptionParser

from wrf import getvar, to_np, vertcross, smooth2d, CoordPair, get_basemap, latlon_coords

_ref_ctable  = ctables.REF_default
_ref_clevels = np.arange(0.,75.,5.)
_min_dbz     = 20.
_variable     = "REFL_10CM_1KM"

datetime = "2011-04-28_00_00_00"
run      = "hrrr_25s_newcode"
ddir     = "/scratch/wicker/HRRR_code/run"
zoom     = [32., 40., -92., -78.]
show_flag = False

parser = OptionParser()

parser.add_option("-e",  "--exp", dest="exp",      type="string", default=ddir, \
                                  help = "Path to WRF run directory")
parser.add_option("-d",  "--dir", dest="dir",      type="string", default=run, \
                                  help = "Experiment directory inside WRF run directory")
parser.add_option("-t",  "--time",dest="datetime", type = "string", 
                                  help = "Usage:  --time 2011-04-27_23_0_0", default=datetime)
parser.add_option(       "--zoom",dest="zoom",     type="int", nargs = 4, default=zoom,
                                  help="bounds (lat/lon) of plot - 4 args required: lat_min lat_max lon_min lon_max)")

parser.add_option(       "--display", dest="display", action="store_true", help="Show plot interactively")

(options, args) = parser.parse_args()

print("WRF directory path:  %s" % options.exp)
print("Run to plot:  %s" % options.dir)
print("Date & time to plot:  %s" % options.datetime)

# Open the NetCDF file
ncfile = Dataset("%s/%s/wrfout_d01_%s" % (options.exp, options.dir, options.datetime))

# Get the WRF variables
dbz = getvar(ncfile, _variable)

# Get the latitude and longitude points
lats, lons = latlon_coords(dbz)

# Create the figure that will have 3 subplots
fig = plt.figure(figsize=(10,7))
ax_dbz = fig.add_subplot(1,1,1)

# Get the basemap object
bm = get_basemap(dbz)

# Convert the lat/lon points in to x/y points in the projection space
x, y = bm(to_np(lons), to_np(lats))

# Make the contour plot for dbz
contour_levels = [25,45]
c1 = bm.contour(x, y, to_np(dbz), levels=contour_levels, colors="black", zorder=3, linewidths=1.0, ax=ax_dbz)

ctt_contours = bm.contourf(x, y, np.ma.masked_less_equal(to_np(dbz),_min_dbz), _ref_clevels, cmap=_ref_ctable, zorder=2, ax=ax_dbz)

# Draw the oceans, land, and states
bm.drawcoastlines(linewidth=0.25, ax=ax_dbz)
bm.drawstates(linewidth=0.25, ax=ax_dbz)
bm.drawcountries(linewidth=0.25, ax=ax_dbz)
bm.fillcontinents(color=np.array([ 0.9375 , 0.9375 , 0.859375]),
                  ax=ax_dbz, lake_color=np.array([ 0.59375 , 0.71484375, 0.8828125 ]))
bm.drawmapboundary(fill_color=np.array([ 0.59375 , 0.71484375, 0.8828125 ]), ax=ax_dbz)

# Draw Parallels
parallels = np.arange(np.amin(lats), 30., 2.5)
bm.drawparallels(parallels, ax=ax_dbz, color="white")

merids = np.arange(-85.0, -72.0, 2.5)
bm.drawmeridians(merids, ax=ax_dbz, color="white")

x_start, y_start = bm(zoom[2], zoom[0])
x_end, y_end = bm(zoom[3], zoom[1])

ax_dbz.set_xlim([x_start, x_end])
ax_dbz.set_ylim([y_start, y_end])

# Set the x-ticks to use latitude and longitude labels.
#coord_pairs = to_np(dbz_cross.coords["xy_loc"])
#x_ticks = np.arange(coord_pairs.shape[0])
#x_labels = [pair.latlon_str() for pair in to_np(coord_pairs)]
#ax_dbz.set_xticks(x_ticks[::20])
#ax_dbz.set_xticklabels(x_labels[::20], rotation=45, fontsize=4)

# Set the x-axis and  y-axis labels
ax_dbz.set_xlabel("Latitude", fontsize=5)
ax_dbz.set_ylabel("Longitude", fontsize=5)

# Add titles
ax_dbz.set_title("RUN:  %s     %s      Max: %4.1f" % (run, _variable, dbz.max()), {"fontsize" : 12})

plt.savefig("%s_%s.png" % (run, datetime))

if(options.display):
    plt.show()
