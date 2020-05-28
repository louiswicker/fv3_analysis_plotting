import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.cm import get_cmap
from netCDF4 import Dataset
import getopt, sys
from optparse import OptionParser
import warnings 
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import cartopy.crs as ccrs
from wrf_file_vars import var_info

warnings.filterwarnings("ignore") 

from wrf import getvar, to_np, vertcross, smooth2d, CoordPair, get_cartopy, get_basemap, latlon_coords

_wrf_file_string = "wrfout_d01_%s"

# Set some defaults

_plot_type  = "pdf"
datetime    = "2011-04-27_03_00_00"
_dir        = "R24_ieva_FINAL"
_var        = ["WMAX"]
_klevel     = 5


# this is used to make pretty plots when variable >> clevels.max
_clip_variable = True

parser = OptionParser()

parser.add_option("-d",  "--dir", dest="dir",      type="string", default=_dir, help = "Path to WRF run directory")
parser.add_option("-t",  "--time",dest="datetime", type = "string", default=datetime, help = "Usage:  --time 2011-04-27_23_00_00")
parser.add_option("-v",  "--var", dest="variable", type = "string", action="append", default=_var, 
                 help = "Supported WRF Variables:  W (max in column), UH (1-6 km max), REF_1KM, CREF, [default=REF_1KM]") 
parser.add_option(       "--zoom",dest="zoom",     type="float", nargs = 4, default=[None],
                                  help="bounds (lat/lon) of plot - 4 args required: lat_min lat_max lon_min lon_max)")
parser.add_option(       "--display", dest="display", action="store_true", help="Show plot interactively")
parser.add_option("-k",  "--klevel",  dest="klevel", type="int", default=_klevel)

(options, args) = parser.parse_args()

# Open the NetCDF file
try:
    print(os.path.join(options.dir, _wrf_file_string % (options.datetime)))
    ncfile = Dataset(os.path.join(options.dir, _wrf_file_string % (options.datetime)))
except:
    print("\n--> Cannot find %s WRF-ARW file" % (_wrf_file_string % (options.datetime)))
    sys.exit(-1)

for variable in (options.variable + args):

    fig = plt.figure(figsize=(7,7))

    print("Now processing %s \n" % variable)
    
    _ctable, _clevels, _llevels, _variable, _ndims, _pf_label, _var_min, _cbar_label = var_info(variable, klevel=options.klevel)


    print("\n=======================WRF PLOT=====================\n")
    print("WRF directory path:   %s"   % options.dir)
    print("WRF Variable to plot: %s"   % _variable)
    print("Date & time to plot:  %s"   % options.datetime)
    print("NAME of plot:         %s"   % _pf_label)
    print("Min value to plot:    %4.2f"% _var_min)
    if options.zoom[0] != None:
        print("Lat Min of plot:      %4.2f" % options.zoom[0])
        print("Lat Max of plot:      %4.2f" % options.zoom[1])
        print("Lon Min of plot:      %4.2f" % options.zoom[2])
        print("Lon Max of plot:      %4.2f" % options.zoom[3])
    print("\n=======================WRF PLOT=====================\n")


    # Get the WRF variable data structure

    if _ndims[0] == 2:
        var = getvar(ncfile, _variable)
    else:
        var = getvar(ncfile, _variable)[options.klevel]

    # Get the latitude and longitude points
    lats, lons = latlon_coords(var)
    cart_proj  = get_cartopy(var)
    ax_plt     = plt.axes(projection=cart_proj)

    # Draw the oceans, land, and states
    reader   = shpreader.Reader('./UScounties/UScounties.shp')
    counties = list(reader.geometries())
    COUNTIES = cfeature.ShapelyFeature(counties, ccrs.PlateCarree())
    ax_plt.add_feature(COUNTIES, facecolor='none', linewidth=0.5, edgecolor='gray')
    ax_plt.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.5, facecolor='none', edgecolor='black')

    # Make the contour plot for var, mask off min values

    mask_var = np.ma.masked_less_equal(to_np(var),_var_min)
    # If Clip, 
    if _clip_variable:
        mask_var = np.clip(mask_var, _clevels.min(), _clevels.max())

    plt.contour(to_np(lons), to_np(lats), mask_var, levels=_llevels, colors="black", transform=ccrs.PlateCarree(), linewidths=0.25)
    plt.contourf(to_np(lons), to_np(lats), mask_var, _clevels, cmap=_ctable, transform=ccrs.PlateCarree())
    plt.colorbar(ax=ax_plt)
    #cbar.set_label('%s   %s' % (_pf_label,_cbar_label))

    if options.zoom[0] != None:
        x_start, y_start = options.zoom[0], options.zoom[1]
        x_end,   y_end   = options.zoom[2], options.zoom[3]

        ax_plt.set_extent([y_start, y_end, x_start, x_end])

#     else:
#         ax_plt.set_xlim(cartopy_xlim(var))
#         ax_plt.set_ylim(cartopy_ylim(var))

    # Add the gridlines

    # Add titles
    ax_plt.set_title("%s\nDate: %s     Var: %s     Max: %4.1f" % (options.dir, options.datetime[:-3], _variable, var.max()), {"fontsize" : 10})

    plt.savefig("%s_%s_%s.%s" % (options.dir, _pf_label, options.datetime[:-3], _plot_type))

    if(options.display):
        plt.show()
