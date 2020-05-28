from matplotlib.cm import get_cmap
import matplotlib
from netCDF4 import Dataset
import numpy as np

# NWS Reflectivity Colors (courtesy MetPy library):

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


def var_info(variable, klevel=5):


    if( variable == "W" ):

        _ctable   = get_cmap("YlOrRd")
        _var_min  = 5.0
        _clevels  = np.arange(-45,45.,5.)
        _llevels  = _clevels[::2]
        _variable = "W"
        _pf_label = "W_K%2.2d" % klevel
        _cbar_label = "($ m s^{-1}$)"
        _ndims      = [3]

    elif( variable == "WMAX" ):

        _ctable   = get_cmap("YlOrRd")
        _var_min  = 5.0
        _clevels  = np.arange(_var_min,45.,5.)
        _llevels  = _clevels[::2]
        _variable = "W_UP_MAX"
        _pf_label = "WMAX"
        _cbar_label = "($ m s^{-1}$)"
        _ndims      = [2]

    elif( variable == "UH16" ):

        _ctable   = matplotlib.colors.ListedColormap([c20, c25, c30, c35, c40, c45, c50, c55, c60, c65, c70])
        _var_min  = 20.0
        _clevels  = np.arange(_var_min,350.,50.)
        _llevels  = [100,200,300]
        _variable = "UP_HELI_MAX16"
        _pf_label = "UHMAX16"
        _cbar_label = "($ m^{2}s^{-2}$)"
        _ndims      = [2]

    elif( variable == "UH" ):

        _ctable   = get_cmap("YlGnBu")
        _var_min  = 20.0
        _clevels  = np.arange(_var_min,500.,50.)
        _llevels  = [100,200,300]
        _variable = "UP_HELI_MAX"
        _pf_label = "UHMAX"
        _cbar_label = "($ m^{2}s^{-2}$)"
        _ndims      = [2]

    elif( variable == "T2" ):

        _ctable   = matplotlib.colors.ListedColormap([c20, c25, c30, c35, c40, c45, c50, c55, c60, c65, c70])
        _clevels  = np.arange(252,312,2.0)
        _llevels  = [273.16]
        _variable = "T2"
        _pf_label = "TEMP_SFC"
        _var_min  = 250.0
        _cbar_label = "(K)"
        _ndims      = [2]

    elif( variable == "CREF" ):

        _ctable   = matplotlib.colors.ListedColormap([c20, c25, c30, c35, c40, c45, c50, c55, c60, c65, c70])
        _clevels  = np.arange(0.,75.,5.)
        _llevels  = [25,45]
        _variable = "REFD_MAX"
        _pf_label = "CREF"
        _var_min  = 20.0
        _cbar_label = ""
        _ndims      = [2]

    else:

        _ctable     = matplotlib.colors.ListedColormap([c20, c25, c30, c35, c40, c45, c50, c55, c60, c65, c70])
        _clevels    = np.arange(20.,75.,5.)
        _llevels    = [25,45]
        _variable   = "REFL_10CM"
        _pf_label   = "REF_K%2.2d" % klevel
        _var_min    = 19.9
        _cbar_label = ""
        _ndims      = [3]
        
    return _ctable, _clevels, _llevels, _variable, _ndims, _pf_label, _var_min, _cbar_label
