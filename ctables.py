from numpy import array
import matplotlib as mpl
LUTSIZE = mpl.rcParams['image.lut']
del mpl

import matplotlib.colors as colors

#These are standard National Weather Service Radar Colortables

NWSRefPrecip = colors.Normalize(5, 75)

NWSRefClearAir = colors.Normalize(-28, 28)

__cmap_data = {

'_NWSRef_data': {
    'blue': [(0.0, 0.92549019607843142, 0.92549019607843142),
             (0.07142857, 0.96470588235294119, 0.96470588235294119),
             (0.14285714, 0.96470588235294119, 0.96470588235294119),
             (0.21428571, 0.0, 0.0),
             (0.28571429, 0.0, 0.0),
             (0.35714286, 0.0, 0.0),
             (0.42857143, 0.0, 0.0),
             (0.50000000, 0.0, 0.0),
             (0.57142857, 0.0, 0.0),
             (0.64285714, 0.0, 0.0),
             (0.71428571, 0.0, 0.0),
             (0.78571429, 0.0, 0.0),
             (0.85714286, 1.0, 1.0),
             (0.92857143, 0.78823529411764703, 0.78823529411764703),
             (1.0, 0.0, 0.0)],
    'green': [(0.0, 0.92549019607843142, 0.92549019607843142),
              (0.07142857, 0.62745098039215685, 0.62745098039215685),
              (0.14285714, 0.0, 0.0),
              (0.21428571, 1.0, 1.0),
              (0.28571429, 0.78431372549019607, 0.78431372549019607),
              (0.35714286, 0.56470588235294117, 0.56470588235294117),
              (0.42857143, 1.0, 1.0),
              (0.50000000, 0.75294117647058822, 0.75294117647058822),
              (0.57142857, 0.56470588235294117, 0.56470588235294117),
              (0.64285714, 0.0, 0.0),
              (0.71428571, 0.0, 0.0),
              (0.78571429, 0.0, 0.0),
              (0.85714286, 0.0, 0.0),
              (0.92857143, 0.33333333333333331, 0.33333333333333331),
              (1.0, 0.0, 0.0)],
    'red': [(0.0, 0.0, 0.0),
            (0.07142857, 0.0039215686274509803, 0.0039215686274509803),
            (0.14285714, 0.0, 0.0),
            (0.21428571, 0.0, 0.0),
            (0.28571429, 0.0, 0.0),
            (0.35714286, 0.0, 0.0),
            (0.42857143, 1.0, 1.0),
            (0.50000000, 0.90588235294117647, 0.90588235294117647),
            (0.57142857, 1.0, 1.0),
            (0.64285714, 1.0, 1.0),
            (0.71428571, 0.83921568627450982, 0.83921568627450982),
            (0.78571429, 0.75294117647058822, 0.75294117647058822),
            (0.85714286, 1.0, 1.0),
            (0.92857143, 0.59999999999999998, 0.59999999999999998),
            (1.0, 0.0, 0.0)]}

'_NWSVel_data':{
    'blue': [(0.0, 0.62352941176470589, 0.62352941176470589),
             (0.071428571428571425, 0.0, 0.0),
             (0.14285714285714285, 0.0, 0.0),
             (0.21428571428571427, 0.0, 0.0),
             (0.2857142857142857, 0.0, 0.0),
             (0.3571428571428571, 0.0, 0.0),
             (0.42857142857142855, 0.0, 0.0),
             (0.5, 0.46666666666666667, 0.46666666666666667),
             (0.5714285714285714, 0.46666666666666667, 0.46666666666666667),
             (0.64285714285714279, 0.0, 0.0),
             (0.71428571428571419, 0.0, 0.0),
             (0.7857142857142857, 0.0, 0.0),
             (0.8571428571428571, 0.0, 0.0),
             (0.92857142857142849, 0.0, 0.0),
             (1.0, 0.0, 0.0)],
    'green': [(0.0, 0.0, 0.0),
              (0.071428571428571425, 1.0, 1.0),
              (0.14285714285714285, 0.90980392156862744, 0.90980392156862744),
              (0.21428571428571427, 0.78431372549019607, 0.78431372549019607),
              (0.2857142857142857, 0.69019607843137254, 0.69019607843137254),
              (0.3571428571428571, 0.56470588235294117, 0.56470588235294117),
              (0.42857142857142855, 0.4392156862745098, 0.4392156862745098),
              (0.5, 0.59215686274509804, 0.59215686274509804),
              (0.5714285714285714, 0.46666666666666667, 0.46666666666666667),
              (0.64285714285714279, 0.0, 0.0),
              (0.71428571428571419, 0.0, 0.0),
              (0.7857142857142857, 0.0, 0.0),
              (0.8571428571428571, 0.0, 0.0),
              (0.92857142857142849, 0.0, 0.0),
              (1.0, 0.0, 0.0)],
    'red': [(0.0, 0.56470588235294117, 0.56470588235294117),
            (0.071428571428571425, 0.0, 0.0),
            (0.14285714285714285, 0.0, 0.0),
            (0.21428571428571427, 0.0, 0.0),
            (0.2857142857142857, 0.0, 0.0),
            (0.3571428571428571, 0.0, 0.0),
            (0.42857142857142855, 0.0, 0.0),
            (0.5, 0.46666666666666667, 0.46666666666666667),
            (0.5714285714285714, 0.59215686274509804, 0.59215686274509804),
            (0.64285714285714279, 0.50196078431372548, 0.50196078431372548),
            (0.71428571428571419, 0.62745098039215685, 0.62745098039215685),
            (0.7857142857142857, 0.72156862745098038, 0.72156862745098038),
            (0.8571428571428571, 0.84705882352941175, 0.84705882352941175),
            (0.92857142857142849, 0.93333333333333335, 0.93333333333333335),
            (1.0, 1.0, 1.0)]}

}

datad    = {}
mylocals = {}
for name in __cmap_data.keys():

    if name.endswith('_data'):
        newname = name[1:-5]
        
        #Put data for colortable into dictionary under new name
        datad[newname] = __cmap_data[name]
        
        print(name)
        #Create colortable from data and place it in local namespace under new name
        mylocals[newname] = colors.LinearSegmentedColormap(newname, __cmap_data[name], LUTSIZE)

#Stolen shamelessly from matplotlib.cm
def get_cmap(name, lut=None):
    if lut is None: lut = LUTSIZE
    
    #If lut is < 0, then return the table with only levels originally defined
    if lut < 0:
        lut = len(datad[name]['red'])
    return colors.LinearSegmentedColormap(name,  datad[name], lut)

#Taken from the matplotlib cookbook

def cmap_map(function,cmap):
    """ Applies function (which should operate on vectors of shape 3:
    [r, g, b], on colormap cmap. This routine will break any discontinuous
    points in a colormap.
    
    Example usage:
    light_jet = cmap_map(lambda x: x/2+0.5, cm.jet)
    """
    cdict = cmap._segmentdata
    step_dict = {}
    # First get the list of points where the segments start or end
    for key in ('red','green','blue'):
        step_dict[key] = map(lambda x: x[0], cdict[key])
    step_list = reduce(lambda x, y: x+y, step_dict.values())
    step_list = array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    reduced_cmap = lambda step : array(cmap(step)[0:3])
    old_LUT = array(map( reduced_cmap, step_list))
    new_LUT = array(map( function, old_LUT))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i,key in enumerate(('red','green','blue')):
        this_cdict = {}
        for j,step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j,i]
            elif new_LUT[j,i]!=old_LUT[j,i]:
                this_cdict[step] = new_LUT[j,i]
        colorvector=  map(lambda x: x + (x[1], ), this_cdict.items())
        colorvector.sort()
        cdict[key] = colorvector

    return colors.LinearSegmentedColormap('colormap',cdict,1024)

if __name__ == '__main__':
    import numpy, pylab
    a=numpy.outer(numpy.arange(0,1,0.01),numpy.ones(10))
    pylab.figure(figsize=(10,7))
    pylab.subplots_adjust(top=0.8,bottom=0.05,left=0.01,right=0.99)
    maps=[m for m in datad.keys() if not m.endswith("_r")]
    maps.sort()
    l=len(maps)+1
    i=1
    for m in maps:
        print(m)
        pylab.subplot(1,l,i)
        pylab.axis("off")
        pylab.imshow(a,aspect='auto',cmap=locals()[m],origin="lower")
        pylab.title(m,rotation=90,fontsize=10)
        i=i+1
    pylab.show()

