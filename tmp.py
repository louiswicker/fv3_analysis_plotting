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

