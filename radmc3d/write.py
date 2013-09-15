def control(incl_dust=None, incl_lines=None, incl_freefree=None, \
        nphot_therm=None, nphot_scat=None, iseed=None, ifast=None, \
        enthres=None, itempdecoup=None, istar_sphere=None, ntemp=None, \
        temp0=None, temp1=None, scattering_mode_max=None, rto_style=None, \
        camera_tracemode=None, camera_nrrefine=None, \
        camera_refine_criterion=None, camera_incl_stars=None, \
        camera_starsphere_nrpix=None, camera_spher_cavity_relres=None, \
        camera_localobs_projection=None, camera_min_dangle=None, \
        camera_max_dangle=None, camera_min_dr=None, \
        camera_diagnostics_subpix=None, camera_secondorder=None, \
        camera_interpol_jnu=None, mc_weighted_photons=None, \
        optimized_motion=None, lines_mode=None, lines_maxdoppler=None, \
        lines_partition_ntempint=None, lines_partition_temp0=None, \
        lines_partition_temp1=None, lines_show_pictograms=None, \
        tgas_eq_tdust=None, subbox_nx=None, subbox_ny=None, subbox_nz=None, \
        subbox_x0=None, subbox_x1=None, subbox_y0=None, subbox_y1=None, \
        subbox_z0=None, subbox_z1=None):

    f = open("radmc3d.inp","w")

    if (incl_dust != None):
        f.write("incl_dust = {0:i}\n".format(incl_dust))
    if (incl_lines != None):
        f.write("incl_lines = {0:i}\n".format(incl_lines))
    if (incl_freefree != None):
        f.write("incl_freefree = {0:i}\n".format(incl_freefree))
    if (nphot_therm != None):
        f.write("nphot_therm = {0:i}\n".format(nphot_therm))
    if (nphot_scat != None):
        f.write("nphot_scat = {0:i}\n".format(nphot_scat))
    if (nphot_spec != None):
        f.write("nphot_spec = {0:i}\n".format(nphot_spec))
    if (iseed != None):
        f.write("iseed = {0:i}\n".format(iseed))
    if (ifast != None):
        f.write("ifast = {0:i}\n".format(ifast))
    if (enthres != None):
        f.write("enthres = {0:f}\n".format(enthres))
    if (itempdecoup != None):
        f.write("itempdecoup = {0:i}\n".format(itempdecoup))
    if (istar_sphere != None):
        f.write("istar_sphere = {0:i}\n".format(istar_sphere))
    if (ntemp != None):
        f.write("ntemp = {0:i}\n".format(ntemp))
    if (temp0 != None):
        f.write("temp0 = {0:f}\n".format(temp0))
    if (temp1 != None):
        f.write("temp1 = {0:f}\n".format(temp1))
    if (scattering_mode_max != None):
        f.write("scattering_mode_max = {0:i}\n".format(scattering_mode_max))
    if (rto_style != None):
        f.write("rto_style = {0:i}\n".format(rto_style))
    if (camera_tracemode != None):
        f.write("camera_tracemode = {0:i}\n".format(camera_tracemode))
    if (camera_nrrefine != None):
        f.write("camera_nrrefine = {0:i}\n".format(camera_nrrefine))
    if (camera_refine_criterion != None):
        f.write("camera_refine_criterion = {0:f}\n".format( \
                camera_refine_criterion))
    if (camera_incl_stars != None):
        f.write("camera_incl_stars = {0:i}\n".format(camera_incl_stars))
    if (camera_starsphere_nrpix != None):
        f.write("camera_starsphere_nrpix = {0:i}\n".format( \
                camera_starsphere_nrpix))
    if (camera_spher_cavity_relres != None):
        f.write("camera_spher_cavity_relres = {0:f}\n".format( \
                camera_spher_cavity_relres))
    if (camera_localobs_projection != None):
        f.write("camera_localobs_projection = {0:i}\n".format( \
                camera_localobs_projection))
    if (camera_min_dangle != None):
        f.write("camera_min_dangle = {0:f}\n".format(camera_min_dangle))
    if (camera_max_dangle != None):
        f.write("camera_max_dangle = {0:f}\n".format(camera_max_dangle))
    if (camera_min_dr != None):
        f.write("camera_min_dr = {0:f}\n".format(camera_min_dr))
    if (camera_diagnostics_subpix != None):
        f.write("camera_diagnostics_subpix = {0:i}\n".format( \
                camera_diagnostics_subpix))
    if (camera_secondorder != None):
        f.write("camera_secondorder = {0:i}\n".format(camera_secondorder))
    if (camera_interpol_jnu != None):
        f.write("camera_interpol_jnu = {0:i}\n".format(camera_interpol_jnu))
    if (mc_weighted_photons != None):
        f.write("mc_weighted_photons = {0:i}\n".format(mc_weighted_photons))
    if (optimized_motion != None):
        f.write("optimized_motion = {0:i}\n".format(optimized_motion))
    if (lines_mode != None):
        f.write("lines_mode = {0:i}\n".format(lines_mode))
    if (lines_maxdoppler != None):
        f.write("lines_maxdoppler = {0:f}\n".format(lines_maxdoppler))
    if (lines_partition_ntempint != None):
        f.write("lines_partition_ntempint = {0:i}\n".format( \
                lines_partition_ntempint))
    if (lines_partition_temp0 != None):
        f.write("lines_partition_temp0 = {0:f}\n".format(lines_partition_temp0))
    if (lines_partition_temp1 != None):
        f.write("lines_partition_temp1 = {0:f}\n".format(lines_partition_temp1))
    if (lines_show_pictograms != None):
        f.write("lines_show_pictograms = {0:i}\n".format(lines_show_pictograms))
    if (tgas_eq_tdust != None):
        f.write("tgas_eq_tdust = {0:i}\n".format(tgas_eq_tdust))
    if (subbox_nx != None):
        f.write("subbox_nx = {0:i}\n".format(subbox_nx))
    if (subbox_ny != None):
        f.write("subbox_ny = {0:i}\n".format(subbox_ny))
    if (subbox_nz != None):
        f.write("subbox_nz = {0:i}\n".format(subbox_nz))
    if (subbox_x0 != None):
        f.write("subbox_x0 = {0:i}\n".format(subbox_x0))
    if (subbox_x1 != None):
        f.write("subbox_x1 = {0:i}\n".format(subbox_x1))
    if (subbox_y0 != None):
        f.write("subbox_y0 = {0:i}\n".format(subbox_y0))
    if (subbox_y1 != None):
        f.write("subbox_y1 = {0:i}\n".format(subbox_y1))
    if (subbox_z0 != None):
        f.write("subbox_z0 = {0:i}\n".format(subbox_z0))
    if (subbox_z1 != None):
        f.write("subbox_z1 = {0:i}\n".format(subbox_z1))

    f.close()

def stars(rstar, mstar, lam, xstar, ystar, zstar, tstar=None, fstar=None):

    nstars = len(rstar)
    nlam = len(lam)

    f = open("stars.inp","w")

    f.write(str(2)+"\n")
    f.write("{0:i}  {1:i}\n".format(nstars, nlam))

    for istar in range (nstars):
        f.write("{0:f}   {1:f}   {2:f}   {3:f}   {4:f}\n".format(rstar[istar], \
                mstar[istar], xstar[istar], ystar[istar], zstar[istar]))

    for ilam in range(nlam):
        f.write("{0:f}\n".format(lam[ilam]))

    for istar in range(nstars):
        if (tstar[istar] != 0):
            f.write("{0:f}\n".format(-tstar))
        else:
            for i in range(nlam):
                f.write("{0:f}\n".format(fstar[ilam]))

    f.close()

def wavelength_micron(lam):

    nlam = len(lam)

    f = open("wavelength_micron.inp","w")

    f.write("{0:i}\n".format(nlam))
    for ilam in range(nlam):
        f.write("{0:f}\n".format(lam[ilam]))

    f.close()

def amr_grid(x, y, z, gridstyle="regular", coordsystem="cartesian", \
        incl_x=False, incl_y=False, incl_z=False):

    nx = x.size-1
    ny = y.size-1
    nz = z.size-1

    f = open("amr_grid.inp","w")

    f.write(str(1)+"\n")

    if (gridstyle == "regular"):
        f.write("0\n")
    elif (gridstyle == "octtree"):
        f.write("1\n")
    elif (gridstyle == "amr"):
        f.write("10\n")

    if (coordsystem == "cartesian"):
        f.write("0\n")
    elif (coordsystem == "spherical"):
        f.write("100\n")
    elif (coordsystem == "cylindrical"):
        f.write("200\n")

    f.write("0\n")
    f.write("{0:i}  {1:i}  {2:i}\n".format(incl_x, incl_y, incl_z))
    f.write("{0:i}  {1:i}  {2:i}\n".format(nx, ny, nz))

    if (gridstyle == "octtree"):
        print("OctTree grids not yet implemented.")
    elif (gridstyle == "amr"):
        print("Layer-style AMR grids not yet implemented.")

    for i in range(nx+1):
        f.write(str(x[i])+"\n")
    for i in range(ny+1):
        f.write(str(y[i])+"\n")
    for i in range(nz+1):
        f.write(str(z[i])+"\n")

    # Insert extra info for octtree and amr grids here...

    f.close()

def dust_density(density, gridstyle="normal"):

    nspecies = len(density)

    if (gridstyle == "normal"):
        nx = density[0].shape[2]
        ny = density[0].shape[1]
        nz = density[0].shape[0]
        ncells = nx*ny*nz

    f = open("dust_density.inp","w")
    f.write("1\n")
    f.write("{0:i}\n".format(ncells))
    f.write("{0:i}\n".format(nspecies))

    for ispec in range(nspecies):
        if (gridstyle == "normal"):
            for iz in range(nz):
                for iy in range(ny):
                    for ix in range(nx):
                        f.write("{0:f}\n".format(density[ispec][iz,iy,ix]))

    f.close()

def dust_temperature(temperature, gridstyle="normal"):

    nspecies = len(temperature)

    if (gridstyle == "normal"):
        nx = temperature[0].shape[2]
        ny = temperature[0].shape[1]
        nz = temperature[0].shape[0]
        ncells = nx*ny*nz

    f = open("dust_temperature.dat","w")
    f.write("1\n")
    f.write("{0:i}\n".format(ncells))
    f.write("{0:i}\n".format(nspecies))

    for ispec in range(nspecies):
        if (gridstyle == "normal"):
            for iz in range(nz):
                for iy in range(ny):
                    for ix in range(nx):
                        f.write("{0:f}\n".format(temperature[ispec][iz,iy,ix]))

    f.close()

def dustopac(opacity):

    nspecies = len(opacity)

    f = open("dustopac.inp","w")

    f.write("2\n")
    f.write("{0:i}\n".format(nspecies))
    f.write("==============================================================\n")
    for i in range(nspecies):
        filetype = opacity[i].split("_")[0]
        species = opacity[i].split("_")[1].split(".")[0]

        if (filetype == "dustkappa"):
            f.write("1\n")
        elif (filetype == "dustkapscatmat"):
            f.write("10\n")
        elif (filetype == "dustopac"):
            f.write("-1\n")

        f.write("0               0=Thermal grain\n")
        f.write("{0:s}\n".format(species))
        f.write("----------------------------------------------------------\n")

    f.close()

def dustkappa(species, lam, kabs, ksca=None, g=None):

    if (ksca != None) and (g != None):
        iformat = 3
    elif (ksca != None) and (g == None):
        iformat = 2
    else:
        iformat = 1

    nlam = len(lam)

    f = open("dustkappa_{0:s}.inp".format(species),"w")

    f.write("{0:i}\n".format(iformat))
    f.write("{0:i}\n".format(nlam))

    for ilam in range(nlam):
        if (iformat == 1):
            f.write("{0:f}   {1:f}\n".format(lam[ilam],kabs[ilam]))
        elif (iformat == 2):
            f.write("{0:f}   {1:f}   {2:f}\n".format(lam[ilam], kabs[ilam], \
                    ksca[ilam]))
        elif (iformat == 3):
            f.write("{0:f}   {1:f}   {2:f}   {3:f}\n".format(lam[ilam], \
                    kabs[ilam], ksca[ilam], g[ilam]))

    f.close()
