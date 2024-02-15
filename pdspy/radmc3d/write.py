def control(incl_dust=None, incl_lines=None, incl_freefree=None, \
        nphot_therm=None, nphot_scat=None, nphot_spec=None, iseed=None, \
        ifast=None, enthres=None, itempdecoup=None, istar_sphere=None, \
        ntemp=None, temp0=None, temp1=None, scattering_mode_max=None, \
        rto_style=None, camera_tracemode=None, camera_nrrefine=None, \
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
        subbox_z0=None, subbox_z1=None, modified_random_walk=None, \
        mrw_gamma=None, mrw_tauthres=None, mrw_count_trigger=None, \
        setthreads=1, camera_scatsrc_allfreq=None, mc_scat_maxtauabs=None, \
        writeimage_unformatted=None, camera_circ_nrphiinf=None, \
        camera_circ_dbdr=None):

    f = open("radmc3d.inp","w")

    if (incl_dust != None):
        f.write("incl_dust = {0:d}\n".format(incl_dust))
    if (incl_lines != None):
        f.write("incl_lines = {0:d}\n".format(incl_lines))
    if (incl_freefree != None):
        f.write("incl_freefree = {0:d}\n".format(incl_freefree))
    if (nphot_therm != None):
        f.write("nphot_therm = {0:d}\n".format(int(nphot_therm)))
    if (nphot_scat != None):
        f.write("nphot_scat = {0:d}\n".format(int(nphot_scat)))
    if (nphot_spec != None):
        f.write("nphot_spec = {0:d}\n".format(int(nphot_spec)))
    if (iseed != None):
        f.write("iseed = {0:d}\n".format(iseed))
    if (ifast != None):
        f.write("ifast = {0:d}\n".format(ifast))
    if (enthres != None):
        f.write("enthres = {0:f}\n".format(enthres))
    if (itempdecoup != None):
        f.write("itempdecoup = {0:d}\n".format(itempdecoup))
    if (istar_sphere != None):
        f.write("istar_sphere = {0:d}\n".format(istar_sphere))
    if (ntemp != None):
        f.write("ntemp = {0:d}\n".format(ntemp))
    if (temp0 != None):
        f.write("temp0 = {0:f}\n".format(temp0))
    if (temp1 != None):
        f.write("temp1 = {0:f}\n".format(temp1))
    if (scattering_mode_max != None):
        f.write("scattering_mode_max = {0:d}\n".format(scattering_mode_max))
    if (rto_style != None):
        f.write("rto_style = {0:d}\n".format(rto_style))
    if (camera_tracemode != None):
        f.write("camera_tracemode = {0:d}\n".format(camera_tracemode))
    if (camera_nrrefine != None):
        f.write("camera_nrrefine = {0:d}\n".format(camera_nrrefine))
    if (camera_refine_criterion != None):
        f.write("camera_refine_criterion = {0:f}\n".format( \
                camera_refine_criterion))
    if (camera_incl_stars != None):
        f.write("camera_incl_stars = {0:d}\n".format(camera_incl_stars))
    if (camera_starsphere_nrpix != None):
        f.write("camera_starsphere_nrpix = {0:d}\n".format( \
                camera_starsphere_nrpix))
    if (camera_spher_cavity_relres != None):
        f.write("camera_spher_cavity_relres = {0:f}\n".format( \
                camera_spher_cavity_relres))
    if (camera_localobs_projection != None):
        f.write("camera_localobs_projection = {0:d}\n".format( \
                camera_localobs_projection))
    if (camera_min_dangle != None):
        f.write("camera_min_dangle = {0:f}\n".format(camera_min_dangle))
    if (camera_max_dangle != None):
        f.write("camera_max_dangle = {0:f}\n".format(camera_max_dangle))
    if (camera_min_dr != None):
        f.write("camera_min_dr = {0:f}\n".format(camera_min_dr))
    if (camera_diagnostics_subpix != None):
        f.write("camera_diagnostics_subpix = {0:d}\n".format( \
                camera_diagnostics_subpix))
    if (camera_secondorder != None):
        f.write("camera_secondorder = {0:d}\n".format(camera_secondorder))
    if (camera_interpol_jnu != None):
        f.write("camera_interpol_jnu = {0:d}\n".format(camera_interpol_jnu))
    if (camera_scatsrc_allfreq != None):
        f.write("camera_scatsrc_allfreq = {0:d}\n".format(camera_scatsrc_allfreq))
    if (camera_circ_nrphiinf != None):
        f.write("camera_circ_nrphiinf = {0:d}\n".format(camera_circ_nrphiinf))
    if (camera_circ_dbdr != None):
        f.write("camera_circ_dbdr = {0:d}\n".format(camera_circ_dbdr))
    if (mc_weighted_photons != None):
        f.write("mc_weighted_photons = {0:d}\n".format(mc_weighted_photons))
    if (optimized_motion != None):
        f.write("optimized_motion = {0:d}\n".format(optimized_motion))
    if (lines_mode != None):
        f.write("lines_mode = {0:d}\n".format(lines_mode))
    if (lines_maxdoppler != None):
        f.write("lines_maxdoppler = {0:f}\n".format(lines_maxdoppler))
    if (lines_partition_ntempint != None):
        f.write("lines_partition_ntempint = {0:d}\n".format( \
                lines_partition_ntempint))
    if (lines_partition_temp0 != None):
        f.write("lines_partition_temp0 = {0:f}\n".format(lines_partition_temp0))
    if (lines_partition_temp1 != None):
        f.write("lines_partition_temp1 = {0:f}\n".format(lines_partition_temp1))
    if (lines_show_pictograms != None):
        f.write("lines_show_pictograms = {0:d}\n".format(lines_show_pictograms))
    if (tgas_eq_tdust != None):
        f.write("tgas_eq_tdust = {0:d}\n".format(tgas_eq_tdust))
    if (subbox_nx != None):
        f.write("subbox_nx = {0:d}\n".format(subbox_nx))
    if (subbox_ny != None):
        f.write("subbox_ny = {0:d}\n".format(subbox_ny))
    if (subbox_nz != None):
        f.write("subbox_nz = {0:d}\n".format(subbox_nz))
    if (subbox_x0 != None):
        f.write("subbox_x0 = {0:d}\n".format(subbox_x0))
    if (subbox_x1 != None):
        f.write("subbox_x1 = {0:d}\n".format(subbox_x1))
    if (subbox_y0 != None):
        f.write("subbox_y0 = {0:d}\n".format(subbox_y0))
    if (subbox_y1 != None):
        f.write("subbox_y1 = {0:d}\n".format(subbox_y1))
    if (subbox_z0 != None):
        f.write("subbox_z0 = {0:d}\n".format(subbox_z0))
    if (subbox_z1 != None):
        f.write("subbox_z1 = {0:d}\n".format(subbox_z1))
    if (modified_random_walk != None):
        f.write("modified_random_walk = {0:d}\n".format(modified_random_walk))
    if (mrw_gamma != None):
        f.write("mrw_gamma = {0:d}\n".format(mrw_gamma))
    if (mrw_count_trigger != None):
        f.write("mrw_count_trigger = {0:d}\n".format(mrw_count_trigger))
    if (mrw_tauthres != None):
        f.write("mrw_tauthres = {0:d}\n".format(mrw_tauthres))
    if (setthreads != 1):
        f.write("setthreads = {0:d}\n".format(setthreads))
    if (mc_scat_maxtauabs != None):
        f.write("mc_scat_maxtauabs = {0:d}\n".format(mc_scat_maxtauabs))
    if (writeimage_unformatted != None):
        f.write("writeimage_unformatted = {0:d}\n".format(writeimage_unformatted))

    f.close()

def stars(rstar, mstar, lam, xstar, ystar, zstar, tstar=None, fstar=None):

    nstars = len(rstar)
    nlam = len(lam)

    f = open("stars.inp","w")

    f.write(str(2)+"\n")
    f.write("{0:d}  {1:d}\n".format(nstars, nlam))

    for istar in range (nstars):
        f.write("{0:e}   {1:e}   {2:e}   {3:e}   {4:e}\n".format(rstar[istar], \
                mstar[istar], xstar[istar], ystar[istar], zstar[istar]))

    for ilam in range(nlam):
        f.write("{0:e}\n".format(lam[ilam]))

    for istar in range(nstars):
        if (tstar[istar] != 0):
            f.write("{0:f}\n".format(-tstar[istar]))
        else:
            for i in range(nlam):
                f.write("{0:e}\n".format(fstar[ilam]))

    f.close()

def wavelength_micron(lam):

    nlam = len(lam)

    f = open("wavelength_micron.inp","w")

    f.write("{0:d}\n".format(nlam))
    for ilam in range(nlam):
        f.write("{0:f}\n".format(lam[ilam]))

    f.close()

def camera_wavelength_micron(lam):

    nlam = len(lam)

    f = open("camera_wavelength_micron.inp","w")

    f.write("{0:d}\n".format(nlam))
    for ilam in range(nlam):
        f.write("{0:f}\n".format(lam[ilam]))

    f.close()

def amr_grid(x, y, z, gridstyle="regular", coordsystem="cartesian"):

    nx = x.size-1
    ny = y.size-1
    nz = z.size-1

    incl_x = int(nx > 1)
    incl_y = int(ny > 1)
    incl_z = int(nz > 1)

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
    f.write("{0:d}  {1:d}  {2:d}\n".format(incl_x, incl_y, incl_z))
    f.write("{0:d}  {1:d}  {2:d}\n".format(nx, ny, nz))

    if (gridstyle == "octtree"):
        print("OctTree grids not yet implemented.")
    elif (gridstyle == "amr"):
        print("Layer-style AMR grids not yet implemented.")

    for i in range(nx+1):
        f.write("{0:12.9e}\n".format(x[i]))
    for i in range(ny+1):
        f.write("{0:12.9e}\n".format(y[i]))
    for i in range(nz+1):
        f.write("{0:12.9e}\n".format(z[i]))

    # Insert extra info for octtree and amr grids here...

    f.close()

def dust_density(density, gridstyle="normal"):

    nspecies = len(density)

    if (gridstyle == "normal"):
        nx, ny, nz = density[0].shape
        ncells = nx*ny*nz

    f = open("dust_density.inp","w")
    f.write("1\n")
    f.write("{0:d}\n".format(ncells))
    f.write("{0:d}\n".format(nspecies))

    for ispec in range(nspecies):
        if (gridstyle == "normal"):
            for iz in range(nz):
                for iy in range(ny):
                    for ix in range(nx):
                        f.write("{0:e}\n".format(density[ispec][ix,iy,iz]))

    f.close()

def dust_temperature(temperature, gridstyle="normal"):

    nspecies = len(temperature)

    if (gridstyle == "normal"):
        nx, ny, nz = temperature[0].shape
        ncells = nx*ny*nz

    f = open("dust_temperature.dat","w")
    f.write("1\n")
    f.write("{0:d}\n".format(ncells))
    f.write("{0:d}\n".format(nspecies))

    for ispec in range(nspecies):
        if (gridstyle == "normal"):
            for iz in range(nz):
                for iy in range(ny):
                    for ix in range(nx):
                        f.write("{0:f}\n".format(temperature[ispec][ix,iy,iz]))

    f.close()

def dustopac(opacity):

    nspecies = len(opacity)

    f = open("dustopac.inp","w")

    f.write("2\n")
    f.write("{0:d}\n".format(nspecies))
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

        f.write("0\n")
        f.write("{0:s}\n".format(species))
        f.write("----------------------------------------------------------\n")

    f.close()

def dustkappa(species, lam, kabs, ksca=None, g=None):

    if (type(ksca) != type(None)) and (type(g) != type(None)):
        iformat = 3
    elif (type(ksca) != type(None)) and (type(g) == type(None)):
        iformat = 2
    else:
        iformat = 1

    nlam = len(lam)

    f = open("dustkappa_{0:s}.inp".format(species),"w")

    f.write("{0:d}\n".format(iformat))
    f.write("{0:d}\n".format(nlam))

    for ilam in range(nlam):
        if (iformat == 1):
            f.write("{0:e}   {1:e}\n".format(lam[ilam],kabs[ilam]))
        elif (iformat == 2):
            f.write("{0:e}   {1:e}   {2:e}\n".format(lam[ilam], kabs[ilam], \
                    ksca[ilam]))
        elif (iformat == 3):
            f.write("{0:e}   {1:e}   {2:e}   {3:e}\n".format(lam[ilam], \
                    kabs[ilam], ksca[ilam], g[ilam]))

    f.close()

def line(species, inpstyle, colpartners):

    nspecies = len(species)

    f = open("lines.inp", "w")

    f.write("2\n")
    f.write("{0:d}\n".format(nspecies))

    for i in range(nspecies):
        f.write("{0:s} {1:s} 0 0 {2:d}\n".format(species[i], inpstyle[i], \
                len(colpartners[i])))
        if (len(colpartners[i]) > 0):
            for j in range(len(colpartners[i])):
                f.write("{0:a}\n".format(colpartners[i][j]))

    f.close()

def molecule(species, name):

    f = open("molecule_{0:s}.inp".format(name), "w")

    f.write("!MOLECULE\n")
    f.write("{0:s}\n".format(name))
    f.write("!MOLECULAR WEIGHT\n")
    f.write("{0:.1f}\n".format(species.mass))
    f.write("!NUMBER OF ENERGY LEVELS\n")
    f.write("{0:d}\n".format(species.J.size))
    f.write("!LEVEL + ENERGIES(cm^-1) + WEIGHT + J\n")

    for i in range(species.J.size):
        f.write("{0:>5d}{1:>16.9f}{2:>10.1f}{3:>10s}\n".format(i+1, \
                species.E[i], species.g[i], species.J[i]))

    f.write("!NUMBER OF RADIATIVE TRANSITIONS\n")
    f.write("{0:d}\n".format(species.J_u.size))
    f.write("!TRANS + UP + LOW + EINSTEINA(s^-1) + FREQ(GHz) + E_u(K)\n")

    for i in range(species.J_u.size):
        f.write("{0:>5d}{1:>6d}{2:>6d}{3:>12.3e}{4:>16.7f}{5:>10.2f}\n".format(\
                i+1, int(species.J_u[i]), int(species.J_l[i]), \
                species.A_ul[i], species.nu[i]/1.0e9, species.E_u[i]))

    f.write("!NUMBER OF COLL PARTNERS\n")
    f.write("{0:d}\n".format(len(species.partners)))
    
    for i in range(len(species.partners)):
        f.write("!COLLISIONS BETWEEN\n")
        f.write("{0:s}".format(species.partners[i]))
        f.write("!NUMBER OF COLL TRANS\n")
        f.write("{0:d}\n".format(species.J_u_coll[i].size))
        f.write("!NUMBER OF COLL TEMPS\n")
        f.write("{0:d}\n".format(species.temp[i].size))
        f.write("!COLL TEMPS\n")
        temp_string = ""
        for j in range(species.temp[i].size):
            temp_string += "{0:>7.1f}".format(species.temp[i][j])
        f.write(temp_string+"\n")
        f.write("!TRANS + UP + LOW + COLLRATES(cm^3 s^-1)\n")
        for j in range(species.J_u_coll[i].size):
            string = "{0:>5d}{1:>6d}{2:>6d} ".format(j+1, \
                    int(species.J_u_coll[i][j]), int(species.J_l_coll[i][j]))
            for k in range(species.temp[i].size):
                string += "{0:10.3e}".format(species.gamma[i][j,k])
            f.write(string+"\n")
        f.write("\n")

    f.close()

def numberdens(n, species, gridstyle="normal"):

    if (gridstyle == "normal"):
        nx, ny, nz = n.shape
        ncells = nx*ny*nz

    f = open("numberdens_{0:s}.inp".format(species),"w")
    f.write("1\n")
    f.write("{0:d}\n".format(ncells))

    if (gridstyle == "normal"):
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    f.write("{0:e}\n".format(n[ix,iy,iz]))

    f.close()

def gas_temperature(T, gridstyle="normal"):

    if (gridstyle == "normal"):
        nx, ny, nz = T.shape
        ncells = nx*ny*nz

    f = open("gas_temperature.inp","w")
    f.write("1\n")
    f.write("{0:d}\n".format(ncells))

    if (gridstyle == "normal"):
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    f.write("{0:e}\n".format(T[ix,iy,iz]))

    f.close()

def microturbulence(a_turb, gridstyle="normal"):

    if (gridstyle == "normal"):
        nx, ny, nz = a_turb.shape
        ncells = nx*ny*nz

    f = open("microturbulence.inp","w")
    f.write("1\n")
    f.write("{0:d}\n".format(ncells))

    if (gridstyle == "normal"):
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    f.write("{0:e}\n".format(a_turb[ix,iy,iz]))

    f.close()

def gas_velocity(v, gridstyle="normal"):

    if (gridstyle == "normal"):
        nx, ny, nz = v[0].shape
        ncells = nx*ny*nz

    f = open("gas_velocity.inp", "w")
    f.write("1\n")
    f.write("{0:d}\n".format(ncells))

    if (gridstyle == "normal"):
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    f.write("{0:e} {1:e} {2:e}\n".format(v[0][ix,iy,iz], \
                            v[1][ix,iy,iz], v[2][ix,iy,iz]))

    f.close()

def scattering_phase(scattering_phase, freq, gridstyle="normal"):

    nfreq = freq.size

    if (gridstyle == "normal"):
        nx, ny, nz = scattering_phase[0].shape
        ncells = nx*ny*nz

    f = open("scattering_phase.out","w")
    f.write("2\n")
    f.write("{0:d}\n".format(ncells))
    f.write("{0:d}\n".format(nspecies))

    f.write("       ".join(freq)+"\n")

    for ifreq in range(nfreq):
        if (gridstyle == "normal"):
            for iz in range(nz):
                for iy in range(ny):
                    for ix in range(nx):
                        f.write("{0:e}\n".format(density[ifreq,ix,iy,iz]))

    f.close()
