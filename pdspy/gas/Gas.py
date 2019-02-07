from pdspy.constants.physics import m_p, c, h
import numpy
import h5py

class Gas:

    def set_properties_from_lambda(self, filename):
        f = open(filename)

        for i in range(3):
            f.readline()

        self.mass = float(f.readline())

        f.readline()
        nlev = int(f.readline())
        f.readline()

        self.J = numpy.empty(nlev, dtype="<U6")
        self.E = numpy.empty(nlev, dtype=float)
        self.g = numpy.empty(nlev, dtype=float)
        for i in range(nlev):
            temp, self.E[i], self.g[i], self.J[i] = tuple(f.readline().split())

        f.readline()
        ntrans = int(f.readline())
        f.readline()

        self.J_u = numpy.empty(ntrans, dtype=int)
        self.J_l = numpy.empty(ntrans, dtype=int)
        self.A_ul = numpy.empty(ntrans, dtype=float)
        self.nu = numpy.empty(ntrans, dtype=float)
        self.E_u = numpy.empty(ntrans, dtype=float)
        for i in range(ntrans):
            temp, self.J_u[i], self.J_l[i], self.A_ul[i], self.nu[i], \
                    self.E_u[i] = tuple(f.readline().split())

        self.nu *= 1.0e9
        self.B_ul = c**2 * self.A_ul / (2*h*self.nu**3)

        f.readline()
        npartners = int(f.readline())

        self.partners = []
        self.temp = []
        self.J_u_coll = []
        self.J_l_coll = []
        self.gamma = []
        for i in range(npartners):
            f.readline()
            self.partners.append(f.readline())
            f.readline()
            ncolltrans = int(f.readline())
            f.readline()
            ncolltemps = int(f.readline())
            f.readline()
            self.temp.append(numpy.array(f.readline().split(), dtype=float))
            f.readline()

            self.J_u_coll.append(numpy.empty(ncolltrans, dtype=int))
            self.J_l_coll.append(numpy.empty(ncolltrans, dtype=int))
            self.gamma.append(numpy.empty((ncolltrans,ncolltemps), \
                    dtype=float))

            for j in range(ncolltrans):
                temp, self.J_u_coll[i][j], self.J_l_coll[i][j], temp2 = \
                        tuple(f.readline().split(None,3))
                self.gamma[i][j,:] = numpy.array(temp2.split())

        f.close()

    def set_properties_from_file(self, filename=None, usefile=None):
        if (usefile == None):
            f = h5py.File(filename, "r")
        else:
            f = usefile

        self.mass = f['mass'][()]

        self.J = f["J"][...]
        self.E = f['E'][...]
        self.g = f['g'][...]

        self.J_u = f['J_u'][...]
        self.J_l = f['J_l'][...]
        self.A_ul = f['A_ul'][...]
        self.nu = f['nu'][...]
        self.E_u = f['E_u'][...]

        self.B_ul = c**2 * self.A_ul / (2*h*self.nu**3)

        self.partners = []
        self.temp = []
        self.J_u_coll = []
        self.J_l_coll = []
        self.gamma = []
        for name in f["CollisionalPartners"]:
            self.partners.append(name)
            self.temp.append(f["CollisionalPartners"][name] \
                    ["Temperature"][...])
            self.J_u_coll.append(f["CollisionalPartners"][name] \
                    ["J_u_coll"][...])
            self.J_l_coll.append(f["CollisionalPartners"][name] \
                    ["J_l_coll"][...])
            self.gamma.append(f["CollisionalPartners"][name]["Gamma"][...])

        if (usefile == None):
            f.close()

    def write(self, filename=None, usefile=None):
        if (usefile == None):
            f = h5py.File(filename, "w")
        else:
            f = usefile

        f['mass'] = self.mass

        J_dset = f.create_dataset("J", (self.J.size,), dtype=h5py.special_dtype(vlen=str))
        J_dset[...] = self.J
        E_dset = f.create_dataset("E", (self.E.size,), dtype='f')
        E_dset[...] = self.E
        g_dset = f.create_dataset("g", (self.g.size,), dtype='f')
        g_dset[...] = self.g

        J_u_dset = f.create_dataset("J_u", (self.J_u.size,), dtype='f')
        J_u_dset[...] = self.J_u
        J_l_dset = f.create_dataset("J_l", (self.J_l.size,), dtype='f')
        J_l_dset[...] = self.J_l
        A_ul_dset = f.create_dataset("A_ul", (self.A_ul.size,), dtype='f')
        A_ul_dset[...] = self.A_ul
        nu_dset = f.create_dataset("nu", (self.nu.size,), dtype='f')
        nu_dset[...] = self.nu
        E_u_dset = f.create_dataset("E_u", (self.E_u.size,), dtype='f')
        E_u_dset[...] = self.E_u

        collisions = f.create_group("CollisionalPartners")
        partners = []
        temp = []
        J_u_coll = []
        J_l_coll = []
        gamma = []
        for i in range(len(self.partners)):
            partners.append(collisions.create_group("{0:s}".format( \
                    self.partners[i])))

            temp.append(partners[i].create_dataset("Temperature", \
                    (self.temp[i].size,), dtype='f'))
            temp[i][...] = self.temp[i]

            J_u_coll.append(partners[i].create_dataset("J_u_coll", \
                    (self.J_u_coll[i].size,), dtype='f'))
            J_u_coll[i][...] = self.J_u_coll[i]

            J_l_coll.append(partners[i].create_dataset("J_l_coll", \
                    (self.J_l_coll[i].size,), dtype='f'))
            J_l_coll[i][...] = self.J_l_coll[i]

            gamma.append(partners[i].create_dataset("Gamma", \
                    self.gamma[i].shape, dtype='f'))
            gamma[i][...] = self.gamma[i]

        if (usefile == None):
            f.close()
