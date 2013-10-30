from pdspy.constants.physics import m_p
import numpy

class Gas:

    def set_properties_from_lambda(self, filename):
        f = open(filename)

        for i in range(3):
            f.readline()

        self.mass = float(f.readline()) * m_p

        f.readline()
        nlev = int(f.readline())
        f.readline()

        self.J = numpy.empty(nlev, dtype=int)
        self.E = numpy.empty(nlev, dtype=float)
        self.g = numpy.empty(nlev, dtype=float)
        for i in range(nlev):
            temp, self.E[i], self.g[i], self.J[i] = tuple(f.readline().split())

        f.readline()
        ntrans = int(f.readline())
        f.readline()

        self.J_u = numpy.empty(ntrans, dtype=int)
        self.J_l = numpy.empty(ntrans, dtype=int)
        self.A = numpy.empty(ntrans, dtype=float)
        self.nu = numpy.empty(ntrans, dtype=float)
        self.E_u = numpy.empty(ntrans, dtype=float)
        for i in range(ntrans):
            temp, self.J_u[i], self.J_l[i], self.A[i], self.nu[i], self.E_u = \
                    tuple(f.readline().split())

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
            self.gamma.append(numpy.empty((npartners,ncolltrans,ncolltemps), \
                    dtype=float))

            for j in range(ncolltrans):
                temp, self.J_u_coll[i][j], self.J_u_coll[i][j], temp2 = \
                        tuple(f.readline().split(None,3))
                self.gamma[i][0,j,:] = numpy.array(temp2.split())

        f.close()
