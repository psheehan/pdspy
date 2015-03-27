import scipy.optimize
import numpy

def leastsq(func, x0, args=(), limits=None, Dfun=None, full_output=0, \
        col_deriv=0, ftol=1.49012e-08, xtol=1.49012e-08, gtol=0.0, maxfev=0, \
        epsfcn=None, factor=100, diag=None):

    def within_bounds(p):
        if (type(limits) != type(None)):
            for i in range(len(p)):
                if ((limits[i]["limited"][0] == True) and \
                        (p[i] < limits[i]["limits"][0])):
                    return False
                elif ((limits[i]["limited"][1] == True) and \
                        (p[i] > limits[i]["limits"][1])):
                    return False

            return True
        else:
            return True

    def residuals(p):
        if within_bounds(p):
            return func(p, *args)
        else:
            return numpy.repeat(1e20, len(func(p, *args)))

    return scipy.optimize.leastsq(residuals, x0, args=(), \
            full_output=full_output, col_deriv=col_deriv, ftol=ftol, \
            xtol=xtol, gtol=gtol, maxfev=maxfev, epsfcn=epsfcn, factor=factor, \
            diag=diag)
