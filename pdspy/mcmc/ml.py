from numpy import sqrt, exp, pi

def ml(x, y, sigma_y, params, model, args=None, limits=None):

    if (type(limits) != type(None)):
        for i in range(params.size):
            if ((limits[i]["limited"][0] == True) and \
                    (params[i] < limits[i]["limits"][0])):
                ml = 0.0
                chisq = 1.0e300
                return ml, chisq
            elif ((limits[i]["limited"][1] == True) and \
                    (params[i] > limits[i]["limits"][1])):
                ml = 0.0
                chisq = 1.0e300
                return ml, chisq

    m = model(x, params, **args)

    mlarr = 1./sqrt(2*pi*sigma_y**2)*\
                exp(-(y-m)**2/(2*sigma_y**2)) 

    #ml = (mlarr**(1./(y.size-params.size))).prod()
    ml = 1.
    chisq = ((y-m)**2/sigma_y**2).sum()

    return ml, chisq
