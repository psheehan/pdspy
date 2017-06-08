import numpy

def ttest(data1, data2, tailed=2):

    n = len(data1)
    m = len(data2)
    nu = n + m - 2

    mean1 = numpy.mean(data1)
    mean2 = numpy.mean(data2)

    S1 = numpy.sum((data1 - mean1)**2) / n
    S2 = numpy.sum((data2 - mean2)**2) / m

    s = (n*S1 + m*S2) / nu

    t = (mean1 - mean2) / (numpy.sqrt(s) * numpy.sqrt(m**-1 + n**-1))

    T = numpy.random.standard_t(nu, size=100000)

    if tailed == 1:
        if t < 0:
            p = len(T[T < t]) / len(T)
        elif t > 0:
            p = len(T[T > t]) / len(T)
    elif tailed == 2:
        if t < 0:
            p = len(T[(T < t) | (T > -t)]) / len(T)
        elif t > 0:
            p = len(T[(T > t) | (T < -t)]) / len(T)

    return t, p
