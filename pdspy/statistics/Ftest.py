import numpy

def Ftest(data1, data2, tailed=2):

    N = len(data1)
    M = len(data2)

    mean1 = numpy.mean(data1)
    mean2 = numpy.mean(data2)

    S1 = numpy.sum((data1 - mean1)**2) / (N - 1)
    S2 = numpy.sum((data2 - mean2)**2) / (M - 1)

    F = S1 / S2
    f = numpy.random.f(N-1, M-1, 100000)

    if tailed == 1:
        if F > 1:
            p = len(f[f > F]) / len(f)
        elif F < 1:
            p = len(f[f < F]) / len(f)
    elif tailed == 2:
        if F > 1:
            p = len(f[(f > F) | (f < 1./F)]) / len(f)
        elif F < 1:
            p = len(f[(f > 1./F) | (f < F)]) / len(f)

    return F, p
