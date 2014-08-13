from math import factorial

def fisher_exact(a,b,c,d):

    return float(factorial(a+b))*float(factorial(c+d))*float(factorial(a+c))* \
            float(factorial(b+d))/(float(factorial(a))*float(factorial(b))* \
            float(factorial(c))*float(factorial(d))*float(factorial(a+b+c+d)))
