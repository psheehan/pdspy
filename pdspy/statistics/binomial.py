from math import factorial

def binomial(N, m, p):

    return factorial(N)/(factorial(m)*factorial(N-m))*p**m*(1-p)**(N-m)
