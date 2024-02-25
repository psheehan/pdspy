import numpy as np
import numba

@numba.jit(nopython=True)
def bhmie(X,REFREL,NANG):
    # Declare parameters:
    # Note: important that MXNANG be consistent with dimension of S1 and S2
    #       in calling routine#
    #      PARAMETER(MXNANG=1000,NMXX=15000)
    MXNANG=1000
    NMXX=15000000
    # Arguments:
    S1 = np.repeat(0.+0.j, 2*MXNANG)
    S2 = np.repeat(0.+0.j, 2*MXNANG)
    # Local variables:
    AMU = np.repeat(0., MXNANG)
    PI = np.repeat(0., MXNANG)
    PI0 = np.repeat(0., MXNANG)
    PI1 = np.repeat(0., MXNANG)
    TAU = np.repeat(0., MXNANG)
    D = np.repeat(0.+0.j, NMXX)

    AN = 0.
    BN = 0.
      
    #***********************************************************************
    # Subroutine BHMIE is the Bohren-Huffman Mie scattering subroutine
    #    to calculate scattering and absorption by a homogenous isotropi#
    #    sphere.
    # Given:
    #    X = 2*pi*a/lambda
    #    REFREL = (complex refr. index of sphere)/(real index of medium)
    #    NANG = number of angles between 0 and 90 degrees
    #           (will calculate 2*NANG-1 directions from 0 to 180 deg.)
    #           if called with NANG<2, will set NANG=2 and will compute
    #           scattering for theta=0,90,180.
    # Returns:
    #    S1(1 - 2*NANG-1) = -i*f_22 (incid. E perp. to scatt. plane,
    #                                scatt. E perp. to scatt. plane)
    #    S2(1 - 2*NANG-1) = -i*f_11 (incid. E parr. to scatt. plane,
    #                                scatt. E parr. to scatt. plane)
    #    QEXT = C_ext/pi*a**2 = efficiency factor for extinction
    #    QSCA = C_sca/pi*a**2 = efficiency factor for scattering
    #    QBACK = (dC_sca/domega)/pi*a**2
    #          = backscattering efficiency [NB: this is (1/4*pi) smaller
    #            than the "radar backscattering efficiency"; see Bohren &
    #            Huffman 1983 pp. 120-123]
    #    GSCA = <cos(theta)> for scattering
    #
    # Original program taken from Bohren and Huffman (1983), Appendix A
    # Modified by B.T.Draine, Princeton Univ. Obs., 90/10/26
    # in order to compute <cos(theta)>
    # 91/05/07 (BTD): Modified to allow NANG=1
    # 91/08/15 (BTD): Corrected error (failure to initialize P)
    # 91/08/15 (BTD): Modified to enhance vectorizability.
    # 91/08/15 (BTD): Modified to make NANG=2 if called with NANG=1
    # 91/08/15 (BTD): Changed definition of QBACK.
    # 92/01/08 (BTD): Converted to full double precision and double complex
    #                 eliminated 2 unneed lines of code
    #                 eliminated redundant variables (e.g. APSI,APSI0)
    #                 renamed RN -> EN = double precision N
    #                 Note that DOUBLE COMPLEX and DCMPLX are not part
    #                 of f77 standard, so this version may not be fully
    #                 portable.  In event that portable version is
    #                 needed, use src/bhmie_f77.f
    # 93/06/01 (BTD): Changed AMAX1 to generic function MAX
    #***********************************************************************
    #*** Safety checks
    if NANG > MXNANG:
      raise ValueError('***Error: NANG > MXNANG in bhmie')
    #      IF(NANG.LT.2)NANG=2
    #*** Obtain pi:
    PII=4.*np.arctan(1.)
    DX=X
    DREFRL=REFREL
    Y=X*DREFRL
    YMOD=abs(Y)
    #
    #*** Series expansion terminated after NSTOP terms
    #    Logarithmic derivatives calculated from NMX on down
    XSTOP=X+4.*X**0.3333+2.
    NMX=int(max(XSTOP,YMOD))+15
    # BTD experiment 91/1/15: add one more term to series and compare results
    #      NMX=AMAX1(XSTOP,YMOD)+16
    # test: compute 7001 wavelengths between .0001 and 1000 micron
    # for a=1.0micron SiC grain.  When NMX increased by 1, only a single
    # computed number changed (out of 4*7001) and it only changed by 1/8387
    # conclusion: we are indeed retaining enough terms in series#
    NSTOP=int(XSTOP)
#
    if NMX > NMXX:
        raise ValueError('Error: NMX > NMXX=',NMXX,' for |m|x=',YMOD)
    #*** Require NANG.GE.1 in order to calculate scattering intensities
    DANG=0.
    if NANG > 1:
        DANG=.5*PII/float(NANG-1)
    for J in range(NANG):
        THETA=float(J-1)*DANG
        AMU[J]=np.cos(THETA)

    for J in range(NANG):
        PI0[J]=0.
        PI1[J]=1.

    NN=2*NANG-1
    for J in range(NN):
        S1[J]=0.+0.j
        S2[J]=0.+0.j
    #
    #*** Logarithmic derivative D(J) calculated by downward recurrence
    #    beginning with initial value (0.,0.) at J=NMX
    #
    D[NMX]=0.+0.j
    NN=NMX-1
    for N in range(NN):
        EN=NMX-N+1
        D[NMX-N]=(EN/Y)-(1./(D[NMX-N+1]+EN/Y))
    #
    #*** Riccati-Bessel functions with real argument X
    #    calculated by upward recurrence
    #
    PSI0=np.cos(DX)
    PSI1=np.sin(DX)
    CHI0=-np.sin(DX)
    CHI1=np.cos(DX)
    XI1=PSI1+-CHI1*1j
    QSCA=0.E0
    GSCA=0.E0
    P=-1.
    for N in range(NSTOP):
        EN=N+1
        FN=(2.E0*EN+1.)/(EN*(EN+1.))
        # for given N, PSI  = psi_n        CHI  = chi_n
        #              PSI1 = psi_{n-1}    CHI1 = chi_{n-1}
        #              PSI0 = psi_{n-2}    CHI0 = chi_{n-2}
        # Calculate psi_n and chi_n
        PSI=(2.E0*EN-1.)*PSI1/DX-PSI0
        CHI=(2.E0*EN-1.)*CHI1/DX-CHI0
        XI=PSI+-CHI*1j
        #
        #*** Store previous values of AN and BN for use
        #    in computation of g=<cos(theta)>
        if N > 0:
            AN1=AN
            BN1=BN
        #
        #*** Compute AN and BN:
        AN=(D[N]/DREFRL+EN/DX)*PSI-PSI1
        AN=AN/((D[N]/DREFRL+EN/DX)*XI-XI1)
        BN=(DREFRL*D[N]+EN/DX)*PSI-PSI1
        BN=BN/((DREFRL*D[N]+EN/DX)*XI-XI1)
        #
        #*** Augment sums for Qsca and g=<cos(theta)>
        QSCA=QSCA+(2.*EN+1.)*(np.abs(AN)**2+np.abs(BN)**2)
        GSCA=GSCA+((2.*EN+1.)/(EN*(EN+1.)))*\
            (AN.real*BN.real+AN.imag*BN.imag)
        if N > 0:
            GSCA=GSCA+((EN-1.)*(EN+1.)/EN)*\
                    (AN1.real*AN.real+AN1.imag*AN.imag+\
                    BN1.real*BN.real+BN1.imag*BN.imag)
        #
        #*** Now calculate scattering intensity pattern
        #    First do angles from 0 to 90
        for J in range(NANG):
            JJ=2*NANG-(J+1)
            PI[J]=PI1[J]
            TAU[J]=EN*AMU[J]*PI[J]-(EN+1.)*PI0[J]
            S1[J]=S1[J]+FN*(AN*PI[J]+BN*TAU[J])
            S2[J]=S2[J]+FN*(AN*TAU[J]+BN*PI[J])
        #
        #*** Now do angles greater than 90 using PI and TAU from
        #    angles less than 90.
        #    P=1 for N=1,3,...; P=-1 for N=2,4,...
        P=-P
        for J in range(NANG-1):
            JJ=2*NANG-(J+1)
            S1[JJ]=S1[JJ]+FN*P*(AN*PI[J]-BN*TAU[J])
            S2[JJ]=S2[JJ]+FN*P*(BN*PI[J]-AN*TAU[J])
        PSI0=PSI1
        PSI1=PSI
        CHI0=CHI1
        CHI1=CHI
        XI1=PSI1+-CHI1*1j
        #
        #*** Compute pi_n for next value of n
        #    For each angle J, compute pi_n+1
        #    from PI = pi_n , PI0 = pi_n-1
        for J in range(NANG):
            PI1[J]=((2.*EN+1.)*AMU[J]*PI[J]-(EN+1.)*PI0[J])/EN
            PI0[J]=PI[J]
    #
    #*** Have summed sufficient terms.
    #    Now compute QSCA,QEXT,QBACK,and GSCA
    GSCA=2.*GSCA/QSCA
    QSCA=(2./(DX*DX))*QSCA
    QEXT=(4./(DX*DX))*(S1[1]).real
    QBACK=(np.abs(S1[2*NANG-1])/DX)**2/PII

    return S1,S2,QEXT,QSCA,QBACK,GSCA
