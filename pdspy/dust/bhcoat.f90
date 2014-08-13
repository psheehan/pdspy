      SUBROUTINE BHCOAT(XX,YY,RRFRL1,RRFRL2,QQEXT,QQSCA,QBACK)
C Arguments:
      REAL(kind=8), intent(in) :: XX,YY
      real(kind=8), intent(out) :: QQEXT,QQSCA,QBACK
      COMPLEX(kind=8), intent(in) :: RRFRL1,RRFRL2
C Local variables:
      INTEGER IFLAG,N,NSTOP
      real(kind=8) ::
     &   CHI0Y,CHI1Y,CHIY,DEL,PSI0Y,PSI1Y,PSIY,QEXT,RN,QSCA,X,Y,YSTOP
      COMPLEX(kind=8) ::
     &   AMESS1,AMESS2,AMESS3,AMESS4,AN,ANCAP,
     &   BN,BNCAP,BRACK,
     &   CHI0X2,CHI0Y2,CHI1X2,CHI1Y2,CHIX2,CHIPX2,CHIPY2,CHIY2,CRACK,
     &   D0X1,D0X2,D0Y2,D1X1,D1X2,D1Y2,DNBAR,GNBAR,II,
     &   REFREL,RFREL1,RFREL2,
     &   XBACK,XI0Y,XI1Y,XIY,
     &   X1,X2,Y2
C***********************************************************************
C
C Subroutine BHCOAT calculates Q_ext, Q_sca, Q_back for coated sphere.
C All bessel functions computed by upward recurrence.
C Input:
C        X = 2*PI*RCORE*REFMED/WAVEL
C        Y = 2*PI*RMANT*REFMED/WAVEL
C        RFREL1 = REFCOR/REFMED
C        RFREL2 = REFMAN/REFMED 
C where  REFCOR = complex refr.index of core)
C        REFMAN = complex refr.index of mantle)
C        REFMED = real refr.index of medium)
C        RCORE = radius of core
C        RMANT = radius of mantle
C        WAVEL = wavelength of light in ambient medium
C
C Routine BHCOAT is taken from Bohren & Huffman (1983)
C Obtained from C.L.Joseph
C
C History:
C 92/11/24 (BTD) Explicit declaration of all variables
C***********************************************************************
c
      DATA DEL/1.D-8/,II/(0.D0,1.D0)/
      X=XX
      Y=YY
      RFREL1=RRFRL1
      RFREL2=RRFRL2
c         -----------------------------------------------------------
c              del is the inner sphere convergence criterion
c         -----------------------------------------------------------
      x1 = rfrel1*x
      x2 = rfrel2*x
      y2 = rfrel2*y
      ystop = y + 4.*y**0.3333 + 2.0
      refrel = rfrel2/rfrel1
      nstop = int(ystop)
c         -----------------------------------------------------------
c              series terminated after nstop terms
c         -----------------------------------------------------------
      d0x1 = COS(x1)/SIN(x1)
      d0x2 = COS(x2)/SIN(x2)
      d0y2 = COS(y2)/SIN(y2)
      psi0y = cos(y)
      psi1y = sin(y)
      chi0y = -sin(y)
      chi1y = cos(y)
      xi0y = psi0y-II*chi0y
      xi1y = psi1y-II*chi1y
      chi0y2 = -SIN(y2)
      chi1y2 = COS(y2)
      chi0x2 = -SIN(x2)
      chi1x2 = COS(x2)
      qsca = 0.0
      qext = 0.0
      xback = (0.0,0.0)
      n = 1
      iflag = 0
  200 rn = n
      psiy = (2.0*rn-1.)*psi1y/y - psi0y
      chiy = (2.0*rn-1.)*chi1y/y - chi0y
      xiy = psiy-II*chiy
      d1y2 = 1.0/(rn/y2-d0y2) - rn/y2
      if (iflag .eq. 1) go to 999
      d1x1 = 1.0/(rn/x1-d0x1) - rn/x1
      d1x2 = 1.0/(rn/x2-d0x2) - rn/x2
      chix2 = (2.0*rn - 1.0)*chi1x2/x2 - chi0x2
      chiy2 = (2.0*rn - 1.0)*chi1y2/y2 - chi0y2
      chipx2 = chi1x2 - rn*chix2/x2
      chipy2 = chi1y2 - rn*chiy2/y2
      ancap = refrel*d1x1 - d1x2
      ancap = ancap/(refrel*d1x1*chix2 - chipx2)
      ancap = ancap/(chix2*d1x2 - chipx2)
      brack = ancap*(chiy2*d1y2 - chipy2)
      bncap = refrel*d1x2 - d1x1
      bncap = bncap/(refrel*chipx2 - d1x1*chix2)
      bncap = bncap/(chix2*d1x2 - chipx2)
      crack = bncap*(chiy2*d1y2 - chipy2)
      amess1 = brack*chipy2
      amess2 = brack*chiy2
      amess3 = crack*chipy2
      amess4 = crack*chiy2
      if (ABS(amess1) .gt. del*ABS(d1y2)) go to 999
      if (ABS(amess2) .gt. del) go to 999
      if (ABS(amess3) .gt. del*ABS(d1y2)) go to 999
      if (ABS(amess4) .gt. del) go to 999
      brack = (0.,0.)
      crack = (0.,0.)
      iflag = 1
  999 dnbar = d1y2 - brack*chipy2
      dnbar = dnbar/(1.0-brack*chiy2)
      gnbar = d1y2 - crack*chipy2
      gnbar = gnbar/(1.0-crack*chiy2)
      an = (dnbar/rfrel2 + rn/y)*psiy - psi1y
      an = an/((dnbar/rfrel2+rn/y)*xiy-xi1y)
      bn = (rfrel2*gnbar + rn/y)*psiy - psi1y
      bn = bn/((rfrel2*gnbar+rn/y)*xiy-xi1y)
      qsca = qsca + (2.0*rn+1.0)*(ABS(an)*ABS(an)+ABS(bn)*ABS(bn))
      xback = xback + (2.0*rn+1.0)*(-1.)**n*(an-bn)
      qext = qext + (2.0*rn + 1.0)*(DBLE(an)+DBLE(bn))
      psi0y = psi1y
      psi1y = psiy
      chi0y = chi1y
      chi1y = chiy
      xi1y = psi1y-II*chi1y
      chi0x2 = chi1x2
      chi1x2 = chix2
      chi0y2 = chi1y2
      chi1y2 = chiy2
      d0x1 = d1x1
      d0x2 = d1x2
      d0y2 = d1y2
      n = n + 1
      if (n-1-nstop) 200,300,300
  300 QQSCA = (2.0/(y*y))*qsca
      QQEXT = (2.0/(y*y))*qext
      qback = (ABS(XBACK))**2
      qback = (1.0/(y*y))*qback
      return
      end
