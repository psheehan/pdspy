      SUBROUTINE BHMIE(X,REFREL,NANG,S1,S2,QEXT,QSCA,QBACK,GSCA)
! Declare parameters:
! Note: important that MXNANG be consistent with dimension of S1 and S2
!       in calling routine!
!      PARAMETER(MXNANG=1000,NMXX=15000)
      INTEGER, PARAMETER :: MXNANG=1000,NMXX=15000000
! Arguments:
      INTEGER, intent(in) :: NANG
      REAL(kind=8), intent(out) :: GSCA,QBACK,QEXT,QSCA
      real(kind=8), intent(in) :: X
      COMPLEX(kind=8), intent(in) :: REFREL
      COMPLEX(kind=8), intent(out), dimension(2*MXNANG-1) :: S1,S2
! Local variables:
      INTEGER :: J,JJ,N,NSTOP,NMX,NN
      real(kind=8) :: CHI,CHI0,CHI1,DANG,DX,EN,FN,P,PII,PSI,PSI0,PSI1,
     &                 THETA,XSTOP,YMOD
      real(kind=8), dimension(mxnang) :: AMU,PI,PI0,PI1,
     &                 TAU
      COMPLEX(kind=8) :: AN,AN1,BN,BN1,DREFRL,XI,XI1,Y
      COMPLEX(kind=8),dimension(nmxx) :: D
      
!***********************************************************************
! Subroutine BHMIE is the Bohren-Huffman Mie scattering subroutine
!    to calculate scattering and absorption by a homogenous isotropi!
!    sphere.
! Given:
!    X = 2*pi*a/lambda
!    REFREL = (complex refr. index of sphere)/(real index of medium)
!    NANG = number of angles between 0 and 90 degrees
!           (will calculate 2*NANG-1 directions from 0 to 180 deg.)
!           if called with NANG<2, will set NANG=2 and will compute
!           scattering for theta=0,90,180.
! Returns:
!    S1(1 - 2*NANG-1) = -i*f_22 (incid. E perp. to scatt. plane,
!                                scatt. E perp. to scatt. plane)
!    S2(1 - 2*NANG-1) = -i*f_11 (incid. E parr. to scatt. plane,
!                                scatt. E parr. to scatt. plane)
!    QEXT = C_ext/pi*a**2 = efficiency factor for extinction
!    QSCA = C_sca/pi*a**2 = efficiency factor for scattering
!    QBACK = (dC_sca/domega)/pi*a**2
!          = backscattering efficiency [NB: this is (1/4*pi) smaller
!            than the "radar backscattering efficiency"; see Bohren &
!            Huffman 1983 pp. 120-123]
!    GSCA = <cos(theta)> for scattering
!
! Original program taken from Bohren and Huffman (1983), Appendix A
! Modified by B.T.Draine, Princeton Univ. Obs., 90/10/26
! in order to compute <cos(theta)>
! 91/05/07 (BTD): Modified to allow NANG=1
! 91/08/15 (BTD): Corrected error (failure to initialize P)
! 91/08/15 (BTD): Modified to enhance vectorizability.
! 91/08/15 (BTD): Modified to make NANG=2 if called with NANG=1
! 91/08/15 (BTD): Changed definition of QBACK.
! 92/01/08 (BTD): Converted to full double precision and double complex
!                 eliminated 2 unneed lines of code
!                 eliminated redundant variables (e.g. APSI,APSI0)
!                 renamed RN -> EN = double precision N
!                 Note that DOUBLE COMPLEX and DCMPLX are not part
!                 of f77 standard, so this version may not be fully
!                 portable.  In event that portable version is
!                 needed, use src/bhmie_f77.f
! 93/06/01 (BTD): Changed AMAX1 to generic function MAX
!***********************************************************************
!*** Safety checks
      IF(NANG.GT.MXNANG)STOP'***Error: NANG > MXNANG in bhmie'
!      IF(NANG.LT.2)NANG=2
!*** Obtain pi:
      PII=4.*ATAN(1.D0)
      DX=X
      DREFRL=REFREL
      Y=X*DREFRL
      YMOD=ABS(Y)
!
!*** Series expansion terminated after NSTOP terms
!    Logarithmic derivatives calculated from NMX on down
      XSTOP=X+4.*X**0.3333+2.
      NMX=int(MAX(XSTOP,YMOD))+15
! BTD experiment 91/1/15: add one more term to series and compare results
!      NMX=AMAX1(XSTOP,YMOD)+16
! test: compute 7001 wavelengths between .0001 and 1000 micron
! for a=1.0micron SiC grain.  When NMX increased by 1, only a single
! computed number changed (out of 4*7001) and it only changed by 1/8387
! conclusion: we are indeed retaining enough terms in series!
      NSTOP=int(XSTOP)
!
      IF(NMX.GT.NMXX)THEN
          WRITE(0,*)'Error: NMX > NMXX=',NMXX,' for |m|x=',YMOD
          STOP
      ENDIF
!*** Require NANG.GE.1 in order to calculate scattering intensities
      DANG=0.
      IF(NANG.GT.1)DANG=.5*PII/DBLE(NANG-1)
      DO 1000 J=1,NANG
          THETA=DBLE(J-1)*DANG
          AMU(J)=COS(THETA)
 1000 CONTINUE
      DO 1100 J=1,NANG
          PI0(J)=0.
          PI1(J)=1.
 1100 CONTINUE
      NN=2*NANG-1
      DO 1200 J=1,NN
          S1(J)=(0.,0.)
          S2(J)=(0.,0.)
 1200 CONTINUE
!
!*** Logarithmic derivative D(J) calculated by downward recurrence
!    beginning with initial value (0.,0.) at J=NMX
!
      D(NMX)=(0.,0.)
      NN=NMX-1
      DO 2000 N=1,NN
          EN=NMX-N+1
          D(NMX-N)=(EN/Y)-(1./(D(NMX-N+1)+EN/Y))
 2000 CONTINUE
!
!*** Riccati-Bessel functions with real argument X
!    calculated by upward recurrence
!
      PSI0=COS(DX)
      PSI1=SIN(DX)
      CHI0=-SIN(DX)
      CHI1=COS(DX)
      XI1=DCMPLX(PSI1,-CHI1)
      QSCA=0.E0
      GSCA=0.E0
      P=-1.
      DO 3000 N=1,NSTOP
          EN=N
          FN=(2.E0*EN+1.)/(EN*(EN+1.))
! for given N, PSI  = psi_n        CHI  = chi_n
!              PSI1 = psi_{n-1}    CHI1 = chi_{n-1}
!              PSI0 = psi_{n-2}    CHI0 = chi_{n-2}
! Calculate psi_n and chi_n
          PSI=(2.E0*EN-1.)*PSI1/DX-PSI0
          CHI=(2.E0*EN-1.)*CHI1/DX-CHI0
          XI=DCMPLX(PSI,-CHI)
!
!*** Store previous values of AN and BN for use
!    in computation of g=<cos(theta)>
          IF(N.GT.1)THEN
              AN1=AN
              BN1=BN
          ENDIF
!
!*** Compute AN and BN:
          AN=(D(N)/DREFRL+EN/DX)*PSI-PSI1
          AN=AN/((D(N)/DREFRL+EN/DX)*XI-XI1)
          BN=(DREFRL*D(N)+EN/DX)*PSI-PSI1
          BN=BN/((DREFRL*D(N)+EN/DX)*XI-XI1)
!
!*** Augment sums for Qsca and g=<cos(theta)>
          QSCA=QSCA+(2.*EN+1.)*(ABS(AN)**2+ABS(BN)**2)
          GSCA=GSCA+((2.*EN+1.)/(EN*(EN+1.)))*
     &         (REAL(AN)*REAL(BN)+IMAG(AN)*IMAG(BN))
          IF(N.GT.1)THEN
              GSCA=GSCA+((EN-1.)*(EN+1.)/EN)*
     &        (REAL(AN1)*REAL(AN)+IMAG(AN1)*IMAG(AN)+
     &         REAL(BN1)*REAL(BN)+IMAG(BN1)*IMAG(BN))
          ENDIF
!
!*** Now calculate scattering intensity pattern
!    First do angles from 0 to 90
          DO 2500 J=1,NANG
              JJ=2*NANG-J
              PI(J)=PI1(J)
              TAU(J)=EN*AMU(J)*PI(J)-(EN+1.)*PI0(J)
              S1(J)=S1(J)+FN*(AN*PI(J)+BN*TAU(J))
              S2(J)=S2(J)+FN*(AN*TAU(J)+BN*PI(J))
 2500     CONTINUE
!
!*** Now do angles greater than 90 using PI and TAU from
!    angles less than 90.
!    P=1 for N=1,3,...; P=-1 for N=2,4,...
          P=-P
          DO 2600 J=1,NANG-1
              JJ=2*NANG-J
              S1(JJ)=S1(JJ)+FN*P*(AN*PI(J)-BN*TAU(J))
              S2(JJ)=S2(JJ)+FN*P*(BN*PI(J)-AN*TAU(J))
 2600     CONTINUE
          PSI0=PSI1
          PSI1=PSI
          CHI0=CHI1
          CHI1=CHI
          XI1=DCMPLX(PSI1,-CHI1)
!
!*** Compute pi_n for next value of n
!    For each angle J, compute pi_n+1
!    from PI = pi_n , PI0 = pi_n-1
          DO 2800 J=1,NANG
              PI1(J)=((2.*EN+1.)*AMU(J)*PI(J)-(EN+1.)*PI0(J))/EN
              PI0(J)=PI(J)
 2800     CONTINUE
 3000 CONTINUE
!
!*** Have summed sufficient terms.
!    Now compute QSCA,QEXT,QBACK,and GSCA
      GSCA=2.*GSCA/QSCA
      QSCA=(2./(DX*DX))*QSCA
      QEXT=(4./(DX*DX))*REAL(S1(1))
      QBACK=(ABS(S1(2*NANG-1))/DX)**2/PII
      RETURN
      END
