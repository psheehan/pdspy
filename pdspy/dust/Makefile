all: bhcoat	bhmie

bhcoat:
	f2py3.4 -m bhcoat -c bhcoat.f90 --fcompiler=gfortran

bhmie:
	f2py3.4 -m bhmie -c bhmie.f90 --fcompiler=gfortran

clean:
	rm -f *.so
