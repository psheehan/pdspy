============
Installation
============

Installing the code with Anaconda
"""""""""""""""""""""""""""""""""

Anaconda is probably the easiest way to install pdspy because the dependency GALARIO is easily conda-installable.

1. Download the code from GitHub: https://github.com/psheehan/pdspy.git

2. In a terminal, in the directory where the code was downloaded to:
   ::

       conda build pdspy -c conda-forge
       conda install pdspy -c conda-forge --use-local

Installing the code with pip
""""""""""""""""""""""""""""

1. In a terminal, run:
   ::

       pip install pdspy

2. Install GALARIO. Unfortunately, GALARIO is not pip-installable, so you will need to follow the instructions `here <https://mtazzari.github.io/galario/>`_.

Installing the code manually
""""""""""""""""""""""""""""

1. Download the code from this webpage. Git clone is recommended if you would like to be able to pull updates:
   ::

       git clone https://github.com/psheehan/pdspy.git

2. Install the Python dependencies:

   * numpy  
   * scipy  
   * matplotlib  
   * emcee  
   * corner  
   * hyperion  
   * h5py  
   * mpi4py  
   * galario  
   * Cython  
   * astropy < 4.0  
   * schwimmbad  
   * dynesty

3. In a terminal, go to the directory where the code was downloaded, and into the code directory. Run:
   ::

        python setup.py install
   
   or

   ::
   
        pip install -e .
   
Other dependencies
""""""""""""""""""

The other codes that are needed to run pdspy are `Hyperion <http://www.hyperion-rt.org>`_ and `RADMC-3D <http://www.ita.uni-heidelberg.de/~dullemond/software/radmc-3d/>`_. If you are a `Homebrew <https://brew.sh>`_ user, you can do this with:
    ::

       brew tap psheehan/science
       brew install hyperion
       brew install radmc3d

