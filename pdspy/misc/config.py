#!/usr/bin/env python
'''
Name    : Config, config.py
Author  : Nickalas Reynolds
Date    : Fall 2017
Misc    : File handles reading in the configuration file and 
          for writing an example configuration file
Example : defaultconfig.py
'''

# imported standard modules
from os.path import isfile,getctime
from os import getcwd,remove
from sys import exit
from glob import glob
from shutil import copyfile
from inspect import getfile
from importlib import import_module
from time import time as ctime

# import custom modules
from . import config_template
from .colours import colours
from ..version import *

# checking python version
assert assertion()
__version__ = package_version()

# main class object for manipulating configuration files
class configuration(object):

    def __init__(self,inputfile=None,cwd=None):
        """
        Initialize the configuration class
        """

        self.time = '{}'.format(str(ctime()).split('.')[0])

        if cwd == None:
            self.cwd = getcwd()
        else:
            self.cwd = cwd

        self.params     = None
        self.loadedfile = None
        self.orig       = None

        if inputfile == None:
            self.example()
            raise RuntimeError('Configuration file not specified, refer to example...')
        else:
            self.inputfile=self.remove_ext(self.find_file(inputfile))

    def get_functions(self):
        '''
        return all defined functions 
        '''
        return dir(self)

    def get_inputs(self):
        '''
        return all input variables initialized
        '''
        return vars(self)

    def load(self):
        """
        load the input file
        """
        self.loadedfile = import_module(self.inputfile)
        self.params     = self.read()
        self.orig       = self.params
        if self.remove:
            remove('{}/{}.py'.format(self.cwd,self.inputfile))

    def read(self):
        """
        read the input file
        """
        return self.loadedfile.config


    def get_params(self):
        """
        Return the parameters from the input file
        """
        return self.params

    def set_params(self,**kwargs):
        """
        Set the parameters in the object
        Only if param exists
        """
        if kwargs is not None:
            if self.params != None:
                temp = self.params
            for key, value in kwargs.items():
                if self.verify_params(key):
                    temp[key] = value

            self.params = temp

    def add_params(self,**kwargs):
        """
        Add new parameters in the object
        Only if param doesn't exist
        """
        if kwargs is not None:
            if self.params != None:
                temp = self.params
            for key, value in kwargs.items():
                if not self.verify_params(key):
                    temp[key] = value

            self.params = temp

    def verify_params(self,*args):
        """
        Verify that the keyword arguments are 
        within the configuration file
        """
        dictkeys = [x for x in self.params]
        for k in args:
            if k not in dictkeys:
                return False
            else:
                pass
        return True

    def remove_ext(self,inputfile):
        """
        Remove File extensions
        """
        return '.'.join(inputfile.split('.')[:-1])

    def find_file(self,inputfile):
        '''
        Find the input file and make sure it is in current directory
        '''
        if isfile(inputfile):
            if (len(inputfile.split('/')) > 1) or ('.py' not in inputfile):
                dest = "{}/config_{}.py".format(self.cwd,self.time)
                copyfile(inputfile,dest)
                self.remove = True
            else:
                self.remove = False
                dest = inputfile
        else:
            raise RuntimeError('Input file not found: {}'.format(inputfile))
        return dest.split('/')[-1]

    def example(self):
        """
        Make the example file in the cwd
        """
        src = "{}.py".format(self.remove_ext(getfile(config_template)))
        dest = "{}/config_template.py".format(self.cwd)
        if not isfile(dest):
            copyfile(src,dest)

if __name__ == "__main__":
    print('Testing module\n')
    print("{}".format(__doc__))
