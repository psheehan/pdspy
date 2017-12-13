#!/usr/bin/env python
'''
Name    : Logger, logger.py
Author  : Nickalas Reynolds
Date    : Fall 2017
Misc    : File will handle misc functions (waiting and input)
Example : logger = utilities.Messenger(verbosity=2,add_timestamp=True,logfile=logfile)
         logger.header1('THIS IS AN EXAMPLE')
'''

# imported standard modules
from sys import version_info
from os import remove
from os.path import isfile
from glob import glob
from datetime import datetime as dtime
import time

# import custom modules
from .colours import colours

# function that creates a logger
class Messenger(object):
    """
    The Messenger class which handles pretty
    logging both to terminal and to a log
    file which was intended for running 
    codes on clusters in batch jobs where
    terminals were not slaved.
    """
    PY2 = version_info[0] == 2
    PY3 = version_info[0] == 3

    use_structure = ".    "
    def __init__(self, verbosity=2, use_colour=True, use_structure=False,add_timestamp=True, logfile=None):
        """
        Setting the parameters for the Messenger class.
        """
        self.verbosity = verbosity

        # specifying colour options
        self.use_colour = use_colour
        if use_colour:
            self.enable_colour()
        else:
            self.disable_colour()

        self.use_structure = use_structure
        self.add_timestamp = add_timestamp
        self.logfile = logfile

        # overrides existing file
        if logfile is not None:
            self.f = open(logfile, 'w') 

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

    def set_verbosity(self, verbosity):
        """
        Set the verbosity level for the class
        """
        self.verbosity = verbosity

    def get_verbosity(self):
        """
        Returns the verbosity level of the class
        """
        return self.verbosity

    def disable_colour(self):
        """
        Turns off all colour formatting.
        """
        self.BOLD    = ''
        self.HEADER1 = ''
        self.HEADER2 = ''
        self.OKGREEN = ''
        self.WARNING = ''
        self.FAIL    = ''
        self.MESSAGE = ''
        self.DEBUG   = ''
        self.CLEAR   = ''

    def enable_colour(self):
        """
        Enable all colour formatting.
        """

        # colour definitions
        self.BOLD    = colours.BOLD
        self.HEADER1 = self.BOLD + colours.HEADER
        self.HEADER2 = self.BOLD + colours.OKBLUE
        self.OKGREEN = colours.OKGREEN
        self.WARNING = colours.WARNING
        self.FAIL    = colours.FAIL
        self.MESSAGE = colours._RST_
        self.DEBUG   = colours.DEBUG
        self.CLEAR   = colours._RST_

    def _get_structure_string(self, level):
        """
        Returns the string of the message with the specified level
        Which is dependent on the verbosity
        """

        string = ''
        if self.use_structure:
            for i in range(level):
                string = string + self.structure_string
        return string

    def _get_time_string(self):
        """
        Returns the detailed datetime for extreme debugging
        """

        string = ''
        if self.add_timestamp:
            string = '[{}] '.format(dtime.today()) 
        return string
    
    def _make_full_msg(self, msg, verb_level):
        """
        Constructs the full string that carries the message
        with the specified verbosity parameters
        """
        struct_string = self._get_structure_string(verb_level)
        time_string = self._get_time_string()
        return time_string + struct_string + msg

    def _write(self, cmod, msg,out=True):
        """
        Write the message to the file and print 
        it to the terminal if it is wanted
        """
        if out:
            print("{}{}{}".format(cmod,msg,self.CLEAR))
        if type(self.logfile) is str:
            self.f.write(msg + '\n')

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    """
    The following commands are the ones
    to use when calling the logger
    will handle writing to the log
    file and to the terminal
    """
    def warn(self, msg, verb_level=2):

        if verb_level <= self.verbosity:
            full_msg = self._make_full_msg(msg, verb_level)
            self._write(self.WARNING, full_msg)

    def header1(self, msg, verb_level=0):

        if verb_level <= self.verbosity:
            full_msg = self._make_full_msg(msg, verb_level)
            self._write(self.HEADER1, full_msg)

    def header2(self, msg, verb_level=1):

        if verb_level <= self.verbosity:
            full_msg = self._make_full_msg(msg, verb_level)
            self._write(self.HEADER2, full_msg)

    def success(self, msg, verb_level=1):

        if verb_level <= self.verbosity:
            full_msg = self._make_full_msg(msg, verb_level)
            self._write(self.OKGREEN, full_msg)

    def failure(self, msg, verb_level=0):

        if verb_level <= self.verbosity:
            full_msg = self._make_full_msg(msg, verb_level)
            self._write(self.FAIL, full_msg)

    def message(self, msg, verb_level=2):

        if verb_level <= self.verbosity:
            full_msg = self._make_full_msg(msg, verb_level)
            self._write(self.MESSAGE, full_msg)

    def debug(self, msg, verb_level=4):

        if verb_level <= self.verbosity:
            full_msg = self._make_full_msg(msg, verb_level)
            self._write(self.DEBUG, full_msg)

    def pyinput(self,message=None,verb_level=0):

        total_Message = "{}: ".format(message)
        if self.PY2:
            out = raw_input(total_Message)
        if self.PY3:
            out = input(total_Message)
        if verb_level <= self.verbosity:  
            full_msg = self._make_full_msg(total_Message, verb_level)
            self._write(self.DEBUG, full_msg,False)      
        return out

    def waiting(self,auto,seconds=1,verb_level=0):
        if not auto:
            self.pyinput('[RET] to continue or CTRL+C to escape')
        elif verb_level <= self.verbosity:  
            self.warn('Will continue in {}s. CTRL+C to escape'.format(seconds))
            time.sleep(seconds)

    def _REMOVE_(self,file):
        """
        This is a restructure of the os.system(rm) or the os.remove command
        such that the files removed are displayed appropriately or not removed
        if the file is not found
        """
        if not typecheck(file):
            for f in glob('*'+file+'*'):
                if isfile(f):
                    try:
                        remove(f)
                        self.debug("Removed file {}".format(f))
                    except OSError:
                        self.debug("Cannot find {} to remove".format(f))
        else:
            for f in file:
                if isfile(f):
                    try:
                        remove(f)
                        self.debug("Removed file {}".format(f))
                    except OSError:
                        self.debug("Cannot find {} to remove".format(f))   

def typecheck(obj): return not isinstance(obj, str) and isinstance(obj, Iterable)

if __name__ == "__main__":
    print('Testing module\n')
    print("{}".format(__doc__))