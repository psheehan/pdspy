#!/usr/bin/env python
'''
Name  : Colours, colours.py
Author: Nickalas Reynolds
Date  : Fall 2017
Misc  : 
    # %&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&
    # Colours
    # %&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&
    # A color init string consists of one or more of the following numeric codes:
    # * Attribute codes:
    #   00=none 01=bold 04=underscore 05=blink 07=reverse 08=concealed
    # * Text color codes:
    #   30=black 31=red 32=green 33=yellow 34=blue 35=magenta 36=cyan 37=white
    # * Background color codes:
    #   40=black 41=red 42=green 43=yellow 44=blue 45=magenta 46=cyan 47=white
    # * Extended color codes for terminals that support more than 16 colors:
    #   (the above color codes still work for these terminals)
    #   ** Text color coding:
    #      38;5;COLOR_NUMBER
    #   ** Background color coding:
    #      48;5;COLOR_NUMBER
    # COLOR_NUMBER is from 0 to 255.
'''

class colours:
    HEADER  = '\033[95m'
    OKBLUE  = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL    = '\033[91m'
    BOLD    = '\033[1m'
    DEBUG   = '\033[96m'
    _RST_   = '\033[0m'  # resets color and format
    _BLD    = '\033[1m'
    _DIM    = '\033[2m'
    _UND    = '\033[4m'
    _BLK    = '\033[5m'  # only in supported terms, otherwise shown as reverse
    _RVS    = '\033[7m'
    _HID    = '\033[8m'
    _BU     = '\033[1m\033[4m'
    _BLD_   = '\033[21m'
    _DIM_   = '\033[22m'
    _UND_   = '\033[22m'
    _BLK_   = '\033[25m'
    _RVS_   = '\033[27m'
    _HID_   = '\033[28m'
    _BU_    = '\033[21m\033[22m'
     
    # Maps 16 color to 256 colors
     
    # Regular Colors
    Black   = '\033[38;5;0m'
    Red     = '\033[38;5;1m'
    Green   = '\033[38;5;2m'
    Yellow  = '\033[38;5;3m'
    Blue    = '\033[38;5;4m'
    Magenta = '\033[38;5;5m'
    Cyan    = '\033[38;5;6m'
    White   = '\033[38;5;7m'
     
    # Background
    On_Black   = '\033[48;5;0m'
    On_Red     = '\033[48;5;1m'
    On_Green   = '\033[48;5;2m'
    On_Yellow  = '\033[48;5;3m'
    On_Blue    = '\033[48;5;4m'
    On_Magenta = '\033[48;5;5m'
    On_Cyan    = '\033[48;5;6m'
    On_White   = '\033[48;5;7m'
     
    # High Intensty
    IBlack   = '\033[38;5;8m'
    IRed     = '\033[38;5;9m'
    IGreen   = '\033[38;5;10m'
    IYellow  = '\033[38;5;11m'
    IBlue    = '\033[38;5;12m'
    IMagenta = '\033[38;5;13m'
    ICyan    = '\033[38;5;14m'
    IWhite   = '\033[38;5;15m'
     
    # High Intensty backgrounds
    On_IBlack   = '\033[48;5;8m'
    On_IRed     = '\033[48;5;9m'
    On_IGreen   = '\033[48;5;10m'
    On_IYellow  = '\033[48;5;11m'
    On_IBlue    = '\033[48;5;12m'
    On_IMagenta = '\033[48;5;13m'
    On_ICyan    = '\033[48;5;14m'
    On_IWhite   = '\033[48;5;15m'

if __name__ == "__main__":
    print('Testing module\n')
    print("{}".format(__doc__))
