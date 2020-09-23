''' Module that handles the processing of configuration files '''

from __future__ import print_function
import re
import os.path
import numpy as np

SEP_LIST = [';', '=', ',', '\t', ' ']


def read_config(filename, relative_to_main=False, params=None, path=''):

    '''
    Reads configuration files.

    :param str filename: The main configuration file.
    :param boolean relative_to_main: True if path names are related to
     the main configuration file, False if path names are related to
     the current configuration file.
    :param dict params: Dictionnary that contains all the parameters.
    :param str path: Path that is used to reconstruct the
     sub-configuration file names.

    :return: A dictionnary containing all the parameters.

    '''

    if params is None:
        params = {}

    # reconstructs the config file name from path argument and filename
    filename = os.path.join(path, filename)

    # if config file is main, initialises the path from main configuration file
    if path == '':
        path = os.path.dirname(filename)

    if not relative_to_main:
        # if path names are related to local path names
        # extracts the new paths
        path = os.path.dirname(filename)

    # Opens the configuration file and extracts the lines
    with open(filename) as file_in:
        lines = file_in.readlines()

    # removes blank spaces at the beginning and end of the lines
    lines = [l.strip() for l in lines]

    # extracts the pattern that matches configuration parameters
    pattern = '^[a-zA-Z]+'
    regexp = re.compile(pattern)

    # loop over the lines
    for l in lines:

        # if param is matched
        if regexp.match(l):

            sep = _find_separator(l)

            if sep is None:
                # if seperator is not found, nothing is done
                message = 'The line %s could not have been processed' % l
                print(message)
                continue

            # extract key and value, remove spurious white spaces
            key, val = l.split(_find_separator(l))
            key = key.strip()
            val = val.strip()

            # if parameter is a new configuration file, read config
            if key.startswith('apecosm.configuration'):
                read_config(val, relative_to_main=relative_to_main, params=params, path=path)
            else:
                # else, if parameter is not already defined, it is
                # added to the parameter list
                if key not in params.keys():
                    params[key] = _convert(val)
                else:
                    message = 'Parameter %s is already defined and equal to %s.\n' % (key, params[key])
                    message += 'Current value %s is ignored.' % val
                    print(message)

    return params


def _convert(val):
    
    # try the conversion into float
    if ',' in val:
        
        val = val.split(',')
        val = [_convert(v) for v in val]
        
        # check if any of the param is a string
        # if so, all vals are converted into str
        test_str = [isinstance(v, str) for v in val]
        if(np.any(test_str)): 
            val = [str(v) for v in val]
            return val

        # if no string, looking for float
        test_float = [isinstance(v, float) for v in val]
        if(np.any(test_float)): 
            val = [float(v) for v in val]
            return val

        return val

    
    # try the conversion into int/bool
    try:
        val = int(val)
        return val
    except ValueError:
        pass
    
    # try the conversion into float
    try:
        val = float(val)
        return val
    except ValueError:
        pass
    
    # try the conversion into float
    try:
        val = float(val)
        return val
    except ValueError:
        pass
    
    return val



def _find_separator(string):

    '''
    Returns the key-value separator for
    a given string. It is defined as the separator
    that splits the string into two parts.

    :param str string: Input string

    :return: The key-value separator. If no separator is found,
    returns None.
    '''

    for sep in SEP_LIST:

        if len(string.split(sep)) == 2:

            return sep

    return None

if __name__ == '__main__':

    dirin = '/home/nbarrier/Modeles/apecosm/svn-apecosm/trunk/tools/config/gyre/'
    dirin = 'config/'
    filename = 'oope.conf'

    x = 'sl, alt, alt'
    x = '2, 1.0'
    toto = _convert(x)
    print(toto)
    print(type(toto[0]))
    print(type(toto[1]))

    #conf = read_config(dirin + filename, relative_to_main=True)
    #print conf
