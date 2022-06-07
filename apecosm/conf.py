''' Module that handles the processing of configuration files '''

from __future__ import print_function
import re
import os.path
import numpy as np

SEP_LIST = [';', '=', ',', '\t', ' ']
BOOL_VALUES = {'true': 1, 'false': 0}


def read_config(filename, relative_to_main=False, params=None, path=''):

    '''
    Reads configuration files.

    :param filename: The main configuration file.
    :type filename: str
    :param relative_to_main: True if path names are related to
     the main configuration file, False if path names are related to
     the current configuration file.
    :type relative_to_main: bool, optiona
    :param params: Dictionnary that contains all the parameters.
    :type params: dict, optional
    :param path: Path that is used to reconstruct the
     sub-configuration file names.
    :type path: str, optional

    :return: A dictionnary containing all the parameters.
    :rtype: dict

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
    lines = [line.strip() for line in lines]

    # extracts the pattern that matches configuration parameters
    pattern = '^[a-zA-Z]+'
    regexp = re.compile(pattern)

    # loop over the lines
    for line in lines:

        # if param is matched
        if regexp.match(line):

            sep = _find_separator(line)

            if sep is None:
                # if seperator is not found, nothing is done
                message = 'The line %s could not have been processed' % line
                print(message)
                continue

            # extract key and value, remove spurious white spaces
            key, val = line.split(_find_separator(line))
            key = key.strip()
            val = val.strip()

            # if parameter is a new configuration file, read config
            if key.startswith('apecosm.configuration'):
                read_config(val, relative_to_main=relative_to_main,
                            params=params, path=path)
            else:
                # else, if parameter is not already defined, it is
                # added to the parameter list
                if key not in params.keys():
                    params[key] = _convert(val)
                else:
                    message = 'Parameter %s is already defined ' % (key)
                    message += 'and equal to %s.\n' % (params[key])
                    message += 'Current value %s is ignored.' % val
                    print(message)

    return params


def _convert(parameter):

    '''
    Converts a string parameter into a single parameter or
    a list of parameters.
    :param parameter: Parameter
    :type parameter: str
    '''

    # try the conversion into float
    if ',' in parameter:

        parameter = parameter.split(',')
        parameter = [_convert(v.strip()) for v in parameter]

        # check if any of the param is a string
        # if so, all vals are converted into str
        test_str = [isinstance(v, str) for v in parameter]

        # if no string, looking for float
        test_float = [isinstance(v, float) for v in parameter]

        # if no string, looking for float
        test_int = [isinstance(v, int) for v in parameter]

        if np.any(test_str):
            parameter = [str(v) for v in parameter]

        elif np.any(test_float):
            parameter = [float(v) for v in parameter]

        elif np.any(test_int):
            parameter = [int(v) for v in parameter]

        return np.array(parameter)

    # try the conversion into int/bool
    try:
        if re.match('[0-9]+', parameter):
            parameter = int(parameter)
        elif re.match('[0-9]+.?[0-9]*', parameter):
            parameter = float(parameter)
        elif parameter.lower() in ['false', 'true']:
            parameter = BOOL_VALUES[parameter]
        return parameter
    except ValueError:
        pass

    # try the conversion into float
    try:
        parameter = float(parameter)
        return parameter
    except ValueError:
        pass

    return parameter


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

    test = _convert('0.05')
    print(type(test), test)

    test = _convert('1006')
    print(type(test), test)

    test = _convert('true')
    print(type(test), test)

    test = _convert('false')
    print(type(test), test)

    test = _convert('false, false, true')
    print(type(test), test.dtype, test)

    test = _convert('0, 3, 5')
    print(type(test), test.dtype, test)

    test = _convert('0.5, 3, 5')
    print(type(test), test.dtype, test)

    # conf = read_config(dirin + filename, relative_to_main=True)
    # print conf
