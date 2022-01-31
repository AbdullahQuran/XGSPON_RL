"""IO.py
"""

import Utils


def open_for_reading(file_name):
    """Open file for reading
    """

    if not Utils.is_str(file_name):
        Utils.error("'{:s}' is not a string".format(file_name))
    try:
        fp = open(file_name)
    except IOError:
        Utils.error("'{:s}' does not exist".format(file_name))

    return fp


def open_for_writing(file_name):
    """Open file for writing
    """

    if not Utils.is_str(file_name):
        Utils.error("'{:s}' is not a string".format(file_name))
    try:
        fp = open(file_name, "w")
    except IOError:
        Utils.error("Cannot open '{:s}' for writing".format(file_name))

    return fp


def close_for_reading(fp):
    """Close file opened for reading
    """

    fp.close()


def close_for_writing(fp):

    """Close file opened for reading
    """

    fp.close()


def file_lines(file_name):

    """Return all file lines, with some filtering
    """

    fp = open_for_reading(file_name)
    lines = fp.readlines()
    close_for_reading(fp)

    # remove whitespaces
    lines2 = Utils.strip_wsnl_list(lines)

    # remove empty lines
    lines3 = []
    for line in lines2:
        if line != "":
            lines3.append(line)

    return lines3
