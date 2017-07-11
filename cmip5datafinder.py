#!/home/valeriu/sdt/bin/python

"""
cmip5datafinder.py
Python 2.7.13
Script that searches for data locally and on valid ESGF nodes. It builds 
cache files using the results of the search.

"""

# -------------------------------------------------------------------------
#      Setup.
# -------------------------------------------------------------------------

# ---- Import standard modules to the python path.
import sys, os, shutil, math, copy, getopt, re, string, popen2, time, errno
import numpy as np
from numpy import loadtxt as lt
from numpy import savetxt as st
from xml.dom import minidom
import subprocess
from datetime import datetime
import time

__author__ = "Valeriu Predoi <valeriu.predoi@ncas.ac.uk>"

# ---- Function usage.
# ---- opts parsing
def usage():
  msg = """\
This is a flexible tool to generate cache files from local datasources (e.g. badc or dkrz 
mounted disks) and ESGF nodes. This makes use of synda for querying ESGF nodes as well.
For problems or queries, email valeriu.predoi@ncas.ac.uk. Have fun!

Code functionality:
1. Given a command line set of arguments or an input file, the code looks for cmip5
files locally and returns the physical paths to the found files;
2. If files are not found, the user has the option to download missing files from ESGF
nodes via synda;
3. Finally, the code writes cache files stored in a directory cache_files_[DATASOURCE]:
           - cache_cmip5_[DATASOURCE].txt local cache file with paths only on server [DATASOURCE]
           - cache_cmip5_combined_[DATASOURCE].txt combined local cache file (synda+DATASOURCE)
           - cache_cmip5_synda_[DATASOURCE].txt synda local cache file
           - cache_err.out errors sdout while caching
           - missing_cache_cmip5_[DATASOURCE].txt local missing files on [SERVER]
           - missing_cache_cmip5_combined_[DATASOURCE].txt missing files (synda+local)
           - missing_cache_cmip5_synda_[DATASOURCE].txt synda missing files
4. Finally-finally it plots the overall, incomplete and missing files by model (png format).

Example run:
python cmip5datafinder.py -p PARAM_FILE --synda --download --dryrun --verbose --datasource badc

Definition of filedescriptor:
To understand the output, by filedescriptor we mean any file indicator
of form e.g. CMIP5_MIROC5_Amon_historical_r1i1p1_2003_2010_hus that is fully
determined by its parameters; there could be multiple .nc files
covering a single filedescriptor, alas there could be just one.
All cache files contain first a file indicator = filedescriptor e.g.

CMIP5_MIROC5_Amon_historical_r1i1p1_2003_2010_hus

Usage:
  cmip5datafinder.py [options]
  -p, --params-file <file>    Namelist file (xml) or text file (txt) or any other input file [REQUIRED] 
                              e.g. for xml: --params-file ESMValTool/nml/namelist_myTest.xml
                              e.g. for text: --params-file example.txt
                              e.g. for yaml: --params-file example.yml
                              This option is REQUIRED if --user-input (command line) is NOT present
  -h, --help                  Display this message and exit
  --user-input                Flag for user defined CMIP file and variables parameters (to be input at command line
                              with --fileparams for each parameter)
                              This option is REQUIRED if --params-file is not present
  --datasource                Name of local data source (example: badc). Available datasources:
                              badc [to add more here, depending where running the code][REQUIRED]
  --synda                     Flag to call synda operations. If not passed, local datasources will be used ONLY
  --download                  Flag to allow download missing data via synda
  --dryrun                    Flag to pass if no download is wanted. Don't pass this if downloads are neeeded!
                              If --dryrun in arguments, all cache files will be written as normal but with
                              NOT-YET-INSTALLED flag per file
  --fileparams                If --user-input is used, this serial option passes one data file argument at a time
                              If --user-input is used, this serial option is REQUIRED
                              e.g. --fileparams CMIP5 --fileparams MPI-ESM-LR --fileparams Amon  --fileparams historical
                              --fileparams r1i1p1 --fileparams 1910 --fileparams 1919
  --uservars                  If --user-input is used, this serial option passes one variable argument at a time
                              If --user-input is used, this serial option is REQUIRED
                              e.g. --uservars tro3
  --verbose                   Flag to show in-code detailed messages

Understand the workflow:
(1) python cmip5datafinder.py -p PARAM_FILE --datasource badc
   looks for files associated with data sources in PARAM_FILE locally on e.g badc only, in dirs in root /badc/cmip5/data/cmip5/output1/
   stores cache files in cache_files_badc/ and creates user-friendly cache cache_PARAM_FILE.txt-badc
(2) python cmip5datafinder.py -p PARAM_FILE --synda --datasource badc
   same as (1), but it adds the local /sdt/data/ to the data lookup targets ONLY for incomplete/missing filedescriptors on badc; creates combined caches 
   with data on badc and in /sdt/;
   creates the same user-friendly cache_PARAM_FILE.txt-badc that this time will include files present in /sdt/ too
(3) python cmip5datafinder.py -p PARAM_FILE --synda --download --datasource badc
   same as (2) only this time the code will search for files that are incomplete or missing from badc AND /sdt/ over the net on ESGF nodes and will
   download them into /sdt/data/ if no --dryrun specified; NOTE that it is impossible to ask for download if prior checks in BOTH
   badc and /sdt/data/ have not been done (this is in place so that wild download will not happen);
(4) python cmip5datafinder.py -p PARAM_FILE --synda --download --dryrun --datasource badc
   same as (3) but no actual downloads happen. 

"""
  print >> sys.stderr, msg

########################################
# ---- Operational functions here ---- #
########################################

# ---- get the path to synda executable
def which_synda(synda):
    """

    This function returns the path to the synda exec
    or aborts the whole program if synda needs to be used
    but its executable is not found.

    """
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(synda)
    if fpath:
        if is_exe(synda):
            #print('We are using the following executable: %s' % synda)
            return synda
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, synda)
            if is_exe(exe_file):
                return exe_file
    return None

# ---- handling the years for files
def time_handling(year1, year1_model, year2, year2_model):
    """
    This function is responsible for finding the correct 
    files for the needed timespan:

    year1 - the start year in files
    year1_model - the needed start year of data
    year2 - the last year in files
    year2_model - the needed last year of data
    WARNINGS:
    we reduce our analysis only to years

    """
    # model interval < data interval / file
    # model requirements completely within data stretch
    if year1 <= int(year1_model) and year2 >= int(year2_model):
        return True,True
    # model interval > data interval / file
    # data stretch completely within model requirements
    elif year1 >= int(year1_model) and year2 <= int(year2_model):
        return True,False
    # left/right overlaps and complete misses
    elif year1 <= int(year1_model) and year2 <= int(year2_model):
        # data is entirely before model
        if year2 <= int(year1_model):
            return False,False
        # data overlaps to the left
        elif year2 >= int(year1_model):
            return True,False
    elif year1 >= int(year1_model) and year2 >= int(year2_model):
        # data is entirely after model
        if year1 >= int(year2_model):
            return False,False
        # data overlaps to the right
        elif year1 <= int(year2_model):
            return True,False

# ---- function to handle various date formats
def date_handling(time1,time2):
    """
    This function deals with different input date formats e.g.
    time1 = 198204 or
    time1 = 19820422 or
    time1 = 198204220511 etc
    More formats can be coded in at this stage.
    Returns year 1 and year 2
    """
    # yyyymm
    if len(list(time1)) == 6 and len(list(time2)) == 6:
        y1 = datetime.strptime(time1, '%Y%m')
        year1 = y1.year
        y2 = datetime.strptime(time2, '%Y%m')
        year2 = y2.year
    else:
        # yyyymmdd
        if len(list(time1)) == 8 and len(list(time2)) == 8:
            y1 = datetime.strptime(time1, '%Y%m%d')
            year1 = y1.year
            y2 = datetime.strptime(time2, '%Y%m%d')
            year2 = y2.year
        # yyyymmddHHMM
        if len(list(time1)) == 12 and len(list(time2)) == 12:
            y1 = datetime.strptime(time1, '%Y%m%d%H%M')
            year1 = y1.year
            y2 = datetime.strptime(time2, '%Y%m%d%H%M')
            year2 = y2.year
    return year1,year2

# ---- cleanup duplicate entries in files
def fix_duplicate_entries(outfile):
    """
    simple fast function to eliminate duplicate entries
    from a cache file
    """
    # ---- fixing the cache file for duplicates
    ar = np.genfromtxt(outfile, dtype=str,delimiter='\n')
    nar = np.unique(ar)
    st(outfile,nar,fmt='%s')

# ---- synda search
def synda_search(model_data,varname):
    """
    This function performs the search for files in synda-standard paths
    It takes exactly two arguments:
    - a model data string of type e.g. 'CMIP5 MPI-ESM-LR Amon amip r1i1p1'
    - a variable name as string e.g. 'tro3'
    It performs the search for files associated with these parameters and returns ALL
    available files. (command example: synda search -f CMIP5 MPI-ESM-LR Amon amip r1i1p1 tro3)

    """
    # this is needed mostly for parallel processes that may
    # go tits-up from time to time due to random path mixes
    if which_synda('synda') is not None:
        pass
    else:
        print >> sys.stderr, "No synda executable found in path. Exiting."
        sys.exit(1)
    synda_search = which_synda('synda') + ' search -f ' + model_data + ' ' + varname
    proc = subprocess.Popen(synda_search, stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()
    if err is not None:
        print >> sys.stderr, "An error has occured while searching for data:"
        print >> sys.stderr, err
        sys.exit(1)
    else:
        return out

# ---- cache via synda
def write_cache_via_synda(searchoutput,varname,year1_model,year2_model,header,outfile,outfile2):
    """
    ----------------------------------------------
    WARNING1: this function is SLOW
    WARNING2: this function is not currently used
    ----------------------------------------------
    This function takes the standard search output from synda (synda_search())
    and parses it to see if/what files exist locally

    The searchoutput argument is a string and is of the form e.g.

    new   221.2 MB  cmip5.output1.MPI-M.MPI-ESM-LR.historical.mon.atmos.Amon.r1i1p1.v20120315.tro3_Amon_MPI-ESM-LR_historical_r1i1p1_195001-195912.nc
    done  132.7 MB  cmip5.output1.MPI-M.MPI-ESM-LR.historical.mon.atmos.Amon.r1i1p1.v20120315.tro3_Amon_MPI-ESM-LR_historical_r1i1p1_200001-200512.nc
    new   221.2 MB  cmip5.output1.MPI-M.MPI-ESM-LR.historical.mon.atmos.Amon.r1i1p1.v20120315.tro3_Amon_MPI-ESM-LR_historical_r1i1p1_185001-185912.nc
    
    ie typical synda file search output. This gets parsed in and analyzed
    against the required model file characterstics and files that comply and 
    exist locally are stored in a cache file for data reading. It also takes the year1_model and year2_model, for time checks.
    It also takes the variable name and the name of a cache file outfile that will be written to disk. 

    """
    # this is needed mostly for parallel processes that may
    # go tits-up from time to time due to random path mixes
    if which_synda('synda') is not None:
        pass
    else:
        print >> sys.stderr, "No synda executable found in path. Exiting."
        sys.exit(1)
    with open(outfile, 'a') as file:
        file.close()
    entries = searchoutput.split('\n')[:-1]
    if len(entries)>0:
        for entry in entries:
            file_name = entry.split()[3]
            if header.split('_')[1] == file_name.split('.')[3]:
                time_range = file_name.split('_')[-1].strip('.nc')
                time1 = time_range.split('-')[0]
                time2 = time_range.split('-')[1]
                year1 = date_handling(time1,time2)[0]
                year2 = date_handling(time1,time2)[1]
                if time_handling(year1, year1_model, year2, year2_model)[0] is True:
                    file_name_complete = ".".join(file_name.split('.')[:10]) + '.' + varname + '.' + ".".join(file_name.split('.')[10:])
                    true_file_name = file_name_complete.split('.')[-2]+'.'+file_name_complete.split('.')[-1]
                    print('Matching file: %s' % true_file_name)
                    # get the most recent file from datasource
                    # synda will always list the most recent filedescriptor first
                    synda_search = which_synda('synda') + ' search -f -l 1 ' + true_file_name
                    proc = subprocess.Popen(synda_search, stdout=subprocess.PIPE, shell=True)
                    (out, err) = proc.communicate()
                    if len(out.split()) > 2:
                        fc = out.split()[3]
                        file_name_complete_final = ".".join(fc.split('.')[:10]) + '.' + varname + '.' + ".".join(fc.split('.')[10:])
                        filepath_complete_0 = '/badc/cmip5/data/c' + file_name_complete_final.replace('.','/').strip('/nc') + '.nc'
                        filepath_complete = "/".join(filepath_complete_0.split('/')[0:13]) + '/latest/' + "/".join(filepath_complete_0.split('/')[14:])
                        print(filepath_complete)
                        # ---- perform a local check file exists in /badc
                        # ---- and write cache
                        crf = filepath_complete.split('/')[-1]
                        # ---- writing only files that match experiment type
                        if header.split('_')[1] == crf.split('_')[2]:
                            if os.path.exists(filepath_complete):
                                with open(outfile, 'a') as file:
                                    file.write(header + ' ' + filepath_complete + ' ' + out.split()[1] + out.split()[2] + '\n')
                                    file.close()
                                print('----------------------------------------------------')
                            else:
                                try:
                                    s = open(filepath_complete)
                                except IOError as ioex:
                                    print 'err message:', os.strerror(ioex.errno)
                                    print('Trying to look one directory up...')
                                    probl = "/".join(filepath_complete.split('/')[0:-1])
                                    fnd = 'find ' + probl +  ' -follow -iname "*.nc"'
                                    proc = subprocess.Popen(fnd, stdout=subprocess.PIPE, shell=True)
                                    (out, err) = proc.communicate()
                                    prs = []
                                    for s in out.split('\n')[0:-1]:
                                        ssp = s.split('/')
                                        av = ssp[-1]
                                        # --- date handling
                                        time_range = av.split('_')[-1].strip('.nc')
                                        time1 = time_range.split('-')[0]
                                        time2 = time_range.split('-')[1]
                                        year1 = date_handling(time1,time2)[0]
                                        year2 = date_handling(time1,time2)[1]
                                        if time_handling(year1, yr1, year2, yr2)[0] is True:
                                            if os.path.exists(s):
                                                prs.append(s)
                                                print('Found rogue file: %s' % s)
                                                with open(outfile, 'a') as file:
                                                    file.write(header + ' ' + s + '\n')
                                    if len(prs)==0:
                                        print('No files found...')
                                        with open(outfile2, 'a') as file:
                                            file.write(header + ' ' + 'ERROR ' + os.strerror(ioex.errno) + ' ' + filepath_complete + '\n')
                                            file.close()
                    else:
                        print('something went wrong with parsing the data entry, calling this a non-existent file')
                        with open(outfile2, 'a') as file:
                            file.write(header + ' ' + 'ERROR ' + file_name_complete + ' could not be found' + '\n')
                            file.close()
    else:
        print >> sys.stderr, "Could not find filedescriptor with the specified parameters on datasource"
        return 0

# ---- function that returns the DRS
def get_drs(dir1, sdir, ic, model, latest_dir):
    """
    Function that returns DRS.
    dir1: root directory - /badc/cmip5/data/cmip5/output1/
    sdir: subdirectory (institution) - MPI-M
    ic: experiment - MPI-ESM-LR
    model: CMIP5 MPI-ESM-LR Amon amip r1i1p1
    latest_dir: on badc is /latest/ - this is known in advance
    and is dependant on where the code is run.
    """
    # 3h
    if model[2] == '3h':
        gdrs = dir1 + sdir + '/' + ic + '/' + model[3] + '/3h/*/*/' + model[4]\
               + latest_dir + model[7] + '/'
    # 6h
    elif model[2] == '6h':
        gdrs = dir1 + sdir + '/' + ic + '/' + model[3] + '/6h/*/*/' + model[4]\
               + latest_dir + model[7] + '/'
    # daily (day)
    # the current implementation does not make the difference
    # between day and cfDay in lower dirs
    elif model[2] == 'day':
        gdrs = dir1 + sdir + '/' + ic + '/' + model[3] + '/day/*/*/' + model[4]\
              + latest_dir + model[7] + '/'
    elif model[2] == 'cfDay':
        gdrs = dir1 + sdir + '/' + ic + '/' + model[3] + '/cfDay/*/*/' + model[4]\
              + latest_dir + model[7] + '/'
    # monthly (mon)
    # very detailed DRS, looking straight into variable dir
    # variable = model[7]
    elif model[2] == 'Amon':
        gdrs = dir1 + sdir\
              + '/' + ic + '/' + model[3] + '/mon/atmos/Amon/' + model[4]\
              + latest_dir + model[7] + '/'
    elif model[2] == 'Omon':
        gdrs = dir1 + sdir\
              + '/' + ic + '/' + model[3] + '/mon/ocean/Omon/' + model[4]\
              + latest_dir + model[7] + '/'
    elif model[2] == 'Lmon':
        gdrs = dir1 + sdir\
              + '/' + ic + '/' + model[3] + '/mon/land/Lmon/' + model[4]\
              + latest_dir + model[7] + '/'
    elif model[2] == 'LImon':
        gdrs = dir1 + sdir\
              + '/' + ic + '/' + model[3] + '/mon/landIce/LImon/' + model[4]\
              + latest_dir + model[7] + '/'
    elif model[2] == 'OImon':
        gdrs = dir1 + sdir\
              + '/' + ic + '/' + model[3] + '/mon/seaIce/OImon/' + model[4]\
              + latest_dir + model[7] + '/'
    # aerosols
    elif model[2] == 'aero':
        gdrs = dir1 + sdir\
              + '/' + ic + '/' + model[3] + '/mon/aerosol/aero/' + model[4]\
              + latest_dir + model[7] + '/'
    else:
        print('Could not establish custom DRS...')
        gdrs = dir1 + sdir + '/' + ic + '/' + model[3] + '/' + model[2] + '/*/*/' + model[4]\
               + latest_dir + model[7] + '/'
        print('Using generalized path: %s' % gdrs)
    return gdrs

# ---- capture ls in the preferred directory
def lsladir(dirname):
    """
    Calling this function once so we save time; called in root dirname.
    It is needed for generalization and not hardcoding the institutions.
    """
    # capture the ls output
    lsd = 'ls -la ' + dirname
    proc = subprocess.Popen(lsd, stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()
    res = out.split('\n')[3:-1]
    return res

# ---- local file finder
def find_local_files(model,out1,dirname1,mfile,latest_dir):
    """
    Function that performs local search for files using `find'
    The depth is as high as possible so that find is fast.
    model: CMIP5 MPI-ESM-LR Amon amip r1i1p1
    mfile: stderr dump file (cache_err.out) - need to capture
    instances of either Permission denied or non-existent dirs;
    latest_dir: latest version directory e.g. /latest/ on badc
    (see above for details)
    """
    flist = []
    for st in out1:
        subdir = st.split()[-1]
        lsd2 = 'ls -la ' + dirname1 + subdir
        FNULL = open(mfile, 'a')
        proc2 = subprocess.Popen(lsd2, stdout=subprocess.PIPE, stderr=FNULL, shell=True)
        (out2, err2) = proc2.communicate()
        # work only with existing dirs or allowed permission dirs
        if len(out2) > 0:
            for st2 in out2.split('\n')[3:-1]:
                findic = st2.split()[-1]
                if findic == model[1]:
                    drs = get_drs(dirname1,subdir, findic, model,latest_dir)
                    # -follow option allows for finding symlinked files
                    strfindic = 'find ' + drs\
                                 +' -follow -type f -iname "*.nc"'
                    proc = subprocess.Popen(strfindic, stdout=subprocess.PIPE, shell=True)
                    (out, err) = proc.communicate()
                    for t in out.split('\n')[0:-1]:
                        flist.append(t)
    return flist
    # ---- done

# ---- cache local data
def write_cache_direct(params_file,ldir,rdir,outfile,outfile2,errfile,ld,verbose=False):
    """
    Function that does direct parsing of available datasource files and establishes
    the paths to the needed files; makes use of find_local_files()
    File versioning is controlled by finding the ld = e.g. /latest/ dir 
    in the badc datasource, this may differ on other clusters and should be correctly
    hardcoded in the code!

    """
    car = np.genfromtxt(params_file, dtype=str, delimiter='\n')
    # ---- eliminate duplicates from input file, if any
    nar = np.unique(car)
    prfile = 'prepended_' + params_file
    if len(nar) == 1:
        with open(prfile, 'a') as file:
            file.write(nar)
            file.write('\n')
            file.write(nar)
    else:
        st(prfile,nar,fmt='%s')
    itemlist = lt(prfile,dtype=str)
    lenitemlist = len(itemlist)
    for item in itemlist:
        arname = find_local_files(item,ldir,rdir,errfile,ld)
        if len(arname) > 0:
            var = item[7]
            header = item[0] + '_'+ item[1] + '_' + item[2]\
                         + '_' + item[3] + '_' + item[4] + '_' + item[5]\
                         + '_' + item[6] + '_' + item[7]
            yr1 = int(item[5])
            yr2 = int(item[6])
            for s in arname:
                ssp = s.split('/')
                av = ssp[-1]
                time_range = av.split('_')[-1].strip('.nc')
                time1 = time_range.split('-')[0]
                time2 = time_range.split('-')[1]
                year1 = date_handling(time1,time2)[0]
                year2 = date_handling(time1,time2)[1]
                # case where the required data completely overlaps
                # available data
                # this case stops the code to make a call to synda for this filedescriptor
                if time_handling(year1, yr1, year2, yr2)[0] is True and time_handling(year1, yr1, year2, yr2)[1] is True:
                    if os.path.exists(s):
                        with open(outfile, 'a') as file:
                            file.write(header + ' ' + s + '\n')
                        if verbose is True:
                            print('Cached file from local datasource: ' + s)
                    else:
                        with open(outfile2, 'a') as file:
                            file.write(header + ' ERROR-MISSING' + '\n')
                        if verbose is True:
                            print('WARNING: missing from local datasource: ' +  header)
                # case where the required data is not fully found
                # ie incomplete data 
                # what we want to do here is cache what we have available
                # but also let synda know there is missing data, maybe
                # she can find it...just maybe
                # also we must make sure she doesnt download what we already have
                if time_handling(year1, yr1, year2, yr2)[0] is True and time_handling(year1, yr1, year2, yr2)[1] is False:
                    if os.path.exists(s):
                        with open(outfile, 'a') as file:
                            file.write(header + ' ' + s + '\n')
                        if verbose is True:
                            print('Cached file from local datasource: ' + s)
                        sfn = s.split('/')[-1]
                        with open(outfile2, 'a') as file:
                            # the INCOMPLETE indicator will be used
                            # to label partially complete filedescriptors so synda can
                            # look for the missing bits and hopefully complete it
                            file.write(header + ' INCOMPLETE ' + sfn + '\n')
                    else:
                        with open(outfile2, 'a') as file:
                            file.write(header + ' ERROR-MISSING' + '\n')
                        if verbose is True:
                            print('WARNING: missing from local datasource: ' +  header)
        else:
            with open(outfile2, 'a') as file:
                # missing entirely
                file.write("_".join(item) + ' ERROR-MISSING' + '\n')
            if verbose is True:
                print('WARNING: missing from local datasource: ' + "_".join(item))
    if os.path.exists(outfile):
        fix_duplicate_entries(outfile)
    else:
        print >> sys.stderr, "WARNING: could not cache any data from local datasource"
    if os.path.exists(outfile2):
        fix_duplicate_entries(outfile2)
    else:
        print >> sys.stderr, "Cached all needed data from local datasource. Looks like there are no missing files, huzzah!"

# ---- print some stats
def print_stats(outfile1,outfile2):
    """
    small function to print some stats at the end
    """
    if os.path.exists(outfile1) and os.path.exists(outfile2):
        ar1 = np.genfromtxt(outfile1, dtype=str,delimiter='\n')
        ar2 = np.genfromtxt(outfile2, dtype=str,delimiter='\n')
        # force to a 1-liner
        if ar1.ndim == 0:
            f = 1
        else:
            f = len(ar1)
        if ar2.ndim == 0:
            m = 1
        else:
            m = len(ar2)
        print('\n###############################################################')
        print('  Found and cached: %i individual .nc files cached' % f)
        print('Missing/incomplete: %i individual datasets NOT cached/incomplete' % m)
        print('#################################################################\n')
    elif os.path.exists(outfile1) and os.path.exists(outfile2) is False:
        ar1 = np.genfromtxt(outfile1, dtype=str,delimiter='\n')
        if ar1.ndim == 0:
            f = 1
        else:
            f = len(ar1)
        print('\n########################################################')
        print('Found and cached: %i individual .nc files cached' % f)
        print('########################################################\n')
    elif os.path.exists(outfile1) is False:
        print('Shoot! No cache written this time around...') 

# ---- synda download
def synda_dll(searchoutput,varname,year1_model,year2_model,header,D,outfile,outfile2,download=False,dryrunOn=False,verbose=False):
    """
    This function takes the standard search output from synda
    and parses it to see if/what files need to be downloaded

    The searchoutput argument is a string and is of the form e.g.

    new   221.2 MB  cmip5.output1.MPI-M.MPI-ESM-LR.historical.mon.atmos.Amon.r1i1p1.v20120315.tro3_Amon_MPI-ESM-LR_historical_r1i1p1_195001-195912.nc
    done  132.7 MB  cmip5.output1.MPI-M.MPI-ESM-LR.historical.mon.atmos.Amon.r1i1p1.v20120315.tro3_Amon_MPI-ESM-LR_historical_r1i1p1_200001-200512.nc
    new   221.2 MB  cmip5.output1.MPI-M.MPI-ESM-LR.historical.mon.atmos.Amon.r1i1p1.v20120315.tro3_Amon_MPI-ESM-LR_historical_r1i1p1_185001-185912.nc
    
    ie typical synda file search output. This gets parsed in and analyzed
    against the required model file characterstics and files that comply can
    be downloaded via synda install. It also takes the year1_model and year2_model, for time checks.
    It also takes the variable name and the name of a cache file outfile that will be written to disk. 
    dryrunOn is the switch from a physical download to just polling the esgf node without any download.

    varname: variable
    D: incomplete filedescriptors: the dictionary that contains the files that are already available locally
    year1_model, year2_model: needed filedescriptor year1 and 2
    header: unique filedescriptor indicator e.g. CMIP5_CNRM-CM5_Amon_historical_r1i1p1_2003_2010_hus
    outfile: cache file
    outfile2: missing cache file
    download: download (either dryrun or for reals) flag 
    
    """
    # this is needed mostly for parallel processes that may
    # go tits-up from time to time due to random path mixes
    if which_synda('synda') is not None:
        pass
    else:
        print >> sys.stderr, "No synda executable found in path. Exiting."
        sys.exit(1)
    entries = searchoutput.split('\n')[:-1]
    if len(entries) > 0:
        for entry in entries:
            label=str(entry.split()[0])
            file_name = entry.split()[3]
            if header.split('_')[1] == file_name.split('.')[3]:
                time_range = file_name.split('_')[-1].strip('.nc')
                time1 = time_range.split('-')[0]
                time2 = time_range.split('-')[1]
                year1 = date_handling(time1,time2)[0]
                year2 = date_handling(time1,time2)[1]
                if time_handling(year1, year1_model, year2, year2_model)[0] is True:
                    if label=='done':
                        file_name_complete = ".".join(file_name.split('.')[:10]) + '.' + varname + '.' + ".".join(file_name.split('.')[10:])
                        filepath_complete = '/sdt/data/c' + file_name_complete.replace('.','/').strip('/nc') + '.nc'
                        fn = filepath_complete.split('/')[-1]
                        # synda should not cache files in dictionary D
                        # these belong to incomplete filedescriptors but are already on disk
                        if fn not in D[header]:
                            with open(outfile, 'a') as file:
                                file.write(header + ' ' + filepath_complete + ' ' + 'INSTALLED' + '\n')
                            if verbose is True:
                                print('File exists in local /sdt/data, path: ' + filepath_complete)
                                # no download #
                    elif label=='new':
                        if download is True:
                            file_name_new = ".".join(file_name.split('.')[:10]) + '.' + varname + '.' + ".".join(file_name.split('.')[10:])
                            filepath_new = '/sdt/data/c' + file_name_new.replace('.','/').strip('/nc') + '.nc'
                            fn = filepath_new.split('/')[-1]
                            # synda should not download files in dictionary D
                            # these belong to incomplete filedescriptors but are already on disk
                            if fn not in D[header]:
                                if dryrunOn is True:
                                    if verbose is True:
                                        print('Needed file %s doesnt exist in local /sdt/data but is on ESGF nodes, enable download to get it' % file_name)
                                        print('Download enabled in dryrun mode...')
                                        print('Synda found file: ' + file_name)
                                        print('If installed, full path would be: ' + filepath_new)
                                    with open(outfile, 'a') as file:
                                        file.write(header + ' ' + filepath_new + ' ' + 'NOT-YET-INSTALLED' + '\n')
                                        # no download, dryrun only #
                                else:
                                    synda_install = which_synda('synda') +  ' install ' + file_name
                                    proc = subprocess.Popen(synda_install, stdout=subprocess.PIPE, stdin=subprocess.PIPE, shell=True)
                                    dll ='\n'
                                    (out, err) = proc.communicate(input=dll)
                                    if err is not None:
                                        print >> sys.stderr, "An error has occured while starting the download:"
                                        print >> sys.stderr, err
                                        sys.exit(1)
                                    else:
                                        with open(outfile, 'a') as file:
                                            file.write(header + ' ' + filepath_new + ' ' + 'INSTALLED' + '\n')
                                    if verbose is True:
                                        print('Needed file %s doesnt exist in local /sdt/data but is on ESGF nodes' % file_name)
                                        print('Download enabled in full install mode...')
                                        print('Downloading file: ' + file_name)
                                        print('Full path: ' + filepath_new)
                                        # yes download #
                else:
                    if verbose is True:
                        print('WARNING: synda - not cached due to requested period mismatch: ' + header + ' ' + file_name)
                    return 0
            else:
                if verbose is True:
                    print('WARNING: synda - not cached due to model mismatch: ' + header + ' ' + file_name)
                return 0
    else:
        if verbose is True:
            print('WARNING: synda - missing data altogether: ' + header)
        return 0
    if os.path.exists(outfile):
        fix_duplicate_entries(outfile)

def cache_merge(file1,file2,finalFile):
    """
    Function that takes two cache files and merges them
    into a single one. Caution -- note the order:
    file1 = local datasource cache
    file2 = local synda cache
    """
    f1 = open(file1, 'r')
    f2 = open(file2, 'r')
    ff = open(finalFile, 'w')
    r1 = f1.readlines()
    r2 = f2.readlines()
    for b in r2:
        for a in r1:
            if a.split()[1].split('/')[-1] == b.split()[1].split('/')[-1]:
                ff.write(a.split()[0] + ' ' + a.split()[1] + '\n')
            else:
                ff.write(a.split()[0] + ' ' + a.split()[1] + '\n')
                ff.write(b.split()[0] + ' ' + b.split()[1] + '\n')
    ff.close()
    if os.path.exists(finalFile):
        fix_duplicate_entries(finalFile)                
    
# ---- final user-friendly cache generator
def final_cache(parfile,ofile1,finalfile):
    """
    Function that generates the final user-friendly
    single cache file; this can easily be used
    in various analyses; file legend:
    Database | data_status | Percent complete | available_data
    ---------------------------------------------
    CMIP5_MIROC5_Amon_historical_r1i1p1_2003_2010_hus (complete,incomplete or missing) [file_list, if available]
    """
    pparfile = 'prepended_' + parfile
    car = open(pparfile, 'r')
    lis = car.readlines()
    ff = open(finalfile, 'w')
    if os.path.exists(ofile1):
        of1 = open(ofile1, 'r')
        ofl1 = of1.readlines()
        o1 = [(a.split()[0],a.split()[1]) for a in ofl1]
        for b in lis:
            tt = []
            hh = []
            header = b.split()[0] + '_'+ b.split()[1] + '_' + b.split()[2]\
                     + '_' + b.split()[3] + '_' + b.split()[4] + '_' + b.split()[5]\
                     + '_' + b.split()[6] + '_' + b.split()[7]

            for h in o1:
                if header == h[0]:
                    y = h[1].split('/')[-1].strip('.nc').split('_')[-1].split('-')
                    # y could be some dodgy stuff if file not proper formatted
                    if len(y) == 2: 
                        yr1 = date_handling(y[0],y[1])[0]
                        yr2 = date_handling(y[0],y[1])[1]
                        tt.append(yr1)
                        tt.append(yr2)
                        hh.append(h[1])
                    else:
                        print('File: _date1-date2.nc not properly formatted...skipping it')
            y1 = int(b.split()[5])
            y2 = int(b.split()[6])
            # let's see how we do with time
            if len(tt) > 0:
                if get_overlap(tt,y1,y2)[1] == 1:
                    # we have contiguous time
                    if get_overlap(tt,y1,y2)[0] == 1:
                        ff.write(header + ' complete 1.0 ' + str(hh) + '\n')
                    else:
                        fdt = get_overlap(tt,y1,y2)[0]
                        ff.write(header + ' incomplete ' + '%.2f' % fdt + ' ' + str(hh) + '\n')
                else:
                    # we have gaps
                    if get_overlap(tt,y1,y2)[0] == 1:
                        ff.write(header + ' complete(DATAGAPS) 1.0 ' + str(hh) + '\n')
                    else:
                        fdt = get_overlap(tt,y1,y2)[0]
                        ff.write(header + ' incomplete(DATAGAPS) ' + '%.2f' % fdt + ' ' + str(hh) + '\n')
            else:
                ff.write(header + ' missing' + '\n')

#---- function that returns the amount of overlap
# between needed data and available data
def get_overlap(tt, my1, my2):
    """
    function that returns the amount of overlap
    between needed data and available data
    Returns a fractional float
    li: list of years from data (1-dim, even number of elements)
    my1,my2: required model years
    """
    nt = len(tt)
    my = float(my2 - my1)
    if nt == 2:
        # single file, no gaps in data
        if min(tt) >= my1 and max(tt) <= my2:
            # completely inside
            df = (max(tt) - min(tt))/my
        elif min(tt) >= my1 and max(tt) >= my2:
            # right plus
            df = (my2 - min(tt))/my
        elif min(tt) <= my1 and max(tt) <= my2:
            # left plus
            df = (max(tt) - my1)/my
        elif my1 >= min(tt) and my2 <= max(tt):
            df = 1
        return df,1
    else:
        #multiple files, checking for gaps in data
        b = max(tt) - min(tt)
        el = [tt[i] - tt[i-1] for i in range(1,nt)]
        if sum(el) == b:
            # multiple files, no gaps in data
            if min(tt) >= my1 and max(tt) <= my2:
                # completely inside
                df = (max(tt) - min(tt))/my
            elif min(tt) >= my1 and max(tt) >= my2:
                # right plus
                df = (my2 - min(tt))/my
            elif min(tt) <= my1 and max(tt) <= my2:
                # left plus
                df = (max(tt) - my1)/my
            elif my1 >= min(tt) and my2 <= max(tt):
                df = 1
            return df,1
        else:
            # there are gaps!!
            # but we dont deal with them here
            df = 1
            print('WARNING: there are gaps in data!')
            print(tt)
            return df,2

        
    #dtl = [tt[i] - tt[i-1] for i in range(1,nt)]
    

def print_final_stats(sfile):
    """
    print some final stats
    To understand the output, by filedescriptor we mean any file indicator
    of form e.g. CMIP5_MIROC5_Amon_historical_r1i1p1_2003_2010_hus that is fully
    determined by its parameters; there could be multiple .nc files
    covering a single filedescriptor, alas there could be just one.
    """
    ff = open(sfile, 'r')
    lff = ff.readlines()
    c = [a for a in lff if a.split()[1] == 'complete']
    ic = [b for b in lff if b.split()[1] == 'incomplete']
    mi = [d for d in lff if d.split()[1] == 'missing']
    gc = [a for a in lff if a.split()[1] == 'complete(DATAGAPS)']
    gic = [b for b in lff if b.split()[1] == 'incomplete(DATAGAPS)']
    prcc = [float(a.split()[2]) for a in lff if a.split()[1] == 'incomplete']
    print('---------------------------')
    if len(gc) != 0 and len(gic) != 0:
        print('============================')
        print('WARNING: THERE ARE DATA GAPS!')
        print('============================')
    print('     Total needed filedescriptors: %i' % len(lff))
    print('         Complete filedescriptors: %i' % len(c))
    print('       Incomplete filedescriptors: %i' % len(ic))
    print('          Missing filedescriptors: %i' % len(mi))
    print('           Complete dbs with gaps: %i' % len(gc))
    print('         Incomplete dbs with gaps: %i' % len(gic))
    print('      Avg coverage for incomplete: %.2f' % np.mean(prcc))
    print('---------------------------')

# ---- plotting the filedescriptors in pie charts
def plotter(cachefile,saveDir):
    """
    simple pie chart plotting function
    """
    # get matplotlib
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    # plot overall
    ff = open(cachefile,'r')
    lff = ff.readlines()
    c = [a for a in lff if a.split()[1] == 'complete']
    ic = [b for b in lff if b.split()[1] == 'incomplete']
    mi = [d for d in lff if d.split()[1] == 'missing']
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = 'complete', 'incomplete', 'missing'
    sizes = [len(c), len(ic), len(mi)]
    explode = (0, 0.1, 0)  # only "explode" the 2nd slice (i.e. 'incomplete')
    # plot
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('Overall data coverage')
    saveLoc = saveDir + '/overall.png'
    plt.savefig(saveLoc)
    # plot only missing
    c2 = [a.split()[0].split('_')[1] for a in lff if a.split()[1] == 'missing']
    c2s = list(set(c2))
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = c2s
    sizes = [c2.count(a) for a in c2s]
    # plot
    fig2, ax2 = plt.subplots()
    ax2.pie(sizes, labels=labels, autopct='%1.1f%%',startangle=90)
    ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('Missing data by model')
    saveLoc = saveDir + '/missing.png'
    plt.savefig(saveLoc)
    # plot only incomplete
    c2 = [a.split()[0].split('_')[1] for a in lff if a.split()[1] == 'incomplete']
    c2s = list(set(c2))
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = c2s
    sizes = [c2.count(a) for a in c2s]
    # plot
    fig3, ax3 = plt.subplots()
    ax3.pie(sizes, labels=labels, autopct='%1.1f%%',startangle=90)
    ax3.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('Incomplete data by model')
    saveLoc = saveDir + '/incomplete.png'
    plt.savefig(saveLoc)

# ---- synda check download
def synda_check_dll():
    """
    Easy checker on current downloads
    """
    print('Your files(s) are being downloaded.')
    print('You can check the download progress with synda queue, see output below')
    synda_queue = 'synda queue'
    proc = subprocess.Popen(synda_queue, stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()
    print(out)
    statusreport = out.split('\n')
    for entry in statusreport:
        if len(entry)>0:
            if entry.split()[0] == 'waiting':
                print('%i files are waiting, totalling %.2f MB disk' % (int(entry.split()[1]),float(entry.split()[2])))
    synda_watch = 'synda watch'
    proc = subprocess.Popen(synda_watch, stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()
    print(out)

# -------------------------------------------------------------------------
#      Parse the command line options.
# -------------------------------------------------------------------------

# ---- Initialise command line argument variables.
params_file       = None
userVars          = False
db                = []
syndacall         = False
download          = False
dryrunOn          = False
fpars             = []
vpars             = []
verbose           = False

# ---- Syntax of options, as required by getopt command.
# ---- Short form.
shortop = "hp:g:r:d:i:n:t:f:m:sc:e:"
# ---- Long form.
longop = [
   "help",
   "params-file=",
   "user-input",
   "datasource=",
   "synda",
   "download",
   "dryrun",
   "fileparams=",
   "uservars=",
   "verbose"
]

# ---- Get command-line arguments.
try:
  opts, args = getopt.getopt(sys.argv[1:], shortop, longop)
except getopt.GetoptError:
  usage()
  sys.exit(1)

# ---- We will record the command line arguments to cmip5datafinder.py in a file called
#      cmip5datafinder.param. This file should be used if a further need to run the code arises
command_string = 'cmip5datafinder.py '

# ---- Parse command-line arguments.  Arguments are returned as strings, so
#      convert type as necessary.
for o, a in opts:
    if o in ("-h", "--help"):
        usage()
        sys.exit(0)
    elif o in ("-p", "--params-file"):
        params_file = a
        command_string = command_string + ' -p ' + a
    elif o in ("--user-input"):
      userVars = True
      command_string = command_string + ' --user-input '
    elif o in ("--datasource"):
        db.append(a)
        command_string = command_string + ' --datasource ' + a
    elif o in ("--synda"):
      syndacall = True
      command_string = command_string + ' --synda '
    elif o in ("--download"):
      download = True
      command_string = command_string + ' --download '
    elif o in ("--dryrun"):
      dryrunOn = True
      command_string = command_string + ' --dryrun '
    elif o in ("--fileparams"):
        fpars.append(a)
        command_string = command_string + ' --fileparams ' + a
    elif o in ("--uservars"):
        vpars.append(a)
        command_string = command_string + ' --uservars ' + a 
    elif o in ("--verbose"):
      verbose = True
      command_string = command_string + ' --verbose '
    else:
        print >> sys.stderr, "Unknown option:", o
        usage()
        sys.exit(1)

# ---- Check that all required arguments are specified, else exit.
if not params_file:
    if not userVars:
        print >> sys.stderr, "No parameter file specified and no user file definitions"
        print >> sys.stderr, "Use --params-file to specify the parameter file or --user-vars followed"
        print >> sys.stderr, "by command-line options for file parameters. Exiting."
        sys.exit(1)
    else:
        print >> sys.stderr, "Using the user's specified file parameters"
        if not fpars:
            print >> sys.stderr, "You need to specify a number of file params e.g. CMIP5,MPI-ESM-LR,Amon,historical,r1i1p1,1910,1919"
            print >> sys.stderr, "Use the --fileparams option for this"
            sys.exit(1)
        if not vpars:
            print >> sys.stderr, "You need to specify a number of variables e.g. tro3"
            print >> sys.stderr, "Use the --uservars option for this"
            sys.exit(1)
if params_file and userVars:
    print >> sys.stderr, "Use --params-file to specify the parameter file OR --user-input followed"
    print >> sys.stderr, "by command-line options for file and variables parameters. Can not use both options! Exiting."
    sys.exit(1)
if not db:
    print >> sys.stderr, "No local datasource to search specified"
    print >> sys.stderr, "Use --datasource to specify a valid datasource e.g. badc or dkrz. Exiting..."
    sys.exit(1)

# -------------------------------------------------------------------------
#      Status message.  Report all supplied arguments.
# -------------------------------------------------------------------------

if verbose is True:
    intro = """\
          This is a flexible tool to generate cache files from local datasources and ESGF nodes.
          This makes use of synda for querying ESGF nodes as well.
          For problems or queries, email valeriu.predoi@ncas.ac.uk. Have fun!
          
          Code functionality:
          1. Given a command line set of arguments or an input file, the code looks for cmip5
          files locally and returns the physical paths to the found files;
          2. If files are not found, the user has the option to download missing files from ESGF
          nodes via synda;
          3. Finally, the code writes cache files:
           - cache_cmip5_[SERVER].txt local cache file
           - cache_cmip5_combined_[SERVER].txt combined cache file (synda+local)
           - cache_cmip5_synda_[SERVER].txt synda cache file
           - cache_err.out error out while caching
           - missing_cache_cmip5_[SERVER].txt local missing files
           - missing_cache_cmip5_combined_[SERVER].txt missing files (synda+local)
           - missing_cache_cmip5_synda_[SERVER].txt synda missing files

          Example run:
          (with param file) python cmip5datafinder.py -p perfmetrics.txt --download --dryrun --verbose --datasource badc
          (with command line args) python cmip5datafinder.py --user-input --fileparams CMIP5 --fileparams bcc-csm1-1 --fileparams --fileparams Amon
          --fileparams historical --fileparams r1i1p1 --fileparams 1982 --fileparams 2014 --uservars clt --uservars tro3 --uservars pr --datasource badc
          --verbose
          """
    print >> sys.stdout, intro
    print >> sys.stdout
    print >> sys.stdout, "####################################################"
    print >> sys.stdout, "#                 CMIP5 Data Finder                #"
    print >> sys.stdout, "####################################################"
    print >> sys.stdout
    print >> sys.stdout, "Parsed input arguments:"
    print >> sys.stdout
    if params_file:
        print >> sys.stdout,"Running with parameters file:", params_file
    else:
        if len(fpars) < 7:
            print >> sys.stderr, "Too few file parameters (CMIP5 needs exacly 7: e.g. CMIP5 MPI-ESM-LR Amon historical r1i1p1 1980 2005)"
            sys.exit(1)
        else:
            print >> sys.stdout,"Running with user-defined file parameters and variables     "
            print >> sys.stdout,"File      :", fpars[0]
            print >> sys.stdout,"Experiment:", fpars[1]
            print >> sys.stdout,"Medium    :", fpars[2]
            print >> sys.stdout,"Type      :", fpars[3]
            print >> sys.stdout,"Ensemble  :", fpars[4]
            print >> sys.stdout,"Year1     :", fpars[5]
            print >> sys.stdout,"Year2     :", fpars[6]
            print >> sys.stdout,"Var(s)    :", vpars[0]
    print >> sys.stdout

# ---- if we are using synda
# ---- Get the synda path or exit here
if syndacall is True:
    if verbose is True:
        print('You are going to use SYNDA to download data...')
        print('Looking up synda executable...')
    if which_synda('synda') is not None:
        print >> sys.stdout, "Synda found...OK" 
        print >> sys.stdout, which_synda('synda')
    else:
        print >> sys.stderr, "No synda executable found in path. Exiting."
        sys.exit(1)

    if verbose is True:
        # ---- Have us some information from the synda configuration file
        # ---- one can add more info if needed, currently just data server
        print('\n---------------------------------------------')
        print('Information about synda configuration:')
        print('---------------------------------------------')
        synda_conf_file = which_synda('synda').rsplit('/',2)[0] + '/conf/sdt.conf'
        print ('Synda conf file %s' % synda_conf_file)
        with open(synda_conf_file, 'r') as file:
            for line in file:
                if line.split('=')[0]=='indexes':
                    data_server = line.split('=')[1]
                    print('ESGF data node: %s' % data_server.split()[0])

# ---- Write ASCII file holding cache_BADC.py command.
pfile = open('cmip5datafinder.param','w')
pfile.write(command_string + "\n")
pfile.close()

# ---- Write cache files ---- #
# ---- hardcoded names so we standardize analyses
# ---- start overall timing
t10 = time.time()
# ---- db is a list and we run on each called datasource
for d in db:
    # we need to firstly remove any pre existent cache dirs
    drb = 'cache_files_' + d
    print('Removing all pre-existent cache directories...')
    if os.path.isdir(drb):
        rrc = 'rm -r ' + drb
        proc = subprocess.Popen(rrc, stdout=subprocess.PIPE, shell=True)
        (out, err) = proc.communicate()
    print('Polling %s datasource...' % d)
    # ...then create new one, standard name cache_files_[SERVER] eg cache_files_badc
    print('We will be writing all needed cache files to %s directory...' % drb)
    mkc = 'mkdir -p ' + drb
    proc = subprocess.Popen(mkc, stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()
    # place the cache files
    pfile2 = drb + '/cache_cmip5_' + d + '.txt'
    pfile3 = drb + '/missing_cache_cmip5_' + d + '.txt'
    if syndacall is True:
        pfile4 = drb + '/cache_cmip5_synda_' + d + '.txt'
        pfile5 = drb + '/missing_cache_cmip5_synda_' + d + '.txt'
    errorfile = drb + '/cache_err.out'
    if params_file:
        nm = 'cache_' + params_file + '-' + d
        if os.path.exists(nm):
            os.remove(nm)
    else:
        nm = 'cache_user.txt-' + d
        if os.path.exists(nm):
            os.remove(nm)

    # ---- get root directory
    if verbose is True:
        print('Using %s as local searchable datasource' % d)
    if d == 'badc':
        host_root = '/badc/cmip5/data/cmip5/output1/'
        ls_host_root = lsladir(host_root)
        # this is a standard for badc
        latestDir = '/latest/'

    # ---- start timer
    t1 = time.time()
    
    # ---- get the params file
    if params_file:
        paramfile, paramfile_extension = os.path.splitext(params_file)

        # ---- txt
        if paramfile_extension=='.txt':
            # ---- Parse a generic text parameters file ---- #
            # ---- with the specified variable(s) ---- #
            ##############################################################
            """
            NOTE: for streamlining
            Build a standardized .txt parameter file as follows:
            each row must be a standard specific file descriptor e.g.
            cmip  experiment type1 type2    ensemble yr1  yr2  variable
            ----------------------------------------------------------
            CMIP5 MPI-ESM-LR Amon historical r1i1p1  1900  1982   tro3 
    
            IT IS IMPORTANT TO KEEP THIS ORDER OTHERWISE THINGS CAN GET VERY MESSY !!!
            """
            if syndacall is True:
                # first poll the local server
                if verbose is True:
                    write_cache_direct(params_file,ls_host_root,host_root,pfile2,pfile3,errorfile,latestDir,verbose)
                else:
                    write_cache_direct(params_file,ls_host_root,host_root,pfile2,pfile3,errorfile,latestDir,verbose=False)
                print_stats(pfile2,pfile3)
                # check for incomplete/missing filedescriptors
                if os.path.exists(pfile3):
                    ar = open(pfile3, 'r')
                    lls = [line for line in ar if line.split()[0].split('_')[0] == 'CMIP5']
                    lenitemlist = len(lls)
                    cat11 = [(p.split()[0],'dope') for p in lls if p.split()[1] == 'ERROR-MISSING']
                    cat21 = [(p.split()[0],p.split()[2]) for p in lls if p.split()[1] == 'INCOMPLETE']
                    # construct two dictionaries:
                    # A: contains all missing filedescriptors
                    # B: contains the incomplete filedescriptors
                    A = {}
                    B = {}
                    for item in cat21:
                        A.setdefault(item[0],[]).append(item[1])
                    for item in cat11:
                        B.setdefault(item[0],[]).append(item[1])
                    # convolve A and B so synda will download only the A's 'dope' (missing)
                    # and the bits from B that are not already on disk
                    Z = dict(A, **B)
                    if verbose is True:
                        print('\n-----------------------------------------------------------------------------------------------------')
                        print('We parsed a missing LOCAL data param file. We have missing/incomplete files for %i filedescriptors: ' % lenitemlist)
                        print('Calling SYNDA to look for data in /sdt/data or download what is not found...')
                        print('-------------------------------------------------------------------------------------------------------')
                    for it in lls:
                        ite = it.split()[0].split('_')
                        v1 = ite[7]
                        header = it.split()[0]
                        model_data = ite[0] + ' '+ ite[1] + ' ' + ite[2]\
                                     + ' ' + ite[3] + ' ' + ite[4]
                        yr1 = int(ite[5])
                        yr2 = int(ite[6])
                        # call synda search
                        outpt = synda_search(model_data,v1)
                        if download is True:
                            if verbose is True:
                                if dryrunOn:
                                    s = synda_dll(outpt,v1,yr1,yr2,header,Z,pfile4,pfile5,download=True,dryrunOn=True,verbose=True)
                                else:
                                    s = synda_dll(outpt,v1,yr1,yr2,header,Z,pfile4,pfile5,download=True,dryrunOn=False,verbose=True)
                            else:
                                if dryrunOn:
                                    s = synda_dll(outpt,v1,yr1,yr2,header,Z,pfile4,pfile5,download=True,dryrunOn=True,verbose=False)
                                else:
                                    s = synda_dll(outpt,v1,yr1,yr2,header,Z,pfile4,pfile5,download=True,dryrunOn=False,verbose=False)
                        else:
                            if verbose is True:
                                s = synda_dll(outpt,v1,yr1,yr2,header,Z,pfile4,pfile5,download=False,dryrunOn=False,verbose=True)
                            else:
                                s = synda_dll(outpt,v1,yr1,yr2,header,Z,pfile4,pfile5,download=False,dryrunOn=False,verbose=False)
                        if s == 0:
                            with open(pfile5, 'a') as file:
                                file.write(header + ' ' + 'ERROR-MISSING' + '\n')
                                file.close()
                    if os.path.exists(pfile4):
                        fix_duplicate_entries(pfile4)
                    if os.path.exists(errorfile):
                        fix_duplicate_entries(errorfile)
                    print_stats(pfile4,pfile5)

                    # final cache merging and cleanup
                    if os.path.exists(pfile2) and os.path.exists(pfile4):
                        # create a composite file using caches from sever and synda
                        compf = drb + '/cache_cmip5_combined_' + d + '.txt'
                        cache_merge(pfile2,pfile4,compf)
                        final_cache(params_file,compf,nm)
                        print_final_stats(nm)
                        plotter(nm,drb)
                    else:
                        # looks like synda didnt find anything extra
                        if os.path.exists(pfile2):
                            cpc = 'cp ' + pfile2 + ' ' + drb + '/cache_cmip5_combined_' + d + '.txt'
                            proc = subprocess.Popen(cpc, stdout=subprocess.PIPE, shell=True)
                            (out, err) = proc.communicate()
                            final_cache(params_file,pfile2,nm)
                            print_final_stats(nm)
                            plotter(nm,drb)
                        else:
                            # looks like there is nothing in local but synda found extra
                            if os.path.exists(pfile4):
                                cpc = 'cp ' + pfile4 + ' ' + drb + '/cache_cmip5_combined_' + d + '.txt'
                                proc = subprocess.Popen(cpc, stdout=subprocess.PIPE, shell=True)
                                (out, err) = proc.communicate()
                                final_cache(params_file,pfile4,nm)
                                print_final_stats(nm)
                                plotter(nm,drb)
                    # in case synda missed some filedescriptors
                    if os.path.exists(pfile5):
                        fix_duplicate_entries(pfile5)
                        cpc = 'cp ' + pfile5 + ' ' + drb + '/missing_cache_cmip5_combined_' + d + '.txt'
                        proc = subprocess.Popen(cpc, stdout=subprocess.PIPE, shell=True)
                        (out, err) = proc.communicate()
                else:
                    # no need to call synda if we found all needed filedescriptors on server
                    print('Cached all needed data from local datasource %s' % d)
                    if os.path.exists(pfile2):
                        cpc = 'cp ' + pfile2 + ' ' + drb + '/cache_cmip5_combined_' + d + '.txt'
                        proc = subprocess.Popen(cpc, stdout=subprocess.PIPE, shell=True)
                        (out, err) = proc.communicate()
                        final_cache(params_file,pfile2,nm)
                        print_final_stats(nm)
                        plotter(nm,drb)
                    #sys.exit(0)
            else:
                # not calling synda at all
                if verbose is True:
                    print('\n-------------------------------------------------------------------------------------')
                    print('We have looked at existing files LOCALLY only: ')
                    print('Here is what we found:')
                    print('---------------------------------------------------------------------------------------')
                    write_cache_direct(params_file,ls_host_root,host_root,pfile2,pfile3,errorfile,latestDir,verbose)
                else:
                    write_cache_direct(params_file,ls_host_root,host_root,pfile2,pfile3,errorfile,latestDir,verbose=False)
                if os.path.exists(errorfile):
                    fix_duplicate_entries(errorfile)
                print_stats(pfile2,pfile3)
                if os.path.exists(pfile2):
                    cpc = 'cp ' + pfile2 + ' ' + drb + '/cache_cmip5_combined_' + d + '.txt'
                    proc = subprocess.Popen(cpc, stdout=subprocess.PIPE, shell=True)
                    (out, err) = proc.communicate()
                    final_cache(params_file,pfile2,nm)
                    print_final_stats(nm)
                    plotter(nm,drb)
                if os.path.exists(pfile3):
                    cpc = 'cp ' + pfile3 + ' ' + drb + '/missing_cache_cmip5_combined_' + d + '.txt'
                    proc = subprocess.Popen(cpc, stdout=subprocess.PIPE, shell=True)
                    (out, err) = proc.communicate()


    elif userVars:
    
        # ---- user command line arguments parsed here
        for vi in vpars:
            if os.path.exists('prepended_temp.txt'):
                os.remove('prepended_temp.txt')
            #if os.path.exists('temp.txt'):
            #    os.remove('temp.txt')
            header = fpars[0] + '_'+ fpars[1] + '_' + fpars[2]\
                     + '_' + fpars[3] + '_' + fpars[4] + '_' + str(fpars[5])+ '_' + str(fpars[6])\
                     + '_' + vi
            model_data = fpars[0] + ' '+ fpars[1] + ' ' + fpars[2]\
                     + ' ' + fpars[3] + ' ' + fpars[4]
            if verbose is True:
                print('Looking at variable %s' % vi)
                print('Model data for: ', model_data + '\n')
            yr1 = fpars[5]
            yr2 = fpars[6]
            tempfile = open('temp.txt', 'a')
            templine = model_data + ' ' + str(yr1) + ' ' + str(yr2) + ' ' + vi + '\n'
            tempfile.write(templine)
            tempfile.close()
            if verbose is True:
                write_cache_direct('temp.txt',ls_host_root,host_root,pfile2,pfile3,errorfile,latestDir,verbose)
            else:
                write_cache_direct('temp.txt',ls_host_root,host_root,pfile2,pfile3,errorfile,latestDir,verbose=False)
            print_stats(pfile2,pfile3)
            if syndacall is True:
                if os.path.exists(pfile3):
                    if verbose is True:
                        print('\n-------------------------------------------------------------------------------------')
                        print('We are missing files for our needed filedescriptor: %s' % model_data + ' ' + str(yr1) + ' ' + str(yr2) + ' ' + vi)
                        print('Calling SYNDA to look for data in /sdt/data or download what is not found...')
                        print('---------------------------------------------------------------------------------------')
                    ar = open(pfile3, 'r')
                    lls = [line for line in ar if line.split()[0].split('_')[0] == 'CMIP5']
                    lenitemlist = len(lls)
                    cat11 = [(p.split()[0],'dope') for p in lls if p.split()[1] == 'ERROR-MISSING']
                    cat21 = [(p.split()[0],p.split()[2]) for p in lls if p.split()[1] == 'INCOMPLETE']
                    A = {}
                    B = {}
                    for item in cat21:
                        A.setdefault(item[0],[]).append(item[1])
                    for item in cat11:
                        B.setdefault(item[0],[]).append(item[1])
                    Z = dict(A, **B)
                    outpt = synda_search(model_data,vi)
                    if download is True:
                        if verbose is True:
                            if dryrunOn:
                                s = synda_dll(outpt,vi,yr1,yr2,header,Z,pfile4,pfile5,download=True,dryrunOn=True,verbose=True)
                            else:
                                s = synda_dll(outpt,vi,yr1,yr2,header,Z,pfile4,pfile5,download=True,dryrunOn=False,verbose=True)
                        else:
                            if dryrunOn:
                                s = synda_dll(outpt,vi,yr1,yr2,header,Z,pfile4,pfile5,download=True,dryrunOn=True,verbose=False)
                            else:
                                s = synda_dll(outpt,vi,yr1,yr2,header,Z,pfile4,pfile5,download=True,dryrunOn=False,verbose=False)
                    else:
                        if verbose is True:
                            s = synda_dll(outpt,vi,yr1,yr2,header,Z,pfile4,pfile5,download=False,dryrunOn=False,verbose=True)
                        else:
                            s = synda_dll(outpt,vi,yr1,yr2,header,Z,pfile4,pfile5,download=False,dryrunOn=False,verbose=False)
                    if s == 0:
                        with open(pfile5, 'a') as file:
                            file.write(header + ' ' + 'ERROR-MISSING' + '\n')
                            file.close()
                    if os.path.exists(pfile4):
                        fix_duplicate_entries(pfile4)
                    if os.path.exists(errorfile):
                        fix_duplicate_entries(errorfile)
                    print_stats(pfile4,pfile5)
                    # final cache merging and cleanup
                    if os.path.exists(pfile2) and os.path.exists(pfile4):
                        # create a composite file using caches from sever and synda
                        compf = drb + '/cache_cmip5_combined_' + d + '.txt'
                        cache_merge(pfile2,pfile4,compf)
                        final_cache('temp.txt',compf,nm)
                        print_final_stats(nm)
                    else:
                        # looks like synda didnt find anything extra
                        if os.path.exists(pfile2):
                            cpc = 'cp ' + pfile2 + ' ' + drb + '/cache_cmip5_combined_' + d + '.txt'
                            proc = subprocess.Popen(cpc, stdout=subprocess.PIPE, shell=True)
                            (out, err) = proc.communicate()
                            final_cache('temp.txt',pfile2,nm)
                            print_final_stats(nm)
                        else:
                            # looks like there is nothing in local but synda found extra
                            if os.path.exists(pfile4):
                                cpc = 'cp ' + pfile4 + ' ' + drb + '/cache_cmip5_combined_' + d + '.txt'
                                proc = subprocess.Popen(cpc, stdout=subprocess.PIPE, shell=True)
                                (out, err) = proc.communicate()
                                final_cache('temp.txt',pfile4,nm)
                                print_final_stats(nm)
                    # in case synda missed some filedescriptors
                    if os.path.exists(pfile5):
                        fix_duplicate_entries(pfile5)
                        cpc = 'cp ' + pfile5 + ' ' + drb + '/missing_cache_cmip5_combined_' + d + '.txt'
                        proc = subprocess.Popen(cpc, stdout=subprocess.PIPE, shell=True)
                        (out, err) = proc.communicate()
                else:
                    # no need to call synda if we found all needed filedescriptors on server
                    print('Cached all data from local datasource %s' % d)
                    if os.path.exists(pfile2):
                        cpc = 'cp ' + pfile2 + ' ' + drb + '/cache_cmip5_combined_' + d + '.txt'
                        proc = subprocess.Popen(cpc, stdout=subprocess.PIPE, shell=True)
                        (out, err) = proc.communicate()
                        final_cache('temp.txt',pfile2,nm)
                        print_final_stats(nm)
            # not calling synda at all
            if verbose is True:
                print('\n-------------------------------------------------------------------------------------')
                print('We have looked at existing files LOCALLY only: ')
                print('Here is what we found:')
                print('---------------------------------------------------------------------------------------')
            if os.path.exists(errorfile):
                fix_duplicate_entries(errorfile)
                print_stats(pfile2,pfile3)
            if os.path.exists(pfile2):
                cpc = 'cp ' + pfile2 + ' ' + drb + '/cache_cmip5_combined_' + d + '.txt'
                proc = subprocess.Popen(cpc, stdout=subprocess.PIPE, shell=True)
                (out, err) = proc.communicate()
                final_cache('temp.txt',pfile2,nm)
                print_final_stats(nm)
            if os.path.exists(pfile3):
                cpc = 'cp ' + pfile3 + ' ' + drb + '/missing_cache_cmip5_combined_' + d + '.txt'
                proc = subprocess.Popen(cpc, stdout=subprocess.PIPE, shell=True)
                (out, err) = proc.communicate()
            #os.remove('temp.txt')
            os.remove('prepended_temp.txt')

    # ---- timing and exit
    t2 = time.time()
    dt = t2 - t1
    if verbose is True:
        print('=================================')
        print('DONE! with datasource %s' % d)
        print('Time elapsed: %.1f seconds' % dt)
        print('=================================')
    print('Time elapsed: %.1f s' % dt)

# ---- finish, cleanup and exit
if params_file:
    prp = 'prepended_' + params_file
    os.remove(prp)
if userVars:
    os.remove('temp.txt')
t20 = time.time()
dt0 = t20 - t10
if verbose is True:
    print('==================================================')
    print('DONE! with all datasources')
    print('Time elapsed: %.1f seconds' % dt0)
    print('If your data is fully cached, you deserve a beer :)')
    print('==================================================')
print('Time elapsed: %.1f s' % dt0)

# ---- end of code
