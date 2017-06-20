#!/home/valeriu/sdt/bin/python

"""

Script that searches for data using a valid ESGF node, and that builds a 
local cache using the results of the search. Currently for BADC archive only.
Can be generalized for any other archive.

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
This is a flexible tool to generate locacal caches from ESGF nodes.
This makes use of synda for querying the local ESGF node (BADC only for now).
It will write a cache file pointing to local physical paths for data.
For queries, email valeriu.predoi@ncas.ac.uk. Have fun!

Usage:
  cache_BADC.py [options]
  -p, --params-file <file>    Namelist file (xml) or text file (txt) or any other input file 
                              e.g. for xml: --params-file ESMValTool/nml/namelist_myTest.xml
                              e.g. for text:
                              e.g. for yaml:
                              This option is REQUIRED if --user-input is not present
  -h, --help                  Display this message and exit
  --user-input                Flag for user defined file and variables parameters (to be inputted at command line)
                              This option is REQUIRED if --params-file is not present
  --fileparams                If --user-input is used, this serial option passes one data file argument at a time
                              If --user-input is used, this serial option is REQUIRED
                              e.g. --fileparams CMIP5 --fileparams MPI-ESM-LR --fileparams Amon  --fileparams historical
                              --fileparams r1i1p1 --fileparams 1910 --fileparams 1919
  --uservars                  If --user-input is used, this serial option passes one variable argument at a time
                              If --user-input is used, this serial option is REQUIRED
                              e.g. --uservars tro3

"""
  print >> sys.stderr, msg

########################################
# ---- Operational functions here ---- #
########################################

# ---- get the path to synda executable
def which_synda(synda):
    """

    This function returns the path to the synda exec
    or aborts the whole program if path is not found.

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
                #print('We are using the following executable: %s' % exe_file)
                return exe_file

    #print('No synda executable found on the system. Aborting program.')
    return None

# ---- handling the years for files
def time_handling(year1, year1_model, year2, year2_model):
    """

    year1 - the start year in files
    year1_model - the needed start year of data
    year2 - the last year in files
    year2_model - the needed last year of data
    we compromise on months

    """
    # model interval < data interval / file
    if year1 <= int(year1_model) and year2 >= int(year2_model):
        return True
    # model interval > data interval / file
    elif year1 >= int(year1_model) and year2 <= int(year2_model):
        return True
    elif year1 <= int(year1_model) and year2 <= int(year2_model):
        # data is entirely before model
        if year2 <= int(year1_model):
            return False
        # data overlaps to the left
        elif year2 >= int(year1_model):
            return True
    elif year1 >= int(year1_model) and year2 >= int(year2_model):
        # data is entirely after model
        if year1 >= int(year2_model):
            return False
        # data overlaps to the right
        elif year1 <= int(year2_model):
            return True

# ---- synda search
def synda_search(model_data,varname,server):
    """
    This function performs the database search for files
    It takes exactly three arguments:
    - a model data string of type e.g. 'CMIP5 MPI-ESM-LR Amon amip r1i1p1'
    - a variable name as string e.g. 'tro3'
    - a server name as string e.g.  'esgf-index1.ceda.ac.uk'
    It performs the search for files associated with these parameters and returns ALL
    available files. This information is stored in *Data_Files* files of type e.g.

    Data_Files_CMIP5_MPI-ESM-LR_Amon_amip_r1i1p1_tro3.txt 

    in a directory called allAvailableFiles_SERVER. This info may be needed for later 
    analyses or manual downloads. It is a good data tracking tool as well.

    """
    # this is needed mostly for parallel processes that may
    # go tits-up from time to time due to random path mixes
    if which_synda('synda') is not None:
        pass
    else:
        print >> sys.stderr, "No synda executable found in path. Exiting."
        sys.exit(1)
    dirname = 'allAvailableFiles_' + server.rstrip()
    mkd = 'mkdir -p ' + dirname
    proc = subprocess.Popen(mkd, stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()
    outfiletitle = './' + dirname + '/Data_Files_' + model_data.replace(' ','_') + '_' + varname + '.txt'
    if os.path.isfile(outfiletitle):
        cachefile = open(outfiletitle, 'a')
    else:
        cachefile = open(outfiletitle, 'w')
    synda_search = which_synda('synda') + ' search -f ' + model_data + ' ' + varname
    proc = subprocess.Popen(synda_search, stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()
    if err is not None:
        print >> sys.stderr, "An error has occured while searching for data:"
        print >> sys.stderr, err
        sys.exit(1)
    else:
        cachefile.write(out)
        return out
    cachefile.close()
    cachefile.save()

# ---- synda download
def write_cache(searchoutput,varname,year1_model,year2_model,header,outfile,outfile2):
    """
    This function takes the standard search output from synda
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
            time_range = file_name.split('_')[-1].strip('.nc')
            time1 = time_range.split('-')[0]
            y1 = datetime.strptime(time1, '%Y%m')
            year1 = y1.year
            time2 = time_range.split('-')[1]
            y2 = datetime.strptime(time2, '%Y%m')           
            year2 = y2.year
            if time_handling(year1, year1_model, year2, year2_model) is True:
                file_name_complete = ".".join(file_name.split('.')[:10]) + '.' + varname + '.' + ".".join(file_name.split('.')[10:])
                true_file_name = file_name_complete.split('.')[-2]+'.'+file_name_complete.split('.')[-1]
                print('Matching file: %s' % true_file_name)
                # get the most recent file from database
                # synda will always list the most recent database first
                synda_search = which_synda('synda') + ' search -f -l 1 ' + true_file_name
                proc = subprocess.Popen(synda_search, stdout=subprocess.PIPE, shell=True)
                (out, err) = proc.communicate()
                if len(out.split()) > 2:
                    fc = out.split()[3]
                    file_name_complete_final = ".".join(fc.split('.')[:10]) + '.' + varname + '.' + ".".join(fc.split('.')[10:])
                    filepath_complete = '/badc/cmip5/data/c' + file_name_complete_final.replace('.','/').strip('/nc') + '.nc'
                    print(filepath_complete)
                    # ---- perform a local check file exists in /badc
                    # ---- and write cache
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
                            with open(outfile2, 'a') as file:
                                file.write(header + ' ' + 'ERROR ' + os.strerror(ioex.errno) + ' ' + filepath_complete + '\n')
                                file.close()
                else:
                    print('something went wrong with parsing the data entry, calling this a non-existent file')
                    with open(outfile2, 'a') as file:
                        file.write(header + ' ' + 'ERROR ' + file_name_complete + ' could not be found' + '\n')
                        file.close()
    else:
        print >> sys.stderr, "Could not find database with the specified parameters on BADC"
        return 0

def fix_duplicate_entries(outfile):
    """
    simple fast function to eliminate duplicate entries
    from a cache file
    """
    # ---- fixing the cache file for duplicates
    ar = np.genfromtxt(outfile, dtype=str,delimiter='\n')
    nar = np.unique(ar)
    st(outfile,nar,fmt='%s')

def print_stats(outfile1,outfile2):
    """
    small function to print some stats at the end
    """
    if os.path.exists(outfile1) and os.path.exists(outfile2):
        ar1 = np.genfromtxt(outfile1, dtype=str,delimiter='\n')
        ar2 = np.genfromtxt(outfile2, dtype=str,delimiter='\n')
        pd = [p for p in ar2 if p.split()[2]+p.split()[3]=='Permissiondenied']
        print('\n########################################################')
        print('netcdf_badc_cache: %i individual .nc files cached' % len(ar1))
        print('missing_badc_cache: %i individual .nc files NOT cached:' % len(ar2))
        print('                   - Permission denied: %i files' % len(pd))
        print('                   - Missing files and databases: %i files' % int(int(len(ar2))-int(len(pd))))
        print('########################################################\n')
    elif os.path.exists(outfile1) and os.path.exists(outfile2) is False:
        ar1 = np.genfromtxt(outfile1, dtype=str,delimiter='\n')
        print('\n########################################################')
        print('netcdf_badc_cache: %i individual .nc files cached' % len(ar1))
        print('########################################################\n')
    elif os.path.exists(outfile1) is False:
        print('Shoot! No cache written...something went wrong here!') 

# -------------------------------------------------------------------------
#      Parse the command line options.
# -------------------------------------------------------------------------

# ---- Initialise command line argument variables.
params_file       = None
userVars          = False
fpars             = []
vpars             = []

# ---- Syntax of options, as required by getopt command.
# ---- Short form.
shortop = "hp:g:r:d:i:n:t:f:m:sc:e:"
# ---- Long form.
longop = [
   "help",
   "params-file=",
   "user-input",
   "fileparams=",
   "uservars="
]

# ---- Get command-line arguments.
try:
  opts, args = getopt.getopt(sys.argv[1:], shortop, longop)
except getopt.GetoptError:
  usage()
  sys.exit(1)

# ---- We will record the command line arguments to cache_BADC.py in a file called
#      cache_BADC.param.
#      This file should be used if a further need to run the code arises
command_string = 'cache_BADC.py '
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
    elif o in ("--fileparams"):
        fpars.append(a)
        command_string = command_string + ' --fileparams ' + a
    elif o in ("--uservars"):
        vpars.append(a)
        command_string = command_string + ' --uservars ' + a 
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

# -------------------------------------------------------------------------
#      Status message.  Report all supplied arguments.
# -------------------------------------------------------------------------

print >> sys.stdout
print >> sys.stdout, "####################################################"
print >> sys.stdout, "#                 BADC Data Caching                #"
print >> sys.stdout, "####################################################"
print >> sys.stdout
print >> sys.stdout, "Parsed input arguments:"
print >> sys.stdout
if params_file:
    print >> sys.stdout,"Running with parameters file:", params_file
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

# ---- Get the synda path or exit here
print('Looking up synda executable...')
if which_synda('synda') is not None:
    print >> sys.stdout, "Synda found...OK" 
    print >> sys.stdout, which_synda('synda')
else:
    print >> sys.stderr, "No synda executable found in path. Exiting."
    sys.exit(1)

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
            print('Data server: %s' % data_server.split()[0])
            if data_server.split()[0] != 'esgf-index1.ceda.ac.uk':
                print >> sys.stderr, "Web server not set to ESGF-BADC! Contact your local support, exiting..."
                sys.exit(1)

# ---- Write ASCII file holding cache_BADC.py command.
pfile = open('cache_BADC.param','w')
pfile.write(command_string + "\n")
pfile.close()

# ---- Write cache file ---- #
pfile2 = 'netcdf_badc_cache.txt'
pfile3 = 'missing_netcdf_badc_cache.txt'

if params_file:
    paramfile, paramfile_extension = os.path.splitext(params_file)
    ################################################################
    # ---- xml (namelist)
    # See below for details about reading xml files
    # This is legacy code for ESMVALTool
    # if you need to read xml param files, standardize them, then add
    # the standardized handling here, using the below syntax as model
    ################################################################ 
        ############################ LEGACY CODE ###############################################
        # if paramfile_extension=='.xml':
        # ---- code for XML inout files
        # ---- this works with SOME ESMVal namelists
        # ---- ESMVal migrates from XML files anyway so this is legacy code
        # ---- Parse the namelist xml parameters file for MODEL data ---- #
        # ---- with the specified variable(s) ---- #
        #xmldoc = minidom.parse(params_file)
        #diaglist = xmldoc.getElementsByTagName('diag')
        #print('\n---------------------------------------------')
        #print('We parsed an XML file with diagnostics tests')
        #print('---------------------------------------------')
        #for diag in diaglist:
        #    itemlist = diag.getElementsByTagName('model')
        #    varlist = diag.getElementsByTagName('variable')
        #    v01 = varlist[0].firstChild.nodeValue
        #    # ---- we need the variable to be as compact as possible
        #    v1=v01.split()[0]
        #    lenitemlist = len(itemlist)
        #    print('We need %i MODEL data files for this diagnostic, using %s as variable:' % (lenitemlist,v1))
        #    print('----------------------------------------------------------------------')
        #    # FIXME bolted-on parallelization is easy to implement at this stage
        #    for i in range(0,lenitemlist):
        #        item = itemlist[i].firstChild.nodeValue
        #        model_data = item.split()[0] + ' '+ item.split()[1] + ' ' + item.split()[2]\
        #             + ' ' + item.split()[3] + ' ' + item.split()[4]
        #        print(model_data + '\n')
        #        yr1 = int(item.split()[5])
        #        yr2 = int(item.split()[6])
        #        outpt = synda_search(model_data,v1,data_server)
        #        s = write_cache(outpt,v1,yr1,yr2,pfile2)
        #        if s == 0:
        #            with open(pfile3, 'a') as file:
        #                file.write(item+'\n')
        #                file.close()
        #######################################################################################
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

        """
        ###############################################################
        # ---- prepend the original params file to eliminate
        # duplicates in the lists
        ar = np.genfromtxt(params_file, dtype=str, delimiter='\n')
        nar = np.unique(ar)
        # ---- really hard for genfromtxt to handle one-liner files, shit
        prfile = 'prepended_' + params_file
        if len(nar) == 1:
            with open(prfile, 'a') as file:
                file.write(nar)
        else:
            st(prfile,nar,fmt='%s')
        itemlist = lt(prfile,dtype=str)
        lenitemlist = len(itemlist)
        print('\n---------------------------------------------------------------------')
        print('We parsed a TXT param file. We need data files for %i data bases: ' % lenitemlist)
        print('-----------------------------------------------------------------------')
        # FIXME bolted-on parallelization is easy to implement at this stage
        for item in itemlist:
            v1 = item[7]
            header = item[0] + '_'+ item[1] + '_' + item[2]\
                     + '_' + item[3] + '_' + item[4] + '_' + item[5]\
                     + '_' + item[6] + '_' + item[7]
            model_data = item[0] + ' '+ item[1] + ' ' + item[2]\
                         + ' ' + item[3] + ' ' + item[4]
            print(model_data + '\n')
            yr1 = int(item[5])
            yr2 = int(item[6])
            outpt = synda_search(model_data,v1,data_server)
            s = write_cache(outpt,v1,yr1,yr2,header,pfile2,pfile3)
            if s == 0:
                with open(pfile3, 'a') as file:
                    file.write(header + ' ' + 'ERROR '  + model_data + ' '\
                               + str(yr1) + ' ' + str(yr2) + ' ' + v1\
                               + ' missing database' + '\n')
                    file.close()
        # ---- clean-up
        if os.path.exists(pfile2):
            fix_duplicate_entries(pfile2)
        if os.path.exists(pfile3):
            fix_duplicate_entries(pfile3)
        print_stats(pfile2,pfile3)
        print('DONE\n')
elif userVars:
    # ---- user command line arguments parsed here
    for vi in vpars:
        print('Looking at variable %s' % vi)
        header = fpars[0] + '_'+ fpars[1] + '_' + fpars[2]\
                 + '_' + fpars[3] + '_' + fpars[4] + '_' + str(fpars[5])+ '_' + str(fpars[6])\
                 + '_' + vi
        model_data = fpars[0] + ' '+ fpars[1] + ' ' + fpars[2]\
                 + ' ' + fpars[3] + ' ' + fpars[4]
        print(model_data + '\n')
        yr1 = fpars[5]
        yr2 = fpars[6]
        outpt = synda_search(model_data,vi,data_server)
        s = write_cache(outpt,vi,yr1,yr2,pfile2,pfile3)
        if s == 0:
            with open(pfile3, 'a') as file:
                file.write(header + ' ' + 'ERROR3: ' + model_data + ' ' + str(yr1) + ' ' + str(yr2) + ' ' + v1 + ' missing database' + '\n')
                file.close()
    # ---- clean-up
    if os.path.exists(pfile2):
        fix_duplicate_entries(pfile2)
    if os.path.exists(pfile3):
        fix_duplicate_entries(pfile3)
    print_stats(pfile2,pfile3)
    print('DONE\n')
