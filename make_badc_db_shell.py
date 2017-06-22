import sys, os, shutil, re, string, popen2, errno
import subprocess
from datetime import datetime
import time

"""
Tool to create a shell script that automates
the finding of netcdf files on /badc eg

find /badc/cmip5/data/cmip5/output1/BCC/bcc-csm1-1 -follow -type f -iname "*.nc" > all_badc_netcdf_CMIP5_bcc-csm1-1.txt
"""


sc = open('all_badc_netcdf_CMIP5.sh','w')
# capture the ls output
dirname1 = '/badc/cmip5/data/cmip5/output1/'
lsd1 = 'ls -la ' + dirname1
proc1 = subprocess.Popen(lsd1, stdout=subprocess.PIPE, shell=True)
(out1, err1) = proc1.communicate()
for st in out1.split('\n')[3:-1]:
    sc.write('# ' + st + '\n')
    subdir = st.split()[-1]
    lsd2 = 'ls -la ' + dirname1 + subdir
    proc2 = subprocess.Popen(lsd2, stdout=subprocess.PIPE, shell=True)
    (out2, err2) = proc2.communicate()
    # skip the Permission denied dirs
    if len(out2) > 0:
        for st2 in out2.split('\n')[3:-1]:
            findic = st2.split()[-1]
            # write the find command to file
            strfindic = 'find /badc/cmip5/data/cmip5/output1/' + subdir + '/' + findic\
                         +' -follow -type f -iname "*.nc" > all_badc_netcdf_CMIP5_' + findic + '.txt'
            sc.write('# ' + '/badc/cmip5/data/cmip5/output1/' + subdir + '/' + findic + '\n')
            sc.write(strfindic + '\n')
            sc.write('\n')
# ---- done

