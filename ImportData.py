##
# import Steinmetz data
#@title Data retrieval
import os
import requests

fname = []
for j in range(3):
  fname.append('steinmetz_part%d.npz'%j)
  url = ["https://osf.io/agvxh/download"]
  url.append("https://osf.io/uv3mw/download")
  url.append("https://osf.io/ehmw2/download")

for j in range(len(url)):
  if not os.path.isfile(fname[j]):
    try:
      r = requests.get(url[j])
    except requests.ConnectionError:
      print("!!! Failed to download data !!!")
    else:
      if r.status_code != requests.codes.ok:
        print("!!! Failed to download data !!!")
      else:
        with open(fname[j], "wb") as fid:
          fid.write(r.content)
##
#@title Data loading
import numpy as np

alldat = np.array([])
for j in range(len(fname)):
  alldat = np.hstack((alldat, np.load('steinmetz_part%d.npz'%j, allow_pickle=True)['dat']))

# select just one of the recordings here. 11 is nice because it has some neurons in vis ctx.
dat = alldat[11]
print(dat.keys())

##
tmp=dat["brain_area"]
##

print(np.unique(tmp))
##
unique_areas=dict()

# regions={}
for dn in range(len(alldat)):
  unique_areas[dn]=np.unique(alldat[dn]["brain_area"])
##
wanted_list=dict()
# wanted_list=['CA1', 'VISp', 'ORB', 'VISam', 'MOs', 'MOp', 'PL', 'RSC', 'SSp', 'LGd', 'SCs',  'SUB', 'SNr', 'CA3', 'DG']
wanted_list=['CA1', 'VISp', 'VISam', 'ORB', 'PL', 'RSP', 'MOs', 'MOp', 'SSp']
##
print(len(unique_areas))
##
print(unique_areas[6])
print(wanted_list[7])
##
wanted_list[7] in unique_areas[6]
##
matchMat=np.zeros((len(unique_areas), len(wanted_list)))

##
for dn in range((len(unique_areas))):
  for an in range((len(wanted_list))):
    if wanted_list[an] in unique_areas[dn]:
      matchMat[dn, an] = 1

##
print(unique_areas[11])
print(unique_areas[12])
print(unique_areas[26])
print(unique_areas[38])

##
import pickle
useSes=np.array([11,12, 26, 38])
for sn in useSes:
  print(sn)
  del dd
  dd=alldat[sn]
  fn="ses" + str(sn+1) + ".p"
  print(fn)
  pickle.dump(dd, open(fn, "wb"))


del dd
##
del ss
ss = pickle.load(open("ses13.p", "rb"))

##
print(alldat[12]["spks"].shape)
print(ss["spks"].shape)

##
# save image as PDFs
# from fpdf import FPDF
# pdf = FPDF()
# # imagelist is the list with all image filenames
# for image in imagelist:
#     pdf.add_page()
#     pdf.image(image,x,y,w,h)
# pdf.output("yourfile.pdf", "F")

## Python <-> Matlab
# .mat -> python
# It’s easy enough to load .mat files in Python via the scipy.io.loadmat function.

# [whatever_data]=pickle.load( open( "myallData.p", "rb" ) )
# import numpy, scipy.io
# scipy.io.savemat('/home/myfiles/mydata.mat', mdict={'whatever_data': whatever_data})

# python -> .mat
# function [a] = loadpickle(filename)
#   if ~exist(filename,'file')
#     error('%s is not a file',filename);
#   end
#   outname = [tempname() '.mat'];
#   pyscript = ['import pickle;import sys;import scipy.io;file=open("' filename '", "rb");dat=pickle.load(file);file.close();scipy.io.savemat("' outname '.dat")'];
# system(['python -c "' pyscript '"']);
# a = load(outname);
# end

# alternatively (which didn't work for me)
# In newer versions of Matlab, there’s an even simpler method thanks to the direct Python support in Matlab.
# Assuming you’ve set up Matlab to use the right Python version, you can simply use:
#
# % Filename is the name of the file.
# fid = py.open(filename,'rb');
# data = py.pickle.load(fid);