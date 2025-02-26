import torch
from lib import *
from aig import *

libmgr = libMgr()

# libmgr.libParse1('./lib/Nangate45_typ.lib')

libmgr.ExcelParser('./lib/Nangate45_typ')

libmgr.libInfo()


# aigmgr = aigMgr()
# aigmgr.cutParser('./aig/uu.txt', libmgr)
# print(aigmgr.level, aigmgr.maxNode, aigmgr.maxCut, aigmgr.maxFanout)
# load = aigmgr.outputLoad(libmgr.libs)
# outputADJ, outputWireADJ = aigmgr.outputADJ()
# weightMaskPre, weightMaskPost = aigmgr.weightMask()
# print(load)