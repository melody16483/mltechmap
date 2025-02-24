import torch
from mltechmap.lib import *

libmgr = libMgr()

# libmgr.libParse1('./lib/Nangate45_typ.lib')

libmgr.ExcelParser('./lib/Nangate45_typ')

libmgr.singleLibInfo()


