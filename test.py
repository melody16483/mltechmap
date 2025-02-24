import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import time
from torchviz import make_dot
from lib import *
from preprocess import *
from aig import *
from mlaig import *

   

if torch.cuda.is_available():
   print('cuda available')
   cuda0 = torch.device('cuda:0')
   

## lib parse in
libmgr = libMgr()
libmgr.ExcelParser('./lib/Nangate45_typ')

##xor example
aigMgr = aigMgr()
input1 = aigMgr.createInputNode()
input2 = aigMgr.createInputNode()
node1 = aigMgr.createAndNode()    
node2 = aigMgr.createAndNode()
node3 = aigMgr.createAndNode()
node1.leftConnect(node2, NEG)
node1.rightConnect(node3, NEG)
node2.leftConnect(input1, NEG)
node2.rightConnect(input2, POS)
node3.leftConnect(input1, POS)
node3.rightConnect(input2, NEG)

##cut add
node1Cut1 = cutNode([[node2, NEG], [node3, NEG]], node1, POS, AND1)
node1Cut2 = cutNode([[node2, NEG], [node3, NEG]], node1, NEG, NAND1)
node1Cut3 = cutNode([[node2, POS], [node3, POS]], node1, NEG, OR1)
node1.cutAdd(node1Cut1, POS)
node1.cutAdd(node1Cut2, NEG)
node1.cutAdd(node1Cut3, NEG)

node2Cut1 = cutNode([[input1, NEG], [input2, POS]], node2, POS, AND1)
node2Cut2 = cutNode([[input1, NEG], [input2, POS]], node2, NEG, NAND1)
node2Cut3 = cutNode([[input1, POS], [input2, NEG]], node2, NEG, OR1)
node2.cutAdd(node2Cut1, POS)
node2.cutAdd(node2Cut2, NEG)
node2.cutAdd(node2Cut3, NEG)

node3Cut1 = cutNode([[input1, POS], [input2, NEG]], node3, POS, AND1)
node3Cut2 = cutNode([[input1, POS], [input2, NEG]], node3, NEG, NAND1)
node3Cut3 = cutNode([[input1, NEG], [input2, POS]], node3, NEG, OR1)
node3.cutAdd(node3Cut1, POS)
node3.cutAdd(node3Cut2, NEG)
node3.cutAdd(node3Cut3, NEG)

##preprocess part
aigMgr.RSConnect(libmgr)
aigMgr.topoSort()
load = aigMgr.outputLoad(libmgr)
outputADJ, outputWireADJ = aigMgr.outputADJ()
weightMaskPre, weightMaskPost = aigMgr.weightMask()



##po required polarity
node1.outputNeg()
aigMgr.PO.append(node1.outputBuf)
outputCond = aigMgr.outputCond()
print(outputCond)

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

##circuit build and to device
choices = choiceNodes(len(aigMgr.sortedAig), aigMgr.maxNode, aigMgr.maxCut, weightMaskPre, weightMaskPost)
choices.to(device)
loads = loadNodes(load, len(aigMgr.sortedAig), aigMgr.maxNode, outputWireADJ, outputADJ)
loads.to(device)
circuitMgr = mlaigMgr(aigMgr, libs, outputCond)
circuitMgr.toCuda(device)
optimizer = optim.SGD([choices.weight], lr=0.01)




epochTotal = 200

print("start")
# print(choices.weight)
t0 = time.time()
for epoch in range(epochTotal):
   
   weight = choices.forward()
   # if epoch == 0:
      # print(weight)
   load = loads.forward(weight, aigMgr) ##check whether the weighted load is correct
   loss = load.sum()
   # dot = make_dot(load, params={"weight": choices.weight})
   # dot.render("computation_graph"+str(epoch), format="png", view=False)

   arrival = torch.zeros(2, aigMgr.level, aigMgr.maxNode).to(device)
   slew = torch.zeros(2, aigMgr.level, aigMgr.maxNode).to(device)


   circuitMgr.forward(arrival, slew, load, weight)
   # dot = make_dot(mlaigMgr.arrival, params={"weight": choices.weight})
   # dot.render("computation_graph"+str(epoch), format="png", view=False)
   poArrival = (arrival*circuitMgr.outputCond)
   loss = poArrival.sum()
   optimizer.zero_grad()
   loss.backward()
   optimizer.step()
   if epoch % 10 == 0:
      print(loss)
   # print(choices.weight)
   # print(f'backward time: {time.time() - t0:.4f}s')

print(f'gradient descent time: {time.time() - t0:.4f}s')
# print("end")
weight = choices.forward()
print(weight)
weight = weight.to("cpu")
aigMgr.showMapResult(weight)

   










