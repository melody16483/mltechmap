import torch
import torch.nn as nn
from aig import *

class mlaigMgr(nn.Module):
    def __init__(self, aigMgr, libs, outputCond):
        super(mlaigMgr, self).__init__()
        # self.arrival = nn.Parameter(torch.zeros(2, aigMgr.level, aigMgr.maxNode), requires_grad=False)
        # self.slew = nn.Parameter(torch.zeros(2, aigMgr.level, aigMgr.maxNode), requires_grad=False)
        self.network = []
        self.outputCond = nn.Parameter(outputCond, requires_grad=False) #2, level, maxNode
        self.maxNode = aigMgr.maxNode
        self.maxLevel = aigMgr.level
        self.maxCut = aigMgr.maxCut
        for i in range(aigMgr.level):
            if i == 0:
                continue
            level = aigMgr.sortedAig[i]
            nodes = mlaigNodeLevel(self.maxNode, self.maxCut, self.maxLevel, level, False, libs)
            self.network.append(nodes)
        
    def forward(self, arrival, slew, loads, choices, verbose = False):
        ## store loads and choices of all level
        ##load (2, level, maxNode)
        ##choices (2, level, maxNode, maxMatch)
        for i in range(len(self.network)):
            nodes = self.network[i]
            load = loads[:, i+1, :]
            choice = choices[:, i+1, :]
            load = load.unsqueeze(-1).unsqueeze(-1)
            load = load.repeat(1, 1, self.maxCut, 5)
            
            arrival, slew = nodes.forward(load, choice, arrival, slew, i+1, self.maxNode, self.maxCut, verbose)
        return 0
    
    # def backward(self, optimizer, epoch):
    #     # print(self.arrival)
    #     poArrival = (self.arrival*self.outputCond)
    #     loss = poArrival.sum()
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #     self.arrival = torch.zeros(2, self.maxLevel, self.maxNode)
    #     self.slew = torch.zeros(2, self.maxLevel, self.maxNode)
    #     if epoch % 5 == 0:
    #         print("loss: ", loss)
    #         # print(loss)
    #     return 0
    
    def toCuda(self, device):
        for level in self.network:
            level.to(device)
        self.to(device)
        return 0

class mlaigNodeLevel(nn.Module):
   def __init__(self, nodeNum, matchSize, level, nodes, regular, libs):
    super(mlaigNodeLevel, self).__init__()
    # self.load = nn.Parameter(torch.zeros(2, nodeNum, matchSize, 5))##from input
    # self.choice = nn.Parameter(torch.zeros(2, nodeNum, matchSize))##from input
    self.loadSCoeff = nn.Parameter(torch.zeros(2, nodeNum, matchSize, 5), requires_grad=False)
    self.slewSCoeff = nn.Parameter(torch.zeros(2, nodeNum, matchSize, 5), requires_grad=False) 
    self.bodySCoeff = nn.Parameter(torch.zeros(2, nodeNum, matchSize, 5), requires_grad=False)
    self.loadACoeff = nn.Parameter(torch.zeros(2, nodeNum, matchSize, 5), requires_grad=False)
    self.slewACoeff = nn.Parameter(torch.zeros(2, nodeNum, matchSize, 5), requires_grad=False) 
    self.bodyACoeff = nn.Parameter(torch.zeros(2, nodeNum, matchSize, 5), requires_grad=False)
    #adj matrix (2, maxNode, maxMatch, 5, 2, level, maxNode) -> (2*maxNode*maxMatch*5, 2, level, maxNode)
    indices = torch.tensor([])
    values = torch.tensor([])
    for i in range(len(nodes)):
        node = nodes[i]
        for j in range(len(node.cutPos)):
            lib = libs[node.cutPos[j].lib]
            inputNum = lib.inputNum
            self.loadSCoeff[0, i, j, 0:inputNum] = lib.loadSCoeff
            self.slewSCoeff[0, i, j, 0:inputNum] = lib.slewSCoeff
            self.bodySCoeff[0, i, j, 0:inputNum] = lib.bodySCoeff
            self.loadACoeff[0, i, j, 0:inputNum] = lib.loadACoeff
            self.slewACoeff[0, i, j, 0:inputNum] = lib.slewACoeff
            self.bodyACoeff[0, i, j, 0:inputNum] = lib.bodyACoeff
            if node.level == 0:
                continue
            for k in range(len(node.cutPos[j].leaves)):
                if node.cutPos[j].leaves[k] == 0:
                    continue
                leaf = node.cutPos[j].leaves[k][0]
                pol = node.cutPos[j].leaves[k][1]
                if leaf.level == 0:
                    continue
                tempi = torch.tensor([
                    [5*matchSize*i + 5*j + k], [pol], [leaf.topoIndex[0]], [leaf.topoIndex[1]]
                ], dtype=torch.int64)
                # tempi = torch.tensor([
                #     [0], [i], [j], [k], [pol], [leaf.topoIndex[0]], [leaf.topoIndex[1]]
                # ], dtype=torch.int64)
                tempv = torch.tensor([1.0], dtype=torch.float32) 
                if indices.shape[0] == 0:
                    indices = tempi
                    values = tempv
                else:
                    indices = torch.cat([indices, tempi], dim=1)
                    values = torch.cat([values, tempv])
        for j in range(len(node.cutNeg)):
            lib = libs[node.cutNeg[j].lib]
            inputNum = lib.inputNum
            self.loadSCoeff[1, i, j, 0:inputNum] = lib.loadSCoeff
            self.slewSCoeff[1, i, j, 0:inputNum] = lib.slewSCoeff
            self.bodySCoeff[1, i, j, 0:inputNum] = lib.bodySCoeff
            self.loadACoeff[1, i, j, 0:inputNum] = lib.loadACoeff
            self.slewACoeff[1, i, j, 0:inputNum] = lib.slewACoeff
            self.bodyACoeff[1, i, j, 0:inputNum] = lib.bodyACoeff
            if node.level == 0:
                continue
            for k in range(len(node.cutNeg[j].leaves)):
                leaf = node.cutNeg[j].leaves[k][0]
                pol = node.cutNeg[j].leaves[k][1]
                tempi = torch.tensor([
                    [5*matchSize*nodeNum + 5*matchSize*i + 5*j + k], [pol], [leaf.topoIndex[0]], [leaf.topoIndex[1]]
                ], dtype=torch.int64)
                # tempi = torch.tensor([
                #     [1], [i], [j], [k], [pol], [leaf.topoIndex[0]], [leaf.topoIndex[1]]
                # ], dtype=torch.int64)
                tempv = torch.tensor([1.0], dtype=torch.float32) 
                if indices.shape[0] == 0:
                    indices = tempi
                    values = tempv
                else:
                    indices = torch.cat([indices, tempi], dim=1)
                    values = torch.cat([values, tempv])             
    self.ADJ = nn.Parameter(torch.sparse_coo_tensor(indices, values, (2*nodeNum*matchSize*5, 2, level, nodeNum)).coalesce(), requires_grad=False)
    # self.ADJ = self.ADJ.to_dense()## might crash --> reshape in sparse
    # self.ADJ = self.ADJ.view((2*nodeNum*matchSize*5, 2, level, nodeNum))
    # self.ADJ = self.ADJ.to_sparse()
    
    # print(self.ADJ)
            
    ## construct adj
    # self.adj = adj #use to represent connect
    #load * loadCoeff + arrival + slew*slewCoeff + constant ==> time arc arrival time
   
      
      

   def forward(self, load, choice, arrival, slew, currentLevel, maxNode, maxCut, verbose = False): ##prev is prev arrival time of each nodes
      arrivalArrangedT = torch.sparse_coo_tensor(self.ADJ.indices(), arrival[self.ADJ.indices()[1], self.ADJ.indices()[2], self.ADJ.indices()[3]], self.ADJ.shape)
      arrivalArrangedT = arrivalArrangedT.to_dense() ## (2*nodenum*match*5)
      arrivalArrangedT = arrivalArrangedT.sum(dim=(1, 2, 3))
      arrivalArranged = arrivalArrangedT.view((2, maxNode, maxCut, 5))
    #   arrivalArranged = arrivalArrangedT.reshape(2, maxNode, maxCut, 5).clone()
      slewArrangedT = torch.sparse_coo_tensor(self.ADJ.indices(), slew[self.ADJ.indices()[1], self.ADJ.indices()[2], self.ADJ.indices()[3]], self.ADJ.shape)
      slewArrangedT = slewArrangedT.to_dense() 
      slewArrangedT = slewArrangedT.sum(dim=(1, 2, 3))
      slewArranged = slewArrangedT.view((2, maxNode, maxCut, 5))
    #   slewArranged = slewArrangedT.reshape(2, maxNode, maxCut, 5).clone()
      
      if verbose:
        print(arrival)
        print(self.ADJ)
        print(arrivalArranged)
      
      
      
      pinArrival = arrivalArranged + slewArranged*self.slewACoeff+load*self.loadACoeff+self.bodyACoeff ## (2, nodenum, match, 5)
      pinSlew = slewArranged*self.slewSCoeff+load*self.loadSCoeff+self.bodySCoeff ## (2, nodenum, match, 5)
      newArrival = torch.logsumexp(pinArrival, dim=-1) ## (2, nodenum, match)
      newSlew = torch.logsumexp(pinSlew, dim=-1) ## (2, nodenum, match)
      
      ##weighted arrival and slew
      if verbose:
        print(pinArrival)
        print(newArrival)
        print(choice)
        print("\n")

      newArrival = (newArrival*choice).sum(dim = -1)
      newSlew = (newSlew*choice).sum(dim = -1)
      arrival[:, currentLevel, :] = newArrival
      slew[:, currentLevel, :] = newSlew
      ##return prev+current aig node
      return arrival, slew
    

   
      

