import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from mltechmap.lib import *


class SparseReshapeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sparse_matrix, new_shape):
        indices = sparse_matrix.indices()
        values = sparse_matrix.values()
        
        ctx.old_shape = sparse_matrix.shape
        ctx.new_shape = new_shape

        old_flat = indices[0]  
        new_indices = torch.empty((len(new_shape), indices.shape[1]), dtype=torch.int64)

        for i, dim in enumerate(reversed(new_shape)):  
            new_indices[-(i+1), :] = old_flat % dim  
            old_flat //= dim  


        return torch.sparse_coo_tensor(new_indices, values, new_shape)

    @staticmethod
    def backward(ctx, grad_output):
        grad_values = grad_output.to_dense().reshape(-1)
        nonzero_indices = torch.nonzero(grad_values, as_tuple=True)  # 找出所有非零索引
        values = grad_values[nonzero_indices]  # 取得非零值
        indices = torch.vstack(nonzero_indices)  # 確保 indices.shape[0] == sparse_dim

        sparse_grad = torch.sparse_coo_tensor(indices, values, grad_values.shape)
        print(indices)
        print(values)
        print(sparse_grad)
        return sparse_grad, None


   
##matchesType represent matches of every generated cut
sparse_reshape = SparseReshapeFunction.apply
class choiceNodes(nn.Module):
   def __init__(self, level, maxNodeNum, maxMatchNum, weightMaskPre, weightMaskPost):
      super(choiceNodes, self).__init__()
      self.weight = nn.Parameter(torch.rand(2, level, maxNodeNum, maxMatchNum, requires_grad=True))
      # self.normWeight = torch.zeros(2, level, maxNodeNum, maxMatchNum)
      self.weightMaskPre = nn.Parameter(weightMaskPre)
      self.weightMaskPost = nn.Parameter(weightMaskPost)
      
      
   def forward(self):
      softmax = nn.Softmax(dim=3)
      
      maskedweight = self.weight*self.weightMaskPre
      infCond = (maskedweight == 0) & (self.weightMaskPost == 1)
      maskedweight = maskedweight.masked_fill(infCond, float('-inf'))
      # maskedweight = maskedweight*self.weightMaskPost
      normWeight = softmax(maskedweight)
      normWeight = normWeight*self.weightMaskPost
      
      
      
      
      
      #return clone choice
      return normWeight.clone()
   
class loadNodes(nn.Module):
   def __init__(self, loads, level, maxNodeNum, wireadj, adj):
      super(loadNodes, self).__init__()
      self.load = nn.Parameter(loads) ##(2, level, maxNodeNum, maxFanoutNum)
      # self.weightLoad = nn.Parameter(torch.zeros(2, level, maxNodeNum)) ## output of the level
      # self.fanoutLib = fanoutLib ## (2, level, maxNodeNum, maxFanoutNum, 2(lib, input index))
      self.fanoutConnect = nn.Parameter(adj.coalesce()) ##(2, level, maxNodeNum, maxFanoutNum, 2, level, maxNode, match)
      self.wireConnect = nn.Parameter(wireadj.coalesce())##(2, level, maxNodeNum, maxFanoutNum, 2, level, maxNode)

      
   def forward(self, weight, aigMgr):
      # expanded_weight_matrix = weight.unsqueeze(0)
      # expanded_weight_matrix =  expanded_weight_matrix.repeat(2*len(aigMgr.sortedAig)*aigMgr.maxNode*aigMgr.maxFanout, 1, 1, 1, 1)

      # o = torch.sparse_coo_tensor(
      #    self.fanoutConnect.indices(), 
      #    self.fanoutConnect.values() * expanded_weight_matrix[self.fanoutConnect.indices()[0], self.fanoutConnect.indices()[1], self.fanoutConnect.indices()[2], self.fanoutConnect.indices()[3], self.fanoutConnect.indices()[4]], 
      #    self.fanoutConnect.shape
      # )
      o = torch.sparse_coo_tensor(
         self.fanoutConnect.indices(), 
         self.fanoutConnect.values() * weight[self.fanoutConnect.indices()[1], self.fanoutConnect.indices()[2], self.fanoutConnect.indices()[3], self.fanoutConnect.indices()[4]], 
         self.fanoutConnect.shape
      )
      o = o.to_dense()
      o = o.sum(dim=(1, 2, 3, 4)) ##(2, level, maxNodeNum, maxFanoutNum)
      # print(o.size())
      # print(self.load)
      # print(o)
      


      o = o.view(2, len(aigMgr.sortedAig), aigMgr.maxNode, aigMgr.maxFanout) # not contain wire output

      weightLoad = (self.load * o).sum(dim = -1)##(2, level, maxNodeNum)
      # print(self.weightLoad)
      # self.weightLoad = self.weightLoad.sum(dim = -1)
      # print(self.weightLoad)
      LoadWire =  torch.sparse_coo_tensor(
         self.wireConnect.indices(), 
         self.wireConnect.values() * weightLoad[self.wireConnect.indices()[1], self.wireConnect.indices()[2], self.wireConnect.indices()[3]], 
         self.wireConnect.shape
      )
      LoadWire = LoadWire.to_dense()##(2, level, node, maxFanout)
      LoadWire = LoadWire.sum(dim=(1, 2, 3))
      LoadWire = LoadWire.view(2, len(aigMgr.sortedAig), aigMgr.maxNode, aigMgr.maxFanout)
      LoadWire = (LoadWire * o).sum(dim = -1)
      # print(LoadWire)
      # LoadWire = (LoadWire * o).sum(dim = -1)
      weightLoad = weightLoad + LoadWire
      # print(self.weightLoad.size(), self.weightLoad.is_contiguous())
      # print(self.weightLoad)
      # self.weightLoad.contiguous()
      


      return weightLoad ##(2, level, maxNodeNum)
         
      