import torch
import torch.nn as nn
import numpy as np




##xor gate
##test library 0:wire, 1:buffer, 2:inv1, 3:inv2, 4:and, 5:nand, 6:or 
##test 
##input signal
input_signal = torch.tensor([[0, 0] ,[-1, -1]])

##input layer is special one, all matches input should come from pos
input_nodeNum = 2
input_matchSize = torch.tensor([[2, 2], [2, 2]]) ##four input and each has two pos match and two neg match
input_matchesType = torch.tensor([[[0, 1], [0, 1]], [[2, 3], [2, 3]]])
input_matchesInput = torch.tensor([[[[[0], [0], [0], [0], [0]], [[0], [0], [0], [0], [0]]], 
                                   [[[0], [0], [0], [0], [0]], [[0], [0], [0], [0], [0]]]], 
                                  [[[[0], [0], [0], [0], [0]], [[0], [0], [0], [0], [0]]], 
                                   [[[0], [0], [0], [0], [0]], [[0], [0], [0], [0], [0]]]]])
input_regular = False

input_level = aigNodeLevel(input_nodeNum, input_matchSize, input_matchesType, input_matchesInput, input_regular)
input_level.checkShape()

level1r_nodeNum = 2
level1r_matchSize = torch.tensor([[1, 1], [2, 2]])
level1r_matchType = torch.tensor([[[4, -1], [4, -1]], [[5, 6], [5, 6]]])
level1r_matchesInput = torch.tensor([[[[[0, 0, 0], [1, 0, 1], [0, -1, -1], [0, -1, -1], [0, -1, -1]], [[0, -1, -1], [0, 0, 1], [0, -1, -1], [0, -1, -1], [0, -1, -1]]], 
                                   [[[1, 0, 0], [0, 0, 1], [0, -1, -1], [0, -1, -1], [0, -1, -1]], [[0, -1, -1], [0, 0, 1], [0, -1, -1], [0, -1, -1], [0, -1, -1]]]], 
                                  [[[[0, 0, 0], [1, 0, 1], [0, -1, -1], [0, -1, -1], [0, -1, -1]], [[1, 0, 0], [0, 0, 1], [0, -1, -1], [0, -1, -1], [0, -1, -1]]], 
                                   [[[1, 0, 0], [0, 0, 1], [0, -1, -1], [0, -1, -1], [0, -1, -1]], [[0, 0, 0], [1, 0, 1], [0, -1, -1], [0, -1, -1], [0, -1, -1]]]]])
level1r_regular = True
level1r = aigNodeLevel(level1r_nodeNum, level1r_matchSize, level1r_matchType, level1r_matchesInput, level1r_regular)
level1r.checkShape()

level1s_nodeNum = 2
level1s_matchSize = torch.tensor([[4, 4], [4, 4]])
level1s_matchType = torch.tensor([[[0, 1, 2, 3], [0, 1, 2, 3]], [[0, 1, 2, 3], [0, 1, 2, 3]]])
level1s_matchesInput = torch.tensor([[[[[0], [0], [0], [0], [0]], 
                                       [[0], [0], [0], [0], [0]], 
                                       [[1], [0], [0], [0], [0]], 
                                       [[1], [0], [0], [0], [0]]], 
                                      [[[0], [0], [0], [0], [0]], 
                                       [[0], [0], [0], [0], [0]], 
                                       [[1], [0], [0], [0], [0]], 
                                       [[1], [0], [0], [0], [0]]]], 
                                     [[[[1], [0], [0], [0], [0]], 
                                       [[1], [0], [0], [0], [0]], 
                                       [[0], [0], [0], [0], [0]], 
                                       [[0], [0], [0], [0], [0]]], 
                                      [[[1], [0], [0], [0], [0]], 
                                       [[1], [0], [0], [0], [0]], 
                                       [[0], [0], [0], [0], [0]], 
                                       [[0], [0], [0], [0], [0]]]]])
level1s_regular = False
level1s = aigNodeLevel(level1s_nodeNum, level1s_matchSize, level1s_matchType, level1s_matchesInput, level1s_regular)
level1s.checkShape()

level2r_nodeNum = 1
level2r_matchSize = torch.tensor([[1], [2]])
level2r_matchType = torch.tensor([[[4, -1]], [[5, 6]]])
level2r_matchesInput = torch.tensor([[[[[1, 1, 0], [1, 1, 1], [0, -1, -1], [0, -1, -1], [0, -1, -1]], [[0, -1, -1], [0, -1, -1], [0, -1, -1], [0, -1, -1], [0, -1, -1]]]], 
                                  [[[[1, 1, 0], [1, 1, 1], [0, -1, -1], [0, -1, -1], [0, -1, -1]], [[0, 1, 0], [0, 1, 1], [0, -1, -1], [0, -1, -1], [0, -1, -1]]]]])
level2r_regular = True
level2r = aigNodeLevel(level2r_nodeNum, level2r_matchSize, level2r_matchType, level2r_matchesInput, level2r_regular)
level2r.checkShape()

##notice that the last singular layer should mark the wanted po polarity --> only record the wanted polarity
level2s_nodeNum = 1
level2s_matchSize = torch.tensor([[0], [4]]) ## in this case only need neg
level2s_matchType = torch.tensor([[[-1, -1, -1, -1]], [[0, 1, 2, 3]]])
level2s_matchesInput = level1s_matchesInput = torch.tensor([[[[[0], [0], [0], [0], [0]],
                                                                  [[0], [0], [0], [0], [0]],
                                                                  [[0], [0], [0], [0], [0]], 
                                                                  [[0], [0], [0], [0], [0]]]],
                                                            [[[[1], [0], [0], [0], [0]],
                                                                  [[1], [0], [0], [0], [0]],
                                                                  [[0], [0], [0], [0], [0]], 
                                                                  [[0], [0], [0], [0], [0]]]]])
level2s_regular = False
level2s = aigNodeLevel(level2s_nodeNum, level2s_matchSize, level2s_matchType, level2s_matchesInput, level2s_regular)
level2s.checkShape()


aigr = [level1r, level2r]
aigs = [input_level, level1s, level2s]

