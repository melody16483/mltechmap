import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import time
import argparse
from torchviz import make_dot
from lib import *
from preprocess import *
from aig import *
from mlaig import *



def get_model_size(model):
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())  
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())  
    size_mb = (param_size + buffer_size) / (1024 * 1024)  
    return size_mb



def argParseIn():
    parser = argparse.ArgumentParser(description='gradient descent technology mapping')
    parser.add_argument('--liberty_file', type=str, default='./lib/Nangate45_typ')
    parser.add_argument('--aig_file', type=str, default='./aig/ILP_test_output.txt')
    args = parser.parse_args()
    return args


def main(args):
    torch.manual_seed(123)
    t0 = time.time()
    libmgr = libMgr()
    libmgr.ExcelParser(args.liberty_file)
    
    aigmgr = aigMgr()
    aigmgr.cutParser(args.aig_file, libmgr) 
    
    load = aigmgr.outputLoad(libmgr.libs)
    outputADJ, outputWireADJ = aigmgr.outputADJ()
    weightMaskPre, weightMaskPost = aigmgr.weightMask()
    
    # print(outputWireADJ.size())
    # return 0
    
    outputCond = aigmgr.outputCond()
    # print(outputCond)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    # ##circuit build and to device
    choices = choiceNodes(len(aigmgr.sortedAig), aigmgr.maxNode, aigmgr.maxCut, weightMaskPre, weightMaskPost)
    choices.to(device)
    loads = loadNodes(load, len(aigmgr.sortedAig), aigmgr.maxNode, outputWireADJ, outputADJ)
    loads.to(device)
    circuitMgr = mlaigMgr(aigmgr, libmgr.libs, outputCond)
    # size = 0
    # for level in circuitMgr.network:
    #     s = get_model_size(level)
    #     # print(s)
    #     size += s
    # print("circuit network size:%d MB", size + get_model_size(circuitMgr))
    # print("load size:%d MB", get_model_size(loads))
    # print("choices size:%d MB", get_model_size(choices))
    circuitMgr.toCuda(device)
    optimizer = optim.AdamW([choices.weight], lr=0.01)
    
    print(f'init time: {time.time() - t0:.4f}s')
    
    epoch = 1000
    t0 = time.time()
    forwardTime = 0
    backwardTime = 0
    for i in range(epoch):
        tf = time.time()
        weight = choices.forward()
        # if epoch == 0:
            # print(weight)
        load = loads.forward(weight, aigmgr) ##check whether the weighted load is correct
        loss = load.sum()
        # dot = make_dot(load, params={"weight": choices.weight})
        # dot.render("computation_graph"+str(epoch), format="png", view=False)

        arrival = torch.zeros(2, aigmgr.level, aigmgr.maxNode).to(device)
        slew = torch.zeros(2, aigmgr.level, aigmgr.maxNode).to(device)


        circuitMgr.forward(arrival, slew, load, weight)
        poArrival = (arrival*circuitMgr.outputCond)
        loss = poArrival.sum()
        forwardTime += time.time() - tf
        tb = time.time()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        backwardTime += time.time() - tb
        if i % 100 == 0:
            print(loss)
    print(f'gradient descent time: {time.time() - t0:.4f}s')
    print(f'forward time: {forwardTime:.4f}s, backward time: {backwardTime:.4f}s')
    print("weighted arrival time sum:", loss)
    return 0





if __name__ == '__main__':
    args = argParseIn()

    main(args)