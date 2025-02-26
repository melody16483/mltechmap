import torch
import numpy as np
from lib import *

class aigMgr:
    def __init__(self):
        self.aig = []
        self.sortedAig = []
        self.PI = [] ##PI sorted
        self.PO = [] ##PO sorted(must be singular node)
        self.maxFanout = -1
        self.maxNode = -1
        self.maxCut = -1
        self.level = -1
    
    def createAndNode(self):
        andTemp = andNode(len(self.aig))
        self.aig.append(andTemp)
        self.aig.append(andTemp.outputBuf)
        return andTemp
    
    def createInputNode(self):
        inputTemp = inputNode(len(self.aig))
        self.aig.append(inputTemp)
        self.aig.append(inputTemp.outputBuf)
        self.PI.append(inputTemp)
        return inputTemp
        
    def topoSort(self, verbose = False):
        for i in range(len(self.PI)):
            self.PI[i].topoIndex = (0, i)
        queue = []
        self.sortedAig.append([])
        for pi in self.PI:
            queue.append(pi)
            self.sortedAig[0].append(pi)
            if len(pi.loadCutPos) > self.maxFanout:
                self.maxFanout = len(pi.loadCutPos)
            if len(pi.loadCutNeg) > self.maxFanout:
                self.maxFanout = len(pi.loadCutNeg)
        
        ##sorting 
        while len(queue) != 0:
            node = queue.pop(0)
            if type(node) == inputNode:
                queue.append(node.outputBuf)    
            elif type(node) == andNode:
                if node.level != -1:##already level
                    continue
                else:##check whether child level
                    if(node.left.level != -1 and node.right.level != -1):
                        node.level = max(node.left.level, node.right.level) + 1
                        queue.append(node.outputBuf)
                        while len(self.sortedAig) <= node.level:
                            self.sortedAig.append([])
                        node.topoIndex = (node.level, len(self.sortedAig[node.level]))
                        self.sortedAig[node.level].append(node)
                        self.maxFanout = max(self.maxFanout, len(node.loadCutPos), len(node.loadCutNeg))
                    else:
                        queue.append(node)
            elif type(node) == bufNode:
                if node.level != -1:
                    continue
                else:
                    if node.input.level == -1:
                        print("something wrong")
                        return 1
                    else:
                        node.level = node.input.level +1
                        while len(self.sortedAig) <= node.level:
                            self.sortedAig.append([])
                        # print(self.sortedAig, node.level)
                        node.topoIndex = (node.level, len(self.sortedAig[node.level]))
                        self.sortedAig[node.level].append(node)
                        self.maxFanout = max(self.maxFanout, len(node.loadCutPos), len(node.loadCutNeg))
                        for output in node.outputNodes:
                            queue.append(output)
        
        #feature record            
        for i in range(len(self.sortedAig)):
            level = self.sortedAig[i]
            if len(level) > self.maxNode:
                self.maxNode = len(level)
            if i == 0:
                continue
            for node in level:
                self.maxCut = max(self.maxCut, len(node.cutPos), len(node.cutNeg))
                for i in range(len(node.cutPos)):
                    cut = node.cutPos[i]
                    cut.cutIndex = i
                for i in range(len(node.cutNeg)):
                    cut = node.cutNeg[i]
                    cut.cutIndex = i
        self.level = len(self.sortedAig)
        if verbose:
            print("maxnode: ", self.maxNode)
            print("maxfanout: ", self.maxFanout)
            print("level: ", self.level)
        return 0
    
    def RSConnect(self, libs):
        for node in self.aig:
            if type(node) == andNode or type(node) == inputNode:
                node.RScutAdd(libs)
        return 0
    
    def outputLoad(self, libs, verbose = False):
        arr = np.zeros((2, self.level, self.maxNode, self.maxFanout))
        for i in range(self.level): ##multi thread?
            level = self.sortedAig[i]
            for j in range(len(level)):
                node = level[j]
                if verbose:
                    print("node %d:"%node.index)
                    print("pos")
                for k in range(len(node.loadCutPos)):
                    posOutCutLib = libs[node.loadCutPos[k][0].lib]
                    inputIndex = node.loadCutPos[k][1]
                    if verbose:
                        print(posOutCutLib['cellName'], inputIndex)
                    arr[0][i][j][k] = posOutCutLib['pins'][inputIndex]['load']
                if verbose:
                    print(arr[0][i][j])
                    print("neg")
                for k in range(len(node.loadCutNeg)):
                    negOutCutLib = libs[node.loadCutNeg[k][0].lib]
                    inputIndex = node.loadCutNeg[k][1]
                    if verbose:
                        print(negOutCutLib['cellName'], inputIndex)
                    arr[1][i][j][k] = negOutCutLib['pins'][inputIndex]['load']   
                if verbose:
                    print(arr[1][i][j])
                    print("\n")
        return torch.tensor(arr)
    
    def outputCond(self):
        outputCond = torch.zeros((2, self.level, self.maxNode))
        for node in self.aig:
            if type(node) == andNode or type(node) == inputNode:
                continue
            if node.outputPolarity == POS:
                outputCond[0, node.topoIndex[0], node.topoIndex[1]] = 1
            if node.outputPolarity == NEG:
                outputCond[1, node.topoIndex[0], node.topoIndex[1]] = 1
        return outputCond
                
    def outputADJ(self):
        indices = torch.empty((5, 0), dtype=torch.int64) 
        values = torch.empty((0), dtype=torch.float32) 
        indicesWire = torch.empty((4, 0), dtype=torch.int64) ##(2*level*node*cut, 2, level, node)
        valuesWire = torch.empty((0), dtype=torch.float32)
        for i in range(self.level):
            level = self.sortedAig[i]
            for j in range(len(level)):
                node = level[j]
                for k in range(len(node.loadCutPos)):
                    outputCut = node.loadCutPos[k][0]
                    ## weight of k th fanout load of pos node at level i index j is weight[outputCut.outputPolarity, level, index, cut index]
                    tempi = torch.tensor([
                        [0 + i*self.maxNode*self.maxFanout + j*self.maxFanout + k], [outputCut.outputPolarity], [outputCut.output.topoIndex[0]], [outputCut.output.topoIndex[1]], [outputCut.cutIndex]
                    ], dtype=torch.int64)
                    # print(outputCut.outputPolarity, outputCut.output.topoIndex[0], outputCut.output.topoIndex[1], outputCut.cutIndex)
                    # tempi = torch.tensor([
                    #     [0], [i], [j], [k], [outputCut.outputPolarity], [outputCut.output.topoIndex[0]], [outputCut.output.topoIndex[1]], [outputCut.cutIndex]
                    # ], dtype=torch.int64)
                    tempv = torch.tensor([1.0], dtype=torch.float32)  
                    if indices.shape[1] == 0:
                        indices = tempi
                        values = tempv
                    else:
                        indices = torch.cat([indices, tempi], dim=1)
                        values = torch.cat([values, tempv])
                    
                    if outputCut.lib == WIRE:
                        ## wire load of k th fanout load of pos node at level i index j is load[0, level, index]
                        tempi = torch.tensor([
                            [0 + i*self.maxNode*self.maxFanout + j*self.maxFanout + k], [0], [outputCut.output.topoIndex[0]], [outputCut.output.topoIndex[1]]
                        ])
                        # tempi = torch.tensor([
                        #     [0], [i], [j], [k], [0], [outputCut.output.topoIndex[0]], [outputCut.output.topoIndex[1]]
                        # ])
                        tempv = torch.tensor([1.0], dtype=torch.float32)  
                        if indicesWire.shape[1] == 0:
                            indicesWire = tempi
                            valuesWire = tempv
                        else:
                            indicesWire = torch.cat([indicesWire, tempi], dim=1)
                            valuesWire = torch.cat([valuesWire, tempv])

                for k in range(len(node.loadCutNeg)):
                    outputCut = node.loadCutNeg[k][0]
                    tempi = torch.tensor([
                        [self.level*self.maxNode*self.maxFanout + i*self.maxNode*self.maxFanout + j*self.maxFanout + k], [outputCut.outputPolarity], [outputCut.output.topoIndex[0]], [outputCut.output.topoIndex[1]], [outputCut.cutIndex]
                    ], dtype=torch.int64)
                    # print(outputCut.cutIndex)
                    # tempi = torch.tensor([
                    #     [1], [i], [j], [k], [outputCut.outputPolarity], [outputCut.output.topoIndex[0]], [outputCut.output.topoIndex[1]], [outputCut.cutIndex]
                    # ], dtype=torch.int64)
                    tempv = torch.tensor([1.0], dtype=torch.float32) 
                    if indices.shape[1] == 0:
                        indices = tempi
                        values = tempv
                    else:
                        indices = torch.cat([indices, tempi], dim=1)
                        values = torch.cat([values, tempv])
                    if outputCut.lib == WIRE:
                    ## wire load of k th fanout load of pos node at level i index j is load[0, level, index]
                        tempi = torch.tensor([
                            [self.level*self.maxNode*self.maxFanout + i*self.maxNode*self.maxFanout + j*self.maxFanout + k], [1], [outputCut.output.topoIndex[0]], [outputCut.output.topoIndex[1]]
                        ])
                        # tempi = torch.tensor([
                        #     [1], [i], [j], [k], [1], [outputCut.output.topoIndex[0]], [outputCut.output.topoIndex[1]]
                        # ])
                        tempv = torch.tensor([1.0], dtype=torch.float32)  
                        if indicesWire.shape[1] == 0:
                            indicesWire = tempi
                            valuesWire = tempv
                        else:
                            indicesWire = torch.cat([indicesWire, tempi], dim=1)
                            valuesWire = torch.cat([valuesWire, tempv])

        # print(indices)
        outputADJ = torch.sparse_coo_tensor(indices, values, (2*self.level*self.maxNode*self.maxFanout, 2, self.level, self.maxNode, self.maxCut))
        # outputADJ = torch.sparse_coo_tensor(indices, values, (2, self.level, self.maxNode, self.maxFanout, 2, self.level, self.maxNode, self.maxCut))
        # # print(sparseMatrix)
        # outputADJ = outputADJ.to_dense()
        # outputADJ = outputADJ.view(2*self.level*self.maxNode*self.maxFanout, 2, self.level, self.maxNode, self.maxCut)
        # outputADJ = outputADJ.to_sparse()
        outputWireADJ = torch.sparse_coo_tensor(indicesWire, valuesWire, (2*self.level*self.maxNode*self.maxFanout, 2, self.level, self.maxNode))
        # outputWireADJ = torch.sparse_coo_tensor(indicesWire, valuesWire, (2, self.level, self.maxNode, self.maxFanout, 2, self.level, self.maxNode))
        # # print(sparseMatrix)
        # outputWireADJ = outputWireADJ.to_dense()
        # outputWireADJ = outputWireADJ.view(2*self.level*self.maxNode*self.maxFanout, 2, self.level, self.maxNode)
        # outputWireADJ = outputWireADJ.to_sparse()

        return outputADJ, outputWireADJ

    def weightMask(self):
        weightMaskPre = torch.zeros((2, self.level, self.maxNode, self.maxCut))
        for node in self.aig:
            if node.level == 0:
                continue
            weightMaskPre[0, node.topoIndex[0], node.topoIndex[1], 0:len(node.cutPos)] = 1
            weightMaskPre[1, node.topoIndex[0], node.topoIndex[1], 0:len(node.cutNeg)] = 1
        weightMaskPost = torch.zeros((2, self.level, self.maxNode, self.maxCut))
        for i in range(len(self.sortedAig)):
            if i == 0:
                continue
            weightMaskPost[:, i, 0:len(self.sortedAig[i]), :] = 1
            
        return weightMaskPre, weightMaskPost

    def showMapResult(self, weight):
        ##weight = (2, level, maxnode, match)
        techmap = dict() ##-->store best choice for each node
        techmapInput = dict()
        for node in self.aig:
            if node.level == 0:
                continue
            
            w = weight[0, node.topoIndex[0], node.topoIndex[1], :]
            best = np.argmax(w.detach().numpy())
            techmap[(node.index, POS)] = node.cutPos[best].lib
            techmapInput[(node.index, POS)] = node.cutPos[best].leaves
            
            w = weight[1, node.topoIndex[0], node.topoIndex[1], :]
            best = np.argmax(w.detach().numpy())
            techmap[(node.index, NEG)] = node.cutNeg[best].lib
            techmapInput[(node.index, NEG)] = node.cutNeg[best].leaves
            
        print(techmap)
        queue = []
        for node in self.PO:
            queue.append(node)
        while len(queue) != 0:
            node = queue.pop(0)
            if type(node) == inputNode:
                continue
            if node.outputPolarity == POS:
                print(node.index, " : ",techmap[(node.index, POS)])
                # for n in techmapInput[(node.index, POS)]:
                    # queue.append(n)
            if node.outputPolarity == NEG:
                print(node.index, " : ",techmap[(node.index, NEG)])


    def cutParser(self, fileName, libmgr):
        f = open(fileName, "r")
        lines = f.readlines()
        piNum = int(lines[0].split("\n")[0])
        andNum = int(lines[1].split("\n")[0])
        # print("piNum: %d, andNum: %d"%(piNum, andNum))
        for i in range(piNum):
            self.createInputNode()
        for i in range(andNum):
            self.createAndNode()
        i = 2
        id = -1
        nMatch = -1
        pol = NONE
        # print(len(lines))
        while i <  len(lines):
            line = lines[i]
            if line.startswith("nodeID"):
                lineSeg = (line.split("\n")[0]).split(", ")
                level = int(lineSeg[1].split(":")[1])
                nMatch = int(lineSeg[2].split(":")[1])
                id = 2*int(lineSeg[0].split(":")[1])
                node = self.aig[id]
                assert(type(node) == andNode or type(node) == inputNode)
                node.level = 2*level
                node.outputBuf.level = 2*level+1
                if len(lineSeg) == 5:
                    pol = int(lineSeg[4].split(":")[1])
                    assert(pol == POS or pol == NEG)
                    node.outputPol(pol)
                i += 1
                if lineSeg[3].split(":")[1] != "and" and lineSeg[3].split(":")[1] != "Co": ##const and ci node don't have cut
                    continue
                else:
                    
                    for j in range(nMatch):
                        line = lines[i]
                        lineSegM = (line.split("\n")[0]).split(", ")
                        outPolM = int(lineSegM[1].split(":")[1])
                        assert(outPolM == POS or outPolM == NEG)
                        outLibM = libmgr.N2I(lineSegM[0].split(":")[1])
                        leaveNum = int(lineSegM[2].split(":")[1])
                        if outLibM == None:
                            print("lib %s not in lib list at line %d"%lineSegM[0].split(":")[1], i)
                            return 1
                        leaves = []
                        i += 1
                        for k in range(leaveNum):
                            line = lines[i]
                            lineSegL = (line.split("\n")[0]).split(", ")
                            # print(lineSegL)
                            idl = 2*int(lineSegL[0].split(":")[1])
                            assert(idl < len(self.aig))
                            leave = self.aig[idl]
                            assert(type(leave) != bufNode)
                            leaves.append([leave, int(lineSegL[1].split(":")[1])])
                            i += 1
                        # print(leaves)
                        cut = cutNode(leaves, node, outPolM, outLibM)
                        ret = node.cutAdd(cut, outPolM)
                        if ret :
                            return 1

        self.RSConnect(libmgr.singleLibs)
        for i in range(len(self.aig)):##add sortedAig
            node = self.aig[i]
            while len(self.sortedAig) <= node.level:
                self.sortedAig.append([])
            node.topoIndex = (node.level, len(self.sortedAig[node.level]))
            self.sortedAig[node.level].append(node)
            self.maxFanout = max(self.maxFanout, len(node.loadCutPos), len(node.loadCutNeg))
            if type(node) == inputNode:
                continue
            self.maxCut = max(len(node.cutPos), len(node.cutNeg), self.maxCut)
            for j in range(len(node.cutPos)):
                cut = node.cutPos[j]
                cut.cutIndex = j
            for j in range(len(node.cutNeg)):
                cut = node.cutNeg[j]
                cut.cutIndex = j
        self.level = len(self.sortedAig)
        for level in self.sortedAig:
            self.maxNode = max(self.maxNode, len(level))
        return 0

class cutNode:
    def __init__(self, leaves, output, outputPolarity , lib, bufCut = False):
        if bufCut == False:
            for i in range(len(leaves)):
                leaves[i][0] = leaves[i][0].outputBuf
        self.leaves = leaves ##(nodes, pol)
        self.output = output
        self.outputPolarity = outputPolarity
        self.lib = lib
        self.cutIndex = -1

class bufNode:
    def __init__(self, input, index):
        self.input = input
        self.level = -1
        self.index = index
        self.topoIndex = (-1, -1)
        self.cutPos = [] ##only come from current and node
        self.cutNeg = [] ##only come from current and node
        self.loadCutPos = []
        self.loadCutNeg = []
        self.outputNodes = []
        self.outputPolarity = NONE
    
    def showBuf(self):
        print("buf node index: ", self.index)
        print("input nodes: %s "%(self.input.index))
        print("outputNodes: ")
        for output in self.outputNodes:
            print(output.index)
        print("\n")

class inputNode:
    def __init__(self, index):
        self.level = 0
        self.index = index
        self.topoIndex = (0, -1)
        self.outputBuf = bufNode(self, index+1)
        self.loadCutPos = []
        self.loadCutNeg = []
        self.outputNodes = [self.outputBuf] ##only one output node
        
    
    def RScutAdd(self, singleLibs):
        for lib in singleLibs:
            if lib['polarity'] == POS:
                cutPOSTemp = cutNode([(self, POS)], self.outputBuf, POS, lib['index'], bufCut=True)
                self.outputBuf.cutPos.append(cutPOSTemp)
                self.loadCutPos.append((cutPOSTemp, 0))
            elif lib['polarity'] == NEG:
                cutPOSTemp = cutNode([(self, POS)], self.outputBuf, NEG, lib['index'], bufCut=True)
                self.outputBuf.cutNeg.append(cutPOSTemp)
                self.loadCutPos.append((cutPOSTemp, 0))
            else:
                print("illegal lib")
                return 1
        return 0

class andNode:
    def __init__(self, index):
        self.left = None
        self.right = None
        self.rightPolarity = NONE
        self.leftPolarity = NONE
        self.level = -1 ##level(?)
        self.index = index ##index in mgr.aig
        self.topoIndex = (-1, -1)
        self.cutPos = [] ## store cut that output is current postive aig node 
        self.cutNeg = [] ## store cut that output is current negative aig node
        self.outputBuf = bufNode(self, index+1)
        self.outputNodes = [self.outputBuf] ##only one output node
        self.loadCutPos = [] ## store the cuts that the current postive aig node connect to(only come from outputbuf) [node, inputIndex]
        self.loadCutNeg = [] ## store the cuts that the current negative aig node connect to (only come from outputbuf) [node, inputIndex]

        
    def leftConnect(self, Node, pol):
        self.left = Node.outputBuf
        self.leftPolarity = pol
        Node.outputBuf.outputNodes.append(self)
    
    def rightConnect(self, Node, pol):
        self.right = Node.outputBuf
        self.rightPolarity = pol
        Node.outputBuf.outputNodes.append(self)
    
    # def outputPos(self):
    #     self.outputBuf.outputPolarity = POS
        
    # def outputNeg(self):
    #     self.outputBuf.outputPolarity = NEG
    
    def outputPol(self, pol):
        self.outputBuf.outputPolarity = pol
    
    def cutAdd(self, cut, pol):
        if pol == NEG:
            self.cutNeg.append(cut)
        elif pol == POS:
            self.cutPos.append(cut)
        else:
            print("illegal operation")
            return 1
        
        for i in range(len(cut.leaves)):
            leave = cut.leaves[i]
            if leave[1] == NEG:
                leave[0].loadCutNeg.append((cut, i))
            elif leave[1] == POS:
                leave[0].loadCutPos.append((cut, i))
            else:
                print("illegal cut")
                return 1
        return 0
    
    def RScutAdd(self, singleLibs):
        for lib in singleLibs:
            if lib['polarity'] == POS:
                cutPOSTemp = cutNode([(self, POS)], self.outputBuf, POS, lib['index'], bufCut = True)
                cutNEGTemp = cutNode([(self, NEG)], self.outputBuf, NEG, lib['index'], bufCut = True)
                self.outputBuf.cutNeg.append(cutNEGTemp)
                self.outputBuf.cutPos.append(cutPOSTemp)
                self.loadCutPos.append((cutPOSTemp, 0))
                self.loadCutNeg.append((cutNEGTemp, 0))
            elif lib['polarity'] == NEG:
                cutPOSTemp = cutNode([(self, POS)], self.outputBuf, NEG, lib['index'], bufCut=True)
                cutNEGTemp = cutNode([(self, NEG)], self.outputBuf, POS, lib['index'], bufCut=True)
                self.outputBuf.cutNeg.append(cutPOSTemp)
                self.outputBuf.cutPos.append(cutNEGTemp)
                self.loadCutPos.append((cutPOSTemp, 0))
                self.loadCutNeg.append((cutNEGTemp, 0))
            else:
                print("illegal lib")
                return 1
        
        return 0
    
    
    def showAnd(self):
        print("and node index: ", self.index)
        print("input nodes: (%s %s) (%s %s)"%( self.left.index, self.leftPolarity, self.right.index, self.rightPolarity))
        print("outputNodes: %s"%(self.outputBuf.index))
        print("\n")
        self.outputBuf.showBuf()
        
        
            
        