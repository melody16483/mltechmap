import torch
import numpy as np
from liberty.parser import parse_liberty
from liberty.types import *
import pandas as pd


##construct a testing library

POS = 0
NEG = 1
NONE = 2

#temp
# WIRE = 0
# BUF = 1
# INV1 = 2
# INV2 = 3
# AND1 = 4
# NAND1 = 5
# OR1 = 6

##nangate default lib temp
WIRE = 0
AND1 = 1
BUF = 25
INV1 = 56 
INV2 = 57
NAND1 = 64
OR1 = 98 


class libMgr:
    def __init__(self):
        self.libs = [] ##list of dict
        self.singleLibs = []
        self.nameToIndex = dict()
        
        ##add wire to lib
        newLib = dict()
        newLib["cellName"] = 'wire'
        self.libs.append(newLib)
        self.nameToIndex['wire'] = len(self.libs) - 1
        newLib["pins"] = [] ##list of dict 
        newLib["index"] = len(self.libs) - 1
        newLib["pinNameToIndex"] = dict()
        
        newPin = dict()
        newPin["pinName"] = 'input'
        newPin["load"] = 0
        newPin["slewACoeff"] = 0
        newPin["loadACoeff"] = 0
        newPin["bodyACoeff"] = 0
        newPin["slewSCoeff"] = 0
        newPin['loadSCoeff'] = 0
        newPin['bodySCoeff'] = 0
        newLib["pins"].append(newPin)
        newLib["pinNameToIndex"]['input'] = len(newLib["pins"]) - 1
    
    def process_timing_table(timing_table, library, cell_name, input_pin_name, output_pin_name, timing_table_name):
        """
        Process a timing table by performing a linear regression fit on the LUT data.
        Returns a dict with regression coefficients and error metrics.
        If grid indices are incomplete, returns None.
        """
        
        # Skip special cases. which are not supported by the regression model.
        template_name = timing_table.args[0]
        if template_name != 'Timing_7_7':
            return None

        # Get the grid arrays and LUT data.
        y_index = timing_table.get_array('index_1')
        x_index = timing_table.get_array('index_2')
        data = timing_table.get_array('values')


        # Convert grid indices to numpy arrays.
        capacitance_index = np.array(y_index)
        input_transition_index = np.array(x_index)

        # Build the grid for regression.
        Xgrid, Ygrid = np.meshgrid(capacitance_index, input_transition_index, indexing='ij')
        X_flat = Xgrid.flatten()
        Y_flat = Ygrid.flatten()
        Z_flat = np.array(data).flatten()

        # Build design matrix for regression: estimated = A * X + B * Y + C
        A_matrix = np.column_stack((X_flat, Y_flat, np.ones_like(X_flat)))
        coeffs, residuals, rank, s = np.linalg.lstsq(A_matrix, Z_flat, rcond=None)
        A_coef, B_coef, C_const = coeffs

        # Record and return the result.
        result_row = {
            'Cell': cell_name,
            'Input': input_pin_name,
            'Type': timing_table_name,
            'A': A_coef,
            'B': B_coef,
            'C': C_const
        }
        return result_row
    
    def libToExcel(self, fileName):
        library = parse_liberty(open(fileName).read())
        timing_tables_name = ['cell_rise', 'cell_fall', 'rise_transition', 'fall_transition']
        table_results = []
        input_pin_cap_results = []

        # Iterate over all cells.
        for cell in library.get_groups('cell'):
            cell_name = cell.args[0]
            cell = select_cell(library, cell_name)
            # Try to select the output pin; if missing, skip this cell.
            output_pin_name = "None"
            for pin in cell.get_groups('pin'):
                direction = pin.get_attribute('direction')
                if direction == 'output':
                    output_pin_name = pin.args[0]
                    break
            if output_pin_name == "None":
                print(f"Cell {cell_name}: no output pin found. Skipping.")
                continue
            output_pin = select_pin(cell, output_pin_name)

            # Iterate over all input pins.
            for pin in cell.get_groups('pin'):
                if pin.get_attribute('direction') == 'output':
                    continue       
                # Get input pin capacitance.             
                input_pin_name = pin.args[0]
                input_pin_capacitance =  pin.get_attribute('capacitance')        
                input_pin_cap_result = { "Cell": cell_name, "Input": input_pin_name, "Capacitance": input_pin_capacitance }
                if input_pin_cap_result is not None:
                    input_pin_cap_results.append(input_pin_cap_result)

                # Process timing tables.
                for timing_table_name in timing_tables_name:
                    try:
                        timing_table = select_timing_table(output_pin, related_pin=input_pin_name,
                                                        table_name=timing_table_name)
                    except Exception as e:
                        print(f"Cell {cell_name} {input_pin_name}->{output_pin_name}: No {timing_table_name} table found. Skipping.")
                        continue
                    result_row = process_timing_table(timing_table, library, cell_name, input_pin_name, output_pin_name, timing_table_name)
                    if result_row is not None:
                        table_results.append(result_row)
                        print(f"Processed {cell_name} {input_pin_name}->{output_pin_name}")  
        
    def ExcelParser(self, fileName):
        cap_file = fileName + ".lib_cell_capacitance.xlsx"
        timing_file = fileName + ".lib_cell_table_regression.xlsx"
        cap = pd.read_excel(cap_file)
        timing = pd.read_excel(timing_file)
        # self.nameToIndex = dict()
        
        for index, row in cap.iterrows():
            cell_name = row['Cell']
            input_pin_name = row['Input']
            cap = row['Capacitance']
            index = self.nameToIndex.get(cell_name)
            if index == None: ## lib not created 
                newLib = dict()
                newLib["cellName"] = cell_name
                self.libs.append(newLib)
                self.nameToIndex[cell_name] = len(self.libs) - 1
                newLib["pins"] = [] ##list of dict 
                newLib["index"] = len(self.libs) - 1
                newLib["pinNameToIndex"] = dict()
                
                newPin = dict()
                newPin["pinName"] = input_pin_name
                newPin["load"] = cap
                newPin["slewACoeff"] = 0
                newPin["loadACoeff"] = 0
                newPin["bodyACoeff"] = 0
                newPin["slewSCoeff"] = 0
                newPin['loadSCoeff'] = 0
                newPin['bodySCoeff'] = 0
                newLib["pins"].append(newPin)
                newLib["pinNameToIndex"][input_pin_name] = len(newLib["pins"]) - 1
            else:
                newPin = dict()
                newPin["pinName"] = input_pin_name
                newPin["load"] = cap
                newPin["slewACoeff"] = 0
                newPin["loadACoeff"] = 0
                newPin["bodyACoeff"] = 0
                newPin["slewSCoeff"] = 0
                newPin['loadSCoeff'] = 0
                newPin['bodySCoeff'] = 0
                self.libs[index]["pins"].append(newPin)
                self.libs[index]["pinNameToIndex"][input_pin_name] = len(self.libs[index]["pins"]) - 1
                
                
            
        ## all library created
        for index, row in timing.iterrows():
            type = row['Type']
            cell_name = row['Cell']
            input_pin_name = row['Input']
            libIndex = self.nameToIndex.get(cell_name)
            if libIndex == None:
                print("something wrong")
                return 1
            lib = self.libs[libIndex]
            pinIndex = lib["pinNameToIndex"].get(input_pin_name)
            pin = lib["pins"][pinIndex]
            if type == "cell_rise" or type == "cell_fall":
                pin["slewACoeff"] = max(pin["slewACoeff"], row['A'])
                pin["loadACoeff"] = max(pin["loadACoeff"], row['B'])
                pin["bodyACoeff"] = max(pin["bodyACoeff"], row['C'])
            elif type == "rise_transition" or type == "fall_transition":
                pin["slewSCoeff"] = max(pin["slewSCoeff"], row['A'])
                pin["loadSCoeff"] = max(pin["loadSCoeff"], row['B'])
                pin["bodySCoeff"] = max(pin["bodySCoeff"], row['C'])
            else:
                print("pin coeff type wrong")
                return 1
            
        for lib in self.libs:
            lib['polarity'] = NONE
            if len(lib['pins']) == 1:
                self.singleLibs.append(lib)
        
        self.LibPolarity()
        
        return 0
    
    def LibParser(self, fileName):
        self.libToExcel(fileName)
        ret = self.ExcelParser(fileName)
        return 0
    
    def LibPolarity(self):
        for lib in self.singleLibs:
            cellName = lib['cellName']
            index = cellName.find('INV')
            if index == -1:
                lib['polarity'] = POS
            else:
                lib['polarity'] = NEG
        return 0
    
    def N2I(self, cellName):
        return self.nameToIndex.get(cellName)
    def libInfo(self):
        for lib in self.libs:
            print(lib['cellName'], ":", lib['index'])
            for pin in lib['pins']:
                print("%s(%s)"%(pin['pinName'], pin['load']), end=', ')
            print("\n")
    
    def singleLibInfo(self):
        for lib in self.singleLibs:
            print(lib['cellName'], ":")
            for pin in lib['pins']:
                print("%s(%s)"%(pin['pinName'], pin['load']), end=', ')
            print("\n")


# class pin:
#     def __init__(self, pinName, slewACoeff, loadACoeff, bodyAcoeff, slewSCoeff, loadSCoeff, bodySCoeff, index, load):
#         self.pinName = pinName
#         self.slewACoeff = slewACoeff
#         self.loadACoeff = loadACoeff
#         self.bodyACoeff = bodyAcoeff
#         self.slewSCoeff = slewSCoeff
#         self.loadSCoeff = loadSCoeff
#         self.bodySCoeff = bodySCoeff
#         self.load = load
#         self.index = index

# class lib:
#     def __init__(self, pins, polarity, inputNum, name):
#         self.name = name
#         self.pins = pins
#         self.index = -1
#         self.polarity = polarity
#         self.inputNum = inputNum
        
        
# wire = lib([0, 0, 0, 0, 0], 0, 0, 0, 0, 0, 0, POS, 1,'wire')
# buf = lib([2, 0, 0, 0, 0], 1, 1, 1, 1, 1, 1, POS, 1, 'buf')
# inv1 = lib([3, 0, 0, 0, 0], 1, 1, 1, 1, 1, 1, NEG, 1, 'inv1')
# inv2 = lib([4, 0, 0, 0, 0], 1, 1, 1, 1, 1, 1, NEG, 1, 'inv2')
# and1 = lib([5, 5, 0, 0, 0], 1, 1, 1, 1, 1, 1, POS, 2, 'and1')
# nand1 = lib([6, 6, 0, 0, 0], 1, 1, 1, 1, 1, 1, NEG, 2, 'nand1')
# or1 = lib([7, 7, 0, 0, 0], 1, 1, 1, 1, 1, 1, POS, 2, 'or1')

# libs = [wire, buf, inv1, inv2, and1, nand1, or1]
# # libs = [buf, inv1, inv2, and1, nand1, or1]

# for i in range(len(libs)):
#     libs[i].index = i

# singleLibs = [wire, buf, inv1, inv2]
# singleLibs = [buf, inv1, inv2]