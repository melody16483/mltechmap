from liberty.parser import parse_liberty
from liberty.types import *
import numpy as np
import argparse
import pandas as pd

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Evaluate LUT regression accuracy for each cell (cell_rise timing table).'
    )
    parser.add_argument('--liberty_file', type=str, default='./lib/Nangate45_typ.lib',
                        help='Path to the liberty file.')
    return parser.parse_args()

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

# Main code.
args = parse_arguments()
liberty_file = args.liberty_file

library = parse_liberty(open(liberty_file).read())

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

# Output table table_results to Excel.
df = pd.DataFrame(table_results, columns=['Cell','Input','Type','A','B','C'])
output_excel =  liberty_file + "_cell_table_regression.xlsx"
df.to_excel(output_excel, index=False)
print(f"Results exported to {output_excel}")

# Output capacitance table_results to Excel.
df = pd.DataFrame(input_pin_cap_results, columns=['Cell','Input','Capacitance'])
output_excel = liberty_file + "_cell_capacitance.xlsx"
df.to_excel(output_excel, index=False)
print(f"Results exported to {output_excel}")