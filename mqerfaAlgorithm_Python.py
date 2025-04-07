from collections import defaultdict
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erfc
from tabulate import tabulate
import csv
import re
import keyboard
import matplotlib.lines as mlines

# Conditions for Output
show_plots = False
show_output = True

def parseInput(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    filename = lines[0].strip()
    date = lines[1].strip()
    comments = [lines[2].strip(), lines[3].strip(), lines[4].strip()]
    file_data = defaultdict(list) 

    num_files = int(lines[5].strip())
    for i in range(num_files):
        startIndex = 6 + i * 6
        hash = str(lines[startIndex + 1])
        for j in range(startIndex + 1, startIndex + 6):
            l = lines[j].strip()
            l = list(map(float, l.split()))
            l = [int(num) if num.is_integer() else num for num in l]
            key = lines[startIndex].strip() + hash
            file_data[key].append(l)

    return filename, date, comments, file_data, num_files

def createOutputFiles(inputfile, date, comments, data_dict):
    
    os.makedirs('MQERFA Results', exist_ok=True)
    os.makedirs('MQERFA Results/csv files', exist_ok=True)
    os.makedirs('MQERFA Results/Plot Images', exist_ok=True)

    types = ['txt'] ## TEMPORARY REMOVED '.FUL' OUTPUT DUE TO LACK OF FITTING DATA ## COMBINED DAT INTO RES
    for t in types:
        header = f"Output filename: {inputfile}.{t}\n\n{date}\n{comments[0]}\n{comments[1]}\n{comments[2]}\n\n"
        with open(f'MQERFA Results/{inputfile}_RESULTS.{t}', 'w') as f:
            f.write(header)

def importSpectralData(fname, channelOffset, startChannel, nChannels):
    extension = '.rpt' # Change to relevant file extension from input file ('.rpt', etc...)
    fname += extension

    # FOR USE WITH '.rpt' EXTENSION ( NO NEED FOR DSA CONVERT )
    pattern = re.compile(r'^\s*\d+:([\s\d]+)')
    numbers = []
    with open(fname, 'r') as f:
        for line in f:
            match = pattern.match(line)
            if match:
                r = re.findall(r'\d+', match.group(1))  # Extract all numbers after the colon
                numbers += r
    id = list(range(startChannel, startChannel + nChannels))  # Includes end
    numbers = numbers[channelOffset + startChannel - 1:channelOffset + startChannel + nChannels - 1]
    numbers = [float(n) for n in numbers]

    return id, numbers

    # FOR USE WITH '.out' EXTENSION
    # x = list(range(startChannel, startChannel + nChannels))  # Includes end
    # y = []
    # with open(fname, "r") as file:
    #     for line in file:
    #         c, second_col = map(float, line.split())
    #         if channelOffset + startChannel <= c < channelOffset + startChannel + nChannels:
    #             y.append(second_col)
    # return x, y         

def fchis(y, y_pred, nParams):
    nfree = len(y) - nParams
    # Initialize chi square
    chisq = 0.0
    
    # Accumulate Chi Square
    for i in range(len(y)):  # FORTRAN loops are inclusive of end point
        chisq = chisq + abs((y[i] - y_pred[i]) * (y[i] - y_pred[i]) / y_pred[i])
    
    # Divide by number of Degrees of Freedom
    free = nfree
    fchis_value = chisq / free
    
    return fchis_value

def fitAlpha(inputfile, fname, channelOffset, startChannel, nChannels, gain, dtype, randSeed1, randSeed2, width, height, A1, A2, A3, A4, A5, A6, nParamsToFit, fittingParams):

    # Importing Raw Data
    x, y = importSpectralData(fname[:-4], channelOffset, startChannel, nChannels)

    # Alpha Fitting Function
    def alphaFunc(x, a1, a2, a3, a4, a5, a6):
        term1 = a2 * (np.exp(-(((x - a1) / width) ** 2)) + height * erfc((x - a1) / width))
        term2 = (0.593 * a2) * (np.exp(-(((x - (a1 - 2165/gain)) / width) ** 2)) + height * erfc((x - (a1 - 2165/gain)) / width))
        term3 = a3 * np.exp(a4 * x) + a5 * np.exp(a6 * x) 

        return term1 + term2 + term3

    # Setting Initial Parameters
    initial_params = [A1, A2, A3, A4, A5, A6]

    # Initial Chi Sqr
    initial_y = np.array([alphaFunc(xi, *initial_params) for xi in x])
    chi_initial = fchis(y, initial_y, 6)

    # Bounding
    bounds = ([-np.inf] * 6, [np.inf] * 6)
    for f in fittingParams:
        bounds[0][f[0] - 1] = initial_params[f[0] - 1] - f[1]
        bounds[1][f[0] - 1] = initial_params[f[0] - 1] + f[1]

    # Fitting Model
    final_params, covariance = curve_fit(alphaFunc, x, y, p0=initial_params, bounds=bounds)
    
    # Final Chi Sqr
    final_y = np.array([alphaFunc(xi, *final_params) for xi in x])
    chi_final = fchis(y, final_y, 6)
    
    # Print the Terminal Output ######################################################################################
    terminalOutput = [
       [[fname, 'Alpha']],
       [['Lower Bound', bounds[0][0], bounds[0][1], bounds[0][2], bounds[0][3], bounds[0][4], bounds[0][5]],
        ['Upper Bound', bounds[1][0], bounds[1][1], bounds[1][2], bounds[1][3], bounds[1][4], bounds[1][5]],
        ['Initial Value', A1, A2, A3, A4, A5, A6],
        ['Final Value', final_params[0], final_params[1], final_params[2], final_params[3], final_params[4], final_params[5]]],
       [['Initial Chi Squared', chi_initial],
        ['Final Chi Squared', chi_final],
        ['Alpha Peak Amplitude', final_params[1]]]
    ]
    terminalOutput = [[[round(element, 5) if isinstance(element, (float, np.floating)) else element for element in row] for row in table] for table in terminalOutput]
    dataOutputStr = 100 * '=' + '\n' + tabulate(terminalOutput[0], tablefmt="pretty", headers=['Filename', 'Fitting Method']) + '\n' +\
                    tabulate(terminalOutput[1], tablefmt='pretty', headers=['A1', 'A2', 'A3', 'A4', 'A5', 'A6']) + '\n' + \
                    tabulate(terminalOutput[2], tablefmt='pretty') + '\n' * 3
    if show_output: print(dataOutputStr)
    ##################################################################################################################

    # Plot the data and fitted function ##############################################################################
    plt.scatter([v + channelOffset for v in x], y, label="Data", color='blue', marker='o', s=10, alpha=0.6)
    plt.plot([v + channelOffset for v in x], final_y, label="Fitted Curve", linestyle='-', color='red', linewidth=1)
    # Plot the vertical line with error-bar-style caps
    plt.errorbar(final_params[0]+channelOffset, (2 * alphaFunc(final_params[0], *final_params) - final_params[1]) / 2, 
             yerr=final_params[1] / 2, fmt='|', color='lightgreen', capsize=3, capthick=1, markersize=5, 
             label=f"Fitted Peak with Amplitude: {round(final_params[1], 2)}") 
    plt.legend()
    plt.title(fname[:-4])
    plt.xlabel('Channel')
    plt.ylabel('Counts')
    plt.savefig('MQERFA Results/Plot Images/' + fname[:-4] + '_alpha_' + '.jpg', dpi=300)
    if show_plots: plt.show()
    plt.clf()  # Clear the figure
    ##################################################################################################################

    # Writing to Output Files ########################################################################################
    # Plot csv File
    cChan = [v + channelOffset for v in x]
    data = list(zip(cChan, y, final_y))
    with open('MQERFA Results/csv files/' + fname[:-4] + '_alpha_' + '.csv', mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Channel Number", "Counts", "Fitted Counts"])  # Headers
        writer.writerows(data)  # Data rows

    # RES Output File
    file_path = 'MQERFA Results/' + inputfile + '_RESULTS.txt'  # Specify the file path here
    with open(file_path, 'a') as file:
        file.write(dataOutputStr)
    ##################################################################################################################

    return

def fitBeta(inputfile, fname, channelOffset, startChannel, nChannels, gain, dtype, randSeed1, randSeed2, width, height, A1, A2, A3, A4, nParamsToFit, fittingParams, coherentPeakPosition, coherentPeakMagnitude):
    
    # Importing Raw Data 
    x, y = importSpectralData(fname[:-4], channelOffset, startChannel, nChannels)

    # Beta Fitting Function
    def betaFunc(x, a1, a2, a3, a4):
        term1 = a1 * (np.exp(-((x-(coherentPeakPosition-3099/gain))/width)**2) + height * erfc((x-(coherentPeakPosition-3099/gain))/width))
        term2 = (0.523 * a1) * (np.exp(-((x-(coherentPeakPosition-3585/gain))/width)**2) + height * erfc((x-(coherentPeakPosition-3585/gain))/width))
        term3 = a2 * np.exp(a3 * x)
        
        POSCA = x - (coherentPeakPosition - 4038/gain)
        POSS = x - (coherentPeakPosition - 2747/gain)

        term4 = 0.030 * coherentPeakMagnitude * np.exp(a4 * POSCA) * erfc(POSCA/width)
        term5 = 0.44 * np.exp(a4 * POSS) * erfc(POSS/width)
        
        return term1 + term2 + term3 + term4 + term5


    # Setting Initial Parameters
    initial_params = [A1, A2, A3, A4]

    # Initial Chi Sqr
    initial_y = np.array([betaFunc(xi, *initial_params) for xi in x])
    chi_initial = fchis(y, initial_y, 4)

    # Bounding
    bounds = ([-np.inf] * 4, [np.inf] * 4)
    for f in fittingParams:
        bounds[0][f[0] - 1] = initial_params[f[0] - 1] - f[1]
        bounds[1][f[0] - 1] = initial_params[f[0] - 1] + f[1]

    # Fitting Model
    final_params, covariance = curve_fit(betaFunc, x, y, p0=initial_params, bounds=bounds)

    # Final Chi Sqr    
    final_y = np.array([betaFunc(xi, *final_params) for xi in x])
    chi_final = fchis(y, final_y, 4)
    
    # Print the Terminal Output ######################################################################################
    terminalOutput = [
       [[fname, 'Beta']],
       [['Lower Bound', bounds[0][0], bounds[0][1], bounds[0][2], bounds[0][3]],
        ['Upper Bound', bounds[1][0], bounds[1][1], bounds[1][2], bounds[1][3]],
        ['Initial Value', A1, A2, A3, A4],
        ['Final Value', final_params[0], final_params[1], final_params[2], final_params[3]]],
       [['Initial Chi Squared', chi_initial],
        ['Final Chi Squared', chi_final],
        ['Beta Peak Amplitude', final_params[0]]]
    ]
    terminalOutput = [[[round(element, 5) if isinstance(element, (float, np.floating)) else element for element in row] for row in table] for table in terminalOutput]
    dataOutputStr = 100 * '=' + '\n' + tabulate(terminalOutput[0], tablefmt="pretty", headers=['Filename', 'Fitting Method']) + '\n' +\
                    tabulate(terminalOutput[1], tablefmt='pretty', headers=['A1', 'A2', 'A3', 'A4']) + '\n' + \
                    tabulate(terminalOutput[2], tablefmt='pretty') + '\n' * 3
    if show_output: print(dataOutputStr)
    ##################################################################################################################

    # Plot the data and fitted function ##############################################################################
    plt.scatter([v + channelOffset for v in x], y, label="Data", color='blue', marker='o', s=10, alpha=0.6)
    plt.plot([v + channelOffset for v in x], final_y, label="Fitted Curve", linestyle='-', color='red', linewidth=1)
    # Plot the vertical line with error-bar-style caps (Beta may not have position therefore cant do this)
    # plt.errorbar(coherentPeakPosition+channelOffset, (2 * betaFunc(coherentPeakPosition, *final_params) - final_params[0]) / 2, 
    #          yerr=final_params[0] / 2, fmt='|', color='lightgreen', capsize=3, capthick=1, markersize=5, 
    #          label=f"Fitted Peak with Amplitude: {round(final_params[0], 2)}") 

    plt.legend()
    plt.title(fname[:-4])
    plt.xlabel('Channel')
    plt.ylabel('Counts')
    plt.savefig('MQERFA Results/Plot Images/' + fname[:-4] + '_beta_' + '.jpg', dpi=300)
    if show_plots: plt.show()
    plt.clf()  # Clear the figure
    ##################################################################################################################

    # Writing to Output Files ########################################################################################
    # Plot csv File
    cChan = [v + channelOffset for v in x]
    data = list(zip(cChan, y, final_y))
    with open('MQERFA Results/csv files/' + fname[:-4] + '_beta_' + '.csv', mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Channel Number", "Counts", "Fitted Counts"])  # Headers
        writer.writerows(data)  # Data rows
    
    # RES Output File
    file_path = 'MQERFA Results/' + inputfile + '_RESULTS.txt'  # Specify the file path here
    with open(file_path, 'a') as file:
        file.write(dataOutputStr)
    ##################################################################################################################

    return

def fitCoherentFloating(inputfile, fname, channelOffset, startChannel, nChannels, gain, dtype, randSeed1, randSeed2, A1, A2, A3, A4, A5, A6, A7, nParamsToFit, fittingParams):
  
    # Importing Raw Data
    x, y = importSpectralData(fname[:-4], channelOffset, startChannel, nChannels)
    
    # Coherent (Floating) Fitting Function
    def coherentFloatingFunc(x, a1, a2, a3, a4, a5, a6, a7):
        term1 = a2 * (np.exp(-((x - a1) / a3)**2) + a7 * erfc((x - a1) / a3))
        term2 = a4 * (np.exp(-((x - (a1 - 668 / gain)) / a3)**2) + a7 * erfc((x - (a1 - 668 / gain)) / a3))
        term3 = 0.509 * (np.exp(-((x - (a1 - 802 / gain)) / a3)**2) + a7 * erfc((x - (a1 - 802 / gain)) / a3))
        term4 = a5 * np.exp(a6 * x)
        
        return term1 + term2 + term3 + term4

    # Setting Initial Parameters
    initial_params = [A1, A2, A3, A4, A5, A6, A7]

    # Initial Chi Sqr
    initial_y = np.array([coherentFloatingFunc(xi, *initial_params) for xi in x])
    chi_initial = fchis(y, initial_y, 7)

    # Bounding
    bounds = ([-np.inf] * 7, [np.inf] * 7)
    for f in fittingParams:
        bounds[0][f[0] - 1] = initial_params[f[0] - 1] - f[1]
        bounds[1][f[0] - 1] = initial_params[f[0] - 1] + f[1]

    # Fitting Model
    final_params, covariance = curve_fit(coherentFloatingFunc, x, y, p0=initial_params, bounds=bounds)
    
    # Final Chi Sqr
    final_y = np.array([coherentFloatingFunc(xi, *final_params) for xi in x])
    chi_final = fchis(y, final_y, 7)
    
    # Print the Terminal Output ######################################################################################
    terminalOutput = [
       [[fname, 'Coherent (Floating)']],
       [['Lower Bound', bounds[0][0], bounds[0][1], bounds[0][2], bounds[0][3], bounds[0][4], bounds[0][5], bounds[0][6]],
        ['Upper Bound', bounds[1][0], bounds[1][1], bounds[1][2], bounds[1][3], bounds[1][4], bounds[1][5], bounds[1][6]],
        ['Initial Value', A1, A2, A3, A4, A5, A6, A7],
        ['Final Value', final_params[0], final_params[1], final_params[2], final_params[3], final_params[4], final_params[5], final_params[6]]],
       [['Initial Chi Squared', chi_initial],
        ['Final Chi Squared', chi_final],
        ['Coherent Peak Amplitude', final_params[1]]]
    ]
    terminalOutput = [[[round(element, 5) if isinstance(element, (float, np.floating)) else element for element in row] for row in table] for table in terminalOutput]
    dataOutputStr = 100 * '=' + '\n' + tabulate(terminalOutput[0], tablefmt="pretty", headers=['Filename', 'Fitting Method']) + '\n' +\
                    tabulate(terminalOutput[1], tablefmt='pretty', headers=['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7']) + '\n' + \
                    tabulate(terminalOutput[2], tablefmt='pretty') + '\n' * 3
    if show_output: print(dataOutputStr)
    ##################################################################################################################

    # Plot the data and fitted function ##############################################################################
    plt.scatter([v + channelOffset for v in x], y, label="Data", color='blue', marker='o', s=10, alpha=0.6)
    plt.plot([v + channelOffset for v in x], final_y, label="Fitted Curve", linestyle='-', color='red', linewidth=1)
    # Plot the vertical line with error-bar-style caps
    plt.errorbar(final_params[0]+channelOffset, (2 * coherentFloatingFunc(final_params[0], *final_params) - final_params[1]) / 2, 
             yerr=final_params[1] / 2, fmt='|', color='lightgreen', capsize=3, capthick=1, markersize=5, 
             label=f"Fitted Peak with Amplitude: {round(final_params[1], 2)}") 
    plt.legend()
    plt.title(fname[:-4])
    plt.xlabel('Channel')
    plt.ylabel('Counts')
    plt.savefig('MQERFA Results/Plot Images/' + fname[:-4] + '_cohF_' + '.jpg', dpi=300)
    if show_plots: plt.show()
    plt.clf()  # Clear the figure
    ##################################################################################################################

    # Writing to Output Files ########################################################################################
    # Plot csv File
    cChan = [v + channelOffset for v in x]
    data = list(zip(cChan, y, final_y))
    with open('MQERFA Results/csv files/' + fname[:-4] + '_cohF_' + '.csv', mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Channel Number", "Counts", "Fitted Counts"])  # Headers
        writer.writerows(data)  # Data rows

    # RES Output File
    file_path = 'MQERFA Results/' + inputfile + '_RESULTS.txt'  # Specify the file path here
    with open(file_path, 'a') as file:
        file.write(dataOutputStr)
    ##################################################################################################################

    return

def main():

    # Get the input file from user
    print('''
    MQERFA Fitting Algorithm (Python)
        
    This terminal or executable file ('.exe') must be in the same directory as the input file.
    All spectral data files ('.rpt' files) being analyzed must also be in this directory. 
    
    Input File ('.txt') must be in the following format:
    
    Ex.)
        alphaOut                                        # Filename
        4/1/2025                                        # Date
        Comment Line 1                                  # Next 3 lines can contain any comments or notes
        Comment Line 2
        Comment Line 3
        1                                               # Number of spectral data files being analyzed
        100-1.rpt                                       # Data file name with '.rpt' extension
        3                                               # Fitting method (beta: 2, alpha: 3, coherent floating: 11)
        1300  112  120  50.3  1  151  27  6.15  0.0303  # All Fitting Parameters on next 4 lines (varies based on fitting method)
        190  50  4e4  -0.01  2e6  0.03
        3
        1  4  4  0.1  6  0.1
        # Include more files here ...

    Please enter the name of the input file:
    ''')

    namefile = input().strip()
    
    # Parse input file and create output files
    inputfile, date, comments, data_dict, n = parseInput(namefile)
    createOutputFiles(inputfile, date, comments, data_dict)

    # Begin Analysis
    for key in data_dict:
        fitType = data_dict[key][0][0]

        if fitType == 3: # ALPHA
            # Initialize Variables
            channelOffset, startChannel, nChannels, gain, dtype, randSeed1, randSeed2, width, height = data_dict[key][1]
            A1, A2, A3, A4, A5, A6 = data_dict[key][2]
            nParamsToFit = data_dict[key][3][0]
            fittingParams = []
            for i in range(nParamsToFit):
                fittingParams.append(tuple(data_dict[key][4][2*i:2*i+2]))
            
            # Fitting
            fitAlpha(inputfile, key[:-2], channelOffset, startChannel, nChannels, gain, dtype, randSeed1, randSeed2, width, height, A1, A2, A3, A4, A5, A6, nParamsToFit, fittingParams)

            
        elif fitType == 2: # BETA
            # Initialize Variables
            channelOffset, startChannel, nChannels, gain, dtype, randSeed1, randSeed2, width, height, coherentPeakPosition, coherentPeakMagnitude = data_dict[key][1]
            A1, A2, A3, A4 = data_dict[key][2]
            nParamsToFit = data_dict[key][3][0]
            fittingParams = []
            for i in range(nParamsToFit):
                fittingParams.append(tuple(data_dict[key][4][2*i:2*i+2]))
         
            # Fitting
            fitBeta(inputfile, key[:-2], channelOffset, startChannel, nChannels, gain, dtype, randSeed1, randSeed2, width, height, A1, A2, A3, A4, nParamsToFit, fittingParams, coherentPeakPosition, coherentPeakMagnitude)
            
        elif fitType == 1: # COHERENT (FIXED)    
            # Initialize Variables
            channelOffset, startChannel, nChannels, gain, dtype, randSeed1, randSeed2, width, height = data_dict[key][1]
            A1, A2, A3, A4, A5 = data_dict[key][2]
            nParamsToFit = data_dict[key][3][0]
            fittingParams = []
            for i in range(nParamsToFit):
                fittingParams.append(tuple(data_dict[key][4][2*i:2*i+2]))
           
        elif fitType == 11: # COHERENT (FLOATING)
            # Initialize Variables
            channelOffset, startChannel, nChannels, gain, dtype, randSeed1, randSeed2 = data_dict[key][1]
            A1, A2, A3, A4, A5, A6, A7 = data_dict[key][2]
            nParamsToFit = data_dict[key][3][0]
            fittingParams = []
            for i in range(nParamsToFit):
                fittingParams.append(tuple(data_dict[key][4][2*i:2*i+2])) 
         
            # Fitting
            fitCoherentFloating(inputfile, key[:-3], channelOffset, startChannel, nChannels, gain, dtype, randSeed1, randSeed2, A1, A2, A3, A4, A5, A6, A7, nParamsToFit, fittingParams)
  
    print(f'''
    Completed Analyzing {n}/{n} Files
    Results stored in "MQERFA Results/"

    Press any key to exit...
    ''')
    keyboard.read_event()  # Wait for any key press

if __name__ == "__main__":
    main()