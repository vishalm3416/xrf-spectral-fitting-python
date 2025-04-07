# MQERFA Spectral Fitting Algorithm (Python)
This repository contains a modern Python implementation of the MQERFA (Multi-Quantitative Elemental Region Fitting Algorithm), used for modeling and deconvolving peaks in X-ray fluorescence (XRF) spectral data. The original algorithm was written in Fortran, and this updated version offers improved performance, enhanced platform compatibility, and accurate peak fitting with low chi-squared error.

## 🔍 Overview
This tool is designed for researchers and analysts working with XRF data, especially in biomedical or environmental applications. It supports modeling alpha, coherent, and beta peaks from raw spectral files and outputs both plots and processed data.

## 📁 Files in This Repository
mqerfaAlgorithm_Python.py – Source code for the spectral fitting algorithm (Python).

mqerfaAlgorithm_Python.exe – Standalone executable built with all dependencies (Python not required).

100-1.rpt – Example raw spectral input file.

testInput.txt – Example input configuration file that defines fitting parameters and calls the .rpt file.

MQERFA Results/ – Output folder containing:

Plot Images/100-1_alpha_.jpg – Example peak fitting plot.

csv files/100-1_alpha_.csv – Peak fitting results in CSV format.

testInput_RESULTS.txt – Summary of all fitting results from the input file.

## 🚀 Running the Program
You can run the program using either the Python script or the compiled executable.

### Option 1: Using the Executable (mqerfaAlgorithm_Python.exe)
Place the .exe file in the same folder as:

Your .txt input file (e.g., testInput.txt)

All .rpt spectral files referenced in the input

Run the .exe from the terminal or by double-clicking.

### Option 2: Running the Python Script
You can run the Python script directly if you have the required packages installed:

## 📊 Output
The algorithm generates:

.csv file(s) containing peak fitting results

.jpg image(s) of the original data with fitted curves

A summary .txt file with fitting results (including chi-squared errors)

All outputs are saved in the MQERFA Results/ folder.

## ✅ Features
🔁 Batch process multiple spectral files

⚡ Fast and optimized Python fitting using least-squares minimization

🖥️ Compatible across Windows systems (Python and .exe versions)

📉 Low chi-squared error and high accuracy for alpha and coherent peak modeling

📂 Organized output with labeled plots and exportable data
