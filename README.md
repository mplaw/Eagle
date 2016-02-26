# Eagle
Python files to analyse data from the EAGLE simulation. 

Hello :) 
This folder contains Python code (2.7.9), that does the following:
1. Create a 2D array of column densities from an EAGLE snapshot
2. Plot the column density distribution functions of a 2D array of Column densities.
3. Group cells in the 2D array (from 1) using a Friends of Friends Algorithm.
4. Plot the auto-correlation of LLSs in a 2D array. 
5. Plot the auto and cross correlation of LLSs and galaxies.

To use these files
============================================================================
1. Open "CreateGrid.py" and change the parameters under the header "SET ME !!!".
2. Run "CreateGrid.py". (will take a long time)
3. Open "CorrelateGrid.py" and change the parameters under "SET ME !!!". 
4. Run "CorrelateGrid.py"
Repeat for steps 3 and 4 for other files.

File descriptions
===========================================================================
CreateGrid		: Code to project an EAGLE snapshot into a 2D grid.
CorrelateGrid		: All correlation related code
MiscellaneousFunctions	: Ancillary functions
PhysicalConstants	: All relevant physical constants, like the speed of light, in cgs units 
SnapshotParameters	: Class & Object for storing snapshot related information.
FoF2			: Use a Friends of friends algorithm to produce groups of LLSs.
Histogram		: Produce plots of the column density distribution function
			  and its first moment.
Voigt			: Simulate absorption by Hydrogen gas (Lyman series only)
			  for an initially flat spectra.

Files are provided "as is".
Author: Matt Law
