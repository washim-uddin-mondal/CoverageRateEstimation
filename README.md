# CNN Based Coverage Estimation
 
This repository contains codes for coverage manifod estimation in cellular networks from real data. We have tested our
code on the base station location data of the following countries:

1. India
2. Brazil
3. Germany
4. USA

The data folder only contains the shapefiles of the above countries. The base station location files are not shared 
due to space restriction. The files are available at: https://www.opencellid.org. Download the csv files and save them in their
respective subfolders as 'BSLocations.csv'. For some of the countries, there are multiple base station
location files. In that case, save them as 'BSLocations0.csv', 'BSLocations1.csv' etc in the same subfolder.

The results are stored in the Results folder (created on the fly). The default values of all the parameters
can be found in Codes/Parameters.py file. Some parameter values can be modified from the command line as well.

# Command Line Options:

Use the following command to see all the options:  
python Scripts/Main.py --help

# Used Softwares/Packages:

python (3.8.3)    
numpy (1.19.5)  
pandas (1.2.8)  
torch (1.8.1)  
matplotlib (3.4.2)  
geopandas (0.6.2)


# Run Experiments

python Scripts/Main.py --coverage --country India --visualise --rerun 10 --fading_shape 1 --seeds 1     
python Scripts/Main.py --coverage --country Germany --lengthX 5 --lengthY 5 --fading_shape 1 --seeds 1   
python Scripts/Main.py --coverage --country USA --lengthX 5 --lengthY 5 --fading_shape 1 --seeds 1   
python Scripts/Main.py --coverage --country Brazil --visualise --rerun 4 --fading_shape 1 --seeds 1   

# Logging

Experiment progresses are logged into the following files:   

Results/USA/Shape1.0/Raw/progress.log   
Results/Brazil/Shape1.0/Raw/progress.log      
Results/India/Shape1.0/Raw/progress.log    
Results/Germany/Shape1.0/Raw/progress.log     

# Progress Summary

To see the progress summary, use the following command:

source progress.sh