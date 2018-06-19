## PROGRAMMING LANGUAGE & REQUIREMENTS
The code uses Python 2.7 and Sci-Kit Learn Library for Neural Network, It also uses the matplotlib and numpy library 

## FILES
The folder contains three data files:
btc_final_day-feature file with only ending day prices
btc_merged_high_low: contains data for running the machine learning algorithm with 11 days as 1 if the prices on 10 day is high and 0 if prices on 10 day is low
cretae_data.py: contains code for preprocessing data and executing the Neural Network code. It is based on the Multilayer Perceptron function. and displays the ROC curve for ten days, and saves the output to the test.out file.
smooth_plot_ten_days.png: contains phot for ten days of the curve for entire dataset.
smooth_plot: contains plot for the entire 195 days
test.out: Contains the output of the neural network

## EXECUTING
To execute just run the create_data.py file by opening the code in Pycharm and run it.
