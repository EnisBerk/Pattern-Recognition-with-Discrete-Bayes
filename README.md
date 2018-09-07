# Pattern-Recognition-with-Discrete-Bayes
Requirements(extra libararies):
pandas
numpy

SUMMARY:
There is a main.py file, one needs to run it only for the project.
Pr_data.txt is required to be in the same folder. With exact name.
Thats ALL. It prints results and saves to a csv file.


DETAILS:
All higher parameters are in the main file, such as class_probabilities and gain matrix.
One can change them accordingly.

Main.py creates a csv file and writes all outputs to that csv file.
Csv file keeps information about each iteration.

Bin_counts are hard coded to 6.
If bin_counts is not given, it calculates bin counts itself. However a  small sample from dataset is used for optimization so in that case bin counts can be smaller as well.
To prevent that we hard code the bin counts.

Learning alpha is 0.01
Iterations continue until there is no change in gain 100 times.

When iterations end, data is sampled one more time with bigger size and everything is recalculated starting with existing border values.



Bayes_funcs.py:
ML stuff: bayes calculation,smoothing, prediction


data_process_funcs.py:
keeps functions such as: reading data, entropy calculation, bin count cal. border calculation,filling memory table vs


pipline.py:
(brings all functions from data_process_funcs.py together in one function): read data,split it, cal_entopy,cal_borders,quantise,train model, make prediction, [we call this function as a starter]

optimiser_funcs.py:
optimisation functions are in this file.
Given a set of borders, they use given alpha to optimize boundaries
There are four types of them.
Original one from slides is called optimise_by_alpha.
Second one, optimise_hill_climb which uses hill climbing algorithm for optimisation,
this speeds up optimisation 5 to 10 times.
Third one is optimise_fixed_time, another version of previous alpha algorithm however it in this one alpha is not constant, it divides distance between borders to equal distance making sure that optimisation will run on fixed time.



