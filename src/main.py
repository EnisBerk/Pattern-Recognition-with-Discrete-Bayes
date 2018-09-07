import pandas as pd
import numpy as np

import csv
from copy import deepcopy,copy
from math import ceil,inf,floor
import random
import itertools 
from pathlib import Path
import time

# dpf: reading data, entropy calculation, bin count cal. border calculation, quantiser border vs
# and filling memory table
import data_process_funcs as dpf
# bayes_funcs: ML stuff, bayes calculation,smoothing, prediction
import bayes_funcs
# pipline, (one function): read data,split it, cal_entopy,cal_borders,quantise,train model, make prediction, [we call this function as a starter]
import pipeline 
# genetic:extra experiment for finding best quantisation levels counts
# after loading data with pipeline, training for the first time, optimisers optimises boundaries, 
#trains model and repeats as long as required
import optimiser_funcs


random.seed(2)
# prior probabilires
clas_pro={0:0.4,1:0.6}
gain_matrix=[[1,-1],[-2,3]]

feature_count=5
range_boundaries=[(0,1)]*feature_count
bin_counts=[6,6,6,6,6]
M=10000
learning_alpha=0.01

#save all results to a csv file
flag=True
i=0
t0=time.time()
while flag:
    results_file="results_"+str(i)
    my_file = Path(results_file+".csv")
    if not my_file.is_file():
        result_handler=open(results_file+".csv","w")
        flag=False
    else:
        i+=1   
wr = csv.writer(result_handler, quoting=csv.QUOTE_ALL)
print("results at ",i)


for sam_i,sampling_rate in enumerate([0.001,0.01,0.1,1]):
    # number of lines to read, limit is count, sampling is percentage, uses lower one
    limit=-1;sampling=sampling_rate
    (theaccuracy,all_borders_first,train_x,train_y,dev_x,dev_y,test_x,test_y,best,bin_counts,K_size,class_data_table,gain)=pipeline.read_train_dev(clas_pro,gain_matrix,bin_counts,limit,sampling,M,feature_count,range_boundaries,print_cond=True)
    best_gain,NOUSEpred_y=optimiser_funcs.train_test_gain_now(train_x,train_y,best_ones,K_size,clas_pro,bin_counts,
                                         dev_x,dev_y,gain_matrix)
    print ("best possible gain ",best_gain)




    if sam_i==0:
        all_borders=all_borders_first
    old_gain=0
    counter4repet=0
    current_total_repetition=0
    while old_gain!=gain or current_total_repetition!=0:
        
        t1=time.time()
        old_gain=gain
        # all_borders,gain=optimiser_funcs.optimise_fixed_time(train_x,train_y,dev_x,dev_y,all_borders,gain,
        # 	K_size,clas_pro,bin_counts,gain_matrix,best,alpha=learning_alpha,M=5,limit_opti=2*M)
        all_borders,gain,theaccuracy=optimiser_funcs.optimise_hill_climb(train_x,train_y,dev_x,dev_y,all_borders,gain,
            K_size,clas_pro,bin_counts,gain_matrix,best,alpha=learning_alpha,percent=1,limit_opti=-1)
        t2=time.time()
        print("accuracy {} gain {}".format(theaccuracy,gain))
        #print(all_borders)
        wr.writerows(all_borders)
        wr.writerow([sampling_rate])
        wr.writerow([best])    
        wr.writerow([gain])
        wr.writerow([theaccuracy])
        wr.writerow([t2-t1])
        result_handler.flush()
        current_total_repetition=0
        for ind_q,list_border in enumerate(all_borders):
            repetation_count=optimiser_funcs.cal_repetition_count(list_border,learning_alpha)
            
            current_total_repetition+=repetation_count
            if repetation_count>0:
                counter4repet+=1
                original_gain=gain
                #     old_gain is equal to gain at this point
#                 if counter4repet>random.randint(5,10):
                if counter4repet>=0:
                    print("repetation_count{} feature {} counter".format(repetation_count,ind_q))
                    all_borders,gain=optimiser_funcs.fix_local_min(wr,gain,train_x,train_y,dev_x,
                                                    dev_y,all_borders,
                                                    K_size,clas_pro,bin_counts,
                                                    gain_matrix,best,ind_q,
                                                    alpha=learning_alpha,trial=2)

                    
                    counter4repet=0
        if old_gain==gain and current_total_repetition==0:
            t1=time.time()
            all_borders,gain,theaccuracy=optimiser_funcs.optimise_by_alpha(train_x,train_y,dev_x,dev_y,all_borders,gain,
            K_size,clas_pro,bin_counts,gain_matrix,best,alpha=learning_alpha,percent=1,limit_opti=-1)
            t2=time.time()
            print("by alpha accuracy {} gain {}".format(theaccuracy,gain))
            #print(all_borders)
            wr.writerows(all_borders)
            wr.writerow([sampling_rate])
            wr.writerow([best])    
            wr.writerow([gain])
            wr.writerow([theaccuracy])
            wr.writerow([t2-t1])
        
                


    #expected_gain=bayes_funcs.true_expected_gain(class_data_table,clas_pro,bin_counts,all_borders,dev_x,pred_y,dev_y,gain_matrix)
    #print(expected_gain)
t4=time.time()
print("total time passed",t4-t0)
result_handler.close()


print_cond=True

if print_cond:
    print("quantising...")
# class_data_table quantisation and storing in dict
# v1.2
class_data_table=dpf.fill_memory_table(train_x,train_y,all_borders,K_size.keys())

# class_data_table=smoot_Dimensional(class_data_table,bin_counts,j,expanding_limit)
if print_cond:
    print("learning...")
data_cond_pro=bayes_funcs.bayes_cal(class_data_table,clas_pro,bin_counts,all_borders)
# data_cond_pro=bayes_cal(class_data_table,K_size)
if print_cond:
    print("predicting...")
pred_y=bayes_funcs.predict(data_cond_pro,test_x,all_borders)


"""
KxK economic gain matrix
   0   1
0  1   -1
1 -2   3

"""
print("results smoothing off, with test data")
theaccuracy = bayes_funcs.accuracy(pred_y,test_y)
best,confusion_matrix_no = bayes_funcs.true_expected_gain2(gain_matrix,test_y,test_y)
gain,confusion_matrix = bayes_funcs.true_expected_gain2(gain_matrix,pred_y,test_y)
if print_cond:
    print("\naccuracy : {:4.3f}".format(theaccuracy))
    print("gain/best: {:4.3f} gain:{} best:{}  \n".format(gain/best,gain,best))
    print(confusion_matrix)
    
print("results smoothing on, with test data")

js=[1,2,3,4]
ks=[]
for j in js:
    ks.append((j*len(test_x))/(20*10000))
    
for k in ks:
    if print_cond:
        print("quantising...")
    # class_data_table quantisation and storing in dict
    # v1.2
    class_data_table=dpf.fill_memory_table(train_x,train_y,all_borders,K_size.keys())
    if print_cond:
        print("smoothing...")
    class_data_table=bayes_funcs.smoot_Dimensional(class_data_table,bin_counts,k)

    if print_cond:
        print("learning...")
    data_cond_pro=bayes_funcs.bayes_cal(class_data_table,clas_pro,bin_counts,all_borders)
    # data_cond_pro=bayes_cal(class_data_table,K_size)
    if print_cond:
        print("predicting...")
    pred_y=bayes_funcs.predict(data_cond_pro,test_x,all_borders)


    """
    KxK economic gain matrix
       0   1
    0  1   -1
    1 -2   3

    """
    theaccuracy = bayes_funcs.accuracy(pred_y,test_y)
    best,confusion_matrix_no = bayes_funcs.true_expected_gain2(gain_matrix,test_y,test_y)
    gain,confusion_matrix = bayes_funcs.true_expected_gain2(gain_matrix,pred_y,test_y)
    if print_cond:
        print("k",k)
        print("\naccuracy : {:4.3f}".format(theaccuracy))
#         print("gain/best: {:4.3f} gain:{} best:{}  \n".format(gain/best,gain,best))
        print("gain",gain)
        print(confusion_matrix)




