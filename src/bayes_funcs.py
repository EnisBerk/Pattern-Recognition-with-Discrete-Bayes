
from copy import deepcopy,copy
import itertools
import data_process_funcs as dpf
import numpy as np
from math import floor

def bayes_cal(class_data_table,clas_pro,bin_counts,all_borders):
    # v1.1
    N_class={}
    all_classes=list(class_data_table.keys())
    for clas in all_classes:
        N_class[clas]=sum(class_data_table[clas].values())
    
    clas_cond_prob=deepcopy(class_data_table)
    
#     N=sum(set(K_size.values()))
#     N=float(N)

#   iterate over all possible bin tuples
    ranges=[]
    for k in bin_counts:
        ranges.append([i for i in range(int(k))])
    
    for d in itertools.product(*ranges):
        for clas in clas_cond_prob:
#         for d in clas_cond_prob[clas]:
            d_count=clas_cond_prob[clas].get(d,0)
            smooth_amount=1
            clas_cond_prob[clas][d]=(d_count+smooth_amount)/N_class[clas]

#     data_pro=[clas_cond_prob[data][clas]*clas_prop[clas](given by prof) sum for all classes]

    data_pro={}
# #   iterate over all possible bin tuples
#     ranges=[]
#     for k in bin_counts:
#         ranges.append([i for i in range(int(k))])
    
    for a_bin in itertools.product(*ranges):
#         doing one smoothing TO-DO
        a_data_prob=0
        for clas in N_class:
            a_data_prob+=(clas_cond_prob[clas][a_bin]*clas_pro[clas])
            
        data_pro[a_bin]=a_data_prob
    
    data_cond_pro={}
#   iterate over all possible bin tuples for data cond class probabilities
    equal_clas_prob=1/len(all_classes)
    for a_bin in itertools.product(*ranges):
        data_cond_pro.setdefault(a_bin,{})
        for clas in N_class:
            clas_cond_prob_value=clas_cond_prob[clas][a_bin] #.get(a_bin,{clas:equal_clas_prob for clas in all_classes})
            data_cond_pro[a_bin][clas]=(clas_cond_prob_value*clas_pro[clas])/data_pro[a_bin]
    
    return data_cond_pro




# I think we need to smooth initial data, to prevent missing data
# so we get initial counts and return smoothed counts since we were gonna turn them into probabilities anyway

"""
class_data_table=
{0: {(1, 6, 2, 4, 1): 6.0,
  (0, 2, 0, 6, 6): 7.0,
  (0, 5, 0, 3, 4): 4.0,
  (3, 4, 6, 2, 2): 5.0,
 1: ...}
"""
# bin_counts=(2,3,4..)
def smoot_Dimensional(class_data_table,bin_counts,j,expanding_limit=12):

    number_of_dimensions=len(bin_counts)
    clas_names=list(class_data_table.keys())
#   class_data_table
    density_tables={}
    for clas in class_data_table.keys():
        density_tables[clas]={}
    
    
#   iterate over all possible bin tuples
    ranges=[]
    for k in bin_counts:
        ranges.append([i for i in range(int(k))])
    
    for a_bin in itertools.product(*ranges):
#       a_bin is one tuple (1, 6, 2, 4, 1)
        bm_values=[0]*len(clas_names)
        for i,clas in enumerate(clas_names):
            bm_values[i]=class_data_table[clas].get(a_bin,0)
        total_bm=sum(bm_values)
#       which dimension we are moving for neighbour  
        dimension_index=0
#     increase or decrease
        sign=1
#     we increase each dimensions value by round_count
        round_count=1.0 # first round 0,1,2..
        counter=1
        while total_bm<=j and counter<=(expanding_limit):
            counter+=1
#           copy original bin
            neighbour_bin=list(copy(a_bin))
#           change a dimension value to access neighbour
            neighbour_bin[dimension_index]+=floor(round_count)*sign
            neighbour_bin=tuple(neighbour_bin)
#           get all class scores from neighbour bin
            for i,clas in enumerate(clas_names):
                bm_values[i]+=class_data_table[clas].get(neighbour_bin,0)
            
            total_bm=sum(bm_values)
            
            dimension_index+=1
            if dimension_index==number_of_dimensions:
                dimension_index=0
                sign*=-1
                round_count+=0.5
                
        for i,clas in enumerate(clas_names):
            density_tables[clas][a_bin] = bm_values[i]/counter
#         we should not return probabilities, bayes func does that
#         density_tables[clas]/=(np.sum(density_tables[clas])/len(density_tables[clas]))
            
    return density_tables



# designed for class 0 and 1 TODO
def predict(data_cond_pro,test_x,all_borders):
    pred_y=[]
    for i,data_point in test_x.iterrows():
        adres=dpf.return_adres(data_point,all_borders)
        prob=data_cond_pro.get(adres,{0:0.5,1:0.5})
#         deterministic bayes
        if prob[0]>prob[1]:
            pred_y.append(0)
        elif prob[0]<prob[1]:
            pred_y.append(1)
        else:
            pred_y.append(randint(0,1))
#         probabilistic bayes
#         guess=random()

#         if prob[0]>guess:
#             pred_y.append(0)
#         else:
#             pred_y.append(1)
        
    return pred_y


def accuracy(y_pred,y_true):
    score = np.sum(np.array(y_pred)==np.array(y_true))
    return score/len(y_pred)

def gain(gain_m,y_pred,y_true):
    expectedgain=0
    for i in range(len(y_pred)):
        expectedgain+=gain_m[y_true[i]][y_pred[i]]
    return expectedgain

def true_expected_gain(gain_m,y_pred,y_true):
    expectedgain=0
    expectedgain_counts=[[0.0,0.0],[0.0,0.0]]

    # gain_matrix=[[1,-1],
            # [-2,3]]
    for i in range(len(y_pred)):
        if y_true[i]==y_pred[i]:
            if y_true[i]==1:
                expectedgain_counts[1][1]+=1
            else:
                expectedgain_counts[0][0]+=1
        else:
            if y_true[i]==1:
                expectedgain_counts[0][1]+=1
            else:
                expectedgain_counts[1][0]+=1
    total=len(y_pred)
    
    for i in range(2):
        for m in range(2):
            thisgain=expectedgain_counts[i][m]
            thisgain/=total
            thisgain*=gain_m[i][m]
            expectedgain+=thisgain
    
    return expectedgain

def true_expected_gain2(gain_m,y_pred,y_true):
    expectedgain=0
    expectedgain_counts=[[0.0,0.0],[0.0,0.0]]

    # gain_matrix=[[1,-1],
            # [-2,3]]
    for i in range(len(y_pred)):
        if y_true[i]==y_pred[i]:
            if y_true[i]==1:
                expectedgain_counts[1][1]+=1
            else:
                expectedgain_counts[0][0]+=1
        else:
            if y_true[i]==1:
                expectedgain_counts[0][1]+=1
            else:
                expectedgain_counts[1][0]+=1
    total=len(y_pred)
    expectedgain_counts_sum=0
    for i in range(2):
        for m in range(2):
            thisgain=expectedgain_counts[i][m]
            expectedgain_counts_sum+=thisgain
            thisgain/=total
            thisgain*=gain_m[i][m]
            expectedgain+=thisgain
    
    confusion_matrix=[[0.0,0.0],[0.0,0.0]]
    for i in range(2):
        for m in range(2):
            confusion_matrix[i][m]=expectedgain_counts[i][m]/expectedgain_counts_sum
    
    
    return expectedgain,confusion_matrix

def expected_gain(class_data_table,clas_pro,bin_counts,all_borders,test_x,y_pred,y_true,gain_m):
    # v1.1
    N_class={}
    all_classes=list(class_data_table.keys())
    for clas in all_classes:
        N_class[clas]=sum(class_data_table[clas].values())
    
    clas_cond_prob=deepcopy(class_data_table)
    
#     N=sum(set(K_size.values()))
#     N=float(N)

#   iterate over all possible bin tuples
    ranges=[]
    for k in bin_counts:
        ranges.append([i for i in range(int(k))])
    
    for d in itertools.product(*ranges):
        for clas in clas_cond_prob:
#         for d in clas_cond_prob[clas]:
            d_count=clas_cond_prob[clas].get(d,0)
            smooth_amount=1
            clas_cond_prob[clas][d]=(d_count+smooth_amount)/N_class[clas]

    
    data_cond_pro={}
#   iterate over all possible bin tuples for data cond class probabilities
    equal_clas_prob=1/len(all_classes)
    for a_bin in itertools.product(*ranges):
        data_cond_pro.setdefault(a_bin,{})
        for clas in N_class:
            clas_cond_prob_value=clas_cond_prob[clas][a_bin] #.get(a_bin,{clas:equal_clas_prob for clas in all_classes})
            data_cond_pro[a_bin][clas]=(clas_cond_prob_value*clas_pro[clas])
    # data_pro[a_bin]
    
    expectedgain=0
    for i in range(len(y_pred)):
        data_point=test_x.iloc[[i]]
        a_bin=dpf.return_adres(data_point,all_borders)
        expectedgain+=(gain_m[y_true[i]][y_pred[i]]*data_cond_pro[a_bin][y_true[i]])

    return expectedgain

    # return data_cond_pro
# gain_matrix=[[1,-1],
            # [-2,3]]


