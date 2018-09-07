import data_process_funcs as dpf
import numpy as np
import bayes_funcs
import time

# if you do not give range_boundaries, then it finds min and max for each feature
#M is memory table size

def read_train_dev(clas_pro,gain_matrix,bin_counts=None,limit=-1,sampling=0.1,M=10000,feature_count=None,range_boundaries=None,print_cond=False):
    # freq=[data_x[m].value_counts() for m in range(5)]
    # in our data, we have 5 features each one of them are between 1 and 0 
    if print_cond:
        print("reading data...")

    alldata_x,alldata_y,K_size = dpf.read_data("pr_data.txt",limit,sampling)
    if range_boundaries==None:
        range_boundaries=alldata_x.apply(lambda x: pd.Series([x.min(), x.max()])).T.values.tolist()
    if feature_count==None:
        feature_count=alldata_x.shape[1]
    
##     shuffling
#     data_x_y=alldata_x.copy()
#     data_x_y[5]=alldata_y
#     data_x_y=data_x_y.sample(frac=1).reset_index(drop=True)
#     alldata_x=data_x_y[[0,1,2,3,4]]
#     alldata_y=data_x_y[5].values.tolist()
    
    
# 
    if print_cond:
        print("processing data...")
    n=len(alldata_x)
    train_end=int((n*3)/10)
    dev_end=int((n*6)/10)
    train_x=alldata_x[:train_end]
    train_y=alldata_y[:train_end]

    dev_x=alldata_x[train_end:dev_end]
    dev_y=alldata_y[train_end:dev_end]
    
    test_x = alldata_x[dev_end:]
    test_y = alldata_y[dev_end:]
    
    
#     print("total size is {}".format(n))
#     print("size of train is {}".format(len(train_x)))
#     print("size of dev is {}".format(len(dev_x)))
#     print("size of test is {}".format(len(test_x)))

    # TODO make multi class
    count=np.sum(dev_y)

    # we will calculate entropy with those bins
    # number_of_bins=n/10
    number_of_bins=10000
    
    K_size={1:count,0:len(dev_y)-count}
    M=number_of_bins

    if bin_counts==None:
        entropies =dpf.cal_entropies(range_boundaries,number_of_bins,train_x) 
        print("entropies",entropies)

        bin_counts=dpf.cal_bin_count(entropies,M)
        print("bin_counts",bin_counts)

    # index_adress((0,2,1,3,4),distent)

    # if print_pro:
    #     print("v_orders calculation")
    all_borders=dpf.cal_vorders(bin_counts,train_x)
#     print("filling memory table")
    """
    class_data_table=
    {0: {(1, 6, 2, 4, 1): 6.0,
      (0, 2, 0, 6, 6): 7.0,
      (0, 5, 0, 3, 4): 4.0,
      (3, 4, 6, 2, 2): 5.0,
     1: ...}
    """
    # deprecated using new one
    # class_data_table=fill_memory_table(K_size,train_x,train_y,all_borders)

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
    pred_y=bayes_funcs.predict(data_cond_pro,dev_x,all_borders)


    """
    KxK economic gain matrix
       0   1
    0  1   -1
    1 -2   3

    """
    theaccuracy = bayes_funcs.accuracy(pred_y,dev_y)
    best = bayes_funcs.true_expected_gain(gain_matrix,dev_y,dev_y)
    gain = bayes_funcs.true_expected_gain(gain_matrix,pred_y,dev_y)
    if print_cond:
        print("\nwith random quantisation points:")
        print("\naccuracy : {:4.3f}".format(theaccuracy))
        print("gain/best: {:4.3f} gain:{} best:{}  \n".format(gain/best,gain,best))
    return theaccuracy,all_borders,train_x,train_y,dev_x,dev_y,test_x,test_y,best,bin_counts,K_size,class_data_table,gain
