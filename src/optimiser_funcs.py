
import data_process_funcs as dpf
import bayes_funcs
import random 
from math import ceil,inf,floor

def cal_repetition_count(list_border,learning_alpha):
    uniq_elements=set()
    for element in list_border[1:-1]:
        uniq_elements.add(int(element*(1/learning_alpha)))
    uniq_elements.add(0*(1/learning_alpha))
    uniq_elements.add(int(0.999*(1/learning_alpha)))
    uniq_elements.add(1*(1/learning_alpha))
    repetation_count=len(list_border)+1-len(uniq_elements)
    return repetation_count


def fix_local_min(wr,original_gain,train_x,train_y,dev_x,
                  dev_y,all_borders,
                  K_size,clas_pro,bin_counts,
                  gain_matrix,best,qindex,
                  alpha=0.01,trial=1):
    func_flag=True
    giventrial=trial
    while func_flag:
        t1=time.time()
        new_q=[-inf,inf]
        for m in range(5):
            new_q.append(random.random())
        new_q.sort()
        old_q=all_borders[qindex][:]
        all_borders[qindex]=new_q[:]
    # reset gain here
        gain=0
        old_gain=original_gain
        while old_gain!=gain or trial>0:

            t1=time.time()
            old_gain=gain
            if trial>1:
                all_borders,gain,theaccuracy=optimiser_funcs.optimise_hill_climb_single(train_x,train_y,dev_x,dev_y,
                                                                     all_borders,gain,K_size,
                                                                     clas_pro,bin_counts,
                                                                     gain_matrix,best,
                                                                     alpha=alpha,
                                                                     percent=1,limit_opti=-1,
                                                                     qindex=qindex)
            else:
                all_borders,gain,theaccuracy=optimiser_funcs.optimise_by_alpha_single(train_x,train_y,dev_x,dev_y,
                                                         all_borders,gain,K_size,
                                                         clas_pro,bin_counts,
                                                         gain_matrix,best,
                                                         alpha=alpha,
                                                         percent=1,limit_opti=-1,
                                                         qindex=qindex)
        
        
            t2=time.time()
            print("Rep. accuracy {} gain {}".format(theaccuracy,gain))
            if old_gain==gain:
                trial-=1

        if original_gain>=gain:
            all_borders[qindex]=old_q[:]
            print ("Stayed",original_gain,gain)
            trial=giventrial
        else:
            func_flag=False
            print ("Fixed",original_gain,gain)
            original_gain=gain
            
            wr.writerows(all_borders)
            wr.writerow([sampling_rate])
            wr.writerow([best])    
            wr.writerow([gain])
            wr.writerow([theaccuracy])
            wr.writerow([t2-t1])
            result_handler.flush()

    return all_borders,original_gain
    


def optimise_fixed_time(train_x,train_y,dev_x,dev_y,all_borders,gain,K_size,clas_pro,bin_counts,gain_matrix,best,alpha=0.01,M=5,limit_opti=-1):
    """
    this part is for optimising quantisation borders:
        we choose a feature and quant. border randomly:
            move border by -(alpha*M), then move it to otherside
            by alpha for 2*M times, 
            calculate decision rule with train_data, calculate expected gain with dev_data
        at the end pick highest expected gain
    """
    if limit_opti==-1:
        limit_opti=2*(M+1)
#     alpha=0.01
#     M=5
    # if it does not improve after limit_opti times then stop trying
#     limit_opti=12
    # populate all boundaries first, to select randomly or to go on until finishing them
    # [[1,2,3],[1,2,3,4,5]...]
    boundaries=[]
    for border in all_borders:
        boundaries.append([i+1 for i in range(len(border)-2)])
     
    boundary_indexes=[i for i in range(len(all_borders))]

    # provided by previous function call
    # best = expected_gain(gain_matrix,dev_y,dev_y)
    while boundary_indexes:
        # print ("left {} feature to optimise".format(len(boundaries)))
    #     choose a feature randomly
        feature_index=random.choice(boundary_indexes)
    #     choose a boundary randomly
        bound_index=random.randint(0,len(boundaries[feature_index])-1)
        boundary=boundaries[feature_index].pop(bound_index)
    #   if feature do not have any other bounadry then remove it
        if boundaries[feature_index]==[]:
            boundary_indexes.remove(feature_index)

    #   get value of the border,
        bkj=all_borders[feature_index][boundary]
    #   move border to left by alpha*(M+1)
        bkj_start=bkj-(alpha*(M+1))
        bkj_end=bkj+(alpha*(M+1))
    #     bkj=bkj-(alpha*(M+1))


    #   if border smaller than previous border or bigger than next border than pass it 
        if bkj_start < all_borders[feature_index][boundary-1] or bkj_end >= all_borders[feature_index][boundary+1]:
            if all_borders[feature_index][boundary-1]>bkj_start:
                bkj_start=all_borders[feature_index][boundary-1]+alpha
            if all_borders[feature_index][boundary+1]<=bkj_end:
                bkj_end=all_borders[feature_index][boundary+1]-alpha
    #         continue
        # print ("feature:{},boundary: {}".format(feature_index,boundary))
        # print("end:{} current:{} start:{}".format(bkj_end,bkj,bkj_start))
        count=0
    #     we will add 2M, 2M-1, 2M-2 ... mulitplied with alpha
    # so we are starting from right side to search
        opti_index=1
        new_bkj=bkj_start+(alpha*opti_index)
        while(new_bkj<=bkj_end):
            
    #       change corresponding border with new value
            all_borders[feature_index][boundary]=new_bkj
                
            class_data_table=dpf.fill_memory_table(train_x,train_y,all_borders,K_size.keys())
    #       print ("learning")
            data_cond_pro=bayes_funcs.bayes_cal(class_data_table,clas_pro,bin_counts,all_borders)

    #         print("predicting")
            pred_y=bayes_funcs.predict(data_cond_pro,dev_x,all_borders)

            new_gain = bayes_funcs.true_expected_gain(gain_matrix,pred_y,dev_y)
            if new_gain<=gain:
                count+=1
                all_borders[feature_index][boundary]=bkj

                # print ("lost:{},bkj:{}".format(gain-new_gain,new_bkj))
                
    #             print ("{} did not update line ".format(count))
            else:
                count=0
                # print("--------------------------improved by {}".format(new_gain-gain))
                gain=new_gain
                bkj=new_bkj

            if count>=limit_opti:
    #             opti_index+=limit_opti
                count=0
                break
            opti_index+=1
            new_bkj=bkj_start+(alpha*opti_index)

    #print("gain/best: {:4.3f} gain:{} best:{}  \n".format(gain/best,gain,best))
    theaccuracy = bayes_funcs.accuracy(pred_y,dev_y)
    #print("\naccuracy : {:4.3f}".format(theaccuracy))
    return all_borders,gain,theaccuracy

def optimise_by_alpha(train_x,train_y,dev_x,dev_y,all_borders,gain,K_size,clas_pro,bin_counts,gain_matrix,best,alpha=0.01,percent=0.5,limit_opti=-1):
    """
    this part is for optimising quantisation borders:
        we choose a feature and quant. border randomly:
            move border by -(alpha*M), then move it to otherside
            by alpha for 2*M times, 
            calculate decision rule with train_data, calculate expected gain with dev_data
        at the end pick highest expected gain
    """
# if it does not improve after limit_opti times then stop trying

    if limit_opti==-1:
        limit_opti=inf
#     limit_opti=12
    # populate all boundaries first, to select randomly or to go on until finishing them
    # passes first one and last one since they are infinite
    # [[1,2,3],[1,2,3,4,5]...]
    boundaries=[]
    for border in all_borders:
        boundaries.append([i+1 for i in range(len(border)-2)])
        
    #all indexes of features [0,1,2,3,4,5]
    feature_indexes=[i for i in range(len(all_borders))]
    
    #while there are features to optimise
    while feature_indexes:

    #     choose a feature randomly
        feature_index=random.choice(feature_indexes)
    #     choose a boundary randomly
        boundary=random.choice(boundaries[feature_index])
        boundaries[feature_index].remove(boundary)
        
        #bound_index=random.randint(0,len(boundaries[feature_index])-1)
        #boundary=boundaries[feature_index].pop(bound_index)
        
    #   if feature do not have any other bounadry then remove it
        if boundaries[feature_index]==[]:
            feature_indexes.remove(feature_index)

    #   get value of the border,
        bkj=all_borders[feature_index][boundary]
    #   move border to left by alpha*(M+1)
        upper_bound=all_borders[feature_index][boundary+1]
        lower_bound=all_borders[feature_index][boundary-1]

        if (upper_bound !=inf) and (lower_bound!=-inf):
            upper_distance=(upper_bound-bkj)*percent

            lower_distance=(bkj-lower_bound)*percent
            
        elif (upper_bound ==inf) and (lower_bound==-inf):
            upper_distance=(1-bkj)*percent
            
            lower_distance=(bkj-0)*percent
        elif (upper_bound ==inf):
            upper_distance=(1-bkj)*percent
            
            lower_distance=(bkj-lower_bound)*percent
        elif (lower_bound==-inf):
            upper_distance=(upper_bound-bkj)*percent

            lower_distance=(bkj-0)*percent

        bkj_end=bkj+upper_distance
        bkj_start=bkj-lower_distance
    #         continue
        # print ("feature:{},boundary: {}".format(feature_index,boundary))
        # print("end:{} current:{} start:{}".format(bkj_end,bkj,bkj_start))
    #     we will add 2M, 2M-1, 2M-2 ... mulitplied with alpha
    # so we are starting from right side to search
        opti_index=1
        new_bkj=bkj_start+(alpha*opti_index)
        while(new_bkj<bkj_end):
            
    #       change corresponding border with new value
            all_borders[feature_index][boundary]=new_bkj
            
            new_gain,pred_y=train_test_gain_now(train_x,train_y,all_borders,K_size,clas_pro,bin_counts,
                                         dev_x,dev_y,gain_matrix)
            #do not put equal so it moves to onto one and another TT
            if new_gain<gain:
                all_borders[feature_index][boundary]=bkj

            else:
                # print("--------------------------improved by {}".format(new_gain-gain))
                gain=new_gain
                bkj=new_bkj

            opti_index+=1
            new_bkj=bkj_start+(alpha*opti_index)


    #print("gain/best: {:4.3f} gain:{} best:{}  \n".format(gain/best,gain,best))
    theaccuracy = bayes_funcs.accuracy(pred_y,dev_y)
    #print("\naccuracy : {:4.3f}".format(theaccuracy))
    return all_borders,gain,theaccuracy
def optimise_by_alpha_single(train_x,train_y,dev_x,dev_y,all_borders,gain,K_size,clas_pro,bin_counts,gain_matrix,best,alpha=0.01,percent=0.5,limit_opti=-1,qindex=0):
    """
    this part is for optimising quantisation borders:
        we choose a feature and quant. border randomly:
            move border by -(alpha*M), then move it to otherside
            by alpha for 2*M times, 
            calculate decision rule with train_data, calculate expected gain with dev_data
        at the end pick highest expected gain
    """
    if limit_opti==-1:
        limit_opti=inf
#     alpha=0.01
#     M=5
    # if it does not improve after limit_opti times then stop trying
#     limit_opti=12
    # populate all boundaries first, to select randomly or to go on until finishing them
    # passes first one and last one since they are infinite
    # [[1,2,3],[1,2,3,4,5]...]
    boundaries=[]
    for border in all_borders:
        boundaries.append([i+1 for i in range(len(border)-2)])
    #CH
    #we do not need all of them #feature_indexes=[i for i in range(len(all_borders))]
    feature_indexes=[qindex]
    # provided by previous function call
    # best = expected_gain(gain_matrix,dev_y,dev_y)
    #CH
    while feature_indexes:

        feature_index=random.choice(feature_indexes)
    #     choose a boundary randomly
        boundary=random.choice(boundaries[feature_index])
        boundaries[feature_index].remove(boundary)
        
    #   if feature do not have any other bounadry then remove it
        if boundaries[feature_index]==[]:
            feature_indexes.remove(feature_index)

    #   get value of the border,
        bkj=all_borders[feature_index][boundary]
    #   move border to left by alpha*(M+1)
        upper_bound=all_borders[feature_index][boundary+1]
        lower_bound=all_borders[feature_index][boundary-1]

        if (upper_bound !=inf) and (lower_bound!=-inf):
            upper_distance=(upper_bound-bkj)*percent

            lower_distance=(bkj-lower_bound)*percent
            
        elif (upper_bound ==inf) and (lower_bound==-inf):
            upper_distance=(1-bkj)*percent
            
            lower_distance=(bkj-0)*percent
        elif (upper_bound ==inf):
            upper_distance=(1-bkj)*percent
            
            lower_distance=(bkj-lower_bound)*percent
        elif (lower_bound==-inf):
            upper_distance=(upper_bound-bkj)*percent

            lower_distance=(bkj-0)*percent
            

        bkj_end=bkj+upper_distance
        bkj_start=bkj-lower_distance
    #         continue

    #     we will add 2M, 2M-1, 2M-2 ... mulitplied with alpha
    # so we are starting from right side to search
        opti_index=1
        new_bkj=bkj_start+(alpha*opti_index)
        while(new_bkj<bkj_end):
    #       change corresponding border with new value
            all_borders[feature_index][boundary]=new_bkj
            
    #         do not need to fill the memory every time
    
            new_gain,pred_y=train_test_gain_now(train_x,train_y,all_borders,K_size,clas_pro,bin_counts,
                                         dev_x,dev_y,gain_matrix)
           #do not put equal so it moves to onto one and another TT
            if new_gain<gain:
                all_borders[feature_index][boundary]=bkj
                
            else:
                gain=new_gain
                bkj=new_bkj

            new_bkj=bkj_start+(alpha*opti_index)
            opti_index+=1

    theaccuracy = bayes_funcs.accuracy(pred_y,dev_y)
    return all_borders,gain,theaccuracy

def train_test_gain_now(train_x,train_y,all_borders,K_size,clas_pro,bin_counts,
                        dev_x,dev_y,gain_matrix):
    class_data_table=dpf.fill_memory_table(train_x,train_y,all_borders,K_size.keys())
    #       print ("learning")
    data_cond_pro=bayes_funcs.bayes_cal(class_data_table,clas_pro,bin_counts,all_borders)
    #         print("predicting")
    pred_y=bayes_funcs.predict(data_cond_pro,dev_x,all_borders)
    new_gain = bayes_funcs.true_expected_gain(gain_matrix,pred_y,dev_y)
    return new_gain,pred_y

            
def optimise_hill_climb(train_x,train_y,dev_x,dev_y,all_borders,gain,K_size,clas_pro,
                        bin_counts,gain_matrix,best,alpha=0.01,percent=0.5,limit_opti=-1):
    """
    this part is for optimising quantisation borders:
        we choose a feature and quant. border randomly:
            move border by -(alpha*M), then move it to otherside
            by alpha for 2*M times, 
            calculate decision rule with train_data, calculate expected gain with dev_data
        at the end pick highest expected gain
    """
    if limit_opti==-1:
        limit_opti=inf
#     alpha=0.01
#     M=5
    # if it does not improve after limit_opti times then stop trying
#     limit_opti=12
    # populate all boundaries first, to select randomly or to go on until finishing them
    # passes first one and last one since they are infinite
    # [[1,2,3],[1,2,3,4,5]...]
    boundaries=[]
    for border in all_borders:
        boundaries.append([i+1 for i in range(len(border)-2)])
        
    feature_indexes=[i for i in range(len(all_borders))]

    # provided by previous function call
    # best = expected_gain(gain_matrix,dev_y,dev_y)
    while feature_indexes:
        #print ("gain {}".format(gain))
        left_count=sum([len(alen) for alen in boundaries ])
        #if left_count%5==0:
        #    print ("left {} feature to optimise".format(left_count))
        #if left_count%7==0:
        #    print (all_borders)
    #     choose a feature randomly
        feature_index=random.choice(feature_indexes)
    #     choose a boundary randomly
        boundary=random.choice(boundaries[feature_index])
        boundaries[feature_index].remove(boundary)
        
    #   if feature do not have any other bounadry then remove it
        if boundaries[feature_index]==[]:
            feature_indexes.remove(feature_index)

    #   get value of the border,
        bkj=all_borders[feature_index][boundary]
    #   move border to left by alpha*(M+1)
        upper_bound=all_borders[feature_index][boundary+1]
        lower_bound=all_borders[feature_index][boundary-1]

        if (upper_bound !=inf) and (lower_bound!=-inf):
            upper_distance=(upper_bound-bkj)*percent

            lower_distance=(bkj-lower_bound)*percent
            
        elif (upper_bound ==inf) and (lower_bound==-inf):
            upper_distance=(1-bkj)*percent
            
            lower_distance=(bkj-0)*percent
        elif (upper_bound ==inf):
            upper_distance=(1-bkj)*percent
            
            lower_distance=(bkj-lower_bound)*percent
        elif (lower_bound==-inf):
            upper_distance=(upper_bound-bkj)*percent

            lower_distance=(bkj-0)*percent

        bkj_end=bkj+upper_distance
        bkj_start=bkj-lower_distance
        
        
        random_point=random.uniform(bkj_start+alpha,bkj_end-alpha)        
        left_side_point = random_point-alpha
        right_side_point = random_point+alpha
        
        all_borders[feature_index][boundary]=left_side_point
        left_side_point_gain,pred_y=train_test_gain_now(train_x,train_y,all_borders,
                                                        K_size,clas_pro,bin_counts,
                                                        dev_x,dev_y,gain_matrix)
        
        all_borders[feature_index][boundary]=right_side_point
        right_side_point_gain,pred_y=train_test_gain_now(train_x,train_y,all_borders,
                                                         K_size,clas_pro,bin_counts,
                                                         dev_x,dev_y,gain_matrix)
        
        if (left_side_point_gain>gain) or (right_side_point_gain>gain):
            
            if left_side_point_gain>=right_side_point_gain:
                if (bkj_start<left_side_point):
                    all_borders[feature_index][boundary]=left_side_point
                    gain=left_side_point_gain
                    bkj=left_side_point
                    flag=True
                    while (flag):
                        left_side_point=bkj-alpha
                        all_borders[feature_index][boundary]=left_side_point

                        if (bkj_start<=left_side_point):
                            left_side_point_gain,pred_y=train_test_gain_now(train_x,train_y,
                                                                            all_borders,K_size,
                                                                            clas_pro,bin_counts,
                                                                            dev_x,dev_y,gain_matrix)
                            #do not put equal so it moves to onto one and another TT
                            if (gain>left_side_point_gain):
                                flag=False
                                all_borders[feature_index][boundary]=bkj
                            else:
                                bkj=left_side_point
                                gain=left_side_point_gain

                        else:
                            flag=False
                            all_borders[feature_index][boundary]=bkj
                else:
                    all_borders[feature_index][boundary]=bkj
                    
            else:
                if (bkj_end>right_side_point):
                    all_borders[feature_index][boundary]=right_side_point
                    gain=right_side_point_gain
                    bkj=right_side_point
                    flag=True
                    while (flag):
                        right_side_point=bkj+alpha
                        all_borders[feature_index][boundary]=right_side_point

                        if (right_side_point<=bkj_end):
                            right_side_point_gain,pred_y=train_test_gain_now(train_x,train_y,
                                                                             all_borders,K_size,
                                                                             clas_pro,bin_counts,
                                                                             dev_x,dev_y,gain_matrix)
                           #do not put equal so it moves to onto one and another TT

                            if (gain>right_side_point_gain):
                                flag=False
                                all_borders[feature_index][boundary]=bkj
                            else:
                                bkj=right_side_point
                                gain=right_side_point_gain

                        else:
                            flag=False
                            all_borders[feature_index][boundary]=bkj
                else:
                    all_borders[feature_index][boundary]=bkj
                    
        else:
            all_borders[feature_index][boundary]=bkj


    #print("gain/best: {:4.3f} gain:{} best:{}  \n".format(gain/best,gain,best))
    theaccuracy = bayes_funcs.accuracy(pred_y,dev_y)
    #print("\naccuracy : {:4.3f}".format(theaccuracy))
    return all_borders,gain,theaccuracy
                            
def optimise_hill_climb_single(train_x,train_y,dev_x,dev_y,all_borders,gain,K_size,clas_pro,
                        bin_counts,gain_matrix,best,alpha=0.01,percent=0.5,limit_opti=-1,qindex=0):
    """
    this part is for optimising quantisation borders:
        we choose a feature and quant. border randomly:
            move border by -(alpha*M), then move it to otherside
            by alpha for 2*M times, 
            calculate decision rule with train_data, calculate expected gain with dev_data
        at the end pick highest expected gain
    """
    if limit_opti==-1:
        limit_opti=inf

    # populate all boundaries first, to select randomly or to go on until finishing them
    # passes first one and last one since they are infinite
    # [[1,2,3],[1,2,3,4,5]...]
    boundaries=[]
    for border in all_borders:
        boundaries.append([i+1 for i in range(len(border)-2)])
        
    #boundary_indexes=[i for i in range(len(all_borders))]
    #optimise only one given features

    feature_indexes=[qindex]
    
    # provided by previous function call
    # best = expected_gain(gain_matrix,dev_y,dev_y)
    while feature_indexes:
        #print ("gain {}".format(gain))
        #left_count=sum([len(alen) for alen in boundaries ])
        #if left_count%5==0:
        #    print ("left {} feature to optimise".format(left_count))
        #if left_count%7==0:
        #    print (all_borders)
        #     choose a feature randomly
        feature_index=random.choice(feature_indexes)

    #     choose a boundary randomly
        boundary=random.choice(boundaries[feature_index])
        boundaries[feature_index].remove(boundary)
        
    #   if feature do not have any other bounadry then remove it
        if boundaries[feature_index]==[]:
            feature_indexes.remove(feature_index)

    #   get value of the border,
        bkj=all_borders[feature_index][boundary]
# upper and lower bounds, all_borders includes inf as well, 
        upper_bound=all_borders[feature_index][boundary+1]
        lower_bound=all_borders[feature_index][boundary-1]

        if (upper_bound !=inf) and (lower_bound!=-inf):
            upper_distance=(upper_bound-bkj)*percent

            lower_distance=(bkj-lower_bound)*percent
            
        elif (upper_bound ==inf) and (lower_bound==-inf):
            upper_distance=(1-bkj)*percent
            
            lower_distance=(bkj-0)*percent
        elif (upper_bound ==inf):
            upper_distance=(1-bkj)*percent
            
            lower_distance=(bkj-lower_bound)*percent
        elif (lower_bound==-inf):
            upper_distance=(upper_bound-bkj)*percent

            lower_distance=(bkj-0)*percent

        bkj_end=bkj+upper_distance
        bkj_start=bkj-lower_distance
        
        random_point=random.uniform(bkj_start+alpha,bkj_end-alpha)        
        left_side_point = random_point-alpha
        right_side_point = random_point+alpha
        
        all_borders[feature_index][boundary]=left_side_point
        left_side_point_gain,pred_y=train_test_gain_now(train_x,train_y,all_borders,
                                                        K_size,clas_pro,bin_counts,
                                                        dev_x,dev_y,gain_matrix)
        
        all_borders[feature_index][boundary]=right_side_point
        right_side_point_gain,pred_y=train_test_gain_now(train_x,train_y,all_borders,
                                                         K_size,clas_pro,bin_counts,
                                                         dev_x,dev_y,gain_matrix)
        
        if (left_side_point_gain>gain) or (right_side_point_gain>gain):
            
            if left_side_point_gain>=right_side_point_gain:
                if (bkj_start<left_side_point):
                    all_borders[feature_index][boundary]=left_side_point
                    gain=left_side_point_gain
                    bkj=left_side_point
                    flag=True
                    while (flag):
                        left_side_point=bkj-alpha
                        all_borders[feature_index][boundary]=left_side_point

                        if (bkj_start<=left_side_point):
                            left_side_point_gain,pred_y=train_test_gain_now(train_x,train_y,
                                                                            all_borders,K_size,
                                                                            clas_pro,bin_counts,
                                                                            dev_x,dev_y,gain_matrix)
                           #do not put equal so it moves to onto one and another TT

                            if (gain>left_side_point_gain):
                                flag=False
                                all_borders[feature_index][boundary]=bkj
                            else:
                                bkj=left_side_point
                                gain=left_side_point_gain

                        else:
                            flag=False
                            all_borders[feature_index][boundary]=bkj
                else:
                    all_borders[feature_index][boundary]=bkj
                    
            else:
                if (bkj_end>right_side_point):
                    all_borders[feature_index][boundary]=right_side_point
                    gain=right_side_point_gain
                    bkj=right_side_point
                    flag=True
                    while (flag):
                        right_side_point=bkj+alpha
                        all_borders[feature_index][boundary]=right_side_point

                        if (right_side_point<=bkj_end):
                            right_side_point_gain,pred_y=train_test_gain_now(train_x,train_y,
                                                                             all_borders,K_size,
                                                                             clas_pro,bin_counts,
                                                                             dev_x,dev_y,gain_matrix)
                            #do not put equal so it moves to onto one and another TT

                            if (gain>right_side_point_gain):
                                flag=False
                                all_borders[feature_index][boundary]=bkj
                            else:
                                bkj=right_side_point
                                gain=right_side_point_gain

                        else:
                            flag=False
                            all_borders[feature_index][boundary]=bkj
                else:
                    all_borders[feature_index][boundary]=bkj
                    
        else:
            all_borders[feature_index][boundary]=bkj


    #print("gain/best: {:4.3f} gain:{} best:{}  \n".format(gain/best,gain,best))
    theaccuracy = bayes_funcs.accuracy(pred_y,dev_y)
    #print("\naccuracy : {:4.3f}".format(theaccuracy))
    return all_borders,gain,theaccuracy