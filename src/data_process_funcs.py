
import random
import pandas as pd
import numpy as np

from math import ceil,inf,floor
import time



def read_data(file_name="pr_data.txt",limit=-1,sampling=1):
# reads data file, loads x values to a data frame and y values to a list
# in each row of given data file, there are 5 floating points (features) and one binary number which is y values
# if given limit, it reads only given number of lines
# returns x values as a dataframe, y values are in a list
    data_file=open(file_name)
    alldata_x=[]
    alldata_y=[]
    a_line=data_file.readline()
    # K_size is a dictionary, keys are class names and values are number of elements in each class
    K_size={}
    #to keep the results same in each run
    while (a_line and limit!=0):
        if random.random()<sampling:
            limit-=1
            line=[float(x) for x in a_line.split()]
    #       features are stored as a tuple, will be dataframe later
            alldata_x.append(tuple(line[:-1]))
    #       y value
            clas=int(line[-1])
            alldata_y.append(clas)
    #       count number of elements in each class
            K_size.setdefault(clas,0)
            K_size[clas]+=1
    #       read next line
        a_line=data_file.readline()
#   
    data_file.close()
    data_x=pd.DataFrame(alldata_x)
    return data_x,alldata_y,K_size



def cal_entropies(range_boundaries,number_of_bins,data_x):
# returns a list of entropies,entropy for each feature
# entropy= -sum(p*log(p) for each bin)
# p = probability = (Number of elements in a bin)/(total elements)= m/N
# if there are zero counts in any bin then: entropy= -sum(p*log(p) for each bin) +( n0-1/2*N*loge(2))

# number_of_bins: Equal Interval Quantize each component to N/10 Levels
# since we are doing equal interval quantization, easiest way is normalising data points then 
# dividing it by range length of the dimension so that calculating corresponding bin of data point


    # number of data points 
    N=float(data_x.shape[0])
    results=[]
#     for each coloumn 
    for col_index in data_x:
#       boundaries provides upper and lower limits for each dimension
        bound=range_boundaries[col_index]
#     bin_len: fixed distance between two boundaries
        bin_len=(bound[1]-bound[0])/number_of_bins

#       normalise data points and divide by boundary_len to find which bin they are falling to
        norm_data=np.floor((data_x[col_index]-bound[0])/bin_len)
#     for each bin, how many elements falling to that particular bin
        counts=norm_data.value_counts()
#     probabilty of an element falling into a bin
        p=counts/N
#     entropy calculation explained at top of the function
        H=-(np.sum(p*np.log2(p)))
    
        n0=number_of_bins-counts.size
        
        if n0!=0:
            H+=(n0-1)/(2*N*np.log(2))
        
        results.append(H)  

    return results


def cal_bin_count(entropies,M):
    # calculates bin counts of each dimension according to entropies
    # f_entropy = an_entropy/total_entropy
    # bin_count = f_entropy**(1/)
    entropies=np.array(entropies)
    pro_ent=entropies/np.sum(entropies)
    bin_counts=np.ceil((M**pro_ent))
    return bin_counts


def cal_vorders(bin_counts,data_x):
    # version v1.1
    # retunrs borderds for all dimensions
    #  store borders list for each coloumn(feature) 
    all_borders=[]

    for col_index in data_x:
        # sort the coloumn, reset indexes 
        sorted_column=data_x[col_index].sort_values().reset_index(drop=True)
        # number of elements in the coloumn
        N=float(sorted_column.shape[0])
        # bin size of the feature(coloumn)
        bin_size=int(bin_counts[col_index])

        elements_in_abin=ceil(int(N/bin_size))

        borders=[-inf]
        # for a binsize=7,N=776, elements_in_abin=111
        # i=1,2,3,4,5,6
        # -inf,111,222,333,444,555,666,inf
        for i in range(1,bin_size):
            index=ceil(elements_in_abin*(i))
            border=sorted_column[index]
            borders.append(border)

        borders.append(inf)
        all_borders.append(borders[:])

    return all_borders


# following three functions are used to turn raw data to structured data points
# first each data point's each feature is quantised with given borders
# for each data point we get a tuple of corresponding quantised bins 
# data stored in a dictionary, keys are data point tuples, values are dictionary with clases are keys
#  and values are number of elements in that class for a data point
def quantizer(x,q_index,borders):
    bin_ad=-1
    # check each border, if data point is smaller than border return border number
    # if it is bigger than that, then move to next border, 
    for border in (borders):
        if border>x:
            return bin_ad
        else:
            bin_ad+=1

# address for a data point    
def return_adres(sample,all_borders):
    adres=[]
    # gets a row of dataframe, returns quantised addres
    for i,x in enumerate(sample):
        thebin=quantizer(x,i,all_borders[i])
        adres.append(thebin)
    adres=tuple(adres)
    return adres

def index_adress(adres,distent):
    index=0
    for i,k in enumerate(adres):
        index+=k*distent[i]
    return index

#  for each data 
def fill_memory_table(data_x,data_y,all_borders,clas_keys):
    # v1.3
    K_names=set(clas_keys)
    # number of classes
    K=len(K_names)
    # class_data_table={clas1:{(address1):2},...
    #                   clas2:{(address1):3,...}}
    class_data_table={}
    for clas in K_names:
        class_data_table.setdefault(clas,{})
#     print(alldata_x)
    for index, data_point in data_x.iterrows():
    
        adres=return_adres(data_point,all_borders)

        clas=data_y[index]
        class_data_table[clas].setdefault(adres,0)
        class_data_table[clas][adres]+=1.0
    return class_data_table

# supposed to be faster than original but slower, TODO check what I am missing
def fill_memory_table2(data_x,data_y,all_borders,clas_keys):
    # v1.2

    # returns quantised alldata_x

    # sort each coloumn in data_x, 
    # compare each item in a col. with first border
    # if it is smaller than border than that's its bin
    # if not, increse border index by until it is

    # t3=time.time()
    cols=[]
    for col_index in data_x:
        # sort the coloumn, reset indexes 
        # print("sort ")
        sorted_column=data_x[col_index].sort_values()

        col_border=all_borders[col_index]

        border_index=1
        bin_index=0
        border_value=col_border[border_index]
        # print("quant")
        for i,x in sorted_column.iteritems():
            while border_value<x:
                border_index+=1
                bin_index+=1
                border_value=col_border[border_index]
            sorted_column[i]=bin_index

        cols.append(sorted_column.copy())

    alldata_x_quantised=pd.concat(cols,axis=1)

    # t4=time.time()
    # print(t4-t3)
        # class_data_table={clas1:{(address1):2},...
    #                   clas2:{(address1):3,...}}
    class_data_table={}
    for clas in clas_keys:
        class_data_table.setdefault(clas,{})
    # print("fill dict ")
    # t1=time.time()

    for index, data_point in alldata_x_quantised.iterrows():
    
        # adres=return_adres(data_point,all_borders)
        adres=tuple(data_point)

        clas=data_y[index]
        class_data_table[clas].setdefault(adres,0)
        class_data_table[clas][adres]+=1.0
    # t2=time.time()
    # print("done_dict")
    # print(t2-t1)
    return class_data_table





























