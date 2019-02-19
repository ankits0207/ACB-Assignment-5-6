# Ankit Sharma
# MT16121
# Python 2.7

import csv
import math
import numpy as np

# Reading a file and creating a list of lists from it
my_file = open('cheung_count_table.txt', 'r')
lines = my_file.readlines()
i = 0
out_list = []
for line in lines:
    if i != 0:
        list_of_elements = line.strip().split()
        in_list = []
        for j in range(1, len(list_of_elements)):
            in_list.append(int(list_of_elements[j]))
        out_list.append(in_list)
    i += 1

# Performing geometric mean normalization over read counts where in each read count is divided by the geometric
# mean of the gene corresponding to that read count across all samples
gm_prod_list = []
for in_list in out_list:
    prod = 1
    for elt in in_list:
        prod *= elt
    gm_prod_list.append(prod)
g_mean_list = []
for gm in gm_prod_list:
    g_mean_list.append(gm**(1/len(out_list[0])*1.0))
for i in range(0, len(out_list)):
    for j in range(0, len(out_list[i])):
        out_list[i][j] = out_list[i][j]/g_mean_list[i]

# Performing log transform over the data
log_transform_list = []
for inner_list in out_list:
    inner_list_of_elts = []
    for element in inner_list:
        inner_list_of_elts.append(math.log(element + 1, 2))
    log_transform_list.append(inner_list_of_elts)

# Writing the log-normalized data to a csv file
with open("log_norm_2.csv","w") as f:
    wr = csv.writer(f)
    wr.writerows(log_transform_list)

my_numpy_array = np.array(out_list)
np.save('My_Log_Normalized_Numpy_Array_Task_2', my_numpy_array)
print 'File written successfully - log_norm_2.csv'
