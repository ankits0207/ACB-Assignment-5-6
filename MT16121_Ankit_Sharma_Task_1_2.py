import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

testsize = 0.33
my_numpy_array_x = np.load('My_Log_Normalized_Numpy_Array_Task_1.npy')
my_numpy_array_x = np.transpose(my_numpy_array_x)

# Reading a file and creating a list of lists from it
my_file = open('Merged_output_file.txt', 'r')
lines = my_file.readlines()
out_list = []
for line in lines:
    list_of_elements = line.strip().split()
    in_list = []
    for j in range(0, len(list_of_elements)):
        in_list.append(int(list_of_elements[j]))
    out_list.append(in_list)
my_numpy_array_y = np.array(out_list)

model = TSNE(n_components=2, random_state=0)
x_transformed = model.fit_transform(my_numpy_array_x)
plt.title('TSNE visualization - ' + 'Task1')
plt.scatter(x_transformed[:, 0], x_transformed[:, 1], my_numpy_array_y.shape[0], c = my_numpy_array_y)
plt.show()

model = PCA(n_components=2)
x_transformed = model.fit_transform(my_numpy_array_x)
plt.title('PCA visualization - ' + 'Task1')
plt.scatter(x_transformed[:, 0], x_transformed[:, 1], c = my_numpy_array_y)
plt.show()

acc_mat = []
for i in range(0, 100):
    x_train, x_test, y_train, y_test = train_test_split(my_numpy_array_x, my_numpy_array_y, test_size=testsize, random_state= 0)
    k_means = KMeans(n_clusters=2, random_state=0).fit(x_train)
    prediction = k_means.predict(x_test)
    acc_mat.append(accuracy_score(prediction, y_test)*100)
    # print 'Iteration ' + str(i)
tot_acc = 0
for a in acc_mat:
    tot_acc += a
print 'Averaged accuracy for KMeans ' + str(tot_acc*1.0/100)
