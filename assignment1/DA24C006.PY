import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## loading the data
data=pd.read_csv("Default Dataset.csv", header=None)
data.columns=pd.Index(['x', 'y'])


# Data Cleaning 

# All values in both x and y axis are between 0 and 1000. In order to make a 1000x1000 matrix the steps I will follow are
# as follows:
# 1) Round the data to the nearest integer. Drop the duplicate rows.
# 2) Discretize the data. Create a 1000x1000 matrix of zeros.
# 3) The entries of the transformed data are a tuple (x, y) where 0<x,y<1000, look at each entry of the dataset, go to 
#    the (x, y)th entry of the matrix and replace 0 by 1.
# 5) Sparse matrix will be created.

data_trans=data.round(0).drop_duplicates().astype(dtype=int)
data_matrix=np.zeros(shape=(1000, 1000))
for idx, coordinate in data_trans.iterrows():
    data_matrix[coordinate['x'], coordinate['y']]=1


### Creating a function that converts data matrix to x,y pairs data (dense representation)
def matrix_to_df(matrix):
    x_max, y_max=matrix.shape
    df_data=[]
    for i in range(x_max):
        for j in range(y_max):
            if (matrix[i, j]==1):
                df_data.append([i, j])
    return pd.DataFrame(df_data, columns=['x', 'y'])


# trans_flip_op is the matrix that is required to flip some matrix horizontally
trans_flip_op=np.zeros(shape=(1000, 1000), dtype=int)
for i in range(1000):
    trans_flip_op[999-i, i]=1

# flipping the matrix horizontally and obtain the dense representation
trans_filp_res_matrix=np.matmul(data_matrix, trans_flip_op)
trans_flip_res_data=matrix_to_df(trans_filp_res_matrix)

# rotating the matrix by 90 degrees and obtaining the dense representation
trans_rot_res_matrix=np.matmul(data_matrix.T, trans_flip_op)
trans_rot_res_data=matrix_to_df(trans_rot_res_matrix)

# Vizualizing the results
fig, axes=plt.subplots(ncols=2, nrows=2)
fig.set_size_inches(w=10, h=10)
axes[0, 0].scatter(data_trans['x'], data_trans['y'])
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('y')
axes[0, 0].set_title('Original Image')

axes[0, 1].scatter(trans_flip_res_data['x'], trans_flip_res_data['y'])
axes[0, 1].set_xlabel('x')
axes[0, 1].set_ylabel('y')
axes[0, 1].set_title('Flipped Horizontally')

axes[1, 0].scatter(trans_rot_res_data['x'], trans_rot_res_data['y'])
axes[1, 0].set_xlabel('x')
axes[1, 0].set_ylabel('y')
axes[1, 0].set_title('Rotated 90 degrees')
fig.tight_layout()
plt.show()