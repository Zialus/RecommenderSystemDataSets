
# coding: utf-8

# In[1]:


import numpy as np
from scipy import sparse


# In[2]:


train_data_file = "jester_train.csv"
test_data_file = "dont_use.csv"


# In[3]:


# jester file should look like
'''
7302,29,7.156
61815,46,6.375
31128,96,2.281
36125,147,-1.781
18007,60,2.188
7387,99,3.594
12007,18,-2.094
'''

m = 63978
n = 150
nnz_train = 1000000
nnz_test = 761439

print("preparing test data")
test_user, test_item, test_rating = np.loadtxt(test_data_file, delimiter=',',
                                               dtype=[('f0', np.int32), ('f1', np.int32), ('f2', np.float)],
                                               skiprows=1, unpack=True)

print("preparing training data")
train_user, train_item, train_rating = np.loadtxt(train_data_file, delimiter=',',
                                                  dtype=[('f0', np.int32), ('f1', np.int32), ('f2', np.float)],
                                                  skiprows=1, unpack=True)


# In[4]:


print(test_user)
print(test_item)
print(test_rating)
print("")
print(np.max(test_user))
print(np.max(test_item))
print(np.max(test_rating))
print("")
print(np.min(test_user))
print(np.min(test_item))
print(np.min(test_rating))
print("")
print(np.unique(test_user).size)
print(np.unique(test_item).size)
print(np.unique(test_rating).size)
print("")
print(test_user.size)

assert test_user.size == nnz_test


# In[5]:


print(train_user)
print(train_item)
print(train_rating)
print("")
print(np.max(train_user))
print(np.max(train_item))
print(np.max(train_rating))
print("")
print(np.min(train_user))
print(np.min(train_item))
print(np.min(train_rating))
print("")
print(np.unique(train_user).size)
print(np.unique(train_item).size)
print(np.unique(train_rating).size)
print("")
print(train_user.size)

assert train_user.size == nnz_train
assert np.max(train_user) == m
assert np.max(train_item) == n


# In[6]:


# 1-based to 0-based
R_test_coo = sparse.coo_matrix((test_rating, (test_user - 1, test_item - 1)))
assert R_test_coo.nnz == nnz_test

outfile_test = open("test.txt", 'w')
for i in range(nnz_test):
    outfile_test.write(str(test_user[i] - 1) + " " + str(test_item[i] - 1) + " " + str(test_rating[i]) + "\n")


# In[7]:


# for test data, we need COO format to calculate test RMSE

R_test_coo.data.astype(np.float32).tofile('R_test_coo.data.bin')
R_test_coo.row.tofile('R_test_coo.row.bin')
R_test_coo.col.tofile('R_test_coo.col.bin')

test_data = np.fromfile('R_test_coo.data.bin', dtype=np.float32)
test_row = np.fromfile('R_test_coo.row.bin', dtype=np.int32)
test_col = np.fromfile('R_test_coo.col.bin', dtype=np.int32)


# In[8]:


print(R_test_coo.data)
print(R_test_coo.row)
print(R_test_coo.col)
print("")
print(test_data)
print(test_row)
print(test_col)


# In[9]:


# 1-based to 0-based
R_train_coo = sparse.coo_matrix((train_rating, (train_user - 1, train_item - 1)))
assert R_train_coo.nnz == nnz_train

outfile_train = open("train.txt", 'w')
for i in range(nnz_train):
    outfile_train.write(str(train_user[i] - 1) + " " + str(train_item[i] - 1) + " " + str(train_rating[i]) + "\n")


# In[10]:


# for training data, we need COO format to calculate training RMSE
# we need CSR format R when calculate X from \Theta
# we need CSC format of R when calculating \Theta from X
R_train_coo.data.astype(np.float32).tofile('R_train_coo.data.bin')
R_train_coo.row.tofile('R_train_coo.row.bin')
R_train_coo.col.tofile('R_train_coo.col.bin')

train_data = np.fromfile('R_train_coo.data.bin', dtype=np.float32)
train_row = np.fromfile('R_train_coo.row.bin', dtype=np.int32)
train_col = np.fromfile('R_train_coo.col.bin', dtype=np.int32)

R_train_csr = R_train_coo.tocsr()
R_train_csc = R_train_coo.tocsc()

R_train_csr.data.astype(np.float32).tofile('R_train_csr.data.bin')
R_train_csr.indices.tofile('R_train_csr.indices.bin')
R_train_csr.indptr.tofile('R_train_csr.indptr.bin')
R_train_csc.data.astype(np.float32).tofile('R_train_csc.data.bin')
R_train_csc.indices.tofile('R_train_csc.indices.bin')
R_train_csc.indptr.tofile('R_train_csc.indptr.bin')

train_csc = np.fromfile('R_train_csc.data.bin', dtype=np.float32)
train_csr = np.fromfile('R_train_csr.data.bin', dtype=np.float32)


# In[11]:


print(np.max(R_train_coo.data))
print(np.max(R_train_coo.row))
print(np.max(R_train_coo.col))
print("")
print(np.min(R_train_coo.data))
print(np.min(R_train_coo.row))
print(np.min(R_train_coo.col))
print("")
print(R_train_coo.data)
print(R_train_coo.row)
print(R_train_coo.col)
print("")
print(train_data)
print(train_row)
print(train_col)
print("")
print(R_train_csr.data)
print(R_train_csr.indptr)
print(R_train_csr.indices)
print("")
print(train_csr)
print("")
print(R_train_csc.data)
print(R_train_csc.indptr)
print(R_train_csc.indices)
print("")
print(train_csc)
print("")
print(np.unique(train_user).size)
print(np.unique(R_train_coo.row).size)
print(np.unique(train_item).size)
print(np.unique(R_train_coo.col).size)


# In[12]:


print("write extra meta file")

outfile_meta = open("meta_modified_all", 'w')
outfile_meta.write(str(m) + " " + str(n) + "\n" + str(nnz_train) + "\n")
outfile_meta.write("""R_train_coo.data.bin
R_train_coo.row.bin
R_train_coo.col.bin
R_train_csr.indptr.bin
R_train_csr.indices.bin
R_train_csr.data.bin
R_train_csc.indptr.bin
R_train_csc.indices.bin
R_train_csc.data.bin
""")
outfile_meta.write(str(nnz_test) + " " + "test.txt\n")

