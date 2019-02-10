
# coding: utf-8

# In[1]:


import numpy as np
from scipy import sparse


# In[2]:


# netflix_mm and netflix_mme should look like this
'''
1 1  3
2 1  5
3 1  4
5 1  3
6 1  3
7 1  4
8 1  3
'''

m = 480189
n = 17770
nnz_train = 99072112
nnz_test = 1408395

train_data_file = "netflix_mm"
test_data_file = "netflix_mme"

print("preparing test data")
test_user, test_item, test_rating = np.loadtxt(test_data_file, dtype=np.int32, unpack=True)

print("preparing training data")
train_user, train_item, train_rating = np.loadtxt(train_data_file, dtype=np.int32, unpack=True)


# In[3]:


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


# In[4]:


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


# In[5]:


R_test_coo = sparse.coo_matrix((test_rating, (test_user, test_item)))
assert R_test_coo.nnz == nnz_test

outfile_test = open("test.txt", 'w')
for i in range(nnz_test):
    outfile_test.write(str(test_user[i]) + " " + str(test_item[i]) + " " + str(test_rating[i]) + "\n")


# In[6]:


# for test data, we need COO format to calculate test RMSE

R_test_coo.data.astype(np.float32).tofile('R_test_coo.data.bin')
R_test_coo.row.tofile('R_test_coo.row.bin')
R_test_coo.col.tofile('R_test_coo.col.bin')

test_data = np.fromfile('R_test_coo.data.bin', dtype=np.float32)
test_row = np.fromfile('R_test_coo.row.bin', dtype=np.int32)
test_col = np.fromfile('R_test_coo.col.bin', dtype=np.int32)


# In[7]:


print(R_test_coo.data)
print(R_test_coo.row)
print(R_test_coo.col)
print("")
print(test_data)
print(test_row)
print(test_col)


# In[8]:


print(np.max(R_test_coo.data))
print(np.max(R_test_coo.row))
print(np.max(R_test_coo.col))
print("")
print(np.min(R_test_coo.data))
print(np.min(R_test_coo.row))
print(np.min(R_test_coo.col))
print("")
print(np.unique(test_user).size)
print(np.unique(R_test_coo.row).size)
print(np.unique(test_item).size)
print(np.unique(R_test_coo.col).size)


# In[9]:


get_ipython().run_cell_magic('time', '', 'R_train_coo = sparse.coo_matrix((train_rating, (train_user, train_item)))\nassert R_train_coo.nnz == nnz_train\n\noutfile_train = open("train.txt", \'w\')\nfor i in range(nnz_train):\n    outfile_train.write(str(train_user[i]) + " " + str(train_item[i]) + " " + str(train_rating[i]) + "\\n")')


# In[10]:


get_ipython().run_cell_magic('time', '', "# for training data, we need COO format to calculate training RMSE\n# we need CSR format R when calculate X from \\Theta\n# we need CSC format of R when calculating \\Theta from X\nR_train_coo.data.astype(np.float32).tofile('R_train_coo.data.bin')\nR_train_coo.row.tofile('R_train_coo.row.bin')\nR_train_coo.col.tofile('R_train_coo.col.bin')\n\nR_train_csr = R_train_coo.tocsr()\nR_train_csc = R_train_coo.tocsc()\n\nR_train_csr.data.astype(np.float32).tofile('R_train_csr.data.bin')\nR_train_csr.indices.tofile('R_train_csr.indices.bin')\nR_train_csr.indptr.tofile('R_train_csr.indptr.bin')\nR_train_csc.data.astype(np.float32).tofile('R_train_csc.data.bin')\nR_train_csc.indices.tofile('R_train_csc.indices.bin')\nR_train_csc.indptr.tofile('R_train_csc.indptr.bin')")


# In[11]:


get_ipython().run_cell_magic('time', '', "train_data = np.fromfile('R_train_coo.data.bin', dtype=np.float32)\ntrain_row = np.fromfile('R_train_coo.row.bin', dtype=np.int32)\ntrain_col = np.fromfile('R_train_coo.col.bin', dtype=np.int32)\n\ntrain_csc_data = np.fromfile('R_train_csc.data.bin', dtype=np.float32)\ntrain_csc_indices = np.fromfile('R_train_csc.indices.bin', dtype=np.int32)\ntrain_csc_indptr = np.fromfile('R_train_csc.indptr.bin', dtype=np.int32)\n\ntrain_csr_data = np.fromfile('R_train_csr.data.bin', dtype=np.float32)\ntrain_csr_indices = np.fromfile('R_train_csr.indices.bin', dtype=np.int32)\ntrain_csr_indptr = np.fromfile('R_train_csr.indptr.bin', dtype=np.int32)")


# In[12]:


print(R_train_coo.data)
print(R_train_coo.row)
print(R_train_coo.col)
print("")
print(train_data)
print(train_row)
print(train_col)
print("")
print(R_train_csr.data)
print(R_train_csr.indices)
print(R_train_csr.indptr)
print("")
print(train_csr_data)
print(train_csr_indices)
print(train_csr_indptr)
print("")
print(R_train_csc.data)
print(R_train_csc.indices)
print(R_train_csc.indptr)
print("")
print(train_csc_data)
print(train_csc_indices)
print(train_csc_indptr)


# In[ ]:


print(np.max(R_train_coo.data))
print(np.max(R_train_coo.row))
print(np.max(R_train_coo.col))
print("")
print(np.min(R_train_coo.data))
print(np.min(R_train_coo.row))
print(np.min(R_train_coo.col))
print("")
print(np.unique(train_user).size)
print(np.unique(R_train_coo.row).size)
print(np.unique(train_item).size)
print(np.unique(R_train_coo.col).size)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'print("write extra meta file")\n\noutfile_meta = open("meta_modified_all", \'w\')\noutfile_meta.write(str(m) + " " + str(n) + "\\n" + str(nnz_train) + "\\n")\noutfile_meta.write("""R_train_coo.data.bin\nR_train_coo.row.bin\nR_train_coo.col.bin\nR_train_csr.indptr.bin\nR_train_csr.indices.bin\nR_train_csr.data.bin\nR_train_csc.indptr.bin\nR_train_csc.indices.bin\nR_train_csc.data.bin\n""")\noutfile_meta.write(str(nnz_test) + " " + "test.txt\\n")')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'print("writing extra meta file")\n\noutfile_meta = open("meta", \'w\')\noutfile_meta.write(str(m) + " " + str(n) + "\\n")\noutfile_meta.write(str(nnz_train) + " " + "train.txt\\n")\noutfile_meta.write(str(nnz_test) + " " + "test.txt\\n")')

