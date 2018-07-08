
# coding: utf-8

# In[1]:


import os
from six.moves import urllib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split


# In[2]:


get_ipython().run_cell_magic('time', '', '# Download the data.\nurl = \'http://files.grouplens.org/datasets/movielens/\'\n\n\ndef maybe_download(filename, expected_bytes):\n    """Download a file if not present, and make sure it\'s the right size."""\n    if not os.path.exists(filename):\n        filename, _ = urllib.request.urlretrieve(url + filename, filename)\n    statinfo = os.stat(filename)\n    if statinfo.st_size == expected_bytes:\n        print(\'Found and verified\', filename)\n    else:\n        print(statinfo.st_size)\n        raise Exception(\'Failed to verify \' + filename + \'. Can you get to it with a browser?\')\n    return filename\n\n\ndata_file = maybe_download(\'ml-20m.zip\', 198702078)')


# In[3]:


get_ipython().run_cell_magic('time', '', "get_ipython().system(u'unzip -o ml-20m.zip')")


# In[4]:


get_ipython().run_cell_magic('time', '', "# file should look like\n'''\nuserId,movieId,rating,timestamp\n1,2,3.5,1112486027\n1,29,3.5,1112484676\n1,32,3.5,1112484819\n1,47,3.5,1112484727\n1,50,3.5,1112484580\n1,112,3.5,1094785740\n1,151,4.0,1094785734\n1,223,4.0,1112485573\n1,253,4.0,1112484940\n'''\nm = 138493\nn = 131262\nnnz_train = 18000236\nnnz_test = 2000027\n\ndata_filename = 'ml-20m/ratings.csv'\n\ndata = pd.read_csv(data_filename, dtype={0:'int32',1:'int32',2:'float32'},usecols=[0,1,2])\n\nuser = data['userId'].values\nitem = data['movieId'].values\nrating = data['rating'].values")


# In[5]:


print(user)
print(item)
print(rating)
print("")
print(np.min(user))
print(np.min(item))
print(np.min(rating))
print("")
print(np.max(user))
print(np.max(item))
print(np.max(rating))
print("")
print(np.unique(user).size)
print(np.unique(item).size)
print(np.unique(rating).size)
print("")
print(user.size)

assert np.max(user) == m
assert np.max(item) == n
assert user.size == nnz_train + nnz_test


# In[6]:


get_ipython().run_cell_magic('time', '', 'user_item = np.vstack((user, item))\n\nuser_item_train, user_item_test, rating_train, rating_test = train_test_split(user_item.T,\n                                                                              rating,\n                                                                              test_size=nnz_test,\n                                                                              random_state=42)')


# In[7]:


get_ipython().run_cell_magic('time', '', 'R_test_coo = sparse.coo_matrix((rating_test, (user_item_test[:, 0], user_item_test[:, 1])))\nassert R_test_coo.nnz == nnz_test\n\noutfile_test = open("test.txt", \'w\')\nfor i in range(nnz_test):\n    outfile_test.write(str((user_item_test[i, 0])) + " " + str((user_item_test[i, 1])) + " " + str(rating_test[i]) + "\\n")')


# In[8]:


get_ipython().run_cell_magic('time', '', "# for test data, we need COO format to calculate test RMSE\n\nR_test_coo.data.astype(np.float32).tofile('R_test_coo.data.bin')\nR_test_coo.row.tofile('R_test_coo.row.bin')\nR_test_coo.col.tofile('R_test_coo.col.bin')\n\ntest_data = np.fromfile('R_test_coo.data.bin', dtype=np.float32)\ntest_row = np.fromfile('R_test_coo.row.bin', dtype=np.int32)\ntest_col = np.fromfile('R_test_coo.col.bin', dtype=np.int32)")


# In[9]:


print(R_test_coo.data)
print(R_test_coo.row)
print(R_test_coo.col)
print("")
print(test_data)
print(test_row)
print(test_col)


# In[10]:


print(np.max(R_test_coo.data))
print(np.max(R_test_coo.row))
print(np.max(R_test_coo.col))
print("")
print(np.min(R_test_coo.data))
print(np.min(R_test_coo.row))
print(np.min(R_test_coo.col))
print("")
print(np.unique(user).size)
print(np.unique(R_test_coo.row).size)
print(np.unique(item).size)
print(np.unique(R_test_coo.col).size)


# In[11]:


get_ipython().run_cell_magic('time', '', 'R_train_coo = sparse.coo_matrix((rating_train, (user_item_train[:, 0], user_item_train[:, 1])))\nassert R_train_coo.nnz == nnz_train\n\noutfile_train = open("train.txt", \'w\')\nfor i in range(nnz_train):\n    outfile_train.write(str((user_item_train[i, 0])) + " " + str((user_item_train[i, 1])) + " " + str(rating_train[i]) + "\\n")')


# In[12]:


get_ipython().run_cell_magic('time', '', "# for training data, we need COO format to calculate training RMSE\n# we need CSR format R when calculate X from \\Theta\n# we need CSC format of R when calculating \\Theta from X\nR_train_coo.data.astype(np.float32).tofile('R_train_coo.data.bin')\nR_train_coo.row.tofile('R_train_coo.row.bin')\nR_train_coo.col.tofile('R_train_coo.col.bin')\n\nR_train_csr = R_train_coo.tocsr()\nR_train_csc = R_train_coo.tocsc()\n\nR_train_csr.data.astype(np.float32).tofile('R_train_csr.data.bin')\nR_train_csr.indices.tofile('R_train_csr.indices.bin')\nR_train_csr.indptr.tofile('R_train_csr.indptr.bin')\nR_train_csc.data.astype(np.float32).tofile('R_train_csc.data.bin')\nR_train_csc.indices.tofile('R_train_csc.indices.bin')\nR_train_csc.indptr.tofile('R_train_csc.indptr.bin')")


# In[13]:


get_ipython().run_cell_magic('time', '', "train_data = np.fromfile('R_train_coo.data.bin', dtype=np.float32)\ntrain_row = np.fromfile('R_train_coo.row.bin', dtype=np.int32)\ntrain_col = np.fromfile('R_train_coo.col.bin', dtype=np.int32)\n\ntrain_csc_data = np.fromfile('R_train_csc.data.bin', dtype=np.float32)\ntrain_csc_indices = np.fromfile('R_train_csc.indices.bin', dtype=np.int32)\ntrain_csc_indptr = np.fromfile('R_train_csc.indptr.bin', dtype=np.int32)\n\ntrain_csr_data = np.fromfile('R_train_csr.data.bin', dtype=np.float32)\ntrain_csr_indices = np.fromfile('R_train_csr.indices.bin', dtype=np.int32)\ntrain_csr_indptr = np.fromfile('R_train_csr.indptr.bin', dtype=np.int32)")


# In[14]:


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


# In[15]:


print(np.max(R_train_coo.data))
print(np.max(R_train_coo.row))
print(np.max(R_train_coo.col))
print("")
print(np.min(R_train_coo.data))
print(np.min(R_train_coo.row))
print(np.min(R_train_coo.col))
print("")
print(np.unique(user).size)
print(np.unique(R_train_coo.row).size)
print(np.unique(item).size)
print(np.unique(R_train_coo.col).size)


# In[16]:


get_ipython().run_cell_magic('time', '', 'print("writing extra meta_modified_all file")\n\noutfile_meta = open("meta_modified_all", \'w\')\noutfile_meta.write(str(m) + " " + str(n) + "\\n" + str(nnz_train) + "\\n")\noutfile_meta.write("""R_train_coo.data.bin\nR_train_coo.row.bin\nR_train_coo.col.bin\nR_train_csr.indptr.bin\nR_train_csr.indices.bin\nR_train_csr.data.bin\nR_train_csc.indptr.bin\nR_train_csc.indices.bin\nR_train_csc.data.bin\n""")\noutfile_meta.write(str(nnz_test) + " " + "test.txt\\n")')


# In[17]:


get_ipython().run_cell_magic('time', '', 'print("writing extra meta file")\n\noutfile_meta = open("meta", \'w\')\noutfile_meta.write(str(m) + " " + str(n) + "\\n")\noutfile_meta.write(str(nnz_train) + " " + "train.txt\\n")\noutfile_meta.write(str(nnz_test) + " " + "test.txt\\n")')

