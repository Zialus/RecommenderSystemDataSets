
# coding: utf-8

# In[1]:


import os
from six.moves import urllib
import numpy as np
from scipy import sparse
from sklearn.model_selection import train_test_split


# In[2]:


# Download the data.
url = 'http://files.grouplens.org/datasets/movielens/'


def maybe_download(filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception('Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename


data_file = maybe_download('ml-20m.zip', 198702078)


# In[3]:


get_ipython().system(u'unzip -o ml-20m.zip')


# In[4]:


# file should look like
'''
userId,movieId,rating,timestamp
1,2,3.5,1112486027
1,29,3.5,1112484676
1,32,3.5,1112484819
1,47,3.5,1112484727
1,50,3.5,1112484580
1,112,3.5,1094785740
1,151,4.0,1094785734
1,223,4.0,1112485573
1,253,4.0,1112484940
'''
m = 138493
n = 131262
nnz_train = 18000236
nnz_test = 2000027

user, item, rating = np.loadtxt('ml-20m/ratings.csv', delimiter=',', skiprows=1,
                                dtype=[('f0', np.int32), ('f1', np.int32), ('f2', np.float)],
                                unpack=True)


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


user_item = np.vstack((user, item))

user_item_train, user_item_test, rating_train, rating_test = train_test_split(user_item.T, rating, test_size=2000027, random_state=42)


# In[7]:


# 1-based to 0-based
R_test_coo = sparse.coo_matrix((rating_test, (user_item_test[:, 0] - 1, user_item_test[:, 1] - 1)))
assert R_test_coo.nnz == nnz_test

outfile_test = open("test.txt", 'w')
for i in range(nnz_test):
    outfile_test.write(str((user_item_test[i, 0] - 1)) + " " + str((user_item_test[i, 1] - 1)) + " " + str(rating_test[i]) + "\n")


# In[8]:


# for test data, we need COO format to calculate test RMSE

R_test_coo.data.astype(np.float32).tofile('R_test_coo.data.bin')
R_test_coo.row.tofile('R_test_coo.row.bin')
R_test_coo.col.tofile('R_test_coo.col.bin')

test_data = np.fromfile('R_test_coo.data.bin', dtype=np.float32)
test_row = np.fromfile('R_test_coo.row.bin', dtype=np.int32)
test_col = np.fromfile('R_test_coo.col.bin', dtype=np.int32)


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


# 1-based to 0-based
R_train_coo = sparse.coo_matrix((rating_train, (user_item_train[:, 0] - 1, user_item_train[:, 1] - 1)))
assert R_train_coo.nnz == nnz_train

outfile_train = open("train.txt", 'w')
for i in range(nnz_train):
    outfile_train.write(str((user_item_train[i, 0] - 1)) + " " + str((user_item_train[i, 1] - 1)) + " " + str(rating_train[i]) + "\n")


# In[ ]:


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


# In[ ]:


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
print(np.unique(user).size)
print(np.unique(R_train_coo.row).size)
print(np.unique(item).size)
print(np.unique(R_train_coo.col).size)


# In[ ]:


print("write extra meta file")

outfile_meta = open("meta_modified_all", 'w')
outfile_meta.write(str(m) + "" + str(n) + "\n" + str(nnz_train) + "\n")
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
outfile_meta.write(str(nnz_test) + "" + "test.txt\n")

