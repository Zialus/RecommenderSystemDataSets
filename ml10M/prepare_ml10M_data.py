
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


data_file = maybe_download('ml-10m.zip', 65566137)


# In[3]:


get_ipython().system(u'unzip -o ml-10m.zip')


# In[4]:


# file should look like
'''
1::122::5::838985046
1::185::5::838983525
1::231::5::838983392
1::292::5::838983421
1::316::5::838983392
1::329::5::838983392
1::355::5::838984474
1::356::5::838983653
1::362::5::838984885
1::364::5::838983707
'''
m = 71567
n = 65133
nnz_train = 9000048
nnz_test = 1000006

user, item, rating = np.loadtxt('ml-10M100K/ratings.dat', delimiter='::',
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

user_item_train, user_item_test, rating_train, rating_test = train_test_split(user_item.T,
                                                                              rating,
                                                                              test_size=nnz_test,
                                                                              random_state=42)


# In[7]:


R_test_coo = sparse.coo_matrix((rating_test, (user_item_test[:, 0], user_item_test[:, 1])))
assert R_test_coo.nnz == nnz_test

outfile_test = open("test.txt", 'w')
for i in range(nnz_test):
    outfile_test.write(str((user_item_test[i, 0])) + " " + str((user_item_test[i, 1])) + " " + str(rating_test[i]) + "\n")


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


R_train_coo = sparse.coo_matrix((rating_train, (user_item_train[:, 0], user_item_train[:, 1])))
assert R_train_coo.nnz == nnz_train

outfile_train = open("train.txt", 'w')
for i in range(nnz_train):
    outfile_train.write(str((user_item_train[i, 0])) + " " + str((user_item_train[i, 1])) + " " + str(rating_train[i]) + "\n")


# In[12]:


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


# In[13]:


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


# In[14]:


print("writing extra meta_modified_all file")

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


# In[15]:


print("writing extra meta file")

outfile_meta = open("meta", 'w')
outfile_meta.write(str(m) + " " + str(n) + "\n")
outfile_meta.write(str(nnz_train) + " " + "train.txt\n")
outfile_meta.write(str(nnz_test) + " " + "test.txt\n")

