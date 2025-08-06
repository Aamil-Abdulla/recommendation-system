#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
from surprise import Dataset, Reader , SVD , accuracy
from surprise.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error , mean_absolute_error
from tensorflow.keras.layers import Input , Embedding , Flatten , Dense , Concatenate
from tensorflow.keras.models import Model 
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')


# In[6]:


import pandas as pd
from surprise import Dataset , SVD , Reader , accuracy


# In[27]:


from scipy.sparse import csr_matrix


# In[28]:


data = Dataset.load_builtin('ml-100k')
raw_ratings = data.raw_ratings
df = pd.DataFrame(raw_ratings , columns=["userId" , "itemId" , "rating","timestamp"])
df.head()


# In[30]:


user_item_matrix = df.pivot(index="userId", columns="itemId", values="rating").fillna(0)
csr_data = csr_matrix(user_item_matrix.values)


# In[ ]:





# In[8]:


from surprise.model_selection import train_test_split
trainset , testset = train_test_split(data , test_size=0.2 , random_state=42)


# In[9]:


model_svd = SVD()
model_svd.fit(trainset)


# In[11]:


pred = model_svd.test(testset)


# In[14]:


print(accuracy.rmse(pred))
print(accuracy.mae(pred))


# In[21]:


prediction = model_svd.predict(uid='196',iid=302)
print(prediction)


# In[22]:


print(f"\nPredicted rating by user 196 for item 302: {prediction.est:.2f}")


# In[ ]:




