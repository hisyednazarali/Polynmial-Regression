#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df =pd.read_csv("D:\\Data Science\\6.Data Cleaning\\claimants.csv")
df.head()


# In[3]:


df.info()


# In[4]:


df.isnull().sum()


# In[5]:


df['CLMAGE'].mean()


# In[6]:


df['CLMAGE'].fillna(28.414, inplace = True)


# In[7]:


df.isnull().sum()


# In[8]:


from sklearn.impute import SimpleImputer


# In[9]:


mean_imputer =SimpleImputer(strategy='mean')
median_imputer=SimpleImputer(strategy ='median')
mode_imputer = SimpleImputer(strategy='most_frequent')


# In[10]:


df['CLMAGE'] = pd.DataFrame(mean_imputer.fit_transform(df[["CLMAGE"]]))
df['CLMAGE'].isnull().sum()


# In[11]:


df['CLMAGE'].median()


# In[12]:


df = pd.read_csv("D:\\Data Science\\6.Data Cleaning\\claimants.csv")
df


# In[13]:


df.isnull().sum()


# In[14]:


df['CLMAGE'].median()


# In[15]:


df["CLMAGE"] = pd.DataFrame(median_imputer.fit_transform(df[["CLMAGE"]]))
df.isnull().sum()


# In[16]:


df['CLMSEX'] =pd.DataFrame(mode_imputer.fit_transform(df[['CLMSEX']]))


# In[17]:


df['CLMSEX'].isnull().sum()


# In[18]:


df['CLMINSUR'] =pd.DataFrame(mode_imputer.fit_transform(df[['CLMINSUR']]))


# In[19]:


df['SEATBELT'] =pd.DataFrame(mode_imputer.fit_transform(df[['SEATBELT']]))


# In[20]:


df.isnull().sum()


# In[21]:


import seaborn as sns
from scipy import stats


# In[22]:


dataset = [11,10,12,14,12,15,14,13,15,102,12,14,17,19,107,10,13,12,14,12,108,12,14,12]


# In[23]:


outliers =[]
def detect_outliers(data):
    for i in data:
        z_score = (i-np.mean(data))/np.std(data)
        if np.abs(z_score)>2:
            outliers.append(i)
    return outliers


# In[24]:


detect_outliers(dataset)


# In[25]:


l =[]
for x in dataset:
    z_score =(x-np.mean(dataset))/np.std(dataset)
    l.append(z_score)


# In[26]:


print(l)


# In[27]:


sorted(dataset)


# In[28]:


q1 = np.percentile(dataset ,25)
q3 = np.percentile(dataset,75)


# In[29]:


q1


# In[30]:


q3


# In[31]:


iqr = q3-q1
print(iqr)


# In[32]:


lower_bound_val = q1-1.5*iqr
upper_bound_val = q3+1.5*iqr
print(lower_bound_val,upper_bound_val)


# In[33]:


sns.boxplot(dataset)
plt.show()


# In[34]:


boston = pd.read_csv("D:\\Data Science\\6.Data Cleaning\\boston.csv")
boston


# In[35]:


boston.info()


# In[36]:


boston.describe()


# In[37]:


boston.head(3)


# In[38]:


sns.boxplot(boston.RM)
plt.title("BOXPLOT")
plt.grid()
plt.show()


# In[39]:


Q3 = boston.RM.quantile(0.75)
Q1 = boston.RM.quantile(0.25)
print(Q3,Q1)


# In[40]:


IQR= Q3-Q1
lower_limit = Q1-1.5*IQR
upper_limit = Q3+.5*IQR
print(lower_limit,upper_limit)


# In[41]:


boston_trimmed = boston[(boston['RM']>lower_limit) & (boston['RM']<upper_limit)]
boston_trimmed


# In[42]:


sns.boxplot(boston_trimmed.RM)
plt.title("BOXPLOT")
plt.show()


# In[43]:


from feature_engine.outliers import Winsorizer


# In[160]:


win = Winsorizer(capping_method='gaussian',tail='both',fold=1.5,variables=['LSTAT'])
boston_t = win.fit_transform(boston[['LSTAT']])


# In[162]:


print(win.left_tail_caps_,win.right_tail_caps_)


# In[163]:


boston_t


# In[165]:


sns.boxplot(boston_t['LSTAT'])
plt.show()


# In[47]:


from feature_engine.outliers import ArbitraryOutlierCapper


# In[48]:


capper  = ArbitraryOutlierCapper(max_capping_dict={'RM':7.5},min_capping_dict={'RM':4.8})
boston_c = capper.fit_transform(boston[['RM']])


# In[49]:


print(capper.right_tail_caps_,capper.left_tail_caps_)


# In[50]:


sns.boxplot(boston_c.RM)
plt.title("Boxplot Graph")
plt.show()


# In[51]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[52]:


df = pd.read_csv("D:\\Data Science\\7.Data Wrangling\\homeprices.csv")
df


# In[53]:


dummies = pd.get_dummies(df.town)
dummies


# In[54]:


df_dummies = pd.concat([df,dummies], axis='columns')
df_dummies


# In[55]:


df_dummies.drop('town',axis ='columns',inplace = True)
df_dummies


# In[56]:


df_dummies.drop('monroe township',axis ='columns',inplace =True)


# In[57]:


df_dummies


# In[58]:


df


# In[59]:


df_dum = pd.get_dummies(df)


# In[60]:


df_dum


# In[61]:


df_dum = pd.get_dummies(df,drop_first=True)
df_dum


# In[62]:


from sklearn.preprocessing import OneHotEncoder


# In[63]:


enc = OneHotEncoder(drop='first')


# In[64]:


enc_df =pd.DataFrame(enc.fit_transform(df[['town']]).toarray())


# In[65]:


df_ohe =df.join(enc_df)


# In[66]:


df_ohe.drop('town',axis ='columns',inplace = True)


# In[67]:


df_ohe


# In[68]:


dfle = df.copy()


# In[69]:


from sklearn.preprocessing import LabelEncoder


# In[70]:


le =LabelEncoder()
dfle.town = le.fit_transform(dfle.town)
dfle


# In[71]:


df_oe = df.copy()


# In[72]:


from sklearn.preprocessing import OrdinalEncoder
oe = OrdinalEncoder(categories=[['robinsville','monroe township','west windsor']])

df_oe.town = oe.fit_transform(df_oe[['town']])
df_oe


# In[73]:


df_m = df.copy()


# In[74]:


df_m['town'] = df_m['town'].map({'monroe township': 1,'west windsor': 2,'robinsville':3})
df_m


# In[75]:


import pandas as pd


# In[76]:


df = pd.read_csv("D:\\Data Science\\7.Data Wrangling\\stroke prediction.csv")
df


# In[77]:


df.info()


# In[78]:


df['age'].value_counts()


# In[79]:


intervals = [0,12,19,30,60,90]
categories =['child','teenager','young_adult','middle_aged','senior_citizen']


# In[80]:


df['Stroke_category']=  pd.cut(x=df['age'],bins =intervals,labels =categories)


# In[81]:


df.head(3)


# In[82]:


df[['age','Stroke_category']]


# In[83]:


data = pd.read_csv("D:\\Data Science\\7.Data Wrangling\\titanic.csv")
data


# In[84]:


data = pd.read_csv("D:\\Data Science\\7.Data Wrangling\\titanic.csv",usecols =['Fare','Age'])


# In[85]:


data


# In[86]:


data.Fare.hist()
plt.show()


# In[87]:


data.Fare.skew()


# In[88]:


data['sqr_Fare'] = data['Fare']**(1/3)
data['sqr_Fare'].hist()


# In[89]:


data.sqr_Fare.skew()


# In[90]:


data['root_Fare']=data['Fare']**(1/5)
data['root_Fare'].hist()
plt.show()


# In[91]:


data.root_Fare.skew()


# In[92]:


data['Log_Fare'] = np.log(data['Fare']+1)
data['Log_Fare'].hist()

plt.show()


# In[93]:


data.Log_Fare.skew()


# In[94]:


data['Rec_Fare'] = 1/(data['Fare']+1)
data.Rec_Fare.hist()
plt.show()


# In[95]:


data['Rec_Fare'].skew()


# In[96]:


from scipy import stats
data['Fare_boxcox'],param=stats.boxcox(data.Fare+1)
data['Fare_boxcox'].hist()


# In[97]:


print("lambda= ",param)


# In[98]:


data['Fare_boxcox'].skew()


# In[99]:


sns.distplot(data['Fare'],kde = True)
plt.show()


# In[100]:


sns.distplot(data['Fare_boxcox'],kde =True)
plt.show()


# In[101]:


df =pd.read_csv("D:\\Data Science\\7.Data Wrangling\\titanic.csv",usecols =['Age'])
df.head()


# In[102]:


df.isnull().sum()


# In[103]:


df.Age.hist()


# In[104]:


df['Age'].fillna(df.Age.median(),inplace =True)


# In[105]:


df.isnull().sum()


# In[106]:


from sklearn.preprocessing import StandardScaler

sc =StandardScaler()


# In[107]:


df['Age_sc'] = sc.fit_transform(df[['Age']])
df['Age_sc']


# In[108]:


plt.hist(df['Age_sc'])
plt.show()


# In[109]:


from sklearn.preprocessing import MinMaxScaler
min_max = MinMaxScaler()


# In[110]:


df['Age_mm'] = min_max.fit_transform(df[["Age"]])
df["Age_mm"]


# In[111]:


from sklearn.preprocessing import RobustScaler
rs = RobustScaler()


# In[112]:


df['Age_rs'] = rs.fit_transform(df[['Age']])
df['Age_rs']


# In[113]:


df['Age'].max()


# In[114]:


from sklearn.preprocessing import MaxAbsScaler

mas =MaxAbsScaler()


# In[115]:


df['Age_mas'] = mas.fit_transform(df[['Age']])
df['Age_mas']


# In[116]:


df.Age.head()


# In[117]:


plt.hist(df['Age_mas'])
plt.show()


# In[118]:


sns.distplot(df['Age_mas'])
plt.show()


# In[119]:


import numpy as np
import pandas as pd


# In[120]:


nb_sample =100
np.random.seed(0)


# In[121]:


#se = pd.Series(np.random.randint(0,100))


# In[122]:


#index = pd.date_range(start=pd.to_datetime('2016-09-24'),periods =nb_sample ,freq='D'))


# In[123]:


se = pd.Series(np.random.randint(0, 100, nb_sample),
index = pd.date_range(start = pd.to_datetime('2016-09-24'),
periods = nb_sample, freq='D'))


# In[124]:


se.head(3)


# In[125]:


se.head(5)


# In[126]:


df  =pd.read_csv("D:\\Data Science\\kraggle Datasets\\weight-height.csv")


# In[127]:


df


# In[128]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[129]:


sns.boxplot(df.Height)
plt.show()


# In[130]:


q1 = np.percentile(df.Height,25)
q3 = np.percentile(df.Height,75)
print(q1,q3)


# In[131]:


IQR = q3-q1
print(IQR)


# In[132]:


df.Height.min()


# In[133]:


df.Height.max()


# In[134]:


lower_value = q1-1.5*(IQR)
upper_value = q3+1.5*(IQR)
print(lower_value,upper_value)


# In[135]:


df_outliers = df[(df['Height']>lower_value) & (df['Height']< upper_value)]


# In[136]:


df_outliers


# In[137]:


sns.boxplot(df_outliers.Height)
plt.show()


# In[138]:


data = pd.read_csv("D:\\Data Science\\kraggle Datasets\\weight-height.csv")


# In[139]:


data


# In[140]:


df.sample(5)


# In[141]:


plt.hist(df.Height,bins=30,rwidth=0.8)

plt.show()


# In[142]:


from scipy.stats import norm

plt.hist(df.Height,bins=30,rwidth=0.8,density= True)
rng = np.arange(df.Height.min(),df.Height.max(),0.1)
plt.plot(rng, norm.pdf(rng,df.Height.mean(),df.Height.std()))

plt.show()


# In[143]:


upper_limit = data.Height.mean()+3*data.Height.std()


# In[144]:


lower_limit = data.Height.mean()-3*data.Height.std()
print(upper_limit,lower_limit)


# In[145]:


data_outliers1 = data[(data.Height>lower_limit) &(df.Height<upper_limit)]


# In[146]:


data_outliers1


# In[147]:


sns.boxplot(df_outliers.Height)
plt.show()


# In[148]:


data.shape[0] - data_outliers1.shape[0]


# In[ ]:




