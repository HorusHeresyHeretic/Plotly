
# coding: utf-8

# In[1]:

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import warnings
warnings.simplefilter('ignore')
# будем отображать графики прямо в jupyter'e
get_ipython().magic('pylab inline')
# увеличим дефолтный размер графиков
from pylab import rcParams    
rcParams['figure.figsize'] = 18, 15
import pandas as pd
import seaborn as sns


# In[2]:

data = pd.read_csv("J:/HiEnd/mlcourse_open-master/mlcourse_open-master/data/adult.data.csv")
print(data.info()) 


# In[3]:

data.head()


# In[4]:

salary_data = data[[x for x in data.columns if 'salary' in x] + ['capital-gain']]
# потренируемся
salary_data.groupby('salary').sum().plot()


# In[5]:

salary_data = data[[x for x in data.columns if 'salary' in x] + ['capital-loss']]
# потренируемся
salary_data.groupby('salary').sum().plot()


# In[6]:

hours_data = data[[x for x in data.columns if 'hours-per-week' in x] + ['capital-loss']]
# потренируемся
hours_data.groupby('hours-per-week').sum().plot()


# In[7]:

hours_data = data[[x for x in data.columns if 'hours-per-week' in x] + ['capital-gain']]
# потренируемся
hours_data.groupby('hours-per-week').sum().plot()


# In[8]:

age_data = data[[x for x in data.columns if 'age' in x] + ['capital-gain']]
# потренируемся
age_data.groupby('age').sum().plot()


# In[9]:

age_data = data[[x for x in data.columns if 'age' in x] + ['capital-loss']]
# потренируемся
age_data.groupby('age').sum().plot()


# In[10]:

cols = ['capital-loss', 'capital-gain', 'age']
sns_plot = sns.pairplot(data[cols])
sns_plot.savefig('pairplot1.png')


# In[11]:

sns.distplot(data['capital-gain'])


# In[12]:

sns.distplot(data['age'])


# In[13]:

sns.jointplot(data['age'], data["capital-gain"])


# In[14]:

age_data = age_data.dropna()
data["capital-gain"] = data["capital-gain"].dropna()


# In[15]:

sns.pairplot(age_data)


# In[16]:

hours_data = hours_data.dropna()
data["capital-loss"] = data["capital-loss"].dropna()


# In[17]:

sns.pairplot(age_data)


# In[18]:

data.head()


# In[19]:

useful_cols = ['age', 'workclass', 'native-country', 'salary', 
               'capital-gain', 'capital-loss', 'hours-per-week' 
              ]
data[useful_cols].head()


# In[20]:

print(data[useful_cols].info())


# In[21]:

age_data = data[[x for x in data.columns if 'age' in x] + ['capital-loss']]
# потренируемся
age_data.groupby('age').sum().plot()


# In[22]:

top_workclass = data["workclass"].value_counts().sort_values(ascending = False).head(5).index.values
sns.boxplot(y="workclass", x="age", data=data[data.workclass.isin(top_workclass)], orient="h")


# In[23]:

top_workclass = data["workclass"].value_counts().sort_values(ascending = False).head(5).index.values
sns.boxplot(y="workclass", x="hours-per-week", data=data[data.workclass.isin(top_workclass)], orient="h")


# In[24]:

# стоит от метить что хит мапы через сводные таблицы умеют рабоать с object ,а вот всё что выше - почти всегда хочет цифру.
rcParams['figure.figsize'] = 20, 15
planet_work = data.pivot_table(
                        index='native-country', 
                        columns='workclass', 
                        values='hours-per-week', 
                        aggfunc=sum).fillna(0).applymap(float)
sns.heatmap(planet_work, annot=True, fmt=".1f", linewidths=.5)


# In[25]:

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly
import plotly.graph_objs as go

init_notebook_mode(connected=True)


# In[28]:

# попробуем посмотреть что получиться
stat_data = data.groupby('age')[['hours-per-week']].sum().join(
    data.groupby('age')[['workclass']].count()
)
stat_data.columns = ['hours-per-week', 'capital-gain']

# создаем линию для числа проданных копий
trace0 = go.Scatter(
    x=stat_data.index,
    y=stat_data['hours-per-week'],
    name='hours-per-week'
)

# создаем линию для числа вышедших игр 
trace1 = go.Scatter(
    x=stat_data.index,
    y=stat_data['capital-gain'],
    name='capital-gain'
)

# определяем массив данных и задаем title графика в layout
data = [trace0, trace1]
layout = {'title': 'Statistics of work hours'}

# cоздаем объект Figure и визуализируем его
fig = go.Figure(data=data, layout=layout)
iplot(fig, show_link=False)


# In[ ]:

# каждая пони ереси должна помнить о системной таблице и не затирать ссылки на ранее созданные объекты. 


# In[29]:

plotly.offline.plot(fig, filename='Statistics of work hours.html', show_link=False)


# In[ ]:



