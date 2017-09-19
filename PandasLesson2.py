
# coding: utf-8

# In[3]:

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


# In[4]:

df = pd.read_csv("J:/HiEnd/mlcourse_open-master/mlcourse_open-master/data/video_games_sales.csv")
df.info()


# In[5]:

# Данные об оценках есть не для всех фильмов, поэтому давайте оставим только те записи, в которых нет пропусков с помощью метода dropna.
df = df.dropna()
print(df.shape)
# смысл метода смотреть здесь https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.dropna.html
# дропает всё что Nun-object


# In[6]:

df.head()


# In[7]:

# опеределяем признаки с которыми будем работать
useful_cols = ['Name', 'Platform', 'Year_of_Release', 'Genre', 
               'Global_Sales', 'Critic_Score', 'Critic_Count',
               'User_Score', 'User_Count', 'Rating'
              ]
df[useful_cols].head()


# In[8]:

print(df[useful_cols].info())


# In[9]:

# Для примера построим график продаж видео игр в различных странах в зависимости от года.
# Для начала отфильтруем только нужные нам столбцы
sales_df = df[[x for x in df.columns if 'Sales' in x] + ['Year_of_Release']]
# затем посчитаем суммарные продажи по годам и у получившегося dataframe вызовем функцию plot без параметров.
sales_df.groupby('Year_of_Release').sum().plot()


# In[10]:

# поэкспериментируем
sales_df = df[[x for x in df.columns if 'Sales' in x] + ['User_Score']]
sales_df.groupby('User_Score').sum().plot()


# In[11]:

sales_df = df[[x for x in df.columns if 'Global_Sales' in x] + ['User_Score']]
sales_df.groupby('User_Score').sum().plot()


# In[12]:

# ларчик просто открывался
# [x for x in df.columns if 'Sales' in x] - поиск подстроки Sales в строках df.colums содержащих objectname_sales.

sales_df = df[[x for x in df.columns if 'Critic' in x] + ['User_Score']]
sales_df.groupby('User_Score').sum().plot()


# In[13]:

# C помощью параметра kind можно изменить тип графика, например, на bar chart.
# Matplotlib позволяет очень гибко настраивать графики. См.документацию
# Например, параметра rot отвечает за угол наклона подписей к оси x.
sales_df = df[[x for x in df.columns if 'Sales' in x] + ['Year_of_Release']]
sales_df.groupby('Year_of_Release').sum().plot(kind='bar', rot=45)


# In[14]:

# Seaborn
# Seaborn — это по сути более высокоуровневое API на базе библиотеки matplotlib.
# Если просто добавить в код import seaborn, то картинки станут гораздо симпатичнее. 
# Также в библиотеке есть достаточно сложные типы визуализации, которые в matplotlib потребовали бы большого количество кода.
# Познакомимся с первым таким "сложным" типом графиков pair plot (scatter plot matrix).
# Эта визуализация поможет нам посмотреть на одной картинке, как связаны между собой различные признаки.
# не забываем преобразоывать User_Score 10015 non-null object в float
df["User_Score"] = df["User_Score"].astype("float")
cols = ['Global_Sales', 'Critic_Score', 'Critic_Count', 'User_Score', 'User_Count']
sns_plot = sns.pairplot(df[cols])
sns_plot.savefig('pairplot.png')
# Для сохранения графиков в файлы стоит использовать метод savefig.


# In[15]:

# С помощью seaborn можно построить и распределение dist plot.
# Для примера посмотрим на распределение оценок критиков Critic_Score. 
# По default'у на графике отображается гистограмма и kernel density estimation.
# https://en.wikipedia.org/wiki/Kernel_density_estimation
sns.distplot(df.Critic_Score)


# In[16]:

sns.distplot(df.Global_Sales)


# In[17]:

sns.jointplot(df.Critic_Score, df.User_Score)


# In[18]:

sns.jointplot(df.Global_Sales, df.User_Score)


# In[19]:

sales_df = sales_df.dropna()
df.User_Score = df.User_Score.dropna()


# In[20]:

# пока не работает, но подумать можно.
# общий смысл в том что sales_df это таблица, а sns и .jointplot любит работать с однострочными series 
sns.pairplot(sales_df)


# In[21]:

df.User_Score


# In[22]:

# Еще один полезный тип графиков — это box plot.
# Давайте сравним оценки игр от критиков для топ-5 крупнейших игровых платформ.
top_platforms = df.Platform.value_counts().sort_values(ascending = False).head(5).index.values
sns.boxplot(y="Platform", x="Critic_Score", data=df[df.Platform.isin(top_platforms)], orient="h")


# In[23]:

# а теперь сравним по оценку критиков по жанру
top_platforms = df.Platform.value_counts().sort_values(ascending = False).head(5).index.values
sns.boxplot(y="Genre", x="Critic_Score", data=df[df.Platform.isin(top_platforms)], orient="h")
# что есть что
# Коробка показывает интерквартильный размах распределения, то есть соответственно 25% (Q1) и 75% (Q3) перцентили
# Черта внутри коробки обозначает медиану распределения.
# Усы отображают весь разброс точек кроме выбросов, то есть минимальные и максимальные значения, которые попадают
# в промежуток (Q1 - 1.5*IQR, Q3 + 1.5*IQR), где IQR = Q3 - Q1 — интерквартильный размах.
# Точками на графике обозначаются выбросы (outliers) — те значения, которые не вписываются в промежуток значений, заданный усами графика.


# In[24]:

# heat map. 
# Heat map позволяет посмотреть на распределение какого-то численного признака по двум категориальным.
# Визуализируем суммарные продажи игр по жанрам и игровым платформам.
platform_genre_sales = df.pivot_table(
                        index='Platform', 
                        columns='Genre', 
                        values='Global_Sales', 
                        aggfunc=sum).fillna(0).applymap(float)
sns.heatmap(platform_genre_sales, annot=True, fmt=".1f", linewidths=.5)


# In[25]:

# и тут меня пробило - ибо сводные таблицы наше всё, а умение строить хит мапы по сводным таблицам - свято как евангеле.
# но юпитер что то тупит, ладно потом потренируюсь в другом дата фрейме.


# In[26]:

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly
import plotly.graph_objs as go

init_notebook_mode(connected=True)


# In[27]:

# посчитаем число вышедших игр и проданных копий по годам
years_df = df.groupby('Year_of_Release')[['Global_Sales']].sum().join(
    df.groupby('Year_of_Release')[['Name']].count()
)
years_df.columns = ['Global_Sales', 'Number_of_Games']

# создаем линию для числа проданных копий
trace0 = go.Scatter(
    x=years_df.index,
    y=years_df.Global_Sales,
    name='Global Sales'
)

# создаем линию для числа вышедших игр 
trace1 = go.Scatter(
    x=years_df.index,
    y=years_df.Number_of_Games,
    name='Number of games released'
)

# определяем массив данных и задаем title графика в layout
data = [trace0, trace1]
layout = {'title': 'Statistics of video games'}

# cоздаем объект Figure и визуализируем его
fig = go.Figure(data=data, layout=layout)
iplot(fig, show_link=False)


# In[28]:

# Plotly — это open-source библиотека, которая позволяет строить интерактивные графики в jupyter.notebook'e без необходимости зарываться в javascript код.
# Прелесть интерактивных графиков заключается в том, что можно посмотреть точное численное значение при наведении мыши, 
# скрыть неинтересные ряды в визуализации, приблизить определенный участок графика и т.д.
# это божественно!

years_df = df.groupby('Critic_Score')[['Global_Sales']].sum().join(
    df.groupby('Critic_Score')[['Name']].count()
)
years_df.columns = ['Global_Sales', 'Number_of_Games']

# создаем линию для числа проданных копий
trace0 = go.Scatter(
    x=years_df.index,
    y=years_df.Global_Sales,
    name='Global Sales'
)

# создаем линию для числа вышедших игр 
trace1 = go.Scatter(
    x=years_df.index,
    y=years_df.Number_of_Games,
    name='Critic_score_number'
)

# определяем массив данных и задаем title графика в layout
data = [trace0, trace1]
layout = {'title': 'Statistics of video games'}

# cоздаем объект Figure и визуализируем его
fig = go.Figure(data=data, layout=layout)
iplot(fig, show_link=False)


# In[29]:

# В plotly строится визуализация объекта Figure, который состоит из данных (массив линий, которые в библиотеке называются traces) 
# и оформления/стиля, за который отвечает объект layout. В простых случаях можно вызывать функцию iplot и просто от массива traces.
# Параметр show_link отвечает за ссылки на online-платформу plot.ly на графиках. 
# Поскольку обычно это функциональность не нужна, то я предпочитаю скрывать ее для предотвращения случайных нажатий.
# Можно сразу сохранить график в виде html-файла.
plotly.offline.plot(fig, filename='years_stats.html', show_link=False)


# In[30]:

# Посмотрим также на рыночную долю игровых платформ, рассчитанную по количеству выпущенных игр и по суммарной выручке.
# Для этого построим bar chart.
# считаем число проданных и вышедших игр по платформам
platforms_df = df.groupby('Platform')[['Global_Sales']].sum().join(
    df.groupby('Platform')[['Name']].count()
)
platforms_df.columns = ['Global_Sales', 'Number_of_Games']
platforms_df.sort_values('Global_Sales', ascending=False, inplace=True)

# создаем traces для визуализации
trace0 = go.Bar(
    x=platforms_df.index,
    y=platforms_df.Global_Sales,
    name='Global Sales'
)

trace1 = go.Bar(
    x=platforms_df.index,
    y=platforms_df.Number_of_Games,
    name='Number of games released'
)

# создаем массив с данными и задаем title для графика и оси x в layout
data = [trace0, trace1]
layout = {'title': 'Share of platforms', 'xaxis': {'title': 'platform'}}

# создаем объект Figure и визуализируем его
fig = go.Figure(data=data, layout=layout)
iplot(fig, show_link=False)


# In[31]:

plotly.offline.plot(fig, filename='Platforms.html', show_link=False)


# In[32]:

# В plotly можно построить и box plot. Рассмотрим распределения оценок критиков в зависимости от жанра игры.
# создаем Box trace для каждого жанра из наших данных
data = []
for genre in df.Genre.unique():
    data.append(
        go.Box(y=df[df.Genre==genre].Critic_Score, name=genre)
    )

# визуализируем данные
iplot(data, show_link = False)


# In[33]:

# Пример визуального анализа данных

df = pd.read_csv("J:/HiEnd/mlcourse_open-master/mlcourse_open-master/data/telecom_churn.csv")
df.head()


# In[34]:

df.shape


# In[35]:

df.info()


# In[36]:

df['Churn'].value_counts()


# In[37]:

df['Churn'].value_counts().plot(kind='bar', label='Churn')
plt.legend()
plt.title('Распределение оттока клиентов');


# In[126]:

# Выделим следующие группы признаков (среди всех кроме Churn ):
# бинарные (object): International plan, Voice mail plan
# категориальные: State
# порядковые: Customer service calls
# количественные: все остальные
df.State


# In[38]:

df["International plan"]


# In[137]:

df["Customer service calls"]


# In[39]:

# Посмотрим на корреляции количественных признаков. 
# По раскрашенной матрице корреляций видно, что такие признаки как Total day charge считаются по проговоренным минутам 
# (Total day minutes). То есть 4 признака можно выкинуть, они не несут полезной информации.
corr_matrix = df.drop(['State', 'International plan', 'Voice mail plan',
                      'Area code'], axis=1).corr()
sns.heatmap(corr_matrix);


# In[40]:

# Теперь посмотрим на распределения всех интересующих нас количественных признаков. 
# На бинарные/категориальные/порядковые признакие будем смотреть отдельно.

features = list(set(df.columns) - set(['State', 'International plan', 'Voice mail plan',  'Area code',
                                      'Total day charge',   'Total eve charge',   'Total night charge',
                                        'Total intl charge', 'Churn']))

df[features].hist(figsize=(20,12));
# Видим, что большинство признаков распределены нормально. Исключения – число звонков в сервисный центр (Customer service calls) 
# (тут больше подходит пуассоновское распределение) и число голосовых сообщений (Number vmail messages, пик в нуле,
# т.е. это те, у кого голосовая почта не подключена).
# Также смещено распределение числа международных звонков (Total intl calls).


# In[41]:

# а без Churn будет редкостная хуета
features = list(set(df.columns) - set(['State', 'International plan', 'Voice mail plan',  'Area code',
                                      'Total day charge',   'Total eve charge',   'Total night charge',
                                        'Total intl charge']))

df[features].hist(figsize=(20,12));


# In[167]:

# Еще полезно строить вот такие картинки, где на главной диагонали рисуются распредления признаков, а вне главной диагонали 
# – диаграммы рассеяния для пар признаков. Бывает, что это приводит к каким-то выводам, но в данном случае все примерно понятно, без сюрпризов.
features = list(set(df.columns) - set(['State', 'International plan', 'Voice mail plan',  'Area code',
                                      'Total day charge',   'Total eve charge',   'Total night charge',
                                        'Total intl charge', 'Churn']))
sns.pairplot(df[features + ['Churn']], hue='Churn');


# In[175]:

_, axes = plt.subplots(1, 2, sharey=True, figsize=(16,6))

sns.boxplot(x='Churn', y='Total day minutes', data=df, ax=axes[0]);
sns.violinplot(x='Churn', y='Total day minutes', data=df, ax=axes[1]);
# Интересное наблюдение: в среднем ушедшие клиенты больше пользуются связью. Возможно, они недовольны тарифами,
# и одной из мер борьбы с оттоком будет понижение тарифных ставок (стоимости мобильной связи). 
# Но это уже компании надо будет проводить дополнительный экономический анализ, действительно ли такие меры будут оправданы.


# In[42]:

# Теперь изобразим распределение числа обращений в сервисный центр (такую картинку мы строили в первой статье).
# Тут уникальных значений признака не много (признак можно считать как количественным целочисленным, так и порядковым),
# и наглядней изобразить распределение с помощью countplot. 
# Наблюдение: доля оттока сильно возрастает начиная с 4 звонков в сервисный центр.
sns.countplot(x='Customer service calls', hue='Churn', data=df);


# In[43]:

# Теперь посмотрим на связь бинарных признаков International plan и Voice mail plan с оттоком. 
# Наблюдение: когда роуминг подключен, доля оттока намного выше, т.е. наличие международного роуминга – сильный признак.
# Про голосовую почту такого нельзя сказать.
_, axes = plt.subplots(1, 2, sharey=True, figsize=(16,6))

sns.countplot(x='International plan', hue='Churn', data=df, ax=axes[0]);
sns.countplot(x='Voice mail plan', hue='Churn', data=df, ax=axes[1]);


# In[44]:

# Наконец, посмотрим, как с оттоком связан категориальный признак State. С ним уже не так приятно работать,
# поскольку число уникальных штатов довольно велико – 51. Можно в начале построить сводную табличку или посчитать процент 
# оттока для каждого штата. Но данных по каждом штату по отдельности маловато (ушедших клиентов всего от 3 до 17 в каждом штате),
# поэтому, возможно, признак State впоследствии не стоит добавлять в модели классификации из-за риска переобучения
# (но мы это будем проверять на кросс-валидации, stay tuned!).
df.groupby(['State'])['Churn'].agg([np.mean]).sort_values(by='mean', ascending=False).T


# In[ ]:

# запоминаем про божественную процедуру df[[x for x in df.columns if 'Sales' in x] и идём читать про математике t-SNE.

