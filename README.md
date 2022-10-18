# Business intelligence - Приложения из магазина Google Play

***Задание: С ростом количества мобильных устройств увеличивается и потребность в разработке приложений для Android. Ежедневно создаются сотни мобильных приложений. Я часто замечал, что мобильное приложение имеющие огромное количество отзывов, в среднем, имеют оценку больше, чем приложение с небольшим количеством отзывов. Поэтому, проверим эту гипотезу проанализировав открытый датасет `googleplaystore`.***

![](https://www.kaggle.com/datasets/lava18/google-play-store-apps?select=googleplaystore.csv)

***Подключение библиотек:***

```Python
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
import numpy as np
```

## Анализ данных

***Вывод таблицы `googleplaystore` с условием `Rating <= 5.0`, с целью избежать случай, когда рейтинг, по ошибке, превышает максимальное значение  `=5.0`***

```Python
googleplaystore = googleplaystore[googleplaystore.Rating <= 5.0]
googleplaystore
```

|       | App                                               | Category              | Rating | Reviews | Size               | Installs    | Type | Price | Content Rating | Genres                    | Last Updated     | Current Ver        | Android Ver               |
| ----- | ------------------------------------------------- | --------------------- | ------ | ------- | ------------------ | ----------- | ---- | ----- | -------------- | ------------------------- | ---------------- | ------------------ | ------------------------- |
| 0     | Photo Editor & Candy Camera & Grid & ScrapBook    | ART\_AND\_DESIGN      | 4.1    | 159     | 19M                | 10,000+     | Free | 0     | Everyone       | Art & Design              | January 7, 2018  | 1.0.0              | 4.0.3 and up              |
| 1     | Coloring book moana                               | ART\_AND\_DESIGN      | 3.9    | 967     | 14M                | 500,000+    | Free | 0     | Everyone       | Art & Design;Pretend Play | January 15, 2018 | 2.0.0              | 4.0.3 and up              |
| 2     | U Launcher Lite – FREE Live Cool Themes, Hide ... | ART\_AND\_DESIGN      | 4.7    | 87510   | 8.7M               | 5,000,000+  | Free | 0     | Everyone       | Art & Design              | August 1, 2018   | 1.2.4              | 4.0.3 and up              |
| 3     | Sketch - Draw & Paint                             | ART\_AND\_DESIGN      | 4.5    | 215644  | 25M                | 50,000,000+ | Free | 0     | Teen           | Art & Design              | June 8, 2018     | Varies with device | 4.2 and up                |
| 4     | Pixel Draw - Number Art Coloring Book             | ART\_AND\_DESIGN      | 4.3    | 967     | 2.8M               | 100,000+    | Free | 0     | Everyone       | Art & Design;Creativity   | June 20, 2018    | 1.1                | 4.4 and up                |
| ...   | ...                                               | ...                   | ...    | ...     | ...                | ...         | ...  | ...   | ...            | ...                       | ...              | ...                | ...                       |
| 10834 | FR Calculator                                     | FAMILY                | 4      | 7       | 2.6M               | 500+        | Free | 0     | Everyone       | Education                 | June 18, 2017    | 1.0.0              | 4.1 and up                |
| 10836 | Sya9a Maroc - FR                                  | FAMILY                | 4.5    | 38      | 53M                | 5,000+      | Free | 0     | Everyone       | Education                 | July 25, 2017    | 1.48               | 4.1 and up                |
| 10837 | Fr. Mike Schmitz Audio Teachings                  | FAMILY                | 5      | 4       | 3.6M               | 100+        | Free | 0     | Everyone       | Education                 | July 6, 2018     | 1                  | 4.1 and up                |
| 10839 | The SCP Foundation DB fr nn5n                     | BOOKS\_AND\_REFERENCE | 4.5    | 114     | Varies with device | 1,000+      | Free | 0     | Mature 17+     | Books & Reference         | January 19, 2015 | Varies with device | Varies with device        |
| 10840 | iHoroscope - 2018 Daily Horoscope & Astrology     | LIFESTYLE             | 4.5    | 398307  | 19M                | 10,000,000+ | Free | 0     | Everyone       | Lifestyle                 | July 25, 2018    | Varies with device | Varies with device |

***Создание описательной статистики:***

 + count - количество не NA/null наблюдений;
 + mean - среднее значение value;
 + std - мера отклонения от среднего значения;
 + min - минимум значений в объекте;
 + max - максимум значений в объекте;

```Python
googleplaystore.describe()
```

|       | Rating      |
| ----- | ----------- |
| count | 9366.000000 |
| mean  | 4.191757    |
| std   | 0.515219    |
| min   | 1.000000    |
| 25%   | 4.000000    |
| 50%   | 4.300000    |
| 75%   | 4.500000    |
| max   | 5.000000    |

***Вывод короткой таблицы по конкретным столбцам:***

```Python
short = googleplaystore[['Rating','Reviews', 'Installs']]
short.head(100)
```

|        | Rating  | Reviews  | Installs    |
| ------ | ------- | -------- | ----------- |
| 0      | 4.1     | 159      | 10,000+     |
| 1      | 3.9     | 967      | 500,000+    |
| 2      | 4.7     | 87510    | 5,000,000+  |
| 3      | 4.5     | 215644   | 50,000,000+ |
| 4      | 4.3     | 967      | 100,000+    |
| ...    | ...     | ...      | ...         |
| 96     | 4.4     | 2680     | 500,000+    |
| 97     | 4       | 1288     | 100,000+    |
| 98     | 4.7     | 18900    | 500,000+    |
| 99     | 4.9     | 49790    | 1,000,000+  |
| 100    | 4.7     | 1150     | 100,000+    |

***В датасете есть проблема: некоторые строки имеют значение `3.0m` - количество скачиваний вместо количества отзывов по ошибке, заменим это значение на `0`(таких значений несколько и они не влияют на результат обучения). Столбец `Reviews` - преобразуем в `int` для обучения.***

```Python
googleplaystore['Reviews'] = googleplaystore['Reviews'].str.replace('3.0M','0', regex = True).astype('int') 
```

***Получение данные о типах столбцов. Убедимся, что столбец количество отзывово имеет тип `int`:***

```Python
googleplaystore.dtypes
```

| App            | object  |
| -------------- | ------- |
| Category       | object  |
| Rating         | float64 |
| Reviews        | int32   |
| Size           | object  |
| Installs       | object  |
| Type           | object  |
| Price          | object  |
| Content Rating | object  |
| Genres         | object  |
| Last Updated   | object  |
| Current Ver    | object  |
| Android Ver    | object  |
| dtype: object  |         |

```Python
visual = googleplaystore[['Reviews', 'Rating']]
visual = visual[visual.Reviews <= 1000000]
visual.hist()
```

![](https://www.dropbox.com/s/vykj9uu64fshdfy/visual.png?dl=0)

***Разделение датафрейма на две случайные выборки train и test (80% и 20%) для обучения и тестирования.***

```Python
data = np.random.rand(len(visual)) < 0.8
train = visual[data]
test = visual[~data]
plt.xlabel("Reviews")
plt.ylabel("Rating")
plt.scatter(visual['Reviews'], visual['Rating'], color = 'red')
```

![](https://www.dropbox.com/s/qdcd12capxep3pj/visual2.png?dl=0)

***Создадим объект линейной регрессии. В качестве `train_x` независимая переменная - количество отзывов, в качестве `train_y` зависимое значение - оценка пользователя приложения.
Передадим их в метод `fit()` для обучения:***

```Python
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['Reviews']])
train_y = np.asanyarray(train[['Rating']])
regr.fit(train_x, train_y)

print('coefficient: ', regr.coef_)
print('intercept: ', regr.intercept_)
```

***После того как отработал `fit()`, мы имеем натренированную модель линейной регрессии. Данная модель предоставит значения коэфициента корреляции `coefficient` и углового коэфициента `intercept`. Это параметры линии, которая аппроксимирует данные. ***

`coefficient:  [[4.92679182e-07]]
intercept:  [4.14091973]`

***Вычислим показатель точности линейной регрессии:***

```Python
accuracy_score = regr.score(train_x, train_y)
print(accuracy_score)
```

`0.02067504467172654`

```Python
plt.scatter(train.Reviews, train.Rating, color = 'blue') #создание точечной диаграммы, используя plt.scatter() с двумя переменными, которые сравниваются в качестве входных аргументов и соответственно цвет, которым они будут отображаться.
plt.rcParams["figure.figsize"] = (8,5.5)
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r' ) # regr.coef_[0][0]*train_x - коэффициент кореляции, regr.intercept_[0] - угловой коэффициент.
#На графике на оси Х - количество оценок. На оси Y -  значение оценки пользователей приложения.
plt.xticks(np.arange(0, 1000000, 100000)) #Так как отзывов слишком много берем шаг в 100000.
plt.yticks(np.arange(0, 5, 1)) # Оценки отображаются с шагом 1 от 0 до 5.
plt.xlabel("Reviews")
plt.ylabel("Rating")
```

![](https://www.dropbox.com/s/yf9qvkzibtxzyef/visual3.png?dl=0)


```python

```
