import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import statistics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import openpyxl

df1 = pd.read_csv('data1.csv')
######## 平均を計算
def mean(vec):
  return np.sum(vec) / len(vec)


######## メイン部分
if (__name__ == "__main__"):

  ###### Pandas ライブラリを使って CSV ファイルを見出しつきで読み込む
  df = pd.read_csv('data1.csv', index_col=0)
  print(df)

  ###### 要素へのアクセスは行列名の場合 at/loc、番号の場合 iat/iloc で可能
  #print(df.at['2022/12', 'Nikkei225'])  # 2022年12月の日経平均を表示
  #print(df.loc[:, 'Nikkei225'])         # 日経平均の列をすべて表示
  #print(df.iloc[132:144, 1])            # 2022年1月～12月の日経平均を表示

  ###### (1) 平均と標準偏差を求める
  xmean = mean(df.loc[:, 'USDJPY'])
  ymean = mean(df.loc[:, 'Nikkei225'])
  print('(1)XとYの平均値と標準偏差を求める')
  print('Xの平均値:', xmean)
  print('Yの平均値:', ymean)

  xstd = statistics.pstdev(df.loc[:,'USDJPY' ])
  ystd = statistics.pstdev(df.loc[:,'Nikkei225'])
  print('Xの標準偏差:', xstd)
  print('Yの標準偏差:', ystd)
  print('----------------------------------------------------------')

  ###### (2) 最小値・最大値・中央値を求める
  #n, bins, _ = plt.hist(df.loc[:, 'USDJPY'], bins=bins, range=(xmin, xmax))
  #plt.show()
  xmax = df.loc[:,'USDJPY'].max()
  xmin = df.loc[:,'USDJPY'].min()
  xmedian = df.loc[:,'USDJPY'].median()
  print('(2)Xの最大値、最小値、中央値とヒストグラムの作成')
  print('Xの最大値:', xmax)
  print('Xの最小値:', xmin)
  print('Xの中央値:', xmedian)

  plt.hist(df.loc[:,'USDJPY'], bins=np.arange(df.loc[:,'USDJPY'].min(), df.loc[:,'USDJPY'].max(), 5))
  plt.title('Histogram of USDJPY')
  plt.xlabel('USDJPY')
  plt.ylabel('Frequency')
  plt.show()
  print('----------------------------------------------------------')

  ###### (3) 散布図を出してみる
  print('(3)散布図を出してみる')
  plt.scatter(df.loc[:, 'USDJPY'], df.loc[:, 'Nikkei225'])
  plt.title('Scatter')
  plt.xlabel('USDJPY')
  plt.ylabel('Nikkei225')
  plt.show()

  ###### (4) 相関係数の計算
  print('(4)相関係数の計算')
  xySokan = df.loc[:,'USDJPY'].corr(df.loc[:,'Nikkei225'])
  print('XとYの相関係数:',xySokan)
  print('----------------------------------------------------------')
  ##### (5) 回帰分析により、xからyを推定できるか。
  print('相関係数が約0.78と十分な正の相関にあることと、ヒストグラムより回帰分析によりｙの値を推定できるといえる。')
  print('----------------------------------------------------------')

  ##### (6)　実際に回帰分析をする。
  print('(6)実際に回帰分析をする。')
  X = df1[['USDJPY']]
  Y = df1[['Nikkei225']]
  model = LinearRegression()
  model.fit(X,Y)
  print('回帰変数：',model.coef_)
  print('切片：',model.intercept_)
  plt.plot(X,Y,'o')
  plt.plot(X,model.predict(X))
  plt.show()
  print('----------------------------------------------------------')

  ##### (7)予測
  print('(7)予測')
  x1 = 148.19
  y1 = model.predict([[x1]])
  print('yの予測した値:',y1[0])
  print('----------------------------------------------------------')

  ##### (8)2011年の決定係数
  print('(8)2011年の決定係数')
  df2 = pd.read_csv('data1.csv',nrows=13)
  X2011 = df2[['USDJPY']]
  Y2011 = df2[['Nikkei225']]
  model2011 = LinearRegression()
  model2011.fit(X2011,Y2011)
  R2_2011 = model2011.score(X2011,Y2011)
  print('2011年の決定係数:', R2_2011)
  print('----------------------------------------------------------')

  ##### (9)各年の決定係数とグラフ
  print('(9)各年の決定係数 (2012年～2022年)とグラフ')
  df3 = pd.read_csv('data1.csv', skiprows=12, nrows=12)
  X2012 = df3['76.94'].values.reshape(-1, 1)
  Y2012 =df3['8455.35'].values.reshape(-1, 1)
  model2012 = LinearRegression()
  model2012.fit(X2012,Y2012)
  R2_2012 = model2012.score(X2012,Y2012)
  print('2012年の決定係数:',R2_2012)

  df4 = pd.read_csv('data1.csv', skiprows=24, nrows=12)
  X2013 = df4['86.74'].values.reshape(-1, 1)
  Y2013 =df4['10395.18'].values.reshape(-1, 1)
  model2013 = LinearRegression()
  model2013.fit(X2013,Y2013)
  R2_2013 = model2013.score(X2013,Y2013)
  print('2013年の決定係数:',R2_2013)

  df5 = pd.read_csv('data1.csv', skiprows=36, nrows=12)
  X2014 = df5['105.30'].values.reshape(-1, 1)
  Y2014 =df5['16291.31'].values.reshape(-1, 1)
  model2014 = LinearRegression()
  model2014.fit(X2014,Y2014)
  R2_2014 = model2014.score(X2014,Y2014)
  print('2014年の決定係数:',R2_2014)

  df6 = pd.read_csv('data1.csv', skiprows=48, nrows=12)
  X2015 = df6['119.68'].values.reshape(-1, 1)
  Y2015 = df6['17450.77'].values.reshape(-1, 1)
  model2015 = LinearRegression()
  model2015.fit(X2015,Y2015)
  R2_2015 = model2015.score(X2015,Y2015)
  print('2015年の決定係数:',R2_2015)

  df7 = pd.read_csv('data1.csv', skiprows=60, nrows=12)
  X2016 = df7['120.30'].values.reshape(-1, 1)
  Y2016 =df7['19033.71'].values.reshape(-1, 1)
  model2016 = LinearRegression()
  model2016.fit(X2016,Y2016)
  R2_2016 = model2016.score(X2016,Y2016)
  print('2016年の決定係数:',R2_2016)

  df8 = pd.read_csv('data1.csv', skiprows=72, nrows=12)
  X2017 = df8['116.87'].values.reshape(-1, 1)
  Y2017 = df8['19114.37'].values.reshape(-1, 1)
  model2017 = LinearRegression()
  model2017.fit(X2017,Y2017)
  R2_2017 = model2017.score(X2017,Y2017)
  print('2017年の決定係数:',R2_2017)

  df9 = pd.read_csv('data1.csv', skiprows=84, nrows=12)
  X2018 = df9['112.67'].values.reshape(-1, 1)
  Y2018 = df9['22764.94'].values.reshape(-1, 1)
  model2018 = LinearRegression()
  model2018.fit(X2018,Y2018)
  R2_2018 = model2018.score(X2018,Y2018)
  print('2018年の決定係数:',R2_2018)

  df10 = pd.read_csv('data1.csv', skiprows=96, nrows=12)
  X2019 = df10['109.56'].values.reshape(-1, 1)
  Y2019 = df10['20014.77'].values.reshape(-1, 1)
  model2019 = LinearRegression()
  model2019.fit(X2019,Y2019)
  R2_2019 = model2019.score(X2019,Y2019)
  print('2019年の決定係数:',R2_2019)

  df11 = pd.read_csv('data1.csv', skiprows=108, nrows=12)
  X2020 = df11['108.61'].values.reshape(-1, 1)
  Y2020 = df11['23656.62'].values.reshape(-1, 1)
  model2020 = LinearRegression()
  model2020.fit(X2020,Y2020)
  R2_2020 = model2020.score(X2020,Y2020)
  print('2020年の決定係数:',R2_2020)

  df12 = pd.read_csv('data1.csv', skiprows=120, nrows=12)
  X2021 = df12['103.24'].values.reshape(-1, 1)
  Y2021 = df12['27444.17'].values.reshape(-1, 1)
  model2021 = LinearRegression()
  model2021.fit(X2021,Y2021)
  R2_2021 = model2021.score(X2021,Y2021)
  print('2021年の決定係数:',R2_2021)

  df13 = pd.read_csv('data1.csv', skiprows=132, nrows=12)
  X2022 = df13['115.08'].values.reshape(-1, 1)
  Y2022 = df13['28791.71'].values.reshape(-1, 1)
  model2022 = LinearRegression()
  model2022.fit(X2022,Y2022)
  R2_2022 = model2022.score(X2022,Y2022)
  print('2022年の決定係数:',R2_2022)

  R2 = []
  R2.append(R2_2011)
  R2.append(R2_2012)
  R2.append(R2_2013)
  R2.append(R2_2014)
  R2.append(R2_2015)
  R2.append(R2_2016)
  R2.append(R2_2017)
  R2.append(R2_2018)
  R2.append(R2_2019)
  R2.append(R2_2020)
  R2.append(R2_2021)
  R2.append(R2_2022)

  years = range(2011,2023)
  
  plt.plot(years,R2)
  plt.title('Graph of R2')
  plt.xlabel('Year')
  plt.ylabel('R2')
  plt.show()

