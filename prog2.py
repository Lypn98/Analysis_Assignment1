import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

df1 = pd.read_csv('data2.csv')
print(df1)


df1_clean = df1.replace({'Strikeout': np.nan, 'Walk': np.nan, 'Double': np.nan}).dropna()
df2 = df1.replace({'Strikeout': np.nan, 'Walk': np.nan}).dropna()

F_group = df1_clean[df1_clean['Result'].isin(['Home Run', 'Flyout', 'Pop Out', 'Sac Fly'])]
G_group = df1_clean[df1_clean['Result'].isin(['Single', 'Groundout', 'Lineout', 'Field Error', 'Forceout'])]

dataX = df1_clean['Launch Angle [deg]']
dataY = df1_clean['Hit distance [feet]']

print(len(F_group))
print(len(G_group))

#####(1)散布図の作成
print('(1)散布図の作成')
plt.scatter(F_group['Launch Angle [deg]'],F_group['Hit distance [feet]'],color='red')
plt.scatter(G_group['Launch Angle [deg]'],G_group['Hit distance [feet]'],color='blue')
plt.xlabel('deg')
plt.ylabel('feet')
plt.show()

#####(2)平均、分散、共分散などの算出
print('(2)平均、分散、共分散などの算出')
AllXmean = statistics.mean(df1_clean['Launch Angle [deg]'])
AllYmean = statistics.mean(df1_clean['Hit distance [feet]'])
Fxmean = statistics.mean(F_group['Launch Angle [deg]'])
Fymean = statistics.mean(F_group['Hit distance [feet]'])
Gxmean = statistics.mean(G_group['Launch Angle [deg]'])
Gymean = statistics.mean(G_group['Hit distance [feet]'])
Xvar = np.var(df1_clean['Launch Angle [deg]'])
Yvar = np.var(df1_clean['Hit distance [feet]'])
XYcov = np.cov(df1_clean['Launch Angle [deg]'],df1_clean['Hit distance [feet]'])[0,1]

#Xvar2 = df1_clean['Launch Angle [deg]'].var()
#Xvar3 = statistics.pvariance(df1_clean['Launch Angle [deg]'])
print('Xの平均:',AllXmean , 'Yの平均:',AllYmean)  
print('F群のXの平均:',Fxmean,'F群のYの平均:',Fymean)
print('G群のXの平均:',Gxmean,'G群のYの平均:',Gymean)
print('Xの分散:',Xvar,'Yの分散',Yvar)
print('XYの共分散:',XYcov)

#print(df1_clean)
print(len(df1_clean))
#print(Xvar2)
#print(Xvar3)

#####(3)線形判別分析
print('(3)線形判別分析')

###行列の要素は全て、教科書に乗っ取り計算している。分散＊要素数＝偏差　を用いている。
#S1 = np.cov(df1_clean[['Launch Angle [deg]', 'Hit distance [feet]']].T)
S = np.array([[Xvar*len(df1_clean),XYcov*len(df1_clean)],[XYcov*len(df1_clean),Yvar*len(df1_clean)]])
#Sb1 = np.cov(F_group[['Launch Angle [deg]', 'Hit distance [feet]']].T) * len(F_group) + np.cov(G_group[['Launch Angle [deg]', 'Hit distance [feet]']].T) * len(G_group)
Sb = np.array([[len(F_group)*((Fxmean - AllXmean)**2)+len(G_group)*((Gxmean - AllXmean)**2),len(F_group)*(Fxmean - AllXmean)*(Fymean - AllYmean)+len(G_group)*(Gxmean - AllXmean)*(Gymean - AllYmean)],[len(F_group)*(Fxmean - AllXmean)*(Fymean - AllYmean)+len(G_group)*(Gxmean - AllXmean)*(Gymean - AllYmean),len(F_group)*((Fymean - AllYmean)**2) + len(G_group)*((Gymean - AllYmean)**2)]])
print('行列S:',S)
#print('S1',S1)
print('行列Sb',Sb)
#print('Sb1',Sb1)

#####(4)線形判別関数
print('(4)線形判別関数')
S_inv = np.linalg.inv(S)
Fmean = np.array([Fxmean,Fymean])
Gmean = np.array([Gxmean,Gymean])
w = S_inv.dot(Fmean - Gmean)
print(w)
print('z = ',w[0] ,'*(x - ',AllXmean,')', '+',w[1],'*(y - ',AllYmean,')')

#####(5)新しいデータの分析
def Z(x,y):
    return w[0]*(x - AllXmean) + w[1]*(y - AllYmean)

Double = df2[df2['Result'] == 'Double']

for index, row in Double.iterrows():
    Doublex = row['Launch Angle [deg]']
    Doubley = row['Hit distance [feet]']
    Result = Z(Doublex,Doubley)
    

    if Result > 0:
        print(f'{index}はF群に属す。')
    elif Result < 0:
        print(f'{index}はG群に属す。')
    else:
        print(f'{index}は判別関数上に位置する。')

#####(6)マハラノビス距離による判別
FXvar = np.var(F_group['Launch Angle [deg]'])
FYvar = np.var(F_group['Hit distance [feet]'])
FXYcov = np.cov(F_group['Launch Angle [deg]'],F_group['Hit distance [feet]'])[0,1]

Sf = np.array([[FXvar,FXYcov],[FXYcov,FYvar]])
Sf_inv = np.linalg.inv(Sf)

def Df(x,y):
    point = np.array([x - Fxmean, y - Fymean])
    return np.sqrt(point@(Sf_inv)@point.T)

GXvar = np.var(G_group['Launch Angle [deg]'])
GYvar = np.var(G_group['Hit distance [feet]'])
GXYcov = np.cov(G_group['Launch Angle [deg]'],G_group['Hit distance [feet]'])[0,1]

Sg = np.array([[GXvar,GXYcov],[GXYcov,GYvar]])
Sg_inv = np.linalg.inv(Sg)
print(FXvar,FYvar,FXYcov,GXvar,GYvar,GXYcov)
def Dg(x,y):
    point1 = np.array([x - Gxmean, y - Gymean])
    return np.sqrt(point1@(Sg_inv)@point1.T)


for index, row in Double.iterrows():
    Doublex = row['Launch Angle [deg]']
    Doubley = row['Hit distance [feet]']
    Resultf = Df(Doublex,Doubley)
    Resultg = Dg(Doublex,Doubley)
    print(Resultf)
    print(Resultg)
    if Resultf > Resultg:
        print(f'{index}はG群に属す。')
    elif Resultf < Resultg:
        print(f'{index}はF群に属す。')
    else:
        print(f'{index}は境界線上にある。')
    
#####(7)主成分分析
pca = PCA(n_components=1)
pca.fit(df1_clean[['Launch Angle [deg]', 'Hit distance [feet]']])

# 第1主成分の方向ベクトル
Vector = pca.components_

# 寄与率
Conration = pca.explained_variance_ratio_[0]

print('(7) 主成分分析')
print('方向ベクトル:',Vector)
print('寄与率:',Conration)
