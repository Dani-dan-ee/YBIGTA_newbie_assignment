########문제1##########
#데이터 불러오기 및 구조 확인
import seaborn as sns
iris = sns.load_dataset('iris')
print(iris.head())

# Species별 Petal Length에 대한 기술통계량 산출
desc = iris.groupby('species')['petal_length'].describe()
count = iris['species'].value_counts()
print(desc)
print(count)

#시각화
import matplotlib.pyplot as plt
sns.boxplot(x='species', y='petal_length', data=iris)
plt.title("Species_Petal_Length_Boxplot")
plt.show()

#정규성 검정
from scipy.stats import shapiro
for species in iris['species'].unique():
    stat, p = shapiro(iris[iris['species'] == species]['petal_length'])
    print(f"{species}: p-value = {p:.4f}")

#등분산성 검정
from scipy.stats import levene
setosa = iris[iris['species'] == 'setosa']['petal_length']
versicolor = iris[iris['species'] == 'versicolor']['petal_length']
virginica = iris[iris['species'] == 'virginica']['petal_length']
stat, p = levene(setosa, versicolor, virginica)
print(f"Levene p-value = {p:.4f}")

#ANOVA 실행
from scipy.stats import f_oneway
f_stat, p = f_oneway(setosa, versicolor, virginica)
print(f"F-statistic = {f_stat:.4f}, p-value = {p:.4f}")

#사후검정
from statsmodels.stats.multicomp import pairwise_tukeyhsd
tukey = pairwise_tukeyhsd(endog=iris['petal_length'], groups=iris['species'], alpha=0.05)
print(tukey.summary())


##############문제2################
##1.데이터 로드 및 기본 탐색
import pandas as pd
import numpy as np

# 데이터 로드
df = pd.read_csv('C:/Users/da396/creditcard.csv')

# 데이터 구조 확인
print(df.head())
print(df.info())
print(df.describe())

# Class 비율 확인
print('정상 거래(Class=0) 건수:', (df['Class'] == 0).sum())
print('사기 거래(Class=1) 건수:', (df['Class'] == 1).sum())
print('전체 거래 중 사기 거래 비율:', round((df['Class'] == 1).mean()*100, 4), '%')

## 2.샘플링
# 사기 거래는 모두 유지
fraud = df[df['Class'] == 1]

# 정상 거래는 10,000건만 무작위 샘플링 
normal = df[df['Class'] == 0].sample(n=10000, random_state=42)

# 두 데이터 합치기
df_sampled = pd.concat([fraud, normal])

# Class 비율 재확인
print(df_sampled['Class'].value_counts(normalize=True))
print(df_sampled['Class'].value_counts())
print('샘플링 후 전체 거래 중 사기 거래 비율:', round((df_sampled['Class'] == 1).mean()*100, 4), '%')

## 3.데이터 전처리
from sklearn.preprocessing import StandardScaler

# Amount 표준화 (Amount_Scaled로 대체)
scaler = StandardScaler()
df_sampled['Amount_Scaled'] = scaler.fit_transform(df_sampled[['Amount']])
df_sampled = df_sampled.drop('Amount', axis=1)

# X, y 분리
X = df_sampled.drop('Class', axis=1)
y = df_sampled['Class']

## 4.학습 데이터와 테스트 데이터 분할
from sklearn.model_selection import train_test_split

# 학습:테스트 = 8:2
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print('학습셋 클래스 비율:\n', y_train.value_counts(normalize=True))
print('테스트셋 클래스 비율:\n', y_test.value_counts(normalize=True))

## 5.SMOTE 적용
from imblearn.over_sampling import SMOTE

# SMOTE 적용 전
print('SMOTE 적용 전 클래스 분포:', y_train.value_counts())

# SMOTE 적용 (random_state=42)
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# SMOTE 적용 후
print('SMOTE 적용 후 클래스 분포:', y_train_sm.value_counts())

## 6.모델 학습 및 평가
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, average_precision_score

# 모델 학습
clf = RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=100)
clf.fit(X_train_sm, y_train_sm)

# 예측값 및 예측 확률
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

# 평가지표 출력
print('classification_report\n', classification_report(y_test, y_pred, digits=4))
print('PR-AUC:', average_precision_score(y_test, y_proba))

# threshold 조정
custom_threshold = 0.2
y_pred_custom = (y_proba >= custom_threshold).astype(int)
print(classification_report(y_test, y_pred_custom, digits=4))
print('조정 후 PR-AUC:', average_precision_score(y_test, y_proba))


