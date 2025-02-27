import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import wget


url = 'https://storage.yandexcloud.net/academy.ai/practica/fake_news.csv'
wget.download(url)

df = pd.read_csv("./fake_news.csv")
df.info()
df.head()

X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #разделение на обучающую и тестовую выборки

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(X_train) #обучение TfidfVectorizer на обучающем наборе данных
tfidf_test = tfidf_vectorizer.transform(X_test) #использование уже обученного TfidfVectorizer для преобразования тестового набора данных

pac = PassiveAggressiveClassifier(max_iter=50) #инициализация классификатора PassiveAggressiveClassifier с максимальным числом итераций 50
pac.fit(tfidf_train, y_train) #обучение модели с использованием векторизованных данных и меток

y_pred = pac.predict(tfidf_test) #предсказание на тестовом наборе данных
score = accuracy_score(y_test, y_pred) #точность модели
print(f'Точность: {score * 100:.2f}%')

cm = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL']) #матрица ошибок
print("Матрица ошибок: \n", cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pac.classes_)
disp.plot(cmap=plt.cm.Greens)
plt.title('Матрица ошибок')
plt.xlabel('Настоящий набор данных')
plt.ylabel('Предсказанный набор данных')
plt.show()