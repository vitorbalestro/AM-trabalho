from pre_processing import load_and_pre_process
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from pre_processing import load_and_pre_process
from sklearn import svm
from sklearn.naive_bayes import CategoricalNB



data_path = "raw_data.csv"
attributes = ["Age","Gender","Height","Weight","CALC","FAVC","FCVC","NCP","SCC","SMOKE","CH2O","family_history_with_overweight","FAF","TUE","CAEC","MTRANS"]


df = load_and_pre_process(data_path)

X = df[attributes]
y = df["NObeyesdad"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

fs = SelectKBest(score_func=chi2, k=8)
fs.fit(X_train, y_train)
X_train_fs = fs.transform(X_train)
X_test_fs = fs.transform(X_test)

clf = svm.SVC(C=1,kernel="linear")
clf = clf.fit(X_train_fs,y_train)

# Fazer previsões
y_pred = clf.predict(X_test_fs)

# Avaliar o modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia: {accuracy}')

conf_matrix = confusion_matrix(y_pred, y_test)
print(conf_matrix)

