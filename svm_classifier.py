from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from pre_processing import load_and_pre_process
from feature_selection import feature_selection
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score


data_path = "raw_data.csv"
attributes = ["Age","Gender","Height","Weight","CALC","FAVC","FCVC","NCP","SCC","SMOKE","CH2O","family_history_with_overweight","FAF","TUE","CAEC","MTRANS"]


df = load_and_pre_process(data_path)

attributes = feature_selection(attributes,df)

X = df[attributes]
y = df["NObeyesdad"]


# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


clf = svm.SVC(C=1, kernel="linear")
scores = cross_val_score(clf, X, y, cv = 10)

print("Média de acurácia: %0.2f \n Desvio padrão: %0.2f \n Melhor acurácia: %0.2f" % (scores.mean(), scores.std(), scores.max()))

clf = clf.fit(X_train,y_train)

# Fazer previsões
y_pred = clf.predict(X_test)

# Avaliar o modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia: {accuracy}')

conf_matrix = confusion_matrix(y_pred, y_test)
print(conf_matrix)

class_labels = clf.classes_
print(class_labels)

"""disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot()
plt.show()"""

sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted class')
plt.ylabel('True class')
plt.show()