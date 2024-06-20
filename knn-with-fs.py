from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from pre_processing import load_and_pre_process
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, chi2


data_path = "raw_data.csv"
attributes = ["Age","Gender","Height","Weight","CALC","FAVC","FCVC","NCP","SCC","SMOKE","CH2O","family_history_with_overweight","FAF","TUE","CAEC","MTRANS"]

classes = {"Insufficient_Weight": 0, "Normal_Weight": 1, "Obesity_Type_I": 2, "Obesity_Type_II": 3, "Obesity_Type_III": 4, "Overweight_Level_I": 5, "Overweight_Level_II": 6}


df = load_and_pre_process(data_path)

# a linha abaixo transforma as entradas da coluna objetivo em inteiros, de acordo com o dicionário de classes acima.
df["NObeyesdad"] = df["NObeyesdad"].apply(lambda x: int(classes[x]))

scaler = StandardScaler()

X = df[attributes]
y = df["NObeyesdad"]

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)

fs = SelectKBest(score_func=chi2, k=8)
fs.fit(X_train, y_train)

X_train_fs = fs.transform(X_train)

X_test_fs = fs.transform(X_test)


clf = KNeighborsClassifier(algorithm="brute", n_neighbors=1,metric="cityblock")
clf.fit(X_train_fs,y_train)

# Fazer previsões
y_pred = clf.predict(X_test_fs)

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