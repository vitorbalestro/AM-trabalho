from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from pre_processing import load_and_pre_process
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay


data_path = "raw_data.csv"
attributes = ["Age","Gender","Height","Weight","CALC","FAVC","FCVC","NCP","SCC","SMOKE","CH2O","family_history_with_overweight","FAF","TUE","CAEC","MTRANS"]

classes = {"Insufficient_Weight": 0, "Normal_Weight": 1, "Obesity_Type_I": 2, "Obesity_Type_II": 3, "Obesity_Type_III": 4, "Overweight_Level_I": 5, "Overweight_Level_II": 6}


df = load_and_pre_process(data_path)

# a linha abaixo transforma as entradas da coluna objetivo em inteiros, de acordo com o dicionário de classes acima.
df["NObeyesdad"] = df["NObeyesdad"].apply(lambda x: int(classes[x]))

scaler = StandardScaler()

X = df[attributes]
y = df["NObeyesdad"]

parameters = {
    'n_neighbors': [1, 5, 10],
    'metric': ['cityblock', 'euclidean', 'cosine','l1', 'l2'],
    #'degree': [2, 3, 4],  # Relevante apenas para kernel 'poly'
    #'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],  # Relevante para 'rbf', 'poly' e 'sigmoid'
    #'coef0': [0, 0.1, 0.5, 1]  # Relevante para 'poly' e 'sigmoid'
}


# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


clf = KNeighborsClassifier()


GS = GridSearchCV(estimator = clf, 
                   param_grid = parameters, 
                   scoring = 'accuracy', 
                   refit = 'accuracy',
                   cv = 10,
                   verbose = 4 ,
                  error_score='raise'
                 )
GS.fit(X_train, y_train)

print(GS.best_params_)
# Fazer previsões
y_pred = GS.predict(X_test)

# Avaliar o modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia: {accuracy}')

conf_matrix = confusion_matrix(y_pred, y_test)
print(conf_matrix)

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot()