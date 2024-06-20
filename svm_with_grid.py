from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from pre_processing import load_and_pre_process
from feature_selection import feature_selection
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay

data_path = "raw_data.csv"
attributes = ["Age","Gender","Height","Weight","CALC","FAVC","FCVC","NCP","SCC","SMOKE","CH2O","family_history_with_overweight","FAF","TUE","CAEC","MTRANS"]


df = load_and_pre_process(data_path)

attributes = feature_selection(attributes,df)

X = df[attributes]
y = df["NObeyesdad"]

parameters = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    #'degree': [2, 3, 4],  # Relevante apenas para kernel 'poly'
    #'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],  # Relevante para 'rbf', 'poly' e 'sigmoid'
    #'coef0': [0, 0.1, 0.5, 1]  # Relevante para 'poly' e 'sigmoid'
}


# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


clf = svm.SVC()


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