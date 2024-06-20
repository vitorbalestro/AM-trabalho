import pandas as pd
import numpy as np
import math
import statistics
from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


data_path = 'raw_data.csv'

df = pd.read_csv(data_path)

# pré-processamento
# o classificador CategoricalNB do scikit learn exige que os atributos categóricos estejam em formato numérico
# por isso, trocamos cada valor categórico por um atributo numérico, tomando o cuidado de não usar o mesmo número para
# valores categóricos diferentes.

df = df.replace("yes",int(1)).replace("no",int(0)).replace("Sometimes",int(2)).replace("Frequently",int(3)).replace("Always",int(4))
df = df.replace("Public_Transportation",int(5)).replace("Walking",int(6)).replace("Automobile",int(7)).replace("Motorbike",int(8)).replace("Bike",int(9))
df = df.replace("Male",int(10)).replace("Female",int(11))

# valores inteiros podem ser lidos como float. corrigimos abaixo

df["NCP"] = df["NCP"].astype("int64")
df["FCVC"] = df["FCVC"].astype("int64")
df["CH2O"] = df["CH2O"].astype("int64")
df["FAF"] = df["FAF"].astype("int64")
df["TUE"] = df["TUE"].astype("int64")

# a divisão de valores de idade segue uma média nos valores, e não nas frequências.
# as faixas obtidas são condizentes com a literatura da gerontologia médica

df['Age'] = df['Age'].apply(lambda x: int(0) if x < 25 else x)
df['Age'] = df['Age'].apply(lambda x: int(1) if x >= 25 and x < 37 else x)
df['Age'] = df['Age'].apply(lambda x: int(2) if x >= 37 and x < 49 else x)
df['Age'] = df['Age'].apply(lambda x: int(3) if x >=49 and x < 71 else x)
df['Age'] = df['Age'].apply(lambda x: int(4) if x >= 71 else x)

df['Weight'] = df['Weight'].apply(lambda x: int(0) if x < 73 else x)
df['Weight'] = df['Weight'].apply(lambda x: int(1) if x >= 73 and x < 106 else x)
df['Weight'] = df['Weight'].apply(lambda x: int(2) if x >= 106 and x < 139 else x)
df['Weight'] = df['Weight'].apply(lambda x: int(3) if x >= 139 and x < 172 else x)
df['Weight'] = df['Weight'].apply(lambda x: int(4) if x >= 172 else x)

df["Height"] = df["Height"].apply(lambda x: int(0) if x < 1.58 else x)
df["Height"] = df["Height"].apply(lambda x: int(1) if x < 1.71 and x >= 1.58 else x)
df["Height"] = df["Height"].apply(lambda x: int(2) if x < 1.84 and x >= 1.71 else x)
df["Height"] = df["Height"].apply(lambda x: int(3) if x < 1.97 and x >= 1.84 else x)
df["Height"] = df["Height"].apply(lambda x: int(4) if x >= 1.97 else x)


# fim do pré-processamento

attributes = ["Age","Gender","Height","Weight","CALC","FAVC","FCVC","NCP","SCC","SMOKE","CH2O","family_history_with_overweight","FAF","TUE","CAEC","MTRANS"]

classes = {"Normal_Weight": 0, "Overweight_Level_I": 1, "Overweight_Level_II": 2, "Obesity_Type_I": 3, "Obesity_Type_II": 4, "Insufficient_Weight": 5, "Obesity_Type_III": 6}

entropy_dict = {}

for attr in attributes:

    values = df[attr].unique()
    entropy_mean = 0.0

    for value in values:
        distrib = [0 for i in range(7)]
        value_lines = df[df[attr] == value].values
        total = np.shape(value_lines)[0]
        for entry in value_lines:
            distrib[classes[entry[-1]]] += 1
    
        prob = [float(distrib[i]/total) for i in range(7)]
    
        entropy = 0.0
        for i in range(7):
            if prob[i] != 0:
                entropy = entropy - prob[i]*math.log(prob[i])
        entropy_mean = entropy_mean + float(entropy / total)       

    entropy_dict[attr] = entropy_mean


values_array = []
for key in entropy_dict.keys():
    values_array.append(entropy_dict[key])

median_entropy = statistics.median(values_array)
below_median_attributes = []

for key in entropy_dict.keys():
    if entropy_dict[key] <= median_entropy:
        below_median_attributes.append(key)

values_array.sort()

upper_threshold = values_array[-2]

below_threshold_attributes = []
for key in entropy_dict.keys():
    if entropy_dict[key] < upper_threshold:
        below_threshold_attributes.append(key)
        print(key)

X = df[below_threshold_attributes]
y = df["NObeyesdad"]


# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


clf = CategoricalNB()
clf = clf.fit(X_train,y_train)

# Fazer previsões
y_pred = clf.predict(X_test)

# Avaliar o modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia: {accuracy}')

conf_matrix = confusion_matrix(y_pred, y_test)
print(conf_matrix)