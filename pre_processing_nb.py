import pandas as pd



def load_and_pre_process(data_path):
    #data_path = 'raw_data.csv'

    df = pd.read_csv(data_path)

    # pré-processamento
    # o classificador CategoricalNB do scikit learn exige que os atributos categóricos estejam em formato numérico
    # por isso, trocamos cada valor categórico por um atributo numérico, tomando o cuidado de não usar o mesmo número para
    # valores categóricos diferentes.

    df = df.replace("yes",int(1)).replace("no",int(0)).replace("Sometimes",int(2)).replace("Frequently",int(3)).replace("Always",int(4))
    df = df.replace("Public_Transportation",int(0)).replace("Walking",int(1)).replace("Automobile",int(2)).replace("Motorbike",int(3)).replace("Bike",int(4))
    df = df.replace("Male",int(0)).replace("Female",int(1))

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

    return df