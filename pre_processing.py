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

    
    return df