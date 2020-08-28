import numpy as np

encoder = {1: "APORTES Y PLANILLAS",
1:"CERTIFICADOS, CONSTANCIAS Y EXTRACTOS",
"ACTUALIZACIÓN DE DATOS":3,
"AFILIACIONES Y TRASLADOS":4,
"SIN CATEGORÍA":5,
"RETIROS":6}

def get_tokens(words, stop_words):
    words_list = str(words).split()
    words_list = [word.lower() for word in words_list if word not in stop_words]
    return words_list


def catch(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except:
        return np.zeros(300)


def vectorize(tokens, wordvectors):
    return np.add.reduce([catch(wordvectors.get_vector, token) for token in tokens]) / len(tokens)


def get_vectors(words, wordvectors, stop_words):
    return vectorize(get_tokens(words, stop_words), wordvectors)


def get_result(proba, label):
    if label == 1:
        return "CS393", "INFORMACIÓN DEL AVANCE DEL TRAMITE DE LA PRESTACIÓN ECONÓMICA"
    elif proba[1]>0.3:
        return "CS258", "ACTUALIZACIÓN DE DATOS"
    else:
        return "CS455", "SOLICITUD DE INFORMACION Y GESTION"

