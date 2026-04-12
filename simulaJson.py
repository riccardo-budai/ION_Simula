
import json
from PyQt6.QtCore import QJsonDocument, QJsonParseError

def read_with_qjsondoc(path):
    try:
        with open(path, 'r') as f:
            data = f.read()

        # Converte la stringa JSON in QJsonDocument
        error = QJsonParseError()
        doc = QJsonDocument.fromJson(data.encode('utf-8'), error)

        if doc.isNull():
            print(f"Errore di parsing JSON: {error.errorString()}")
            return None

        # Converte il QJsonDocument in una struttura dati Python
        if doc.isObject():
            # Ritorna un dizionario Python
            return doc.object().toVariantMap()
        elif doc.isArray():
            # Ritorna una lista Python
            return doc.array().toVariantList()

    except Exception as e:
        print(f"Errore di lettura: {e}")
        return None


from PyQt6.QtCore import QJsonDocument
import json  # Usiamo il modulo standard per la conversione


def write_with_qjsondoc(path, python_data):
    # 1. Converte il dizionario/lista Python in QJsonDocument
    if isinstance(python_data, dict):
        doc = QJsonDocument.fromVariant(python_data)
    elif isinstance(python_data, list):
        doc = QJsonDocument.fromVariant(python_data)
    else:
        print("Dati non supportati per QJsonDocument.")
        return

    # 2. Scrive il documento formattato (in formato human-readable)
    json_bytes = doc.toJson(QJsonDocument.JsonFormat.Indented)

    try:
        with open(path, 'wb') as f:
            f.write(json_bytes)
        print(f"Documento scritto con successo in: {path}")
    except Exception as e:
        print(f"Errore di scrittura: {e}")

read_with_qjsondoc('anomaly/cap_1.json')