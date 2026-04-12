import google.generativeai as genai

# Inserisci qui la tua chiave
API_KEY = 'AIzaSyCd0zvMqCrVv4LPP15F__5RcZOQfV2h1Tw'

'''
def check_models():
    try:
        genai.configure(api_key=API_KEY)
        print(f"{'NOME MODELLO':<30} | DESCRIZIONE")
        print("-" * 60)
        nummod = 0
        for m in genai.list_models():
            # Filtriamo solo i modelli che generano testo/chat
            if 'generateContent' in m.supported_generation_methods:
                # Puliamo il nome rimuovendo 'models/'
                clean_name = m.name.replace('models/', '')
                print(f"{nummod} : {clean_name}")
                found = True
                nummod += 1

    except Exception as e:
        print(f"Errore: {e}")


if __name__ == "__main__":
    check_models()
'''
import google.generativeai as genai
import os

# --- CONFIGURAZIONE ---
# Inserisci qui la tua API Key (oppure usa os.getenv se l'hai nelle variabili d'ambiente)
# MY_API_KEY = "INSERISCI_QUI_LA_TUA_API_KEY"

genai.configure(api_key=API_KEY)

# --- LA TUA LISTA DI MODELLI ---
# Ho inserito la tua lista in un array per poterli selezionare facilmente
available_models = [
    "gemini-2.5-flash",  # 0
    "gemini-2.5-pro",  # 1
    "gemini-2.0-flash-exp",  # 2
    "gemini-2.0-flash",  # 3 (Molto stabile e veloce)
    "gemini-2.0-flash-001",  # 4
    "gemini-2.0-flash-lite-001",  # 5
    "gemini-2.0-flash-lite",  # 6
    "gemini-2.0-flash-lite-preview-02-05",  # 7
    "gemini-2.0-flash-lite-preview",  # 8
    "gemini-2.0-pro-exp",  # 9 (Molto potente)
    "gemini-2.0-pro-exp-02-05",  # 10
    "gemini-exp-1206",  # 11
    "gemini-2.5-flash-preview-tts",  # 12
    "gemini-2.5-pro-preview-tts",  # 13
    "learnlm-2.0-flash-experimental",  # 14
    "gemma-3-1b-it",  # 15 (Modelli open source hostati)
    "gemma-3-4b-it",  # 16
    "gemma-3-12b-it",  # 17
    "gemma-3-27b-it",  # 18
    "gemma-3n-e4b-it",  # 19
    "gemma-3n-e2b-it",  # 20
    "gemini-flash-latest",  # 21
    "gemini-flash-lite-latest",  # 22
    "gemini-pro-latest",  # 23
    "gemini-2.5-flash-lite",  # 24
    "gemini-2.5-flash-image-preview",  # 25
    "gemini-2.5-flash-image",  # 26
    "gemini-2.5-flash-preview-09-2025",  # 27
    "gemini-2.5-flash-lite-preview-09-2025",  # 28
    "gemini-3-pro-preview",  # 29 (Wow!)
    "gemini-3-pro-image-preview",  # 30
    "nano-banana-pro-preview",  # 31 (Curioso!)
    "gemini-robotics-er-1.5-preview",  # 32
    "gemini-2.5-computer-use-preview-10-2025"  # 33
]


def test_model(model_index):
    # Controllo validità indice
    if model_index < 0 or model_index >= len(available_models):
        print("Indice non valido.")
        return

    model_name = available_models[model_index]
    print(f"\n--- Testando il modello: {model_name} ---")

    try:
        # 1. Inizializzazione del modello
        model = genai.GenerativeModel(model_name)

        # 2. Generazione del contenuto
        # Nota: per modelli "image" o "tts" la chiamata potrebbe dover cambiare,
        # ma per i modelli testuali standard questa è la procedura.
        response = model.generate_content("Raccontami una barzelletta sulla programmazione in una frase.")

        # 3. Stampa risultato
        print(f"Risposta:\n{response.text}")
        print("------------------------------------------------")

    except Exception as e:
        print(f"ERRORE: Non è stato possibile usare {model_name}.")
        print(f"Dettaglio: {e}")
        print(
            "Nota: Alcuni modelli in lista potrebbero essere riservati (whitelisted) o richiedere parametri specifici.")


# --- ESECUZIONE ---
if __name__ == "__main__":
    # Esempio: Testiamo Gemini 2.0 Flash (indice 3 della tua lista)
    test_model(3)

    # Esempio: Testiamo Gemini 2.0 Pro Experimental (indice 9)
    test_model(9)

    # Puoi decommentare qui sotto per provare un modello "futuristico" (potrebbe dare errore se non hai accesso)
    # test_model(0) # Gemini 2.5 Flash