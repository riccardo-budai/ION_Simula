import google.generativeai as genai

# 1. Metti qui la tua nuova chiave API
CHIAVE_API = 'AIzaSyCd0zvMqCrVv4LPP15F__5RcZOQfV2h1Tw'
genai.configure(api_key=CHIAVE_API)

print("Sto interrogando i server di Google...\nEcco i modelli che puoi usare per il tuo Tutor:")
print("-" * 40)

# 2. Chiediamo a Google la lista dei modelli
for modello in genai.list_models():
    # Filtriamo solo i modelli in grado di generare testo (generateContent)
    if 'generateContent' in modello.supported_generation_methods:
        print(modello.name)

print("-" * 40)
print("Scegli uno di questi nomi (togliendo 'models/') e usalo nel tuo codice!")