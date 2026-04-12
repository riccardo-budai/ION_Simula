import fitz

pdf_path = "help_docs/paper/Tut_Dwave.pdf"  # Assicurati che il percorso sia giusto

try:
    doc = fitz.open(pdf_path)
    toc = doc.get_toc()  # Estrae la Table of Contents
    doc.close()

    if toc:
        print("EVVIVA! Trovato un indice strutturato:")
        for item in toc:
            # item è una lista: [livello, titolo, pagina]
            livello, titolo, pagina = item
            print(f"Pagina {pagina}: {titolo}")
    else:
        print("PECCATO: Questo PDF non ha un indice interno (Bookmarks).")
        print("Dovremo usare il metodo manuale o analizzare il file Word.")

except Exception as e:
    print(f"Errore: {e}")