import subprocess
import os
import platform
import fitz  # PyMuPDF


def trova_libreoffice():
    """Cerca l'eseguibile di LibreOffice."""
    sistema = platform.system()
    if sistema == "Windows":
        percorsi = [
            r"C:\Program Files\LibreOffice\program\soffice.exe",
            r"C:\Program Files (x86)\LibreOffice\program\soffice.exe"
        ]
        for p in percorsi:
            if os.path.exists(p): return p
    elif sistema == "Linux":
        return "libreoffice"
    elif sistema == "Darwin":
        return "/Applications/LibreOffice.app/Contents/MacOS/soffice"
    return None


def converti_in_pdf(input_docx, output_folder):
    """Converte DOCX in PDF usando LibreOffice headless."""
    soffice = trova_libreoffice()
    if not soffice:
        return False, "LibreOffice non trovato."

    input_abs = os.path.abspath(input_docx)
    output_abs = os.path.abspath(output_folder)

    # Crea la cartella se non esiste
    os.makedirs(output_abs, exist_ok=True)

    cmd = [
        soffice, "--headless", "--convert-to", "pdf",
        "--outdir", output_abs, input_abs
    ]

    print(f"🔄 [1/2] Conversione in corso: {os.path.basename(input_docx)}...")
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Nome file atteso
        nome_base = os.path.splitext(os.path.basename(input_docx))[0]
        pdf_path = os.path.join(output_abs, nome_base + ".pdf")

        if os.path.exists(pdf_path):
            return True, pdf_path
        else:
            return False, "File PDF non generato."
    except Exception as e:
        return False, str(e)


def auto_genera_toc(pdf_path):
    """
    Analizza il PDF, cerca testi grandi (Titoli) e crea il TOC.
    """
    print(f"🔨 [2/2] Analisi struttura e rigenerazione Indice (TOC)...")

    doc = fitz.open(pdf_path)
    toc_nuovo = []

    # 1. Analizziamo la grandezza dei font per capire cosa è un titolo
    font_counts = {}

    # Campioniamo le prime 10 pagine per trovare la dimensione del testo "normale"
    for page in doc[:10]:
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if "lines" in b:
                for line in b["lines"]:
                    for span in line["spans"]:
                        size = round(span["size"], 1)
                        text = span["text"].strip()
                        if len(text) > 1:  # Ignora caratteri vuoti
                            font_counts[size] = font_counts.get(size, 0) + len(text)

    # Assumiamo che il font più frequente sia il "Corpo del testo"
    if not font_counts:
        print("⚠️ PDF vuoto o illeggibile.")
        return

    dimensione_corpo = max(font_counts, key=font_counts.get)
    soglia_titolo = dimensione_corpo + 1.5  # Un titolo deve essere almeno 1.5pt più grande del corpo

    print(f"   -> Dimensione testo normale: {dimensione_corpo}pt")
    print(f"   -> Cerco titoli > {soglia_titolo}pt")

    # 2. Scansione vera e propria per creare il TOC
    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        # Ordiniamo i blocchi verticalmente (per leggere dall'alto in basso)
        blocks.sort(key=lambda b: b["bbox"][1])

        for b in blocks:
            if "lines" in b:
                for line in b["lines"]:
                    for span in line["spans"]:
                        if span["size"] > soglia_titolo:
                            text = span["text"].strip()
                            # Filtri di pulizia (evita numeri di pagina o testi spuri)
                            if len(text) > 3 and len(text) < 100:
                                # Aggiungi al TOC: [Livello, Titolo, Pagina]
                                # Nota: fitz vuole la pagina base-1 nel TOC
                                toc_nuovo.append([1, text, page_num + 1])

    if toc_nuovo:
        doc.set_toc(toc_nuovo)
        # Salviamo su un file temporaneo e rinominiamo per evitare corruzione
        temp_name = pdf_path.replace(".pdf", "_temp.pdf")
        doc.save(temp_name)
        doc.close()
        os.replace(temp_name, pdf_path)
        print(f"✅ Indice creato con {len(toc_nuovo)} voci!")
    else:
        print("⚠️ Nessun titolo rilevato automaticamente.")
        doc.close()


# --- BLOCCO PRINCIPALE ---
if __name__ == "__main__":
    # Configurazione
    file_word = "help_docs/paper/Tut_Sep.docx"
    cartella_output = "help_docs/paper/"

    # 1. Converti
    successo, risultato = converti_in_pdf(file_word, cartella_output)

    if successo:
        pdf_generato = risultato
        print(f"✅ PDF base creato: {pdf_generato}")

        # 2. Ripara (Genera il TOC automaticamente)
        auto_genera_toc(pdf_generato)

        print("\n🎉 Finito! Ora il tuo PDF ha un indice navigabile.")
    else:
        print(f"❌ Errore: {risultato}")