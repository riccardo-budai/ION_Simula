"""
    schema to view paragraphs of pdf file content
"""
import streamlit as st
import fitz
import os

st.set_page_config(layout="wide", page_title="Auto-PDF Help Desk")

# --- PERCORSO FILE ---
base_dir = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(base_dir, "help_docs", "paper", "Tut_Sep.pdf")


# --- 1. FUNZIONI DI UTILITÀ ---
def get_pdf_toc(pdf_path):
    """Estrae l'indice (Titoli) e i numeri di pagina dal PDF."""
    if not os.path.exists(pdf_path):
        return []
    try:
        doc = fitz.open(pdf_path)
        toc = doc.get_toc()  # Restituisce una lista [livello, titolo, pagina]
        doc.close()
        return toc
    except:
        return []


def get_page_image(pdf_path, page_num):
    """Renderizza la pagina come immagine."""
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_num - 1)  # fitz conta da 0, il TOC conta da 1
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        return pix.tobytes("png")
    except:
        return None


# --- 2. LOGICA PRINCIPALE ---

st.title("🤖 Help Desk Automatico da PDF")

# Carichiamo l'indice (TOC) una volta sola
toc_data = get_pdf_toc(PDF_PATH)

if not toc_data:
    st.error("Nessun indice trovato nel PDF! Assicurati che il PDF abbia i segnalibri.")
else:
    # Creiamo un dizionario semplice { "Titolo Capitolo": NumeroPagina }
    # Filtriamo magari solo i livelli 1 e 2 per non avere troppa roba
    menu_options = {f"{item[1]} (Pag. {item[2]})": item[2] for item in toc_data}

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.header("📄 Navigazione PDF")

        # Il menu a tendina ora si riempie da solo!
        selected_label = st.selectbox("Vai al capitolo:", list(menu_options.keys()))

        target_page = menu_options[selected_label]

        # Mostra l'immagine
        img_data = get_page_image(PDF_PATH, target_page)
        if img_data:
            st.image(img_data, caption=selected_label, width='content')

    with col_right:
        st.header("bibliografia Dinamica")
        st.info("In questo scenario, i riferimenti devono essere generici o linkati al titolo.")

        # Qui potresti avere un dizionario più semplice che mappa
        # parole chiave del titolo a dei link, se necessario.
        st.markdown(f"Stai leggendo la sezione: **{selected_label}**")
        st.markdown("Cerca riferimenti correlati nel database bibliografico...")