import os

# Función para cargar todos los documentos desde archivos individuales
def load_documents_from_directory(directory: str):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                content = file.read().strip()
                documents.append({"title": filename, "content": content})
    return documents