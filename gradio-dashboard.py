import gradio as gr
import pandas as pd
from docx import Document  # For reading .docx files
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os
import docx


load_dotenv()

# === Load and prepare database ===
mags = pd.read_csv("cleaned_Mags.csv")
raw_documents = TextLoader("tagged_description.txt", encoding="utf-8").load()
text_splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator="\n")
documents = text_splitter.split_documents(raw_documents)
db_mags = Chroma.from_documents(documents, embedding=OpenAIEmbeddings())


# === Utility: Extract text from file ===
def extract_text(file):
    if file.name.endswith(".txt"):
        with open(file.name, "r", encoding="utf-8") as f:
            return f.read()
    elif file.name.endswith(".docx"):
        doc = Document(file.name)
        return "\n".join([para.text for para in doc.paragraphs])
    else:
        return None


# === Semantic Recommendation Function ===
def retrieve_recommendations_from_text(text, initial_top_k=50, final_top_k=8):
    recs = db_mags.similarity_search(text, k=initial_top_k)
    mags_list = [
        int(doc.page_content.strip().split()[0])
        for doc in recs
        if doc.page_content.strip().split()[0].isdigit()
    ]
    return mags[mags["row_number"].isin(mags_list)].head(final_top_k)


# === Main Interface Function ===
def recommend_from_file(uploaded_file):
    try:
        file_text = extract_text(uploaded_file)
        if not file_text:
            return [("❌", "Unsupported file type. Please upload a .txt or .docx file.")]
    except Exception as e:
        return [("❌", f"Error reading file: {e}")]

    recommendations = retrieve_recommendations_from_text(file_text)

    results = []
    for _, row in recommendations.iterrows():
        name = row.get("name", "Untitled Magazine")
        snippet = " ".join(row["description"].split()[:30]) + "..."
        results.append((None, f"**{name}**\n\n{snippet}"))

    return results


# === Gradio UI ===
with gr.Blocks(theme=gr.themes.Soft()) as dashboard:
    gr.Markdown("# 📝 Magazine Submission Recommender")
    gr.Markdown("Upload a `.txt` or `.docx` file and get personalized submission suggestions.")

    with gr.Row():
        file_input = gr.File(label="Upload your story", file_types=[".txt", ".docx"])
        submit_button = gr.Button("🔍 Find Matches")

    output = gr.Gallery(label="Recommended Magazines", columns=1, rows=6)

    submit_button.click(fn=recommend_from_file, inputs=file_input, outputs=output)


if __name__ == "__main__":
    dashboard.launch()
