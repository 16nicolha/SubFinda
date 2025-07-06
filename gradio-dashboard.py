import gradio as gr
import pandas as pd
from docx import Document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os
import re
from openai import OpenAI
from langchain_core.embeddings import Embeddings

# === Load environment variables ===
load_dotenv()

# === Set up key===
client = OpenAI(
    api_key=os.getenv(""),
    project=os.getenv("")
)

# === Embedding wrapper ===
class MyOpenAIEmbedding(Embeddings):
    def embed_documents(self, texts):
        return [client.embeddings.create(input=t, model="text-embedding-3-small").data[0].embedding for t in texts]
    def embed_query(self, text):
        return client.embeddings.create(input=text, model="text-embedding-3-small").data[0].embedding

# === Load magazine data ===
mags = pd.read_csv("final_mags.csv")
mags["row_number"] = mags["row_number"].astype(pd.Int64Dtype())  # Handle float → int + NaN safely
mags = mags.dropna(subset=["row_number"])  # Drop NaNs
mags["row_number"] = mags["row_number"].astype(int)
mags["large_thumbnail"] = mags["url of thumbnail"].fillna("cover-not-found.jpg") + "&fife=w800"

# === Load tagged descriptions and build vector DB ===
raw_documents = TextLoader("tagged_description.txt", encoding="utf-8").load()
text_splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator="\n")
documents = text_splitter.split_documents(raw_documents)

embedding = MyOpenAIEmbedding()
db_mags = Chroma.from_documents(documents, embedding=embedding)

print("Vector DB built with", len(db_mags.get()['documents']), "documents.")

# === File reader ===
def extract_text(file):
    try:
        if file.name.endswith(".txt"):
            with open(file.name, "r", encoding="utf-8") as f:
                return f.read()
        elif file.name.endswith(".docx"):
            doc = Document(file.name)
            return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        print("Error reading file:", e)
    return None

# 1. Load and preprocess your documents
documents = text_splitter.split_documents(raw_documents)

# 2. Set up the embedding
embedding = MyOpenAIEmbedding()

# 3. Create the vector DB (make sure this is before your functions use it)
db = Chroma.from_documents(documents, embedding=embedding)

# === Semantic matching ===
def retrieve_recommendations_from_text(query, top_k=16):
    docs = db.similarity_search(query, k=top_k * 2)

    row_ids = []
    for doc in docs:
        first_token = doc.page_content.strip().split()[0].replace('"', '')
        try:
            row_ids.append(int(first_token))
        except ValueError:
            continue

    print("Row IDs from vector DB:", row_ids)

    if not row_ids:
        print("⚠️ No valid row numbers extracted.")
        return pd.DataFrame()

    # Ensure row_number is clean integer
    mags_clean = mags.dropna(subset=["row_number"]).copy()
    mags_clean["row_number"] = mags_clean["row_number"].astype(int)

    matches = mags_clean[mags_clean["row_number"].isin(row_ids)]

    # Preserve ranking order and avoid duplicates
    seen = set()
    ranked_matches = []
    for rid in row_ids:
        if rid in seen:
            continue
        seen.add(rid)
        match = matches[matches["row_number"] == rid]
        if not match.empty:
            ranked_matches.append(match)

    if not ranked_matches:
        print("⚠️ No matching magazines found.")
        return pd.DataFrame()

    final_df = pd.concat(ranked_matches).head(top_k)

    # Optional: filter by genre
    # final_df = final_df[final_df["genres"].str.contains("Horror", na=False, case=False)]

    return final_df

# === Render HTML cards ===
def render_gallery_full_card(mags_list, page=0, per_page=4):
    start = page * per_page
    end = start + per_page
    cards = ""
    for mag in mags_list[start:end]:
        name = mag.get("market_name", "Unnamed")
        desc = mag.get("description", "No description.")
        genres = mag.get("genres", "N/A")
        followers = str(mag.get("followers") or "N/A")
        year = str(mag.get("founded") or "N/A")
        country = mag.get("country", "N/A")
        rate = str(mag.get("acceptance rate") or "N/A")
        link = mag.get("submission_guidelines", "#")
        thumb = mag.get("large_thumbnail", "cover-not-found.jpg")

        cards += f"""
        <div style="flex: 1 1 100%; padding: 24px; background: white; border-radius: 16px; box-shadow: 0 6px 12px rgba(0,0,0,0.08); margin-bottom: 32px; display: flex; gap: 24px; align-items: flex-start;">
            <img src="{thumb}" alt="{name}" style="width: 240px; height: auto; border-radius: 12px;" />
            <div style="flex: 1;">
                <h2>{name}</h2>
                <p style="margin-bottom: 16px;">{desc}</p>
                <ul style="padding: 0; list-style: none; line-height: 1.8;">
                    <li><strong>Genres:</strong> {genres}</li>
                    <li><strong>Followers:</strong> {followers}</li>
                    <li><strong>Founded:</strong> {year}</li>
                    <li><strong>Country:</strong> {country}</li>
                    <li><strong>Acceptance Rate:</strong> {rate}</li>
                </ul>
                <p><a href="{link}" target="_blank" style="color: #2563eb;">📬 Submit Here</a></p>
            </div>
        </div>
        """
    return f"<div style='display: flex; flex-direction: column; gap: 24px;'>{cards}</div>"

# === Main logic ===
def recommend(uploaded_file):
    file_text = extract_text(uploaded_file)
    if not file_text:
        return "<p>Could not read your file.</p>", [], 0

    matches = retrieve_recommendations_from_text(file_text)
    if matches.empty:
        return "<p>No matches found.</p>", [], 0

    data = [row.to_dict() for _, row in matches.iterrows()]
    html = render_gallery_full_card(data, page=0)
    return html, data, 0

def paginate(direction, mags_data, current_page):
    if not mags_data:
        return "<p>No data available.</p>", current_page

    new_page = max(0, current_page + direction)
    max_page = (len(mags_data) - 1) // 4
    new_page = min(new_page, max_page)
    return render_gallery_full_card(mags_data, new_page), new_page

# === UI ===
with gr.Blocks(theme=gr.themes.Soft()) as dashboard:
    gr.Markdown("# 📘SubFinda")
    gr.Markdown("Upload a `.txt` or `.docx` story and get personalized magazine matches:")

    with gr.Row():
        file_input = gr.File(label="Upload your story", file_types=[".txt", ".docx"])
        submit_btn = gr.Button("🔍 Find Matches")

    gallery_html = gr.HTML()
    hidden_data = gr.State()
    current_page = gr.Number(value=0, visible=False)

    with gr.Row():
        prev_btn = gr.Button("⬅️ Prev")
        next_btn = gr.Button("Next ➡️")

    submit_btn.click(fn=recommend, inputs=file_input, outputs=[gallery_html, hidden_data, current_page])
    prev_btn.click(fn=paginate, inputs=[gr.State(-1), hidden_data, current_page], outputs=[gallery_html, current_page])
    next_btn.click(fn=paginate, inputs=[gr.State(1), hidden_data, current_page], outputs=[gallery_html, current_page])

query = "funny sci-fi story with robots"
docs = db.similarity_search(query, k=5)
for doc in docs:
    print(doc.page_content[:200])

print(documents[0].page_content)
print(documents[1].page_content)
print(documents[2].page_content)
if __name__ == "__main__":
    dashboard.launch()
