from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import unidecode
import re
from datetime import datetime 
from rapidfuzz import fuzz

app = Flask(__name__)

# Load Data & Model
df = pd.read_csv("D:/project-folder1/model/metadata.csv")
embeddings = np.load("D:/project-folder1/model/name_embeddings.npy")
model = SentenceTransformer("D:/project-folder1/model/sentence_model")
# Load FAISS index
index = faiss.read_index("D:/project-folder1/model/faiss.index")

#model = SentenceTransformer("all-MiniLM-L6-v2")  # Automatically downloaded and cached
# Clean name function
import unidecode

def clean_text(text):
    if pd.isna(text):
        return ""
    text = unidecode.unidecode(text)
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = text.strip()
    return text


def decide_final_match(row, threshold):

    
    if row['cosine_sim'] >= threshold:
        if row['dob_match'] or row['id_match']:
            return True
    return False
    
def find_similar_names(query_name, input_dob=None, input_id=None, top_k=5, threshold=0.7,fuzzy_threshold=0.7):
    # Clean & encode query
    name_clean = clean_text(query_name)
    query_vec = model.encode([name_clean]).astype("float32")
    query_vec = np.array(query_vec, dtype='float32')
    if query_vec.ndim == 1:
        query_vec = query_vec.reshape(1, -1)
    dob_str = None
    if input_dob:
        dob_str = pd.to_datetime(input_dob).date().strftime('%Y-%m-%d')
    # Get nearest neighbors to determine cluster
    D, I = index.search(query_vec, 5)
    nearest_cluster = df.iloc[I[0]]['cluster_id'].mode()[0]  # Most common cluster among nearest

    # Filter the dataset to the same cluster
    df_cluster = df[df['cluster_id'] == nearest_cluster].copy()

    # Compute similarity only in that cluster
    cluster_embeddings = embeddings[df['cluster_id'] == nearest_cluster]
    sims = cosine_similarity(query_vec, cluster_embeddings)[0]
    df_cluster['cosine_sim'] = sims

      # Add fuzzy partial ratio
    df_cluster['fuzzy_score'] = df_cluster['alias_name'].apply(
        lambda x: fuzz.partial_ratio(name_clean, clean_text(str(x)))
    )

    # Apply DOB and ID matching
    df_cluster['dob_match'] = df_cluster['DateLst'].astype(str) == dob_str if dob_str else False
    df_cluster['id_match'] = df_cluster['IDLst'].astype(str) == str(input_id) if input_id else False

    # Filter with both thresholds
    filtered = df_cluster[
        (df_cluster['cosine_sim'] >= threshold) | (df_cluster['fuzzy_score'] >= fuzzy_threshold)
    ].copy()

    filtered = df_cluster[
        (df_cluster['cosine_sim'] >= threshold) & 
        (df_cluster['fuzzy_score'] >= fuzzy_threshold)
    ].nlargest(top_k, 'cosine_sim')

    # Decision logic
    def decide_final_match(row):
        if row['cosine_sim'] >= threshold and (row['dob_match'] or row['id_match']):
            return "Strong Match"
        elif row['cosine_sim'] >= threshold or row['fuzzy_score'] >= fuzzy_threshold:
            return "Probable Match"
        else:
            return "Weak or No Match"

    filtered['match_decision'] = filtered.apply(decide_final_match, axis=1)

    return filtered[['NameList', 'alias_name', 'DateLst', 'IDLst', 'cosine_sim', 'fuzzy_score', 'dob_match', 'id_match', 'match_decision']].to_dict(orient='records')

from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index_route():
    if request.method == "POST":
        name = request.form.get("name")
        dob = request.form.get("dob")
        id_val = request.form.get("id")

        input_dob = None
        if dob:
            try:
                input_dob = datetime.strptime(dob, "%Y-%m-%d").date()
            except ValueError:
                input_dob = None

        # Pass the *correctly parsed* input_dob to your function
        matches = find_similar_names(name, input_dob, id_val)

        return render_template("index.html", matches=matches)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)


