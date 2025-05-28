from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load Data & Model
df = pd.read_csv("D:/project-folder/model/metadata.csv")
embeddings = np.load("D:/project-folder/model/name_embeddings.npy")
#model = SentenceTransformer("D:/project-folder/model/sentence_model")
model = SentenceTransformer("all-MiniLM-L6-v2")  # Automatically downloaded and cached
# Clean name function
def clean_name(name):
    return re.sub(r"[^\w\s]", "", name.lower()).strip()

def find_similar_names(query_name, input_dob=None, input_id=None, top_k=5, threshold=0.7):
    # Clean the query name
    name_clean = clean_name(query_name)  # Assuming clean_name function is defined elsewhere
    query_vec = model.encode([name_clean]).astype("float32")

    # Calculate cosine similarity between the query and all stored embeddings
    sims = cosine_similarity(query_vec, embeddings)[0]
    df['cosine_sim'] = sims  # Add cosine similarity to DataFrame

    # Get top_k matches based on cosine similarity
    top_matches = df.nlargest(top_k, 'cosine_sim').copy()

    # Rule-based filters: Match by Date of Birth (dob) and ID
    top_matches['dob_match'] = top_matches['DateLst'].astype(str) == str(input_dob) if input_dob else False
    top_matches['id_match'] = top_matches['IDLst'].astype(str) == str(input_id) if input_id else False

    # Filter out results that do not meet the cosine similarity threshold
    filtered = top_matches[top_matches['cosine_sim'] >= threshold]

    if filtered.empty:
        return "No match found"

    # Return relevant columns as a dictionary of top matches (including NameList, alias_name, etc.)
    return filtered[['NameList', 'alias_name', 'DateLst', 'IDLst', 'cosine_sim', 'dob_match', 'id_match']].to_dict(orient='records')

from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        name = request.form.get("name")
        dob = request.form.get("dob")
        id_val = request.form.get("id")

        # Call the find_similar_names function to get matches
        matches = find_similar_names(name, dob, id_val)

        # Pass the result to the HTML template
        return render_template("index.html", matches=matches)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)


