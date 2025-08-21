import streamlit as st
from src.retrieval import Retriever
from src.memory import ChatMemory
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="Financial Policy Chatbot", page_icon="📘")

st.title(" Financial Policy Chatbot")
st.write("Ask about the budget, debt, taxation, assets, or superannuation policies.")

# ---------------------------
# Init session state
# ---------------------------
if "retriever" not in st.session_state:
    st.session_state.retriever = Retriever()
if "memory" not in st.session_state:
    st.session_state.memory = ChatMemory(k=6)
if "chat" not in st.session_state:
    st.session_state.chat = []

# ---------------------------
# Concise Answer Generator
# ---------------------------
def make_answer(query, hits):
    if not hits:
        return "I couldn't find that in the policy document."

    # Take top 2 chunks
    texts = [h[1]["text"] for h in hits[:2]]
    combined = " ".join(texts)

    # Split into sentences
    sentences = [s.strip() for s in combined.replace("\n", " ").split(". ") if len(s.split()) > 4]

    # Score with TF-IDF
    vectorizer = TfidfVectorizer().fit([query] + sentences)
    query_vec = vectorizer.transform([query])
    sent_vecs = vectorizer.transform(sentences)
    sims = (sent_vecs @ query_vec.T).toarray().ravel()

    # Select top 1–2 most relevant sentences
    top_idx = np.argsort(-sims)[:2]
    best_sentences = [sentences[i] for i in top_idx]
    summary = " ".join(best_sentences).strip()
    if not summary.endswith("."):
        summary += "."

    # Conversational wrapper
    if "debt" in query.lower():
        answer = f"The policy explains debt management as follows: {summary}"
    elif "interest" in query.lower():
        answer = f"On interest, the document states: {summary}"
    elif "tax" in query.lower() or "gsp" in query.lower():
        answer = f"In terms of taxation, {summary}"
    elif "asset" in query.lower():
        answer = f"Regarding net assets, {summary}"
    elif "superannuation" in query.lower() or "funding" in query.lower():
        answer = f"The superannuation funding target is: {summary}"
    elif "budget" in query.lower() or "surplus" in query.lower():
        answer = f"According to the budget policy, {summary}"
    else:
        answer = f"The policy states: {summary}"

    # Add citation
    first_chunk = hits[0][1]
    answer += f"\n\n_Source: Section {first_chunk.get('section','N/A')}, Page {first_chunk.get('page','?')}_"
    return answer

# ---------------------------
# Chat Input
# ---------------------------
user_input = st.text_input("Your question:", placeholder="e.g. What are the Government’s strategic financial priorities?")
if st.button("Ask") and user_input:
    aug = st.session_state.memory.maybe_augment(user_input)
    hits = st.session_state.retriever.search(aug, top_k=4)
    answer = make_answer(aug, hits)
    st.session_state.chat.append(("You", user_input))
    st.session_state.chat.append(("Bot", answer))
    st.session_state.memory.add(user_input, answer)

# ---------------------------
# Display Chat
# ---------------------------
for role, msg in st.session_state.chat:
    if role == "You":
        st.markdown(f"**You:** {msg}")
    else:
        st.markdown(f"**Bot:** {msg}")
