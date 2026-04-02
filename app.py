import nltk

# These MUST come before any other logic
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
# --- 1. EXPANDED KNOWLEDGE BASE ---
faq_data = {
    "What is CodeAlpha?": "CodeAlpha is a software development company that provides virtual internship opportunities in AI, Web Dev, and more.",
    "What is the deadline for tasks?": "The deadline for all tasks in this internship batch is May 10, 2026.",
    "How many tasks are there?": "There are typically 3 to 4 tasks assigned depending on your specific internship domain.",
    "Do I get a certificate?": "Yes, a completion certificate is issued after all tasks are successfully submitted and reviewed.",
    "Where do I submit my work?": "You must submit your GitHub repository links through the official CodeAlpha submission portal or via the instructions in your selection email.",
    "What are the internship domains?": "CodeAlpha offers internships in Artificial Intelligence, Python Development, Java, Web Development, and Mobile App Development.",
    "Is this internship paid?": "CodeAlpha primarily offers unpaid virtual internships focused on skill-building and portfolio development.",
    "How do I contact support?": "You can reach out to CodeAlpha through their official LinkedIn page or the contact email provided in your offer letter.",
    "Can I use AI to help with tasks?": "Yes, using AI as a learning assistant is encouraged, but you must understand the code and be able to explain it.",
    "What happens if I miss the deadline?": "Late submissions might not be eligible for a certificate. It is best to complete tasks before May 10."
}

questions = list(faq_data.keys())

# --- 2. NLP PREPROCESSING ---
def preprocess(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w.isalnum() and w not in stop_words]
    return " ".join(tokens)

# --- 3. THE CHAT LOGIC ---
def get_bot_response(user_input):
    if not user_input.strip():
        return "Please type a question!"

    # Prepare data for math comparison
    processed_docs = [preprocess(q) for q in questions]
    processed_user = preprocess(user_input)
    
    # If the user input is empty after cleaning (e.g., just "hi")
    if not processed_user:
        return "Hello! Ask me anything about the CodeAlpha internship, tasks, or deadlines."

    # Math: Vectorization & Cosine Similarity
    vectorizer = TfidfVectorizer()
    all_docs = processed_docs + [processed_user]
    tfidf_matrix = vectorizer.fit_transform(all_docs)
    
    # Compare the last vector (user) with all previous vectors (FAQs)
    probs = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    best_match_idx = probs.argmax()
    confidence = probs[0][best_match_idx]

    # Confidence Threshold: Only answer if similarity is > 0.2
    if confidence > 0.2:
        return faq_data[questions[best_match_idx]]
    else:
        return "I'm not quite sure about that. Try asking about deadlines, certificates, or how to submit tasks!"

# --- 4. STREAMLIT UI ---
st.set_page_config(page_title="CodeAlpha FAQ Bot", page_icon="🤖")
st.title("🤖 CodeAlpha Internship Assistant")
st.markdown("---")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about the deadline..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response = get_bot_response(prompt)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
