import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Setup NLTK (NLP tools)
nltk.download('punkt')
nltk.download('stopwords')

# --- 1. DATASET: Your FAQ Knowledge Base ---
# You can change these to any topic (e.g., CodeAlpha, a Coffee Shop, etc.)
faq_data = {
    "What is CodeAlpha?": "CodeAlpha is a leading software development company dedicated to driving innovation in emerging technologies.",
    "What internships do you offer?": "We offer internships in AI, Web Development, Java Programming, and Python Development.",
    "How long is the internship?": "The internship typically lasts for 4 weeks, providing hands-on real-world experience.",
    "Do I get a certificate?": "Yes, upon successful completion of the tasks, interns receive a certificate from CodeAlpha.",
    "How do I submit my tasks?": "Tasks should be uploaded to GitHub and the links submitted through the official portal or email provided."
}

questions = list(faq_data.keys())

# --- 2. NLP PREPROCESSING FUNCTION ---
def preprocess(text):
    tokens = word_tokenize(text.lower())
    # Remove punctuation and stop words (the, is, at, etc.)
    tokens = [word for word in tokens if word.isalnum() and word not in stopwords.words('english')]
    return " ".join(tokens)

# --- 3. THE AI MATCHING ENGINE ---
def get_bot_response(user_input):
    # Preprocess all questions + the user's question
    processed_questions = [preprocess(q) for q in questions]
    processed_user_input = preprocess(user_input)
    
    # Vectorization (Turning text into math)
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(processed_questions + [processed_user_input])
    
    # Calculate Similarity between user input and all FAQ questions
    similarities = cosine_similarity(vectors[-1], vectors[:-1])
    
    # Find the best match
    index = similarities.argmax()
    score = similarities[0][index]
    
    if score > 0.3: # Confidence threshold
        return faq_data[questions[index]]
    else:
        return "I'm sorry, I don't have information on that. Try asking about our internships or certificates!"

# --- 4. STREAMLIT UI ---
st.set_page_config(page_title="AI FAQ Chatbot", page_icon="🤖")
st.title("🤖 CodeAlpha FAQ Chatbot")
st.write("Ask me anything about the CodeAlpha internship!")

# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("How do I submit my tasks?"):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get AI response
    response = get_bot_response(prompt)
    
    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
