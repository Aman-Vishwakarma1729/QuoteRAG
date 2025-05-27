import streamlit as st
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import json
import os
import ast
import time
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re

# Page configuration
st.set_page_config(
    page_title="QuoteRAG - Intelligent Quote Search",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #4CAF50, #2196F3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .quote-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #4CAF50;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .quote-text {
        font-size: 1.1rem;
        font-style: italic;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    
    .quote-author {
        font-weight: bold;
        color: #34495e;
        text-align: right;
    }
    
    .stats-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
    
    .search-tips {
        background-color: #e8g4fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_data():
    """Load the model, data, and FAISS index with caching"""
    with st.spinner("🔄 Loading AI model and quote database..."):
        # Model loading logic
        model_path = os.path.join(os.getcwd(), 'model_artifacts', 'finetuned_sentence_transformer', 'final_model')
        
        if not os.path.exists(model_path):
            checkpoint_dir = os.path.join(os.getcwd(), 'model_artifacts', 'finetuned_sentence_transformer')
            if os.path.exists(checkpoint_dir):
                checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith('checkpoint-epoch-')]
                if checkpoints:
                    latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))[-1]
                    model_path = os.path.join(checkpoint_dir, latest_checkpoint)
                else:
                    model_path = 'all-MiniLM-L6-v2'
            else:
                model_path = 'all-MiniLM-L6-v2'
        
        try:
            model = SentenceTransformer(model_path)
            model_status = "✅ Fine-tuned model loaded"
        except Exception as e:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            model_status = "⚠️ Using base model (fine-tuned model not available)"
        
        # Load data
        processed_data = os.path.join(os.getcwd(), 'dataset', 'processed_quotes_data_1.csv')
        df = pd.read_csv(processed_data)
        
        # Clean data
        df = df.dropna(subset=['quote', 'tags'])
        df = df[df['quote'].apply(lambda x: isinstance(x, str) and len(x.strip()) > 10)]
        df = df[df['tags'].apply(lambda x: isinstance(x, str) and x.startswith('['))]
        
        def parse_tags(tags_str):
            try:
                tags = ast.literal_eval(tags_str)
                return tags if isinstance(tags, list) else []
            except:
                return []
        
        df['parsed_tags'] = df['tags'].apply(parse_tags)
        df = df[df['parsed_tags'].apply(len) > 0]
        df = df.reset_index(drop=True)
        
        # Load or create FAISS index
        index_path = 'quotes_index.faiss'
        embeddings_path = 'quotes_embeddings.npy'
        
        if os.path.exists(index_path) and os.path.exists(embeddings_path):
            index = faiss.read_index(index_path)
            embeddings = np.load(embeddings_path)
        else:
            quotes = df['quote'].tolist()
            embeddings = model.encode(quotes, convert_to_numpy=True, show_progress_bar=False)
            
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings)
            
            faiss.write_index(index, index_path)
            np.save(embeddings_path, embeddings)
        
        return model, df, index, embeddings, model_status

@st.cache_resource
def load_llm():
    """Load LLM with caching"""
    try:
        llm = pipeline('text-generation', 
                      model='microsoft/DialoGPT-medium', 
                      device=-1,
                      max_length=512)
        return llm, "DialoGPT-medium"
    except:
        llm = pipeline('text-generation', 
                      model='gpt2', 
                      device=-1,
                      max_length=512)
        return llm, "GPT-2"

def search_quotes(query, model, df, index, k=10):
    """Search for relevant quotes"""
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)
    
    retrieved_df = df.iloc[indices[0]]
    results = []
    
    for idx, (_, row) in enumerate(retrieved_df.iterrows()):
        results.append({
            'quote': row['quote'],
            'author': row.get('author', 'Unknown'),
            'tags': row['parsed_tags'],
            'similarity_score': float(1 / (1 + distances[0][idx])),  # Convert distance to similarity
            'distance': float(distances[0][idx])
        })
    
    return results

def generate_response(query, retrieved_quotes, llm):
    """Generate AI response based on retrieved quotes"""
    context_parts = []
    for i, r in enumerate(retrieved_quotes[:3]):  # Use top 3 quotes
        context_parts.append(f"{i+1}. \"{r['quote']}\" - {r['author']}")
    
    context = "\n".join(context_parts)
    
    prompt = f"""Based on these relevant quotes, provide a thoughtful response to the user's query.

Query: {query}

Relevant Quotes:
{context}

Response:"""

    try:
        response = llm(prompt, 
                      max_new_tokens=150,
                      do_sample=True,
                      temperature=0.7,
                      pad_token_id=50256)
        
        generated_text = response[0]['generated_text']
        
        if "Response:" in generated_text:
            answer = generated_text.split("Response:")[-1].strip()
        else:
            answer = generated_text.replace(prompt, "").strip()
            
        return answer
    except Exception as e:
        return f"Here are some relevant quotes I found for your query about '{query}':"

def display_quote_card(quote_data, show_similarity=True):
    """Display a quote in a nice card format"""
    similarity_badge = f"**Similarity: {quote_data['similarity_score']:.1%}**" if show_similarity else ""
    
    st.markdown(f"""
    <div class="quote-card">
        <div class="quote-text">"{quote_data['quote']}"</div>
        <div class="quote-author">— {quote_data['author']}</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        if quote_data['tags']:
            st.write("🏷️ **Tags:**", ", ".join(quote_data['tags'][:5]))
    with col2:
        if show_similarity:
            st.metric("Similarity", f"{quote_data['similarity_score']:.1%}")

def main():
    # Header
    st.markdown('<h1 class="main-header">💬 QuoteRAG</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Intelligent Quote Search & Discovery</p>', unsafe_allow_html=True)
    
    # Load model and data
    model, df, index, embeddings, model_status = load_model_and_data()
    llm, llm_name = load_llm()
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 System Status")
        st.info(model_status)
        st.info(f"🤖 LLM: {llm_name}")
        
        st.header("📊 Database Stats")
        total_quotes = len(df)
        unique_authors = df['author'].nunique() if 'author' in df.columns else 0
        all_tags = [tag for tags in df['parsed_tags'] for tag in tags]
        unique_tags = len(set(all_tags))
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Quotes", total_quotes)
            st.metric("Unique Authors", unique_authors)
        with col2:
            st.metric("Unique Tags", unique_tags)
            avg_tags = len(all_tags) / len(df) if len(df) > 0 else 0
            st.metric("Avg Tags/Quote", f"{avg_tags:.1f}")
        
        st.header("🎯 Search Options")
        search_mode = st.radio(
            "Search Mode:",
            ["🔍 Smart Search", "👤 By Author", "🏷️ By Tag", "📈 Analytics"]
        )
        
        num_results = st.slider("Number of results", 1, 20, 5)
        
        if st.button("🔄 Refresh Data"):
            st.cache_resource.clear()
            st.rerun()
    
    # Main content area
    if search_mode == "🔍 Smart Search":
        st.header("🔍 Smart Quote Search")
        
        # Search tips
        with st.expander("💡 Search Tips", expanded=False):
            st.markdown("""
            <div class="search-tips">
                <strong>Try these search types:</strong><br>
                • <strong>Topic-based:</strong> "quotes about love", "wisdom and life"<br>
                • <strong>Author-specific:</strong> "Einstein quotes about science"<br>
                • <strong>Emotion-based:</strong> "inspirational quotes", "motivational sayings"<br>
                • <strong>Situation-based:</strong> "quotes for difficult times", "success quotes"
            </div>
            """, unsafe_allow_html=True)
        
        # Search interface
        col1, col2 = st.columns([4, 1])
        with col1:
            query = st.text_input(
                "What kind of quotes are you looking for?",
                placeholder="e.g., inspirational quotes about perseverance",
                key="search_query"
            )
        with col2:
            st.write("")  # Spacing
            search_clicked = st.button("🔍 Search", type="primary")
        
        # Example queries
        
        # Search results
        if query and (search_clicked or query):
            with st.spinner("🤔 Thinking and searching..."):
                results = search_quotes(query, model, df, index, num_results)
                ai_response = generate_response(query, results, llm)
            
            # AI Response
            st.header("🤖 AI Response")
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%); 
                        color: white; padding: 1.5rem; border-radius: 10px; margin: 1rem 0;">
                {ai_response}
            </div>
            """, unsafe_allow_html=True)
            
            # Search Results
            st.header(f"📚 Found {len(results)} Relevant Quotes")
            
            for i, result in enumerate(results):
                with st.container():
                    display_quote_card(result)
                    
                    # Additional actions
                    col1, col2, col3 = st.columns([1, 1, 2])
                    with col1:
                        if st.button(f"👍", key=f"like_{i}"):
                            st.success("Liked!")
                    with col2:
                        if st.button(f"📋", key=f"copy_{i}"):
                            st.info("Quote copied to clipboard!")
                    with col3:
                        st.write("")  # Spacing
                    
                    st.divider()
    
    elif search_mode == "👤 By Author":
        st.header("👤 Search by Author")
        
        if 'author' in df.columns:
            authors = sorted(df['author'].dropna().unique())
            selected_author = st.selectbox("Select an author:", [""] + list(authors))
            
            if selected_author:
                author_quotes = df[df['author'] == selected_author].head(num_results)
                
                st.subheader(f"Quotes by {selected_author}")
                st.write(f"Found {len(df[df['author'] == selected_author])} quotes by this author")
                
                for _, row in author_quotes.iterrows():
                    quote_data = {
                        'quote': row['quote'],
                        'author': row['author'],
                        'tags': row['parsed_tags'],
                        'similarity_score': 1.0
                    }
                    display_quote_card(quote_data, show_similarity=False)
                    st.divider()
        else:
            st.warning("Author information not available in the dataset.")
    
    elif search_mode == "🏷️ By Tag":
        st.header("🏷️ Search by Tag")
        
        all_tags = [tag for tags in df['parsed_tags'] for tag in tags]
        tag_counts = Counter(all_tags)
        popular_tags = [tag for tag, count in tag_counts.most_common(50)]
        
        selected_tag = st.selectbox("Select a tag:", [""] + popular_tags)
        
        if selected_tag:
            tag_quotes = df[df['parsed_tags'].apply(
                lambda tags: selected_tag in tags
            )].head(num_results)
            
            st.subheader(f"Quotes tagged with '{selected_tag}'")
            st.write(f"Found {len(df[df['parsed_tags'].apply(lambda tags: selected_tag in tags)])} quotes with this tag")
            
            for _, row in tag_quotes.iterrows():
                quote_data = {
                    'quote': row['quote'],
                    'author': row.get('author', 'Unknown'),
                    'tags': row['parsed_tags'],
                    'similarity_score': 1.0
                }
                display_quote_card(quote_data, show_similarity=False)
                st.divider()
    
    elif search_mode == "📈 Analytics":
        st.header("📈 Quote Database Analytics")
        
        # Tag cloud
        all_tags = [tag for tags in df['parsed_tags'] for tag in tags]
        tag_counts = Counter(all_tags)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏷️ Most Popular Tags")
            top_tags = tag_counts.most_common(15)
            
            if top_tags:
                tags_df = pd.DataFrame(top_tags, columns=['Tag', 'Count'])
                fig = px.bar(tags_df, x='Count', y='Tag', orientation='h',
                           title="Top Tags by Frequency")
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📊 Quote Length Distribution")
            quote_lengths = df['quote'].apply(len)
            
            fig = px.histogram(x=quote_lengths, nbins=30, 
                             title="Distribution of Quote Lengths")
            fig.update_xaxes(title="Characters")
            fig.update_yaxes(title="Number of Quotes")
            st.plotly_chart(fig, use_container_width=True)
        
        # Author statistics
        if 'author' in df.columns:
            st.subheader("👤 Top Authors by Quote Count")
            author_counts = df['author'].value_counts().head(10)
            
            fig = px.bar(x=author_counts.values, y=author_counts.index, 
                        orientation='h', title="Most Quoted Authors")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Random quote of the day
        st.subheader("💎 Random Quote of the Day")
        if st.button("🎲 Get Random Quote"):
            random_quote = df.sample(1).iloc[0]
            quote_data = {
                'quote': random_quote['quote'],
                'author': random_quote.get('author', 'Unknown'),
                'tags': random_quote['parsed_tags'],
                'similarity_score': 1.0
            }
            display_quote_card(quote_data, show_similarity=False)
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #666;">💬 QuoteRAG </p>', 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()