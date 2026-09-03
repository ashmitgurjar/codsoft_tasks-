import os
import io
import time
import torch
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.dataset import load_corpus, CharVocab
from src.rnn_model import CharRNN
from src.trainer import RNNTrainer
from src.text_generator import CharTextGenerator
from src.handwriting_renderer import HandwritingRenderer

# Page Configuration
st.set_page_config(
    page_title="Handwritten Text Generation | Char-RNN Engine",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for rich aesthetics
st.markdown(
    """
    <style>
    /* Main Background & Fonts */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        color: #f8fafc;
    }
    
    /* Card Container */
    .custom-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(12px);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    /* Header Gradient */
    .main-title {
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.6rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }
    
    /* Generated Text Container */
    .generated-text-box {
        background-color: #020617;
        border-left: 4px solid #6366f1;
        border-radius: 8px;
        padding: 18px;
        font-family: 'Courier New', Courier, monospace;
        font-size: 1.05rem;
        line-height: 1.6;
        color: #e2e8f0;
        white-space: pre-wrap;
        margin-top: 15px;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.5);
    }
    
    /* Metric Badges */
    .metric-badge {
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 12px;
        padding: 12px 18px;
        text-align: center;
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #818cf8;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

MODEL_DIR = "models"
DEFAULT_CORPUS_PATH = "data/handwritten_corpus.txt"


@st.cache_resource
def ensure_default_model():
    """Ensure a baseline pre-trained model checkpoint exists for instant generation."""
    checkpoint_path = os.path.join(MODEL_DIR, "char_rnn.pt")
    vocab_path = os.path.join(MODEL_DIR, "vocab.json")
    
    if not os.path.exists(checkpoint_path) or not os.path.exists(vocab_path):
        os.makedirs(MODEL_DIR, exist_ok=True)
        text_corpus = load_corpus(DEFAULT_CORPUS_PATH)
        vocab = CharVocab(list(text_corpus))
        model = CharRNN(vocab_size=len(vocab), embed_dim=128, hidden_dim=256, num_layers=2, rnn_type="LSTM")
        trainer = RNNTrainer(model=model, vocab=vocab, lr=0.003)
        trainer.train(text_corpus, epochs=25, batch_size=32, save_dir=MODEL_DIR)
        
    return RNNTrainer.load_checkpoint(checkpoint_path, vocab_path)


def main():
    # Title Section
    st.markdown('<div class="main-title">✍️ Handwritten Text Generation Engine</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Character-Level Recurrent Neural Network (LSTM/GRU) for Text Synthesis & Visual Handwriting Document Rendering</div>',
        unsafe_allow_html=True,
    )

    # Load baseline model
    model, vocab = ensure_default_model()
    text_generator = CharTextGenerator(model=model, vocab=vocab)
    renderer = HandwritingRenderer()

    # Sidebar Controls
    st.sidebar.image("https://img.icons8.com/color/96/000000/signature.png", width=70)
    st.sidebar.title("🎛️ Control Panel")
    st.sidebar.markdown("Configure Char-RNN generation parameters & visual rendering aesthetics.")

    # Main Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "✍️ Text & Handwriting Generator",
        "🚀 Model Training Studio",
        "📊 Dataset & Corpus Explorer",
        "🔬 Architecture Diagnostics",
    ])

    # TAB 1: GENERATOR & RENDERER
    with tab1:
        st.subheader("Generate & Render Handwritten Text")
        
        col_input, col_params = st.columns([1.2, 1])
        
        with col_input:
            prompt_input = st.text_input("Enter Seed Prompt:", value="Handwriting is", help="Initial characters fed into the RNN to warm up state.")
            num_chars_input = st.slider("Characters to Generate:", min_value=50, max_value=1000, value=300, step=25)
            
        with col_params:
            temperature = st.slider("Temperature (Creativity):", min_value=0.1, max_value=1.5, value=0.75, step=0.05, help="Higher values increase randomness/creativity, lower values make output more deterministic.")
            top_k = st.slider("Top-K Sampling (0 = Disabled):", min_value=0, max_value=40, value=0, step=5)
            top_p = st.slider("Top-P Nucleus Sampling (0.0 = Disabled):", min_value=0.0, max_value=1.0, value=0.9, step=0.05)

        st.markdown("---")
        
        # Generation Action
        if st.button("⚡ Generate Handwritten Text", type="primary", use_container_width=True):
            with st.spinner("Generating character sequence with RNN..."):
                start_time = time.time()
                generated_text = text_generator.generate(
                    prompt=prompt_input,
                    num_chars=num_chars_input,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
                inference_time = (time.time() - start_time) * 1000
                st.session_state["last_generated_text"] = generated_text
                st.session_state["inference_time"] = inference_time

        # Display Text Results if available
        if "last_generated_text" in st.session_state:
            gen_text = st.session_state["last_generated_text"]
            inf_time = st.session_state.get("inference_time", 0.0)
            
            # Metrics bar
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            with mcol1:
                st.markdown(f'<div class="metric-badge"><div class="metric-val">{len(gen_text)}</div><div class="metric-lbl">Total Characters</div></div>', unsafe_allow_html=True)
            with mcol2:
                st.markdown(f'<div class="metric-badge"><div class="metric-val">{len(gen_text.split())}</div><div class="metric-lbl">Word Count</div></div>', unsafe_allow_html=True)
            with mcol3:
                st.markdown(f'<div class="metric-badge"><div class="metric-val">{inf_time:.1f} ms</div><div class="metric-lbl">Inference Latency</div></div>', unsafe_allow_html=True)
            with mcol4:
                st.markdown(f'<div class="metric-badge"><div class="metric-val">{inf_time / max(1, len(gen_text)):.2f} ms</div><div class="metric-lbl">Per Token Speed</div></div>', unsafe_allow_html=True)
                
            st.markdown("##### Raw Generated Text Sequence:")
            st.markdown(f'<div class="generated-text-box">{gen_text}</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            st.subheader("🖼️ Visual Handwritten Document Renderer")
            
            rcol1, rcol2, rcol3, rcol4 = st.columns(4)
            with rcol1:
                paper_style = st.selectbox("Paper Background Style:", options=["ruled", "parchment", "clean", "grid"], index=0)
            with rcol2:
                ink_color = st.selectbox("Ink Color:", options=["blue", "black", "red", "sepia", "green"], index=0)
            with rcol3:
                font_size = st.slider("Font Size:", min_value=16, max_value=40, value=26)
            with rcol4:
                line_spacing = st.slider("Line Spacing:", min_value=25, max_value=60, value=38)
                
            # Render handwritten page image
            rendered_image = renderer.render_page(
                text=gen_text,
                paper_style=paper_style,
                ink_color=ink_color,
                font_size=font_size,
                line_spacing=line_spacing,
            )
            
            st.image(rendered_image, caption="Visual Handwritten Document Synthesis", use_container_width=True)
            
            # Download Image Button
            img_buffer = io.BytesIO()
            rendered_image.save(img_buffer, format="PNG")
            img_bytes = img_buffer.getvalue()
            
            st.download_button(
                label="📥 Download Handwritten Page PNG",
                data=img_bytes,
                file_name="generated_handwriting_document.png",
                mime="image/png",
                use_container_width=True,
            )

    # TAB 2: MODEL TRAINING STUDIO
    with tab2:
        st.subheader("🚀 Train Recurrent Neural Network on Handwritten Dataset")
        st.markdown("Train a custom character-level RNN on new dataset examples or the built-in handwriting corpus.")
        
        tcol1, tcol2 = st.columns([1, 1])
        
        with tcol1:
            dataset_source = st.radio("Dataset Source:", ["Built-in Handwriting Corpus", "Upload Custom Text Dataset (.txt)"])
            
            if dataset_source == "Built-in Handwriting Corpus":
                training_text = load_corpus(DEFAULT_CORPUS_PATH)
                st.info(f"Loaded built-in handwriting corpus: **{len(training_text):,} characters**")
            else:
                uploaded_file = st.file_uploader("Upload Text File", type=["txt", "csv"])
                if uploaded_file is not None:
                    training_text = uploaded_file.read().decode("utf-8", errors="ignore")
                    st.success(f"Uploaded custom dataset: **{len(training_text):,} characters**")
                else:
                    training_text = load_corpus(DEFAULT_CORPUS_PATH)
                    st.warning("No file uploaded yet. Defaulting to built-in dataset.")

            rnn_type = st.selectbox("RNN Architecture:", options=["LSTM", "GRU", "RNN"], index=0)
            epochs_input = st.slider("Training Epochs:", min_value=5, max_value=100, value=25, step=5)
            
        with tcol2:
            hidden_dim = st.select_slider("Hidden Dimension:", options=[64, 128, 256, 512], value=256)
            num_layers = st.slider("Recurrent Layers:", min_value=1, max_value=3, value=2)
            learning_rate = st.select_slider("Learning Rate:", options=[0.0005, 0.001, 0.002, 0.005, 0.01], value=0.002)
            batch_size = st.select_slider("Batch Size:", options=[16, 32, 64, 128], value=32)

        if st.button("🚀 Launch Model Training", type="primary", use_container_width=True):
            st.markdown("---")
            st.markdown("#### Training Progress")
            
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            sample_box = st.empty()
            
            # Metrics charts layout
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                loss_chart_spot = st.empty()
            with chart_col2:
                perp_chart_spot = st.empty()

            custom_vocab = CharVocab(list(training_text))
            new_model = CharRNN(
                vocab_size=len(custom_vocab),
                embed_dim=128,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                rnn_type=rnn_type,
            )
            trainer = RNNTrainer(model=new_model, vocab=custom_vocab, lr=learning_rate)
            
            epoch_losses = []
            epoch_perps = []
            
            def ui_callback(epoch, total_epochs, loss, perplexity, sample_text):
                progress_bar.progress(epoch / total_epochs)
                status_text.markdown(f"**Epoch [{epoch}/{total_epochs}]** — Loss: `{loss:.4f}` | Perplexity: `{perplexity:.2f}`")
                sample_box.markdown(f"**Live Generation Sample:** *\"{sample_text[:100]}...\"*")
                
                epoch_losses.append(loss)
                epoch_perps.append(perplexity)
                
                # Update charts
                loss_df = pd.DataFrame({"Epoch": range(1, len(epoch_losses) + 1), "Loss": epoch_losses})
                perp_df = pd.DataFrame({"Epoch": range(1, len(epoch_perps) + 1), "Perplexity": epoch_perps})
                
                loss_chart_spot.line_chart(loss_df.set_index("Epoch"))
                perp_chart_spot.line_chart(perp_df.set_index("Epoch"))

            trainer.train(
                text_corpus=training_text,
                epochs=epochs_input,
                seq_len=80,
                batch_size=batch_size,
                save_dir=MODEL_DIR,
                progress_callback=ui_callback,
            )
            
            st.success("🎉 Model Training Complete! Checkpoint saved to `models/char_rnn.pt`.")
            st.cache_resource.clear()

    # TAB 3: DATASET & CORPUS EXPLORER
    with tab3:
        st.subheader("📊 Character Vocabulary & Corpus Analysis")
        
        corpus_text = load_corpus(DEFAULT_CORPUS_PATH)
        char_counts = pd.Series(list(corpus_text)).value_counts()
        
        ecol1, ecol2 = st.columns([1, 1.2])
        
        with ecol1:
            st.markdown(f"- **Total Corpus Characters:** `{len(corpus_text):,}`")
            st.markdown(f"- **Unique Characters (Vocab Size):** `{len(vocab)}`")
            st.markdown(f"- **Vocabulary Characters:** `{repr(''.join(vocab.idx2char))}`")
            
            st.markdown("##### Character Index Mapping")
            vocab_df = pd.DataFrame({"Index": range(len(vocab)), "Character": [repr(c) for c in vocab.idx2char]})
            st.dataframe(vocab_df, height=300, use_container_width=True)
            
        with ecol2:
            st.markdown("##### Character Frequency Distribution")
            st.bar_chart(char_counts.head(30))
            
        st.markdown("##### Corpus Raw Text Sample")
        st.text_area("Corpus Preview", corpus_text[:1000], height=200)

    # TAB 4: ARCHITECTURE DIAGNOSTICS
    with tab4:
        st.subheader("🔬 RNN Model Architecture Breakdown")
        
        dcol1, dcol2 = st.columns(2)
        with dcol1:
            st.markdown(
                f"""
                <div class="custom-card">
                    <h4>Network Specification</h4>
                    <ul>
                        <li><b>Architecture:</b> {model.rnn_type}</li>
                        <li><b>Vocabulary Size:</b> {model.vocab_size}</li>
                        <li><b>Embedding Dimension:</b> {model.embed_dim}</li>
                        <li><b>Hidden Dimension:</b> {model.hidden_dim}</li>
                        <li><b>Recurrent Layers:</b> {model.num_layers}</li>
                        <li><b>Total Parameters:</b> {model.get_num_params():,}</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
        with dcol2:
            st.markdown(
                """
                <div class="custom-card">
                    <h4>Sampling & Inference Features</h4>
                    <ul>
                        <li><b>Temperature Softmax:</b> Scaled output logit probabilities</li>
                        <li><b>Top-K Filtering:</b> Restricts sampling to top K logits</li>
                        <li><b>Top-P Nucleus:</b> Cumulative probability mass thresholding</li>
                        <li><b>Auto-regressive Warm-up:</b> Seed prompt hidden state tracking</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
