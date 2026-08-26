"""
Visual-Doc Assistant — Phase 3, Step 9C: Performance Upgrade
====================================================================
Step 9B made multi-PDF upload work correctly, but indexing was slow. The
root cause, in order of impact:

  1. THE BIG ONE — embed_page_image() ran a SEPARATE model forward pass
     per page. Phase 2, Step 5 even printed an estimate of "~3-5 seconds
     PER PAGE on a T4" for exactly this reason: one image in, one model
     call, one image out, repeated in a Python loop. A 26-page PDF was
     always going to take 1.5-2+ minutes; a 100-page PDF, 5-8 minutes.
  2. Every page did its own `collection.get(ids=[doc_id])` dedup check —
     one ChromaDB round trip per page instead of one for the whole file.
  3. PDF -> image rendering (poppler) ran single-threaded.
  4. Pages were rasterized at 300 DPI, but ColPali's vision encoder
     (SigLIP-So400m/14 inside PaliGemma) resizes every image to a FIXED
     internal resolution regardless of input size — so 300 DPI costs
     rasterization + preprocessing time without buying any retrieval
     quality. It only matters for how sharp the page looks to Gemini.
  5. A full re-upload of an already-fully-indexed PDF still rasterized
     and re-checked every page before discovering there was nothing new
     to add.

This cell overwrites the SAME app.py path Step 10 launches, fixing all
five without changing any retrieval/generation behavior:

  1. TRUE batched embedding — process_images() now receives a LIST of
     pages per call, so one model forward pass embeds an entire batch
     (default 4, same batch size Phase 2 already used, just now actually
     batched at the tensor level) instead of one page at a time.
  2. One collection.get(ids=[...all pages...]) call per file instead of
     one per page.
  3. convert_from_bytes(..., thread_count=N) renders multiple pages of a
     PDF in parallel via poppler.
  4. Default rendering DPI lowered 300 -> 200 (adjustable in the sidebar,
     with the reasoning above shown right next to the slider).
  5. A cheap pdfinfo_from_bytes() page-count check up front skips
     rasterizing/embedding entirely when a file has already been fully
     indexed (e.g. an accidental duplicate upload).

Everything else — embed_query, retrieve_top_k_pages, build_multimodal_input,
generate_answer, SYSTEM_INSTRUCTION, the Document Library sidebar, chat
history — is unchanged from Step 9B.
"""
import os
import shutil
import base64
import hashlib
import time

import streamlit as st
import torch
from PIL import Image
import chromadb
from pdf2image import convert_from_bytes, pdfinfo_from_bytes
from colpali_engine.models import ColPali, ColPaliProcessor
from google import genai

# ============================================================================
# Config -- same Drive paths as Phases 1-3, so this shares the existing index
# ============================================================================
MODEL_NAME       = "vidore/colpali-v1.2"
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_FOLDER     = "/content/drive/MyDrive/Visual-Doc/processed_images"
DB_PATH          = "/content/drive/MyDrive/Visual-Doc/chroma_db_storage"

GEMINI_MODEL     = "gemini-3.6-flash"
DEFAULT_DPI      = 200   # Was 300. ColPali resizes internally anyway (see above) —
                          # this is purely a rasterization/Gemini-legibility knob now.
DEFAULT_BATCH    = 4     # Same batch size Phase 2 documented; now truly batched.
RENDER_THREADS   = max(1, os.cpu_count() or 4)

os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(DB_PATH, exist_ok=True)

SYSTEM_INSTRUCTION = (
    "You are the Visual-Doc Assistant, a technical documentation Q&A system. "
    "Answer the user's question using ONLY the page image(s) provided below — "
    "do not use outside knowledge. Reference the source document and page "
    "number when relevant. If the provided pages don't contain the answer, "
    "say so directly instead of guessing."
)

st.set_page_config(page_title="Visual-Doc Assistant", page_icon="📄", layout="wide")


# ============================================================================
# Cached resources -- load once per session, not once per query
# ============================================================================
@st.cache_resource(show_spinner="Loading ColPali model (first load only)...")
def load_colpali():
    model = ColPali.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda:0"
    ).eval()
    processor = ColPaliProcessor.from_pretrained(MODEL_NAME)
    return model, processor


@st.cache_resource(show_spinner="Connecting to ChromaDB...")
def load_collection():
    db_client = chromadb.PersistentClient(path=DB_PATH)
    return db_client.get_or_create_collection(
        name="visual_doc_collection", metadata={"hnsw:space": "cosine"}
    )


model, processor = load_colpali()
collection = load_collection()


@st.cache_resource(show_spinner="Checking index for legacy records...")
def migrate_legacy_metadata():
    """One-time backfill for Phase 2's pre-doc_id_prefix records. Unchanged
    from Step 9B — see that cell's docstring for details."""
    if collection.count() == 0:
        return 0
    all_data = collection.get(include=["metadatas"])
    updates_ids, updates_meta = [], []
    for doc_id, meta in zip(all_data["ids"], all_data["metadatas"]):
        if "doc_id_prefix" not in meta:
            legacy_prefix = f"{meta.get('doc_name', 'legacy')}__legacy"
            new_meta = dict(meta)
            new_meta["doc_id_prefix"] = legacy_prefix
            updates_ids.append(doc_id)
            updates_meta.append(new_meta)
    if updates_ids:
        collection.update(ids=updates_ids, metadatas=updates_meta)
    return len(updates_ids)


migrated_count = migrate_legacy_metadata()


# ============================================================================
# Embedding functions
# ============================================================================
@torch.no_grad()
def embed_page_images_batch(images: list) -> list:
    """
    Embed MULTIPLE page images in a single model forward pass. This is the
    main speed fix: Step 9B (and Phase 2 before it) called this once per
    page in a Python loop; every page paid the full model-call overhead on
    its own. Batching amortizes that overhead across `len(images)` pages.
    """
    inputs = processor.process_images(images).to(DEVICE)
    batch_embeddings = model(**inputs)                    # (B, N_patches, 128)
    pooled = batch_embeddings.mean(dim=1)                  # (B, 128)
    return [p.cpu().float().numpy() for p in pooled]


@torch.no_grad()
def embed_query(query_text: str):
    inputs = processor.process_queries([query_text]).to(DEVICE)
    token_embeddings = model(**inputs)                     # (1, N_tokens, 128)
    pooled = token_embeddings.mean(dim=1).squeeze(0)        # (128,)
    return pooled.cpu().float().numpy()


# ============================================================================
# PDF ingestion pipeline -- optimized
# ============================================================================
def _file_hash(file_bytes: bytes) -> str:
    return hashlib.md5(file_bytes).hexdigest()[:10]


def index_pdf(uploaded_file, dpi: int, batch_size: int, progress_bar=None) -> dict:
    """
    Convert an uploaded PDF into page images and embed + store every page
    in ChromaDB, with all five optimizations described in this cell's
    docstring applied.
    """
    file_bytes = uploaded_file.getvalue()
    doc_hash = _file_hash(file_bytes)
    doc_name = os.path.splitext(uploaded_file.name)[0]
    doc_id_prefix = f"{doc_name}__{doc_hash}"
    doc_image_folder = os.path.join(IMAGE_FOLDER, doc_id_prefix)

    t_start = time.time()

    # ---- Optimization 5: skip everything if this exact file is already
    # fully indexed (cheap metadata-only page count, no rasterization) ----
    try:
        total_pages = pdfinfo_from_bytes(file_bytes)["Pages"]
    except Exception:
        total_pages = None  # fall back to full conversion below if pdfinfo fails

    if total_pages is not None:
        already = collection.get(where={"doc_id_prefix": doc_id_prefix}, include=[])
        if len(already["ids"]) >= total_pages:
            if progress_bar is not None:
                progress_bar.progress(1.0, text=f"'{uploaded_file.name}' already fully indexed — skipped")
            return {
                "doc_name": doc_name, "doc_id_prefix": doc_id_prefix,
                "total_pages": total_pages, "added": 0, "skipped": total_pages,
                "elapsed": time.time() - t_start,
            }

    os.makedirs(doc_image_folder, exist_ok=True)

    # ---- Optimization 3: parallel PDF -> image rendering ----
    pages = convert_from_bytes(file_bytes, dpi=dpi, thread_count=RENDER_THREADS)
    total_pages = len(pages)

    # ---- Optimization 2: one dedup check for the whole file, not one per page ----
    candidate_ids = [f"{doc_id_prefix}__page_{i + 1}" for i in range(total_pages)]
    existing_ids = set(collection.get(ids=candidate_ids, include=[])["ids"])

    to_process = [
        (i + 1, page) for i, page in enumerate(pages)
        if f"{doc_id_prefix}__page_{i + 1}" not in existing_ids
    ]
    skipped = total_pages - len(to_process)
    added = 0

    # ---- Optimization 1: true batched embedding ----
    for batch_start in range(0, len(to_process), batch_size):
        batch = to_process[batch_start: batch_start + batch_size]
        batch_page_nums = [pn for pn, _ in batch]
        batch_images_rgb = [pg.convert("RGB") for _, pg in batch]

        # Save PNGs for this batch (needed for Gemini's input later)
        batch_paths = []
        for page_num, rgb_img in zip(batch_page_nums, batch_images_rgb):
            image_path = os.path.join(doc_image_folder, f"page_{page_num}.png")
            if not os.path.exists(image_path):
                rgb_img.save(image_path, "PNG")
            batch_paths.append(image_path)

        batch_embeddings = embed_page_images_batch(batch_images_rgb)

        collection.add(
            ids=[f"{doc_id_prefix}__page_{pn}" for pn in batch_page_nums],
            embeddings=[e.tolist() for e in batch_embeddings],
            metadatas=[{
                "page_number": pn,
                "image_path": path,
                "filename": uploaded_file.name,
                "doc_name": doc_name,
                "doc_id_prefix": doc_id_prefix,
            } for pn, path in zip(batch_page_nums, batch_paths)],
            documents=[f"{doc_name} — page {pn}" for pn in batch_page_nums],
        )
        added += len(batch)

        if progress_bar is not None:
            done = batch_start + len(batch)
            progress_bar.progress(
                min(done / max(len(to_process), 1), 1.0),
                text=f"Embedding '{uploaded_file.name}' — {done}/{len(to_process)} new pages "
                     f"({skipped} already indexed)",
            )

    if progress_bar is not None:
        progress_bar.progress(1.0, text=f"'{uploaded_file.name}' done — {added} added, {skipped} skipped")

    return {
        "doc_name": doc_name, "doc_id_prefix": doc_id_prefix,
        "total_pages": total_pages, "added": added, "skipped": skipped,
        "elapsed": time.time() - t_start,
    }


def get_indexed_documents() -> dict:
    if collection.count() == 0:
        return {}
    all_meta = collection.get(include=["metadatas"])["metadatas"]
    docs = {}
    for m in all_meta:
        key = (m.get("doc_name", "unknown"), m.get("doc_id_prefix", "unknown"))
        docs[key] = docs.get(key, 0) + 1
    return dict(sorted(docs.items(), key=lambda kv: kv[0][0].lower()))


def delete_document(doc_id_prefix: str) -> int:
    matches = collection.get(where={"doc_id_prefix": doc_id_prefix})
    ids_to_delete = matches["ids"]
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
    folder = os.path.join(IMAGE_FOLDER, doc_id_prefix)
    if os.path.isdir(folder):
        shutil.rmtree(folder, ignore_errors=True)
    return len(ids_to_delete)


def reset_index():
    ids = collection.get()["ids"]
    if ids:
        collection.delete(ids=ids)
    if os.path.isdir(IMAGE_FOLDER):
        shutil.rmtree(IMAGE_FOLDER, ignore_errors=True)
        os.makedirs(IMAGE_FOLDER, exist_ok=True)


# ============================================================================
# Retrieval + generation -- unchanged from Step 9B
# ============================================================================
def retrieve_top_k_pages(query_text: str, k: int = 3, doc_filter=None):
    query_embedding = embed_query(query_text)
    where = {"doc_id_prefix": {"$in": doc_filter}} if doc_filter else None

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=k,
        where=where,
        include=["metadatas", "distances"],
    )

    pages = []
    if results["ids"] and results["ids"][0]:
        for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
            pages.append({
                "page_number": meta["page_number"],
                "image_path": meta["image_path"],
                "doc_name": meta["doc_name"],
                "similarity": 1 - dist,
            })
    return pages


def build_multimodal_input(query_text: str, pages: list):
    content = [{"type": "text", "text": f"Question: {query_text}"}]
    for p in pages:
        with open(p["image_path"], "rb") as f:
            image_bytes = f.read()
        content.append({
            "type": "image",
            "data": base64.b64encode(image_bytes).decode("utf-8"),
            "mime_type": "image/png",
        })
    return content


def generate_answer(gemini_client, query_text: str, pages: list) -> str:
    try:
        interaction = gemini_client.interactions.create(
            model=GEMINI_MODEL,
            input=build_multimodal_input(query_text, pages),
            system_instruction=SYSTEM_INSTRUCTION,
        )
        return interaction.output_text
    except Exception as e:
        return f"Gemini request failed: {e}"


# ============================================================================
# Session state
# ============================================================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_uploads" not in st.session_state:
    st.session_state.processed_uploads = set()


# ============================================================================
# Sidebar — Document Library
# ============================================================================
st.sidebar.title("📁 Document Library")

if migrated_count:
    st.sidebar.caption(f"Migrated {migrated_count} legacy record(s) from Phase 2.")

if shutil.which("pdftoppm") is None:
    st.sidebar.error(
        "poppler-utils not found on this system.\n\n"
        "Install it before uploading PDFs:\n"
        "`apt-get install -y poppler-utils` (Linux/Colab)\n"
        "`brew install poppler` (macOS)"
    )

with st.sidebar.expander("⚡ Indexing speed settings", expanded=False):
    st.caption(
        "ColPali's vision encoder resizes every page to a fixed internal "
        "resolution, so a lower DPI here speeds up rasterization and disk "
        "I/O without changing retrieval quality. It only affects how sharp "
        "the page looks when Gemini reads it — 150–200 is normally plenty "
        "for text-heavy pages."
    )
    render_dpi = st.slider("Rendering DPI", min_value=100, max_value=300, value=DEFAULT_DPI, step=25)
    embed_batch_size = st.slider(
        "Embedding batch size", min_value=1, max_value=8, value=DEFAULT_BATCH,
        help="Pages embedded per model forward pass. Higher = faster on a strong GPU, "
             "but uses more VRAM. Lower this if you hit a CUDA out-of-memory error.",
    )

uploaded_files = st.sidebar.file_uploader(
    "Add PDFs to the knowledge base",
    type=["pdf"],
    accept_multiple_files=True,
    help="Any number of files. Each page is rendered as an image and embedded directly with ColPali — no OCR.",
)

if uploaded_files:
    for uf in uploaded_files:
        upload_key = f"{uf.name}_{uf.size}"
        if upload_key in st.session_state.processed_uploads:
            continue

        progress_bar = st.sidebar.progress(0.0, text=f"Preparing '{uf.name}'...")
        try:
            result = index_pdf(uf, dpi=render_dpi, batch_size=embed_batch_size, progress_bar=progress_bar)
            progress_bar.progress(
                1.0,
                text=(f"✅ {uf.name}: {result['added']} new pages, {result['skipped']} already indexed "
                      f"({result['elapsed']:.1f}s)"),
            )
        except Exception as e:
            progress_bar.empty()
            st.sidebar.error(f"Failed to index '{uf.name}': {e}")
        finally:
            st.session_state.processed_uploads.add(upload_key)

st.sidebar.divider()
st.sidebar.subheader("Indexed Documents")

docs = get_indexed_documents()
if not docs:
    st.sidebar.caption("No documents indexed yet — upload one or more PDFs above.")
else:
    for (doc_name, doc_id_prefix), page_count in docs.items():
        row = st.sidebar.columns([4, 1])
        row[0].markdown(f"**{doc_name}**  \n{page_count} pages indexed")
        if row[1].button("🗑️", key=f"del_{doc_id_prefix}", help=f"Remove '{doc_name}'"):
            removed = delete_document(doc_id_prefix)
            st.sidebar.success(f"Removed {removed} pages from '{doc_name}'.")
            st.rerun()

    if st.sidebar.button("Clear ALL documents", type="secondary"):
        reset_index()
        st.session_state.chat_history = []
        st.rerun()

st.sidebar.divider()
st.sidebar.subheader("⚙️ Settings")

api_key = st.sidebar.text_input(
    "Gemini API key", type="password",
    value=os.environ.get("GEMINI_API_KEY", ""),
    help="Get one at aistudio.google.com/apikey",
)
top_k = st.sidebar.slider("Pages to retrieve (k)", min_value=1, max_value=8, value=3)

doc_names = [d[0] for d in docs.keys()]
selected_docs = st.sidebar.multiselect(
    "Search only in (optional)",
    options=doc_names,
    default=[],
    help="Leave empty to search across every indexed document.",
)
doc_filter = (
    [dip for (dn, dip) in docs.keys() if dn in selected_docs]
    if selected_docs else None
)

if st.sidebar.button("Clear chat history"):
    st.session_state.chat_history = []
    st.rerun()

st.sidebar.divider()
st.sidebar.caption(f"Total pages indexed: {collection.count()} · Device: {DEVICE}")


# ============================================================================
# Main area
# ============================================================================
st.title("📄 Visual-Doc Assistant")
st.caption("Multimodal RAG over your own documents — ColPali page-image retrieval + Gemini generation. No OCR.")

if not docs:
    st.info("👈 Upload one or more PDFs from the sidebar to get started. Any number of documents can be added, at any time.")

for turn in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(turn["query"])
    with st.chat_message("assistant"):
        st.markdown(turn["answer"])
        with st.expander(f"📑 Source pages ({len(turn['pages'])})", expanded=False):
            cols = st.columns(min(len(turn["pages"]), 4) or 1)
            for i, p in enumerate(turn["pages"]):
                with cols[i % len(cols)]:
                    img = Image.open(p["image_path"]).convert("RGB")
                    st.image(
                        img,
                        caption=f"{p['doc_name']} · p.{p['page_number']} · sim {p['similarity']:.3f}",
                        use_container_width=True,
                    )

query = st.chat_input("Ask a question about your documents...")

if query:
    if not api_key:
        st.error("Please enter your Gemini API key in the sidebar.")
    elif not docs:
        st.warning("Please upload at least one PDF first.")
    else:
        with st.chat_message("user"):
            st.markdown(query)

        gemini_client = genai.Client(api_key=api_key)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant pages..."):
                pages = retrieve_top_k_pages(query, k=top_k, doc_filter=doc_filter)

            if not pages:
                st.warning("No relevant pages found in the selected document(s).")
            else:
                with st.spinner("Generating answer with Gemini..."):
                    answer = generate_answer(gemini_client, query, pages)

                st.markdown(answer)
                with st.expander(f"📑 Source pages ({len(pages)})", expanded=True):
                    cols = st.columns(min(len(pages), 4) or 1)
                    for i, p in enumerate(pages):
                        with cols[i % len(cols)]:
                            img = Image.open(p["image_path"]).convert("RGB")
                            st.image(
                                img,
                                caption=f"{p['doc_name']} · p.{p['page_number']} · sim {p['similarity']:.3f}",
                                use_container_width=True,
                            )

                st.session_state.chat_history.append({
                    "query": query, "answer": answer, "pages": pages,
                })
