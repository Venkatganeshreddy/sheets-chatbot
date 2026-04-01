import streamlit as st
import os
import json
import time
import re
import requests
import gspread
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google.oauth2.service_account import Credentials
from dotenv import load_dotenv

load_dotenv()

# Support both local .env and Streamlit Cloud secrets
def get_secret(key, default=None):
    # Try Streamlit secrets first (for cloud deployment)
    try:
        return st.secrets[key]
    except Exception:
        pass
    # Fall back to env vars (for local dev)
    return os.getenv(key, default)

OPENROUTER_API_KEY = get_secret("OPENROUTER_API_KEY")
SHEET_URL = get_secret("SHEET_URL")

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]

MAX_CONTEXT_CHARS = 600_000


def get_sheet_id(url):
    return url.split("/d/")[1].split("/")[0]


def get_gc():
    # Check if running on Streamlit Cloud (secrets available)
    if "gcp_service_account" in st.secrets:
        gcp_raw = st.secrets["gcp_service_account"]
        # Convert AttrDict to a plain dict recursively
        gcp_info = json.loads(json.dumps(dict(gcp_raw)))
        creds = Credentials.from_service_account_info(gcp_info, scopes=SCOPES)
        return gspread.authorize(creds)
    # Local dev: use credentials file
    credentials_path = os.getenv("GOOGLE_CREDENTIALS_PATH", "credentials.json")
    creds = Credentials.from_service_account_file(credentials_path, scopes=SCOPES)
    return gspread.authorize(creds)


def fetch_spreadsheet_full(gc, sheet_id):
    resp = gc.http_client.request(
        "get",
        f"https://sheets.googleapis.com/v4/spreadsheets/{sheet_id}",
        params={
            "includeGridData": "true",
            "fields": (
                "properties.title,"
                "sheets.properties.title,"
                "sheets.data.rowData.values("
                "formattedValue,hyperlink,note,"
                "textFormatRuns,dataValidation,"
                "userEnteredValue,chipRuns"
                ")"
            ),
        },
    )
    return resp.json()


def parse_spreadsheet(content, is_linked=False):
    spreadsheet_title = content.get("properties", {}).get("title", "Unknown")
    all_data = {}
    chip_links = []

    for sheet in content.get("sheets", []):
        tab_name = sheet.get("properties", {}).get("title", "Unknown")
        grid_list = sheet.get("data", [])

        if not grid_list:
            all_data[tab_name] = {
                "headers": [], "rows": [], "hyperlinks": [],
                "notes": [], "dropdowns": [],
            }
            continue

        row_data = grid_list[0].get("rowData", [])
        if not row_data:
            all_data[tab_name] = {
                "headers": [], "rows": [], "hyperlinks": [],
                "notes": [], "dropdowns": [],
            }
            continue

        all_rows = []
        for row in row_data:
            cells = row.get("values", [])
            all_rows.append([c.get("formattedValue", "") for c in cells])

        headers = all_rows[0] if all_rows else []
        data_rows = all_rows[1:] if len(all_rows) > 1 else []

        col_name = lambda ci: headers[ci] if ci < len(headers) else f"Col{ci}"

        hyperlinks = []
        notes = []
        dropdowns = []

        for ri, row in enumerate(row_data):
            for ci, cell in enumerate(row.get("values", [])):
                text = cell.get("formattedValue", "")
                cn = col_name(ci)

                link = cell.get("hyperlink")
                if link:
                    hyperlinks.append({"row": ri, "col": cn, "text": text, "url": link})

                for run in cell.get("textFormatRuns", []):
                    run_link = run.get("format", {}).get("link", {}).get("uri")
                    if run_link and run_link != link:
                        hyperlinks.append({"row": ri, "col": cn, "text": text, "url": run_link, "type": "smart_chip"})

                for chip in cell.get("chipRuns", []):
                    uri = chip.get("chip", {}).get("richLinkProperties", {}).get("uri", "")
                    if uri:
                        hyperlinks.append({"row": ri, "col": cn, "text": text, "url": uri, "type": "file_chip"})
                        if not is_linked and "/spreadsheets/d/" in uri:
                            linked_id = uri.split("/d/")[1].split("/")[0]
                            chip_links.append({"sheet_id": linked_id, "label": text, "url": uri, "source_tab": tab_name})

                note = cell.get("note")
                if note:
                    notes.append({"row": ri, "col": cn, "text": text, "note": note})

                dv = cell.get("dataValidation")
                if dv and ri <= 1:
                    condition = dv.get("condition", {})
                    dv_type = condition.get("type", "")
                    if dv_type in ("ONE_OF_LIST", "ONE_OF_RANGE"):
                        vals = condition.get("values", [])
                        dropdown_source = ", ".join(v.get("userEnteredValue", "") for v in vals[:5])
                        dropdowns.append({"col": cn, "type": dv_type, "source": dropdown_source})

        all_data[tab_name] = {
            "headers": headers, "rows": data_rows, "hyperlinks": hyperlinks,
            "notes": notes, "dropdowns": dropdowns,
        }

    return spreadsheet_title, all_data, chip_links


def format_tab_text(tab_name, info, source_label=""):
    """Format a single tab into searchable text with header:value pairs per row."""
    headers = info["headers"]
    rows = info["rows"]
    hyperlinks = info["hyperlinks"]
    notes = info["notes"]

    if not headers and not rows:
        return ""

    prefix = f"[{source_label}] " if source_label else ""
    section = f"{prefix}Sheet: {tab_name}\n"

    if headers:
        section += f"Columns: {' | '.join(headers)}\n"

    # Format each row with column_name: value pairs for clarity
    for i, row in enumerate(rows[:500]):
        if headers:
            pairs = []
            for j, cell in enumerate(row):
                if cell.strip():
                    col = headers[j] if j < len(headers) else f"Col{j}"
                    pairs.append(f"{col}: {cell}")
            if pairs:
                section += f"Row {i+1}: {' | '.join(pairs)}\n"
        else:
            row_str = " | ".join(cell for cell in row)
            if row_str.strip(" |"):
                section += f"Row {i+1}: {row_str}\n"

    if hyperlinks:
        section += "Links:\n"
        for h in hyperlinks:
            lt = f" [{h.get('type', '')}]" if h.get("type") else ""
            section += f'  {h["col"]} "{h["text"][:60]}" -> {h["url"]}{lt}\n'

    if notes:
        section += "Notes:\n"
        for n in notes:
            section += f'  {n["col"]} "{n["text"]}": {n["note"]}\n'

    return section


@st.cache_data(ttl=300)
def load_everything():
    """Load main sheet + linked sheets. Returns list of (chunk_id, chunk_text) for retrieval."""
    gc = get_gc()
    sheet_id = get_sheet_id(SHEET_URL)

    # 1. Main spreadsheet
    main_content = fetch_spreadsheet_full(gc, sheet_id)
    main_title, main_data, chip_links = parse_spreadsheet(main_content, is_linked=False)

    # Build chunks: each tab = one chunk
    chunks = []
    for tab_name, info in main_data.items():
        text = format_tab_text(tab_name, info, source_label=f"MAIN: {main_title}")
        if text.strip():
            chunks.append({
                "id": f"main/{tab_name}",
                "label": f"{main_title} > {tab_name}",
                "text": text,
                "is_main": True,
            })

    # 2. Linked spreadsheets
    seen_ids = {sheet_id}
    unique_links = []
    for cl in chip_links:
        if cl["sheet_id"] not in seen_ids:
            seen_ids.add(cl["sheet_id"])
            unique_links.append(cl)

    linked_loaded = 0
    failed_links = []
    progress = st.progress(0, text="Loading linked sheets...")

    for i, cl in enumerate(unique_links):
        progress.progress(
            (i + 1) / max(len(unique_links), 1),
            text=f"Loading {i+1}/{len(unique_links)}: {cl['label'][:40]}..."
        )
        try:
            linked_content = fetch_spreadsheet_full(gc, cl["sheet_id"])
            linked_title, linked_tabs, _ = parse_spreadsheet(linked_content, is_linked=True)
            linked_loaded += 1

            for tab_name, info in linked_tabs.items():
                text = format_tab_text(tab_name, info, source_label=f"LINKED: {linked_title}")
                if text.strip():
                    chunks.append({
                        "id": f"linked/{cl['sheet_id']}/{tab_name}",
                        "label": f"{linked_title} > {tab_name}",
                        "text": text,
                        "is_main": False,
                        "url": cl["url"],
                    })
        except Exception as e:
            err = str(e)
            if "429" in err:
                time.sleep(12)
                try:
                    linked_content = fetch_spreadsheet_full(gc, cl["sheet_id"])
                    linked_title, linked_tabs, _ = parse_spreadsheet(linked_content, is_linked=True)
                    linked_loaded += 1
                    for tab_name, info in linked_tabs.items():
                        text = format_tab_text(tab_name, info, source_label=f"LINKED: {linked_title}")
                        if text.strip():
                            chunks.append({
                                "id": f"linked/{cl['sheet_id']}/{tab_name}",
                                "label": f"{linked_title} > {tab_name}",
                                "text": text,
                                "is_main": False,
                                "url": cl["url"],
                            })
                except Exception:
                    failed_links.append({"label": cl["label"], "error": "rate limited"})
            elif "403" in err or "404" in err:
                failed_links.append({"label": cl["label"], "error": "no access"})
            else:
                failed_links.append({"label": cl["label"], "error": err[:60]})

        if i < len(unique_links) - 1:
            time.sleep(1.5)

    progress.empty()

    stats = {
        "main_title": main_title,
        "main_tabs": len(main_data),
        "total_linked_found": len(unique_links),
        "linked_loaded": linked_loaded,
        "failed_links": failed_links,
        "total_chunks": len(chunks),
    }

    return chunks, stats


def build_retriever(chunks):
    """Build TF-IDF index over all chunks for fast retrieval."""
    texts = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=20000,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    return vectorizer, tfidf_matrix


def retrieve_relevant_chunks(query, chunks, vectorizer, tfidf_matrix, max_chars=MAX_CONTEXT_CHARS):
    """Find the most relevant chunks for a query using TF-IDF + keyword matching."""
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    query_lower = query.lower()
    query_words = set(re.findall(r'\w+', query_lower))
    # Filter out very short/common words for matching
    meaningful_words = {w for w in query_words if len(w) > 2}

    boosted_scores = []
    for i, chunk in enumerate(chunks):
        score = float(scores[i])
        chunk_lower = chunk["text"].lower()

        # Strong boost for exact phrase match
        if query_lower in chunk_lower:
            score += 0.8

        # Boost for keyword matches (proportional to how many match)
        keyword_hits = sum(1 for w in meaningful_words if w in chunk_lower)
        if meaningful_words:
            score += (keyword_hits / len(meaningful_words)) * 0.4

        boosted_scores.append((i, score))

    # Sort by score descending
    boosted_scores.sort(key=lambda x: x[1], reverse=True)

    # Step 1: Always include ALL main sheet tabs first (they're the core data)
    selected = []
    total_chars = 0
    main_budget = int(max_chars * 0.5)  # Reserve 50% for main sheets

    # Add main tabs sorted by relevance
    main_indices = [(idx, sc) for idx, sc in boosted_scores if chunks[idx]["is_main"]]
    for idx, score in main_indices:
        chunk = chunks[idx]
        chunk_len = len(chunk["text"])
        if total_chars + chunk_len > main_budget:
            continue
        selected.append({**chunk, "score": score})
        total_chars += chunk_len

    # Step 2: Fill remaining budget with top linked chunks
    linked_budget = max_chars - total_chars
    selected_ids = {s["id"] for s in selected}

    for idx, score in boosted_scores:
        chunk = chunks[idx]
        if chunk["id"] in selected_ids:
            continue
        chunk_len = len(chunk["text"])
        if total_chars + chunk_len > max_chars:
            if not chunk["is_main"] and total_chars > main_budget:
                continue
            # Truncate if it's the first linked chunk and it's huge
            if chunk_len > linked_budget:
                selected.append({**chunk, "text": chunk["text"][:linked_budget], "score": score})
                break
        selected.append({**chunk, "score": score})
        total_chars += chunk_len
        selected_ids.add(chunk["id"])

    return selected


def chat_with_openrouter(messages, relevant_context):
    system_msg = f"""You are a helpful assistant that answers questions based on live Google Sheets data.
You have access to data from a MAIN spreadsheet and its LINKED spreadsheets (connected via smart chips).
Below is the most relevant data for the user's question. Each row is formatted as column_name: value pairs.

CRITICAL RULES:
- Answer ONLY based on the data below. Do NOT guess or assume.
- Be precise about matching: if the user asks about "CDU 2025", match EXACTLY "CDU 2025" — do NOT return data for "CDU 2024" or other years.
- When multiple rows match partially, list ALL of them and highlight the differences.
- Always cite the exact sheet/tab name AND row number.
- Include full URLs when referencing links.
- Cross-reference between main and linked sheets when relevant.
- If the data below doesn't contain the answer, say so and suggest what sheet might have it.
- Pay attention to column names — each row has "ColumnName: value" format. Use the column name to identify what each value means.

RELEVANT SPREADSHEET DATA:
{relevant_context}"""

    api_messages = [{"role": "system", "content": system_msg}]
    api_messages.extend(messages)

    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "google/gemini-2.0-flash-001",
            "messages": api_messages,
            "max_tokens": 4096,
        },
        timeout=120,
    )

    if resp.status_code != 200:
        return f"API Error ({resp.status_code}): {resp.text}"

    data = resp.json()
    return data["choices"][0]["message"]["content"]


# --- UI ---

st.set_page_config(
    page_title="AOL Chatbot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimal CSS — let Streamlit's light theme do the heavy lifting
st.markdown("""
<style>
    /* Only hide footer and deploy button, keep header for sidebar toggle */
    footer {visibility: hidden;}
    .stDeployButton {display: none;}

    /* Header banner */
    .aol-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 1.5rem 2rem;
        border-radius: 14px;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        color: white;
    }
    .aol-title { font-size: 1.4rem; font-weight: 700; margin: 0; letter-spacing: 0.5px; }
    .aol-sub { color: #94a3b8; font-size: 0.82rem; margin: 4px 0 0 0; }
    .aol-live {
        display: inline-flex; align-items: center; gap: 6px;
        background: rgba(34,197,94,0.15); color: #4ade80;
        padding: 6px 14px; border-radius: 20px;
        font-size: 0.75rem; font-weight: 600;
    }
    .aol-dot {
        width: 8px; height: 8px; background: #4ade80;
        border-radius: 50%; display: inline-block;
        animation: blink 2s infinite;
    }
    @keyframes blink { 0%,100%{opacity:1;} 50%{opacity:0.3;} }

    /* Stat cards row */
    .stats { display: flex; gap: 12px; margin-bottom: 1.2rem; }
    .scard {
        flex: 1; text-align: center;
        background: white; border: 1px solid #e2e8f0;
        border-radius: 12px; padding: 16px 8px;
    }
    .scard b { display: block; font-size: 1.5rem; color: #0f172a; }
    .scard span { font-size: 0.68rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.8px; }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<div class="aol-header">
    <div>
        <div class="aol-title">AOL - CHATBOT</div>
        <p class="aol-sub">NIAT 2025 &mdash; Program Design, Package &amp; Implementation Intelligence</p>
    </div>
    <div class="aol-live"><span class="aol-dot"></span> LIVE</div>
</div>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("#### 🎓 AOL Chatbot")
    st.caption("NIAT 2025 Data Intelligence")
    st.divider()
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        if "retriever" in st.session_state:
            del st.session_state["retriever"]
        st.rerun()

# --- Load Data ---
with st.status("Loading live data...", expanded=True) as status:
    st.write("Connecting to Google Sheets...")
    try:
        chunks, stats = load_everything()
    except Exception as e:
        st.error(f"Failed to load: {e}")
        st.info("Share the sheet (and linked sheets) with the service account email.")
        st.stop()

    st.write("Building search index...")
    if "retriever" not in st.session_state or st.session_state.get("chunk_count") != len(chunks):
        vectorizer, tfidf_matrix = build_retriever(chunks)
        st.session_state["retriever"] = (vectorizer, tfidf_matrix)
        st.session_state["chunk_count"] = len(chunks)
    else:
        vectorizer, tfidf_matrix = st.session_state["retriever"]

    status.update(
        label=f"Ready — {stats['total_chunks']} sections from {stats['main_tabs']} tabs + {stats['linked_loaded']} linked sheets",
        state="complete",
        expanded=False,
    )

# --- Stat Cards ---
main_chunks = [c for c in chunks if c["is_main"]]
linked_chunks = [c for c in chunks if not c["is_main"]]
total_chars = sum(len(c["text"]) for c in chunks)

st.markdown(f"""
<div class="stats">
    <div class="scard"><b>{stats['main_tabs']}</b><span>Main Tabs</span></div>
    <div class="scard"><b>{stats['linked_loaded']}</b><span>Linked Sheets</span></div>
    <div class="scard"><b>{stats['total_chunks']}</b><span>Indexed Sections</span></div>
    <div class="scard"><b>{total_chars // 1000}K</b><span>Characters</span></div>
</div>
""", unsafe_allow_html=True)

# --- Sidebar Details ---
with st.sidebar:
    st.divider()

    # Status
    st.markdown(f"**Status:** ✅ Online — {stats['total_chunks']} sections")

    if stats["failed_links"]:
        st.markdown(f"**⚠️ {len(stats['failed_links'])} sheets inaccessible**")
        with st.expander("Show details"):
            for fl in stats["failed_links"]:
                st.caption(f"• {fl['label'][:40]} — {fl['error']}")

    st.divider()

    # Main sheets
    st.markdown("**📋 Main Sheets**")
    for c in main_chunks:
        name = c["label"].split(">")[-1].strip() if ">" in c["label"] else c["label"]
        st.caption(f"• {name}")

    st.divider()

    # Linked sheets
    st.markdown(f"**🔗 Linked Sheets ({len(linked_chunks)})**")
    with st.expander("View all"):
        for c in linked_chunks:
            name = c["label"].split(">")[-1].strip() if ">" in c["label"] else c["label"]
            st.caption(f"• {name[:50]}")

    st.divider()
    st.caption("Data cached 5 min. Click Refresh for latest.")
    st.caption("Powered by NxtWave AOL Intelligence")

# --- Chat ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    with st.chat_message("assistant", avatar="🎓"):
        st.markdown(
            "**Welcome to AOL Chatbot!** I have access to all your NIAT 2025 data — "
            "BOS Trackers, Implementation details, Curriculum sheets, Course data, and more.\n\n"
            "**Try asking:**\n"
            '- *"What is the BOS status for MRV University?"*\n'
            '- *"Which universities have Full Delivery mode?"*\n'
            '- *"Show me the curriculum link for SGU"*\n'
            '- *"Who is the AOA for CDU 2025?"*'
        )

for msg in st.session_state.messages:
    avatar = "🎓" if msg["role"] == "assistant" else None
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about universities, BOS status, curricula, implementation..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🎓"):
        with st.spinner("Searching across all sheets..."):
            relevant = retrieve_relevant_chunks(prompt, chunks, vectorizer, tfidf_matrix)
            context_parts = [r["text"] for r in relevant]
            relevant_context = "\n\n---\n\n".join(context_parts)

            api_messages = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ]
            response = chat_with_openrouter(api_messages, relevant_context)

        st.markdown(response)

        with st.expander(f"📚 Sources ({len(relevant)} sections)"):
            for r in relevant:
                score = r.get("score", 0)
                st.caption(f"• {r['label']} — relevance: {score:.2f}")

    st.session_state.messages.append({"role": "assistant", "content": response})
