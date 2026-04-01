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
    page_title="AOL Chatbot | NxtWave",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    footer {visibility: hidden;}
    .stDeployButton {display: none;}

    html, body, [class*="css"] {
        font-family: 'Space Grotesk', -apple-system, sans-serif !important;
    }

    .stApp { background: #0a0a0f; }

    /* ── Animated mesh background ── */
    .mesh-bg {
        position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        pointer-events: none; z-index: 0;
        background:
            radial-gradient(ellipse 600px 400px at 20% 20%, rgba(99,102,241,0.08) 0%, transparent 70%),
            radial-gradient(ellipse 500px 500px at 80% 80%, rgba(236,72,153,0.06) 0%, transparent 70%),
            radial-gradient(ellipse 400px 300px at 60% 30%, rgba(34,197,94,0.05) 0%, transparent 70%);
    }

    /* ── Hero ── */
    .hero2 {
        position: relative;
        background: linear-gradient(135deg, rgba(15,23,42,0.9), rgba(30,41,59,0.9));
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.06);
        padding: 2.2rem 2.8rem;
        border-radius: 24px;
        margin-bottom: 1.5rem;
        display: flex; align-items: center; justify-content: space-between;
        overflow: hidden;
    }
    .hero2::before {
        content: ''; position: absolute; inset: 0;
        background: linear-gradient(135deg, rgba(99,102,241,0.1) 0%, transparent 50%, rgba(236,72,153,0.08) 100%);
        border-radius: 24px;
    }
    .hero2-left { position: relative; z-index: 1; }
    .hero2-tag {
        display: inline-block;
        background: linear-gradient(135deg, #6366f1, #ec4899);
        padding: 4px 12px; border-radius: 6px;
        font-size: 0.6rem; font-weight: 600; letter-spacing: 2.5px;
        color: white; text-transform: uppercase; margin-bottom: 10px;
    }
    .hero2 h1 {
        font-family: 'Space Grotesk', sans-serif;
        color: #ffffff; font-size: 2.2rem; font-weight: 700;
        margin: 0; line-height: 1.1; letter-spacing: -1px;
    }
    .hero2 h1 span {
        background: linear-gradient(135deg, #818cf8, #c084fc, #f472b6);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .hero2-sub {
        color: #64748b; font-size: 0.88rem; margin: 10px 0 0 0;
        font-weight: 400;
    }
    .hero2-sub strong { color: #94a3b8; }
    .hero2-right { position: relative; z-index: 1; text-align: right; }
    .live2 {
        display: inline-flex; align-items: center; gap: 8px;
        background: rgba(34,197,94,0.1);
        border: 1px solid rgba(34,197,94,0.25);
        padding: 8px 18px; border-radius: 50px;
        font-size: 0.72rem; font-weight: 600; color: #4ade80;
        letter-spacing: 1.5px;
    }
    .live2-dot {
        width: 8px; height: 8px; background: #4ade80; border-radius: 50%;
        box-shadow: 0 0 12px #4ade80; animation: pulse2 2s infinite;
    }
    @keyframes pulse2 { 0%,100%{box-shadow:0 0 12px #4ade80;} 50%{box-shadow:0 0 2px #4ade80;} }
    .hero2-meta {
        margin-top: 8px; font-size: 0.65rem; color: #475569;
        font-family: 'JetBrains Mono', monospace;
    }

    /* ── Glowing stat cards ── */
    .sg { display: grid; grid-template-columns: repeat(4,1fr); gap: 14px; margin-bottom: 1.5rem; }
    .sc {
        background: rgba(15,23,42,0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 18px; padding: 22px 16px;
        text-align: center; position: relative;
        transition: all 0.4s cubic-bezier(0.4,0,0.2,1);
        overflow: hidden;
    }
    .sc::before {
        content: ''; position: absolute; inset: 0;
        border-radius: 18px; opacity: 0;
        transition: opacity 0.4s;
    }
    .sc:nth-child(1)::before { background: linear-gradient(135deg, rgba(99,102,241,0.15), transparent); }
    .sc:nth-child(2)::before { background: linear-gradient(135deg, rgba(236,72,153,0.15), transparent); }
    .sc:nth-child(3)::before { background: linear-gradient(135deg, rgba(34,197,94,0.15), transparent); }
    .sc:nth-child(4)::before { background: linear-gradient(135deg, rgba(251,191,36,0.15), transparent); }
    .sc:hover::before { opacity: 1; }
    .sc:hover { transform: translateY(-4px); border-color: rgba(255,255,255,0.12); }
    .sc-icon { font-size: 1.8rem; margin-bottom: 8px; position: relative; z-index: 1; }
    .sc-num {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2rem; font-weight: 700; color: #f1f5f9;
        line-height: 1; margin-bottom: 4px; position: relative; z-index: 1;
    }
    .sc-label {
        font-size: 0.65rem; color: #64748b; text-transform: uppercase;
        letter-spacing: 1.2px; font-weight: 500; position: relative; z-index: 1;
    }

    /* ── Welcome ── */
    .wc {
        background: rgba(15,23,42,0.5);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 20px; padding: 2rem 2.5rem;
        margin-bottom: 1rem;
    }
    .wc h3 { color: #f1f5f9; font-size: 1.3rem; font-weight: 700; margin: 0 0 6px 0; }
    .wc p { color: #94a3b8; font-size: 0.88rem; line-height: 1.7; margin: 0 0 16px 0; }
    .wc-grid { display: flex; flex-wrap: wrap; gap: 10px; }
    .wc-chip {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        color: #cbd5e1; padding: 10px 18px; border-radius: 12px;
        font-size: 0.82rem; font-weight: 500;
        transition: all 0.25s; cursor: default;
    }
    .wc-chip:hover {
        background: rgba(99,102,241,0.12);
        border-color: rgba(99,102,241,0.3);
        color: #e0e7ff;
        transform: translateY(-1px);
    }

    /* ── Chat ── */
    .stChatMessage {
        border-radius: 16px !important;
        background: rgba(15,23,42,0.4) !important;
        border: 1px solid rgba(255,255,255,0.04) !important;
    }
    .stChatMessage p, .stChatMessage li, .stChatMessage span {
        color: #e2e8f0 !important;
    }
    .stChatMessage strong { color: #f8fafc !important; }
    .stChatMessage a { color: #818cf8 !important; }
    [data-testid="stChatInput"] > div {
        background: rgba(15,23,42,0.6) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 14px !important;
        color: #f1f5f9 !important;
    }
    [data-testid="stChatInput"] > div:focus-within {
        border-color: rgba(99,102,241,0.5) !important;
        box-shadow: 0 0 20px rgba(99,102,241,0.15) !important;
    }
    [data-testid="stChatInput"] textarea {
        color: #f1f5f9 !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #475569 !important;
    }

    /* ── Expander in chat ── */
    .streamlit-expanderHeader { color: #94a3b8 !important; font-size: 0.82rem !important; }
    .streamlit-expanderContent p, .streamlit-expanderContent span { color: #94a3b8 !important; }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a, #1e293b) !important;
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    section[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
    section[data-testid="stSidebar"] .stMarkdown p { color: #94a3b8 !important; }
    section[data-testid="stSidebar"] .stCaption p { color: #64748b !important; }

    .sb2-brand {
        display: flex; align-items: center; gap: 12px;
        padding-bottom: 20px; margin-bottom: 16px;
        border-bottom: 1px solid rgba(255,255,255,0.06);
    }
    .sb2-icon {
        width: 42px; height: 42px; border-radius: 14px;
        background: linear-gradient(135deg, #6366f1, #ec4899);
        display: flex; align-items: center; justify-content: center;
        font-size: 1.2rem; flex-shrink: 0;
    }
    .sb2-name { font-size: 1rem; font-weight: 700; color: #f1f5f9 !important; line-height: 1.2; }
    .sb2-sub { font-size: 0.7rem; color: #64748b !important; margin-top: 2px; }

    .sb2-status {
        background: rgba(34,197,94,0.1);
        border: 1px solid rgba(34,197,94,0.2);
        border-radius: 12px; padding: 12px 16px;
        font-size: 0.8rem; color: #4ade80 !important;
        font-weight: 500; margin: 12px 0;
    }
    .sb2-section {
        font-size: 0.6rem; font-weight: 700; color: #475569 !important;
        text-transform: uppercase; letter-spacing: 2px;
        margin: 20px 0 10px 0;
    }
    .sb2-item {
        font-size: 0.78rem; color: #94a3b8 !important;
        padding: 6px 12px; border-radius: 8px; margin: 2px 0;
        transition: all 0.2s;
    }
    .sb2-item:hover {
        background: rgba(255,255,255,0.04);
        color: #e2e8f0 !important;
    }

    /* ── Status widget ── */
    [data-testid="stStatusWidget"] { background: rgba(15,23,42,0.6) !important; border-radius: 14px !important; }
    [data-testid="stStatusWidget"] p, [data-testid="stStatusWidget"] span { color: #94a3b8 !important; }

    /* ── Spinner ── */
    .stSpinner > div { color: #94a3b8 !important; }

    /* ── Buttons in sidebar ── */
    section[data-testid="stSidebar"] .stButton button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: white !important; border: none !important;
        border-radius: 12px !important; font-weight: 600 !important;
        padding: 10px 0 !important;
        transition: all 0.3s !important;
    }
    section[data-testid="stSidebar"] .stButton button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 20px rgba(99,102,241,0.4) !important;
    }
</style>
<div class="mesh-bg"></div>
""", unsafe_allow_html=True)

# --- Hero ---
st.markdown("""
<div class="hero2">
    <div class="hero2-left">
        <div class="hero2-tag">NxtWave AOL</div>
        <h1>Ask your <span>NIAT Data</span></h1>
        <p class="hero2-sub">Instant answers from <strong>all sheets, subsheets &amp; linked documents</strong></p>
    </div>
    <div class="hero2-right">
        <div class="live2"><span class="live2-dot"></span> LIVE</div>
        <div class="hero2-meta">v2.0 // RAG-powered</div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("""
    <div class="sb2-brand">
        <div class="sb2-icon">🎓</div>
        <div>
            <div class="sb2-name">AOL Chatbot</div>
            <div class="sb2-sub">NIAT Data Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Refresh Data", use_container_width=True):
        st.cache_data.clear()
        if "retriever" in st.session_state:
            del st.session_state["retriever"]
        st.rerun()

# --- Load ---
with st.status("Connecting to live data...", expanded=True) as status:
    st.write("Fetching from Google Sheets...")
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

    status.update(label=f"Ready — {stats['total_chunks']} sections indexed", state="complete", expanded=False)

# --- Stats ---
main_chunks = [c for c in chunks if c["is_main"]]
linked_chunks = [c for c in chunks if not c["is_main"]]
total_chars = sum(len(c["text"]) for c in chunks)

st.markdown(f"""
<div class="sg">
    <div class="sc"><div class="sc-icon">📋</div><div class="sc-num">{stats['main_tabs']}</div><div class="sc-label">Main Tabs</div></div>
    <div class="sc"><div class="sc-icon">🔗</div><div class="sc-num">{stats['linked_loaded']}</div><div class="sc-label">Linked Sheets</div></div>
    <div class="sc"><div class="sc-icon">🧠</div><div class="sc-num">{stats['total_chunks']}</div><div class="sc-label">Indexed</div></div>
    <div class="sc"><div class="sc-icon">⚡</div><div class="sc-num">{total_chars // 1000}K</div><div class="sc-label">Characters</div></div>
</div>
""", unsafe_allow_html=True)

# --- Sidebar Details ---
with st.sidebar:
    st.markdown('<div class="sb2-status">✅ Online — All systems operational</div>', unsafe_allow_html=True)

    if stats["failed_links"]:
        with st.expander(f"⚠️ {len(stats['failed_links'])} inaccessible"):
            for fl in stats["failed_links"]:
                st.caption(f"• {fl['label'][:40]} — {fl['error']}")

    st.markdown('<div class="sb2-section">Main Sheets</div>', unsafe_allow_html=True)
    main_html = ""
    for c in main_chunks:
        name = c["label"].split(">")[-1].strip() if ">" in c["label"] else c["label"]
        main_html += f'<div class="sb2-item">{name}</div>'
    st.markdown(main_html, unsafe_allow_html=True)

    st.markdown(f'<div class="sb2-section">Linked Sheets ({len(linked_chunks)})</div>', unsafe_allow_html=True)
    with st.expander("View all"):
        lh = ""
        for c in linked_chunks:
            name = c["label"].split(">")[-1].strip() if ">" in c["label"] else c["label"]
            lh += f'<div class="sb2-item">{name[:48]}</div>'
        st.markdown(lh, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        '<p style="font-size:0.65rem;color:#475569;text-align:center;">'
        'Cache: 5 min<br>'
        '<span style="background:linear-gradient(135deg,#6366f1,#ec4899);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-weight:700;">Powered by NxtWave</span></p>',
        unsafe_allow_html=True,
    )

# --- Chat ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown("""
    <div class="wc">
        <h3>👋 Hey! Ask me anything about NIAT</h3>
        <p>I can search across all your sheets, linked documents, curricula, BOS trackers,
        implementation plans, and more — instantly.</p>
        <div class="wc-grid">
            <div class="wc-chip">BOS status for MRV?</div>
            <div class="wc-chip">Universities with Full Delivery</div>
            <div class="wc-chip">Curriculum link for SGU</div>
            <div class="wc-chip">AOA for CDU 2025</div>
            <div class="wc-chip">AICTE framework universities</div>
            <div class="wc-chip">Student count for Yenepoya</div>
            <div class="wc-chip">Implementation wave details</div>
            <div class="wc-chip">Prod sequence for BITS</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

for msg in st.session_state.messages:
    avatar = "🎓" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask anything about NIAT data..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
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

        with st.expander(f"Sources ({len(relevant)})"):
            for r in relevant:
                score = r.get("score", 0)
                st.caption(f"• {r['label']} — {score:.2f}")

    st.session_state.messages.append({"role": "assistant", "content": response})
