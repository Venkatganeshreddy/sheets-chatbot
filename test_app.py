"""
Comprehensive unit tests for sheets-chatbot app.py
Tests all core functions without requiring Google Sheets credentials or Streamlit.
"""
import re
import json
import sys
import traceback

# We need to mock streamlit before importing app functions
import types

# Create a mock streamlit module
mock_st = types.ModuleType("streamlit")
mock_st.cache_data = lambda **kwargs: (lambda f: f)  # no-op decorator
mock_st.secrets = {}
mock_st.progress = lambda *a, **kw: None
mock_st.write = lambda *a, **kw: None

# Inject mock before importing app functions
sys.modules["streamlit"] = mock_st

# Mock other imports that aren't needed for logic tests
mock_gspread = types.ModuleType("gspread")
sys.modules["gspread"] = mock_gspread
mock_google_oauth = types.ModuleType("google.oauth2.service_account")
mock_google_oauth.Credentials = None
sys.modules["google"] = types.ModuleType("google")
sys.modules["google.oauth2"] = types.ModuleType("google.oauth2")
sys.modules["google.oauth2.service_account"] = mock_google_oauth
mock_dotenv = types.ModuleType("dotenv")
mock_dotenv.load_dotenv = lambda: None
sys.modules["dotenv"] = mock_dotenv
mock_requests = types.ModuleType("requests")
sys.modules["requests"] = mock_requests

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Now import the functions we want to test
# We'll define them locally to avoid streamlit initialization issues
exec(open("app.py", encoding="utf-8").read().split("# --- UI ---")[0])

# ============================================================
# TEST HELPERS
# ============================================================
passed = 0
failed = 0
errors = []

def test(name, condition, detail=""):
    global passed, failed, errors
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        errors.append(f"{name}: {detail}")
        print(f"  FAIL: {name} -- {detail}")


# ============================================================
# TEST 1: normalize_query
# ============================================================
print("\n=== TEST 1: normalize_query ===")

q1 = normalize_query("what is the BOS status of CDU 2024 sem-4")
test("sem-4 expanded", "semester 4" in q1, f"Got: {q1}")
test("sem 4 expanded", "sem 4" in q1, f"Got: {q1}")
test("original preserved", "sem-4" in q1, f"Got: {q1}")
test("BOS preserved", "BOS" in q1, f"Got: {q1}")

q2 = normalize_query("sem4 details")
test("sem4 expanded", "semester 4" in q2, f"Got: {q2}")

q3 = normalize_query("semester 4 curriculum")
test("semester 4 unchanged", "semester 4" in q3, f"Got: {q3}")

q4 = normalize_query("Sem-2 status")
test("case insensitive", "semester 2" in q4, f"Got: {q4}")

q5 = normalize_query("no semester here")
test("no false expansion", q5 == "no semester here", f"Got: {q5}")


# ============================================================
# TEST 2: parse_spreadsheet with None formattedValue
# ============================================================
print("\n=== TEST 2: parse_spreadsheet (None handling) ===")

mock_content = {
    "properties": {"title": "Test Sheet"},
    "sheets": [{
        "properties": {"title": "Tab1"},
        "data": [{
            "rowData": [
                {"values": [
                    {"formattedValue": "Name"},
                    {"formattedValue": "Status"},
                    {"formattedValue": None},  # Explicit null
                ]},
                {"values": [
                    {"formattedValue": "CDU 2024"},
                    {"formattedValue": "Completed"},
                    {},  # Missing key
                ]},
                {"values": [
                    {"formattedValue": None},
                    {"formattedValue": "Pending"},
                ]},
            ]
        }]
    }]
}

title, data, chip_links = parse_spreadsheet(mock_content)
test("title parsed", title == "Test Sheet")
test("tab parsed", "Tab1" in data)
test("headers handle None", data["Tab1"]["headers"] == ["Name", "Status", ""],
     f"Got: {data['Tab1']['headers']}")
test("row1 parsed", data["Tab1"]["rows"][0] == ["CDU 2024", "Completed", ""],
     f"Got: {data['Tab1']['rows'][0]}")
test("row2 None->empty", data["Tab1"]["rows"][1] == ["", "Pending"],
     f"Got: {data['Tab1']['rows'][1]}")


# ============================================================
# TEST 3: format_tab_text with None-safe data
# ============================================================
print("\n=== TEST 3: format_tab_text (None safety) ===")

tab_info = {
    "headers": ["University", "BOS Status", "Semester", "Doc Link"],
    "rows": [
        ["CDU 2024", "Completed", "Sem-4", "https://docs.google.com/abc"],
        ["CDU 2024", "Pending", "Sem-3", ""],
        ["", "Done", "", ""],  # Row with some empty cells
        [None, "Active", None, None],  # None values (shouldn't happen after parse fix, but test anyway)
    ],
    "hyperlinks": [
        {"row": 1, "col": "Doc Link", "text": "Curriculum Doc", "url": "https://docs.google.com/abc"},
    ],
    "notes": [],
    "dropdowns": [],
}

try:
    text = format_tab_text("Tracker", tab_info, source_label="MAIN: Test")
    test("format_tab_text succeeds", True)
    test("contains CDU 2024", "CDU 2024" in text, f"Missing CDU 2024")
    test("contains BOS Status", "BOS Status" in text, f"Missing BOS Status")
    test("contains Sem-4", "Sem-4" in text, f"Missing Sem-4")
    test("contains doc link", "https://docs.google.com/abc" in text, f"Missing doc link")
    test("handles None row", "Active" in text, f"Missing Active from None row")
except Exception as e:
    test("format_tab_text succeeds", False, f"Crashed: {e}")


# ============================================================
# TEST 4: create_tab_chunks - small tab (single chunk)
# ============================================================
print("\n=== TEST 4: create_tab_chunks (small tab) ===")

small_info = {
    "headers": ["University", "BOS Status", "Semester"],
    "rows": [
        ["CDU 2024", "Completed", "Sem-4"],
        ["SGU 2025", "Pending", "Sem-1"],
    ],
    "hyperlinks": [],
    "notes": [],
    "dropdowns": [],
}

chunks_small = create_tab_chunks("Tracker", small_info, source_label="MAIN: Test", chunk_id_prefix="main")
test("single chunk for small tab", len(chunks_small) == 1, f"Got {len(chunks_small)} chunks")
test("chunk has id", "id" in chunks_small[0])
test("chunk has text", "CDU 2024" in chunks_small[0]["text"])


# ============================================================
# TEST 5: create_tab_chunks - large tab (multiple sub-chunks)
# ============================================================
print("\n=== TEST 5: create_tab_chunks (large tab, sub-chunking) ===")

large_rows = []
for i in range(100):
    large_rows.append([f"Univ-{i}", f"Status-{i}", f"Sem-{i % 8}"])

large_info = {
    "headers": ["University", "BOS Status", "Semester"],
    "rows": large_rows,
    "hyperlinks": [
        {"row": 0, "col": "University", "text": "Header Link", "url": "https://example.com/header"},
        {"row": 5, "col": "BOS Status", "text": "Link5", "url": "https://example.com/5"},
        {"row": 50, "col": "BOS Status", "text": "Link50", "url": "https://example.com/50"},
    ],
    "notes": [
        {"row": 0, "col": "University", "text": "Header", "note": "Header note"},
        {"row": 10, "col": "BOS Status", "text": "Status-9", "note": "Important note"},
    ],
    "dropdowns": [],
}

chunks_large = create_tab_chunks("BigTracker", large_info, source_label="MAIN: Test", chunk_id_prefix="main")
test("multiple sub-chunks", len(chunks_large) == 3, f"Expected 3, got {len(chunks_large)}")  # 100 rows / 40 = 3 chunks

# Check first sub-chunk has header row links
first_chunk_text = chunks_large[0]["text"]
test("first chunk has header link", "https://example.com/header" in first_chunk_text,
     "Header-row link missing from first chunk")
test("first chunk has header note", "Header note" in first_chunk_text,
     "Header-row note missing from first chunk")

# Check row 5's link is in first chunk (rows 1-40)
test("row 5 link in chunk 1", "https://example.com/5" in first_chunk_text,
     "Row 5 link missing from first chunk")

# Check row 50's link is in second chunk (rows 41-80)
second_chunk_text = chunks_large[1]["text"]
test("row 50 link in chunk 2", "https://example.com/50" in second_chunk_text,
     "Row 50 link missing from second chunk")

# Check chunk IDs are unique
chunk_ids = [c["id"] for c in chunks_large]
test("unique chunk IDs", len(chunk_ids) == len(set(chunk_ids)), f"Duplicate IDs: {chunk_ids}")

# Check all rows are covered
all_text = " ".join(c["text"] for c in chunks_large)
test("row 1 present", "Univ-0" in all_text)
test("row 40 present", "Univ-39" in all_text)
test("row 41 present", "Univ-40" in all_text)
test("row 100 present", "Univ-99" in all_text)


# ============================================================
# TEST 6: create_tab_chunks - empty tab
# ============================================================
print("\n=== TEST 6: create_tab_chunks (empty tab) ===")

empty_info = {"headers": [], "rows": [], "hyperlinks": [], "notes": [], "dropdowns": []}
chunks_empty = create_tab_chunks("Empty", empty_info, chunk_id_prefix="main")
test("empty tab -> no chunks", len(chunks_empty) == 0)

one_header = {"headers": ["A", "B"], "rows": [], "hyperlinks": [], "notes": [], "dropdowns": []}
chunks_header_only = create_tab_chunks("HeaderOnly", one_header, chunk_id_prefix="main")
# format_tab_text with no rows returns just headers line - should have text
test("header-only tab", len(chunks_header_only) <= 1)


# ============================================================
# TEST 7: build_retriever - normal and empty
# ============================================================
print("\n=== TEST 7: build_retriever ===")

test_chunks = [
    {"id": "1", "text": "CDU 2024 BOS Status Completed Semester 4", "is_main": True},
    {"id": "2", "text": "SGU 2025 Curriculum Pending Semester 1", "is_main": True},
    {"id": "3", "text": "CDU 2024 Document shared BOS meeting notes link", "is_main": False},
]

vectorizer, tfidf_matrix = build_retriever(test_chunks)
test("vectorizer created", vectorizer is not None)
test("matrix shape", tfidf_matrix.shape[0] == 3, f"Got shape: {tfidf_matrix.shape}")

# Empty chunks
v_empty, m_empty = build_retriever([])
test("empty returns None vectorizer", v_empty is None)
test("empty returns None matrix", m_empty is None)


# ============================================================
# TEST 8: retrieve_relevant_chunks - BOS status query
# ============================================================
print("\n=== TEST 8: retrieve_relevant_chunks (BOS status query) ===")

test_chunks_full = [
    {"id": "main/Tracker/rows_1_40", "text": """[MAIN: AOL] Sheet: Tracker (Rows 1-40)
Columns: University | BOS Status | Semester | Doc Link
Row 1: University: CDU 2024 | BOS Status: Completed | Semester: Sem-4 | Doc Link: https://docs.google.com/bos-doc
Row 2: University: CDU 2024 | BOS Status: Pending | Semester: Sem-3
Row 3: University: SGU 2025 | BOS Status: In Progress | Semester: Sem-1""", "is_main": True, "label": "AOL > Tracker"},

    {"id": "main/Curriculum/rows_1_40", "text": """[MAIN: AOL] Sheet: Curriculum (Rows 1-40)
Columns: Subject | Code | Credits
Row 1: Subject: Mathematics | Code: MTH101 | Credits: 4
Row 2: Subject: Physics | Code: PHY101 | Credits: 3""", "is_main": True, "label": "AOL > Curriculum"},

    {"id": "linked/abc123/BOS_Docs/rows_1_40", "text": """[LINKED: BOS Documents] Sheet: BOS_Docs (Rows 1-40)
Columns: University | Meeting Date | Document | Status
Row 1: University: CDU 2024 | Meeting Date: 2024-01-15 | Document: https://docs.google.com/bos-minutes | Status: Approved
Row 2: University: SGU 2025 | Meeting Date: 2024-03-20 | Document: https://docs.google.com/sgu-minutes | Status: Draft""", "is_main": False, "label": "BOS Documents > BOS_Docs"},
]

v, m = build_retriever(test_chunks_full)

# Query 1: BOS status of CDU 2024 sem-4
results1 = retrieve_relevant_chunks("what is the BOS status of CDU 2024 sem-4", test_chunks_full, v, m)
test("returns results", len(results1) > 0, f"Got {len(results1)} results")

if results1:
    top_result = results1[0]
    test("top result has BOS data", "BOS Status" in top_result["text"])
    test("top result has CDU 2024", "CDU 2024" in top_result["text"])
    test("top result has Sem-4", "Sem-4" in top_result["text"])

    # Check that Tracker (with BOS Status column) ranks higher than Curriculum
    tracker_idx = next((i for i, r in enumerate(results1) if "Tracker" in r["id"]), -1)
    curriculum_idx = next((i for i, r in enumerate(results1) if "Curriculum" in r["id"]), -1)
    if tracker_idx >= 0 and curriculum_idx >= 0:
        test("Tracker ranks above Curriculum", tracker_idx < curriculum_idx,
             f"Tracker at {tracker_idx}, Curriculum at {curriculum_idx}")

# Query 2: Document shared at time of BOS
results2 = retrieve_relevant_chunks("what was the document shared at the time of BOS", test_chunks_full, v, m)
test("doc query returns results", len(results2) > 0)

if results2:
    # Check BOS Documents chunk ranks high
    all_text = " ".join(r["text"] for r in results2)
    test("doc query finds BOS docs", "bos-minutes" in all_text or "bos-doc" in all_text,
         f"No BOS document URLs found in results")

# Query 3: Empty result for nonsense query
results3 = retrieve_relevant_chunks("xyzzy foobar nonexistent", test_chunks_full, v, m)
# Should still return some results (keyword matching might be 0 but TF-IDF could find something)
test("nonsense query handled gracefully", isinstance(results3, list))

# Query 4: Empty chunks
results4 = retrieve_relevant_chunks("test query", [], None, None)
test("empty chunks returns empty", results4 == [], f"Got: {results4}")


# ============================================================
# TEST 9: Retrieval scoring - column header boost
# ============================================================
print("\n=== TEST 9: Column header boost ===")

if results1:
    # The Tracker chunk has "BOS Status" in its Columns line
    # The BOS_Docs chunk has "Status" and "Document" in its Columns line
    # Both should score higher than Curriculum for a "BOS status" query
    scores = {r["id"]: r["score"] for r in results1}
    tracker_score = scores.get("main/Tracker/rows_1_40", 0)
    curriculum_score = scores.get("main/Curriculum/rows_1_40", 0)
    test("column boost works", tracker_score > curriculum_score,
         f"Tracker: {tracker_score:.3f}, Curriculum: {curriculum_score:.3f}")


# ============================================================
# TEST 10: parse_spreadsheet with hyperlinks and chips
# ============================================================
print("\n=== TEST 10: parse_spreadsheet (hyperlinks, chips, linked detection) ===")

mock_with_chips = {
    "properties": {"title": "Main Sheet"},
    "sheets": [{
        "properties": {"title": "Overview"},
        "data": [{
            "rowData": [
                {"values": [
                    {"formattedValue": "University"},
                    {"formattedValue": "Linked Doc"},
                ]},
                {"values": [
                    {"formattedValue": "CDU"},
                    {
                        "formattedValue": "BOS Document",
                        "hyperlink": "https://docs.google.com/spreadsheets/d/LINKED_SHEET_ID_123/edit",
                        "chipRuns": [{
                            "chip": {
                                "richLinkProperties": {
                                    "uri": "https://docs.google.com/spreadsheets/d/CHIP_SHEET_ID_456/edit"
                                }
                            }
                        }]
                    },
                ]},
            ]
        }]
    }]
}

title, data, chip_links = parse_spreadsheet(mock_with_chips, is_linked=False)
test("detects chip linked sheet", len(chip_links) > 0, f"Got {len(chip_links)} chip links")
if chip_links:
    test("correct linked sheet ID", chip_links[0]["sheet_id"] == "CHIP_SHEET_ID_456",
         f"Got: {chip_links[0]['sheet_id']}")

# When is_linked=True, should NOT follow chips
_, _, chip_links2 = parse_spreadsheet(mock_with_chips, is_linked=True)
test("linked sheets don't recurse", len(chip_links2) == 0, f"Got {len(chip_links2)} chip links")


# ============================================================
# TEST 11: Edge case - tab with exactly ROWS_PER_CHUNK rows
# ============================================================
print("\n=== TEST 11: Boundary - exactly ROWS_PER_CHUNK rows ===")

exact_rows = [[f"Val-{i}", f"S-{i}"] for i in range(ROWS_PER_CHUNK)]
exact_info = {"headers": ["A", "B"], "rows": exact_rows, "hyperlinks": [], "notes": [], "dropdowns": []}
chunks_exact = create_tab_chunks("Exact", exact_info, chunk_id_prefix="main")
test("exactly ROWS_PER_CHUNK -> single chunk", len(chunks_exact) == 1,
     f"Got {len(chunks_exact)} chunks for {ROWS_PER_CHUNK} rows")

# One more row -> 2 chunks
plus_one = exact_rows + [["Extra", "Row"]]
plus_info = {"headers": ["A", "B"], "rows": plus_one, "hyperlinks": [], "notes": [], "dropdowns": []}
chunks_plus = create_tab_chunks("PlusOne", plus_info, chunk_id_prefix="main")
test("ROWS_PER_CHUNK+1 -> 2 chunks", len(chunks_plus) == 2,
     f"Got {len(chunks_plus)} chunks for {ROWS_PER_CHUNK + 1} rows")


# ============================================================
# TEST 12: Semester normalization in retrieval
# ============================================================
print("\n=== TEST 12: Semester normalization in retrieval ===")

sem_chunks = [
    {"id": "1", "text": "University: CDU | Semester: semester 4 | Status: Done", "is_main": True, "label": "BOS Tracker Sem 4"},
    {"id": "2", "text": "University: SGU | Semester: Sem-1 | Status: Active", "is_main": True, "label": "BOS Tracker Sem 1"},
]
v_sem, m_sem = build_retriever(sem_chunks)

# "sem-4" should match "semester 4" thanks to normalization
r_sem = retrieve_relevant_chunks("CDU sem-4", sem_chunks, v_sem, m_sem)
test("sem-4 matches semester 4", len(r_sem) > 0)
if r_sem:
    all_texts = " ".join(r["text"] for r in r_sem)
    test("CDU semester 4 chunk in results", "semester 4" in all_texts and "CDU" in all_texts,
         f"Results: {all_texts[:100]}")


# ============================================================
# RESULTS
# ============================================================
print(f"\n{'='*60}")
print(f"RESULTS: {passed} passed, {failed} failed out of {passed + failed} tests")
if errors:
    print(f"\nFailed tests:")
    for e in errors:
        print(f"  - {e}")
print(f"{'='*60}")

sys.exit(0 if failed == 0 else 1)
