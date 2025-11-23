# # doc_analyzer.py
# # Document Analyzer module for GovGuide (cleaner UI)

# import io
# import time
# import json
# import uuid
# import logging
# from typing import List, Dict, Any

# from PIL import Image
# import streamlit as st

# # Optional heavy deps - import if available
# try:
#     import pdfplumber
# except Exception:
#     pdfplumber = None

# try:
#     import pytesseract
# except Exception:
#     pytesseract = None

# try:
#     import cv2 # type: ignore
#     _HAS_CV2 = True
# except Exception:
#     cv2 = None
#     _HAS_CV2 = False

# # PDF report generator (optional)
# try:
#     from reportlab.lib.pagesizes import A4
#     from reportlab.pdfgen import canvas
# except Exception:
#     reportlab = None

# # Import the LLM calling function from app.py
# CALL_LLM_AVAILABLE = False
# try:
#     from app import call_llm
#     CALL_LLM_AVAILABLE = True
# except ImportError:
#     # This should not happen if app.py is correctly executed first, 
#     # but include a safety check for development/testing environments.
#     pass

# # Safety check: If app.py didn't define call_llm, use a stub that prints an error.
# if not CALL_LLM_AVAILABLE:
#     def call_llm(prompt: str) -> str:
#         st.error("LLM connection not established. Check app.py for `call_llm`.")
#         return '{"title": "Connection Error", "bullets": ["LLM connection failed."], "clauses": [], "actions": [], "department": "System"}'

# # Configuration
# MAX_SUMMARY_CHUNKS = 4 # Hardcode to a reasonable default
# CHUNK_SIZE = 1200
# OVERLAP = 200

# # Logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("doc_analyzer")

# # -----------------------------
# # Utility / preprocessing
# # -----------------------------
# def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> List[str]:
#     """Split text into overlapping chunks for LLM processing."""
#     if not text:
#         return []
#     text = text.replace('\r', '\n')
#     if len(text) <= chunk_size:
#         return [text]
#     chunks = []
#     start = 0
#     l = len(text)
#     while start < l:
#         end = min(start + chunk_size, l)
#         chunks.append(text[start:end])
#         if end == l:
#             break
#         start = max(end - overlap, end)
#     # filter tiny chunks
#     return [c.strip() for c in chunks if len(c.strip()) > 50]

# # -----------------------------
# # OCR / Extraction (using original logic)
# # -----------------------------
# def extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
#     """Best-effort extraction from PDFs."""
#     if pdfplumber is None: return ""
#     try:
#         text_pages = []
#         with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
#             for p in pdf.pages:
#                 try: t = p.extract_text()
#                 except Exception: t = None
#                 if t: text_pages.append(t)
#         return "\n\n".join(text_pages)
#     except Exception as e:
#         logger.exception("PDF extraction failed: %s", e)
#         return ""

# def _preprocess_image_for_ocr(pil_img: Image.Image) -> Image.Image:
#     """Use OpenCV preprocessing if available for better OCR results."""
#     if not _HAS_CV2:
#         return pil_img
#     try:
#         import numpy as np
#         arr = np.array(pil_img.convert("RGB"))
#         gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
#         # upscale
#         h, w = gray.shape
#         scale = 1.5 if max(h, w) < 2000 else 1.0
#         if scale != 1.0:
#             gray = cv2.resize(gray, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
#         # denoise and adaptive threshold
#         den = cv2.bilateralFilter(gray, 9, 75, 75)
#         th = cv2.adaptiveThreshold(den, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
#         return Image.fromarray(th)
#     except Exception as e:
#         logger.exception("OpenCV preprocessing failed: %s", e)
#         return pil_img

# def extract_text_from_image_bytes(file_bytes: bytes) -> str:
#     """Run OCR using pytesseract."""
#     if pytesseract is None: return ""
#     try:
#         pil = Image.open(io.BytesIO(file_bytes)).convert("RGB")
#     except Exception as e:
#         logger.exception("Failed to open image bytes: %s", e)
#         return ""
#     proc_img = _preprocess_image_for_ocr(pil)
#     try:
#         txt = pytesseract.image_to_string(proc_img, config="--psm 3")
#         return txt or ""
#     except Exception as e:
#         logger.exception("pytesseract failed: %s", e)
#         # try raw
#         try: return pytesseract.image_to_string(pil)
#         except Exception: return ""

# # -----------------------------
# # LLM Prompt (force JSON output)
# # -----------------------------
# PROMPT_TEMPLATE = (
#     "You are a civic document summarizer. Given the document text, return a JSON object EXACTLY "
#     "with the following keys: title (string), bullets (array of short strings), clauses (array of strings), "
#     "actions (array of short strings), department (string). Do NOT include any extra commentary or text outside "
#     "the JSON object. If any field is unknown, return an empty string or empty array for that key.\n\n"
#     "Example:\n"
#     "{\"title\":\"Notice about water supply\",\"bullets\":[\"point1\",\"point2\"],\"clauses\":[\"clause1\"],"
#     "\"actions\":[\"action1\"],\"department\":\"Municipal Corporation\"}\n\n"
#     "Document text:\n\n"
# )

# # -----------------------------
# # Parse LLM output (JSON-first)
# # -----------------------------
# def parse_llm_text(text: str) -> Dict[str, Any]:
#     """Try to parse JSON output first."""
#     out = {"title": None, "bullets": [], "clauses": [], "actions": [], "department": None}
#     if not text or not text.strip():
#         return out
    
#     # Try JSON parsing
#     try:
#         # Find first JSON object in text
#         start = text.find('{')
#         end = text.rfind('}')
#         if start != -1 and end != -1 and end > start:
#             candidate = text[start:end+1]
#             parsed = json.loads(candidate)
#             out['title'] = parsed.get('title') if isinstance(parsed.get('title'), str) else None
#             out['bullets'] = parsed.get('bullets') if isinstance(parsed.get('bullets'), list) else []
#             out['clauses'] = parsed.get('clauses') if isinstance(parsed.get('clauses'), list) else []
#             out['actions'] = parsed.get('actions') if isinstance(parsed.get('actions'), list) else []
#             out['department'] = parsed.get('department') if isinstance(parsed.get('department'), str) else None
#             return out
#     except Exception as e:
#         logger.warning("JSON parse failed: %s", e)
#         # If JSON fails, use simple text fallback for bullets/title
#         out['title'] = "Analysis Incomplete (JSON Parse Error)"
#         out['bullets'] = [f"Could not parse AI response: {e}. Check extracted text."]
#         return out
#     return out

# def safe_call_llm(prompt: str) -> str:
#     """Call call_llm with error handling and logs. Returns empty string on failure."""
#     try:
#         t0 = time.time()
#         resp = call_llm(prompt)
#         t1 = time.time()
#         logger.info("LLM call took %.2fs", t1 - t0)
#         return resp if isinstance(resp, str) and resp.strip() else ""
#     except Exception as e:
#         logger.exception("LLM call failed: %s", e)
#         return ""

# # -----------------------------
# # Summarization orchestration (Simplified)
# # -----------------------------
# def summarize_text_with_llm(text: str, max_chunks: int = MAX_SUMMARY_CHUNKS) -> Dict[str, Any]:
#     """Simplified summarization: send first N chunks to the LLM for a single pass."""
    
#     chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
#     if not chunks:
#         return {"title": "Document Empty", "bullets": ["No readable text was extracted."], "clauses": [], "actions": [], "department": "System"}

#     # Use only the first MAX_SUMMARY_CHUNKS to save tokens/time
#     context = "\n\n".join(chunks[:max_chunks])
    
#     prompt = PROMPT_TEMPLATE + context
#     raw = safe_call_llm(prompt)
    
#     parsed = parse_llm_text(raw)
    
#     return parsed

# # -----------------------------
# # PDF Report generator
# # -----------------------------
# def build_pdf_report(fname: str, analysis_dict: Dict[str, Any]) -> bytes | None:
#     try:
#         from reportlab.lib.pagesizes import A4
#         from reportlab.pdfgen import canvas
#     except Exception:
#         return None
#     buffer = io.BytesIO()
#     c = canvas.Canvas(buffer, pagesize=A4)
#     width, height = A4
#     margin = 40
#     y = height - margin
#     c.setFont("Helvetica-Bold", 14)
#     c.drawString(margin, y, f"Document Analysis Report: {fname}")
#     y -= 30
#     def write_block(title, lines):
#         nonlocal y, c
#         c.setFont("Helvetica-Bold", 12)
#         c.drawString(margin, y, title)
#         y -= 16
#         c.setFont("Helvetica", 11)
#         for line in lines:
#             for sub in split_lines(line, 90):
#                 c.drawString(margin + 10, y, sub)
#                 y -= 14
#                 if y < 80:
#                     c.showPage()
#                     y = height - margin
#         y -= 8
        
#     write_block("TITLE:", [analysis_dict.get("title") or "(No title)"])
#     write_block("SUMMARY:", analysis_dict.get("bullets", []))
#     write_block("KEY CLAUSES:", analysis_dict.get("clauses", []))
#     write_block("ACTIONS:", analysis_dict.get("actions", []))
#     write_block("DEPARTMENT:", [analysis_dict.get("department") or "(Not identified)"])
#     c.save()
#     buffer.seek(0)
#     return buffer.read()

# def split_lines(text: str, width: int):
#     words = text.split()
#     lines = []
#     cur = []
#     cur_len = 0
#     for w in words:
#         if cur_len + len(w) + 1 > width:
#             lines.append(' '.join(cur))
#             cur = [w]
#             cur_len = len(w)
#         else:
#             cur.append(w)
#             cur_len += len(w) + 1
#     if cur:
#         lines.append(' '.join(cur))
#     return lines

# # -----------------------------
# # Streamlit UI function (Cleaned Up)
# # -----------------------------
# def run_document_analyzer_tab():
#     st.title("Document Analyzer ")
#     st.write("Upload a government notice, bill, or civic document (PDF / image). The AI will summarize it, list key clauses, and produce action items.")

#     uploaded_file = st.file_uploader("Choose a PDF or image file", type=['pdf','png','jpg','jpeg'])
    
#     if uploaded_file is None:
#         st.info("Upload a document to begin analysis.")
#         return

#     fname = uploaded_file.name
#     bytes_data = uploaded_file.read()
#     st.info(f"Processing **{fname}**...")

#     # Extraction Logic
#     extracted_text = ""
#     if fname.lower().endswith('.pdf') and pdfplumber is not None:
#         with st.spinner('Extracting digital text from PDF...'):
#             extracted_text = extract_text_from_pdf_bytes(bytes_data)
    
#     if not extracted_text.strip():
#         # Fallback to OCR if no text found or if it's an image
#         if pytesseract is not None:
#             with st.spinner('Running OCR on document/image (for scanned text)...'):
#                 extracted_text = extract_text_from_image_bytes(bytes_data)
#         else:
#             st.warning("No OCR library found (pytesseract). Cannot process images or scanned PDFs.")

#     if not extracted_text.strip():
#         st.error('No readable text extracted from the document.')
#         return

#     # Show text preview in a controlled expander
#     with st.expander("Preview Extracted Text"):
#         st.text_area("Extracted text preview", value=extracted_text[:3000], height=200, disabled=True)

#     # Summarize
#     with st.spinner('Generating summary and actions with AI...'):
#         # Pass hardcoded MAX_SUMMARY_CHUNKS (4)
#         analysis = summarize_text_with_llm(extracted_text, max_chunks=MAX_SUMMARY_CHUNKS)
    
#     # Display results
#     st.markdown("----")
#     st.header('Analysis Results')
    
#     # Title
#     title = analysis.get('title') or "Document Analysis"
#     st.subheader(f'**{title}**')
    
#     # Summary
#     st.markdown("### 📋 Summary")
#     for b in analysis.get('bullets', []):
#         st.write(f'• {b}')

#     # Key Clauses
#     st.markdown("### 🔑 Key Clauses")
#     if analysis.get('clauses'):
#         for c in analysis.get('clauses', []):
#             st.write(f'• {c}')
#     else:
#         st.info("No specific key clauses were identified.")

#     # Actionable Steps
#     st.markdown("### ✅ Actionable Steps")
#     if analysis.get('actions'):
#         for a in analysis.get('actions', []):
#             st.write(f'**{a}**')
#     else:
#         st.info("No clear action items were identified.")

#     # Responsible Department
#     st.markdown("### 🏛️ Responsible Department")
#     st.write(analysis.get('department') or 'Not identified')

#     # Follow-up Q&A
#     st.markdown('---')
#     st.subheader('Ask a follow-up question')
#     follow_q = st.text_input('Ask a question (context: analyzed document)')
#     if st.button('Ask'):
#         if follow_q.strip():
#             context = ' '.join(chunk_text(extracted_text)[:MAX_SUMMARY_CHUNKS])
#             follow_prompt = (
#                 "You are a civic assistant. Use the following document context to answer the question concisely:\n\n"
#                 f"Context:\n{context}\n\nUser question: {follow_q}\nAnswer in simple language suitable for citizens."
#             )
#             with st.spinner("Getting answer..."):
#                 follow_raw = safe_call_llm(follow_prompt)
#             st.info(follow_raw or "No answer returned from the AI.")
#         else:
#             st.warning("Please type a question.")

#     # Generate Report
#     st.markdown('---')
#     if st.button('Generate & Download PDF Report'):
#         with st.spinner('Creating PDF report...'):
#             report_bytes = build_pdf_report(fname, analysis)
#             if report_bytes is None:
#                 st.error('PDF generator (reportlab) is not available on this environment. Please install it.')
#             else:
#                 st.success('Report ready — click download below')
#                 st.download_button('Download Report (PDF)', data=report_bytes, file_name=f"analysis_{uuid.uuid4().hex}.pdf", mime='application/pdf')

# # If run directly, provide a simple streamlit page for testing
# if __name__ == '__main__':
#     st.set_page_config(page_title='Document Analyzer (Standalone)')
#     run_document_analyzer_tab()




















































# doc_analyzer.py  (modified)
import io
import time
import json
import uuid
import logging
from typing import List, Dict, Any

from PIL import Image
import streamlit as st

# Optional heavy deps - import if available
try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    import pytesseract
except Exception:
    pytesseract = None

try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except Exception:
    cv2 = None
    _HAS_CV2 = False

# PDF report generator (optional)
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
except Exception:
    reportlab = None

# Import LLM client (no circular import)
try:
    from llm_client import call_llm
except Exception:
    # graceful fallback if llm_client not available
    def call_llm(prompt: str) -> str:
        st.error("LLM client unavailable. Please ensure llm_client.py is present and configured.")
        return ""

# Configuration
MAX_SUMMARY_CHUNKS = 4
CHUNK_SIZE = 1200
OVERLAP = 200

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("doc_analyzer")

# -----------------------------
# Utility / preprocessing
# -----------------------------
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> List[str]:
    """Split text into overlapping chunks for LLM processing."""
    if not text:
        return []
    text = text.replace('\r', '\n')
    if len(text) <= chunk_size:
        return [text.strip()]
    chunks = []
    start = 0
    l = len(text)
    while start < l:
        end = min(start + chunk_size, l)
        chunks.append(text[start:end].strip())
        if end == l:
            break
        # fix overlap: step back by overlap, but not before 0
        start = end - overlap if (end - overlap) > 0 else 0
        if start >= end:
            # fail-safe to avoid infinite loops
            start = end
    # filter tiny chunks
    return [c for c in chunks if len(c) > 50]

# -----------------------------
# OCR / Extraction (using original logic)
# -----------------------------
def extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
    """Best-effort extraction from PDFs."""
    if pdfplumber is None:
        return ""
    try:
        text_pages = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for p in pdf.pages:
                try:
                    t = p.extract_text()
                except Exception:
                    t = None
                if t:
                    text_pages.append(t)
        return "\n\n".join(text_pages)
    except Exception as e:
        logger.exception("PDF extraction failed: %s", e)
        return ""

def _preprocess_image_for_ocr(pil_img: Image.Image) -> Image.Image:
    """Use OpenCV preprocessing if available for better OCR results."""
    if not _HAS_CV2:
        return pil_img
    try:
        import numpy as np
        arr = np.array(pil_img.convert("RGB"))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        # upscale if small
        h, w = gray.shape
        scale = 1.5 if max(h, w) < 2000 else 1.0
        if scale != 1.0:
            gray = cv2.resize(gray, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
        # denoise and adaptive threshold
        den = cv2.bilateralFilter(gray, 9, 75, 75)
        th = cv2.adaptiveThreshold(den, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
        return Image.fromarray(th)
    except Exception as e:
        logger.exception("OpenCV preprocessing failed: %s", e)
        return pil_img

def extract_text_from_image_bytes(file_bytes: bytes) -> str:
    """Run OCR using pytesseract."""
    if pytesseract is None:
        return ""
    try:
        pil = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as e:
        logger.exception("Failed to open image bytes: %s", e)
        return ""
    proc_img = _preprocess_image_for_ocr(pil)
    try:
        txt = pytesseract.image_to_string(proc_img, config="--psm 3")
        return txt or ""
    except Exception as e:
        logger.exception("pytesseract failed: %s", e)
        try:
            return pytesseract.image_to_string(pil)
        except Exception:
            return ""

# -----------------------------
# LLM Prompt (force JSON output)
# -----------------------------
PROMPT_TEMPLATE = (
    "You are a civic document summarizer. Given the document text, return a JSON object EXACTLY "
    "with the following keys: title (string), bullets (array of short strings), clauses (array of strings), "
    "actions (array of short strings), department (string). Do NOT include any extra commentary or text outside "
    "the JSON object. If any field is unknown, return an empty string or empty array for that key.\n\n"
    "Example:\n"
    "{\"title\":\"Notice about water supply\",\"bullets\":[\"point1\",\"point2\"],\"clauses\":[\"clause1\"],"
    "\"actions\":[\"action1\"],\"department\":\"Municipal Corporation\"}\n\n"
    "Document text:\n\n"
)

# -----------------------------
# Parse LLM output (JSON-first)
# -----------------------------
def parse_llm_text(text: str) -> Dict[str, Any]:
    """Try to parse JSON output first and provide useful fallback info."""
    out = {"title": "", "bullets": [], "clauses": [], "actions": [], "department": ""}
    if not text or not text.strip():
        return out

    # Try JSON parsing
    try:
        # Find the first JSON object in text (balanced braces)
        start = text.find('{')
        if start == -1:
            raise ValueError("No JSON object found in response.")
        # attempt to find matching closing brace by simple rfind
        end = text.rfind('}')
        if end == -1 or end <= start:
            raise ValueError("No closing brace for JSON found.")
        candidate = text[start:end+1]
        parsed = json.loads(candidate)
        out['title'] = parsed.get('title') or ""
        out['bullets'] = parsed.get('bullets') if isinstance(parsed.get('bullets'), list) else []
        out['clauses'] = parsed.get('clauses') if isinstance(parsed.get('clauses'), list) else []
        out['actions'] = parsed.get('actions') if isinstance(parsed.get('actions'), list) else []
        out['department'] = parsed.get('department') or ""
        return out
    except Exception as e:
        logger.warning("JSON parse failed: %s; returning fallback result.", e)
        # Fallback: give highest-level text as one bullet for debugging
        cleaned = text.strip().replace('\n', ' ')
        out['title'] = "Analysis (raw output)"
        out['bullets'] = [cleaned[:300] + ("..." if len(cleaned) > 300 else "")]
        return out

def safe_call_llm(prompt: str) -> str:
    """Call call_llm with error handling and logs. Returns empty string on failure."""
    try:
        t0 = time.time()
        resp = call_llm(prompt)
        t1 = time.time()
        logger.info("LLM call took %.2fs", t1 - t0)
        return resp if isinstance(resp, str) else str(resp)
    except Exception as e:
        logger.exception("LLM call failed: %s", e)
        return ""

# -----------------------------
# Summarization orchestration (Simplified)
# -----------------------------
def summarize_text_with_llm(text: str, max_chunks: int = MAX_SUMMARY_CHUNKS) -> Dict[str, Any]:
    """Simplified summarization: send first N chunks to the LLM for a single pass."""
    chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
    if not chunks:
        return {"title": "Document Empty", "bullets": ["No readable text was extracted."], "clauses": [], "actions": [], "department": "System"}

    # Use only the first N chunks to conserve tokens/time
    context = "\n\n".join(chunks[:max_chunks])
    prompt = PROMPT_TEMPLATE + context
    raw = safe_call_llm(prompt)
    parsed = parse_llm_text(raw)
    return parsed

# -----------------------------
# PDF Report generator
# -----------------------------
def build_pdf_report(fname: str, analysis_dict: Dict[str, Any]) -> bytes | None:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
    except Exception:
        return None
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    margin = 40
    y = height - margin
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y, f"Document Analysis Report: {fname}")
    y -= 30
    def write_block(title, lines):
        nonlocal y, c
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, y, title)
        y -= 16
        c.setFont("Helvetica", 11)
        for line in lines:
            for sub in split_lines(line, 90):
                c.drawString(margin + 10, y, sub)
                y -= 14
                if y < 80:
                    c.showPage()
                    y = height - margin
        y -= 8

    write_block("TITLE:", [analysis_dict.get("title") or "(No title)"])
    write_block("SUMMARY:", analysis_dict.get("bullets", []))
    write_block("KEY CLAUSES:", analysis_dict.get("clauses", []))
    write_block("ACTIONS:", analysis_dict.get("actions", []))
    write_block("DEPARTMENT:", [analysis_dict.get("department") or "(Not identified)"])
    c.save()
    buffer.seek(0)
    return buffer.read()

def split_lines(text: str, width: int):
    words = text.split()
    lines = []
    cur = []
    cur_len = 0
    for w in words:
        if cur_len + len(w) + 1 > width:
            lines.append(' '.join(cur))
            cur = [w]
            cur_len = len(w)
        else:
            cur.append(w)
            cur_len += len(w) + 1
    if cur:
        lines.append(' '.join(cur))
    return lines

# -----------------------------
# Streamlit UI function (Cleaned Up; no emojis)
# -----------------------------
def run_document_analyzer_tab():
    st.title("Document Analyzer")
    st.write("Upload a government notice, bill, or civic document (PDF / image). The AI will summarize it, list key clauses, and produce action items.")

    uploaded_file = st.file_uploader("Choose a PDF or image file", type=['pdf','png','jpg','jpeg'])

    if uploaded_file is None:
        st.info("Upload a document to begin analysis.")
        return

    fname = uploaded_file.name
    bytes_data = uploaded_file.read()
    st.info(f"Processing **{fname}**...")

    # Extraction Logic
    extracted_text = ""
    if fname.lower().endswith('.pdf') and pdfplumber is not None:
        with st.spinner('Extracting digital text from PDF...'):
            extracted_text = extract_text_from_pdf_bytes(bytes_data)

    if not extracted_text.strip():
        # Fallback to OCR if no text found or if it's an image
        if pytesseract is not None:
            with st.spinner('Running OCR on document/image (for scanned text)...'):
                extracted_text = extract_text_from_image_bytes(bytes_data)
        else:
            st.warning("No OCR library found (pytesseract). Cannot process images or scanned PDFs.")

    if not extracted_text.strip():
        st.error('No readable text extracted from the document.')
        return

    # Show text preview in a controlled expander
    with st.expander("Preview Extracted Text"):
        st.text_area("Extracted text preview", value=extracted_text[:3000], height=200, disabled=True)

    # Summarize
    with st.spinner('Generating summary and actions with AI...'):
        analysis = summarize_text_with_llm(extracted_text, max_chunks=MAX_SUMMARY_CHUNKS)

    # Display results
    st.markdown("----")
    st.header('Analysis Results')

    # Title
    title = analysis.get('title') or "Document Analysis"
    st.subheader(f'{title}')

    # Summary
    st.markdown("### Summary")
    bullets = analysis.get('bullets', [])
    if bullets:
        for b in bullets:
            st.write(f"- {b}")
    else:
        st.info("No summary points found.")

    # Key Clauses
    st.markdown("### Key Clauses")
    if analysis.get('clauses'):
        for c in analysis.get('clauses', []):
            st.write(f"- {c}")
    else:
        st.info("No specific key clauses were identified.")

    # Actionable Steps
    st.markdown("### Actionable Steps")
    if analysis.get('actions'):
        for idx, a in enumerate(analysis.get('actions', []), start=1):
            st.write(f"{idx}. {a}")
    else:
        st.info("No clear action items were identified.")

    # Responsible Department
    st.markdown("### Responsible Department")
    st.write(analysis.get('department') or 'Not identified')

    # Follow-up Q&A
    st.markdown('---')
    st.subheader('Ask a follow-up question')
    follow_q = st.text_input('Ask a question (context: analyzed document)')
    if st.button('Ask'):
        if follow_q.strip():
            context = ' '.join(chunk_text(extracted_text)[:MAX_SUMMARY_CHUNKS])
            follow_prompt = (
                "You are a civic assistant. Use the following document context to answer the question concisely:\n\n"
                f"Context:\n{context}\n\nUser question: {follow_q}\nAnswer in simple language suitable for citizens."
            )
            with st.spinner("Getting answer..."):
                follow_raw = safe_call_llm(follow_prompt)
            st.write(follow_raw or "No answer returned from the AI.")
        else:
            st.warning("Please type a question.")

    # Generate Report
    st.markdown('---')
    if st.button('Generate & Download PDF Report'):
        with st.spinner('Creating PDF report...'):
            report_bytes = build_pdf_report(fname, analysis)
            if report_bytes is None:
                st.error('PDF generator (reportlab) is not available on this environment. Please install it.')
            else:
                st.success('Report ready — click download below')
                st.download_button('Download Report (PDF)', data=report_bytes, file_name=f"analysis_{uuid.uuid4().hex}.pdf", mime='application/pdf')

# If run directly, provide a simple streamlit page for testing
if __name__ == '__main__':
    st.set_page_config(page_title='Document Analyzer (Standalone)')
    run_document_analyzer_tab()
