# llm_client.py
import os
import logging
from typing import Optional

logger = logging.getLogger("llm_client")
logger.setLevel(logging.INFO)

# lazy import of SDK to avoid failing at import-time on environments without it
_genai = None
_llm_instance = None
_initialized = False

def init_llm(api_key: "Optional[str]"):
    """
    Initialize the Google Generative AI client. Call once at app startup.
    Accepts api_key from env or st.secrets.
    """
    global _genai, _llm_instance, _initialized
    if _initialized:
        return
    if not api_key:
        logger.warning("No API key provided to init_llm.")
        _initialized = True
        return
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        # create a model handle; adjust model name as required
        _genai = genai
        try:
            _llm_instance = genai.GenerativeModel("gemini-2.5-flash")
        except Exception:
            # If GenerativeModel not available, keep _genai and call genai.generate_text as fallback
            _llm_instance = None
        _initialized = True
        logger.info("LLM initialized successfully.")
    except Exception as e:
        logger.exception("Failed to initialize LLM: %s", e)
        _initialized = True

def call_llm(prompt: str, temperature: float = 0.2, max_output_tokens: int = 512) -> str:
    """
    Call the LLM and return a plain text response.
    Always return a string (may be empty on failure).
    """
    global _genai, _llm_instance
    if _genai is None and _llm_instance is None:
        logger.warning("LLM not configured. Returning empty string.")
        return ""
    try:
        # Prefer _llm_instance if available
        if _llm_instance is not None:
            # Use generate_content (older code path) — keep generation_config for compatibility
            response = _llm_instance.generate_content(prompt, generation_config={"temperature": temperature, "max_output_tokens": max_output_tokens})
            # some SDKs return object with .text or .candidates[0].content
            if hasattr(response, "text"):
                return response.text or ""
            if hasattr(response, "candidates") and len(response.candidates) > 0:
                c = response.candidates[0]
                if isinstance(c, dict) and "content" in c:
                    return c["content"]
                if hasattr(c, "content"):
                    return c.content
            # fallback to string conversion
            return str(response)
        else:
            # Fallback to genai.generate_text or genai.generate if available
            if hasattr(_genai, "generate_text"):
                resp = _genai.generate_text(model="gemini-2.5-flash", prompt=prompt, temperature=temperature, max_output_tokens=max_output_tokens)
                # many SDKs return { "candidates":[{"text":"..."}] } or similar
                if isinstance(resp, dict):
                    if resp.get("candidates"):
                        return resp["candidates"][0].get("text", "")
                # fallback
                return str(resp)
            else:
                # Last-resort: call a generic API method if exists
                return str(_genai)
    except Exception as e:
        logger.exception("LLM call failed: %s", e)
        return ""
