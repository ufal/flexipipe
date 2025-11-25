"""
CText REST Backend for South African Languages.

The CText service provides multiple technologies per language:
- Tokeniser
- Sentence Separator  
- Language identifier
- Part of speech (XPOS)
- Named entity recogniser (NER)
- Phrase chunker
- Optical character recognition
- Universal part of speech (UPOS) - available but not currently used
- Lemmatiser - available but not currently used

Service endpoints:
- Languages: https://v-ctx-lnx10.nwu.ac.za:8443/CTexTWebAPI/services/languages
- Technologies per language: https://v-ctx-lnx10.nwu.ac.za:8443/CTexTWebAPI/services/coretechs?lang=<Language>

TODO: 
- Investigate API response format for cores that provide UPOS/lemma/NER
- Use native UPOS from service instead of mapping XPOS->UPOS
- Use native lemma from service instead of using form as lemma
- Add support for NER if available in response
"""

from __future__ import annotations

import re
import time
import ssl
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..backend_spec import BackendSpec
from ..conllu import conllu_to_document, document_to_conllu
from ..doc import Document, Sentence, Token
from ..neural_backend import BackendManager, NeuralResult
from ..language_utils import build_model_entry

try:
    import requests
    from requests.auth import HTTPBasicAuth
    from requests.adapters import HTTPAdapter
    from urllib3.util.ssl_ import create_urllib3_context
except ImportError as exc:  # pragma: no cover - import error handling
    raise ImportError(
        "CText REST backend requires the 'requests' package. "
        "Install it with: pip install requests"
    ) from exc


def _document_to_plain_text(document: Document) -> str:
    """Reconstruct plain text from a Document (preserving sentence texts when available)."""
    sentences = []
    for sent in document.sentences:
        if sent.text:
            sentences.append(sent.text.strip())
            continue
        parts: list[str] = []
        for tok in sent.tokens:
            parts.append(tok.form)
            if tok.space_after is not False:
                parts.append(" ")
        sentences.append("".join(parts).strip())
    return "\n".join(filter(None, sentences)) or document.id or ""


def _load_xpos_mapping(mapping_file: Optional[Path] = None) -> tuple[Dict[str, str], Dict[str, str]]:
    """
    Load XPOS to UPOS and FEATS mapping from TSV file.
    
    NOTE: This mapping is a workaround. The CText service now provides native UPOS
    via different API cores. We should use the native UPOS instead of mapping.
    See module docstring for TODO details.
    
    Returns:
        Tuple of (xpos_to_upos, xpos_to_feats) dictionaries
    """
    xpos_to_upos: Dict[str, str] = {}
    xpos_to_feats: Dict[str, str] = {}
    
    if mapping_file is None:
        # Try to find the mapping file in common locations
        possible_paths = [
            Path(__file__).parent.parent.parent / "Resources" / "CText-UD.tsv",
            Path(__file__).parent.parent / "Resources" / "CText-UD.tsv",
        ]
        for path in possible_paths:
            if path.exists():
                mapping_file = path
                break
    
    if mapping_file and mapping_file.exists():
        import csv
        with open(mapping_file, 'r', encoding='utf-8') as f:
            tsv_reader = csv.reader(f, delimiter='\t')
            for line in tsv_reader:
                if len(line) > 1:
                    xpos = line[0]
                    upos = line[1]
                    xpos_to_upos[xpos] = upos
                    if len(line) > 2:
                        feats = line[2]
                        xpos_to_feats[xpos] = feats
    
    return xpos_to_upos, xpos_to_feats


class CTextRESTBackend(BackendManager):
    """Neural backend that delegates tagging to a CText REST service."""

    def __init__(
        self,
        endpoint_url: str,
        *,
        language: str,
        auth_token: Optional[str] = None,
        auth_header: Optional[str] = None,
        timeout: float = 30.0,
        batch_size: int = 50,
        mapping_file: Optional[Path] = None,
        session: Optional[requests.Session] = None,
        log_requests: bool = False,
        verify_ssl: bool = False,  # Default to False due to SSL certificate issues
    ):
        if not endpoint_url:
            raise ValueError("endpoint_url is required for CText REST backend")
        self.endpoint_url = endpoint_url
        if not language:
            raise ValueError("CText REST backend requires a language code. Provide --ctext-language.")
        self.language = language
        self.auth_token = auth_token or "SGllcmRpZSBzYWwgd2Vyaw=="  # Default from run_ctext.py
        self.auth_header = auth_header or "Basic"
        self.timeout = timeout
        self.batch_size = max(1, int(batch_size))
        self.verify_ssl = verify_ssl
        self.session = session or self._build_session()
        self.log_requests = log_requests
        self._model_name = f"ctext-{language}"
        self._backend_name = "ctext"
        
        # Load XPOS mapping
        self.xpos_to_upos, self.xpos_to_feats = _load_xpos_mapping(mapping_file)
        
        # Cache for authentication token
        self._cached_token: Optional[str] = None
        self._token_expiry: Optional[float] = None

    @property
    def _backend_info(self) -> str:
        """Used by CLI to describe this backend."""
        return f"ctext: {self.language}"

    def supports_training(self) -> bool:  # pragma: no cover - trivial
        return False
    
    def train(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - not supported
        """Training is not supported for CText REST backend."""
        raise NotImplementedError("CText REST backend does not support training")

    def _build_session(self) -> requests.Session:
        # If SSL verification is disabled, use a simple session without custom adapter
        # This avoids SSL version negotiation issues
        if not self.verify_ssl:
            sess = requests.Session()
            # Suppress urllib3 SSL warnings when verification is disabled
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            return sess
        
        # Only use custom adapter if SSL verification is enabled
        class FlexibleTLSAdapter(HTTPAdapter):
            def init_poolmanager(self, connections, maxsize, block=False, **pool_kwargs):
                try:
                    ctx = create_urllib3_context()
                    # Don't restrict TLS versions too much - let the server negotiate
                    if hasattr(ctx, "minimum_version"):
                        try:
                            ctx.minimum_version = ssl.TLSVersion.MINIMUM_SUPPORTED
                        except (AttributeError, ValueError):
                            ctx.minimum_version = ssl.TLSVersion.TLSv1
                    if hasattr(ctx, "maximum_version"):
                        try:
                            ctx.maximum_version = ssl.TLSVersion.MAXIMUM_SUPPORTED
                        except (AttributeError, ValueError):
                            try:
                                ctx.maximum_version = ssl.TLSVersion.TLSv1_2
                            except (AttributeError, ValueError):
                                pass
                    
                    super().init_poolmanager(
                        connections,
                        maxsize,
                        block=block,
                        ssl_context=ctx,
                        **pool_kwargs,
                    )
                except Exception:
                    # If SSL context creation fails, fall back to default
                    pool_kwargs.pop('ssl_context', None)
                    super().init_poolmanager(
                        connections,
                        maxsize,
                        block=block,
                        **pool_kwargs,
                    )

            def proxy_manager_for(self, *args, **kwargs):
                try:
                    ctx = create_urllib3_context()
                    if hasattr(ctx, "minimum_version"):
                        try:
                            ctx.minimum_version = ssl.TLSVersion.MINIMUM_SUPPORTED
                        except (AttributeError, ValueError):
                            ctx.minimum_version = ssl.TLSVersion.TLSv1
                    if hasattr(ctx, "maximum_version"):
                        try:
                            ctx.maximum_version = ssl.TLSVersion.MAXIMUM_SUPPORTED
                        except (AttributeError, ValueError):
                            try:
                                ctx.maximum_version = ssl.TLSVersion.TLSv1_2
                            except (AttributeError, ValueError):
                                pass
                    kwargs["ssl_context"] = ctx
                except Exception:
                    pass
                return super().proxy_manager_for(*args, **kwargs)

        sess = requests.Session()
        sess.mount("https://", FlexibleTLSAdapter())
        return sess

    def _get_auth_token(self) -> str:
        """Get authentication token from CText service."""
        # Use cached token if still valid (cache for 1 hour)
        if self._cached_token and self._token_expiry and time.time() < self._token_expiry:
            return self._cached_token
        
        # Try multiple possible auth endpoints
        auth_urls = [
            "https://v-ctx-lnx10.nwu.ac.za:8443/CTexTWebAPI/services/setuser",
            "https://v-ctx-lnx7.nwu.ac.za:8443/CTexTWebAPI/services/setuser",
        ]
        
        auth_header_value = f"{self.auth_header} {self.auth_token}"
        
        if self.log_requests:
            print(f"[ctext] Getting auth token from CText service")
        
        last_exc = None
        for auth_url in auth_urls:
            try:
                if self.log_requests:
                    print(f"[ctext] Trying auth endpoint: {auth_url}")
                
                response = self.session.post(
                    auth_url,
                    headers={
                        "Authorization": auth_header_value,
                        "Connection": "keep-alive",
                        "Content-Type": "application/json",
                    },
                    json={},  # Send empty JSON body like the web interface
                    timeout=self.timeout,
                    verify=self.verify_ssl,  # Respect SSL verification setting
                )
                response.raise_for_status()
                data = response.json()
                
                # Try different response formats
                token = None
                if isinstance(data, dict):
                    # Try 'token' field (array or direct)
                    token_data = data.get('token')
                    if isinstance(token_data, list) and token_data:
                        token = token_data[0]
                    elif isinstance(token_data, str):
                        token = token_data
                    # Try 'authToken' field
                    if not token:
                        token = data.get('authToken')
                
                if not token:
                    if self.log_requests:
                        print(f"[ctext] Unexpected auth response format: {data}")
                    continue  # Try next URL
                
                # Cache token for 1 hour
                self._cached_token = token
                self._token_expiry = time.time() + 3600
                
                if self.log_requests:
                    print(f"[ctext] Auth token obtained from {auth_url} (cached for 1 hour)")
                
                return token
            except requests.RequestException as exc:
                last_exc = exc
                if self.log_requests:
                    print(f"[ctext] Auth endpoint {auth_url} failed: {exc}")
                continue  # Try next URL
        
        # If all URLs failed, raise the last exception
        raise RuntimeError(f"CText authentication failed (tried {len(auth_urls)} endpoints): {last_exc}") from last_exc

    def tag(
        self,
        document: Document,
        *,
        overrides: Optional[Dict[str, object]] = None,
        preserve_pos_tags: bool = False,
        components: Optional[list[str]] = None,
        use_raw_text: bool = False,
    ) -> NeuralResult:
        start = time.time()

        batches = self._split_document(document)
        aggregated_doc = Document(id=document.id, meta=dict(document.meta))
        total_tokens = 0

        # Get auth token once
        token = self._get_auth_token()

        for batch_index, batch_doc in enumerate(batches):
            text = _document_to_plain_text(batch_doc) if use_raw_text else _document_to_plain_text(batch_doc)
            
            if self.log_requests:
                print(f"[ctext] GET {self.endpoint_url} (batch {batch_index + 1}/{len(batches)})")
                if batch_index == 0:
                    print(f"[ctext] Language: {self.language}")
                    print(f"[ctext] Text preview: {text[:200]}...")

            request_start = time.time()
            response = self._get_tagged_text(text, token)
            request_elapsed = time.time() - request_start

            if self.log_requests:
                print(f"[ctext] Response status: {response.status_code}")
                print(f"[ctext] Request duration: {request_elapsed:.2f}s")

            chunk_doc = self._parse_response(response, batch_doc)
            if self.log_requests and batch_index == 0 and chunk_doc.sentences:
                first_sentence = chunk_doc.sentences[0]
                token_forms = " ".join(tok.form for tok in first_sentence.tokens)
                print("[ctext] First sentence returned:")
                print(f"  text={first_sentence.text or token_forms}")
                print(f"  tokens={token_forms}")
            
            aggregated_doc.sentences.extend(chunk_doc.sentences)
            total_tokens += sum(len(sent.tokens) for sent in chunk_doc.sentences)

        elapsed = time.time() - start
        stats = {
            "elapsed_seconds": elapsed,
            "backend": "ctext-rest",
            "token_count": total_tokens,
        }
        return NeuralResult(document=aggregated_doc, stats=stats)

    def _split_document(self, document: Document) -> list[Document]:
        """Split document sentences into batches to avoid oversized payloads."""
        if not document.sentences:
            return [document]
        batches: list[Document] = []
        for start_idx in range(0, len(document.sentences), self.batch_size):
            batch = Document(
                id=f"{document.id or 'doc'}-batch-{len(batches)+1}",
                meta=dict(document.meta),
                sentences=[*document.sentences[start_idx:start_idx + self.batch_size]],
            )
            batches.append(batch)
        return batches

    def _get_tagged_text(self, text: str, token: str) -> requests.Response:
        """
        Make GET request to CText tagging service.
        
        NOTE: Currently uses core=pos which only returns XPOS tags.
        The CText service now also provides:
        - Universal part of speech (UPOS) - available via different core or in response
        - Lemmatiser - available via different core or in response  
        - Named entity recogniser (NER) - available via different core
        
        TODO: Investigate API response format when using cores that provide UPOS/lemma/NER.
        The service lists available technologies at:
        https://v-ctx-lnx10.nwu.ac.za:8443/CTexTWebAPI/services/coretechs?lang=<Language>
        
        Using native UPOS/lemma from the service would be much better than mapping XPOS->UPOS
        and using form as lemma.
        """
        import urllib.parse
        encoded_text = urllib.parse.quote(text)
        
        # Try multiple endpoints - use the one that worked for auth, plus fallbacks
        # Extract base URL from endpoint_url, but also try alternatives
        base_urls = []
        if self.endpoint_url:
            base_urls.append(self.endpoint_url)
        # Add fallback endpoints
        if "v-ctx-lnx10" not in self.endpoint_url:
            base_urls.append("https://v-ctx-lnx10.nwu.ac.za:8443/CTexTWebAPI/services")
        if "v-ctx-lnx7" not in self.endpoint_url:
            base_urls.append("https://v-ctx-lnx7.nwu.ac.za:8443/CTexTWebAPI/services")
        
        # Try both header name variations (Authtoken and authToken)
        # The web interface uses authToken, but the API might accept both
        headers_to_try = [
            {"authToken": token, "Connection": "keep-alive"},
            {"Authtoken": token, "Connection": "keep-alive"},
            {"Authorization": f"Bearer {token}", "Connection": "keep-alive"},
        ]
        
        last_exc = None
        for base_url in base_urls:
            url = f"{base_url}?core=pos&lang={self.language}&text={encoded_text}"
            
            for headers in headers_to_try:
                try:
                    if self.log_requests:
                        if base_url != base_urls[0] or headers != headers_to_try[0]:
                            print(f"[ctext] Trying endpoint: {base_url} with header: {list(headers.keys())[0]}")
                    
                    response = self.session.get(
                        url,
                        headers=headers,
                        timeout=self.timeout,
                        verify=self.verify_ssl,  # Respect SSL verification setting
                    )
                    response.raise_for_status()
                    if self.log_requests and (base_url != base_urls[0] or headers != headers_to_try[0]):
                        print(f"[ctext] Success with endpoint: {base_url} and header: {list(headers.keys())[0]}")
                    return response
                except requests.HTTPError as exc:
                    last_exc = exc
                    # If it's a 401/403, try next header format or endpoint
                    if exc.response is not None and exc.response.status_code in (401, 403):
                        if self.log_requests:
                            print(f"[ctext] Auth error with {base_url} and header {list(headers.keys())[0]}, trying next...")
                        continue
                    # For other HTTP errors, try next endpoint/header
                    if self.log_requests:
                        print(f"[ctext] HTTP {exc.response.status_code} with {base_url} and header {list(headers.keys())[0]}, trying next...")
                    continue
                except requests.RequestException as exc:
                    last_exc = exc
                    # For network/SSL errors, try next endpoint/header
                    if self.log_requests:
                        error_type = type(exc).__name__
                        print(f"[ctext] {error_type} with {base_url} and header {list(headers.keys())[0]}, trying next...")
                    continue
        
        # If all combinations failed, raise the last exception
        if last_exc:
            if isinstance(last_exc, requests.HTTPError):
                error_msg = f"CText REST returned HTTP {last_exc.response.status_code}"
                if last_exc.response is not None:
                    try:
                        error_data = last_exc.response.json()
                        error_msg += f": {error_data}"
                    except ValueError:
                        error_msg += f": {last_exc.response.text[:500]}"
                raise RuntimeError(error_msg) from last_exc
            else:
                raise RuntimeError(f"CText REST request failed (tried {len(base_urls)} endpoints with {len(headers_to_try)} header formats each): {last_exc}") from last_exc
        else:
            raise RuntimeError("CText REST request failed: unknown error")

    def _parse_response(self, response: requests.Response, original_doc: Document) -> Document:
        """Parse CText JSON response into a Document."""
        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError(f"CText REST returned invalid JSON: {exc}") from exc
        
        if 'PoS Tagger-streams' not in data:
            raise RuntimeError(f"CText REST response missing 'PoS Tagger-streams' field: {response.text[:500]}")
        
        poslist = data['PoS Tagger-streams']
        
        doc = Document(id=original_doc.id or "ctext", meta=dict(original_doc.meta))
        sentences: List[Sentence] = []
        current_sentence = Sentence(id="", text="", tokens=[])
        current_text = ""
        last_form = ""
        ord_num = 0
        
        # Track remaining text to determine spacing
        remaining_text = _document_to_plain_text(original_doc) if original_doc.sentences else ""
        
        for line in poslist:
            if line == '':
                continue
            
            if line == '\n':
                # End of sentence
                if current_sentence.tokens:
                    current_sentence.text = current_text.strip()
                    sentences.append(current_sentence)
                current_sentence = Sentence(id="", text="", tokens=[])
                current_text = ""
                last_form = ""
                ord_num = 0
                continue
            
            # Parse tab-separated line: form\txpos
            parts = line.split('\t')
            if not parts:
                continue
            
            form = parts[0]
            xpos = parts[1] if len(parts) > 1 else "_"
            
            # Determine space_after based on original text
            space_after: Optional[bool] = None
            if last_form:
                # Check if remaining text starts with form (no space) or has space before form
                remaining_text = remaining_text.lstrip()
                if remaining_text.startswith(form):
                    space_after = False
                else:
                    space_after = True
                last_form = ""
                remaining_text = remaining_text.lstrip(form)
            else:
                if ord_num > 0:
                    # Check spacing from remaining text
                    remaining_text = remaining_text.lstrip()
                    if remaining_text.startswith(form):
                        space_after = False
                    else:
                        space_after = True
                    remaining_text = remaining_text.lstrip(form)
                else:
                    # First token - check if it starts the remaining text
                    remaining_text = remaining_text.lstrip()
                    if remaining_text.startswith(form):
                        remaining_text = remaining_text.lstrip(form)
                    space_after = None  # First token, don't set space_after
            
            # Map XPOS to UPOS and FEATS
            # TODO: The CText service now provides native UPOS and lemma via different cores.
            # We should use those instead of mapping XPOS->UPOS and using form as lemma.
            # See _get_tagged_text() docstring for more details.
            upos = self.xpos_to_upos.get(xpos, "_")
            feats = self.xpos_to_feats.get(xpos, "_")
            
            # Try without trailing numbers if not found
            if upos == "_" and xpos != "_":
                xpos_base = re.sub(r'\d+$', '', xpos)
                upos = self.xpos_to_upos.get(xpos_base, "_")
                feats = self.xpos_to_feats.get(xpos_base, feats)
            
            token = Token(
                id=ord_num + 1,
                form=form,
                lemma=form,  # TODO: CText service now provides lemmatiser - use native lemma instead
                upos=upos,  # TODO: CText service now provides UPOS - use native UPOS instead of mapping
                xpos=xpos,
                feats=feats,
                head=0,
                deprel="",
                space_after=space_after,
            )
            
            current_sentence.tokens.append(token)
            current_text += form
            if space_after:
                current_text += " "
            
            last_form = form
            ord_num += 1
        
        # Add last sentence if any
        if current_sentence.tokens:
            current_sentence.text = current_text.strip()
            # Last token should have space_after = None
            if current_sentence.tokens:
                current_sentence.tokens[-1].space_after = None
            sentences.append(current_sentence)
        
        doc.sentences = sentences
        return doc


CTEXT_LANGUAGE_SPECS = [
    ("afrikaans", "af", "Afrikaans"),
    ("english", "en", "English"),
    ("isindebele", "nr", "isiNdebele"),
    ("sesotho", "st", "Sesotho"),
    ("sesotho_sa_leboa", "nso", "Sesotho sa Leboa"),
    ("setswana", "tn", "Setswana"),
    ("siswati", "ss", "SiSwati"),
    ("venda", "ven", "Venda"),
    ("xhosa", "xh", "isiXhosa"),
    ("xitsonga", "tso", "Xitsonga"),
    ("zulu", "zu", "isiZulu"),
]


def get_ctext_model_entries(
    *,
    use_cache: bool = True,
    refresh_cache: bool = False,
) -> Dict[str, Dict[str, Any]]:
    entries: Dict[str, Dict[str, Any]] = {}
    for key, iso, name in CTEXT_LANGUAGE_SPECS:
        model_name = f"ctext-{iso}"
        entry = build_model_entry(
            "ctext",
            model_name,
            language_code=iso,
            language_name=name,
            description="CText REST service model",
        )
        entry["model_size_bytes"] = 0
        entries[key] = entry
    return entries


def list_ctext_models(*args: Any, **kwargs: Any) -> int:
    entries = get_ctext_model_entries()
    print("CText models:")
    print(f"{'Model':<15} {'ISO':<6} {'Language':<25}")
    print("-" * 50)
    for entry in entries.values():
        model = entry.get("model", "-")
        iso = entry.get("language_iso") or "-"
        language = entry.get("language_name") or "-"
        print(f"{model:<15} {iso:<6} {language:<25}")
    print("\nModels are hosted by the CText REST service; no local downloads are required.")
    return 0


def _create_ctext_backend(
    *,
    endpoint_url: str,
    language: str,
    auth_token: Optional[str] = None,
    auth_header: Optional[str] = None,
    timeout: float = 30.0,
    batch_size: int = 50,
    mapping_file: Optional[Path] = None,
    session: Optional[requests.Session] = None,
    log_requests: bool = False,
    verify_ssl: bool = False,
    **kwargs: Any,
) -> CTextRESTBackend:
    """Instantiate the CText REST backend."""
    from ..backend_utils import validate_backend_kwargs
    
    validate_backend_kwargs(kwargs, "CText", allowed_extra=["download_model", "training"])
    
    return CTextRESTBackend(
        endpoint_url=endpoint_url,
        language=language,
        auth_token=auth_token,
        auth_header=auth_header,
        timeout=timeout,
        batch_size=batch_size,
        mapping_file=mapping_file,
        session=session,
        log_requests=log_requests,
        verify_ssl=verify_ssl,
    )


BACKEND_SPEC = BackendSpec(
    name="ctext",
    description="CText - REST service for POS tagging (South African languages)",
    factory=_create_ctext_backend,
    get_model_entries=get_ctext_model_entries,
    list_models=list_ctext_models,
    supports_training=False,
    is_rest=True,
)

