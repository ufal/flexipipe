"""Utilities for safe file reading with encoding detection."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

try:
    import chardet
    HAS_CHARDET = True
except ImportError:
    HAS_CHARDET = False


def detect_file_encoding(file_path: Path, sample_size: int = 8192) -> Optional[str]:
    """
    Detect the encoding of a file.
    
    Args:
        file_path: Path to the file
        sample_size: Number of bytes to sample for detection
        
    Returns:
        Detected encoding name (e.g., 'utf-8', 'utf-16', 'latin-1') or None if detection fails
    """
    try:
        with open(file_path, "rb") as f:
            sample = f.read(sample_size)
        
        if not sample:
            return "utf-8"  # Default for empty files
        
        # Check for BOM (Byte Order Mark)
        if sample.startswith(b"\xff\xfe"):
            return "utf-16-le"  # UTF-16 Little Endian
        elif sample.startswith(b"\xfe\xff"):
            return "utf-16-be"  # UTF-16 Big Endian
        elif sample.startswith(b"\xef\xbb\xbf"):
            return "utf-8-sig"  # UTF-8 with BOM
        
        # Use chardet for detection if available
        if HAS_CHARDET:
            try:
                result = chardet.detect(sample)
                if result and result.get("encoding"):
                    confidence = result.get("confidence", 0)
                    encoding = result["encoding"].lower()
                    
                    # Only trust high-confidence detections
                    if confidence > 0.7:
                        # Normalize common encodings
                        if encoding in ("utf-8", "utf8"):
                            return "utf-8"
                        elif encoding in ("utf-16", "utf16"):
                            # Default to LE for UTF-16 without BOM
                            return "utf-16-le"
                        elif encoding in ("latin-1", "iso-8859-1", "iso8859-1"):
                            return "latin-1"
                        elif encoding in ("cp1252", "windows-1252"):
                            return "cp1252"
                        else:
                            return encoding
            except Exception:
                # chardet failed, continue with defaults
                pass
        
        # Default fallback
        return "utf-8"
    except Exception:
        return "utf-8"  # Safe default


def read_text_file(
    file_path: Path,
    *,
    encoding: Optional[str] = None,
    errors: str = "replace",
    fallback_encodings: Optional[list[str]] = None,
    return_encoding: bool = False,
) -> Union[str, tuple[str, str]]:
    """
    Read a text file with automatic encoding detection and graceful error handling.
    
    Args:
        file_path: Path to the file to read
        encoding: Explicit encoding to use (if None, will auto-detect)
        errors: How to handle encoding errors ('strict', 'ignore', 'replace', 'surrogateescape')
        fallback_encodings: List of encodings to try if the primary encoding fails
        
    return_encoding: If True, return a tuple of (content, encoding) instead of just content
        
    Returns:
        File contents as a string, or tuple of (content, encoding) if return_encoding=True
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        OSError: If the file cannot be read (after all fallbacks fail)
    """
    if fallback_encodings is None:
        fallback_encodings = ["utf-8", "latin-1", "cp1252", "utf-16-le", "utf-16-be"]
    
    # Determine encoding to try
    encodings_to_try = []
    if encoding:
        encodings_to_try.append(encoding)
    else:
        # Auto-detect
        detected = detect_file_encoding(file_path)
        if detected:
            encodings_to_try.append(detected)
        # Add fallbacks
        encodings_to_try.extend(fallback_encodings)
    
    # Remove duplicates while preserving order
    seen = set()
    encodings_to_try = [enc for enc in encodings_to_try if enc not in seen and not seen.add(enc)]
    
    # Try each encoding
    last_error = None
    for enc in encodings_to_try:
        try:
            with open(file_path, "r", encoding=enc, errors=errors) as f:
                content = f.read()
                if return_encoding:
                    return content, enc
                return content
        except UnicodeDecodeError as e:
            last_error = e
            continue
        except Exception as e:
            # For non-encoding errors, re-raise immediately
            raise
    
    # If we get here, all encodings failed
    if last_error:
        raise UnicodeDecodeError(
            last_error.encoding,
            last_error.object,
            last_error.start,
            last_error.end,
            f"Could not decode file '{file_path}' with any of the attempted encodings: {', '.join(encodings_to_try)}. "
            f"Last error: {last_error.reason}"
        )
    else:
        raise OSError(f"Could not read file '{file_path}'")


def open_text_file(
    file_path: Path,
    mode: str = "r",
    *,
    encoding: Optional[str] = None,
    errors: str = "replace",
    **kwargs,
):
    """
    Open a text file with automatic encoding detection.
    
    This is a context manager that returns a file handle with the correct encoding.
    For reading, it will auto-detect encoding. For writing, it defaults to UTF-8.
    
    Args:
        file_path: Path to the file
        mode: File mode ('r', 'w', 'a', etc.)
        encoding: Explicit encoding (if None, will auto-detect for 'r', default to 'utf-8' for 'w')
        errors: How to handle encoding errors
        **kwargs: Additional arguments to pass to open()
        
    Returns:
        File handle (context manager)
    """
    if "r" in mode or "a" in mode:
        # Reading or appending - detect encoding if not specified
        if encoding is None:
            encoding = detect_file_encoding(file_path) or "utf-8"
    else:
        # Writing - default to UTF-8
        if encoding is None:
            encoding = "utf-8"
    
    return open(file_path, mode, encoding=encoding, errors=errors, **kwargs)

