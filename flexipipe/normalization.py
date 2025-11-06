"""
Normalization module for FlexiPipe.
"""
from typing import List, Dict, Optional
from pathlib import Path

def _derive_inflection_suffixes_from_vocab(vocab: Dict, max_suffix_len: int = 4, min_count: int = 3) -> List[str]:
    """Derive common inflection suffixes from vocab form->reg mappings in a language-agnostic way.

    We look for entries where both the surface form and its reg end with the same suffix
    and the stems differ (e.g., seruicio->servicio while preserving plural 's').
    We collect suffixes up to max_suffix_len and keep those observed at least min_count times.
    """
    def get_reg(entry):
        if isinstance(entry, list):
            if entry and isinstance(entry[0], dict):
                return entry[0].get('reg', None)
        elif isinstance(entry, dict):
            return entry.get('reg', None)
        return None

    counts: Dict[str, int] = {}
    for form, entry in vocab.items():
        if not isinstance(form, str):
            continue
        reg = get_reg(entry)
        if not reg or reg == '_' or reg.lower() == form.lower():
            continue
        f = form.lower()
        r = reg.lower()
        max_k = min(max_suffix_len, len(f), len(r))
        for k in range(1, max_k + 1):
            sfx = f[-k:]
            if r.endswith(sfx):
                if f[:-k] != r[:-k]:
                    counts[sfx] = counts.get(sfx, 0) + 1
    frequent = [s for s, c in counts.items() if c >= min_count]
    frequent.sort(key=lambda s: (-counts[s], -len(s), s))
    return frequent



def normalize_word(word: str, vocab: Dict, conservative: bool = True, similarity_threshold: float = 0.8,
                   inflection_suffixes: Optional[List[str]] = None) -> Optional[str]:
    """
    Normalize orthographic variant to standard form using vocabulary.
    
    Priority order (conservative mode):
    1. Explicit normalization mapping in vocabulary (reg field)
    2. Morphological variations of known mappings (e.g., "mysterio"->"misterio" allows "mysterios"->"misterios")
    3. Check if word is already normalized (exists in vocab without reg)
    4. Pattern-aware Levenshtein distance matching (only in non-conservative mode)
    
    This is especially important for historic documents where normalization depends on
    transcription standards, region, period, and register - the local vocabulary can
    provide domain-specific normalization mappings.
    
    Args:
        word: Word to normalize
        vocab: Vocabulary dictionary (can include reg field for explicit mappings)
        conservative: If True, only use explicit mappings and morphological variations (default: True)
        similarity_threshold: Similarity threshold for normalization (higher = more conservative)
    
    Returns:
        Normalized form if found, None otherwise
    """
    word_lower = word.lower()
    
    def get_reg_from_entry(entry):
        """Extract reg (normalization) from vocabulary entry (handles single dict or array)."""
        if isinstance(entry, list):
            # Array format: check first entry (most frequent)
            if entry and isinstance(entry[0], dict):
                return entry[0].get('reg', None)
        elif isinstance(entry, dict):
            return entry.get('reg', None)
        return None
    
    def word_exists_as_reg_in_vocab(word_to_check: str) -> bool:
        """Check if word appears as a 'reg' value anywhere in vocab (means it's already normalized)."""
        word_check_lower = word_to_check.lower()
        for entry in vocab.values():
            if isinstance(entry, list):
                for analysis in entry:
                    reg = analysis.get('reg')
                    if reg and reg.lower() == word_check_lower:
                        return True
            elif isinstance(entry, dict):
                reg = entry.get('reg')
                if reg and reg.lower() == word_check_lower:
                    return True
        return False
    
    # Step 0: Early check - if word appears as a 'reg' value, it's already normalized
    # This prevents normalizing words that are themselves normalized forms
    if word_exists_as_reg_in_vocab(word):
        return None  # Word is already a normalized form, don't normalize it
    
    # Step 1: Check for explicit normalization mapping in vocabulary
    # Try exact case first, then lowercase
    if word in vocab:
        reg = get_reg_from_entry(vocab[word])
        if reg and reg != '_' and reg != word:
            return reg
    
    if word_lower in vocab:
        reg = get_reg_from_entry(vocab[word_lower])
        if reg and reg != '_' and reg.lower() != word_lower:
            return reg
    
    # Step 2: Check morphological variations of known mappings
    # If "mysterio" -> "misterio" is in vocab, also normalize "mysterios" -> "misterios"
    # This is safe because it's based on explicit mappings
    if conservative:
        # Use provided suffixes, otherwise derive from vocab
        suffixes_to_try = inflection_suffixes or _derive_inflection_suffixes_from_vocab(vocab)
        
        # Try removing suffixes to find base form
        for suffix in suffixes_to_try:
            if len(word_lower) > len(suffix) + 2 and word_lower.endswith(suffix):
                base_form = word_lower[:-len(suffix)]
                # Check if base form has a normalization mapping
                if base_form in vocab:
                    reg = get_reg_from_entry(vocab[base_form])
                    if reg and reg != '_' and reg.lower() != base_form:
                        # Apply same suffix to normalized form
                        normalized = reg.lower() + suffix
                        # CRITICAL: Verify the normalized form exists in vocab (as key or reg value)
                        # Don't normalize if result doesn't exist in vocab - prevents incorrect normalizations
                        if normalized in vocab or word_exists_as_reg_in_vocab(normalized):
                            return normalized
                
                # Also try with original case
                if word and len(word) > len(suffix):
                    base_form_orig = word[:-len(suffix)]
                    if base_form_orig in vocab:
                        reg = get_reg_from_entry(vocab[base_form_orig])
                        if reg and reg != '_' and reg != base_form_orig:
                            normalized = reg + suffix
                            # CRITICAL: Verify the normalized form exists in vocab (as key or reg value)
                            normalized_lower = normalized.lower()
                            if normalized in vocab or normalized_lower in vocab or word_exists_as_reg_in_vocab(normalized):
                                return normalized
    
    # Step 3: Check if word is already normalized (exists in vocab without reg)
    # If word exists in vocab and has no reg, it's already the normalized form
    if word in vocab or word_lower in vocab:
        entry = vocab.get(word) or vocab.get(word_lower)
        reg = get_reg_from_entry(entry)
        if not reg or reg == '_':
            return None  # Already normalized/standard form
    
    # Step 4: Pattern-aware similarity matching (only in non-conservative mode)
    if not conservative:
        # Use Levenshtein distance with pattern-aware substitutions
        normalized = find_pattern_aware_normalization(word, vocab, threshold=similarity_threshold)
        if normalized:
            return normalized
    
    # Conservative mode: don't normalize if no explicit mapping found
    return None



def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]



def find_pattern_aware_normalization(word: str, vocab: Dict, threshold: float = 0.8) -> Optional[str]:
    """
    Find normalization using pattern-aware approach with frequency-based rules.
    
    IMPORTANT: Only applies normalization patterns that are:
    1. Used frequently (at least 3 times in vocab)
    2. Result in a normalized form that exists in vocab (as key or reg value)
    
    Extracts normalization patterns from vocab (form -> reg mappings) and only
    applies frequent patterns. This prevents rare transformations like "conocido -> conociendo"
    from being over-applied.
    
    Args:
        word: Word to normalize
        vocab: Vocabulary dictionary (with reg fields for normalization mappings)
        threshold: Minimum similarity threshold (0.0-1.0) - not used for pattern matching
    
    Returns:
        Normalized form if found, None otherwise
    """
    word_lower = word.lower()
    
    # Step 1: Extract normalization patterns from vocab (form -> reg mappings)
    # Count how often each transformation pattern is used
    normalization_patterns = {}  # (char_from, char_to, position) -> count
    pattern_to_reg = {}  # (char_from, char_to, position) -> list of (form, reg) examples
    
    for vocab_form, vocab_entry in vocab.items():
        # Extract reg from entry
        reg = None
        if isinstance(vocab_entry, list):
            if vocab_entry and isinstance(vocab_entry[0], dict):
                reg = vocab_entry[0].get('reg')
        elif isinstance(vocab_entry, dict):
            reg = vocab_entry.get('reg')
        
        if reg and reg != '_' and reg.lower() != vocab_form.lower():
            # We have a normalization mapping: vocab_form -> reg
            form_lower = vocab_form.lower()
            reg_lower = reg.lower()
            
            # Find character differences (substitution patterns)
            # Try to identify which characters changed
            if len(form_lower) == len(reg_lower):
                # Same length: character substitution
                for i, (c1, c2) in enumerate(zip(form_lower, reg_lower)):
                    if c1 != c2:
                        pattern_key = (c1, c2, 'subst')
                        normalization_patterns[pattern_key] = normalization_patterns.get(pattern_key, 0) + 1
                        if pattern_key not in pattern_to_reg:
                            pattern_to_reg[pattern_key] = []
                        pattern_to_reg[pattern_key].append((vocab_form, reg))
    
    # Filter patterns: only keep those used at least 3 times (frequent patterns)
    frequent_patterns = {k: v for k, v in normalization_patterns.items() if v >= 3}
    
    if not frequent_patterns:
        # No frequent patterns found - fall back to simple checks
        return None
    
    # Step 2: Try to apply frequent patterns to the word
    # Check if applying any frequent pattern results in a word that exists in vocab
    candidates = []
    
    for pattern_key, pattern_count in frequent_patterns.items():
        char_from, char_to, pattern_type = pattern_key
        if pattern_type == 'subst' and char_from in word_lower:
            # Try applying this substitution pattern
            normalized_candidate = word_lower.replace(char_from, char_to)
            
            # CRITICAL: Only proceed if normalized form exists in vocab (as key or reg value)
            exists_in_vocab = False
            if normalized_candidate in vocab:
                exists_in_vocab = True
            else:
                # Check if it exists as a reg value in any vocab entry
                for vocab_entry in vocab.values():
                    reg = None
                    if isinstance(vocab_entry, list):
                        for analysis in vocab_entry:
                            reg = analysis.get('reg')
                            if reg and reg.lower() == normalized_candidate:
                                exists_in_vocab = True
                                break
                    elif isinstance(vocab_entry, dict):
                        reg = vocab_entry.get('reg')
                        if reg and reg.lower() == normalized_candidate:
                            exists_in_vocab = True
                            break
                    if exists_in_vocab:
                        break
            
            if exists_in_vocab:
                # This pattern is frequent AND the normalized form exists - use it
                candidates.append((normalized_candidate, pattern_count))
    
    if not candidates:
        return None
    
    # Sort by pattern frequency (most frequent first)
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]



