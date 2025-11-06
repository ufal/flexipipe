"""
Contractions module for FlexiPipe.
"""
from typing import List, Dict, Optional
from pathlib import Path

def split_contraction(form: str, vocab: Dict, aggressive: bool = False, language: Optional[str] = None) -> Optional[List[str]]:
    """
    Split contraction into component words (e.g., "destas" -> ["de", "estas"]).
    
    Uses vocabulary to identify potential contractions and split them.
    Handles both modern languages (with rules) and historic texts.
    
    Args:
        form: Word form that might be a contraction
        vocab: Vocabulary dictionary
        aggressive: If True, use more aggressive splitting patterns for historic texts
        language: Language code (e.g., 'es', 'pt', 'ltz') for language-specific rules
    
    Returns:
        List of split words if contraction detected, None otherwise
    """
    form_lower = form.lower()
    
    # Check if form is already in vocabulary as a single word
    # If it exists as a single word, it's ambiguous - prefer keeping as single word
    # unless we have strong evidence it's a contraction
    exists_as_single_word = form in vocab or form_lower in vocab
    
    # Language-specific patterns (modern languages)
    if language:
        split_result = _split_contraction_language_specific(form, form_lower, vocab, language, exists_as_single_word, aggressive)
        if split_result:
            return split_result
    
    # Common contraction patterns (language-agnostic)
    # These are patterns that often indicate contractions
    contraction_patterns = []
    
    if aggressive:
        # More aggressive patterns for historic texts
        # Spanish: destas, dellos, etc.
        contraction_patterns.extend([
            (r'^d([aeiou])', ['de', r'\1']),  # de + vowel
            (r'^([aeiou])l([aeiou])', [r'\1', 'el', r'\2']),  # vowel + el + vowel (aggressive)
            (r'^([aeiou])n([aeiou])', [r'\1', 'en', r'\2']),  # vowel + en + vowel (aggressive)
        ])
    
    # Standard patterns (more conservative)
    contraction_patterns.extend([
        # Portuguese/Spanish: d'água, faze-lo, etc.
        (r"^([a-z]+)-([a-z]+)$", None),  # hyphenated words (check if parts exist)
        (r"^([a-z]+)'([a-z]+)$", None),   # apostrophe contractions (check if parts exist)
        # Check for common prefixes that might be contractions
        (r'^([a-z]{1,2})([a-z]{3,})$', None),  # Short prefix + longer word
    ])
    
    # Try to split based on patterns
    for pattern, replacement in contraction_patterns:
        if replacement is None:
            # Pattern-based splitting: check if parts exist in vocabulary
            if '-' in form:
                parts = form.split('-')
                if len(parts) == 2:
                    part1, part2 = parts
                    # Check if both parts exist in vocab (or are common words)
                    if (part1.lower() in vocab or len(part1) <= 2) and \
                       (part2.lower() in vocab or len(part2) <= 2):
                        return [part1, part2]
            
            if "'" in form:
                parts = form.split("'")
                if len(parts) == 2:
                    part1, part2 = parts
                    if (part1.lower() in vocab or len(part1) <= 2) and \
                       (part2.lower() in vocab or len(part2) <= 2):
                        return [part1, part2]
            
            # Try splitting at common boundaries
            # Common prefixes: de, a, en, con, etc.
            common_prefixes = ['de', 'a', 'en', 'con', 'por', 'para', 'del', 'al', 'da', 'do']
            for prefix in common_prefixes:
                if form_lower.startswith(prefix) and len(form_lower) > len(prefix) + 2:
                    remainder = form_lower[len(prefix):]
                    # If ambiguous, only split if remainder clearly exists in vocab
                    if exists_as_single_word:
                        if remainder in vocab:
                            return [prefix, remainder]
                        # Otherwise, keep as single word (ambiguous)
                        continue
                    else:
                        if remainder in vocab or len(remainder) >= 3:
                            return [prefix, remainder]
        else:
            # Direct replacement pattern
            match = re.match(pattern, form_lower)
            if match:
                # Build the split based on replacement pattern
                split_words = []
                for repl in replacement:
                    if repl.startswith('\\'):
                        # Backreference
                        group_num = int(repl[1:])
                        if group_num <= len(match.groups()):
                            split_words.append(match.group(group_num))
                    else:
                        split_words.append(repl)
                if len(split_words) > 1:
                    return split_words
    
    # If no pattern matched, try vocabulary-based splitting
    # Look for words that could be the start of this form
    # This is more expensive but useful for historic texts
    if aggressive:
        for vocab_word in vocab.keys():
            if len(vocab_word) >= 2 and form_lower.startswith(vocab_word) and len(form_lower) > len(vocab_word) + 2:
                remainder = form_lower[len(vocab_word):]
                # If ambiguous, only split if remainder clearly exists in vocab
                if exists_as_single_word:
                    if remainder in vocab:
                        return [vocab_word, remainder]
                    # Otherwise, keep as single word (ambiguous)
                    continue
                else:
                    if remainder in vocab or len(remainder) >= 3:
                        return [vocab_word, remainder]
    
    return None



def _split_contraction_language_specific(form: str, form_lower: str, vocab: Dict, language: str, exists_as_single_word: bool, aggressive: bool = False) -> Optional[List[str]]:
    """
    Language-specific contraction splitting rules.
    
    Args:
        form: Original form
        form_lower: Lowercase form
        vocab: Vocabulary dictionary
        language: Language code ('es', 'pt', 'ltz', etc.)
        exists_as_single_word: Whether the form exists as a single word in vocab
    
    Returns:
        List of split words if contraction detected, None otherwise
    """
    # Luxembourgish: d'XXX is always de + XXX
    if language == 'ltz' or language == 'lb':
        if form_lower.startswith("d'") and len(form_lower) > 2:
            remainder = form_lower[2:]
            if remainder and (remainder in vocab or len(remainder) >= 3):
                return ["de", remainder]
        # Also handle d' at start of capitalized words
        if form.startswith("D'") and len(form) > 2:
            remainder = form[2:].lower()
            if remainder and (remainder in vocab or len(remainder) >= 3):
                return ["De", remainder.capitalize()]
    
    # Portuguese: verb-lo, verb-la, etc. (hyphenated clitics)
    if language == 'pt':
        # Pattern: verb-lo, verb-la, verb-las, verb-los, verb-me, verb-te, verb-nos, verb-vos
        clitic_patterns = [
            (r'^([a-z]+)-lo$', r'\1', 'lo'),
            (r'^([a-z]+)-la$', r'\1', 'la'),
            (r'^([a-z]+)-las$', r'\1', 'las'),
            (r'^([a-z]+)-los$', r'\1', 'los'),
            (r'^([a-z]+)-me$', r'\1', 'me'),
            (r'^([a-z]+)-te$', r'\1', 'te'),
            (r'^([a-z]+)-nos$', r'\1', 'nos'),
            (r'^([a-z]+)-vos$', r'\1', 'vos'),
        ]
        
        for pattern, verb_group, clitic in clitic_patterns:
            match = re.match(pattern, form_lower)
            if match:
                verb_part = match.group(1)
                # If ambiguous (exists as single word), prefer keeping as single word
                # unless verb part clearly exists in vocab
                if exists_as_single_word:
                    # Only split if verb part exists in vocab (strong evidence)
                    if verb_part in vocab:
                        return [verb_part, clitic]
                    # Otherwise, keep as single word (ambiguous)
                    return None
                else:
                    # Not ambiguous, safe to split if verb part looks valid
                    if verb_part in vocab or len(verb_part) >= 3:
                        return [verb_part, clitic]
    
    # Spanish: dámelo = dé + me + lo (no hyphen, verb + clitics)
    if language == 'es':
        # Spanish clitics: me, te, se, nos, os, le, la, lo, les, las, los
        # Pattern: verb ending in vowel + clitics
        # Common: dáme, dámelo, dámela, etc.
        
        # Try to split at clitic boundaries
        spanish_clitics = ['me', 'te', 'se', 'nos', 'os', 'le', 'la', 'lo', 'les', 'las', 'los']
        
        # Check if form ends with known clitics
        for clitic in spanish_clitics:
            if form_lower.endswith(clitic) and len(form_lower) > len(clitic):
                verb_part = form_lower[:-len(clitic)]
                
                # Check if verb part exists in vocab (e.g., "dá" from "dar")
                # Also check if verb_part + "r" exists (infinitive form)
                verb_inf = verb_part + 'r'
                verb_exists = verb_part in vocab or verb_inf in vocab
                
                # If ambiguous (exists as single word), only split if verb clearly exists
                if exists_as_single_word:
                    if verb_exists:
                        return [verb_part, clitic]
                    # Otherwise, keep as single word (ambiguous, e.g., "kárate" vs "kára" + "te")
                    return None
                else:
                    # Not ambiguous, split if verb part looks valid
                    if verb_exists or len(verb_part) >= 2:
                        # Check for multiple clitics (e.g., dámelo = dá + me + lo)
                        # Try to find another clitic in the middle
                        remaining = verb_part
                        clitics_found = [clitic]
                        
                        # Check for second clitic
                        for clitic2 in spanish_clitics:
                            if remaining.endswith(clitic2) and len(remaining) > len(clitic2):
                                verb_base = remaining[:-len(clitic2)]
                                if verb_base in vocab or verb_base + 'r' in vocab:
                                    return [verb_base, clitic2, clitic]
                        
                        return [verb_part, clitic]
        
        # Historic Spanish: destas, dellos, etc. (aggressive mode)
        if aggressive:
            if form_lower.startswith('d') and len(form_lower) > 3:
                # Try "de" + remainder
                remainder = form_lower[1:]  # Remove 'd'
                if remainder and (remainder in vocab or len(remainder) >= 3):
                    # Check if it's ambiguous (exists as single word)
                    if exists_as_single_word:
                        # Only split if remainder clearly exists in vocab
                        if remainder in vocab:
                            return ["de", remainder]
                        return None
                    return ["de", remainder]
    
    return None



