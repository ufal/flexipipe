"""
Viterbi module for FlexiPipe.
"""
import re
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

def viterbi_tag_sentence(sentence: List[str], vocab: Dict, transition_probs: Dict, 
                         tag_type: str = 'upos') -> List[str]:
    """
    Tag a sentence using Viterbi algorithm.
    
    Args:
        sentence: List of word forms
        vocab: Vocabulary dictionary (form -> analysis or list of analyses)
        transition_probs: Transition probabilities dict with 'upos' or 'xpos' keys
        tag_type: 'upos' or 'xpos'
    
    Returns:
        List of predicted tags
    """
    if not sentence:
        return []
    
    # Get transition probabilities for this tag type
    trans_probs = transition_probs.get(tag_type, {})
    start_probs = transition_probs.get('start', {})
    
    # Collect all possible tags from vocab
    all_tags = set()
    for entry in vocab.values():
        if isinstance(entry, list):
            for analysis in entry:
                tag = analysis.get(tag_type, '_')
                if tag != '_':
                    all_tags.add(tag)
        else:
            tag = entry.get(tag_type, '_')
            if tag != '_':
                all_tags.add(tag)
    
    # Add tags from transitions
    for prev_tag in trans_probs.keys():
        all_tags.add(prev_tag)
        for curr_tag in trans_probs[prev_tag].keys():
            all_tags.add(curr_tag)
    
    all_tags = sorted(all_tags)  # For consistent ordering
    
    if not all_tags:
        # No tags available, return all '_'
        return ['_'] * len(sentence)
    
    # Calculate raw tag frequencies from vocabulary (for OOV fallback)
    # This is better than uniform distribution - use actual tag distribution in vocab
    tag_frequencies = {}
    total_tag_count = 0
    for entry in vocab.values():
        if isinstance(entry, list):
            for analysis in entry:
                tag = analysis.get(tag_type, '_')
                if tag != '_':
                    count = analysis.get('count', 1)
                    tag_frequencies[tag] = tag_frequencies.get(tag, 0) + count
                    total_tag_count += count
        else:
            tag = entry.get(tag_type, '_')
            if tag != '_':
                count = entry.get('count', 1)
                tag_frequencies[tag] = tag_frequencies.get(tag, 0) + count
                total_tag_count += count
    
    # Create default emission probabilities based on tag frequencies
    # If a tag doesn't appear in vocab, give it a very small probability (smoothing)
    default_emission = {}
    smoothing = 0.1  # Small smoothing value for unseen tags
    if total_tag_count > 0:
        for tag in all_tags:
            freq = tag_frequencies.get(tag, 0)
            default_emission[tag] = (freq + smoothing) / (total_tag_count + smoothing * len(all_tags))
    else:
        # No frequency data - use uniform as absolute last resort
        uniform = 1.0 / len(all_tags)
        default_emission = {tag: uniform for tag in all_tags}
    
    # Build emission probabilities for each word
    # emission[word_idx][tag] = log probability
    emission = []
    for word in sentence:
        word_emission = {}
        word_lower = word.lower()
        
        # Get entry from vocab (try exact case, then lowercase)
        entry = vocab.get(word) or vocab.get(word_lower)
        
        if entry:
            # Word in vocab - use frequency-based emission
            # BUT: Filter out analyses that don't have the requested tag_type
            # This prevents using entries without XPOS for XPOS tagging
            if isinstance(entry, list):
                # Multiple analyses - filter to only those with the tag_type
                analyses_with_tag = [a for a in entry if a.get(tag_type) and a.get(tag_type) != '_']
                if analyses_with_tag:
                    # Use only analyses that have the tag
                    total_count = sum(a.get('count', 1) for a in analyses_with_tag)
                    for analysis in analyses_with_tag:
                        tag = analysis.get(tag_type, '_')
                        if tag != '_':
                            count = analysis.get('count', 1)
                            # Use log probability (add small smoothing)
                            prob = (count + 0.1) / (total_count + 0.1 * len(all_tags))
                            word_emission[tag] = word_emission.get(tag, 0) + prob
                # If no analyses have the tag, treat as OOV (fall through to OOV handling)
            else:
                # Single analysis - check if it has the tag
                tag = entry.get(tag_type, '_')
                if tag != '_':
                    count = entry.get('count', 1)
                    # Normalize to probability
                    prob = (count + 0.1) / (count + 0.1 * len(all_tags))
                    word_emission[tag] = prob
                # If entry doesn't have the tag, treat as OOV (fall through to OOV handling)
            
            # Normalize to probabilities and convert to log space
            total_prob = sum(word_emission.values())
            if total_prob > 0:
                for tag in word_emission:
                    word_emission[tag] = word_emission[tag] / total_prob
            else:
                # Uniform distribution if no counts
                uniform = 1.0 / len(all_tags)
                for tag in all_tags:
                    word_emission[tag] = uniform
        else:
            # OOV word - check if it's punctuation-only first
            # Historic texts often have punctuation marks not in vocabulary
            # (e.g., "!" in ODE corpus, various historic punctuation marks)
            # Check if word contains only punctuation/symbol characters (no letters, digits)
            is_punctuation_only = bool(word and not re.search(r'[a-zA-Z0-9]', word) and re.search(r'[^\s]', word))
            
            if is_punctuation_only:
                # Punctuation-only OOV word
                word_emission = {}
                
                if tag_type == 'upos':
                    # For UPOS, always use PUNCT for punctuation
                    if 'PUNCT' in all_tags:
                        word_emission['PUNCT'] = 1.0
                    else:
                        # If PUNCT not in tags, use most common tag as fallback
                        word_emission = default_emission.copy()
                else:
                    # For XPOS, find punctuation tags (typically start with 'F' or are short)
                    # Look for punctuation tags in all_tags
                    punct_tags = [tag for tag in all_tags if (
                        tag.startswith('F') and len(tag) <= 4  # Short tags starting with F
                        or tag in ('PUNCT', 'Punct', 'punct')  # Common punctuation tag names
                    )]
                    
                    if punct_tags:
                        # Use uniform distribution over available punctuation tags
                        prob = 1.0 / len(punct_tags)
                        for tag in punct_tags:
                            word_emission[tag] = prob
                    else:
                        # No punctuation tags found - try to infer from common patterns
                        # Default to a generic punctuation tag if available
                        # Otherwise fall back to tag frequency distribution
                        word_emission = default_emission.copy()
            else:
                # Not punctuation - try suffix-based matching
                # OOV word - try to use suffix-based tag frequencies (for suffixing languages)
                # This is much better than uniform distribution or general similarity matching
                # In suffixing languages, the ending is morphologically significant
                # Example: words ending in -tido are likely past participles (verbs)
                word_emission = {}
                word_lower = word.lower()
                
                # Try suffix-based tag frequency lookup (like lemmatization pattern matching)
                # Try progressively shorter suffixes (longest first) to find tag distribution
                # This is more reliable than general similarity matching for suffixing languages
                suffix_tag_counts = {}
                total_suffix_count = 0
                
                # Try suffixes from 6 chars down to 2 chars (morphologically significant endings)
                for suffix_len in range(6, 1, -1):
                    if len(word_lower) >= suffix_len:
                        suffix = word_lower[-suffix_len:]
                        # Find all words in vocab ending with this suffix
                        suffix_words = []
                        for vocab_word, vocab_entry in vocab.items():
                            if vocab_word.lower().endswith(suffix) and vocab_word.lower() != word_lower:
                                suffix_words.append((vocab_word, vocab_entry))
                        
                        # If we found words with this suffix, use their tag distribution
                        if suffix_words:
                            for vocab_word, vocab_entry in suffix_words:
                                if isinstance(vocab_entry, list):
                                    for analysis in vocab_entry:
                                        tag = analysis.get(tag_type, '_')
                                        if tag != '_':
                                            count = analysis.get('count', 1)
                                            suffix_tag_counts[tag] = suffix_tag_counts.get(tag, 0) + count
                                            total_suffix_count += count
                                else:
                                    tag = vocab_entry.get(tag_type, '_')
                                    if tag != '_':
                                        count = vocab_entry.get('count', 1)
                                        suffix_tag_counts[tag] = suffix_tag_counts.get(tag, 0) + count
                                        total_suffix_count += count
                            
                            # Found suffix matches - use their tag distribution (stop at longest suffix)
                            break
                
                # Convert suffix-based tag counts to probabilities
                if suffix_tag_counts and total_suffix_count > 0:
                    for tag, count in suffix_tag_counts.items():
                        word_emission[tag] = count / total_suffix_count
                
                # If no similar words found, or if we have very low similarity, use pattern-based heuristics
                if not word_emission or max(word_emission.values()) < 0.3:
                    # Pattern-based heuristics for common endings
                    # This helps assign more reasonable tags for OOV words
                    word_lower = word.lower()
                    
                    # Spanish/Portuguese verb endings (common in historic texts)
                    if word_lower.endswith(('ado', 'ada', 'ados', 'adas', 'ido', 'ida', 'idos', 'idas')):
                        # Past participles - likely to be ADJ or VERB
                        for tag in all_tags:
                            if tag.startswith('A') or tag.startswith('V'):  # Adjective or Verb
                                word_emission[tag] = word_emission.get(tag, 0) + 0.5
                    
                    # Preposition-like patterns (very short words, single letters)
                    if len(word_lower) <= 2 or word_lower in ('de', 'a', 'en', 'por', 'para', 'con', 'sin', 'sobre'):
                        for tag in all_tags:
                            if tag.startswith('SP'):  # Preposition
                                word_emission[tag] = word_emission.get(tag, 0) + 0.3
                
                # Fallback: if still no emission probabilities, use tag frequency distribution
                if not word_emission:
                    # Use default emission based on tag frequencies in vocabulary
                    # This is much better than uniform - reflects actual tag distribution
                    word_emission = default_emission.copy()
                else:
                    # Normalize the emission probabilities we found
                    total_prob = sum(word_emission.values())
                    if total_prob > 0:
                        for tag in word_emission:
                            word_emission[tag] = word_emission[tag] / total_prob
                    else:
                        # Fallback to tag frequencies if normalization fails
                        word_emission = default_emission.copy()
        
        # Convert to log space
        log_emission = {}
        import math
        for tag, prob in word_emission.items():
            log_emission[tag] = math.log(max(prob, 1e-10))  # Avoid log(0)
        
        emission.append(log_emission)
    
    # Viterbi algorithm
    n = len(sentence)
    if n == 0:
        return []
    
    # Initialize DP table: viterbi[position][tag] = best log probability
    viterbi = [{} for _ in range(n)]
    backpointer = [{} for _ in range(n)]
    
    # Initialization: first word
    import math
    for tag in all_tags:
        # Start probability (log)
        start_prob = start_probs.get(tag, 1.0 / len(all_tags))
        start_log = math.log(max(start_prob, 1e-10))
        
        # Emission probability (log)
        emit_log = emission[0].get(tag, math.log(1e-10))
        
        viterbi[0][tag] = start_log + emit_log
        backpointer[0][tag] = None
    
    # Recursion: fill DP table
    for t in range(1, n):
        for curr_tag in all_tags:
            best_prob = float('-inf')
            best_prev_tag = None
            
            for prev_tag in all_tags:
                # Transition probability (log)
                trans_prob = trans_probs.get(prev_tag, {}).get(curr_tag, 1e-10)
                trans_log = math.log(max(trans_prob, 1e-10))
                
                # Emission probability (log)
                emit_log = emission[t].get(curr_tag, math.log(1e-10))
                
                # Total probability
                prob = viterbi[t-1][prev_tag] + trans_log + emit_log
                
                if prob > best_prob:
                    best_prob = prob
                    best_prev_tag = prev_tag
            
            viterbi[t][curr_tag] = best_prob
            backpointer[t][curr_tag] = best_prev_tag
    
    # Termination: find best final tag
    best_final_tag = max(all_tags, key=lambda tag: viterbi[n-1][tag])
    
    # Backtrack: reconstruct best path
    path = [best_final_tag]
    for t in range(n-1, 0, -1):
        prev_tag = backpointer[t][path[0]]
        path.insert(0, prev_tag)
    
    return path



