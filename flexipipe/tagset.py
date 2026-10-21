"""
Tagset definition parser for FlexiPipe.

Reads TEITOK-style tagset XML files and creates mappings from XPOS to UPOS and FEATS.
Supports position-based tagsets (EAGLES style).
"""
import xml.etree.ElementTree as ET
from typing import Dict, Optional, Tuple
from pathlib import Path
from collections import defaultdict


# Mapping from EAGLES main categories to UD UPOS
EAGLES_TO_UPOS = {
    'A': 'ADJ',      # Adjective
    'R': 'ADV',      # Adverb
    'D': 'DET',      # Determiner
    'N': 'NOUN',     # Noun (common)
    'NP': 'PROPN',   # Proper Noun
    'V': 'VERB',     # Verb
    'P': 'PRON',     # Pronoun
    'C': 'CCONJ',   # Conjunction
    'I': 'INTJ',     # Interjection
    'S': 'ADP',      # Preposition
    'F': 'PUNCT',    # Punctuation
    'Z': 'NUM',      # Numeral
    'X': 'X',        # Other
}


def parse_teitok_tagset(tagset_file: Path) -> Dict:
    """
    Parse a TEITOK-style tagset XML file.
    
    Args:
        tagset_file: Path to tagset XML file
        
    Returns:
        Dictionary with:
        - 'main_categories': Main category mappings
        - 'position_defs': Position definitions for feature extraction
        - 'tagset_file': Path to tagset file
    """
    tree = ET.parse(tagset_file)
    root = tree.getroot()
    
    # Find tagset element
    tagset_elem = root.find('.//tagset') or root
    if tagset_elem is None:
        raise ValueError(f"Could not find <tagset> element in {tagset_file}")
    
    positions_elem = tagset_elem.find('positions')
    if positions_elem is None:
        raise ValueError(f"Could not find <positions> element in {tagset_file}")
    
    # Parse position-based tagset
    main_categories = {}
    position_defs = {}  # main_cat -> {pos_index -> {char_key: {ud_feat, ud_value, display}}}
    
    for item in positions_elem.findall('item'):
        key = item.get('key')
        maintag = item.get('maintag', '0')
        display = item.get('display', '')
        
        if key:
            main_categories[key] = {
                'maintag': maintag,
                'display': display,
                'upos': EAGLES_TO_UPOS.get(key, 'X')
            }
            
            # Initialize position definitions for this main category
            if key not in position_defs:
                position_defs[key] = {}
            
            # Handle multi-category items (e.g., NP for proper nouns)
            multi_elem = item.find('multi')
            if multi_elem is not None:
                for multi_item in multi_elem.findall('item'):
                    multi_key = multi_item.get('key')
                    if multi_key:
                        main_categories[multi_key] = {
                            'maintag': maintag,
                            'display': multi_item.get('display', ''),
                            'upos': EAGLES_TO_UPOS.get(multi_key, EAGLES_TO_UPOS.get(key, 'X'))
                        }
                        # Share position definitions with parent category
                        if multi_key not in position_defs:
                            position_defs[multi_key] = {}
            
            # Parse position definitions for this main category
            for pos_item in item.findall('item[@pos]'):
                pos_index = int(pos_item.get('pos'))
                if pos_index not in position_defs[key]:
                    position_defs[key][pos_index] = {}
                
                pos_display = pos_item.get('display', '').lower()
                
                # Map common feature names to UD FEATS
                ud_feat_name = _map_feature_name_to_ud(pos_display)
                
                # Parse value definitions - store by value key (the character in the XPOS tag)
                values = pos_item.findall('item')
                sibling_displays = frozenset(v.get('display', '').strip().lower() for v in values)
                for value_item in values:
                    value_key = value_item.get('key', '')
                    value_display = value_item.get('display', '').strip()
                    
                    # Skip items with key="0" and empty display (non-relevant features)
                    if value_key == '0' and not value_display:
                        continue
                    
                    if not value_key:
                        continue
                    position_defs[key][pos_index][value_key] = {
                        'display': value_display,
                        'ud_feat': ud_feat_name,
                        # the UD features this value stands for (None: none)
                        'ud_feats': _map_feature_value_to_ud(ud_feat_name, value_display, value_key, sibling_displays),
                        # a word type that changes the UPOS (a subordinating conjunction, a proper noun)
                        'upos': _TYPE_UPOS.get((EAGLES_TO_UPOS.get(key), value_display.lower()))
                                if pos_display in ('type', 'tipo') else None,
                    }
            
            # Also copy position definitions to multi-category items
            if multi_elem is not None:
                for multi_item in multi_elem.findall('item'):
                    multi_key = multi_item.get('key')
                    if multi_key and key in position_defs:
                        position_defs[multi_key] = position_defs[key].copy()
    
    return {
        'main_categories': main_categories,
        'position_defs': position_defs,
        'tagset_file': str(tagset_file)
    }


def _map_feature_name_to_ud(feature_name: str) -> Optional[str]:
    """Map feature name to UD FEATS name."""
    feature_lower = feature_name.lower()
    
    # Common mappings (including abbreviations and multiple languages)
    mappings = {
        'gender': 'Gender',
        'género': 'Gender',
        'gen': 'Gender',  # Abbreviation (Czech, Portuguese)
        'number': 'Number',
        'número': 'Number',
        'num': 'Number',  # Abbreviation (Czech, Portuguese)
        'person': 'Person',
        'persona': 'Person',
        'mood': 'Mood',
        'modo': 'Mood',
        'tense': 'Tense',
        'tiempo': 'Tense',
        'case': 'Case',
        'caso': 'Case',
        'degree': 'Degree',
        'grado': 'Degree',
        'type': None,  # Type is usually not a UD feature
        'tipo': None,
        'form': 'Form',
        'forma': 'Form',
    }
    
    return mappings.get(feature_lower)


# Feature values by their description in the tagset (lowercased) → the UD features they stand for; None: no UD value
# (ambiguous, "any", unspecified …: left out rather than invented). A description not listed here is used only if it is
# itself a UD value of the feature. Mood and tense descriptions may set other features (UD: infinitive, gerund and
# participle are VerbForms; the conditional is a Mood).
_VALUES = {
    'Gender': {
        'masculine': {'Gender': 'Masc'}, 'masculino': {'Gender': 'Masc'}, 'masc': {'Gender': 'Masc'},
        'feminine': {'Gender': 'Fem'}, 'femenino': {'Gender': 'Fem'}, 'feminino': {'Gender': 'Fem'}, 'fem': {'Gender': 'Fem'},
        'neuter': {'Gender': 'Neut'}, 'neutral': {'Gender': 'Neut'}, 'neutro': {'Gender': 'Neut'},
        'common': {'Gender': 'Com'}, 'común': {'Gender': 'Com'}, 'comum': {'Gender': 'Com'},  # see _common_is_either
        'masc/fem': {'Gender': 'Fem,Masc'}, 'masculine or feminine': {'Gender': 'Fem,Masc'},
        'feminine or neuter': {'Gender': 'Fem,Neut'}, 'not feminine': {'Gender': 'Masc,Neut'},
        'nonneuter': {'Gender': 'Com'},
        'masculine animate': {'Gender': 'Masc', 'Animacy': 'Anim'},
        'masculine inanimate': {'Gender': 'Masc', 'Animacy': 'Inan'},
    },
    'Number': {
        'singular': {'Number': 'Sing'}, 'plural': {'Number': 'Plur'}, 'dual': {'Number': 'Dual'},
        'invariable': None, 'impersonal-invariable': None, 'invariável': None, 'common': None, 'singular or plural': None,
    },
    'Person': {
        'first': {'Person': '1'}, 'primera': {'Person': '1'}, 'primeira': {'Person': '1'},
        'second': {'Person': '2'}, 'segunda': {'Person': '2'},
        'third': {'Person': '3'}, 'tercera': {'Person': '3'}, 'terceira': {'Person': '3'},
    },
    'Mood': {
        'indicative': {'Mood': 'Ind'}, 'indicativo': {'Mood': 'Ind'},
        'subjunctive': {'Mood': 'Sub'}, 'subjuntivo': {'Mood': 'Sub'}, 'conjuntivo': {'Mood': 'Sub'},
        'imperative': {'Mood': 'Imp'}, 'imperativo': {'Mood': 'Imp'},
        'conditional': {'Mood': 'Cnd'}, 'condicional': {'Mood': 'Cnd'}, 'optative': {'Mood': 'Opt'},
        'infinitive': {'VerbForm': 'Inf'}, 'infinitivo': {'VerbForm': 'Inf'},
        'gerund': {'VerbForm': 'Ger'}, 'gerundio': {'VerbForm': 'Ger'}, 'gerúndio': {'VerbForm': 'Ger'},
        'participle': {'VerbForm': 'Part'}, 'participial': {'VerbForm': 'Part'}, 'pastparticiple': {'VerbForm': 'Part'},
        'participio': {'VerbForm': 'Part'}, 'particípio': {'VerbForm': 'Part'},
    },
    'Tense': {
        'present': {'Tense': 'Pres'}, 'presente': {'Tense': 'Pres'},
        'imperfective': {'Tense': 'Imp'}, 'imperfect': {'Tense': 'Imp'}, 'pretérito imperfecto': {'Tense': 'Imp'},
        'pretérito imperfeito': {'Tense': 'Imp'},
        'future': {'Tense': 'Fut'}, 'futuro': {'Tense': 'Fut'},
        'past': {'Tense': 'Past'}, 'pretérito perfecto': {'Tense': 'Past'}, 'pretérito perfeito': {'Tense': 'Past'},
        'pluperfect': {'Tense': 'Pqp'}, 'pretérito pluscuamperfecto': {'Tense': 'Pqp'}, 'pretérito mais-que-perfeito': {'Tense': 'Pqp'},
        'conditional': {'Mood': 'Cnd'}, 'condicional': {'Mood': 'Cnd'},
        'past or present': None,
    },
    'Case': {
        'nominative': {'Case': 'Nom'}, 'nominativo': {'Case': 'Nom'}, 'genitive': {'Case': 'Gen'},
        'dative': {'Case': 'Dat'}, 'dativo': {'Case': 'Dat'}, 'accusative': {'Case': 'Acc'}, 'acusativo': {'Case': 'Acc'},
        'vocative': {'Case': 'Voc'}, 'locative': {'Case': 'Loc'}, 'prepositional': {'Case': 'Loc'},
        'instrumental': {'Case': 'Ins'}, 'ablative': {'Case': 'Abl'}, 'partitive': {'Case': 'Par'},
        'illative': {'Case': 'Ill'}, 'inessive': {'Case': 'Ine'}, 'elative': {'Case': 'Ela'}, 'allative': {'Case': 'All'},
        'adessive': {'Case': 'Ade'}, 'ablative case': {'Case': 'Abl'}, 'essive': {'Case': 'Ess'},
        'translative': {'Case': 'Tra'}, 'terminative': {'Case': 'Ter'}, 'abessive': {'Case': 'Abe'},
        'comitative': {'Case': 'Com'}, 'ergative': {'Case': 'Erg'}, 'absolutive': {'Case': 'Abs'},
        'genitive or dative': {'Case': 'Dat,Gen'}, 'accusative or dative': {'Case': 'Acc,Dat'},
        # Romance pronouns after a preposition (mí, ti): UD Spanish / Portuguese / Catalan's PrepCase
        'oblique': {'PrepCase': 'Pre'}, 'oblicuo': {'PrepCase': 'Pre'}, 'oblíquo': {'PrepCase': 'Pre'},
    },
    'Degree': {
        'positive': {'Degree': 'Pos'}, 'comparative': {'Degree': 'Cmp'}, 'comparativo': {'Degree': 'Cmp'},
        'superlative': {'Degree': 'Sup'}, 'superlativo': {'Degree': 'Sup'}, 'suerlativo': {'Degree': 'Sup'},
        'absolute': {'Degree': 'Abs'}, 'diminutive': {'Degree': 'Dim'}, 'diminutivo': {'Degree': 'Dim'},
        'augmentative': {'Degree': 'Aug'}, 'aumentativo': {'Degree': 'Aug'},
    },
}

# UD's values per feature (universaldependencies.org/u/feat), for descriptions not in _VALUES
_UD_VALUES = {
    'Gender': {'Com', 'Fem', 'Masc', 'Neut'}, 'Number': {'Coll', 'Count', 'Dual', 'Grpa', 'Grpl', 'Inv', 'Pauc', 'Plur', 'Ptan', 'Sing', 'Tri'},
    'Person': {'0', '1', '2', '3', '4'}, 'Mood': {'Adm', 'Cnd', 'Des', 'Imp', 'Ind', 'Int', 'Irr', 'Jus', 'Nec', 'Opt', 'Pot', 'Prp', 'Qot', 'Sub'},
    'Tense': {'Fut', 'Imp', 'Past', 'Pqp', 'Pres'}, 'Degree': {'Abs', 'Aug', 'Cmp', 'Dim', 'Equ', 'Pos', 'Sup'},
    'Case': {'Abs', 'Acc', 'Erg', 'Nom', 'Abe', 'Ben', 'Cau', 'Cmp', 'Cns', 'Com', 'Dat', 'Dis', 'Equ', 'Gen', 'Ins', 'Par', 'Tem',
             'Tra', 'Voc', 'Abl', 'Add', 'Ade', 'All', 'Del', 'Ela', 'Ess', 'Ill', 'Ine', 'Lat', 'Loc', 'Per', 'Sbe', 'Sbl', 'Spl',
             'Sub', 'Sup', 'Ter'},
}

# A word-type description that changes the UPOS of its main category: (category UPOS, description) → UPOS
_TYPE_UPOS = {('CCONJ', d): 'SCONJ' for d in ('subordinate', 'subordinating', 'subordinada', 'subordinante',
                                             'subordinativa', 'subordinator')}
_TYPE_UPOS.update({('NOUN', d): 'PROPN' for d in ('proper', 'propio', 'próprio')})

_FINITE_MOODS = {'Ind', 'Sub', 'Imp', 'Cnd', 'Opt'}


def _common_is_either(displays) -> bool:
    """EAGLES 'common' next to masculine and feminine (Romance: invariable for gender) means either of them — no UD
    value; next to neuter only (Scandinavian) it is UD's Gender=Com."""
    return {'masculine', 'feminine'} <= displays or {'masculino', 'femenino'} <= displays


def _map_feature_value_to_ud(feat_name: Optional[str], value_display: str, value_key: str,
                             sibling_displays=frozenset()) -> Optional[Dict[str, str]]:
    """The UD features a value stands for ({feature: value}), or None where it has no UD value."""
    if feat_name is None or value_key == '0':
        return None
    value_lower = value_display.lower().strip()
    table = _VALUES.get(feat_name, {})
    if feat_name == 'Gender' and value_lower in ('common', 'común', 'comum') and _common_is_either(sibling_displays):
        return None
    if value_lower in table:
        return table[value_lower]
    if feat_name == 'Person' and value_key in ('1', '2', '3'):
        return {'Person': value_key}
    allowed = _UD_VALUES.get(feat_name, set())
    for candidate in (value_display.strip(), value_display.strip().capitalize()):
        if candidate in allowed:
            return {feat_name: candidate}
    return None


def xpos_to_upos_feats(xpos: str, tagset_def: Dict) -> Tuple[str, str]:
    """
    Convert XPOS tag to UPOS and FEATS using tagset definition.
    
    Args:
        xpos: XPOS tag (e.g., "NCMS000" for Spanish)
        tagset_def: Tagset definition from parse_teitok_tagset()
        
    Returns:
        Tuple of (upos, feats) where feats is in UD format (e.g., "Gender=Masc|Number=Sing")
    """
    if not xpos or xpos == '_':
        return '_', '_'
    
    main_categories = tagset_def.get('main_categories', {})
    position_defs = tagset_def.get('position_defs', {})
    
    # Determine main category (first character or first two for multi-char categories)
    main_cat = None
    remaining = ''
    if len(xpos) >= 2 and xpos[:2] in main_categories:
        # Check if this multi-char category shares position definitions with a single-char parent
        multi_cat = xpos[:2]
        single_cat = xpos[0]
        multi_upos = main_categories.get(multi_cat, {}).get('upos')
        single_upos = main_categories.get(single_cat, {}).get('upos')
        
        if (single_cat in main_categories and 
            multi_cat in position_defs and 
            single_cat in position_defs and
            position_defs[multi_cat] == position_defs[single_cat] and
            multi_upos == single_upos):
            # Multi-category shares position defs AND UPOS with single - treat second char as position 1
            main_cat = single_cat
            remaining = xpos[1:]  # Include the second character in remaining
        else:
            # True multi-char category (different position defs or different UPOS)
            main_cat = multi_cat
            remaining = xpos[2:]
    elif len(xpos) >= 1:
        main_cat = xpos[0]
        remaining = xpos[1:]
    else:
        return 'X', '_'
    
    # Get UPOS from main category
    cat_info = main_categories.get(main_cat)
    if not cat_info:
        return 'X', '_'
    
    upos = cat_info.get('upos', 'X')
    
    # Extract features from remaining positions
    feats_dict = {}
    
    # Get position definitions for this main category
    cat_pos_defs = position_defs.get(main_cat, {})
    
    for char_index, char in enumerate(remaining, start=1):
        char_info = cat_pos_defs.get(char_index, {}).get(char)
        if not char_info:
            continue
        if char_info.get('upos'):
            upos = char_info['upos']
        for feat, value in (char_info.get('ud_feats') or {}).items():
            if feats_dict.get(feat) == 'Cnd':
                continue  # the conditional (EAGLES: a tense) wins over the mood position's indicative
            feats_dict[feat] = value
    if feats_dict.get('Mood') in _FINITE_MOODS and 'VerbForm' not in feats_dict:
        feats_dict['VerbForm'] = 'Fin'
    if 'VerbForm' in feats_dict and feats_dict['VerbForm'] != 'Fin':
        feats_dict.pop('Mood', None)
    
    # Convert feats_dict to UD format string (features in UD's order: alphabetical, case-insensitive)
    if feats_dict:
        feats = '|'.join(f"{k}={v}" for k, v in sorted(feats_dict.items(), key=lambda kv: kv[0].lower()))
    else:
        feats = '_'
    
    return upos, feats

