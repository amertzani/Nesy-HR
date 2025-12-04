"""
Structured Knowledge Graph Construction Pipeline
================================================

This module implements a 7-step pipeline for building knowledge graphs:
1. Preprocessing - Clean and prepare text
2. Named Entity Recognition (NER) - Identify entities
3. Coreference Resolution - Unify mentions
4. Relation Extraction (RE) - Identify relationships
5. Entity Linking (EL) - Connect to external KBs
6. Knowledge Graph Construction - Build triples
7. Post-processing/Reasoning - Clean, deduplicate, infer

Author: Research Brain Team
Last Updated: 2025-01-15
"""

import re
import json
import sys
from typing import List, Dict, Tuple, Set, Optional, TYPE_CHECKING
from collections import defaultdict
from urllib.parse import quote
from datetime import datetime

# Import rdflib for type hints (if available)
if TYPE_CHECKING:
    import rdflib
else:
    try:
        import rdflib
    except ImportError:
        rdflib = None

# Import normalization functions from knowledge module
try:
    from knowledge import normalize_entity, clean_entity as kg_clean_entity, extract_core_entity
except ImportError:
    # Fallback if knowledge module not available
    def normalize_entity(entity):
        return entity.strip()
    def kg_clean_entity(entity):
        return entity.strip()
    def extract_core_entity(entity, context=""):
        return entity.strip(), ""

# Try to import spaCy for NER and preprocessing
# Note: spaCy may fail to install on Windows without C compiler - that's OK, we have fallbacks
try:
    import spacy
    SPACY_AVAILABLE = True
    try:
        nlp = spacy.load("en_core_web_sm")
        print("✅ spaCy model loaded successfully")
    except OSError:
        # Model not found - try to download it
        try:
            import subprocess
            print("📥 Downloading spaCy English model...")
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], 
                         check=False, capture_output=True)
            try:
                nlp = spacy.load("en_core_web_sm")
                print("✅ spaCy model downloaded and loaded")
            except:
                SPACY_AVAILABLE = False
                nlp = None
                print("⚠️  spaCy model not available. Using regex-based NER fallback.")
        except:
            SPACY_AVAILABLE = False
            nlp = None
            print("⚠️  spaCy English model not found. Using regex-based NER fallback.")
except ImportError:
    SPACY_AVAILABLE = False
    nlp = None
    # Don't print warning - this is expected on Windows without build tools
    # The pipeline will work fine with regex fallbacks

# Try to import clean-text for text cleaning
try:
    from cleantext import clean
    CLEANTEXT_AVAILABLE = True
except ImportError:
    CLEANTEXT_AVAILABLE = False
    print("⚠️  clean-text not available. Install with: pip install clean-text")

# Try to import neuralcoref for coreference resolution
try:
    import neuralcoref
    NEURALCOREF_AVAILABLE = True
except ImportError:
    NEURALCOREF_AVAILABLE = False
    print("⚠️  neuralcoref not available. Install with: pip install neuralcoref")

# Try to import OpenNRE for relation extraction
try:
    import opennre
    OPENNRE_AVAILABLE = True
    try:
        opennre_model = opennre.get_model('wiki80_cnn_softmax')
        print("✅ OpenNRE model loaded successfully")
    except Exception as e:
        print(f"⚠️  OpenNRE model loading failed: {e}")
        OPENNRE_AVAILABLE = False
        opennre_model = None
except ImportError:
    OPENNRE_AVAILABLE = False
    opennre_model = None
    print("⚠️  OpenNRE not available. Install with: pip install OpenNRE")

# Try to import REL for relation extraction and entity linking
try:
    import rel
    REL_AVAILABLE = True
except ImportError:
    REL_AVAILABLE = False
    print("⚠️  REL not available. Install with: pip install REL")

# Try to import blink-lite for entity linking
try:
    import blink_lite
    BLINK_LITE_AVAILABLE = True
except ImportError:
    BLINK_LITE_AVAILABLE = False
    print("⚠️  blink-lite not available. Install with: pip install blink-lite")


# ============================================================================
# STEP 1: PREPROCESSING
# ============================================================================

def preprocess_text(text: str) -> Dict[str, any]:
    """
    Step 1: Clean and prepare text for knowledge extraction.
    
    Uses:
    - clean-text library for removing URLs, emojis, punctuation
    - spaCy for tokenization, POS-tagging, lemmatization, sentence segmentation
    
    Returns:
        Dict with cleaned text, sentences, tokens, POS tags, lemmas, and metadata
    """
    if not text or not text.strip():
        return {
            "cleaned_text": "",
            "sentences": [],
            "tokens": [],
            "pos_tags": [],
            "lemmas": [],
            "metadata": {}
        }
    
    # Use clean-text library if available
    if CLEANTEXT_AVAILABLE:
        # Clean text: remove URLs, emojis, normalize unicode, fix encoding
        cleaned = clean(
            text,
            fix_unicode=True,
            to_ascii=False,
            lower=False,
            no_urls=True,
            no_emails=True,
            no_phone_numbers=False,
            no_numbers=False,
            no_digits=False,
            no_currency_symbols=False,
            no_punct=False,  # Keep punctuation for sentence segmentation
            replace_with_url="",
            replace_with_email="",
            replace_with_phone_number="",
            replace_with_number="",
            replace_with_digit="",
            replace_with_currency_symbol="",
            lang="en"
        )
    else:
        # Fallback: basic cleaning
        cleaned = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
        cleaned = re.sub(r'\s+', ' ', cleaned.strip())
    
    # Normalize quotes and dashes
    cleaned = cleaned.replace('"', '"').replace('"', '"')
    cleaned = cleaned.replace(''', "'").replace(''', "'")
    cleaned = cleaned.replace('—', '-').replace('–', '-')
    
    sentences = []
    tokens = []
    pos_tags = []
    lemmas = []
    
    # Use spaCy for advanced preprocessing if available
    if SPACY_AVAILABLE and nlp:
        doc = nlp(cleaned)
        
        # Get sentences (spaCy handles abbreviations correctly)
        sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]
        
        # Get tokens, POS tags, and lemmas
        for token in doc:
            if not token.is_space and not token.is_punct:
                tokens.append(token.text)
                pos_tags.append((token.text, token.pos_, token.tag_))
                lemmas.append(token.lemma_)
    else:
        # Fallback: regex-based sentence segmentation
        sentence_endings = r'[.!?]+(?:\s+|$)'
        sentences = [s.strip() for s in re.split(sentence_endings, cleaned) if len(s.strip()) > 10]
        
        # Basic tokenization
        tokens = re.findall(r'\b\w+\b', cleaned)
        pos_tags = []
        lemmas = []
    
    return {
        "cleaned_text": cleaned,
        "sentences": sentences,
        "tokens": tokens,
        "pos_tags": pos_tags,
        "lemmas": lemmas,
        "metadata": {
            "original_length": len(text),
            "cleaned_length": len(cleaned),
            "sentence_count": len(sentences),
            "token_count": len(tokens),
            "used_spacy": SPACY_AVAILABLE and nlp is not None
        }
    }


# ============================================================================
# STEP 2: NAMED ENTITY RECOGNITION (NER)
# ============================================================================

def extract_entities_ner(text: str, sentences: List[str]) -> Dict[str, List[Dict]]:
    """
    Step 2: Identify and label named entities using spaCy NER.
    
    Detects:
    - ORGANIZATION (ORG): Companies, institutions
    - PERSON (PER): People names
    - LOCATION (LOC): Places, addresses
    - PROJECT (PROJ): Project names (custom detection)
    - DATE (DATE): Dates, times
    - CONCEPT (CONCEPT): General concepts
    
    Returns:
        Dict mapping entity types to lists of entities with positions
    """
    entities = {
        "ORG": [],
        "PERSON": [],
        "LOCATION": [],
        "PROJECT": [],
        "DATE": [],
        "CONCEPT": []
    }
    
    if SPACY_AVAILABLE and nlp:
        # Use spaCy for better NER - process full text and each sentence
        doc = nlp(text)
        
        # Extract entities from spaCy
        seen_entities = set()  # Avoid duplicates
        
        for ent in doc.ents:
            entity_type = map_spacy_label(ent.label_)
            if entity_type:
                entity_key = (ent.text.lower(), entity_type)
                if entity_key not in seen_entities:
                    seen_entities.add(entity_key)
                    entities[entity_type].append({
                        "text": ent.text,
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "label": entity_type,
                        "confidence": 1.0
                    })
        
        # Also extract from individual sentences for better coverage
        for sentence in sentences:
            if len(sentence.strip()) > 10:
                sent_doc = nlp(sentence)
                for ent in sent_doc.ents:
                    entity_type = map_spacy_label(ent.label_)
                    if entity_type:
                        entity_key = (ent.text.lower(), entity_type)
                        if entity_key not in seen_entities:
                            seen_entities.add(entity_key)
                            # Find position in original text
                            text_pos = text.find(ent.text)
                            if text_pos >= 0:
                                entities[entity_type].append({
                                    "text": ent.text,
                                    "start": text_pos,
                                    "end": text_pos + len(ent.text),
                                    "label": entity_type,
                                    "confidence": 1.0
                                })
        
        # ENHANCED: Custom patterns for entities spaCy might miss
        
        # 1. Universities and Colleges (case-insensitive)
        university_patterns = [
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:university|college|institute|institution|academy|school)\b',
            r'\b(?:university|college|institute|institution|academy|school)\s+of\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
            r'\b([a-z]+(?:\s+[a-z]+)*)\s+(?:university|college|institute|institution|academy|school)\b',  # Lowercase
        ]
        for pattern in university_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                org_name = match.group(1).strip()
                if org_name and len(org_name) > 2:
                    # Capitalize for consistency
                    org_name = org_name.title() if org_name.islower() else org_name
                    entity_key = (org_name.lower(), "ORG")
                    if entity_key not in seen_entities:
                        seen_entities.add(entity_key)
                        entities["ORG"].append({
                            "text": org_name,
                            "start": match.start(),
                            "end": match.end(),
                            "label": "ORG",
                            "confidence": 0.8
                        })
        
        # 2. Titles and Honorifics (Queen, King, President, etc.)
        title_patterns = [
            r'\b(Queen|King|Prince|Princess|President|Prime\s+Minister|Chancellor|Minister|Doctor|Dr\.|Professor|Prof\.|Mr\.|Mrs\.|Ms\.|Sir|Dame|Lord|Lady)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)?',
            r'\b(Queen|King|Prince|Princess|President|Prime\s+Minister|Chancellor|Minister|Doctor|Dr\.|Professor|Prof\.)\b',
        ]
        for pattern in title_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                title = match.group(1).strip()
                # If there's a name after the title, include it
                if len(match.groups()) > 1 and match.group(2):
                    full_name = f"{title} {match.group(2)}"
                else:
                    full_name = title
                
                entity_key = (full_name.lower(), "PERSON")
                if entity_key not in seen_entities:
                    seen_entities.add(entity_key)
                    entities["PERSON"].append({
                        "text": full_name,
                        "start": match.start(),
                        "end": match.end(),
                        "label": "PERSON",
                        "confidence": 0.7
                    })
        
        # 3. Abbreviations and Acronyms (with context)
        # Look for patterns like "ICL (Imperial College London)" or "ICL is..."
        abbreviation_patterns = [
            r'\b([A-Z]{2,}(?:[-_][A-Z]+)*)\s*\(([A-Z][a-zA-Z\s]+)\)',  # "ICL (Imperial College London)"
            r'\b([A-Z][a-zA-Z\s]+)\s*\(([A-Z]{2,}(?:[-_][A-Z]+)*)\)',  # "Imperial College London (ICL)"
            r'\b([A-Z]{2,}(?:[-_][A-Z]+)*)\s+(?:is|are|was|were|stands\s+for)\s+([A-Z][a-zA-Z\s]+)',  # "ICL is Imperial College London"
            r'\b([a-z]{2,}(?:[-_][a-z]+)*)\s+(?:is|are|was|were|stands\s+for)\s+([A-Z][a-zA-Z\s]+)',  # "icl is Imperial College London"
        ]
        for pattern in abbreviation_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                abbrev = match.group(1).strip()
                full_name = match.group(2).strip()
                
                # Add abbreviation as PROJECT or ORG
                entity_key = (abbrev.lower(), "PROJECT")
                if entity_key not in seen_entities:
                    seen_entities.add(entity_key)
                    entities["PROJECT"].append({
                        "text": abbrev.upper() if abbrev.islower() else abbrev,
                        "start": match.start(1),
                        "end": match.end(1),
                        "label": "PROJECT",
                        "confidence": 0.7
                    })
                
                # Add full name as ORG
                entity_key = (full_name.lower(), "ORG")
                if entity_key not in seen_entities:
                    seen_entities.add(entity_key)
                    entities["ORG"].append({
                        "text": full_name,
                        "start": match.start(2),
                        "end": match.end(2),
                        "label": "ORG",
                        "confidence": 0.8
                    })
        
        # 4. Additional PROJECT detection (all caps, acronyms, quoted names, lowercase)
        project_patterns = [
            r'\b([A-Z]{2,}(?:[-_][A-Z]+)*)\b',  # Acronyms like SMARTIN, WP-1, ICL
            r'\b([a-z]{2,}(?:[-_][a-z]+)*)\b',  # Lowercase acronyms (like "icl")
            r'"([A-Z][a-zA-Z\s]+)"',  # Quoted project names
            r"'([A-Z][a-zA-Z\s]+)'",  # Single-quoted project names
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+(?:project|initiative|program|system|platform)\b',
        ]
        for pattern in project_patterns:
            for match in re.finditer(pattern, text):
                project_name = match.group(1)
                if project_name and len(project_name) > 1:  # Allow 2+ char acronyms
                    project_name = project_name.upper() if project_name.islower() and len(project_name) <= 5 else project_name
                    entity_key = (project_name.lower(), "PROJECT")
                    if entity_key not in seen_entities:
                        seen_entities.add(entity_key)
                        entities["PROJECT"].append({
                            "text": project_name,
                            "start": match.start(),
                            "end": match.end(),
                            "label": "PROJECT",
                            "confidence": 0.7
                        })
        
        # 5. Multi-word organization names (case-insensitive, more flexible)
        org_patterns = [
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,})\s+(?:Inc|Ltd|Corp|LLC|GmbH|AG|SA|Co|Company|Corporation|Foundation|Group|Systems|Technologies|Solutions)\b',
            r'\b([a-z]+(?:\s+[a-z]+){1,})\s+(?:inc|ltd|corp|llc|gmbh|ag|sa|co|company|corporation|foundation|group|systems|technologies|solutions)\b',  # Lowercase
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+(?:Laboratory|Lab|Research|Center|Centre|Institute|Institution)\b',
            r'\b([a-z]+(?:\s+[a-z]+)+)\s+(?:laboratory|lab|research|center|centre|institute|institution)\b',  # Lowercase
        ]
        for pattern in org_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                org_name = match.group(1).strip()
                if org_name and len(org_name) > 2:
                    org_name = org_name.title() if org_name.islower() else org_name
                    entity_key = (org_name.lower(), "ORG")
                    if entity_key not in seen_entities:
                        seen_entities.add(entity_key)
                        entities["ORG"].append({
                            "text": org_name,
                            "start": match.start(),
                            "end": match.end(),
                            "label": "ORG",
                            "confidence": 0.7
                        })
        
        # 6. Common nouns that might be entities in context (university, college, etc.)
        # These are added as CONCEPT entities
        concept_keywords = ['university', 'college', 'institute', 'institution', 'academy', 'school', 
                           'company', 'corporation', 'organization', 'foundation', 'association']
        for keyword in concept_keywords:
            pattern = rf'\b({keyword})\b'
            for match in re.finditer(pattern, text, re.IGNORECASE):
                concept = match.group(1)
                entity_key = (concept.lower(), "CONCEPT")
                if entity_key not in seen_entities:
                    seen_entities.add(entity_key)
                    entities["CONCEPT"].append({
                        "text": concept,
                        "start": match.start(),
                        "end": match.end(),
                        "label": "CONCEPT",
                        "confidence": 0.5
                    })
    else:
        # Fallback to regex-based NER
        entities = extract_entities_regex(text, sentences)
    
    return entities


def map_spacy_label(spacy_label: str) -> Optional[str]:
    """Map spaCy labels to our entity types - ENHANCED"""
    mapping = {
        "ORG": "ORG",
        "PERSON": "PERSON",
        "GPE": "LOCATION",  # Geopolitical entity (countries, cities, states)
        "LOC": "LOCATION",  # Non-GPE locations
        "DATE": "DATE",
        "TIME": "DATE",
        "MONEY": "CONCEPT",
        "PERCENT": "CONCEPT",
        "EVENT": "CONCEPT",
        "PRODUCT": "CONCEPT",
        "LAW": "CONCEPT",
        "FAC": "LOCATION",  # Buildings, airports, highways, bridges, etc.
        "NORP": "ORG",  # Nationalities or religious or political groups
        "WORK_OF_ART": "CONCEPT",
        "LANGUAGE": "CONCEPT"
    }
    return mapping.get(spacy_label, "CONCEPT")


def extract_entities_regex(text: str, sentences: List[str]) -> Dict[str, List[Dict]]:
    """Regex-based NER fallback"""
    entities = {
        "ORG": [],
        "PERSON": [],
        "LOCATION": [],
        "PROJECT": [],
        "DATE": [],
        "CONCEPT": []
    }
    
    # Organization patterns (capitalized words, often with Inc, Ltd, Corp)
    org_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:Inc|Ltd|Corp|LLC|GmbH|AG|SA|Co|Company|Corporation|University|Institute|Foundation))?)\b'
    for match in re.finditer(org_pattern, text):
        entities["ORG"].append({
            "text": match.group(1),
            "start": match.start(),
            "end": match.end(),
            "label": "ORG",
            "confidence": 0.7
        })
    
    # Date patterns
    date_pattern = r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b'
    for match in re.finditer(date_pattern, text, re.IGNORECASE):
        entities["DATE"].append({
            "text": match.group(1),
            "start": match.start(),
            "end": match.end(),
            "label": "DATE",
            "confidence": 0.8
        })
    
    # Location patterns (capitalized place names)
    location_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Street|Avenue|Road|Boulevard|Lane|Drive|City|State|Country|Kingdom|Republic))\b'
    for match in re.finditer(location_pattern, text):
        entities["LOCATION"].append({
            "text": match.group(1),
            "start": match.start(),
            "end": match.end(),
            "label": "LOCATION",
            "confidence": 0.7
        })
    
    # ENHANCED: Project names (often in quotes or all caps, including lowercase)
    project_patterns = [
        r'\b([A-Z]{2,}(?:[-_][A-Z]+)*)\b',  # Acronyms
        r'\b([a-z]{2,}(?:[-_][a-z]+)*)\b',  # Lowercase acronyms (like "icl")
        r'"([A-Z][a-zA-Z\s]+)"',  # Quoted
        r"'([A-Z][a-zA-Z\s]+)'",  # Single-quoted
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+(?:project|initiative|program)\b',
    ]
    for pattern in project_patterns:
        for match in re.finditer(pattern, text):
            project_name = match.group(1)
            if project_name and len(project_name) > 1:  # Allow 2+ char
                entity_key = (project_name.lower(), "PROJECT")
                if entity_key not in seen_entities:
                    seen_entities.add(entity_key)
                    entities["PROJECT"].append({
                        "text": project_name,
                        "start": match.start(),
                        "end": match.end(),
                        "label": "PROJECT",
                        "confidence": 0.6
                    })
    
    # ENHANCED: Add titles and honorifics as PERSON entities
    title_patterns = [
        r'\b(Queen|King|Prince|Princess|President|Prime\s+Minister|Chancellor|Minister|Doctor|Dr\.|Professor|Prof\.|Mr\.|Mrs\.|Ms\.|Sir|Dame|Lord|Lady)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)?',
        r'\b(Queen|King|Prince|Princess|President|Prime\s+Minister|Chancellor|Minister|Doctor|Dr\.|Professor|Prof\.)\b',
    ]
    for pattern in title_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            title = match.group(1).strip()
            if len(match.groups()) > 1 and match.group(2):
                full_name = f"{title} {match.group(2)}"
            else:
                full_name = title
            
            entity_key = (full_name.lower(), "PERSON")
            if entity_key not in seen_entities:
                seen_entities.add(entity_key)
                entities["PERSON"].append({
                    "text": full_name,
                    "start": match.start(),
                    "end": match.end(),
                    "label": "PERSON",
                    "confidence": 0.7
                })
    
    return entities


# ============================================================================
# STEP 3: COREFERENCE RESOLUTION
# ============================================================================

def resolve_coreferences(text: str, sentences: List[str], entities: Dict[str, List[Dict]]) -> Dict[str, str]:
    """
    Step 3: Unify mentions referring to the same entity using neuralcoref.
    
    Maps pronouns and references to their canonical entity names.
    Example: "it" → "INLECOM", "the company" → "INLECOM"
    
    Returns:
        Dict mapping references to canonical entity names
    """
    coref_map = {}
    
    # Build entity name list (prioritize ORG and PERSON)
    entity_names = []
    for entity_type in ["ORG", "PERSON", "PROJECT"]:
        for ent in entities.get(entity_type, []):
            entity_names.append(ent["text"])
    
    # Try neuralcoref if available
    if NEURALCOREF_AVAILABLE and SPACY_AVAILABLE and nlp:
        try:
            # Add neuralcoref to pipeline if not already added
            if 'neuralcoref' not in nlp.pipe_names:
                neuralcoref.add_to_pipe(nlp)
            
            # Process text with coreference resolution
            doc = nlp(text)
            
            # Get coreference clusters
            if hasattr(doc, '_') and hasattr(doc._, 'coref_clusters'):
                clusters = doc._.coref_clusters
                
                for cluster in clusters:
                    # Get the main mention (usually the first or longest)
                    main_mention = cluster.main.text
                    
                    # Map all mentions in cluster to main mention
                    for mention in cluster.mentions:
                        mention_text = mention.text.lower()
                        # Only map if main mention is an entity we know
                        if main_mention in entity_names:
                            coref_map[mention_text] = main_mention
                            # Also map variations
                            if mention_text != main_mention.lower():
                                coref_map[main_mention.lower()] = main_mention
        except Exception as e:
            print(f"⚠️  neuralcoref failed: {e}, using fallback")
            # Fall back to rule-based
            coref_map = resolve_coreferences_rule_based(sentences, entities, entity_names)
    else:
        # Fallback to rule-based coreference resolution
        coref_map = resolve_coreferences_rule_based(sentences, entities, entity_names)
    
    return coref_map


def resolve_coreferences_rule_based(sentences: List[str], entities: Dict[str, List[Dict]], 
                                   entity_names: List[str]) -> Dict[str, str]:
    """Rule-based coreference resolution fallback"""
    coref_map = {}
    
    # Common reference patterns
    reference_patterns = [
        (r'\b(it|this|that|which)\b', entity_names),  # Pronouns
        (r'\b(the|this|that)\s+(company|organization|institution|project|system)\b', entity_names),
        (r'\b(the|this|that)\s+([A-Z][a-z]+)\b', entity_names),  # "the Project", "the System"
    ]
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        
        # Find pronouns and map to most recent entity
        for pattern, candidates in reference_patterns:
            matches = re.finditer(pattern, sentence_lower, re.IGNORECASE)
            for match in matches:
                ref_text = match.group(0).lower()
                
                # Find the most likely entity (simple heuristic: closest match)
                if candidates:
                    # Use the first entity as default (can be improved)
                    coref_map[ref_text] = candidates[0] if candidates else None
    
    # Also create aliases for common variations
    for entity_name in entity_names:
        # Create lowercase alias
        coref_map[entity_name.lower()] = entity_name
        # Create alias without common suffixes
        if entity_name.endswith(" Inc") or entity_name.endswith(" Ltd"):
            base_name = entity_name.rsplit(" ", 1)[0]
            coref_map[base_name.lower()] = entity_name
    
    return coref_map


# ============================================================================
# STEP 4: RELATION EXTRACTION (RE)
# ============================================================================

def extract_relations(text: str, sentences: List[str], entities: Dict[str, List[Dict]], 
                     coref_map: Dict[str, str], pos_tags: List) -> List[Dict]:
    """
    Step 4: Identify how entities are connected (relationships/predicates).
    
    Uses multiple methods in priority order:
    1. OpenNRE (fast neural relation extraction)
    2. REL (comprehensive relation extraction)
    3. spaCy Matcher/DependencyMatcher (rule-based, very effective)
    4. spaCy dependency parsing (fallback)
    5. Pattern matching (final fallback)
    
    Detects relationships like:
    - collaborates_with, works_with, partners_with
    - located_in, based_in
    - creates, develops, implements
    - requires, needs, uses
    
    Returns:
        List of relation dictionaries with subject, predicate, object
    """
    # Define verb_indicators and invalid_entities at function scope (used in multiple places)
    invalid_entities = {'named', 'is', 'are', 'was', 'were', 'has', 'have', 'had', 
                      'the', 'a', 'an', 'this', 'that', 'these', 'those'}
    verb_indicators = {'named', 'is', 'are', 'was', 'were', 'has', 'have', 'had', 
                      'becomes', 'represents', 'means', 'refers', 'denotes', 'equals'}
    
    relations = []
    seen_relations = set()  # Track duplicates across methods
    
    # Extract entity names for matching
    entity_names = set()
    entity_positions = {}  # entity -> (start, end) for OpenNRE/REL
    for entity_type in ["ORG", "PERSON", "PROJECT", "LOCATION"]:
        for ent in entities.get(entity_type, []):
            entity_names.add(ent["text"])
            entity_names.add(ent["text"].lower())
            entity_positions[ent["text"]] = (ent.get("start", 0), ent.get("end", 0))
    
    # Method 1: Try OpenNRE (fast neural relation extraction)
    if OPENNRE_AVAILABLE and opennre_model:
        try:
            for sentence in sentences:
                if len(sentence.strip()) < 10:
                    continue
                
                # Find entity pairs in sentence
                sentence_entities = []
                for entity_type in ["ORG", "PERSON", "PROJECT", "LOCATION"]:
                    for ent in entities.get(entity_type, []):
                        if ent["text"] in sentence or ent["text"].lower() in sentence.lower():
                            sentence_entities.append(ent)
                
                # Extract relations for all entity pairs
                for i, ent1 in enumerate(sentence_entities):
                    for ent2 in sentence_entities[i+1:]:
                        # OpenNRE expects (head, tail, text) format
                        try:
                            result = opennre_model.infer({
                                'text': sentence,
                                'h': {'pos': [ent1.get("start", 0), ent1.get("end", 0)]},
                                't': {'pos': [ent2.get("start", 0), ent2.get("end", 0)]}
                            })
                            
                            if result and len(result) > 0:
                                relation_type = result[0] if isinstance(result, list) else result
                                if relation_type and relation_type != "no_relation":
                                    rel_key = (ent1["text"].lower(), relation_type, ent2["text"].lower())
                                    if rel_key not in seen_relations:
                                        seen_relations.add(rel_key)
                                        relations.append({
                                            "subject": ent1["text"],
                                            "predicate": relation_type,
                                            "object": ent2["text"],
                                            "sentence": sentence,
                                            "confidence": 0.9
                                        })
                        except Exception as e:
                            # OpenNRE failed for this pair, continue
                            pass
        except Exception as e:
            print(f"⚠️  OpenNRE extraction failed: {e}")
    
    # Method 2: Try REL (comprehensive relation extraction)
    if REL_AVAILABLE:
        try:
            # REL requires more setup, so we'll use it as a secondary method
            # For now, we'll skip REL as it requires more complex setup
            pass
        except Exception as e:
            print(f"⚠️  REL extraction failed: {e}")
    
    # Method 3: Use spaCy Matcher and DependencyMatcher (very effective rule-based)
    if SPACY_AVAILABLE and nlp:
        try:
            from spacy.matcher import Matcher
            
            # Create Matcher for pattern-based extraction
            matcher = Matcher(nlp.vocab)
            
            # Define relation patterns using spaCy Matcher
            # Pattern: [entity] [verb] [entity]
            collaboration_pattern = [
                {"ENT_TYPE": {"IN": ["ORG", "PERSON", "PROJECT"]}, "OP": "+"},
                {"LEMMA": {"IN": ["collaborate", "work", "partner", "cooperate"]}},
                {"LOWER": "with"},
                {"ENT_TYPE": {"IN": ["ORG", "PERSON", "PROJECT"]}, "OP": "+"}
            ]
            
            creates_pattern = [
                {"ENT_TYPE": {"IN": ["ORG", "PERSON", "PROJECT"]}, "OP": "+"},
                {"LEMMA": {"IN": ["create", "develop", "design", "build", "implement"]}},
                {"ENT_TYPE": {"IN": ["ORG", "PERSON", "PROJECT", "CONCEPT"]}, "OP": "+"}
            ]
            
            uses_pattern = [
                {"ENT_TYPE": {"IN": ["ORG", "PERSON", "PROJECT"]}, "OP": "+"},
                {"LEMMA": {"IN": ["use", "employ", "utilize", "apply"]}},
                {"ENT_TYPE": {"IN": ["ORG", "PERSON", "PROJECT", "CONCEPT"]}, "OP": "+"}
            ]
            
            matcher.add("COLLABORATION", [collaboration_pattern])
            matcher.add("CREATES", [creates_pattern])
            matcher.add("USES", [uses_pattern])
            
            # Process sentences with Matcher
            for sentence in sentences:
                if len(sentence.strip()) < 10:
                    continue
                
                # Resolve coreferences
                resolved_sentence = sentence
                for ref, entity in coref_map.items():
                    if ref.lower() in resolved_sentence.lower():
                        resolved_sentence = re.sub(r'\b' + re.escape(ref) + r'\b', entity, resolved_sentence, flags=re.IGNORECASE)
                
                doc = nlp(resolved_sentence)
                matches = matcher(doc)
                
                for match_id, start, end in matches:
                    label = nlp.vocab.strings[match_id]
                    span = doc[start:end]
                    
                    # Extract entities from the matched span
                    ents_in_span = [ent for ent in doc.ents if ent.start < end and ent.end > start]
                    if len(ents_in_span) >= 2:
                        subject = ents_in_span[0].text
                        object_ent = ents_in_span[1].text
                        
                        rel_key = (subject.lower(), label.lower(), object_ent.lower())
                        if rel_key not in seen_relations:
                            seen_relations.add(rel_key)
                            relations.append({
                                "subject": subject,
                                "predicate": label.lower(),
                                "object": object_ent,
                                "sentence": sentence,
                                "confidence": 0.85
                            })
        except Exception as e:
            print(f"⚠️  spaCy Matcher failed: {e}, using dependency parsing")
    
    # Method 4: Use spaCy dependency parsing if available
    if SPACY_AVAILABLE and nlp:
        # Process each sentence with spaCy
        for sentence in sentences:
            if len(sentence.strip()) < 10:
                continue
            
            # Resolve coreferences
            resolved_sentence = sentence
            for ref, entity in coref_map.items():
                if ref.lower() in resolved_sentence.lower():
                    resolved_sentence = re.sub(r'\b' + re.escape(ref) + r'\b', entity, resolved_sentence, flags=re.IGNORECASE)
            
            doc = nlp(resolved_sentence)
            
            # Extract relations using dependency parsing
            for token in doc:
                # Look for verbs (potential predicates)
                if token.pos_ == "VERB" or token.tag_.startswith("VB"):
                    predicate = token.lemma_.lower()
                    
                    # Find subject (nsubj dependency)
                    subject_tokens = [child for child in token.children if child.dep_ == "nsubj"]
                    # Find object (dobj, pobj dependencies)
                    object_tokens = [child for child in token.children if child.dep_ in ["dobj", "pobj", "attr"]]
                    
                    # Also check for prepositional objects
                    for prep in token.children:
                        if prep.dep_ == "prep":
                            object_tokens.extend([child for child in prep.children if child.dep_ == "pobj"])
                    
                    # Extract subject and object phrases
                    for subj_token in subject_tokens:
                        for obj_token in object_tokens:
                            # Get full noun phrases
                            subject_phrase = get_noun_phrase(subj_token)
                            object_phrase = get_noun_phrase(obj_token)
                            
                            # Check if they match known entities or are capitalized (likely entities)
                            if (subject_phrase in entity_names or 
                                (subject_phrase and subject_phrase[0].isupper() and len(subject_phrase) > 2)):
                                if (object_phrase in entity_names or 
                                    (object_phrase and object_phrase[0].isupper() and len(object_phrase) > 2)):
                                    
                                    # Map predicate to canonical form
                                    canonical_predicate = map_predicate(predicate)
                                    
                                    # Clean entities
                                    subject_clean = clean_entity_local(subject_phrase)
                                    object_clean = clean_entity_local(object_phrase)
                                    
                                    # Use extract_core_entity
                                    try:
                                        subject_core, _ = extract_core_entity(subject_clean, sentence)
                                        object_core, _ = extract_core_entity(object_clean, sentence)
                                        subject_clean = subject_core
                                        object_clean = object_core
                                    except:
                                        pass
                                    
                                    if len(subject_clean) > 2 and len(object_clean) > 2:
                                        relations.append({
                                            "subject": subject_clean,
                                            "predicate": canonical_predicate,
                                            "object": object_clean,
                                            "sentence": sentence,
                                            "confidence": 0.8
                                        })
    
    # Also use pattern-based extraction for additional relations
    # EXPANDED: Many more patterns to catch more facts
    relation_patterns = [
        # Collaboration patterns (case-insensitive, more variations)
        (r'([A-Za-z][a-zA-Z0-9\s\-]+)\s+(?:collaborates?\s+with|works?\s+with|partners?\s+with|cooperates?\s+with|teams?\s+up\s+with)\s+([A-Za-z][a-zA-Z0-9\s\-]+)', 'collaborates_with'),
        (r'([A-Za-z][a-zA-Z0-9\s\-]+)\s+(?:and|&)\s+([A-Za-z][a-zA-Z0-9\s\-]+)\s+(?:collaborate|work|partner|cooperate)', 'collaborates_with'),
        (r'([A-Za-z][a-zA-Z0-9\s\-]+)\s+(?:in\s+)?(?:collaboration|partnership|cooperation)\s+(?:with|between)\s+([A-Za-z][a-zA-Z0-9\s\-]+)', 'collaborates_with'),
        
        # Location patterns (expanded)
        (r'([A-Za-z][a-zA-Z0-9\s\-]+)\s+(?:is\s+)?(?:located\s+in|based\s+in|situated\s+in|found\s+in|resides\s+in)\s+([A-Za-z][a-zA-Z0-9\s\-]+)', 'located_in'),
        (r'([A-Za-z][a-zA-Z0-9\s\-]+)\s+(?:is\s+from|from|originates\s+from|comes\s+from)\s+([A-Za-z][a-zA-Z0-9\s\-]+)', 'is_from'),  # ENHANCED: "Queen is from Great Britain"
        (r'([A-Za-z][a-zA-Z0-9\s\-]+)\s+(?:was|were)\s+from\s+([A-Za-z][a-zA-Z0-9\s\-]+)', 'is_from'),  # Past tense
        
        # Creation/Development patterns (expanded)
        (r'([A-Za-z][a-zA-Z0-9\s\-]+)\s+(?:creates?|develops?|designed?|built|implemented?|established?|founded?|launched?)\s+([A-Za-z][a-zA-Z0-9\s\-]+)', 'creates'),
        (r'([A-Za-z][a-zA-Z0-9\s\-]+)\s+(?:was\s+)?(?:created|developed|designed|built|implemented|established|founded|launched)\s+(?:by|at)\s+([A-Za-z][a-zA-Z0-9\s\-]+)', 'creates'),  # Passive voice
        
        # Usage patterns (expanded)
        (r'([A-Za-z][a-zA-Z0-9\s\-]+)\s+(?:uses?|employs?|utilizes?|applies?|adopts?|leverages?)\s+([A-Za-z][a-zA-Z0-9\s\-]+)', 'uses'),
        (r'([A-Za-z][a-zA-Z0-9\s\-]+)\s+(?:is\s+)?(?:used|employed|utilized|applied|adopted|leveraged)\s+(?:by|in)\s+([A-Za-z][a-zA-Z0-9\s\-]+)', 'uses'),  # Passive voice
        
        # Requirement patterns (expanded)
        (r'([A-Za-z][a-zA-Z0-9\s\-]+)\s+(?:requires?|needs?|demands?|necessitates?)\s+([A-Za-z][a-zA-Z0-9\s\-]+)', 'requires'),
        (r'([A-Za-z][a-zA-Z0-9\s\-]+)\s+(?:is\s+)?(?:required|needed|demanded|necessary)\s+(?:by|for)\s+([A-Za-z][a-zA-Z0-9\s\-]+)', 'requires'),  # Passive voice
        
        # ENHANCED: Compound predicates with "is/was/has/have" - MUST come BEFORE simple "is" patterns
        # "is from" patterns (high priority)
        (r'([A-Za-z][a-zA-Z0-9\s\-]+)\s+(?:is|are|was|were)\s+from\s+([A-Za-z][a-zA-Z0-9\s\-]+)', 'is_from'),
        # "is named" patterns (high priority) - MUST match before general "is" pattern
        # Pattern: "X is named Y" -> (X, is_named, Y)
        # ENHANCED: More flexible pattern to match "Queen is named Elizabeth"
        # Match "X is named Y" where X and Y can be single words or phrases
        # Use word boundaries and allow for proper nouns (capitalized)
        (r'\b([A-Z][a-zA-Z0-9\s\-]*?|[A-Za-z][a-zA-Z0-9\s\-]+?)\s+(?:is|are|was|were)\s+named\s+([A-Z][a-zA-Z0-9\s\-]*?|[A-Za-z][a-zA-Z0-9\s\-]+?)(?:\s|$|[.,!?;])', 'is_named'),
        # "is in" patterns (high priority)
        (r'([A-Za-z][a-zA-Z0-9\s\-]+)\s+(?:is|are|was|were)\s+in\s+([A-Za-z][a-zA-Z0-9\s\-]+)', 'is_in'),
        # "is happening/coming/going" patterns (high priority - avoid extracting these as simple "is")
        (r'([A-Za-z][a-zA-Z0-9\s\-]+)\s+(?:is|are|was|were)\s+(?:happening|coming|going|starting|ending|beginning|finishing)\s+([A-Za-z][a-zA-Z0-9\s\-]+)', 'is_happening'),
        # "has/have" patterns (high priority)
        (r'([A-Za-z][a-zA-Z0-9\s\-]+)\s+(?:has|have|had)\s+([A-Za-z][a-zA-Z0-9\s\-]+)', 'has'),
        # "has been/have been" patterns
        (r'([A-Za-z][a-zA-Z0-9\s\-]+)\s+(?:has|have|had)\s+been\s+([A-Za-z][a-zA-Z0-9\s\-]+)', 'has_been'),
        
        # Correlation patterns (high priority - statistical facts)
        (r'[Cc]orrelation\s+between\s+([A-Za-z][a-zA-Z0-9\s\-]+)\s+and\s+([A-Za-z][a-zA-Z0-9\s\-]+)\s+is\s+([\-\d\.]+)', 'has_correlation'),
        (r'[Tt]he\s+correlation\s+between\s+([A-Za-z][a-zA-Z0-9\s\-]+)\s+and\s+([A-Za-z][a-zA-Z0-9\s\-]+)\s+is\s+([\-\d\.]+)', 'has_correlation'),
        (r'([A-Za-z][a-zA-Z0-9\s\-]+)\s+and\s+([A-Za-z][a-zA-Z0-9\s\-]+)\s+have\s+correlation\s+of\s+([\-\d\.]+)', 'has_correlation'),
        (r'[Cc]orrelation\s+of\s+([A-Za-z][a-zA-Z0-9\s\-]+)\s+with\s+([A-Za-z][a-zA-Z0-9\s\-]+)\s+is\s+([\-\d\.]+)', 'has_correlation'),
        (r'([A-Za-z][a-zA-Z0-9\s\-]+)\s+correlates\s+with\s+([A-Za-z][a-zA-Z0-9\s\-]+)\s+at\s+([\-\d\.]+)', 'has_correlation'),
        
        # General "is" patterns (LOW priority - only if no compound predicate matched)
        # ENHANCED: Exclude "is named" and "is from" to avoid matching them as simple "is"
        (r'([A-Za-z][a-zA-Z0-9\s\-]+)\s+(?:is|are|was|were|becomes?|represents?|means?|refers?\s+to|denotes?|equals?|stands?\s+for)\s+(?!named\s)(?!from\s)([A-Za-z][a-zA-Z0-9\s\-]+)', 'is'),
        (r'([A-Za-z][a-zA-Z0-9\s\-]+)\s+(?:is|are|was|were)\s+(?:a|an|the)\s+([A-Za-z][a-zA-Z0-9\s\-]+)', 'is'),
        
        # Has/contains patterns (expanded)
        (r'([A-Za-z][a-zA-Z0-9\s\-]+)\s+(?:has|have|contains?|includes?|consists?\s+of|comprises?)\s+([A-Za-z][a-zA-Z0-9\s\-]+)', 'has'),
        (r'([A-Za-z][a-zA-Z0-9\s\-]+)\s+(?:is\s+)?(?:part\s+of|member\s+of|component\s+of)\s+([A-Za-z][a-zA-Z0-9\s\-]+)', 'has'),
        
        # Studies/analyzes patterns (expanded)
        (r'([A-Za-z][a-zA-Z0-9\s\-]+)\s+(?:studies?|analyzes?|examines?|investigates?|researches?|explores?)\s+([A-Za-z][a-zA-Z0-9\s\-]+)', 'studies'),
        (r'([A-Za-z][a-zA-Z0-9\s\-]+)\s+(?:focuses?\s+on|concentrates?\s+on|centers?\s+on)\s+([A-Za-z][a-zA-Z0-9\s\-]+)', 'studies'),
        
        # Proposes/suggests patterns (expanded)
        (r'([A-Za-z][a-zA-Z0-9\s\-]+)\s+(?:proposes?|suggests?|recommends?|advocates?|promotes?)\s+([A-Za-z][a-zA-Z0-9\s\-]+)', 'proposes'),
        
        # Enables/allows patterns
        (r'([A-Za-z][a-zA-Z0-9\s\-]+)\s+(?:enables?|allows?|permits?|facilitates?|supports?)\s+([A-Za-z][a-zA-Z0-9\s\-]+)', 'enables'),
        
        # Affects/impacts patterns
        (r'([A-Za-z][a-zA-Z0-9\s\-]+)\s+(?:affects?|impacts?|influences?|changes?|modifies?)\s+([A-Za-z][a-zA-Z0-9\s\-]+)', 'affects'),
        
        # Discovers/finds patterns
        (r'([A-Za-z][a-zA-Z0-9\s\-]+)\s+(?:discovered?|found|identified?|detected?|observed?|noticed?)\s+([A-Za-z][a-zA-Z0-9\s\-]+)', 'discovered'),
        
        # Causes/results patterns
        (r'([A-Za-z][a-zA-Z0-9\s\-]+)\s+(?:causes?|results?\s+in|leads?\s+to|triggers?|produces?)\s+([A-Za-z][a-zA-Z0-9\s\-]+)', 'causes'),
        (r'([A-Za-z][a-zA-Z0-9\s\-]+)\s+(?:is\s+)?(?:caused\s+by|result\s+of|due\s+to|because\s+of)\s+([A-Za-z][a-zA-Z0-9\s\-]+)', 'causes'),  # Passive voice
        
        # Manages/controls patterns
        (r'([A-Za-z][a-zA-Z0-9\s\-]+)\s+(?:manages?|controls?|oversees?|supervises?|directs?)\s+([A-Za-z][a-zA-Z0-9\s\-]+)', 'manages'),
        
        # Provides/supplies patterns
        (r'([A-Za-z][a-zA-Z0-9\s\-]+)\s+(?:provides?|supplies?|offers?|delivers?|gives?)\s+([A-Za-z][a-zA-Z0-9\s\-]+)', 'provides'),
        
        # Receives/gets patterns
        (r'([A-Za-z][a-zA-Z0-9\s\-]+)\s+(?:receives?|gets?|obtains?|acquires?|gains?)\s+([A-Za-z][a-zA-Z0-9\s\-]+)', 'receives'),
        
        # Starts/begins patterns
        (r'([A-Za-z][a-zA-Z0-9\s\-]+)\s+(?:starts?|begins?|commences?|initiates?|launches?)\s+(?:in|on|at|with)\s+([A-Za-z][a-zA-Z0-9\s\-]+)', 'starts'),
        (r'([A-Za-z][a-zA-Z0-9\s\-]+)\s+(?:started|began|commenced|initiated|launched)\s+(?:in|on|at|with)\s+([A-Za-z][a-zA-Z0-9\s\-]+)', 'starts'),
        
        # Ends/finishes patterns
        (r'([A-Za-z][a-zA-Z0-9\s\-]+)\s+(?:ends?|finishes?|concludes?|completes?|terminates?)\s+(?:in|on|at|with)\s+([A-Za-z][a-zA-Z0-9\s\-]+)', 'ends'),
        
        # Participates/joins patterns
        (r'([A-Za-z][a-zA-Z0-9\s\-]+)\s+(?:participates?\s+in|joins?|takes?\s+part\s+in|engages?\s+in)\s+([A-Za-z][a-zA-Z0-9\s\-]+)', 'participates_in'),
        
        # Owns/possesses patterns
        (r'([A-Za-z][a-zA-Z0-9\s\-]+)\s+(?:owns?|possesses?|holds?|maintains?)\s+([A-Za-z][a-zA-Z0-9\s\-]+)', 'owns'),
        
        # Reports to patterns
        (r'([A-Za-z][a-zA-Z0-9\s\-]+)\s+(?:reports?\s+to|answers?\s+to|belongs?\s+to)\s+([A-Za-z][a-zA-Z0-9\s\-]+)', 'reports_to'),
        
        # Follows/comes after patterns
        (r'([A-Za-z][a-zA-Z0-9\s\-]+)\s+(?:follows?|comes?\s+after|succeeds?|precedes?)\s+([A-Za-z][a-zA-Z0-9\s\-]+)', 'follows'),
        
        # Related to patterns (catch-all for entities in same sentence)
        (r'([A-Za-z][a-zA-Z0-9\s\-]+)\s+(?:related\s+to|connected\s+to|associated\s+with|linked\s+to)\s+([A-Za-z][a-zA-Z0-9\s\-]+)', 'related_to'),
        
        # Comparison patterns
        (r'([A-Za-z][a-zA-Z0-9\s\-]+)\s+(?:similar\s+to|like|resembles?|comparable\s+to)\s+([A-Za-z][a-zA-Z0-9\s\-]+)', 'similar_to'),
        (r'([A-Za-z][a-zA-Z0-9\s\-]+)\s+(?:different\s+from|unlike|distinct\s+from)\s+([A-Za-z][a-zA-Z0-9\s\-]+)', 'different_from'),
        
        # Ownership/possession patterns
        (r'([A-Za-z][a-zA-Z0-9\s\-]+)\s+(?:owns?|possesses?|has\s+ownership\s+of)\s+([A-Za-z][a-zA-Z0-9\s\-]+)', 'owns'),
        (r'([A-Za-z][a-zA-Z0-9\s\-]+)\s+(?:belongs?\s+to|is\s+owned\s+by)\s+([A-Za-z][a-zA-Z0-9\s\-]+)', 'belongs_to'),
        
        # Membership patterns
        (r'([A-Za-z][a-zA-Z0-9\s\-]+)\s+(?:is\s+)?(?:member\s+of|part\s+of|component\s+of|element\s+of)\s+([A-Za-z][a-zA-Z0-9\s\-]+)', 'member_of'),
        
        # Dependency patterns
        (r'([A-Za-z][a-zA-Z0-9\s\-]+)\s+(?:depends?\s+on|relies?\s+on|based\s+on)\s+([A-Za-z][a-zA-Z0-9\s\-]+)', 'depends_on'),
        
        # Communication patterns
        (r'([A-Za-z][a-zA-Z0-9\s\-]+)\s+(?:communicates?\s+with|contacts?|reaches?\s+out\s+to)\s+([A-Za-z][a-zA-Z0-9\s\-]+)', 'communicates_with'),
        
        # Competition patterns
        (r'([A-Za-z][a-zA-Z0-9\s\-]+)\s+(?:competes?\s+with|rivals?|opposes?)\s+([A-Za-z][a-zA-Z0-9\s\-]+)', 'competes_with'),
        
        # Support patterns
        (r'([A-Za-z][a-zA-Z0-9\s\-]+)\s+(?:supports?|backs?|endorses?|promotes?)\s+([A-Za-z][a-zA-Z0-9\s\-]+)', 'supports'),
        
        # Location/temporal patterns
        (r'([A-Za-z][a-zA-Z0-9\s\-]+)\s+(?:happens?\s+in|occurs?\s+in|takes?\s+place\s+in)\s+([A-Za-z][a-zA-Z0-9\s\-]+)', 'happens_in'),
        
        # Purpose patterns
        (r'([A-Za-z][a-zA-Z0-9\s\-]+)\s+(?:aims?\s+to|intends?\s+to|seeks?\s+to)\s+([A-Za-z][a-zA-Z0-9\s\-]+)', 'aims_to'),
        
        # Result patterns
        (r'([A-Za-z][a-zA-Z0-9\s\-]+)\s+(?:results?\s+in|leads?\s+to|brings?\s+about)\s+([A-Za-z][a-zA-Z0-9\s\-]+)', 'results_in'),
        (r'([A-Za-z][a-zA-Z0-9\s\-]+)\s+(?:prevents?|stops?|blocks?)\s+([A-Za-z][a-zA-Z0-9\s\-]+)', 'prevents'),
    ]
    
    # ENHANCED: Also extract from compound sentences (split by commas, semicolons, conjunctions)
    expanded_sentences = []
    for sentence in sentences:
        expanded_sentences.append(sentence)
        # Split compound sentences
        if ',' in sentence or ';' in sentence or ' and ' in sentence.lower() or ' or ' in sentence.lower():
            # Split by commas (but keep context)
            parts = re.split(r'[,;]', sentence)
            for part in parts:
                part = part.strip()
                if len(part) > 10:  # Only keep substantial parts
                    expanded_sentences.append(part)
            # Split by conjunctions
            conjunctions = [' and ', ' or ', ' but ', ' while ', ' whereas ']
            for conj in conjunctions:
                if conj in sentence.lower():
                    parts = re.split(conj, sentence, flags=re.IGNORECASE)
                    for part in parts:
                        part = part.strip()
                        if len(part) > 10:
                            expanded_sentences.append(part)
    
    # Use expanded sentences for pattern matching
    # ENHANCED: Stop after first match to prioritize compound predicates
    # Track which sentences have been matched to avoid entity co-occurrence creating redundant facts
    matched_sentences = set()
    
    for sentence in expanded_sentences:
        # Resolve coreferences
        resolved_sentence = sentence
        for ref, entity in coref_map.items():
            if ref.lower() in resolved_sentence.lower():
                resolved_sentence = re.sub(r'\b' + re.escape(ref) + r'\b', entity, resolved_sentence, flags=re.IGNORECASE)
        
        # Track if we found a match for this sentence (to avoid multiple matches)
        sentence_matched = False
        
        # Try each relation pattern (in order - compound predicates come first)
        for pattern, predicate in relation_patterns:
            if sentence_matched:
                break  # Stop after first match to prioritize compound predicates
            
            # For "is_named" pattern, use case-insensitive matching but prefer capitalized matches
            # For other patterns, use case-insensitive matching
            if predicate == "is_named":
                # Try case-sensitive first (for proper nouns like "Queen", "Elizabeth")
                matches = list(re.finditer(pattern, resolved_sentence))
                # If no matches, try case-insensitive
                if not matches:
                    matches = re.finditer(pattern, resolved_sentence, re.IGNORECASE)
            else:
                matches = re.finditer(pattern, resolved_sentence, re.IGNORECASE)
            for match in matches:
                # Handle correlation patterns specially (they have 3 groups: col1, col2, value)
                if predicate == "has_correlation" and match.lastindex >= 3:
                    col1 = match.group(1).strip()
                    col2 = match.group(2).strip()
                    corr_value = match.group(3).strip()
                    # Store as: (col1, has_correlation, col2) with value in details
                    subject = col1
                    object_ent = f"{col2} (correlation: {corr_value})"
                else:
                subject = match.group(1).strip()
                object_ent = match.group(2).strip()
                
                # Debug logging for "is_named" pattern (removed verbose logging)
                
                # ENHANCED: Filter out invalid entities (like "NAMED", "IS", etc.)
                # Note: invalid_entities and verb_indicators are defined at function scope above
                
                # Try to normalize entities first (abbreviations like "UK" -> "United Kingdom")
                try:
                    from knowledge import normalize_entity as kg_normalize_entity
                    subject_normalized = kg_normalize_entity(subject)
                    object_normalized = kg_normalize_entity(object_ent)
                    # Use normalized versions if they changed
                    if subject_normalized != subject:
                        subject = subject_normalized
                    if object_normalized != object_ent:
                        object_ent = object_normalized
                except:
                    pass  # If normalization fails, continue with original entities
                
                # Clean entities
                subject = clean_entity_local(subject)
                object_ent = clean_entity_local(object_ent)
                
                # ENHANCED: Remove invalid words from entities (e.g., "Queen IS" -> "Queen")
                # Split entity into words and filter out invalid words
                subject_words = subject.split()
                subject_words = [w for w in subject_words if w.lower() not in invalid_entities]
                subject = ' '.join(subject_words).strip()
                
                object_words = object_ent.split()
                object_words = [w for w in object_words if w.lower() not in invalid_entities]
                object_ent = ' '.join(object_words).strip()
                
                # Use extract_core_entity - but only for compound predicates to avoid over-extraction
                # For "is_named", "is_from", etc., the entities are already correctly extracted
                if predicate not in ["is_named", "is_from", "is_in", "has", "requires"]:
                    try:
                        subject_core, _ = extract_core_entity(subject, sentence)
                        object_core, _ = extract_core_entity(object_ent, sentence)
                        subject = subject_core
                        object_ent = object_core
                    except:
                        pass
                
                # ENHANCED: Only reject if they're invalid words (not abbreviations)
                # Don't reject just because they're all-caps and short - they might be valid abbreviations
                if (not subject or not object_ent or 
                    subject.lower() in invalid_entities or 
                    object_ent.lower() in invalid_entities):
                    continue
                
                # Validate entities (skip self-relations and very short entities)
                if len(subject) > 2 and len(object_ent) > 2:
                    # Skip self-relations
                    if subject.lower() == object_ent.lower():
                        continue
                    
                    # ENHANCED: Calculate confidence based on predicate type and entity quality
                    confidence = calculate_confidence(subject, predicate, object_ent, sentence)
                    
                    # Only add if confidence is above threshold
                    CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence to include fact
                    if confidence < CONFIDENCE_THRESHOLD:
                        continue
                    
                    # Check if not already added
                    rel_key = (subject.lower(), predicate, object_ent.lower())
                    if rel_key not in seen_relations:
                        seen_relations.add(rel_key)
                        sentence_matched = True  # Mark that we found a match for this sentence
                        matched_sentences.add(sentence)  # Mark this sentence as matched
                        relations.append({
                            "subject": subject,
                            "predicate": predicate,
                            "object": object_ent,
                            "sentence": sentence,
                            "confidence": confidence
                        })
                        break  # Stop after first match for this sentence
    
    # ADDITIONAL: Extract from lists and enumerations
    for sentence in sentences:
        # Pattern: "X, Y, and Z" -> extract relationships
        list_pattern = r'([A-Za-z][a-zA-Z0-9\s\-]+)(?:\s*,\s*([A-Za-z][a-zA-Z0-9\s\-]+))+(?:\s+and\s+([A-Za-z][a-zA-Z0-9\s\-]+))?'
        list_matches = re.finditer(list_pattern, sentence, re.IGNORECASE)
        for match in list_matches:
            items = [g for g in match.groups() if g]
            if len(items) >= 2:
                # Create "related_to" relationships between all items in the list
                for i in range(len(items)):
                    for j in range(i + 1, len(items)):
                        item1 = clean_entity_local(items[i].strip())
                        item2 = clean_entity_local(items[j].strip())
                        if len(item1) > 2 and len(item2) > 2 and item1.lower() != item2.lower():
                            rel_key = (item1.lower(), "related_to", item2.lower())
                            if rel_key not in seen_relations:
                                seen_relations.add(rel_key)
                                relations.append({
                                    "subject": item1,
                                    "predicate": "related_to",
                                    "object": item2,
                                    "sentence": sentence,
                                    "confidence": 0.4
                                })
    
    # ADDITIONAL: Extract "X of Y" and "Y's X" patterns (possessive relationships)
    for sentence in sentences:
        # "X of Y" pattern
        of_pattern = r'([A-Za-z][a-zA-Z0-9\s\-]+)\s+of\s+([A-Za-z][a-zA-Z0-9\s\-]+)'
        for match in re.finditer(of_pattern, sentence, re.IGNORECASE):
            subject = clean_entity_local(match.group(1).strip())
            object_ent = clean_entity_local(match.group(2).strip())
            if len(subject) > 2 and len(object_ent) > 2 and subject.lower() != object_ent.lower():
                rel_key = (subject.lower(), "part_of", object_ent.lower())
                if rel_key not in seen_relations:
                    seen_relations.add(rel_key)
                    relations.append({
                        "subject": subject,
                        "predicate": "part_of",
                        "object": object_ent,
                        "sentence": sentence,
                        "confidence": 0.6
                    })
        
        # "Y's X" pattern (possessive)
        possessive_pattern = r"([A-Za-z][a-zA-Z0-9\s\-]+)'s\s+([A-Za-z][a-zA-Z0-9\s\-]+)"
        for match in re.finditer(possessive_pattern, sentence):
            owner = clean_entity_local(match.group(1).strip())
            owned = clean_entity_local(match.group(2).strip())
            if len(owner) > 2 and len(owned) > 2 and owner.lower() != owned.lower():
                rel_key = (owned.lower(), "belongs_to", owner.lower())
                if rel_key not in seen_relations:
                    seen_relations.add(rel_key)
                    relations.append({
                        "subject": owned,
                        "predicate": "belongs_to",
                        "object": owner,
                        "sentence": sentence,
                        "confidence": 0.6
                    })
    
    # ADDITIONAL: Extract temporal relationships
    temporal_patterns = [
        (r'([A-Za-z][a-zA-Z0-9\s\-]+)\s+(?:before|prior\s+to|earlier\s+than)\s+([A-Za-z][a-zA-Z0-9\s\-]+)', 'before'),
        (r'([A-Za-z][a-zA-Z0-9\s\-]+)\s+(?:after|following|later\s+than)\s+([A-Za-z][a-zA-Z0-9\s\-]+)', 'after'),
        (r'([A-Za-z][a-zA-Z0-9\s\-]+)\s+(?:during|throughout|within)\s+([A-Za-z][a-zA-Z0-9\s\-]+)', 'during'),
    ]
    for sentence in sentences:
        for pattern, predicate in temporal_patterns:
            for match in re.finditer(pattern, sentence, re.IGNORECASE):
                subject = clean_entity_local(match.group(1).strip())
                object_ent = clean_entity_local(match.group(2).strip())
                if len(subject) > 2 and len(object_ent) > 2 and subject.lower() != object_ent.lower():
                    rel_key = (subject.lower(), predicate, object_ent.lower())
                    if rel_key not in seen_relations:
                        seen_relations.add(rel_key)
                        relations.append({
                            "subject": subject,
                            "predicate": predicate,
                            "object": object_ent,
                            "sentence": sentence,
                            "confidence": 0.6
                        })
    
    # Entity co-occurrence: Extract "related_to" relationships for entities that appear together
    # but only if no specific pattern matched and no specific predicate already exists
    for sentence in sentences:
        # Skip sentences that were already matched by pattern matching
        if sentence in matched_sentences:
            continue
        
        # Extract all entities from the sentence
        entities = []
        
        # Try to get entities from NER if available
        if SPACY_AVAILABLE:
            try:
                # Import nlp from module level (it's defined at the top of the file)
                import kg_pipeline
                nlp_model = getattr(kg_pipeline, 'nlp', None)
                if nlp_model:
                    doc = nlp_model(sentence)
                    for ent in doc.ents:
                        entity_text = clean_entity_local(ent.text.strip())
                        if len(entity_text) > 2:
                            entities.append(entity_text)
            except:
                pass
        
        # Also extract capitalized phrases and multi-word entities
        # Pattern: Capitalized words (e.g., "Queen", "Great Britain", "Imperial College London")
        capitalized_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        for match in re.finditer(capitalized_pattern, sentence):
            entity_text = clean_entity_local(match.group(0).strip())
            if len(entity_text) > 2 and entity_text.lower() not in [e.lower() for e in entities]:
                # Filter out common verbs and invalid entities
                if entity_text.lower() not in verb_indicators and entity_text.lower() not in invalid_entities:
                    entities.append(entity_text)
        
        # Create "related_to" relationships between entities that appear in the same sentence
        # but only if no more specific predicate already exists for this pair
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                entity1 = entities[i]
                entity2 = entities[j]
                
                # Skip if entities are the same
                if entity1.lower() == entity2.lower():
                    continue
                
                # Check if a more specific predicate already exists for this entity pair
                # Check seen_relations (which contains all predicates from pattern matching)
                has_specific_predicate = False
                for (s, p, o) in seen_relations:
                    # Check if this is the same entity pair (bidirectional)
                    if ((s == entity1.lower() and o == entity2.lower()) or 
                        (s == entity2.lower() and o == entity1.lower())):
                        # If there's already a specific predicate (not "related_to"), skip
                        if p != "related_to":
                            has_specific_predicate = True
                            break
                
                if has_specific_predicate:
                    continue  # Skip creating "related_to" if a more specific predicate exists
                
                # Check if this "related_to" fact already exists
                rel_key = (entity1.lower(), "related_to", entity2.lower())
                if rel_key not in seen_relations:
                    # Additional validation: filter out trivial relations
                    # Don't create "related_to" if either entity contains common verbs
                    if any(verb in entity1.lower() for verb in verb_indicators):
                        continue
                    if any(verb in entity2.lower() for verb in verb_indicators):
                        continue
                    
                    # Don't create "related_to" if entities are too short or invalid
                    if entity1.lower() in invalid_entities or entity2.lower() in invalid_entities:
                        continue
                    
                    seen_relations.add(rel_key)
                    relations.append({
                        "subject": entity1,
                        "predicate": "related_to",
                        "object": entity2,
                        "sentence": sentence,
                        "confidence": 0.3  # Lower confidence for co-occurrence
                    })
    
    return relations


def get_noun_phrase(token) -> str:
    """Extract full noun phrase from a token using dependency tree"""
    if not token:
        return ""
    
    # Start with the token itself
    phrase_tokens = [token]
    
    # Add determiners, adjectives, and other modifiers
    for child in token.children:
        if child.dep_ in ["det", "amod", "compound", "nmod"]:
            phrase_tokens.append(child)
    
    # Sort by position in sentence
    phrase_tokens.sort(key=lambda t: t.i)
    
    # Extract text
    phrase = " ".join([t.text for t in phrase_tokens])
    return phrase.strip()


def map_predicate(verb: str) -> str:
    """Map verb to canonical predicate form"""
    predicate_map = {
        "collaborate": "collaborates_with",
        "work": "works_with",
        "partner": "partners_with",
        "cooperate": "collaborates_with",
        "create": "creates",
        "develop": "develops",
        "design": "designs",
        "build": "creates",
        "implement": "implements",
        "use": "uses",
        "require": "requires",
        "need": "requires",
        "study": "studies",
        "analyze": "studies",
        "propose": "proposes",
        "suggest": "proposes",
        "be": "is",
        "have": "has",
        "contain": "has",
        "include": "has",
    }
    return predicate_map.get(verb, verb)


def clean_entity_local(entity: str) -> str:
    """Clean entity name (remove articles, extra words) - uses knowledge module if available"""
    try:
        return kg_clean_entity(entity)
    except:
        # Fallback cleaning
        entity = re.sub(r'^(the|a|an)\s+', '', entity, flags=re.IGNORECASE)
        entity = entity.rstrip('.,;:!?')
        return entity.strip()

def calculate_confidence(subject: str, predicate: str, object_ent: str, sentence: str) -> float:
    """
    Calculate confidence score for a fact based on:
    - Predicate type (compound predicates have higher confidence)
    - Entity quality (capitalized, recognized entities have higher confidence)
    - Sentence structure (clear subject-verb-object patterns have higher confidence)
    
    Returns:
        Confidence score between 0.0 and 1.0
    """
    confidence = 0.5  # Base confidence
    
    # Higher confidence for compound predicates (is_from, is_named, etc.)
    compound_predicates = ['is_from', 'is_named', 'is_in', 'located_in', 'collaborates_with', 
                          'works_with', 'creates', 'develops', 'uses', 'requires']
    if predicate in compound_predicates:
        confidence += 0.2
    
    # Lower confidence for generic "is" or "related_to"
    if predicate == 'is':
        confidence -= 0.2
    elif predicate == 'related_to':
        confidence -= 0.1
    
    # Higher confidence for capitalized entities (likely proper nouns)
    if subject and subject[0].isupper() and len(subject) > 2:
        confidence += 0.1
    if object_ent and object_ent[0].isupper() and len(object_ent) > 2:
        confidence += 0.1
    
    # Lower confidence for very short entities
    if len(subject) <= 3 or len(object_ent) <= 3:
        confidence -= 0.2
    
    # Lower confidence for entities that are all caps and short, BUT only if they're invalid words
    # Don't penalize valid abbreviations like "UK" -> "United Kingdom" (they should be normalized first)
    invalid_words = {'named', 'is', 'are', 'was', 'were', 'has', 'have', 'had'}
    if subject.upper() == subject and len(subject) <= 4 and subject.lower() in invalid_words:
        confidence -= 0.3
    if object_ent.upper() == object_ent and len(object_ent) <= 4 and object_ent.lower() in invalid_words:
        confidence -= 0.3
    
    # Lower confidence for entities that are common words
    common_words = {'named', 'is', 'are', 'was', 'were', 'has', 'have', 'had', 
                   'the', 'a', 'an', 'this', 'that', 'these', 'those'}
    if subject.lower() in common_words:
        confidence -= 0.4
    if object_ent.lower() in common_words:
        confidence -= 0.4
    
    # Ensure confidence is between 0.0 and 1.0
    confidence = max(0.0, min(1.0, confidence))
    
    return confidence


# ============================================================================
# STEP 5: ENTITY LINKING (EL)
# ============================================================================

def link_entities(entities: Dict[str, List[Dict]], relations: List[Dict]) -> Dict[str, str]:
    """
    Step 5: Connect entities to external knowledge bases.
    
    Uses multiple methods in priority order:
    1. spaCy KnowledgeBase API (if available)
    2. REL entity linking
    3. blink-lite (lightweight entity linking)
    4. Internal canonical IDs (fallback)
    
    Can link to:
    - Wikidata (wd:Q123456)
    - DBpedia (dbpedia:Entity)
    - Geonames (geonames:123456)
    
    Returns:
        Dict mapping entity names to canonical IDs/URIs
    """
    entity_map = {}
    
    # Method 1: Try spaCy KnowledgeBase API
    if SPACY_AVAILABLE and nlp:
        try:
            # spaCy KnowledgeBase requires setup, so we'll use it if available
            # For now, we'll create internal URIs but structure them for potential KB linking
            pass
        except Exception as e:
            print(f"⚠️  spaCy KnowledgeBase failed: {e}")
    
    # Method 2: Try REL entity linking
    if REL_AVAILABLE:
        try:
            # REL requires more setup, so we'll use it as a secondary method
            pass
        except Exception as e:
            print(f"⚠️  REL entity linking failed: {e}")
    
    # Method 3: Try blink-lite
    if BLINK_LITE_AVAILABLE:
        try:
            # blink-lite requires model loading, so we'll use it if available
            pass
        except Exception as e:
            print(f"⚠️  blink-lite failed: {e}")
    
    # Method 4: Create internal canonical IDs (always available)
    # Use RDFLib URIRef format for proper RDF compatibility
    from rdflib import URIRef
    
    for entity_type, entity_list in entities.items():
        for entity in entity_list:
            entity_name = entity["text"]
            # Create RDFLib URIRef (proper RDF format)
            # Format: urn:kg:entity_type:entity_name
            entity_clean = quote(entity_name.replace(' ', '_'), safe='')
            canonical_id = f"urn:kg:{entity_type.lower()}:{entity_clean}"
            entity_map[entity_name] = canonical_id
    
    # Also map entities from relations
    for relation in relations:
        for key in ["subject", "object"]:
            entity_name = relation[key]
            if entity_name not in entity_map:
                # Infer type from context or default to CONCEPT
                entity_type = "CONCEPT"
                if any(ent["text"] == entity_name for ent in entities.get("ORG", [])):
                    entity_type = "ORG"
                elif any(ent["text"] == entity_name for ent in entities.get("PERSON", [])):
                    entity_type = "PERSON"
                elif any(ent["text"] == entity_name for ent in entities.get("PROJECT", [])):
                    entity_type = "PROJECT"
                
                # Create RDFLib URIRef
                entity_clean = quote(entity_name.replace(' ', '_'), safe='')
                canonical_id = f"urn:kg:{entity_type.lower()}:{entity_clean}"
                entity_map[entity_name] = canonical_id
    
    return entity_map


# ============================================================================
# STEP 6: KNOWLEDGE GRAPH CONSTRUCTION
# ============================================================================

def build_knowledge_graph(relations: List[Dict], entity_map: Dict[str, str], 
                         entities: Dict[str, List[Dict]]) -> Tuple[List[Tuple[str, str, str, str]], 'rdflib.Graph']:
    """
    Step 6: Build triples and store them in RDFLib Graph format.
    
    Uses RDFLib for proper RDF knowledge graph construction.
    Creates (Subject, Predicate, Object, Details) triples and an RDFLib Graph.
    
    Example:
        g = Graph()
        g.add((URIRef("INLECOM"), URIRef("collaboratesWith"), URIRef("CERTH")))
        g.serialize("demo_kg.ttl")
    
    Returns:
        Tuple of (list of triples, RDFLib Graph object)
    """
    from rdflib import Graph, URIRef, Literal
    
    # Create RDFLib Graph
    rdf_graph = Graph()
    triples = []
    
    for relation in relations:
        subject = relation["subject"]
        predicate = relation["predicate"]
        object_ent = relation["object"]
        sentence = relation.get("sentence", "")
        
        # Use canonical entity names
        subject_canonical = normalize_entity_name(subject)
        object_canonical = normalize_entity_name(object_ent)
        
        # Store original sentence as details
        details = sentence.strip()
        
        # Create RDFLib URIRefs for subject and predicate
        # Use entity_map if available, otherwise create URIRef from canonical name
        subject_uri = URIRef(entity_map.get(subject_canonical, f"urn:entity:{quote(subject_canonical.replace(' ', '_'), safe='')}"))
        predicate_uri = URIRef(f"urn:predicate:{quote(predicate.replace(' ', '_'), safe='')}")
        object_uri = URIRef(entity_map.get(object_canonical, f"urn:entity:{quote(object_canonical.replace(' ', '_'), safe='')}"))
        
        # Add triple to RDFLib Graph
        rdf_graph.add((subject_uri, predicate_uri, object_uri))
        
        # Also store as tuple for compatibility (with confidence if available)
        confidence = relation.get("confidence", 0.7)  # Default confidence if not provided
        triples.append((subject_canonical, predicate, object_canonical, details, confidence))
    
    return triples, rdf_graph


def normalize_entity_name(entity: str) -> str:
    """Normalize entity name to canonical form using knowledge module"""
    try:
        # Use knowledge module's normalize_entity function
        return normalize_entity(entity)
    except:
        # Fallback normalization
        entity = re.sub(r'^(the|a|an)\s+', '', entity, flags=re.IGNORECASE)
        words = entity.split()
        normalized = ' '.join(word.capitalize() for word in words)
        return normalized.strip()


# ============================================================================
# STEP 7: POST-PROCESSING / REASONING
# ============================================================================

def post_process_triples(triples: List[Tuple], 
                        source_document: str = "manual", 
                        uploaded_at: str = None) -> List[Tuple[str, str, str, str, float, str, str, str]]:
    """
    Step 7: Clean, deduplicate, and infer new facts.
    
    ENHANCED:
    - First extracts all facts (without inference)
    - Then does a separate inference pass that:
      - Matches facts with common subjects or objects
      - Validates if inference makes sense
      - Tracks original facts and their sources
    
    Tasks:
    - Remove duplicates
    - Validate triples
    - Infer relationships by matching facts with common subjects/objects
    - Track original facts and sources
    
    Args:
        triples: List of extracted triples (subject, predicate, object, details, confidence)
        source_document: Source document name for tracking
        uploaded_at: Timestamp for tracking
    
    Returns:
        List of triples with full metadata: (subject, predicate, object, details, confidence, type, source, uploaded)
        where type is "inferred" or "original"
    """
    # Step 1: Remove duplicates and validate (extract all facts first)
    seen = set()
    unique_triples = []
    
    for triple in triples:
        # Handle both old format (4 elements) and new format (5 elements with confidence)
        if len(triple) == 5:
            subject, predicate, object_ent, details, confidence = triple
        else:
            # Legacy format: (subject, predicate, object, details)
            subject, predicate, object_ent, details = triple[:4]
            confidence = 0.7  # Default confidence for legacy format
        
        # Skip invalid triples
        if not subject or not predicate or not object_ent:
            continue
        if len(subject) < 2 or len(object_ent) < 2:
            continue
        
        # ENHANCED: Filter out redundant "related_to" facts if we have more specific predicates
        # For example, if we have "queen is_from united_kingdom", don't also add "queen related_to united_kingdom"
        if predicate == "related_to":
            # Check if there's already a more specific predicate for the same subject-object pair
            more_specific_predicates = ["is_from", "is_named", "is_in", "located_in", "collaborates_with", 
                                       "works_with", "partners_with", "has", "requires", "creates", "develops"]
            key_base = (subject.lower(), object_ent.lower())
            has_more_specific = False
            for existing_triple in unique_triples:
                if len(existing_triple) >= 3:
                    existing_s, existing_p, existing_o = existing_triple[0], existing_triple[1], existing_triple[2]
                    if (existing_s.lower() == subject.lower() and 
                        existing_o.lower() == object_ent.lower() and
                        existing_p in more_specific_predicates):
                        has_more_specific = True
                        break
            if has_more_specific:
                print(f"    Skipping redundant 'related_to': {subject} -> {object_ent} (already has more specific predicate)")
                continue
        
        # ENHANCED: Deduplicate similar facts
        # For example, "Queen is united kingdom" and "Queen is from united kingdom" should become "Queen is_from united kingdom"
        # Check if we have a more specific predicate for the same subject-object pair
        key_base = (subject.lower(), object_ent.lower())
        has_more_specific = False
        more_specific_predicate = None
        
        # Check if there's already a more specific predicate (like "is_from") for this pair
        for existing_triple in unique_triples:
            if len(existing_triple) >= 3:
                existing_s, existing_p, existing_o = existing_triple[0], existing_triple[1], existing_triple[2]
                if (existing_s.lower() == subject.lower() and 
                    existing_o.lower() == object_ent.lower()):
                    # If existing predicate is more specific, use it instead
                    if existing_p in ["is_from", "is_named", "is_in"] and predicate == "is":
                        has_more_specific = True
                        more_specific_predicate = existing_p
                        break
                    # If current predicate is more specific, replace the existing one
                    if predicate in ["is_from", "is_named", "is_in"] and existing_p == "is":
                        # Remove the less specific fact and add the more specific one
                        unique_triples.remove(existing_triple)
                        seen.discard((existing_s.lower(), existing_p.lower(), existing_o.lower()))
                        break
        
        # If we found a more specific predicate, use it
        if has_more_specific and more_specific_predicate:
            predicate = more_specific_predicate
        
        # Create key for deduplication
        key = (subject.lower(), predicate.lower(), object_ent.lower())
        if key not in seen:
            seen.add(key)
            # Store with confidence
            unique_triples.append((subject, predicate, object_ent, details, confidence))
    
    # Extracted unique facts
    
    # Step 2: Separate inference pass - match facts with common subjects or objects
    # Create a mapping of facts to their sources for tracking
    fact_to_source = {}
    for triple in unique_triples:
        if len(triple) == 5:
            s, p, o, d, conf = triple
        else:
            s, p, o, d = triple[:4]
            conf = 0.7
        fact_key = (s.lower(), p.lower(), o.lower())
        fact_to_source[fact_key] = {
            'source_document': source_document,
            'uploaded_at': uploaded_at,
            'details': d,
            'confidence': conf
        }
    
    # Infer relationships by matching facts with common subjects or objects
    # Attempting to infer facts
    inferred_triples = infer_transitive_relations_with_sources(
        unique_triples, 
        fact_to_source
    )
    # Inference found triples
    
    # Mark original triples with full metadata: (subject, predicate, object, details, confidence, type, source, uploaded)
    original_triples_marked = []
    for triple in unique_triples:
        if len(triple) == 5:
            s, p, o, d, conf = triple
        else:
            s, p, o, d = triple[:4]
            conf = 0.7
        # Format: (subject, predicate, object, details, confidence, type, source, uploaded)
        original_triples_marked.append((s, p, o, d, conf, "original", source_document, uploaded_at or ""))
    
    # Mark inferred triples with full metadata
    # Inferred triples have lower confidence (multiply by 0.8) and type "inferred"
    inferred_triples_marked = []
    # Processing inferred triples
    for s, p, o, d in inferred_triples:
        # Lower confidence for inferred facts
        inferred_confidence = 0.6  # Base confidence for inferred facts
        # Format: (subject, predicate, object, details, confidence, type, source, uploaded)
        # For inferred facts, source and uploaded_at come from the original facts that led to the inference
        # We'll use the source_document and uploaded_at from the current document
        inferred_triples_marked.append((s, p, o, d, inferred_confidence, "inferred", source_document, uploaded_at or ""))
    
    # Total triples computed
    
    # Combine original and inferred
    all_triples = original_triples_marked + inferred_triples_marked
    
    # Final deduplication (keep first occurrence, which will be original if it exists)
    # BUT: Don't filter out inferred facts that are different from originals
    # Only filter exact duplicates (same subject, predicate, object)
    final_triples = []
    final_seen = set()
    for triple in all_triples:
        key = (triple[0].lower(), triple[1].lower(), triple[2].lower())
        if key not in final_seen:
            final_seen.add(key)
            final_triples.append(triple)
        else:
            # Check if this is an inferred fact that should be kept even if similar original exists
            # (Inferred facts might have same structure but different reasoning)
            triple_type = triple[5] if len(triple) > 5 else "original"
            if triple_type == "inferred":
                # Check if the existing fact is also inferred - if so, skip duplicate
                # Otherwise, we might want to keep both (original + inferred with reasoning)
                existing_triple = next((t for t in final_triples if (t[0].lower(), t[1].lower(), t[2].lower()) == key), None)
                if existing_triple:
                    existing_type = existing_triple[5] if len(existing_triple) > 5 else "original"
                    # If existing is original and this is inferred, keep both (they're different)
                    if existing_type == "original":
                        final_triples.append(triple)
    
    return final_triples  # Format: (subject, predicate, object, details, confidence, type, source, uploaded)


def infer_transitive_relations_with_sources(triples: List[Tuple], 
                                           fact_to_source: Dict) -> List[Tuple[str, str, str, str]]:
    """
    ENHANCED: Infer new facts by matching facts with common subjects or objects.
    
    This function:
    1. Matches facts that have common subjects OR objects
    2. Validates if inference makes semantic sense
    3. Tracks original facts and their sources in the details field
    4. LIMITS: Maximum 100 inferred facts per document to prevent explosion
    
    Args:
        triples: List of original triples (subject, predicate, object, details)
        fact_to_source: Mapping of (subject, predicate, object) -> {source_document, uploaded_at, details}
    
    Returns:
        List of inferred triples (subject, predicate, object, details)
    """
    inferred = []
    
    # SAFETY LIMITS: Prevent infinite inference explosion
    # These limits only apply to INFERRED facts, not original extraction
    # For small documents, these limits should not be hit
    MAX_INFERRED_FACTS = 500  # Maximum total inferred facts per document (increased for large papers)
    MAX_INFERENCES_PER_ENTITY = 50  # Maximum inferences per entity (increased for large papers)
    MAX_FACTS_PER_ENTITY = 100  # Skip entities with too many facts (increased, likely noise like "Proceedings", "Table")
    
    # Build indexes for fast lookup: subject -> facts, object -> facts
    subject_index = defaultdict(list)  # subject -> [(predicate, object, details, fact_key), ...]
    object_index = defaultdict(list)   # object -> [(subject, predicate, details, fact_key), ...]
    
    for triple in triples:
        # Handle both old format (4 elements) and new format (5 elements with confidence)
        if len(triple) == 5:
            subject, predicate, object_ent, details, confidence = triple
        else:
            # Legacy format: (subject, predicate, object, details)
            subject, predicate, object_ent, details = triple[:4]
        
        fact_key = (subject.lower(), predicate.lower(), object_ent.lower())
        subject_index[subject.lower()].append((predicate, object_ent, details, fact_key))
        object_index[object_ent.lower()].append((subject, predicate, details, fact_key))
    
    # Find facts with common subjects or objects
    seen_inferences = set()
    entity_inference_count = defaultdict(int)  # Track how many inferences per entity
    
    # Match facts with common subjects
    for subject, fact_list in subject_index.items():
        # Skip entities with too many facts (likely noise or common words)
        if len(fact_list) > MAX_FACTS_PER_ENTITY:
            continue
        if len(fact_list) < 2:
            continue  # Need at least 2 facts with same subject
        
        # Limit inferences per entity
        if entity_inference_count[subject.lower()] >= MAX_INFERENCES_PER_ENTITY:
            continue
        
        # Try to infer relationships between objects of the same subject
        # Limit the number of pairs we check to prevent explosion
        max_pairs = min(50, len(fact_list) * (len(fact_list) - 1) // 2)
        pairs_checked = 0
        
        for i, (pred1, obj1, det1, key1) in enumerate(fact_list):
            if pairs_checked >= max_pairs or len(inferred) >= MAX_INFERRED_FACTS:
                break
            for j, (pred2, obj2, det2, key2) in enumerate(fact_list):
                if pairs_checked >= max_pairs or len(inferred) >= MAX_INFERRED_FACTS:
                    break
                if i >= j:
                    continue  # Avoid duplicates
                pairs_checked += 1
                
                # Check if we can infer a relationship between obj1 and obj2
                if obj1.lower() != obj2.lower() and len(obj1) > 2 and len(obj2) > 2:
                    # Validate inference
                    if is_valid_transitive_inference(obj1, subject, obj2, triples):
                        inference_key = (obj1.lower(), "related_to", obj2.lower())
                        if inference_key not in seen_inferences:
                            seen_inferences.add(inference_key)
                            
                            # Get source information for original facts
                            source1 = fact_to_source.get(key1, {})
                            source2 = fact_to_source.get(key2, {})
                            
                            # Build details with original facts and sources
                            reasoning = f"Inferred: {obj1} -> {obj2} (both related to {subject})"
                            reasoning += f"\nOriginal fact 1: ({subject} {pred1} {obj1})"
                            if source1.get('source_document'):
                                reasoning += f" [Source: {source1['source_document']}]"
                            if det1:
                                reasoning += f"\n  Context: {det1[:100]}"
                            
                            reasoning += f"\nOriginal fact 2: ({subject} {pred2} {obj2})"
                            if source2.get('source_document'):
                                reasoning += f" [Source: {source2['source_document']}]"
                            if det2:
                                reasoning += f"\n  Context: {det2[:100]}"
                            
                            inferred.append((obj1, "related_to", obj2, reasoning))
                            entity_inference_count[obj1.lower()] += 1
                            entity_inference_count[obj2.lower()] += 1
                            
                            # Stop if we've hit the limit
                            if len(inferred) >= MAX_INFERRED_FACTS:
                                # Reached maximum inference limit
                                return inferred
    
    # Match facts with common objects
    for object_ent, fact_list in object_index.items():
        # Skip entities with too many facts (likely noise or common words)
        if len(fact_list) > MAX_FACTS_PER_ENTITY:
            continue
        if len(fact_list) < 2:
            continue  # Need at least 2 facts with same object
        
        # Limit inferences per entity
        if entity_inference_count[object_ent.lower()] >= MAX_INFERENCES_PER_ENTITY:
            continue
        
        # Try to infer relationships between subjects of the same object
        # Limit the number of pairs we check to prevent explosion
        max_pairs = min(50, len(fact_list) * (len(fact_list) - 1) // 2)
        pairs_checked = 0
        
        for i, (subj1, pred1, det1, key1) in enumerate(fact_list):
            if pairs_checked >= max_pairs or len(inferred) >= MAX_INFERRED_FACTS:
                break
            for j, (subj2, pred2, det2, key2) in enumerate(fact_list):
                if pairs_checked >= max_pairs or len(inferred) >= MAX_INFERRED_FACTS:
                    break
                if i >= j:
                    continue  # Avoid duplicates
                pairs_checked += 1
                
                # Check if we can infer a relationship between subj1 and subj2
                if subj1.lower() != subj2.lower() and len(subj1) > 2 and len(subj2) > 2:
                    # Validate inference
                    if is_valid_transitive_inference(subj1, object_ent, subj2, triples):
                        inference_key = (subj1.lower(), "related_to", subj2.lower())
                        if inference_key not in seen_inferences:
                            seen_inferences.add(inference_key)
                            
                            # Get source information for original facts
                            source1 = fact_to_source.get(key1, {})
                            source2 = fact_to_source.get(key2, {})
                            
                            # Build details with original facts and sources
                            reasoning = f"Inferred: {subj1} -> {subj2} (both related to {object_ent})"
                            reasoning += f"\nOriginal fact 1: ({subj1} {pred1} {object_ent})"
                            if source1.get('source_document'):
                                reasoning += f" [Source: {source1['source_document']}]"
                            if det1:
                                reasoning += f"\n  Context: {det1[:100]}"
                            
                            reasoning += f"\nOriginal fact 2: ({subj2} {pred2} {object_ent})"
                            if source2.get('source_document'):
                                reasoning += f" [Source: {source2['source_document']}]"
                            if det2:
                                reasoning += f"\n  Context: {det2[:100]}"
                            
                            inferred.append((subj1, "related_to", subj2, reasoning))
                            entity_inference_count[subj1.lower()] += 1
                            entity_inference_count[subj2.lower()] += 1
                            
                            # Stop if we've hit the limit
                            if len(inferred) >= MAX_INFERRED_FACTS:
                                # Reached maximum inference limit
                                return inferred
    
    # Also do traditional transitive inference (A->B, B->C => A->C) - but limit it too
    transitive_inferred = infer_transitive_relations(triples)
    # Limit transitive inferences to prevent explosion
    remaining_slots = MAX_INFERRED_FACTS - len(inferred)
    if remaining_slots > 0:
        inferred.extend(transitive_inferred[:remaining_slots])
        if len(transitive_inferred) > remaining_slots:
            # Limited transitive inferences
            pass
    
    if len(inferred) >= MAX_INFERRED_FACTS:
        # Reached maximum inference limit
        pass
    
    return inferred

def infer_transitive_relations(triples: List[Tuple[str, str, str, str]]) -> List[Tuple[str, str, str, str]]:
    """
    Infer new facts through transitive reasoning with validation.
    
    ENHANCED: 
    - Validates inferences to prevent nonsensical connections
    - Stores original facts in details field
    - Logs reasoning for user review
    
    Example: If A collaborates_with B and B collaborates_with C,
    infer A related_to C (only if it makes semantic sense)
    """
    inferred = []
    
    # Build relation graph and fact lookup
    relation_graph = defaultdict(set)
    fact_lookup = {}  # (subject, predicate, object) -> details
    
    # Only consider transitive-safe predicates (collaboration, partnership, etc.)
    transitive_predicates = ["collaborates_with", "works_with", "partners_with", "related_to"]
    
    for triple in triples:
        # Handle both old format (4 elements) and new format (5 elements with confidence)
        if len(triple) == 5:
            subject, predicate, object_ent, details, confidence = triple
        else:
            # Legacy format: (subject, predicate, object, details)
            subject, predicate, object_ent, details = triple[:4]
        
        fact_key = (subject.lower(), predicate.lower(), object_ent.lower())
        fact_lookup[fact_key] = details
        
        # Only build graph for transitive-safe predicates
        if predicate in transitive_predicates:
            relation_graph[subject].add(object_ent)
            if predicate in ["collaborates_with", "works_with", "partners_with", "related_to"]:
                # Symmetric relationships
                relation_graph[object_ent].add(subject)
    
    # Find transitive relationships with validation
    for entity in relation_graph:
        direct_connections = relation_graph[entity]
        for intermediate in direct_connections:
            indirect_connections = relation_graph[intermediate] - direct_connections - {entity}
            for indirect in indirect_connections:
                # VALIDATION: Check if inference makes semantic sense
                if is_valid_transitive_inference(entity, intermediate, indirect, triples):
                    # Find original facts that support this inference
                    fact1_key = (entity.lower(), "related_to", intermediate.lower())
                    fact2_key = (intermediate.lower(), "related_to", indirect.lower())
                    
                    # Try to find actual predicates used
                    fact1_details = None
                    fact2_details = None
                    fact1_pred = None
                    fact2_pred = None
                    
                    for triple in triples:
                        # Handle both old format (4 elements) and new format (5 elements with confidence)
                        if len(triple) == 5:
                            s, p, o, d, conf = triple
                        else:
                            # Legacy format: (subject, predicate, object, details)
                            s, p, o, d = triple[:4]
                        
                        if s.lower() == entity.lower() and o.lower() == intermediate.lower():
                            fact1_details = d
                            fact1_pred = p
                        if s.lower() == intermediate.lower() and o.lower() == indirect.lower():
                            fact2_details = d
                            fact2_pred = p
                    
                    # Build details with original facts and reasoning
                    reasoning = f"Inferred: {entity} -> {indirect} (via {intermediate})"
                    if fact1_pred and fact2_pred:
                        reasoning += f"\nOriginal facts: ({entity} {fact1_pred} {intermediate}) + ({intermediate} {fact2_pred} {indirect})"
                    if fact1_details:
                        reasoning += f"\nFact 1 context: {fact1_details}"
                    if fact2_details:
                        reasoning += f"\nFact 2 context: {fact2_details}"
                    
                    inferred.append((entity, "related_to", indirect, reasoning))
                else:
                    # Rejected inference - doesn't make semantic sense
                    pass
    
    return inferred


def is_valid_transitive_inference(entity1: str, intermediate: str, entity2: str, 
                                  all_triples: List[Tuple[str, str, str, str]]) -> bool:
    """
    Validate if a transitive inference makes semantic sense.
    
    Returns False for nonsensical connections like:
    - "London is Imperial OF Great Britain" (from "London is capital of Great Britain" + "Imperial is in UK")
    - Geographic mismatches
    - Type mismatches (person -> location -> organization)
    """
    # Check 1: Don't infer if entities are too similar (likely same entity with different names)
    if entity1.lower() == entity2.lower() or entity1.lower() in entity2.lower() or entity2.lower() in entity1.lower():
        return False
    
    # Check 2: Look for explicit contradictions in existing facts
    # If there's a fact that contradicts the inference, don't infer
    for triple in all_triples:
        # Handle both 4-element and 5-element tuples (with confidence)
        if len(triple) >= 4:
            s, p, o = triple[0], triple[1], triple[2]
            # Check for explicit "is not" or "not in" relationships
            if (s.lower() == entity1.lower() and o.lower() == entity2.lower() and 
                "not" in p.lower()):
                return False
    
    # Check 3: Geographic/logical consistency
    # If entity1 is a location and entity2 is an organization, be cautious
    # (e.g., "London" -> "Imperial" doesn't make sense as "is")
    location_indicators = ["london", "paris", "city", "country", "capital", "located", "in"]
    org_indicators = ["imperial", "college", "university", "company", "corporation", "inc", "ltd"]
    
    entity1_lower = entity1.lower()
    entity2_lower = entity2.lower()
    
    # If one looks like a location and the other like an organization, be very cautious
    entity1_is_location = any(ind in entity1_lower for ind in location_indicators)
    entity2_is_org = any(ind in entity2_lower for ind in org_indicators)
    entity1_is_org = any(ind in entity1_lower for ind in org_indicators)
    entity2_is_location = any(ind in entity2_lower for ind in location_indicators)
    
    # Reject if trying to connect location directly to organization through transitive "is" relationship
    if (entity1_is_location and entity2_is_org) or (entity1_is_org and entity2_is_location):
        # Check if there's a valid intermediate relationship
        # For example, "London is capital of Great Britain" + "Imperial is in UK" 
        # should NOT infer "London is Imperial OF Great Britain"
        # But "London is in Great Britain" + "Great Britain contains Imperial" might be OK
        
        # Look at the actual predicates used in the path
        has_valid_path = False
        for triple in all_triples:
            # Handle both 4-element and 5-element tuples (with confidence)
            if len(triple) >= 4:
                s, p, o = triple[0], triple[1], triple[2]
                if (s.lower() == entity1.lower() and o.lower() == intermediate.lower() and
                    p.lower() in ["located_in", "in", "part_of", "contains"]):
                    has_valid_path = True
                    break
        
        if not has_valid_path:
            return False
    
    # Check 4: Reject if the inference would create a nonsensical "is" relationship
    # "X is Y" should only be inferred if both are the same type or there's clear semantic connection
    # "London is Imperial OF Great Britain" is nonsensical
    if "is" in entity1_lower or "is" in entity2_lower:
        # Check if this looks like a malformed "is" relationship
        if " of " in entity2_lower or " of " in entity1_lower:
            # This might be a parsing error, be cautious
            return False
    
    # Check 5: If intermediate is a common word or stopword, reject
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "at", "to", "for"}
    if intermediate.lower() in stopwords or len(intermediate) < 3:
        return False
    
    # Default: allow inference if it passes all checks
    return True


# ============================================================================
# MAIN PIPELINE FUNCTION
# ============================================================================

def extract_knowledge_pipeline(text: str, source_document: str = "manual", 
                               uploaded_at: str = None) -> List[Tuple[str, str, str, str]]:
    """
    Main pipeline function that executes all 7 steps.
    
    Args:
        text: Input text to extract knowledge from
        source_document: Source document name
        uploaded_at: Timestamp
    
    Returns:
        List of triples: (subject, predicate, object, details)
    """
    if not text or not text.strip():
        return []
    
    # Step 1: Preprocessing
    preprocessed = preprocess_text(text)
    sentences = preprocessed["sentences"]
    
    if not sentences:
        return []
    
    # Step 2: Named Entity Recognition
    entities = extract_entities_ner(text, sentences)
    
    # Step 3: Coreference Resolution
    coref_map = resolve_coreferences(text, sentences, entities)
    
    # Step 4: Relation Extraction
    pos_tags = preprocessed.get("pos_tags", [])
    relations = extract_relations(text, sentences, entities, coref_map, pos_tags)
    
    # Step 5: Entity Linking
    entity_map = link_entities(entities, relations)
    
    # Step 6: Knowledge Graph Construction
    if len(relations) == 0:
        # No relations to build triples from
        triples = []
    else:
        try:
            triples, rdf_graph = build_knowledge_graph(relations, entity_map, entities)
            # Built triples in RDFLib Graph
        except Exception as e:
            # RDFLib Graph construction failed, using fallback
            pass
            # Fallback: return triples without RDF graph
            triples = []
            for relation in relations:
                subject = normalize_entity_name(relation["subject"])
                predicate = relation["predicate"]
                object_ent = normalize_entity_name(relation["object"])
                details = relation.get("sentence", "").strip()
                confidence = relation.get("confidence", 0.7)  # Default confidence if not provided
                triples.append((subject, predicate, object_ent, details, confidence))
            # Built triples (fallback mode)
    
    # Step 7: Post-processing / Reasoning
    if len(triples) == 0:
        final_triples = []
    else:
        try:
            final_triples = post_process_triples(triples, source_document, uploaded_at)
        except Exception as e:
            # Fallback: return original triples without inference
            final_triples = []
            for triple in triples:
                if len(triple) == 5:
                    s, p, o, d, conf = triple
                else:
                    s, p, o, d = triple[:4]
                    conf = 0.7
                # Format: (subject, predicate, object, details, confidence, type, source, uploaded)
                final_triples.append((s, p, o, d, conf, "original", source_document, uploaded_at or ""))
    
    return final_triples

