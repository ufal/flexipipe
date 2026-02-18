"""Integration with udapi for CoNLL-U processing, transformation, and evaluation.

udapi (Universal Dependencies API) is a Python framework from Charles University (UFAL)
for working with Universal Dependencies data. This module provides integration points
for udapi's useful features that complement flexipipe.

Key udapi features that could enhance flexipipe:
- Tree transformations (deproj, flatten, proj)
- Evaluation metrics (conll17, conll18, f1, parsing)
- Multiple output formats (HTML, TikZ, TextModeTrees)
- CoNLL-U processing utilities
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .doc import Document


def apply_udapi_transform(
    document: "Document",
    transform_name: str,
    **kwargs: Any,
) -> "Document":
    """Apply a udapi transformation block to a flexipipe document.
    
    Args:
        document: flexipipe Document to transform
        transform_name: Name of udapi transform block (e.g., 'deproj', 'flatten', 'proj')
        **kwargs: Additional arguments for the transform block
        
    Returns:
        Transformed Document
        
    Example:
        >>> from flexipipe.udapi_integration import apply_udapi_transform
        >>> doc = load_document("input.conllu")
        >>> transformed = apply_udapi_transform(doc, "deproj")
    """
    try:
        from udapi.core.document import Document as UdapiDocument
        from .conllu import document_to_conllu, conllu_to_document
    except ImportError:
        raise SystemExit(
            "udapi integration requires the 'udapi' package. "
            "Install it with: pip install udapi"
        )
    
    # Convert flexipipe Document to CoNLL-U
    conllu_text = document_to_conllu(document, create_implicit_mwt=False)
    
    # Load into udapi
    udapi_doc = UdapiDocument()
    udapi_doc.from_conllu_string(conllu_text)
    
    # Apply transform
    transform_map = {
        "deproj": "udapi.block.transform.deproj.Deproj",
        "flatten": "udapi.block.transform.flatten.Flatten",
        "proj": "udapi.block.transform.proj.Proj",
    }
    
    if transform_name not in transform_map:
        raise ValueError(
            f"Unknown transform: {transform_name}. "
            f"Available: {', '.join(transform_map.keys())}"
        )
    
    # Import and apply transform
    module_path, class_name = transform_map[transform_name].rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    transform_class = getattr(module, class_name)
    transform_block = transform_class(**kwargs)
    transform_block.process_document(udapi_doc)
    
    # Convert back to CoNLL-U and then to flexipipe Document
    transformed_conllu = udapi_doc.to_conllu_string()
    return conllu_to_document(transformed_conllu, doc_id=document.id)


def evaluate_with_udapi(
    predicted: "Document",
    gold: "Document",
    eval_type: str = "f1",
    **kwargs: Any,
) -> Dict[str, Any]:
    """Evaluate predicted document against gold using udapi evaluation blocks.
    
    Args:
        predicted: Predicted flexipipe Document
        gold: Gold-standard flexipipe Document
        eval_type: Type of evaluation ('f1', 'conll17', 'conll18', 'parsing')
        **kwargs: Additional arguments for the evaluation block
        
    Returns:
        Dictionary with evaluation metrics
        
    Example:
        >>> from flexipipe.udapi_integration import evaluate_with_udapi
        >>> metrics = evaluate_with_udapi(predicted_doc, gold_doc, eval_type="f1")
        >>> print(f"UAS: {metrics.get('uas')}, LAS: {metrics.get('las')}")
    """
    try:
        from udapi.core.document import Document as UdapiDocument
        from .conllu import document_to_conllu
    except ImportError:
        raise SystemExit(
            "udapi integration requires the 'udapi' package. "
            "Install it with: pip install udapi"
        )
    
    # Convert to CoNLL-U
    pred_conllu = document_to_conllu(predicted, create_implicit_mwt=False)
    gold_conllu = document_to_conllu(gold, create_implicit_mwt=False)
    
    # Load into udapi
    pred_udapi = UdapiDocument()
    pred_udapi.from_conllu_string(pred_conllu)
    gold_udapi = UdapiDocument()
    gold_udapi.from_conllu_string(gold_conllu)
    
    # Apply evaluation block
    eval_map = {
        "f1": "udapi.block.eval.f1.F1",
        "conll17": "udapi.block.eval.conll17.Conll17",
        "conll18": "udapi.block.eval.conll18.Conll18",
        "parsing": "udapi.block.eval.parsing.Parsing",
    }
    
    if eval_type not in eval_map:
        raise ValueError(
            f"Unknown eval type: {eval_type}. "
            f"Available: {', '.join(eval_map.keys())}"
        )
    
    # Import and apply eval block
    module_path, class_name = eval_map[eval_type].rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    eval_class = getattr(module, class_name)
    eval_block = eval_class(**kwargs)
    
    # Process both documents
    eval_block.process_document(gold_udapi)
    eval_block.process_document(pred_udapi)
    
    # Extract metrics (udapi eval blocks store metrics internally)
    # This is a simplified version - actual implementation would need to
    # extract metrics from the eval_block object
    return {
        "eval_type": eval_type,
        "note": "Metrics would be extracted from udapi eval_block object",
    }


def list_udapi_features() -> Dict[str, List[str]]:
    """List available udapi features that could be useful for flexipipe.
    
    Returns:
        Dictionary mapping feature categories to lists of available features
    """
    features = {
        "transforms": [],
        "eval_blocks": [],
        "write_formats": [],
        "other_features": [],
    }
    
    try:
        import udapi
        
        # Check transforms
        try:
            from udapi.block import transform
            import pkgutil
            features["transforms"] = [
                m.name for m in pkgutil.iter_modules(transform.__path__)
            ]
        except Exception:
            pass
        
        # Check eval blocks
        try:
            from udapi.block import eval
            import pkgutil
            features["eval_blocks"] = [
                m.name for m in pkgutil.iter_modules(eval.__path__)
            ]
        except Exception:
            pass
        
        # Check write formats
        try:
            from udapi.block import write
            import pkgutil
            features["write_formats"] = [
                m.name for m in pkgutil.iter_modules(write.__path__)
            ]
        except Exception:
            pass
        
        features["other_features"] = [
            "CoNLL-U I/O (load_conllu, to_conllu_string, from_conllu_string)",
            "Tree/node manipulation",
            "Coreference handling",
        ]
    except ImportError:
        pass
    
    return features
