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
    eval_type: str = "parsing",  # Default to parsing for UAS/LAS
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
    
    # udapi evaluators expect both trees in the same bundle with different zones
    # Load both documents separately, then merge them into a single document
    # with both trees in the same bundle but different zones
    gold_udapi = UdapiDocument()
    gold_udapi.from_conllu_string(gold_conllu)
    
    pred_udapi = UdapiDocument()
    pred_udapi.from_conllu_string(pred_conllu)
    
    # Create a new combined document
    combined_udapi = UdapiDocument()
    gold_zone = "gold"
    pred_zone = "predicted"
    
    # Get number of sentences (should match)
    gold_bundles = list(gold_udapi.bundles)
    pred_bundles = list(pred_udapi.bundles)
    
    # Create bundles with both gold and predicted trees
    for i, (gold_bundle, pred_bundle) in enumerate(zip(gold_bundles, pred_bundles)):
        combined_bundle = combined_udapi.create_bundle()
        
        # Copy gold tree to combined bundle with zone
        gold_tree = list(gold_bundle.trees)[0]
        combined_gold_tree = combined_bundle.create_tree(zone=gold_zone)
        # Copy nodes from gold_tree to combined_gold_tree
        for gold_node in gold_tree.descendants:
            new_node = combined_gold_tree.create_child(
                form=gold_node.form,
                lemma=gold_node.lemma,
                upos=gold_node.upos,
                xpos=gold_node.xpos,
                feats=gold_node.feats,
                deprel=gold_node.deprel,
            )
            # Set parent (will be set after all nodes are created)
            if gold_node.parent and gold_node.parent.ord > 0:
                # We'll set parent after creating all nodes
                pass
        
        # Set parents for gold tree
        gold_nodes_list = list(combined_gold_tree.descendants)
        for idx, gold_node in enumerate(gold_tree.descendants):
            if gold_node.parent and gold_node.parent.ord > 0:
                parent_idx = gold_node.parent.ord - 1
                if parent_idx < len(gold_nodes_list):
                    gold_nodes_list[idx].parent = gold_nodes_list[parent_idx]
            elif gold_node.parent and gold_node.parent.ord == 0:
                # Root node
                gold_nodes_list[idx].parent = combined_gold_tree
        
        # Copy predicted tree to combined bundle with zone
        pred_tree = list(pred_bundle.trees)[0]
        combined_pred_tree = combined_bundle.create_tree(zone=pred_zone)
        # Copy nodes from pred_tree to combined_pred_tree
        for pred_node in pred_tree.descendants:
            new_node = combined_pred_tree.create_child(
                form=pred_node.form,
                lemma=pred_node.lemma,
                upos=pred_node.upos,
                xpos=pred_node.xpos,
                feats=pred_node.feats,
                deprel=pred_node.deprel,
            )
        
        # Set parents for predicted tree
        pred_nodes_list = list(combined_pred_tree.descendants)
        for idx, pred_node in enumerate(pred_tree.descendants):
            if pred_node.parent and pred_node.parent.ord > 0:
                parent_idx = pred_node.parent.ord - 1
                if parent_idx < len(pred_nodes_list):
                    pred_nodes_list[idx].parent = pred_nodes_list[parent_idx]
            elif pred_node.parent and pred_node.parent.ord == 0:
                # Root node
                pred_nodes_list[idx].parent = combined_pred_tree
    
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
    
    # Both F1 and Parsing evaluators need gold_zone
    eval_block = eval_class(gold_zone=gold_zone, **kwargs)
    
    # Process the combined document (contains both gold and predicted trees)
    eval_block.process_document(combined_udapi)
    
    # Extract metrics from eval_block
    # udapi eval blocks store metrics in different ways depending on the type
    metrics = {
        "eval_type": eval_type,
    }
    
    # For Parsing evaluator, extract UAS/LAS from internal counters
    if eval_type == "parsing":
        if hasattr(eval_block, 'total') and eval_block.total > 0:
            if hasattr(eval_block, 'correct_uas'):
                metrics['uas'] = eval_block.correct_uas / eval_block.total
            if hasattr(eval_block, 'correct_las'):
                metrics['las'] = eval_block.correct_las / eval_block.total
            if hasattr(eval_block, 'correct_ulas'):
                metrics['ulas'] = eval_block.correct_ulas / eval_block.total
            metrics['total_nodes'] = eval_block.total
    
    # For F1 evaluator, extract token-level metrics
    elif eval_type == "f1":
        if hasattr(eval_block, 'correct') and hasattr(eval_block, 'pred') and hasattr(eval_block, 'gold'):
            # F1 evaluator provides precision, recall, f1 as properties
            if hasattr(eval_block, 'precision'):
                metrics['precision'] = eval_block.precision
            if hasattr(eval_block, 'recall'):
                metrics['recall'] = eval_block.recall
            if hasattr(eval_block, 'f1'):
                metrics['f1'] = eval_block.f1
            metrics['correct'] = eval_block.correct
            metrics['predicted'] = eval_block.pred
            metrics['gold'] = eval_block.gold
    
    # Try direct attribute access for common metrics (fallback)
    for attr in ['uas', 'las', 'upos', 'xpos', 'lemma', 'feats', 'sentences', 'words', 'tokens']:
        if attr not in metrics and hasattr(eval_block, attr):
            value = getattr(eval_block, attr)
            if value is not None:
                metrics[attr] = value
    
    return metrics


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
