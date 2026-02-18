"""Optional SVG renderers that require additional dependencies.

These renderers can be registered dynamically if their dependencies are available.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from .doc import Sentence


def register_optional_renderers() -> Dict[str, Any]:
    """Register optional renderers if their dependencies are available.
    
    Returns:
        Dict of renderer_name -> renderer_instance for available optional renderers
    """
    optional_renderers = {}
    
    # Try to register deplacy renderer
    try:
        import deplacy
        from .svg_renderers import SVGRenderer
        
        class DeplacyRenderer:
            """Dependency visualization using deplacy package."""
            
            def render(
                self,
                sentence: "Sentence",
                token_id_to_idx: Dict[int, int],
                options: Dict[str, Any],
            ) -> str:
                """Render using deplacy package."""
                # deplacy works with spaCy Doc objects
                # For now, we'll note that deplacy integration would require
                # converting Sentence to spaCy Doc format
                # This is a placeholder that shows how it could be integrated
                raise NotImplementedError(
                    "deplacy renderer requires converting Sentence to spaCy Doc format. "
                    "This is a proof-of-concept for renderer swapping."
                )
        
        optional_renderers["deplacy"] = DeplacyRenderer()
    except ImportError:
        pass
    
    # Try to register Graphviz renderer
    try:
        from graphviz import Digraph
        from .svg_renderers import SVGRenderer
        
        class GraphvizRenderer:
            """Dependency visualization using Graphviz (via DOT format)."""
            
            def render(
                self,
                sentence: "Sentence",
                token_id_to_idx: Dict[int, int],
                options: Dict[str, Any],
            ) -> str:
                """Render using Graphviz DOT format converted to SVG."""
                dot = Digraph(format='svg')
                dot.attr(rankdir='TB')  # Top to bottom
                dot.attr('node', shape='box', style='rounded')
                
                # Add nodes (tokens)
                for idx, token in enumerate(sentence.tokens):
                    label = f"{token.form}\\n{token.upos or token.xpos or ''}"
                    dot.node(str(idx), label)
                
                # Add edges (dependency relations)
                for idx, token in enumerate(sentence.tokens):
                    if token.head and token.head > 0 and token.deprel:
                        head_idx = token_id_to_idx.get(token.head)
                        if head_idx is not None and head_idx != idx:
                            dot.edge(str(head_idx), str(idx), label=token.deprel)
                
                # Render to SVG string
                svg_bytes = dot.pipe(format='svg')
                return svg_bytes.decode('utf-8')
        
        optional_renderers["graphviz"] = GraphvizRenderer()
    except ImportError:
        pass
    
    # Try to register udapi renderer
    # udapi is from Charles University (UFAL) and provides CoNLL-U processing
    # udapi's HTML output uses JavaScript to generate SVG (js-treex-view)
    # For now, we use displacy for SVG, but udapi provides excellent CoNLL-U processing
    try:
        from udapi.core.document import Document as UdapiDocument
        from .conllu import document_to_conllu
        from .doc import Document
        
        class UdapiRenderer:
            """Dependency visualization using udapi package.
            
            udapi provides multiple output formats:
            - HTML with JavaScript-based SVG (udapi.block.write.html)
            - TikZ (udapi.block.write.tikz) - LaTeX-based tree visualization
            - TextModeTrees (udapi.block.write.textmodetrees) - text-based
            
            This renderer uses displacy for SVG output, leveraging udapi's
            CoNLL-U processing capabilities.
            """
            
            def render(
                self,
                sentence: "Sentence",
                token_id_to_idx: Dict[int, int],
                options: Dict[str, Any],
            ) -> str:
                """Render using udapi package."""
                # Create a temporary Document with just this sentence
                temp_doc = Document(id="temp", meta={})
                temp_doc.sentences = [sentence]
                
                # Convert to CoNLL-U format
                conllu_text = document_to_conllu(temp_doc, model_info=None, create_implicit_mwt=False)
                
                # Load into udapi for processing (could apply transforms here)
                udapi_doc = UdapiDocument()
                udapi_doc.from_conllu_string(conllu_text)
                
                # For SVG output, use displacy
                # Note: udapi's HTML uses JavaScript to generate SVG, which requires a browser
                # For CLI SVG output, displacy is more appropriate
                try:
                    from spacy import displacy
                    words = []
                    arcs = []
                    
                    for idx, token in enumerate(sentence.tokens):
                        words.append({
                            "text": token.form,
                            "tag": token.upos or token.xpos or "",
                        })
                    
                    for idx, token in enumerate(sentence.tokens):
                        if token.head and token.head > 0 and token.deprel:
                            head_idx = token_id_to_idx.get(token.head)
                            if head_idx is not None and head_idx != idx:
                                arcs.append({
                                    "start": min(idx, head_idx),
                                    "end": max(idx, head_idx),
                                    "label": token.deprel,
                                    "dir": "left" if head_idx < idx else "right",
                                })
                    
                    displacy_data = {
                        "words": words,
                        "arcs": arcs,
                    }
                    return displacy.render(displacy_data, style="dep", jupyter=False, manual=True)
                except Exception as e:
                    raise SystemExit(f"udapi renderer failed: {e}")
        
        optional_renderers["udapi"] = UdapiRenderer()
    except ImportError:
        # udapi might not be installed, that's okay
        pass
    except Exception as e:
        # Log the exception for debugging, but don't fail
        # udapi might be installed but API is different, skip for now
        import sys
        if "--debug" in sys.argv:
            print(f"Warning: udapi renderer registration failed: {e}", file=sys.stderr)
        pass
    
    # Try to register conllview renderer
    # conllview is a popular tool for visualizing CoNLL-U files
    # It typically works by converting CoNLL-U format to SVG
    try:
        import conllview
        from .conllu import document_to_conllu
        from .doc import Document
        
        class ConllviewRenderer:
            """Dependency visualization using conllview package."""
            
            def render(
                self,
                sentence: "Sentence",
                token_id_to_idx: Dict[int, int],
                options: Dict[str, Any],
            ) -> str:
                """Render using conllview package."""
                # Create a temporary Document with just this sentence
                temp_doc = Document(id="temp", meta={})
                temp_doc.sentences = [sentence]
                
                # Convert to CoNLL-U format
                conllu_text = document_to_conllu(temp_doc, model_info=None, create_implicit_mwt=False)
                
                # Use conllview to render SVG
                # Try different possible API patterns
                if hasattr(conllview, 'render'):
                    return conllview.render(conllu_text)
                elif hasattr(conllview, 'to_svg'):
                    return conllview.to_svg(conllu_text)
                elif hasattr(conllview, 'visualize'):
                    return conllview.visualize(conllu_text)
                elif callable(conllview):
                    # conllview might be a function
                    return conllview(conllu_text)
                else:
                    raise NotImplementedError("conllview API not recognized")
        
        optional_renderers["conllview"] = ConllviewRenderer()
    except ImportError:
        # conllview might not be installed, that's okay
        pass
    except Exception:
        # conllview might be installed but API is different, skip for now
        pass
    
    return optional_renderers
