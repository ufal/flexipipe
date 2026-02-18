"""SVG rendering system for dependency tree visualizations.

This module provides a pluggable system for rendering dependency trees as SVG.
Backends can register custom renderers, and users can select renderers via
the --svg-style option.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Protocol, Tuple

if TYPE_CHECKING:
    from .doc import Document, Sentence, Token


class SVGRenderer(Protocol):
    """Protocol for SVG renderers.
    
    A renderer takes a sentence and returns an SVG string.
    """
    
    def render(
        self,
        sentence: "Sentence",
        token_id_to_idx: Dict[int, int],
        options: Dict[str, Any],
    ) -> str:
        """Render a sentence as SVG.
        
        Args:
            sentence: The sentence to render
            token_id_to_idx: Mapping from token.id to list index
            options: Renderer-specific options (e.g., {"boxes": True, "root": True})
            
        Returns:
            SVG string (should be a complete, valid SVG document)
        """
        ...


class DisplacyDepRenderer:
    """Arrow-style dependency visualization using spaCy's displacy."""
    
    def render(
        self,
        sentence: "Sentence",
        token_id_to_idx: Dict[int, int],
        options: Dict[str, Any],
    ) -> str:
        """Render using spaCy's displacy dep style."""
        try:
            from spacy import displacy
        except ImportError:
            raise SystemExit("SVG output requires spaCy. Install it with: pip install spacy")
        
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


class TreeStyleRenderer:
    """Hierarchical tree-style visualization."""
    
    def render(
        self,
        sentence: "Sentence",
        token_id_to_idx: Dict[int, int],
        options: Dict[str, Any],
    ) -> str:
        """Render as hierarchical tree."""
        if not sentence.tokens:
            return ""
        
        # Parse options
        show_boxes = options.get("boxes", False)
        show_root = options.get("root", False)
        horizontal_order = options.get("horizontal", False)
        flush_bottom = options.get("flush_bottom", False)
        node_width = options.get("node_width", 80)
        level_height = options.get("level_height", 120)
        text_height = options.get("text_height", 20)
        padding = options.get("padding", 20)
        
        # Build tree structure
        children: Dict[int, List[int]] = {}
        root_idx = None
        
        for idx, token in enumerate(sentence.tokens):
            if token.head == 0 or token.head is None:
                root_idx = idx
            else:
                head_idx = token_id_to_idx.get(token.head)
                if head_idx is not None:
                    if head_idx not in children:
                        children[head_idx] = []
                    children[head_idx].append(idx)
        
        if root_idx is None:
            root_idx = 0
        
        # Calculate layout
        node_positions: Dict[int, Tuple[float, float]] = {}
        
        if horizontal_order:
            # Horizontal layout: nodes in sentence order
            x_start = padding
            for idx, token in enumerate(sentence.tokens):
                # Find depth of this node
                depth = self._calculate_depth(idx, children, root_idx)
                x = x_start + idx * (node_width + padding)
                y = depth * level_height
                node_positions[idx] = (x, y)
            max_x = len(sentence.tokens) * (node_width + padding) + padding
        else:
            # Hierarchical tree layout
            def calculate_layout(node_idx: int, level: int, x_start: float) -> float:
                """Calculate positions recursively."""
                node_children = children.get(node_idx, [])
                
                if not node_children:
                    x = x_start
                    node_positions[node_idx] = (x, level * level_height)
                    return x_start + node_width + padding
                
                child_x = x_start
                for child_idx in node_children:
                    child_x = calculate_layout(child_idx, level + 1, child_x)
                
                if node_children:
                    first_child_x, _ = node_positions[node_children[0]]
                    last_child_x, _ = node_positions[node_children[-1]]
                    parent_x = (first_child_x + last_child_x) / 2
                else:
                    parent_x = x_start
                
                node_positions[node_idx] = (parent_x, level * level_height)
                return max(child_x, parent_x + node_width + padding)
            
            max_x = calculate_layout(root_idx, 0, padding)
        
        # Calculate max Y
        if node_positions:
            max_y_pos = max(y for _, y in node_positions.values())
            if flush_bottom:
                # Flush words to bottom
                max_y = max_y_pos + text_height * 2 + 15 + 5
            else:
                max_y = max_y_pos + 50
        else:
            max_y = 50
        
        # Build SVG
        svg_lines = [
            '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"',
            f'width="{max_x + padding}" height="{max_y}"',
            f'viewBox="0 0 {max_x + padding} {max_y}">',
            '<style>',
        ]
        
        if show_boxes:
            svg_lines.append('  .tree-node { fill: #f0f0f0; stroke: #333; stroke-width: 1px; }')
        else:
            svg_lines.append('  .tree-node { fill: none; stroke: none; }')
        
        svg_lines.extend([
            '  .tree-text { font-family: Arial, sans-serif; font-size: 12px; text-anchor: middle; fill: #000; }',
            '  .tree-tag { font-family: Arial, sans-serif; font-size: 10px; fill: #666; text-anchor: middle; }',
            '  .tree-edge { stroke: #333; stroke-width: 1.5px; fill: none; }',
            '  .tree-label { font-family: Arial, sans-serif; font-size: 9px; fill: #0066cc; text-anchor: middle; }',
            '  .tree-root { font-family: Arial, sans-serif; font-size: 10px; fill: #999; text-anchor: middle; }',
            '</style>',
        ])
        
        # Draw edges
        for idx, token in enumerate(sentence.tokens):
            if token.head and token.head > 0 and token.deprel:
                head_idx = token_id_to_idx.get(token.head)
                if head_idx is not None and head_idx != idx and idx in node_positions and head_idx in node_positions:
                    x1, y1 = node_positions[head_idx]
                    x2, y2 = node_positions[idx]
                    x1 += node_width / 2
                    x2 += node_width / 2
                    
                    y1_start = y1 + text_height + 15 + 5
                    y2_end = y2 + text_height - 10
                    
                    svg_lines.append(f'<line class="tree-edge" x1="{x1}" y1="{y1_start}" x2="{x2}" y2="{y2_end}"/>')
                    label_x = (x1 + x2) / 2
                    label_y = (y1_start + y2_end) / 2 - 5
                    svg_lines.append(f'<text class="tree-label" x="{label_x}" y="{label_y}">{token.deprel}</text>')
        
        # Draw root indicator if requested
        if show_root and root_idx is not None and root_idx in node_positions:
            x, y = node_positions[root_idx]
            svg_lines.append(f'<text class="tree-root" x="{x + node_width/2}" y="{y - 5}">ROOT</text>')
        
        # Draw nodes
        for idx, token in enumerate(sentence.tokens):
            if idx not in node_positions:
                continue
            x, y = node_positions[idx]
            
            svg_lines.append(
                f'<rect class="tree-node" x="{x}" y="{y}" width="{node_width}" height="{text_height * 2 + 10}" '
                f'fill="{"#f0f0f0" if show_boxes else "none"}" '
                f'stroke="{"#333" if show_boxes else "none"}" stroke-width="{"1" if show_boxes else "0"}px"/>'
            )
            
            text_y = y + text_height
            svg_lines.append(f'<text class="tree-text" x="{x + node_width/2}" y="{text_y}">{token.form}</text>')
            
            tag = token.upos or token.xpos or ""
            if tag:
                tag_y = y + text_height + 15
                svg_lines.append(f'<text class="tree-tag" x="{x + node_width/2}" y="{tag_y}">{tag}</text>')
        
        svg_lines.append('</svg>')
        return '\n'.join(svg_lines)
    
    def _calculate_depth(self, node_idx: int, children: Dict[int, List[int]], root_idx: int) -> int:
        """Calculate depth of a node in the tree."""
        if node_idx == root_idx:
            return 0
        
        # Find parent
        for parent_idx, child_list in children.items():
            if node_idx in child_list:
                return 1 + self._calculate_depth(parent_idx, children, root_idx)
        
        return 1  # Default depth if parent not found


class DeplacyRenderer:
    """Dependency visualization using deplacy package.
    
    Requires: pip install deplacy
    """
    
    def render(
        self,
        sentence: "Sentence",
        token_id_to_idx: Dict[int, int],
        options: Dict[str, Any],
    ) -> str:
        """Render using deplacy package."""
        try:
            import deplacy
        except ImportError:
            raise SystemExit(
                "deplacy renderer requires the 'deplacy' package. "
                "Install it with: pip install deplacy"
            )
        
        # Convert flexipipe Sentence to a format deplacy can use
        # deplacy works with spaCy Doc objects, so we need to create a minimal Doc
        # or use deplacy's direct rendering methods
        
        # Try to get spaCy Doc from document metadata if available
        # Otherwise, we'll need to convert our Sentence to a format deplacy understands
        
        # For now, fall back to displacy if we can't get a spaCy Doc
        # In a full implementation, we'd convert Sentence to spaCy Doc format
        try:
            from spacy import displacy
            # Build displacy format (same as DisplacyDepRenderer)
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
            
            # Note: deplacy.serve() starts a web server, which isn't ideal for CLI
            # deplacy.dot() returns DOT format, not SVG
            # For now, we'll use displacy but note that deplacy could be integrated
            # if we convert to spaCy Doc format or use deplacy's rendering differently
            return displacy.render(displacy_data, style="dep", jupyter=False, manual=True)
        except Exception as e:
            raise SystemExit(f"deplacy rendering failed: {e}")


class GraphvizRenderer:
    """Dependency visualization using Graphviz (via DOT format).
    
    Requires: pip install graphviz
    Note: Also requires Graphviz system package (brew install graphviz on macOS)
    """
    
    def render(
        self,
        sentence: "Sentence",
        token_id_to_idx: Dict[int, int],
        options: Dict[str, Any],
    ) -> str:
        """Render using Graphviz DOT format converted to SVG."""
        try:
            from graphviz import Digraph
        except ImportError:
            raise SystemExit(
                "Graphviz renderer requires the 'graphviz' package. "
                "Install it with: pip install graphviz"
            )
        
        # Create a directed graph
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
        try:
            svg_bytes = dot.pipe(format='svg')
            return svg_bytes.decode('utf-8')
        except Exception as e:
            raise SystemExit(f"Graphviz rendering failed: {e}. Make sure Graphviz is installed: brew install graphviz (macOS) or apt-get install graphviz (Linux)")


# Registry for custom renderers
_renderer_registry: Dict[str, SVGRenderer] = {
    "dep": DisplacyDepRenderer(),
    "tree": TreeStyleRenderer(),
    # Alternative renderers (optional dependencies)
    # "deplacy": DeplacyRenderer(),  # Uncomment if deplacy is available
    # "graphviz": GraphvizRenderer(),  # Uncomment if graphviz is available
}


def register_renderer(name: str, renderer: SVGRenderer) -> None:
    """Register a custom SVG renderer.
    
    Args:
        name: Renderer name (used in --svg-style)
        renderer: Renderer instance implementing SVGRenderer protocol
    
    Example:
        >>> from flexipipe.svg_renderers import register_renderer, DisplacyDepRenderer
        >>> register_renderer("my_custom", DisplacyDepRenderer())
    """
    _renderer_registry[name] = renderer


def register_optional_renderers() -> None:
    """Register optional renderers if their dependencies are available.
    
    This function attempts to register renderers that require optional dependencies
    like 'deplacy' or 'graphviz'. If the dependencies aren't installed, they're skipped.
    
    Example:
        >>> from flexipipe.svg_renderers import register_optional_renderers
        >>> register_optional_renderers()  # Registers graphviz/deplacy if available
    """
    try:
        from . import svg_renderers_optional
        optional = svg_renderers_optional.register_optional_renderers()
        for name, renderer in optional.items():
            register_renderer(name, renderer)
    except ImportError:
        pass


def get_renderer(name: str) -> Optional[SVGRenderer]:
    """Get a renderer by name.
    
    Args:
        name: Renderer name
        
    Returns:
        Renderer instance or None if not found
    """
    return _renderer_registry.get(name)


def parse_svg_style(style_str: str) -> Tuple[str, Dict[str, any]]:
    """Parse --svg-style option.
    
    Supports formats:
    - "dep" -> ("dep", {})
    - "tree" -> ("tree", {})
    - "tree,boxes" -> ("tree", {"boxes": True})
    - "tree,boxes,root" -> ("tree", {"boxes": True, "root": True})
    - "tree,horizontal,flush_bottom" -> ("tree", {"horizontal": True, "flush_bottom": True})
    
    Args:
        style_str: Style string (may include comma-separated options)
        
    Returns:
        Tuple of (renderer_name, options_dict)
    """
    parts = [p.strip() for p in style_str.split(",")]
    if not parts:
        return ("dep", {})
    
    renderer_name = parts[0]
    options: Dict[str, Any] = {}
    
    # Parse boolean options
    boolean_options = {
        "boxes": True,
        "root": True,
        "horizontal": True,
        "flush_bottom": True,
        "displacy": True,  # Force displacy renderer
    }
    
    for part in parts[1:]:
        if part in boolean_options:
            options[part] = True
        elif "=" in part:
            # Key=value option
            key, value = part.split("=", 1)
            key = key.strip()
            try:
                # Try to parse as number
                if "." in value:
                    options[key] = float(value)
                else:
                    options[key] = int(value)
            except ValueError:
                # Keep as string
                options[key] = value
    
    return (renderer_name, options)


def render_document_svg(
    document: "Document",
    style: str = "dep",
    backend_renderer: Optional[Callable[[], Optional[str]]] = None,
) -> str:
    """Render a document as SVG.
    
    Args:
        document: Document to render
        style: SVG style string (e.g., "dep", "tree", "tree,boxes")
        backend_renderer: Optional backend-specific renderer function
            that returns SVG string or None
        
    Returns:
        Combined SVG string for all sentences
    """
    """Render a document as SVG.
    
    Args:
        document: Document to render
        style: SVG style string (e.g., "dep", "tree", "tree,boxes")
        backend_renderer: Optional backend-specific renderer function
            that returns SVG string or None
        
    Returns:
        Combined SVG string for all sentences
    """
    # Try backend-specific renderer first (e.g., UD-Kanbun's to_svg())
    if backend_renderer:
        try:
            svg_result = backend_renderer()
            if svg_result:
                return svg_result
        except Exception:
            pass
    
    # Parse style
    renderer_name, options = parse_svg_style(style)
    
    # "displacy" is an alias for "dep" (displacy renderer)
    if renderer_name == "displacy":
        renderer_name = "dep"
    
    # Get renderer
    renderer = get_renderer(renderer_name)
    if not renderer:
        raise SystemExit(f"Unknown SVG renderer: {renderer_name}. Available: {', '.join(_renderer_registry.keys())}")
    
    # Check for dependencies
    has_dependencies = False
    for sent in document.sentences:
        for token in sent.tokens:
            if token.head or token.deprel:
                has_dependencies = True
                break
        if has_dependencies:
            break
    
    if not has_dependencies:
        raise SystemExit("SVG output requires dependency relations (head and deprel). The document does not have dependency parsing.")
    
    # Render each sentence
    svg_parts = []
    for sent in document.sentences:
        if not sent.tokens:
            continue
        
        # Build token_id_to_idx mapping
        token_id_to_idx = {}
        for idx, token in enumerate(sent.tokens):
            token_id_to_idx[token.id] = idx
        
        # Render sentence
        try:
            svg_content = renderer.render(sent, token_id_to_idx, options)
            if svg_content:
                svg_parts.append(svg_content)
        except Exception as e:
            if not svg_parts:
                raise SystemExit(f"SVG rendering failed: {e}")
    
    if not svg_parts:
        raise SystemExit("SVG output failed: could not generate SVG from document")
    
    # Combine multiple SVGs into a single valid SVG document
    if len(svg_parts) == 1:
        return svg_parts[0]
    
    # Multiple sentences: combine into a single SVG
    svg_contents = []
    max_width = 0
    total_height = 0
    spacing = 50
    
    for svg_part in svg_parts:
        content_match = re.search(r'<svg[^>]*>(.*)</svg>', svg_part, re.DOTALL)
        if not content_match:
            g_match = re.search(r'<g[^>]*>(.*)</g>', svg_part, re.DOTALL)
            if g_match:
                svg_content = g_match.group(1)
                height_val = 500
                svg_contents.append((svg_content, height_val))
                total_height += height_val + spacing
                continue
            continue
        
        svg_content = content_match.group(1)
        
        width_match = re.search(r'width="([^"]+)"', svg_part)
        height_match = re.search(r'height="([^"]+)"', svg_part)
        viewbox_match = re.search(r'viewBox="([^"]+)"', svg_part)
        
        width_val = 1000
        height_val = 500
        
        if width_match:
            try:
                width_val = float(width_match.group(1).replace("px", ""))
            except ValueError:
                pass
        
        if height_match:
            try:
                height_val = float(height_match.group(1).replace("px", ""))
            except ValueError:
                pass
        
        if viewbox_match:
            viewbox_parts = viewbox_match.group(1).split()
            if len(viewbox_parts) >= 4:
                try:
                    width_val = float(viewbox_parts[2])
                    height_val = float(viewbox_parts[3])
                except (ValueError, IndexError):
                    pass
        
        max_width = max(max_width, width_val)
        svg_contents.append((svg_content, height_val))
        total_height += height_val + spacing
    
    # Build combined SVG
    combined_content = []
    current_y = 0
    
    for svg_content, height_val in svg_contents:
        combined_content.append(f'<g transform="translate(0, {current_y})">{svg_content}</g>')
        current_y += height_val + spacing
    
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{max_width}" height="{total_height - spacing}" '
        f'viewBox="0 0 {max_width} {total_height - spacing}">\n'
        + "\n".join(combined_content) + "\n</svg>"
    )
