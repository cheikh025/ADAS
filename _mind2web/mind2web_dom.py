"""Mind2Web DOM pruning and compact tree serialization.

Adapted from OSU-NLP-Group/Mind2Web's ``src/data_utils/dom_utils.py`` under
the MIT license. The pruning defaults and neighborhood semantics are preserved.
"""

from __future__ import annotations

import copy
import re
from typing import Iterable

from lxml import etree


class Mind2WebDOMError(RuntimeError):
    """Raised when cleaned Mind2Web HTML cannot produce the required context."""


def get_descendants(
    node: etree._Element,
    max_depth: int,
    current_depth: int = 0,
) -> list:
    if current_depth > max_depth:
        return []
    descendants = []
    for child in node:
        descendants.append(child)
        descendants.extend(get_descendants(child, max_depth, current_depth + 1))
    return descendants


def prune_tree(
    dom_tree: etree._Element,
    candidate_set: Iterable[str],
    max_depth: int = 5,
    max_children: int = 50,
    max_sibling: int = 3,
) -> etree._Element:
    """Apply the official Mind2Web candidate-neighborhood pruning algorithm."""

    candidate_set = {str(candidate_id) for candidate_id in candidate_set}
    nodes_to_keep: set[str] = set()
    missing: list[str] = []
    for candidate_id in candidate_set:
        matches = dom_tree.xpath(
            '//*[@backend_node_id=$candidate_id]', candidate_id=candidate_id
        )
        if not matches:
            missing.append(candidate_id)
            continue
        candidate_node = matches[0]
        nodes_to_keep.add(candidate_node.attrib["backend_node_id"])
        nodes_to_keep.update(
            item.attrib.get("backend_node_id", "")
            for item in candidate_node.xpath("ancestor::*")
        )
        nodes_to_keep.update(
            [
                item.attrib.get("backend_node_id", "")
                for item in get_descendants(candidate_node, max_depth)
            ][:max_children]
        )
        parent = candidate_node.getparent()
        if parent is not None:
            siblings = [item for item in parent.getchildren() if item.tag != "text"]
            sibling_index = siblings.index(candidate_node)
            nodes_to_keep.update(
                item.attrib.get("backend_node_id", "")
                for item in siblings[
                    max(0, sibling_index - max_sibling) :
                    sibling_index + max_sibling + 1
                ]
            )
    if missing:
        raise Mind2WebDOMError(
            "Ranked candidate nodes are missing from cleaned_html: "
            + ", ".join(sorted(missing)[:10])
        )

    new_tree = copy.deepcopy(dom_tree)
    for node in new_tree.xpath("//*")[::-1]:
        if node.tag != "text":
            node_id = node.attrib.get("backend_node_id", "")
            is_keep = node_id in nodes_to_keep
            is_candidate = node_id in candidate_set
        else:
            parent = node.getparent()
            parent_id = (
                parent.attrib.get("backend_node_id", "")
                if parent is not None
                else ""
            )
            is_keep = parent_id in nodes_to_keep
            is_candidate = parent_id in candidate_set
        if not is_keep and node.getparent() is not None:
            node.getparent().remove(node)
        else:
            if not is_candidate or node.tag == "text":
                node.attrib.pop("backend_node_id", None)
            if (
                len(node.attrib) == 0
                and not any(child.tag == "text" for child in node.getchildren())
                and node.getparent() is not None
                and node.tag != "text"
                and len(node.getchildren()) <= 1
            ):
                for child in node.getchildren():
                    node.addprevious(child)
                node.getparent().remove(node)
    return new_tree


def get_attribute_repr(
    node: etree._Element,
    max_value_length: int = 5,
    max_length: int = 20,
) -> None:
    attr_values_set: set[str] = set()
    attr_values = ""
    for attr in [
        "role",
        "aria_role",
        "type",
        "alt",
        "aria_description",
        "aria_label",
        "label",
        "title",
        "name",
        "text_value",
        "value",
        "placeholder",
        "input_checked",
        "input_value",
        "option_selected",
        "class",
    ]:
        if attr in node.attrib and node.attrib[attr] is not None:
            value = node.attrib[attr].lower()
            if value in {"hidden", "none", "presentation", "null", "undefined"}:
                continue
            if value.startswith("http"):
                continue
            value_tokens = value.split()
            value = " ".join(
                token for token in value_tokens if len(token) < 15
            ).split()
            value = " ".join(value[:max_value_length])
            if value and value not in attr_values_set:
                attr_values_set.add(value)
                attr_values += value + " "
    uid = node.attrib.get("backend_node_id", "")
    node.attrib.clear()
    if uid:
        node.attrib["id"] = uid
    if attr_values:
        node.attrib["meta"] = " ".join(attr_values.split()[:max_length])


def get_tree_repr(
    tree: str | etree._Element,
    max_value_length: int = 5,
    max_length: int = 20,
    id_mapping: dict[str, int] | None = None,
    keep_html_brackets: bool = False,
) -> tuple[str, dict[str, int]]:
    """Return the official compact tree representation and local ID mapping."""

    if id_mapping is None:
        id_mapping = {}
    if isinstance(tree, str):
        tree = etree.fromstring(tree)
    else:
        tree = copy.deepcopy(tree)
    for node in tree.xpath("//*"):
        if node.tag != "text":
            if "backend_node_id" in node.attrib:
                node_id = node.attrib["backend_node_id"]
                if node_id not in id_mapping:
                    id_mapping[node_id] = len(id_mapping)
                node.attrib["backend_node_id"] = str(id_mapping[node_id])
            get_attribute_repr(node, max_value_length, max_length)
        else:
            node.text = " ".join((node.text or "").split()[:max_length])
    tree_repr = etree.tostring(tree, encoding="unicode")
    tree_repr = tree_repr.replace('"', " ")
    tree_repr = (
        tree_repr.replace("meta= ", "")
        .replace("id= ", "id=")
        .replace(" >", ">")
    )
    tree_repr = re.sub(r"<text>(.*?)</text>", r"\1", tree_repr)
    if not keep_html_brackets:
        tree_repr = tree_repr.replace("/>", "$/$>")
        tree_repr = re.sub(r"</(.+?)>", r")", tree_repr)
        tree_repr = re.sub(r"<(.+?)>", r"(\1", tree_repr)
        tree_repr = tree_repr.replace("$/$", ")")

    html_escape_table = [
        ("&quot;", '"'),
        ("&amp;", "&"),
        ("&lt;", "<"),
        ("&gt;", ">"),
        ("&nbsp;", " "),
        ("&ndash;", "-"),
        ("&rsquo;", "'"),
        ("&lsquo;", "'"),
        ("&ldquo;", '"'),
        ("&rdquo;", '"'),
        ("&#39;", "'"),
        ("&#40;", "("),
        ("&#41;", ")"),
    ]
    for escaped, value in html_escape_table:
        tree_repr = tree_repr.replace(escaped, value)
    tree_repr = re.sub(r"\s+", " ", tree_repr).strip()
    return tree_repr, id_mapping


def format_pruned_html(
    cleaned_html: str,
    candidate_ids: Iterable[str],
) -> tuple[str, dict[str, str]]:
    """Parse, prune, and serialize one page plus its candidate snippets."""

    candidate_ids = [str(candidate_id) for candidate_id in candidate_ids]
    if not candidate_ids:
        return "", {}
    try:
        dom_tree = etree.fromstring(cleaned_html)
    except (etree.XMLSyntaxError, ValueError, TypeError) as exc:
        raise Mind2WebDOMError("Unable to parse Mind2Web cleaned_html.") from exc

    pruned_tree = prune_tree(dom_tree, candidate_ids)
    tree_repr, id_mapping = get_tree_repr(pruned_tree, id_mapping={})
    candidate_reprs: dict[str, str] = {}
    for node in pruned_tree.xpath("//*[@backend_node_id]"):
        node_id = str(node.attrib["backend_node_id"])
        node_repr, _ = get_tree_repr(node, id_mapping=id_mapping)
        candidate_reprs[node_id] = " ".join(node_repr.split()[:10])
    missing = sorted(set(candidate_ids) - candidate_reprs.keys())
    if missing:
        raise Mind2WebDOMError(
            "Pruned tree lost ranked candidate nodes: " + ", ".join(missing[:10])
        )
    return tree_repr, candidate_reprs
