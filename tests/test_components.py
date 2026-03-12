"""Tests for component parsing and reassembly."""

import pytest
from collections import OrderedDict
from prompt_evolver.components import parse_components, reassemble, list_component_names


def test_parse_components_basic():
    """Test basic component parsing."""
    prompt = """Preamble text here.
<!-- @component: greeting -->
Hello student!
<!-- @component: instruction -->
Let's learn math.
"""
    preamble, components = parse_components(prompt)

    assert "Preamble text here." in preamble
    assert len(components) == 2
    assert "greeting" in components
    assert "instruction" in components
    assert "Hello student!" in components["greeting"]


def test_parse_components_no_tags():
    """Test parsing prompt with no component tags."""
    prompt = "This is a plain prompt with no tags."
    preamble, components = parse_components(prompt)

    assert preamble == ""
    assert len(components) == 1
    assert "__all__" in components
    assert components["__all__"] == prompt


def test_reassemble_basic():
    """Test reassembling prompt without replacement."""
    preamble = "Preamble\n"
    components = OrderedDict([
        ("greeting", "<!-- @component: greeting -->\nHello!"),
        ("instruction", "<!-- @component: instruction -->\nLearn.")
    ])

    result = reassemble(preamble, components)

    assert "Preamble" in result
    assert "Hello!" in result
    assert "Learn." in result


def test_reassemble_with_replacement():
    """Test reassembling with component replacement."""
    preamble = "Preamble\n"
    components = OrderedDict([
        ("greeting", "<!-- @component: greeting -->\nHello!"),
        ("instruction", "<!-- @component: instruction -->\nLearn.")
    ])

    result = reassemble(preamble, components, target="greeting", replacement="Hi there!")

    assert "Hi there!" in result
    assert "<!-- @component: greeting -->" in result
    assert "Learn." in result


def test_list_component_names():
    """Test listing component names."""
    prompt = """<!-- @component: intro -->
Text
<!-- @component: body -->
More text
<!-- @component: conclusion -->
End"""

    names = list_component_names(prompt)

    assert len(names) == 3
    assert names == ["intro", "body", "conclusion"]


def test_parse_reassemble_round_trip():
    """Test that parse -> reassemble preserves the prompt."""
    original = """Preamble text.
<!-- @component: greeting -->
Hello student!
<!-- @component: instruction -->
Let's learn math."""

    preamble, components = parse_components(original)
    result = reassemble(preamble, components)

    assert result == original
