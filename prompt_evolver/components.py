"""Parse and reassemble prompt components delimited by <!-- @component: name --> tags."""

import re
from collections import OrderedDict
from typing import Optional


COMPONENT_PATTERN = re.compile(r'<!--\s*@component:\s*(\w+)\s*-->')


def parse_components(prompt_text: str) -> tuple[str, OrderedDict]:
    """Split prompt into preamble (untagged) and named components.

    Returns:
        (preamble, components) where preamble is frozen text before the first tag,
        and components is an OrderedDict of {name: text}.
    """
    matches = list(COMPONENT_PATTERN.finditer(prompt_text))

    if not matches:
        return "", OrderedDict([("__all__", prompt_text)])

    preamble = prompt_text[:matches[0].start()]

    components = OrderedDict()
    for i, match in enumerate(matches):
        name = match.group(1)
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(prompt_text)
        components[name] = prompt_text[start:end]

    return preamble, components


def reassemble(preamble: str, components: OrderedDict, target: Optional[str] = None, replacement: Optional[str] = None) -> str:
    """Reassemble full prompt from preamble + components.

    If target and replacement are provided, substitute that component's text.
    The replacement is wrapped with the original tag so the marker is preserved.
    """
    parts = [preamble]
    for name, text in components.items():
        if name == target and replacement is not None:
            tag = f"<!-- @component: {name} -->"
            if not replacement.startswith(tag):
                parts.append(tag + "\n" + replacement)
            else:
                parts.append(replacement)
        else:
            parts.append(text)
    return "".join(parts)


def list_component_names(prompt_text: str) -> list[str]:
    """Return list of component names found in prompt."""
    return COMPONENT_PATTERN.findall(prompt_text)
