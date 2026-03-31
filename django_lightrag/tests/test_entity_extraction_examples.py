from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_examples_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "prompts"
        / "entity_extraction_examples.py"
    )
    spec = spec_from_file_location("entity_extraction_examples", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_render_entity_extraction_examples_uses_structured_signature_format() -> None:
    module = _load_examples_module()
    rendered = module.render_entity_extraction_examples(
        tuple_delimiter="<|#|>",
        completion_delimiter="<|COMPLETE|>",
        entity_types='"Person", "Organization", "Location"',
    )

    assert '"extraction_output": [' in rendered
    assert '"input_text":' in rendered
    assert '"keywords": [' in rendered
    assert "entity<|#|>" not in rendered
    assert "relation<|#|>" not in rendered
    assert "<|COMPLETE|>" not in rendered
