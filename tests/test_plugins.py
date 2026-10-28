"""Custom node plugins from the data folder's custom_nodes/ (report item F22)."""

import sys
import textwrap

import pytest
from fastapi.testclient import TestClient

from backend import main, paths, plugins
from backend.engine.graph import execute_prompt, validate_prompt
from backend.nodes import NODE_REGISTRY

PLUGIN = '''
from backend.nodes import NodeBase, register_node


@register_node("Shout Text")
class {cls}(NodeBase):
    CATEGORY = "plugins"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {{"required": {{"text": ("STRING", {{"default": "hi"}})}}}}

    def run(self, text):
        return {{"text": [text.upper()], "result": (text.upper(),)}}
'''


@pytest.fixture
def plugin_dir(isolated_data_dirs):
    folder = isolated_data_dirs["custom_nodes"]
    folder.mkdir()
    yield folder
    # Forget what the test registered or imported, so other tests see only built-ins.
    for key in [k for k, cls in NODE_REGISTRY.items() if getattr(cls, "PLUGIN", None)]:
        del NODE_REGISTRY[key]
    for name in [m for m in sys.modules if m == plugins.MODULE_PREFIX or m.startswith(plugins.MODULE_PREFIX + ".")]:
        del sys.modules[name]
    plugins.LOADED.clear()


def _write(path, text):
    path.write_text(textwrap.dedent(text), encoding="utf-8")


def test_a_plugin_file_registers_nodes_that_run(plugin_dir):
    _write(plugin_dir / "shout.py", PLUGIN.format(cls="ShoutTextTest"))
    (results,) = plugins.load_custom_nodes()
    assert (results.name, results.nodes, results.error) == ("shout", ["ShoutTextTest"], None)
    assert NODE_REGISTRY["ShoutTextTest"].PLUGIN == "shout"
    outputs = execute_prompt(validate_prompt({"n": {"class_type": "ShoutTextTest", "inputs": {"text": "hey"}}}))
    assert outputs["n"]["text"] == ["HEY"]


def test_a_plugin_package_can_use_relative_imports(plugin_dir):
    package = plugin_dir / "tools"
    package.mkdir()
    _write(package / "helpers.py", "SUFFIX = '!'\n")
    _write(package / "__init__.py", '''
        from backend.nodes import NodeBase, register_node

        from .helpers import SUFFIX


        @register_node("Exclaim")
        class ShoutPackageTest(NodeBase):
            CATEGORY = "plugins"
            RETURN_TYPES = ("STRING",)
            FUNCTION = "run"
            OUTPUT_NODE = True

            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {"text": ("STRING", {"default": "hi"})}}

            def run(self, text):
                return {"text": [text.upper() + SUFFIX], "result": (text,)}
    ''')
    (result,) = plugins.load_custom_nodes()
    assert result.error is None and result.nodes == ["ShoutPackageTest"]
    outputs = execute_prompt(validate_prompt({"n": {"class_type": "ShoutPackageTest", "inputs": {"text": "a"}}}))
    assert outputs["n"]["text"] == ["A!"]


@pytest.mark.parametrize(("failure", "message"), [
    ("raise RuntimeError('boom')", "RuntimeError: boom"),
    ("raise SystemExit('plugin requested exit')", "SystemExit: plugin requested exit"),
])
def test_a_broken_plugin_is_reported_and_the_rest_still_load(plugin_dir, failure, message):
    _write(plugin_dir / "a_broken.py", failure)
    _write(plugin_dir / "b_good.py", PLUGIN.format(cls="ShoutAfterBrokenTest"))
    _write(plugin_dir / "_helper.py", "raise SystemExit('never imported')\n")
    (plugin_dir / "notes.txt").write_text("ignored")
    broken, good = plugins.load_custom_nodes()
    assert broken.name == "a_broken" and message in broken.error and broken.nodes == []
    assert good.error is None and good.nodes == ["ShoutAfterBrokenTest"]
    assert "pixelflow_custom_nodes.a_broken" not in sys.modules


def test_a_plugin_cannot_replace_a_built_in_node(plugin_dir):
    original = NODE_REGISTRY["SaveImage"]
    _write(plugin_dir / "evil.py", PLUGIN.format(cls="SaveImage"))
    (result,) = plugins.load_custom_nodes()
    assert NODE_REGISTRY["SaveImage"] is original
    assert "refused to replace existing node(s): SaveImage" in result.error


def test_loading_can_be_turned_off(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "ROOT_DIR", tmp_path)
    assert plugins.custom_nodes_enabled({})
    assert not plugins.custom_nodes_enabled({plugins.DISABLE_ENV: "1"})
    (tmp_path / "settings.json").write_text('{"custom_nodes": false}', encoding="utf-8")
    assert not plugins.custom_nodes_enabled({})


def test_startup_loads_plugins_and_the_api_reports_them(plugin_dir, monkeypatch):
    monkeypatch.setenv("PIXELFLOW_TEST_MODE", "1")
    _write(plugin_dir / "shout.py", PLUGIN.format(cls="ShoutApiTest"))
    main.load_plugins()  # what the lifespan runs at startup
    client = TestClient(main.app, base_url="http://127.0.0.1:8288")
    nodes = client.get("/api/nodes").json()
    assert nodes["ShoutApiTest"]["plugin"] == "shout"
    assert nodes["SaveImage"]["plugin"] is None
    system = client.get("/api/system").json()
    assert system["custom_nodes"] == [
        {"name": "shout", "path": str(plugin_dir / "shout.py"), "nodes": ["ShoutApiTest"], "error": None}
    ]
