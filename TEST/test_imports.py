"""Every view module and shared module must import cleanly (no Streamlit runtime).

We deliberately do NOT import app.py here: importing it executes top-level
st.set_page_config / st.navigation / pg.run(), which require a live runtime.
"""

import glob
import importlib
import os

import pytest

_VIEWS = sorted(
    "views." + os.path.basename(f)[:-3]
    for f in glob.glob(os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                    "views", "_*.py"))
    if not os.path.basename(f).startswith("__")
)


def test_core_modules_import():
    for mod in ("utils", "data_loader"):
        importlib.import_module(mod)


@pytest.mark.parametrize("modname", _VIEWS)
def test_view_imports(modname):
    mod = importlib.import_module(modname)
    assert hasattr(mod, "render"), f"{modname} has no render(store)"


def test_views_package_exports_pages():
    import views
    for i in range(1, 15):
        assert any(name.startswith(f"page_{i}_") for name in dir(views)), \
            f"views package missing page_{i}_*"
