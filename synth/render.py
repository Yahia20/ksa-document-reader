"""Render HTML templates to PNG with headless Chrome (Playwright)."""
from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from playwright.sync_api import sync_playwright

TEMPLATES = Path(__file__).resolve().parent / "templates"


class Renderer:
    def __init__(self):
        self.env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(["html"]))
        self._pw = sync_playwright().start()
        # Use the installed Chrome; file access lets the page load the local fonts.
        self._browser = self._pw.chromium.launch(channel="chrome", args=["--allow-file-access-from-files"])
        self._page = self._browser.new_page(device_scale_factor=1)
        self._tmp = TEMPLATES / "_render.html"

    def render(self, template: str, context: dict, out: Path) -> None:
        html = self.env.get_template(template).render(**context)
        self._tmp.write_text(html, encoding="utf-8")
        self._page.set_viewport_size({"width": 1400, "height": 900})
        self._page.goto(self._tmp.as_uri())
        self._page.evaluate("document.fonts.ready")
        size = self._page.evaluate(
            "[document.body.scrollWidth, document.body.scrollHeight]")
        self._page.set_viewport_size({"width": size[0], "height": size[1]})
        out.parent.mkdir(parents=True, exist_ok=True)
        self._page.screenshot(path=str(out), full_page=True)

    def close(self) -> None:
        self._browser.close()
        self._pw.stop()
        self._tmp.unlink(missing_ok=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
