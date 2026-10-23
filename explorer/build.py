"""Inject outputs/explorer_data.json into the template -> outputs/scout_room.html"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
html = (ROOT / "explorer" / "template.html").read_text()
data = (ROOT / "outputs" / "explorer_data.json").read_text()
(ROOT / "outputs" / "scout_room.html").write_text(html.replace("__DATA__", data.replace("</", "<\\/")))
print("wrote outputs/scout_room.html")
