"""Turn typst's HTML export into a self-contained artifact page.

`typst compile --format html` emits a whole document: a `<head>` carrying the MathML
support stylesheet, and a `<body>` of semantic HTML -- `<h2>`/`<h3>`, `<figure>` with
base64 `<img>`, `<table>`, and inline `<svg>` for the cetz diagrams. The Artifact host
supplies its own `<!doctype html><head></head><body>` skeleton, so what it wants is the
page *content*: a title, a stylesheet, and the body's inner HTML.

This script does exactly that and nothing else. It does not rewrite the document -- the
typst source stays the single description of what the report says, and this file is only
responsible for how it looks.
"""

from __future__ import annotations

import pathlib
import re

HERE = pathlib.Path(__file__).parent
EXPORT = HERE / "report.html"
ARTIFACT = HERE / "report.artifact.html"

TITLE = "Planar Fault Cost"

# The palette is the study's own instrument: every headline quantity is a *signed*
# difference -- early against late, too deep against too shallow -- so the identity is
# diverging, in the manner of an anomaly chart. Deep-sea blue is the curved model (the
# reference), rust is the flat model (the error). Neutrals carry a slight blue bias so
# they read as chosen rather than inherited.
STYLE = """
:root {
  --ground: #f6f8fa;
  --surface: #ffffff;
  --ink: #131820;
  --muted: #58626f;
  --rule: #dce2e9;
  --rule-strong: #b9c3cf;
  --cool: #17527d;
  --warm: #a93a26;
  --shadow: rgba(19, 24, 32, 0.06);

  --serif: "Iowan Old Style", "Palatino Linotype", Palatino, "Book Antiqua",
           "URW Palladio L", Georgia, serif;
  --mono: "SF Mono", "IBM Plex Mono", "Cascadia Mono", "Roboto Mono", Menlo,
          Consolas, monospace;

  --measure: 34rem;
  --wide: 58rem;

  /* The matplotlib figures are baked on #fcfcfb and the cetz diagram's strokes are
     baked light-theme too, so plates keep one ground in both themes rather than
     pretending to adapt. A scientific figure reads as a plate on the page; what it
     must not do is put dark ink on a dark ground. */
  --plate: #fcfcfb;
}

@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --ground: #0e1319;
    --surface: #151b23;
    --ink: #e4e9f0;
    --muted: #8a94a2;
    --rule: #232c36;
    --rule-strong: #3a4552;
    --cool: #63ade0;
    --warm: #d4634a;
    --shadow: rgba(0, 0, 0, 0.4);
  }
}

:root[data-theme="dark"] {
  --ground: #0e1319;
  --surface: #151b23;
  --ink: #e4e9f0;
  --muted: #8a94a2;
  --rule: #232c36;
  --rule-strong: #3a4552;
  --cool: #63ade0;
  --warm: #d4634a;
  --shadow: rgba(0, 0, 0, 0.4);
}

body {
  background: var(--ground);
  color: var(--ink);
  font-family: var(--serif);
  font-size: 1.0625rem;
  line-height: 1.62;
  margin: 0;
  padding: 4rem 1.5rem 7rem;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 1.45rem;
  text-rendering: optimizeLegibility;
  -webkit-font-smoothing: antialiased;
}

/* Every child is centred on the measure; wide things opt out individually. */
body > * {
  width: 100%;
  max-width: var(--measure);
  margin: 0;
}

h2 {
  max-width: var(--wide);
  font-size: clamp(1.85rem, 4.4vw, 2.7rem);
  line-height: 1.12;
  font-weight: 600;
  letter-spacing: -0.015em;
  text-wrap: balance;
  margin: 0 0 0.8rem;
  padding-bottom: 1.5rem;
  border-bottom: 2px solid var(--rule-strong);
}

h3 {
  font-size: 1.02rem;
  font-family: var(--mono);
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--cool);
  margin: 2.6rem 0 0.1rem;
  text-wrap: balance;
}

p { margin: 0; }

strong { font-weight: 600; }

em { font-style: italic; }

code {
  font-family: var(--mono);
  font-size: 0.88em;
  background: color-mix(in srgb, var(--cool) 9%, transparent);
  padding: 0.1em 0.34em;
  border-radius: 3px;
}

/* Numbers set in mono in the source: keep them lining up and stop them shouting. */
p span[style*="monospace"], td span[style*="monospace"] {
  font-variant-numeric: tabular-nums;
}

ul {
  margin: 0;
  padding-left: 1.15rem;
  display: flex;
  flex-direction: column;
  gap: 0.55rem;
}

li::marker { color: var(--muted); }

/* --- figures ------------------------------------------------------------- */

figure {
  max-width: var(--wide);
  margin: 1.1rem 0;
  display: flex;
  flex-direction: column;
  gap: 0.7rem;
}

figure img {
  width: 100%;
  height: auto;
  display: block;
  background: var(--plate);
  border: 1px solid var(--rule);
  border-radius: 2px;
}

figcaption {
  font-size: 0.875rem;
  line-height: 1.5;
  color: var(--muted);
  max-width: 46rem;
}

/* typst numbers its own captions; the label is doing real work so let it read as one. */
figcaption { text-indent: 0; }

/* The cetz diagram comes through as inline SVG with light-theme strokes baked in, so
   it gets the same plate ground as the raster figures. */
svg {
  max-width: 100%;
  height: auto;
  display: block;
  margin: 0.6rem auto 0;
  background: var(--plate);
  border: 1px solid var(--rule);
  border-radius: 2px;
  padding: 1.4rem 0.75rem;
  box-sizing: border-box;
}

p.note {
  max-width: 40rem;
  margin: 0 auto;
  font-size: 0.875rem;
  line-height: 1.5;
  color: var(--muted);
  text-align: center;
}

/* --- tables -------------------------------------------------------------- */

table {
  max-width: var(--wide);
  width: 100%;
  border-collapse: collapse;
  font-size: 0.95rem;
  margin: 0.6rem 0;
}

th, td {
  padding: 0.55rem 0.9rem;
  text-align: left;
  vertical-align: baseline;
  border: none;
}

th {
  font-family: var(--mono);
  font-size: 0.76rem;
  font-weight: 600;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  color: var(--muted);
  border-bottom: 1px solid var(--rule-strong);
}

tr + tr td { border-top: 1px solid var(--rule); }

td:first-child { padding-left: 0; }
th:first-child { padding-left: 0; }
td:last-child, th:last-child { padding-right: 0; }

/* Data tables: figures right-aligned and lining, so a column can be scanned. */
.numeric table td:not(:first-child),
.numeric table th:not(:first-child) {
  text-align: right;
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}

.numeric { max-width: var(--wide); }
.numeric table { margin: 0.6rem 0; }

/* Anything wide scrolls in its own box rather than the page. */
.numeric, figure, table { overflow-x: auto; }

@media (max-width: 640px) {
  body { padding: 2.5rem 1.1rem 4rem; font-size: 1rem; }
  h3 { margin-top: 2.1rem; }
  th, td { padding: 0.45rem 0.55rem; }
}

@media (prefers-reduced-motion: reduce) {
  * { animation: none !important; transition: none !important; }
}
"""


def main() -> None:
    """Assemble the artifact page from typst's export."""
    source = EXPORT.read_text(encoding="utf-8")

    # typst's own stylesheet supports the MathML it emits; keep it, and put ours after
    # so ours wins on anything they both touch.
    head_styles = re.findall(r"<style>(.*?)</style>", source, flags=re.S)
    typst_style = "\n".join(head_styles)

    body = re.search(r"<body[^>]*>(.*)</body>", source, flags=re.S)
    if body is None:
        raise ValueError(
            f"{EXPORT} has no <body>; typst's export format has changed and this "
            "script's assumption about it no longer holds"
        )

    ARTIFACT.write_text(
        f"<title>{TITLE}</title>\n<style>\n{typst_style}\n{STYLE}\n</style>\n"
        f"{body.group(1)}\n",
        encoding="utf-8",
    )
    size_mb = ARTIFACT.stat().st_size / 1024**2
    print(f"wrote {ARTIFACT} ({size_mb:.2f} MB)")
    if size_mb > 15.0:
        raise ValueError(
            f"{size_mb:.1f} MB exceeds the 16 MB an artifact may be; shrink the figures"
        )


if __name__ == "__main__":
    main()
