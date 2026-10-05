"""Fetch NIST PSTAR reference data for liquid water and cache it as JSON."""
import json
import re
import urllib.parse
import urllib.request
import ssl

MATNO = "276"  # Water, Liquid
URL = "https://physics.nist.gov/cgi-bin/Star/ap_table.pl"
ENERGIES = [1, 2, 5, 10, 20, 30, 50, 80, 100, 150, 200, 250, 300]
ctx = ssl.create_default_context()


def post(extra):
    body = {"prog": "PSTAR", "matno": MATNO, "Energies": "\n".join(str(e) for e in ENERGIES)}
    body.update(extra)
    data = urllib.parse.urlencode(body).encode()
    req = urllib.request.Request(URL, data=data, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=90, context=ctx) as r:
        return r.read().decode("utf-8", "replace")


def strip(html):
    t = re.sub(r"<script.*?</script>", " ", html, flags=re.S | re.I)
    t = re.sub(r"<[^>]+>", " ", t)
    t = t.replace("&nbsp;", " ").replace("&amp;", "&")
    return re.sub(r"[ \t]+", " ", t)


def parse_rows(html):
    """Return list of numeric rows from the PSTAR result table."""
    rows = []
    for tr in re.findall(r"(?s)<tr[^>]*>(.*?)</tr>", html):
        cells = [re.sub(r"<[^>]+>", "", c).replace("&nbsp;", " ").strip()
                 for c in re.findall(r"(?s)<t[dh][^>]*>(.*?)</t[dh]>", tr)]
        if len(cells) >= 3:
            try:
                vals = [float(c) for c in cells[:4]]
                rows.append(vals)
            except ValueError:
                pass
    return rows


out = {}
print("=== total stopping power ===")
h = post({"GraphType": "StopPwr", "Graph3": "on", "Graph1": "on", "Graph2": "on"})
rows = parse_rows(h)
for r in rows:
    print(r)
out["stopping"] = rows

print("=== CSDA range ===")
h2 = post({"GraphType": "Range", "Graph5": "on", "Graph4": "on"})
rows2 = parse_rows(h2)
for r in rows2:
    print(r)
out["range"] = rows2

with open(r"<REPO_ROOT>\source\pstar_water.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1)
print("saved")
