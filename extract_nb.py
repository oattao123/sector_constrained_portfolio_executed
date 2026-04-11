import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

with open("Summary.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

with open("nb_out.txt", "w", encoding="utf-8") as out:
    for i, cell in enumerate(cells):
        out.write(f"\n{'=' * 60}\n")
        out.write(f"CELL {i} | TYPE: {cell['cell_type']}\n")
        out.write("=" * 60 + "\n")
        source = "".join(cell["source"])
        out.write("--- SOURCE ---\n")
        out.write(source[:5000] + "\n")
        if cell["cell_type"] == "code" and cell.get("outputs"):
            out.write("\n--- OUTPUTS ---\n")
            for output in cell["outputs"]:
                if output.get("output_type") == "stream":
                    text = "".join(output.get("text", []))
                    out.write(text[:5000] + "\n")
                elif output.get("output_type") in ("display_data", "execute_result"):
                    data = output.get("data", {})
                    if "text/plain" in data:
                        out.write("".join(data["text/plain"])[:3000] + "\n")
                    if "text/html" in data:
                        out.write("[HTML OUTPUT]\n")
                        out.write("".join(data["text/html"])[:2000] + "\n")

print("Done. Output written to nb_out.txt")
