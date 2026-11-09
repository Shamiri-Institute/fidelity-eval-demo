"""Minimal notebook executor (no jupyter available): runs code cells in order in one
namespace and records stdout, the last expression's rich repr, and matplotlib figures."""
import ast, base64, contextlib, io, json, sys, traceback
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src, dst = sys.argv[1], sys.argv[2]
nb = json.load(open(src))
outputs = []
def _display(obj):
    data = {"text/plain": repr(obj)}
    if hasattr(obj, "_repr_html_"):
        h = obj._repr_html_()
        if h: data["text/html"] = h
    outputs.append({"output_type": "display_data", "data": data, "metadata": {}})

ns = {"display": _display, "__name__": "__main__"}
plt.show = lambda *a, **k: None
count = 0; failed = False
for cell in nb["cells"]:
    if cell["cell_type"] != "code": continue
    count += 1; outputs = []
    code = "".join(cell["source"])
    buf = io.StringIO()
    try:
        tree = ast.parse(code)
        last = tree.body[-1] if tree.body and isinstance(tree.body[-1], ast.Expr) else None
        body = tree.body[:-1] if last else tree.body
        with contextlib.redirect_stdout(buf):
            exec(compile(ast.Module(body, []), f"cell{count}", "exec"), ns)
            val = eval(compile(ast.Expression(last.value), f"cell{count}", "eval"), ns) if last else None
        if buf.getvalue():
            outputs.insert(0, {"output_type": "stream", "name": "stdout", "text": buf.getvalue()})
        for num in plt.get_fignums():
            b = io.BytesIO(); plt.figure(num).savefig(b, format="png", bbox_inches="tight")
            outputs.append({"output_type": "display_data", "metadata": {},
                            "data": {"image/png": base64.b64encode(b.getvalue()).decode(), "text/plain": "<Figure>"}})
        plt.close("all")
        if val is not None:
            data = {"text/plain": repr(val)}
            if hasattr(val, "_repr_html_") and val._repr_html_(): data["text/html"] = val._repr_html_()
            outputs.append({"output_type": "execute_result", "execution_count": count, "data": data, "metadata": {}})
    except Exception as e:
        failed = True
        outputs.append({"output_type": "error", "ename": type(e).__name__, "evalue": str(e),
                        "traceback": traceback.format_exc().splitlines()})
        print(f"--- cell {count} FAILED ---\n{code[:300]}\n{traceback.format_exc()}", file=sys.stderr)
    cell["execution_count"] = count; cell["outputs"] = outputs
    # echo plain text for review
    print(f"===== cell {count}")
    for o in outputs:
        if o["output_type"] == "stream": print(o["text"], end="")
        elif "data" in o and "image/png" not in o["data"]: print(o["data"]["text/plain"][:3000])
        elif "data" in o: print("[figure]")
json.dump(nb, open(dst, "w"), indent=1, ensure_ascii=False)
sys.exit(1 if failed else 0)
