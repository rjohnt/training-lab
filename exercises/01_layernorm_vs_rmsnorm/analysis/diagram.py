"""Generate a conceptual diagram; no measured timings are represented."""
from html import escape
from pathlib import Path

out = Path(__file__).resolve().parents[1] / "diagram/theory.svg"
out.parent.mkdir(parents=True, exist_ok=True)
a = ['''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1100 690"><defs>
<pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse"><path d="M40 0H0V40" fill="none" stroke="#1e293b" stroke-width=".5"/></pattern>
<marker id="arrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0,10 3.5,0 7" fill="#94a3b8"/></marker>
<style>text{font-family:'SF Mono','Cascadia Code',monospace;fill:#e2e8f0}</style></defs>
<rect width="100%" height="100%" fill="#0f172a"/><rect width="100%" height="100%" fill="url(#grid)"/>''']
def text(x, y, s, size=14, color="#e2e8f0", anchor="start"):
    a.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" style="fill:{color}" text-anchor="{anchor}">{escape(s)}</text>')
def box(x, y, w, title, detail, color):
    a.append(f'<rect x="{x}" y="{y}" width="{w}" height="70" rx="8" fill="#0f172a"/>')
    a.append(f'<rect x="{x}" y="{y}" width="{w}" height="70" rx="8" fill="{color}" fill-opacity=".12" stroke="{color}"/>')
    text(x+w/2, y+28, title, 15, anchor="middle")
    text(x+w/2, y+49, detail, 11, "#94a3b8", "middle")
text(30, 36, "LayerNorm vs RMSNorm: what changes in training?", 22)
text(30, 62, "Conceptual operation and storage diagram — not measured execution time", 13, "#94a3b8")
for y, norm, first, detail, color in ((110,"LayerNorm","Center each row","z = x - mean(x)","#22d3ee"),
                                    (230,"RMSNorm","Keep input values","z = x","#34d399")):
    text(35, y+40, norm, 17, color)
    for x1, x2 in ((435,475),(740,780)):
        a.append(f'<path d="M{x1} {y+35}H{x2}" stroke="#94a3b8" marker-end="url(#arrow)"/>')
    box(180,y,255,first,detail,color)
    box(475,y,265,"Scale by inverse RMS","z / sqrt(mean(z²) + eps)",color)
    box(780,y,280,"Apply learned scale","y = normalized(z) × weight",color)
text(35, 340, "Same shapes, epsilon, scale, dtype and loss. LayerNorm bias disabled.", 14)
text(35, 365, "Each norm has its own output and gradient reference; outputs need not match.", 13, "#94a3b8")
text(35, 410, "Training keeps information that inference can release", 19)
for x1,x2 in ((285,330),(565,610),(845,890)):
    a.append(f'<path d="M{x1} 477H{x2}" stroke="#94a3b8" marker-end="url(#arrow)"/>')
box(35,442,250,"Inputs + weights","baseline allocation","#22d3ee")
box(330,442,235,"Forward + loss","output and saved tensors","#a78bfa")
box(610,442,235,"Backward","input and weight gradients","#34d399")
box(890,442,175,"Release","saved buffers","#94a3b8")
text(35, 557, "Peak live allocation ≠ reserved allocator pool ≠ DRAM bytes transferred", 16, "#fbbf24")
text(35, 588, "Saved-tensor accounting deduplicates storage shared by views and inputs.", 13, "#94a3b8")
text(35, 613, "Compare native, eager and compiled families separately; implementation matters.", 13, "#94a3b8")
text(35, 648, "Timed passes exclude hooks and snapshots. Optimizer updates are outside this study.", 12, "#94a3b8")
a.append('</svg>')
out.write_text('\n'.join(a)+'\n')
