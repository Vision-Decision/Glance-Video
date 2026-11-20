"""Turn the raw Glance outputs and ablation rows into the blog widgets' data files.

    ~/Developer/Glance/.venv/bin/python outputs/blog/widgets/make_widget_data.py <blog>/content/widgets/glance

Reads only files under outputs/ (and the images next to this script); writes hero.json, video.json, dial.json,
phrasing.json and the WebP images the widgets show. Every number is copied or computed from those files.
"""
import json, os, sys
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(sys.argv[1])
ABL = os.path.join(HERE, "..", "..", "ablation")
os.makedirs(os.path.join(OUT, "f"), exist_ok=True)


def r(x, n=4):
    return round(float(x), n)


def webp(src, dst, width, q=78):
    im = Image.open(src).convert("RGB")
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    im.save(os.path.join(OUT, dst), "WEBP", quality=q, method=6)
    return im.size


# The editor screenshot is shown cropped to its toolbar and dialog, where every
# question's target is; boxes move with the crop, and one wholly outside it has none.
CROP = {"editor": (150, 60, 900, 420)}


def cropped(name, box):
    if name not in CROP:
        return [r(v, 1) for v in box]
    x0, y0, x1, y1 = CROP[name]
    b = [min(max(box[0], x0), x1) - x0, min(max(box[1], y0), y1) - y0, min(max(box[2], x0), x1) - x0, min(max(box[3], y0), y1) - y0]
    return [r(v, 1) for v in b] if b[2] - b[0] > 2 and b[3] - b[1] > 2 else None


def show(name, src, dst, width, q=80):
    """Write the picture as shown: cropped if it has a crop, then scaled."""
    im = Image.open(src).convert("RGB")
    if name in CROP:
        im = im.crop(CROP[name])
    size = list(im.size)
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    im.save(os.path.join(OUT, dst), "WEBP", quality=q, method=6)
    return size


def question(q, x, name=""):
    k = x["kind"]
    if k == "box":
        return {"kind": "box", "text": x["text"], "p": r(x["present_prob"]), "box": cropped(name, x["box"]),
                "iou": r(x["iou_prob"]), "conf": r(x["confidence"])}
    if k == "yesno":
        return {"kind": "yesno", "text": x["question"], "p": r(x["p_yes"])}
    return {"kind": k, "text": x["question"], "probs": [[a, r(b)] for a, b in x["probs"].items()], "answer": x["answer"]}


# ---- hero: three scenes, eight questions each, one ask() call per scene
hero = json.load(open(os.path.join(HERE, "hero.json")))
credits = {
    "umbrella": ["“Dinner under the umbrella.”", "Peter Moons", "https://www.flickr.com/photos/bw-art/7488569650",
                 "CC BY 2.0", "https://creativecommons.org/licenses/by/2.0/", "COCO val2017 #275198，缩放，叠加了模型输出"],
    "bunny": ["Big Buck Bunny 2:16", "© 2008 Blender Foundation", "https://peach.blender.org/",
              "CC BY 3.0", "https://creativecommons.org/licenses/by/3.0/", ""],
    "editor": ["Glance 仓库的示例界面", "examples/gui_mock.html", "https://github.com/Vision-Decision/Glance",
               "Apache-2.0", "https://github.com/Vision-Decision/Glance/blob/main/LICENSE", ""],
}
labels = {"umbrella": "照片", "bunny": "动画", "editor": "软件界面"}
widths = {"umbrella": 640, "bunny": 640, "editor": 750}
scenes = []
for name, s in hero["scenes"].items():
    src = os.path.join(HERE, s["image"])
    size = show(name, src, f"{name}.webp", widths[name], 82)
    scenes.append({"id": name, "label": labels[name], "image": f"{name}.webp", "size": size,
                   "mps_s": s["mps_secs_median"], "credit": credits[name],
                   "questions": [question(None, x, name) for x in s["results"]]})
# 67 ms: one image, 8 questions, ablation_v2/latency/latency_table.md (the same benchmark as the vLLM baseline).
json.dump({"model": hero["model"], "gpu_ms_8q": 67, "gpu": "RTX PRO 6000", "scenes": scenes},
          open(os.path.join(OUT, "hero.json"), "w"), ensure_ascii=False, separators=(",", ":"))

# ---- video: 120 Big Buck Bunny frames, four box questions, hand-labelled truth
bbb = json.load(open(os.path.join(HERE, "bbb4.json")))
gt = json.load(open(os.path.join(HERE, "bbb_gt.json")))
names = [("rabbit", "the big white rabbit", 0), ("squirrel", "a flying squirrel", 1), ("butterfly", "a butterfly", 2),
         ("car", "a red car", 3)]
for row in bbb["rows"]:
    webp(os.path.join(HERE, "bbb4", f"t{row['t']:03d}.jpg"), f"f/t{row['t']:03d}.webp", 448, 62)
video = {"t": [row["t"] for row in bbb["rows"]], "size": [640, 360], "secs": [row["secs"] for row in bbb["rows"]],
         "queries": []}
for key, text, i in names:
    xs = [row["results"][i] for row in bbb["rows"]]
    assert all(x["text"] == text for x in xs)
    video["queries"].append({"id": key, "text": text, "p": [r(x["present_prob"]) for x in xs],
                             "box": [[r(v, 1) for v in x["box"]] for x in xs], "gt": gt["gt"][key]})
json.dump(video, open(os.path.join(OUT, "video.json"), "w"), ensure_ascii=False, separators=(",", ":"))

# ---- dial: refusal of present targets vs rejection of no-target expressions, per output head
main = {row["run"]: row for row in json.load(open(os.path.join(ABL, "main_table.json")))}
heads = [("gate", "ours_neg25", "门控"), ("digits", "ar_digits_neg25", "数字"), ("hier", "ar_hier_neg25", "层级"),
         ("atomic100", "ar_atomic100_neg25", "原子 100"), ("atomic1000", "ar_atomic1000_neg25", "原子 1000")]
grid = np.concatenate([np.geomspace(0.05, 60, 70)])
dial = {"n_present": None, "n_absent": None, "heads": []}
for key, run, label in heads:
    U = json.load(open(os.path.join(ABL, run, "rows.json")))["grefcoco_val_merged"]
    pos = np.array([x["present"] and x["n_targets"] == 1 for x in U])
    neg = np.array([not x["present"] for x in U])
    dial["n_present"], dial["n_absent"] = int(pos.sum()), int(neg.sum())
    s = np.array([1 - x["p_present"] for x in U]) if key == "gate" else np.array([x["p_none"] for x in U])
    sp, sn = np.sort(s[pos]), s[neg]
    curve = []
    for ref in grid:  # threshold that refuses ref% of present targets
        th = np.quantile(sp, 1 - ref / 100)
        curve.append([r(100 * (sp >= th).mean(), 3), r(100 * (sn >= th).mean(), 2), r(1 - th if key == "gate" else th, 4)])
    h = {"id": key, "label": label, "curve": curve, "lat": [r(main[run]["lat_1q_ms"], 1), r(main[run]["lat_8q_ms"], 1)],
         "steps": main[run].get("steps_1q", 0), "auroc": r(100 * main[run]["auroc"], 1),
         "at1": r(100 * main[run]["N_acc@1%refusal"], 1), "acc": r(100 * main[run]["refcoco_val"], 1), "rules": {}}
    h["rules"]["threshold"] = [r(100 * (s[pos] >= .5).mean(), 2), r(100 * (s[neg] >= .5).mean(), 1)]
    if key != "gate":
        C = json.load(open(os.path.join(ABL, run, "rows_c.json")))["grefcoco_val_merged"]
        g = np.array([x["greedy_none"] for x in U]); m = np.array([x["seqmap_none"] for x in C])
        h["rules"]["greedy"] = [r(100 * g[pos].mean(), 2), r(100 * g[neg].mean(), 1)]
        h["rules"]["map"] = [r(100 * m[pos].mean(), 2), r(100 * m[neg].mean(), 1)]
        # the table's own numbers, as a check
        assert abs(h["rules"]["greedy"][1] - 100 * main[run]["N_acc"]) < 0.06, (run, h["rules"]["greedy"], main[run]["N_acc"])
        assert abs(h["rules"]["map"][0] - 100 * main[run]["refusal_seqmap_val"]) < 0.06
    dial["heads"].append(h)
json.dump(dial, open(os.path.join(OUT, "dial.json"), "w"), ensure_ascii=False, separators=(",", ":"))

# ---- phrasing: the same thing, asked different ways (real outputs from probe.py)
probe = json.load(open(os.path.join(HERE, "probe.json")))
by = {name: {x["text"]: x for x in rows} for name, rows in probe.items()}
groups = {  # the cake first: there every wording works
    "cake": [("甜点", ["dessert", "甜点", "something sweet", "甜的东西"]),
             ("喝的", ["something to drink", "喝的东西", "饮料"]),
             ("图里没有的", ["a pizza", "披萨", "a phone"])],
    "umbrella": [("番茄酱", ["ketchup", "番茄酱", "the ketchup bottle", "a red bottle"]),
                 ("喝的", ["水瓶", "something to drink", "喝的东西", "a glass of water"]),
                 ("面包", ["面包", "the bread", "a basket of bread"])],
}
# The editor probes were run separately (outputs/probes/run_probe.py) on the same screenshot.
ed = json.load(open(os.path.join(HERE, "..", "..", "probes", "editor_glance_v1.json")))["probes"]["results"]
by["editor"] = {x["text"]: x for x in ed}
groups["editor"] = [("导出", ["the Export button", "导出按钮", "the button that saves the picture as a PNG"]),
                    ("保存", ["the Save button in the dialog", "对话框里的保存按钮", "the button that keeps my edits"]),
                    ("取消", ["the Cancel button in the dialog", "取消按钮", "the button that closes the dialog without doing anything"])]
pcred = {"umbrella": credits["umbrella"],
         "cake": ["“Ice capuccino and durian durian cake for #tea”", "Walter Lim", "https://www.flickr.com/photos/coolinsights/8645114496",
                  "CC BY 2.0", "https://creativecommons.org/licenses/by/2.0/", "COCO val2017 #54605，缩放，叠加了模型输出"]}
pcred["editor"] = credits["editor"]
phr = {"scenes": []}
for name, gs in groups.items():
    img = os.path.join(HERE, "scenes", name + (".png" if name == "editor" else ".jpg"))
    file = f"{name}.webp"
    size = show(name, img, file, widths.get(name, 640), 82)  # the same file as the hero's, where it has one
    phr["scenes"].append({"id": name, "image": file, "size": size, "credit": pcred[name],
                          "groups": [{"label": lab, "asks": [{"text": t, "p": r(by[name][t]["present_prob"]),
                                                              "box": cropped(name, by[name][t]["box"])} for t in ts]}
                                     for lab, ts in gs]})
json.dump(phr, open(os.path.join(OUT, "phrasing.json"), "w"), ensure_ascii=False, separators=(",", ":"))
print("ok", OUT)
