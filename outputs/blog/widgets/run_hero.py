import json, sys, time, os
from glance import Glance, Box, YesNo, Choice, Score
W = sys.argv[1]
m = Glance.from_pretrained("Vision-Decision/Glance-4B-preview")
S = {
 "umbrella": ("scenes/umbrella.jpg", [Box("the rainbow umbrella"), Box("彩虹色的伞"), Box("ketchup"), Box("左边站着的女人"),
              Box("a dog"), Box("a birthday cake"), YesNo("Is it raining?"),
              Choice("Where are they eating?", ["in a garden", "in a restaurant", "on a beach", "in an office"])]),
 "bunny": ("scenes/bbb136.jpg", [Box("the big white rabbit"), Box("白色的大兔子"), Box("the purple butterfly"), Box("a flying squirrel"),
              Box("树上的松鼠"), Box("a red car"), Box("a bird"), YesNo("Is it night?")]),
 "editor": ("scenes/editor.png", [YesNo("Is a save dialog open?"), Box("the Export button"), Box("导出按钮"), Box("the Save button in the dialog"),
              Box("the Delete all button"), Box("a login form"),
              Choice("Task: export the photo as a PNG file. What is the next action?", ["click", "type", "scroll", "done"]),
              Score("How risky would it be to click 'Delete all' now?", ["low", "medium", "high", "critical"])]),
}
out = {"model": "Vision-Decision/Glance-4B-preview", "device": str(m.device), "scenes": {}}
for name, (img, qs) in S.items():
    p = os.path.join(W, img)
    m.ask(p, qs[:1])  # warm-up
    t = []
    for _ in range(3):
        s = time.time(); r = m.ask(p, qs); t.append(time.time() - s)
    from PIL import Image
    w, h = Image.open(p).size
    out["scenes"][name] = {"image": img, "size": [w, h], "mps_secs_median": round(sorted(t)[1], 2), "n": len(qs),
                          "results": [dict(vars(x)) for x in r]}
    print(name, out["scenes"][name]["mps_secs_median"], flush=True)
    for x in r: print("  ", x, flush=True)
json.dump(out, open(os.path.join(W, "hero.json"), "w"), indent=1, ensure_ascii=False, default=list)
