import json, sys, os
from glance import Glance, Box
W = sys.argv[1]
m = Glance.from_pretrained("Vision-Decision/Glance-4B-preview")
P = {
 "umbrella": ["the ketchup bottle", "番茄酱", "a red bottle", "the red squeeze bottle", "ketchup", "the sauce", "调味料", "something to drink", "喝的东西", "a glass of water", "a water bottle", "水瓶",
              "something to keep the rain off", "挡雨的东西", "somewhere to sit", "可以坐的地方", "a chair", "椅子", "the bread", "面包", "the bowl of bread", "a basket of bread",
              "甜点", "dessert", "a woman", "the woman standing on the left", "左边站着的女人", "a dog", "a cat", "a birthday cake", "a car", "汽车", "雨伞", "伞"],
 "cake": ["甜点", "dessert", "something sweet", "甜的东西", "something to drink", "喝的东西", "a drink", "饮料", "something to eat", "吃的", "the glass of water", "一杯水", "the blue glass", "蓝色的杯子",
          "the whipped cream", "奶油", "a napkin", "餐巾纸", "two forks", "叉子", "a pizza", "披萨", "a person", "一个人", "a phone", "手机", "a spoon", "勺子"],
 "wine": ["dessert", "甜点", "something sweet", "甜的东西", "something to drink", "喝的东西", "a glass of wine", "红酒", "葡萄酒", "a sandwich", "三明治", "the spoon", "勺子", "a car", "a pizza", "披萨"],
}
out = {}
for name, qs in P.items():
    p = os.path.join(W, "scenes", name + ".jpg")
    r = m.ask(p, [Box(q) for q in qs])
    out[name] = [dict(vars(x)) for x in r]
    for x in r: print(name, x, flush=True)
json.dump(out, open(os.path.join(W, "probe.json"), "w"), indent=1, ensure_ascii=False, default=list)
