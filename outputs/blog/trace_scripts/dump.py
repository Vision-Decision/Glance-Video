import json, os, sys
DATA="~/Developer/decision-vison/data/research/video_index_v1_1"
def events(m,iid):
    d=os.path.join(DATA,"traces",m,m,iid.replace("^","_"))
    ev=[]
    for fn in ("stdout.jsonl","events.jsonl"):
        p=os.path.join(d,fn)
        if not os.path.exists(p): continue
        for line in open(p,errors="ignore"):
            try: e=json.loads(line)
            except Exception: continue
            t=e.get("type")
            if t=="assistant":
                for c in e["message"]["content"]:
                    if c["type"]=="text" and c["text"].strip(): ev.append(["say",c["text"]])
                    elif c["type"]=="tool_use":
                        inp=c["input"]; ev.append(["cmd",inp.get("command") or json.dumps(inp)])
            elif t=="user" and isinstance(e.get("message",{}).get("content"),list):
                for c in e["message"]["content"]:
                    if c.get("type")=="tool_result":
                        cc=c.get("content")
                        if isinstance(cc,list): cc=" ".join(z.get("text","[image]") if z.get("type")=="text" else "[image]" for z in cc)
                        ev.append(["out",str(cc)])
            elif t=="item.completed":
                it=e["item"]
                if it["type"]=="command_execution":
                    ev.append(["cmd",it["command"]]); ev.append(["out",it.get("aggregated_output","")])
                elif it["type"] in("agent_message","reasoning"): ev.append(["say" if it["type"]=="agent_message" else "reasoning",it.get("text","")])
                else: ev.append([it["type"],json.dumps(it)])
            elif t=="message" and e.get("role")=="assistant": ev.append(["say",e.get("content","")])
            elif t=="tool_use":
                p_=e.get("parameters",{}); ev.append(["cmd",f"{e.get('tool_name')}: "+(p_.get("command") or json.dumps(p_))])
            elif t=="tool_result": ev.append(["out",str(e.get("output",""))])
            elif t in ("error","turn.failed","result") or "error" in str(t): ev.append(["raw:"+str(t),json.dumps(e)[:2000]])
    m2=[]
    for k,v in ev:
        if m2 and k=="say" and m2[-1][0]=="say" and m.startswith("gemini"): m2[-1][1]+=v
        else: m2.append([k,v])
    return m2
if __name__=="__main__":
    m,iid=sys.argv[1],sys.argv[2]
    for k,v in events(m,iid):
        print(f"<<{k}>> {v if k!='out' else v[:1500]}")
