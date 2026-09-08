#!/bin/zsh
# End-to-end API checks against a running lily server (start it on 127.0.0.1:8123
# or set LILY_URL). Writes responses under tools/e2e/out/.
set -u
URL=${LILY_URL:-http://127.0.0.1:8123}
S=${LILY_E2E_OUT:-$(dirname "$0")/out}; mkdir -p $S

echo "== health"; curl -s $URL/health; echo
echo "== models"; curl -s $URL/v1/models | python3 -c 'import json,sys; print([m["id"] for m in json.load(sys.stdin)["data"]])'

echo "== 1. non-streaming chat, thinking off"
curl -s $URL/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"Qwen3.8-Flash-Next","messages":[{"role":"user","content":"What is the capital of Switzerland? Answer in one sentence."}],
  "max_tokens":64,"temperature":0,"chat_template_kwargs":{"enable_thinking":false},"prompt_cache_key":"conv-a"}' | tee $S/r1.json | python3 -c '
import json,sys; r=json.load(sys.stdin); c=r["choices"][0]; print(repr(c["message"]["content"])); print("finish", c["finish_reason"], "usage", r["usage"])'

echo "== 2a. same conversation extended with the same template settings (cache extension)"
curl -s $URL/v1/chat/completions -H 'Content-Type: application/json' -d "$(python3 -c '
import json; r=json.load(open("'$S'/r1.json")); prev=r["choices"][0]["message"]["content"]
print(json.dumps({"model":"Qwen3.8-Flash-Next","messages":[
 {"role":"user","content":"What is the capital of Switzerland? Answer in one sentence."},
 {"role":"assistant","content":prev},
 {"role":"user","content":"And of Austria? One sentence."}],
 "max_tokens":60,"temperature":0,"chat_template_kwargs":{"enable_thinking":False},"prompt_cache_key":"conv-a"}))')" | python3 -c '
import json,sys; r=json.load(sys.stdin); print(repr(r["choices"][0]["message"]["content"]), r["usage"])'

echo "== 2b. thinking on (effort medium keeps the prompt prefix), streaming with usage"
curl -sN $URL/v1/chat/completions -H 'Content-Type: application/json' -d "$(python3 -c '
import json; r=json.load(open("'$S'/r1.json")); prev=r["choices"][0]["message"]["content"]
print(json.dumps({"model":"Qwen3.8-Flash-Next","messages":[
 {"role":"user","content":"What is the capital of Switzerland? Answer in one sentence."},
 {"role":"assistant","content":prev},
 {"role":"user","content":"And of Austria? One sentence."}],
 "max_tokens":400,"stream":True,"stream_options":{"include_usage":True},"reasoning_effort":"medium","prompt_cache_key":"conv-a"}))')" > $S/r2.sse
python3 - <<EOF
import json
reasoning=content=""; finish=None; usage=None; n=0
for line in open("$S/r2.sse"):
    line=line.strip()
    if not line.startswith("data: "): continue
    body=line[6:]
    if body=="[DONE]": print("got [DONE]"); continue
    j=json.loads(body); n+=1
    if j.get("usage"): usage=j["usage"]
    for ch in j.get("choices",[]):
        d=ch.get("delta",{}); reasoning+=d.get("reasoning_content","") or ""; content+=d.get("content","") or ""
        if ch.get("finish_reason"): finish=ch["finish_reason"]
print("chunks",n,"finish",finish); print("reasoning:",repr(reasoning[:200])); print("content:",repr(content)); print("usage",usage)
EOF

echo "== 3. tool call"
curl -s $URL/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"Qwen3.8-Flash-Next",
  "messages":[{"role":"system","content":"You are a coding agent. Use tools when needed."},{"role":"user","content":"Read the file src/main.rs and tell me what it does."}],
  "tools":[{"type":"function","function":{"name":"read_file","description":"Read a file from disk","parameters":{"type":"object","properties":{"path":{"type":"string","description":"path to the file"},"limit":{"type":"integer","description":"max lines"}},"required":["path"]}}}],
  "max_tokens":600,"chat_template_kwargs":{"enable_thinking":false},"temperature":0}' | tee $S/r3.json | python3 -c '
import json,sys; r=json.load(sys.stdin); c=r["choices"][0]; m=c["message"]
print("finish", c["finish_reason"]); print("content", repr(m.get("content"))); print("tool_calls", json.dumps(m.get("tool_calls")))'

echo "== 4. tool result round trip (streamed tool call deltas)"
curl -sN $URL/v1/chat/completions -H 'Content-Type: application/json' -d "$(python3 -c '
import json; r=json.load(open("'$S'/r3.json")); m=r["choices"][0]["message"]
msgs=[{"role":"system","content":"You are a coding agent. Use tools when needed."},{"role":"user","content":"Read the file src/main.rs and tell me what it does."}, m]
for tc in (m.get("tool_calls") or []):
    msgs.append({"role":"tool","tool_call_id":tc["id"],"content":"fn main() { println!(\"hello\"); }"})
print(json.dumps({"model":"Qwen3.8-Flash-Next","messages":msgs,"tools":[{"type":"function","function":{"name":"read_file","parameters":{"type":"object","properties":{"path":{"type":"string"},"limit":{"type":"integer"}},"required":["path"]}}}],
 "max_tokens":300,"stream":True,"chat_template_kwargs":{"enable_thinking":False},"temperature":0}))')" > $S/r4.sse
python3 - <<EOF
import json
content=""; calls=[]; finish=None
for line in open("$S/r4.sse"):
    line=line.strip()
    if not line.startswith("data: ") or line=="data: [DONE]": continue
    j=json.loads(line[6:])
    for ch in j.get("choices",[]):
        d=ch.get("delta",{}); content+=d.get("content","") or ""; calls+=d.get("tool_calls",[]) or []
        if ch.get("finish_reason"): finish=ch["finish_reason"]
print("finish",finish); print("content:",repr(content[:300])); print("tool_calls:",calls)
EOF

echo "== 5. completions endpoint"
curl -s $URL/v1/completions -H 'Content-Type: application/json' -d '{"model":"x","prompt":"The three primary colours are","max_tokens":20,"temperature":0,"stop":["\n"]}' | python3 -c 'import json,sys; r=json.load(sys.stdin); print(repr(r["choices"][0]["text"]), r["choices"][0]["finish_reason"], r["usage"])'

echo "== 6. sampling with seed replays"
for i in 1 2; do curl -s $URL/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"x","messages":[{"role":"user","content":"Write one short line about autumn."}],"max_tokens":40,"seed":7,"temperature":1.0,"chat_template_kwargs":{"enable_thinking":false}}' | python3 -c 'import json,sys; print(repr(json.load(sys.stdin)["choices"][0]["message"]["content"]))'; done

echo "== 7. invalid requests"
curl -s $URL/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"x","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"x"}}]}]}'; echo
curl -s $URL/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"x","messages":[{"role":"user","content":"hi"}],"n":2}'; echo
curl -s $URL/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"x","messages":[{"role":"assistant","content":"hi"}]}'; echo

echo "== 8. cancellation: client drops a streaming request after 1.5s"
curl -sN --max-time 1.5 $URL/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"x","messages":[{"role":"user","content":"Count from 1 to 500, one number per line."}],"max_tokens":2000,"stream":true,"chat_template_kwargs":{"enable_thinking":false}}' | wc -l
sleep 1
echo "== 8b. speculation statistics in usage (completion_tokens_details when the draft head is on)"
curl -s $URL/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"x","messages":[{"role":"user","content":"Write a Python function that reverses a list, with a docstring."}],"max_tokens":120,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' | python3 -c 'import json,sys; r=json.load(sys.stdin); print(r["usage"].get("completion_tokens_details"), "completion", r["usage"]["completion_tokens"]); print(repr(r["choices"][0]["message"]["content"][:160]))'

echo "== 9. regenerate the very first prompt (identical prompt, should resume from its checkpoint)"
curl -s $URL/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"Qwen3.8-Flash-Next","messages":[{"role":"user","content":"What is the capital of Switzerland? Answer in one sentence."}],
  "max_tokens":64,"temperature":0,"chat_template_kwargs":{"enable_thinking":false},"prompt_cache_key":"conv-a"}' | python3 -c 'import json,sys; r=json.load(sys.stdin); print(repr(r["choices"][0]["message"]["content"]), r["usage"])'
echo "== done"
