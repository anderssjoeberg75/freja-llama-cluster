#!/usr/bin/env bash
# Smoke test: response quality, speed (timings) and endpoint availability.
URL=http://127.0.0.1:11434
curl -s $URL/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"gemma4:12b","messages":[{"role":"user","content":"Förklara kort vad blodtryck är, på svenska."}],"max_tokens":300}' \
  | python3 -c 'import sys,json; r=json.load(sys.stdin); print(r["choices"][0]["message"]["content"][:500]); t=r["timings"]; print("prompt t/s:", round(t["prompt_per_second"],1), "| gen t/s:", round(t["predicted_per_second"],1), "| draft accepted:", t.get("draft_n_accepted"), "/", t.get("draft_n"))'
echo "---"
BODY='{"model":"gemma4:12b","prompt":"hej","messages":[{"role":"user","content":"hej"}],"stream":false}'
for p in /api/tags /api/chat /api/generate /v1/models /v1/chat/completions; do
  code=$(curl -s -o /dev/null -w '%{http_code}' -X POST -H 'Content-Type: application/json' -d "$BODY" "$URL$p")
  echo "$p -> $code"
done
