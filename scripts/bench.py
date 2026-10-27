import urllib.request, json, time

url = 'http://127.0.0.1:11434/v1/chat/completions'
prompt = '''Förklara i detalj hur en relationsdatabas indexerar data med B-träd och LSM-träd.
Beskriv skillnaderna i läs- och skrivprestanda, hur siddelningar (page splits) och komprimering fungerar,
samt varför LSM-träd ofta föredras i distribuerade databaser som Cassandra och RocksDB medan B-träd
dominerar i traditionella relationsdatabaser som PostgreSQL och MySQL.'''

payload = {
    'model': 'gemma4:12b',
    'messages': [{'role': 'user', 'content': prompt}],
    'max_tokens': 350,
    'temperature': 0.7
}

req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json'})
t0 = time.time()
with urllib.request.urlopen(req) as resp:
    res = json.loads(resp.read().decode('utf-8'))
t1 = time.time()

usage = res.get('usage', {})
pt = usage.get('prompt_tokens', 0)
ct = usage.get('completion_tokens', 0)
dt = t1 - t0
print(f'Done in {dt:.2f}s | Prompt: {pt} tokens | Completion: {ct} tokens | Gen speed: {ct/dt:.2f} t/s')
