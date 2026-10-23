import argparse
import csv
import json
import os
import sys

STIMULI_SETS = ['simple', 'conjunctive', 'disjunctive']

RULES = {
    'simple': 'Category 1: shape is a square. Category 2: shape is a triangle.',
    'conjunctive': 'Category 1: square, green, semi-transparent. Category 2: square, red, opaque.',
    'disjunctive': 'Category 1: (green square) or (red disc). Category 2: (yellow square) or (green disc).'
}

PROVIDER_DEFAULTS = {
    'anthropic': {'env': 'ANTHROPIC_API_KEY', 'model': 'claude-sonnet-4-5'},
    'openai': {'env': 'OPENAI_API_KEY', 'model': 'gpt-4o'},
    'gemini': {'env': 'GOOGLE_API_KEY', 'model': 'gemini-3.6-flash'}
}

def load_items(stimuli_dir, set_name):
    path = os.path.join(stimuli_dir, set_name, 'labels.csv')
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    return [
        {
            'shape': r['shape'], 'color': r['color'], 'size': r['size'],
            'texture': r['texture'], 'transparency': r['transparency'],
            'category': int(r['category'])
        }
        for r in rows
    ]

def describe(item):
    return f"{item['transparency']} {item['color']} {item['texture']} {item['size']} {item['shape']}"

def extract_guess(reply):
    for ch in reply:
        if ch in '12':
            return ch
    return '?'

def make_client(provider):
    if provider == 'anthropic':
        from anthropic import Anthropic
        return Anthropic()
    if provider == 'openai':
        from openai import OpenAI
        return OpenAI()
    if provider == 'gemini':
        from google import genai
        return genai.Client()
    raise ValueError(f'unknown provider {provider}')

def call_anthropic(client, model, messages):
    resp = client.messages.create(model=model, max_tokens=200, messages=messages)
    return ''.join(block.text for block in resp.content if block.type == 'text').strip()

def call_openai(client, model, messages):
    resp = client.chat.completions.create(model=model, max_tokens=200, messages=messages)
    return resp.choices[0].message.content.strip()

def call_gemini(client, model, messages):
    from google.genai import types
    contents = [
        types.Content(
            role='model' if m['role'] == 'assistant' else 'user',
            parts=[types.Part.from_text(text=m['content'])]
        )
        for m in messages
    ]
    resp = client.models.generate_content(model=model, contents=contents)
    return resp.text.strip()

CALL_FUNCS = {'anthropic': call_anthropic, 'openai': call_openai, 'gemini': call_gemini}

import time

def call_with_retry(call_model, client, model, messages, max_attempts=5):
    delay = 5
    for attempt in range(1, max_attempts + 1):
        try:
            return call_model(client, model, messages)
        except Exception as e:
            if 'RESOURCE_EXHAUSTED' in str(e) or '429' in str(e):
                raise
            if attempt == max_attempts:
                raise
            print(f"  request failed ({e}), retrying in {delay}s (attempt {attempt}/{max_attempts})...")
            time.sleep(delay)
            delay = min(delay * 2, 60)

def run(provider, stimuli_dir, set_name, model, out_dir, max_trials=None, delay_seconds=0):
    client = make_client(provider)
    call_model = CALL_FUNCS[provider]
    items = load_items(stimuli_dir, set_name)
    if max_trials:
        items = items[:max_trials]

    messages = [{
        'role': 'user',
        'content': (
            'We are running a categorization learning task. I will show you objects one at a time, '
            'each described by shape, color, size, texture and transparency. Each object belongs to '
            'Category 1 or Category 2, but I will not tell you the rule. Guess the category for each '
            'object. Reply with ONLY the digit 1 or 2, nothing else. After your guess I will tell you '
            f"whether you were right and the true category. Object 1: {describe(items[0])}. Your guess?"
        )
    }]

    log_rows = []
    correct_count = 0
    for i, item in enumerate(items):
        reply = call_with_retry(call_model, client, model, messages)
        guess = extract_guess(reply)
        correct = guess == str(item['category'])
        correct_count += correct
        log_rows.append({**item, 'trial': i + 1, 'guess': guess, 'correct': correct})
        print(f"trial {i+1}/{len(items)}: {describe(item)} -> guess {guess}, actual {item['category']}, "
              f"{'correct' if correct else 'wrong'}, running acc {correct_count/(i+1):.2f}")

        messages.append({'role': 'assistant', 'content': reply})
        feedback = f"Correct, it was category {item['category']}." if correct \
            else f"Incorrect, it was actually category {item['category']}."
        if i < len(items) - 1:
            feedback += f" Object {i+2}: {describe(items[i+1])}. Your guess?"
        else:
            feedback += (" That was the last training object. Based on everything you saw, what rule "
                         "distinguishes category 1 from category 2? State it as precisely as you can "
                         "in one or two sentences.")
        messages.append({'role': 'user', 'content': feedback})
        if delay_seconds:
            time.sleep(delay_seconds)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f'{provider}_{set_name}_trials.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        w.writeheader()
        w.writerows(log_rows)

    try:
        stated_rule = call_with_retry(call_model, client, model, messages)
        print('\nstated rule:', stated_rule)
        print('actual rule:', RULES[set_name])
    except Exception as e:
        print(f'\ncould not get the stated rule ({e}), but trial data is already saved')
        stated_rule = None

    with open(os.path.join(out_dir, f'{provider}_{set_name}_result.json'), 'w') as f:
        json.dump({
            'provider': provider, 'model': model, 'set': set_name,
            'accuracy': correct_count / len(items),
            'stated_rule': stated_rule, 'actual_rule': RULES[set_name]
        }, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--set', choices=STIMULI_SETS, default='simple')
    parser.add_argument('--provider', choices=list(PROVIDER_DEFAULTS), default='anthropic')
    parser.add_argument('--model', default=None)
    parser.add_argument('--stimuli-dir', default='stimuli')
    parser.add_argument('--out-dir', default='results')
    parser.add_argument('--max-trials', type=int, default=None, help='Limit number of trials, useful for tight free-tier quotas')
    parser.add_argument('--delay-seconds', type=float, default=0, help='Pause between calls to stay under per-minute rate limits')
    args = parser.parse_args()

    defaults = PROVIDER_DEFAULTS[args.provider]
    if not os.environ.get(defaults['env']):
        sys.exit(f"Set the {defaults['env']} environment variable first.")

    run(args.provider, args.stimuli_dir, args.set, args.model or defaults['model'], args.out_dir,
        max_trials=args.max_trials, delay_seconds=args.delay_seconds)
