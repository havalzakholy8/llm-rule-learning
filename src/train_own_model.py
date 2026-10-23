import argparse
import csv
import itertools
import json
import os
import random

import torch
import torch.nn as nn
from sklearn.tree import DecisionTreeClassifier, export_text

SHAPES = ['square', 'triangle', 'disc']
COLORS = ['red', 'green', 'blue', 'yellow']
SIZES = ['small', 'medium', 'large']
TEXTURES = ['solid', 'striped', 'dotted']
TRANSPARENCY = ['opaque', 'semi']

FEATURE_NAMES = (
    [f'shape={s}' for s in SHAPES] + [f'color={c}' for c in COLORS] +
    [f'size={s}' for s in SIZES] + [f'texture={t}' for t in TEXTURES] +
    [f'transparency={t}' for t in TRANSPARENCY]
)

def encode(item):
    return (
        [1.0 if item['shape'] == s else 0.0 for s in SHAPES] +
        [1.0 if item['color'] == c else 0.0 for c in COLORS] +
        [1.0 if item['size'] == s else 0.0 for s in SIZES] +
        [1.0 if item['texture'] == t else 0.0 for t in TEXTURES] +
        [1.0 if item['transparency'] == t else 0.0 for t in TRANSPARENCY]
    )

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

class Net(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

def train(items, epochs, lr, seed):
    random.seed(seed)
    torch.manual_seed(seed)
    model = Net(len(FEATURE_NAMES))
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    log_rows = []
    trial = 0
    order = list(range(len(items)))

    for epoch in range(epochs):
        random.shuffle(order)
        for idx in order:
            item = items[idx]
            x = torch.tensor([encode(item)], dtype=torch.float32)
            y = torch.tensor([item['category'] - 1], dtype=torch.long)

            with torch.no_grad():
                logits = model(x)
                guess = int(torch.argmax(logits, dim=1).item()) + 1
            correct = guess == item['category']
            trial += 1
            log_rows.append({'epoch': epoch + 1, 'trial': trial, **item, 'guess': guess, 'correct': correct})

            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()

        epoch_rows = log_rows[-len(items):]
        epoch_acc = sum(r['correct'] for r in epoch_rows) / len(items)
        print(f'epoch {epoch+1}/{epochs}: accuracy on this pass = {epoch_acc:.2f}')

    return model, log_rows

def full_stimulus_space():
    combos = itertools.product(SHAPES, COLORS, SIZES, TEXTURES, TRANSPARENCY)
    return [
        {'shape': s, 'color': c, 'size': sz, 'texture': t, 'transparency': tr}
        for s, c, sz, t, tr in combos
    ]

def extract_rule(model):
    space = full_stimulus_space()
    X = [encode(item) for item in space]
    with torch.no_grad():
        preds = torch.argmax(model(torch.tensor(X, dtype=torch.float32)), dim=1).tolist()
    tree = DecisionTreeClassifier(max_depth=3, random_state=0)
    tree.fit(X, preds)
    return export_text(tree, feature_names=FEATURE_NAMES)

def probe_conjunctive_shortcuts(model):
    probes = [
        {'shape': 'square', 'color': 'green', 'size': 'medium', 'texture': 'solid', 'transparency': 'opaque'},
        {'shape': 'square', 'color': 'red', 'size': 'medium', 'texture': 'solid', 'transparency': 'semi'}
    ]
    X = [encode(p) for p in probes]
    with torch.no_grad():
        preds = (torch.argmax(model(torch.tensor(X, dtype=torch.float32)), dim=1) + 1).tolist()
    return [{'stimulus': p, 'predicted_category': c} for p, c in zip(probes, preds)]

def save_learning_curve(log_rows, path, window=10):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    corrects = [1 if r['correct'] else 0 for r in log_rows]
    rolling = [sum(corrects[max(0, i-window+1):i+1]) / min(window, i+1) for i in range(len(corrects))]
    plt.figure(figsize=(8, 4))
    plt.plot(rolling)
    plt.xlabel('trial (across all epochs)')
    plt.ylabel(f'rolling accuracy (window={window})')
    plt.title('Learning curve')
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--set', choices=['simple', 'conjunctive', 'disjunctive'], default='simple')
    parser.add_argument('--stimuli-dir', default='stimuli')
    parser.add_argument('--out-dir', default='results')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    items = load_items(args.stimuli_dir, args.set)
    model, log_rows = train(items, args.epochs, args.lr, args.seed)

    rule_text = extract_rule(model)
    print('\nextracted rule (decision tree fit to the network\'s own predictions over all possible stimuli):')
    print(rule_text)

    os.makedirs(args.out_dir, exist_ok=True)
    save_learning_curve(log_rows, os.path.join(args.out_dir, f'network_{args.set}_learning_curve.png'))

    with open(os.path.join(args.out_dir, f'network_{args.set}_trials.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        w.writeheader()
        w.writerows(log_rows)

    result = {
        'set': args.set,
        'epochs': args.epochs,
        'final_epoch_accuracy': sum(r['correct'] for r in log_rows[-len(items):]) / len(items),
        'extracted_rule': rule_text
    }
    if args.set == 'conjunctive':
        result['shortcut_probes'] = probe_conjunctive_shortcuts(model)

    with open(os.path.join(args.out_dir, f'network_{args.set}_result.json'), 'w') as f:
        json.dump(result, f, indent=2)
