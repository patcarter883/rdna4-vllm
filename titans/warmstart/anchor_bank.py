"""CAM step 1-2 — the SCALED anchor bank (the shared alignment key for the committee probe).

The committee relative-representation probe (`probe_relrep.py`) aligns models of different vocab
and hidden dim by forwarding a FIXED set of anchor texts through every frozen base and comparing the
resulting per-anchor residual geometry. For the smoke we used 16 anchors; the real canonical-Z atlas
needs the rel-rep matrix `R` to be HIGH-RANK and content-diverse, so this module scales the bank to
~128 anchors spanning the content axes the atlas must cover:

  - factual / encyclopedic NL          (knowledge-store content, use-case A)
  - science / math statements          (symbolic + technical geometry)
  - code (multiple languages)          (a distinct residual regime)
  - narrative / literary prose         (long-range NL, use-case C)
  - dialogue / instruction             (chat geometry)
  - lists / structured / tabular       (positional structure, use-case C)
  - multilingual                       (vocab-stress for tokenizer-agnostic alignment)

The bank is FIXED + reproducible: it is a literal curated list (no RNG needed — a static list IS the
reproducible artifact), order-stable, and saved to ckpt/probe/anchor_bank.pt so the atlas builder and
every committee member (local + cloud) provably see the identical anchors in the identical order.

`get_anchors(n)` returns the first `n` (default all). `save_bank(path)` dumps {anchors, sha} so a
later/cloud probe can assert byte-identical anchors. The bank is intentionally a flat python list of
strings — diversity by construction, not by sampling, so it cannot drift between runs.
"""
import hashlib
import os

import torch

# --- The curated, content-diverse anchor bank (static => reproducible by construction) -------------
# Grouped by content axis for readability; the flat ANCHORS list (built below) preserves this order.

_FACTUAL = [
    "The capital of France is Paris.",
    "Mount Everest is the highest mountain above sea level on Earth.",
    "The Pacific Ocean is the largest and deepest of the world's oceans.",
    "William Shakespeare wrote the tragedy Hamlet around the year 1600.",
    "The human heart has four chambers and pumps blood through the body.",
    "The Great Wall of China was built over many centuries by successive dynasties.",
    "Tokyo is the most populous metropolitan area in the world.",
    "The Amazon rainforest produces a significant fraction of the planet's oxygen.",
    "The Roman Empire reached its greatest territorial extent under Trajan.",
    "Antarctica is the coldest, driest, and windiest continent on Earth.",
    "The printing press was invented by Johannes Gutenberg in the fifteenth century.",
    "The Nile is one of the longest rivers in the world, flowing through northeast Africa.",
    "The United Nations was founded in 1945 after the Second World War.",
    "Penicillin, the first true antibiotic, was discovered by Alexander Fleming.",
    "The Sahara is the largest hot desert on the planet.",
    "Leonardo da Vinci painted the Mona Lisa during the Italian Renaissance.",
]

_SCIENCE_MATH = [
    "Water boils at one hundred degrees Celsius at sea level.",
    "The mitochondrion is the powerhouse of the cell.",
    "In 1969 humans first walked on the surface of the Moon.",
    "The integral of x squared is x cubed over three plus a constant.",
    "Quantum entanglement links the states of two distant particles.",
    "A triangle has three sides and its interior angles sum to one hundred eighty degrees.",
    "Photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen.",
    "Gravity causes objects to accelerate toward the Earth at nine point eight meters per second squared.",
    "The speed of light in a vacuum is approximately three hundred thousand kilometers per second.",
    "Entropy in an isolated system tends to increase over time.",
    "DNA is composed of four nucleotide bases: adenine, thymine, guanine, and cytosine.",
    "The derivative of the sine function is the cosine function.",
    "Newton's second law states that force equals mass times acceleration.",
    "A prime number is a natural number greater than one with no divisors other than one and itself.",
    "The Pythagorean theorem relates the squares of the sides of a right triangle.",
    "Electrons carry a negative charge and orbit the nucleus of an atom.",
    "The boiling point of a liquid decreases as atmospheric pressure decreases.",
    "Euler's identity states that e to the i pi plus one equals zero.",
]

_CODE = [
    "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
    "for i in range(10):\n    print(i * i)",
    "import numpy as np\narr = np.zeros((3, 3))\nprint(arr.shape)",
    "SELECT name, age FROM users WHERE age > 30 ORDER BY name;",
    "public static void main(String[] args) { System.out.println(\"Hello\"); }",
    "const sum = (a, b) => a + b;\nconsole.log(sum(2, 3));",
    "fn main() {\n    let v = vec![1, 2, 3];\n    println!(\"{:?}\", v);\n}",
    "git commit -m \"fix: handle empty input edge case\" && git push origin main",
    "class Stack:\n    def __init__(self):\n        self.items = []\n    def push(self, x):\n        self.items.append(x)",
    "#include <stdio.h>\nint main() { printf(\"%d\\n\", 42); return 0; }",
    "try:\n    result = risky()\nexcept ValueError as e:\n    log.error(e)",
    "echo $PATH | tr ':' '\\n' | sort | uniq",
    "x = [n**2 for n in range(20) if n % 2 == 0]",
    "async function fetchData(url) { const r = await fetch(url); return r.json(); }",
    "@dataclass\nclass Point:\n    x: float\n    y: float",
    "while True:\n    line = input()\n    if not line:\n        break",
]

_NARRATIVE = [
    "She sold seashells by the seashore on a bright summer morning.",
    "The river port shipped grain north to the city every autumn.",
    "The orchestra tuned their instruments before the conductor raised his baton.",
    "The ancient library held scrolls in a hundred forgotten languages.",
    "He wandered the empty streets long after the last train had departed.",
    "Rain fell softly on the tin roof as the old man told his story.",
    "The lighthouse beam swept across the dark, restless waves all night.",
    "A single candle flickered in the window of the distant farmhouse.",
    "They buried the treasure beneath the gnarled oak at the edge of the field.",
    "The detective studied the photograph for a long time before speaking.",
    "Morning mist clung to the valley as the shepherd led his flock uphill.",
    "The letter arrived three years too late, its ink faded and brittle.",
    "Children chased fireflies across the meadow until the stars came out.",
    "The clockmaker worked by lamplight, fitting each tiny gear in silence.",
    "Snow began to fall just as the travelers reached the mountain pass.",
    "The garden had grown wild in the years since anyone had tended it.",
]

_DIALOGUE = [
    "Could you please summarize the main argument of this article in two sentences?",
    "I'm not sure I follow — can you explain that step a little more slowly?",
    "Let's schedule the meeting for Thursday afternoon if that works for everyone.",
    "What would you recommend for someone just starting to learn the guitar?",
    "Honestly, I think we should reconsider the whole approach from the beginning.",
    "Thank you for your help; I really appreciate you taking the time today.",
    "Write a polite email declining the invitation but leaving the door open.",
    "Can you walk me through how to reset my password step by step?",
    "Given the budget constraints, which option do you think makes the most sense?",
    "Translate the following sentence into formal written English, please.",
    "Why did the experiment fail, and what would you change next time?",
    "Give me three reasons why this plan might not work as intended.",
]

_STRUCTURED = [
    "Ingredients: 2 cups flour, 1 cup sugar, 3 eggs, 1 teaspoon vanilla extract.",
    "Step 1: open the panel. Step 2: disconnect the cable. Step 3: replace the fuse.",
    "Name: Ada Lovelace; Born: 1815; Field: Mathematics; Known for: First algorithm.",
    "Monday: gym. Tuesday: groceries. Wednesday: dentist. Thursday: report due.",
    "Pros: cheaper, faster, simpler. Cons: less flexible, harder to extend later.",
    "Q1 revenue 4.2M, Q2 revenue 4.8M, Q3 revenue 5.1M, Q4 revenue 5.9M.",
    "Latitude 48.8566, Longitude 2.3522, Elevation 35 meters, City Paris.",
    "Error 404: the requested resource could not be found on this server.",
    "Chapter 1: Origins. Chapter 2: Conflict. Chapter 3: Resolution. Chapter 4: Aftermath.",
    "Red apples 3, green pears 5, ripe bananas 6, fresh oranges 4, total 18.",
    "Priority: high. Status: open. Assignee: unassigned. Due: end of week.",
    "Version 2.1.0 — fixed login bug, added dark mode, improved load time.",
]

_MULTILINGUAL = [
    "La capitale de la France est Paris.",
    "El gato negro duerme sobre la mesa de madera.",
    "Der schnelle braune Fuchs springt über den faulen Hund.",
    "Il sole splende luminoso sul mare calmo questa mattina.",
    "東京は日本の首都であり、世界で最も人口の多い都市の一つです。",
    "学而时习之，不亦说乎？有朋自远方来，不亦乐乎？",
    "Книга лежит на старом деревянном столе у окна.",
    "A rápida raposa marrom salta sobre o cão preguiçoso.",
    "Het regende zachtjes terwijl de oude man zijn verhaal vertelde.",
    "في الصباح الباكر، غادر المسافرون القرية الصغيرة بهدوء.",
    "Yapay zeka, makinelerin insan benzeri görevleri yapmasını sağlar.",
    "초록색 사과 세 개와 잘 익은 바나나 여섯 개가 식탁 위에 있다.",
]

ANCHORS = (
    _FACTUAL + _SCIENCE_MATH + _CODE + _NARRATIVE + _DIALOGUE + _STRUCTURED + _MULTILINGUAL
)

# Category label per anchor (parallel to ANCHORS) — carried in the saved bank so the atlas builder /
# diagnostics can group rel-rep structure by content axis.
CATEGORIES = (
    ["factual"] * len(_FACTUAL)
    + ["science_math"] * len(_SCIENCE_MATH)
    + ["code"] * len(_CODE)
    + ["narrative"] * len(_NARRATIVE)
    + ["dialogue"] * len(_DIALOGUE)
    + ["structured"] * len(_STRUCTURED)
    + ["multilingual"] * len(_MULTILINGUAL)
)
assert len(CATEGORIES) == len(ANCHORS)


def anchor_sha(anchors=None):
    """Stable content hash of the (ordered) anchor list — the reproducibility witness."""
    anchors = ANCHORS if anchors is None else anchors
    h = hashlib.sha256()
    for a in anchors:
        h.update(a.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def get_anchors(n=None):
    """Return the first n anchors (default all), order-stable."""
    return ANCHORS if n is None else ANCHORS[:n]


def get_categories(n=None):
    return CATEGORIES if n is None else CATEGORIES[:n]


def save_bank(path):
    """Persist the fixed bank {anchors, categories, sha, n} so every (local+cloud) probe can assert
    byte-identical anchors before merging cards into the atlas."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {"anchors": ANCHORS, "categories": CATEGORIES, "sha": anchor_sha(), "n": len(ANCHORS)},
        path,
    )
    return path


if __name__ == "__main__":
    from collections import Counter

    print(f"anchor bank: n={len(ANCHORS)}  sha={anchor_sha()[:16]}")
    for cat, c in Counter(CATEGORIES).items():
        print(f"  {cat:14s} {c}")
