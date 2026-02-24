# Evaluation Datasets Research

## Overview

This document captures the dataset choices for the evaluation plugin, which will measure retrieval quality on top of the FAISS index (coming in Tom's plugin). Two datasets are chosen — one text, one image — both well-known benchmarks that ML practitioners will recognise, and small enough to run locally without LLM-scale infrastructure.

---

## Text Retrieval: BeIR/scifact

### Why scifact

**BEIR** (Benchmarking Information Retrieval) is the industry standard for evaluating text embedding models. Cohere, Voyage AI, OpenAI, and Mistral all report on it. SciFact is the smallest dataset in the BEIR collection that's still meaningful:

| Property | Value |
|----------|-------|
| Corpus size | 5,183 documents |
| Test queries | 300 |
| Domain | Scientific fact-checking |
| Task | Given a claim, retrieve supporting/refuting evidence |
| HuggingFace ID | `BeIR/scifact` |
| Licence | CC BY 4.0 |
| Size on disk | ~10MB |
| Requires login | No |

**Why not larger BEIR datasets:**
- MS MARCO: 8.8M passages — too large for local iteration
- Natural Questions: 300K+ corpus — heavy
- SciFact gives a credible result while being runnable in seconds

**Upgrade path** when we want to stress test: `BeIR/trec-covid` (50 queries, 171K corpus, same format).

### What to Load

```
corpus (subset="corpus"):
    doc_id: str
    text: str
    title: str

queries (subset="queries"):
    query_id: str
    text: str

qrels (subset="qrels", split="test"):
    query_id: str
    doc_id: str
    score: int  # 0 = not relevant, 1 = relevant, 2 = highly relevant
```

### Ground Truth Format

BEIR uses a standard qrels format:

```python
# qrels as a nested dict: { query_id: { doc_id: relevance_score } }
{
    "0": {"4983": 1, "5129": 2},
    "1": {"2031": 1},
    ...
}
```

Most queries have 1–3 relevant documents against a corpus of 5K. For binary evaluation, treat score >= 1 as relevant.

### Loading with HuggingFace `datasets`

```python
from datasets import load_dataset

corpus = load_dataset("BeIR/scifact", "corpus", split="corpus")
queries = load_dataset("BeIR/scifact", "queries", split="queries")
qrels = load_dataset("BeIR/scifact-qrels", split="test")

# corpus rows: {"_id": "4983", "title": "...", "text": "..."}
# queries rows: {"_id": "0", "text": "Antigens adjust protein expression..."}
# qrels rows: {"query-id": "0", "corpus-id": "4983", "score": 1}
```

Note: BEIR datasets have two separate HuggingFace repos per dataset:
- `BeIR/scifact` — corpus and queries
- `BeIR/scifact-qrels` — relevance labels

### Evaluation Metrics (BEIR standard)

| Metric | K values | Notes |
|--------|----------|-------|
| **NDCG@10** | Primary | Standard BEIR leaderboard metric |
| **Recall@100** | Secondary | Measures retrieval ceiling |
| Recall@10 | Reported | Good for top-K retrieval quality |
| MRR | Optional | When there's typically one right answer |

---

## Image Retrieval: nlphuji/flickr30k

### Why Flickr30K

**Flickr30K** (Young et al., 2014) is one of the most cited image-text retrieval benchmarks with 5000+ citations. It's used to evaluate CLIP, ALIGN, BLIP, Florence, and every significant vision-language model. Any ML practitioner working with multimodal models will recognise it.

| Property | Value |
|----------|-------|
| Total images | 31,783 |
| Captions per image | 5 human-written |
| Test split | 1,000 images (5,000 captions) |
| Domain | General (Flickr photos) |
| HuggingFace ID | `nlphuji/flickr30k` |
| Test split size on disk | ~300MB |
| Full dataset size | ~2GB |
| Requires login | Yes (HuggingFace account, free) |

**Why not COCO:** MS-COCO has 330K images (~10GB) — same format, just much larger. Flickr30K gives the same benchmark credibility at 1/5th the size.

**Why Flickr30K pairs with our CLIP embedder:** CLIP was trained on image-text pairs and Flickr30K is the canonical text-to-image retrieval benchmark. Reporting R@1/R@5/R@10 on Flickr30K is instantly interpretable by anyone familiar with CLIP.

### What to Load (test split)

```
Each row (test split only — 1,000 images):
    image: PIL.Image          # The actual image
    caption: str              # One of the 5 captions
    img_id: int               # Unique image identifier
    filename: str             # Original Flickr filename
    split: str                # "test"
```

Since each image has 5 captions, the test split has 5,000 rows total (1,000 images × 5 captions).

### Ground Truth Format

For **text-to-image retrieval** (given a caption, find the matching image):
- Each caption has exactly 1 correct image (the one it was written for)
- The 999 other test images are distractors
- Relevance: `retrieved_img_id == query_caption_img_id`

```python
# Ground truth is implicitly encoded in img_id:
# For caption with img_id=123456, the correct retrieval is image 123456
# All other images are negatives

# To build ground truth lookup:
ground_truth = {caption_idx: img_id for caption_idx, img_id in enumerate(captions_df["img_id"])}
```

For **image-to-text retrieval** (given an image, find its matching captions):
- Each image has 5 correct captions
- Treat any caption with the same `img_id` as relevant

### Loading with HuggingFace `datasets`

```python
from datasets import load_dataset
import io
from PIL import Image

# Load only the test split (300MB vs 2GB for full)
dataset = load_dataset("nlphuji/flickr30k", split="test")

# Each row: {"image": PIL.Image, "caption": str, "img_id": int, "filename": str, "split": str}
# Convert PIL image to bytes for our pipeline:
for row in dataset:
    img_bytes = io.BytesIO()
    row["image"].save(img_bytes, format="PNG")
    image_data = img_bytes.getvalue()
```

Note: `nlphuji/flickr30k` requires a free HuggingFace account. Alternative mirror: `Multimodal-Fatima/Flickr30k_test_80k` (no login, test subset only).

### Evaluation Metrics (Flickr30K standard)

| Metric | Notes |
|--------|-------|
| **R@1** | Primary — standard Flickr30K metric, top-1 accuracy |
| **R@5** | Primary — standard Flickr30K metric |
| **R@10** | Primary — standard Flickr30K metric |
| MRR | Optional, less commonly reported for image retrieval |

These three numbers are what every CLIP paper reports on Flickr30K, making our results directly comparable to published baselines.

**CLIP baseline (ViT-B/32, text-to-image):**
- R@1: ~65%, R@5: ~87%, R@10: ~92%

This gives a known reference point to verify the evaluation pipeline is working correctly.

---

## Evaluation Metrics: Implementation Plan

All metrics will be implemented from scratch — simple formulas, no extra dependencies.

### Recall@K

```python
def recall_at_k(relevant_ids: set, ranked_ids: list, k: int) -> float:
    retrieved = set(ranked_ids[:k])
    return len(relevant_ids & retrieved) / len(relevant_ids) if relevant_ids else 0.0
```

### MRR (Mean Reciprocal Rank)

```python
def reciprocal_rank(relevant_ids: set, ranked_ids: list) -> float:
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0
```

### NDCG@K

```python
import math

def ndcg_at_k(relevant_scores: dict, ranked_ids: list, k: int) -> float:
    dcg = sum(
        relevant_scores.get(doc_id, 0) / math.log2(rank + 1)
        for rank, doc_id in enumerate(ranked_ids[:k], start=1)
    )
    ideal = sorted(relevant_scores.values(), reverse=True)[:k]
    idcg = sum(score / math.log2(rank + 1) for rank, score in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else 0.0
```

---

## Dependencies to Add Later

```toml
[project.optional-dependencies]
eval = [
    "datasets>=2.0.0",   # HuggingFace dataset loading
    # Pillow already in 'advanced' — reuse for image decoding
]
```

Metric implementations require only Python stdlib (`math`).

---

## Summary

| | Text | Image |
|--|------|-------|
| **Dataset** | BeIR/scifact | Flickr30K |
| **Size** | ~10MB | ~300MB (test split) |
| **Queries** | 300 | 5,000 captions |
| **Corpus** | 5,183 docs | 1,000 images |
| **Primary metric** | NDCG@10 | R@1, R@5, R@10 |
| **Known baseline** | BEIR leaderboard | CLIP ViT-B/32 results |
| **HuggingFace** | `BeIR/scifact` + `BeIR/scifact-qrels` | `nlphuji/flickr30k` |
| **Free access** | Yes | Yes (free HF account) |

---

## Next Steps 

1. Build `HuggingFaceTextDataset` FeatureGroup — loads scifact corpus/queries/qrels into mloda
2. Build `HuggingFaceImageDataset` FeatureGroup — loads Flickr30K test split into mloda
3. Build `RetrievalEvaluator` FeatureGroup — computes Recall@K, MRR, NDCG against FAISS results
4. Wire together: Dataset → Embed → FAISS index → Evaluate
