# Multimodal Research: Images

**Research Goal:** Determine if mloda's swappable component pattern (3 alternatives per component) extends to image pipelines for multimodal database support.

**Date Started:** February 3, 2026  
**Date Completed:** February 4, 2026  
**Researcher:** Manoj Krishna Mohan

---

## Executive Summary

✅ **Research Status:** Complete - ready for implementation

**Key Findings:**
1. Image pipeline follows same pattern as text: `images__pii_redacted__embedded__stored`
2. Three swappable providers identified for each component
3. No chunking needed (images are atomic)
4. Row-by-row processing with disk I/O (memory constraints)
5. CLIP embeddings enable multimodal search (text queries → image results)

**Components Ready:**
- ✅ **Ingestion:** LocalFile, S3, URL (3 providers)
- ✅ **PII Redaction:** YuNet, GroundedSAM, CloudAPI (3 providers)
- ✅ **Embedding:** CLIP, DINOv2, ResNet (3 providers)
- ⚠️ **Storage:** Needs verification (hypothesis: same as text)

**Architecture Fit:**
- Proposal 8 (Framework Split): Less relevant for images - PIL/OpenCV are framework-agnostic
- Proposal 9 (Provider Swapping): Perfect fit - CLIP↔DINOv2↔ResNet, YuNet↔GroundedSAM↔CloudAPI

---

## Image Pipeline Structure

**Text pipeline (reference):**
```
docs__pii_redacted__chunked__embedded__stored
```

**Image pipeline (proposed):**
```
images__pii_redacted__embedded__stored
```

**Key difference:** No chunking for images (atomic units)

---

## Research Questions

### 1. Image Embeddings
**Question:** Are there 3 viable image embedding providers/models?

**Known options:**
- CLIP (OpenAI's multimodal model)
- ResNet (traditional CNN)
- ?

**To research:**
- [ ] What are the available image embedding models?
- [ ] Which have good library support?
- [ ] What are embedding dimensions for each?
- [ ] Performance/quality comparison
- [ ] API vs local deployment options

### 2. PII Redaction for Images
**Question:** What constitutes PII in images and what are 3 provider options?

**Potential PII types:**
- Faces
- License plates
- Document text (SSN, credit cards in photos)
- Location metadata (EXIF data)

**To research:**
- [ ] Face detection/blurring libraries
- [ ] License plate detection
- [ ] OCR + text PII detection (hybrid approach?)
- [ ] Metadata scrubbing

### 3. Storage
**Question:** Do the same vectorDBs work for image embeddings?

**Hypothesis:** Yes, vectorDBs are format-agnostic (just vectors)

**To verify:**
- [ ] Confirm pgvector works with image embeddings
- [ ] Confirm FAISS works with image embeddings
- [ ] Any image-specific storage considerations?

### 4. Data Flow
**Question:** What does the data structure look like through the pipeline?

**Input:**
```python
data = {
    'image_id': [1, 2, 3],
    'image_path': ['/img1.jpg', '/img2.png', '/img3.jpg']
}
```

**After PII redaction:**
```python
# Option A: Modified paths
data = {
    'image_id': [1, 2, 3],
    'image_path': ['/redacted/img1.jpg', '/redacted/img2.png', '/redacted/img3.jpg']
}

# Option B: Metadata tracking
data = {
    'image_id': [1, 2, 3],
    'image_path': ['/img1.jpg', '/img2.png', '/img3.jpg'],
    'pii_detected': [True, False, True]
}
```

**After embedding:**
```python
data = {
    'image_id': [1, 2, 3],
    'image_path': [...],
    'embedding': [vec1, vec2, vec3]  # numpy arrays
}
```

---

## Architecture Validation

### Does Proposal 9 (Provider Inheritance) work?

**Text example:**
```python
BaseEmbedder
├── OpenAITextEmbedder
├── CohereTextEmbedder
└── SentenceTransformersEmbedder
```

**Image equivalent:**
```python
BaseImageEmbedder
├── CLIPImageEmbedder
├── ResNetImageEmbedder
└── ???  # Need third option
```

**Question:** Is there a viable third image embedding provider?

### Does Proposal 8 (Framework Split) work?

**Text example:**
```python
BaseChunker
├── PandasChunker
├── PolarsChunker
└── SparkChunker
```

**Question:** Do we need framework variants for image operations?
- Image loading is typically done with PIL/OpenCV (framework-agnostic)
- Embedding happens in the model (framework-agnostic)
- Only storage layer interacts with DataFrame-like structures

**Hypothesis:** Proposal 8 may be less relevant for images, or only applies at storage layer.

---

## Multimodal Database Vision

**End goal:** Single database handling both text and images

**Questions:**
- How do text and image embeddings coexist in the same vectorDB?
- Separate collections/tables or unified?
- Cross-modal search (text query → image results)?
- Metadata schema for mixed modalities?



---

## Research Notes

### Image Embeddings

**Research completed:** February 3, 2026

**Available models (5+ viable options):**

1. **CLIP (OpenAI/OpenCLIP)** - Multimodal winner
   - Variants: ViT-B/32, ViT-L/14, RN50
   - Embedding dimensions: 512 or 768
   - **Critical advantage:** Same embedding space for text AND images (enables text→image search)
   - Library: `pip install open-clip-torch` or `pip install git+https://github.com/openai/CLIP.git`
   - Use case: Multimodal search, zero-shot classification

2. **DINOv2 (Meta)** - Accuracy champion
   - 93% accuracy on Food-101 vs CLIP's 88%
   - Best for fine-grained classification (70% on 10,000 species vs CLIP's 15%)
   - Self-supervised learning approach
   - **Limitation:** Image-only (no text search capability)
   - Library: Available through HuggingFace

3. **ResNet (Classic CNN)** - Speed champion
   - ResNet-18: 512 dimensions
   - ResNet-50: 2048 dimensions
   - Smallest and fastest to compute
   - Good for image-to-image similarity
   - **Limitation:** Image-only
   - Library: Built into PyTorch/torchvision

4. **Vision Transformer (ViT)** - Modern architecture
   - Multiple sizes: Base, Large, Huge
   - Consistent performance across datasets
   - Balances accuracy and efficiency
   - **Limitation:** Image-only

5. **ConvNeXt** - Modern CNN hybrid
   - 93% accuracy, outperforms ViT-B on some tasks
   - Modern CNN architecture with transformer-like design
   - **Limitation:** Image-only

**Key distinction for RAG/multimodal database:**
- **CLIP:** Multimodal (text queries retrieve images)
- **All others:** Image-only (only image→image similarity)

**Recommendation for mloda-RAG:**
CLIP is essential for multimodal database because it enables text-based image search. Other models (DINOv2, ResNet) could be alternatives for pure image-to-image similarity use cases.

**Three swappable providers:**
1. **CLIPEmbedder** - For multimodal search
2. **DINOv2Embedder** - For accuracy-critical image classification
3. **ResNetEmbedder** - For speed/resource-constrained environments

**Questions remaining:**
- Do all 3 need to support the same interface?
- How to handle text queries with non-CLIP models? (fallback? error?)
- Embedding dimension differences (512 vs 768 vs 2048) - does storage layer care?

### Image Ingestion


**Purpose:** Accept image sources and create data structure with paths/references.

**Input sources:**
1. **Local filesystem** - Direct file paths
2. **S3/Cloud storage** - S3 URIs (s3://bucket/key)
3. **HTTP/HTTPS URLs** - Remote images

**Data structure after ingestion:**
```python
data = {
    'image_id': [1, 2, 3],
    'image_path': ['/local/img1.jpg', 's3://bucket/img2.jpg', 'https://site.com/img3.jpg'],
    'storage_type': ['local', 's3', 'http']  # Optional metadata
}
```

**Three swappable providers:**
1. **LocalFileIngestion** - Reads from local filesystem
2. **S3Ingestion** - Reads from S3 buckets (using boto3)
3. **URLIngestion** - Downloads from HTTP/HTTPS

**Key decisions:**
- Keep as paths (not loaded images) - memory efficient
- Each downstream step loads as needed
- Matches text pattern (strings as lightweight references)

**Libraries:**
- PIL/Pillow for image loading when needed
- boto3 for S3 access
- urllib/requests for URL downloads

---

### PII Redaction

**Research completed:** February 3, 2026

**Use cases:**
- Healthcare: Patient photos
- Surveillance: Privacy protection
- **Note:** Optional component (not universal like text PII)

**Traditional Face Detection Options (5+ libraries):**

1. **YuNet (OpenCV)** - Speed champion
   - Fast enough for real-time applications
   - Great accuracy for frontal faces
   - Built into OpenCV

2. **Dlib HoG** - Fast but limited
   - Fast face detection based on Histogram of Oriented Gradients
   - Fails on faces smaller than ~70x70 pixels
   - Good for minimizing false positives

3. **Dlib CNN (MMOD)** - Robust
   - Handles framing occlusions and various angles
   - More accurate than HoG but slower
   - Good with non-frontal faces

4. **RetinaFace** - Accuracy champion
   - Incredibly accurate, especially for small faces
   - Slower than YuNet but significantly more accurate
   - Good balance for production use

5. **OpenCV DNN** - Modern baseline
   - Deep learning detector, overcomes Haar cascade limitations
   - Good general-purpose option

**Prompt-Based Detection (Flexible):**

**Grounded SAM (Grounding DINO + SAM):**
- Combines zero-shot detection with flexible segmentation
- Text prompts: "face", "license plate", "phone number", "text"
- Enables detection and segmentation of ANY PII type with one model
- Zero-shot capabilities (works without training)
- Slower than specialized models but highly flexible

**How it works:**
1. Text prompt → Grounding DINO detects bounding boxes
2. Bounding boxes → SAM creates segmentation masks
3. Apply Gaussian blur to masked regions
4. Save redacted image

**Three swappable providers:**
1. **YuNetPIIDetector** - Fast, face-only (real-time capable)
2. **GroundedSAMDetector** - Flexible, prompt-based (any PII type: faces, plates, text)
3. **CloudAPIPIIDetector** - Managed service (AWS Rekognition, Azure Face API)

**Libraries:**
- OpenCV (cv2) for traditional detection + blurring
- Grounding DINO + SAM for prompt-based
- boto3 for AWS Rekognition
- Azure Computer Vision SDK

**Blurring technique:** `cv2.GaussianBlur()` applied to detected regions

### Storage Considerations

**Hypothesis:** VectorDBs are format-agnostic - they only store vectors.

**Image embeddings should work identically to text embeddings in:**
- pgvector (PostgreSQL extension)
- FAISS (Facebook AI Similarity Search)
- Qdrant
- ChromaDB
- DuckDB with vss extension

**Storage layer requirements:**
1. Accept vectors of various dimensions (512, 768, 2048)
2. Store metadata (image_id, image_path, storage_type)
3. Support similarity search (cosine, L2)
4. Handle both text and image embeddings (multimodal)

**To verify before implementation:**
- Test FAISS with image embeddings (CLIP 512-dim, ResNet 2048-dim)
- Confirm pgvector handles different dimensions
- Test metadata storage patterns

**Expected outcome:** No changes needed - vectors are vectors regardless of source

---

## Data Flow & Memory Management

**Critical architectural insight:** Images are heavyweight (5MB each × 1000 = 5GB)

**Text pipeline (lightweight):**
```python
# All in memory
data = pd.DataFrame({'text': ['doc1', 'doc2', ...]})  # ~1KB per doc
data['chunks'] = chunk(data['text'])  # Expands rows
data['embeddings'] = embed(data['chunks'])  # ~1KB per chunk
store(data)
```

**Image pipeline (heavyweight):**
```python
# Row-by-row with disk I/O
for img_path in data['image_path']:
    img = load(img_path)              # Load ONE image (5MB)
    img_redacted = pii_redact(img)    # Process
    save(img_redacted, new_path)      # Save to disk
    del img, img_redacted             # Free memory (critical)
    
    # Next step loads from new_path
    img = load(new_path)
    embedding = embed(img)            # Process
    store(embedding)                  # Store vector (512 floats = 2KB)
    del img                           # Free memory
```

**Key difference:** Accept disk I/O overhead to maintain bounded memory

**Pattern is identical, only memory management differs:**
- Text: Batch in memory (lightweight strings)
- Images: Row-by-row with disk (heavyweight binary)

---

## Architecture Mapping

### Proposal 9 (Provider Inheritance) - Perfect Fit

**Embedding providers:**
```python
BaseImageEmbedder
├── CLIPImageEmbedder        # Multimodal search
├── DINOv2ImageEmbedder      # Accuracy champion
└── ResNetImageEmbedder       # Speed champion
```

**PII providers:**
```python
BasePIIDetector
├── YuNetPIIDetector          # Fast, face-only
├── GroundedSAMDetector       # Flexible, prompt-based
└── CloudAPIPIIDetector        # Managed service
```

**Ingestion providers:**
```python
BaseImageIngestion
├── LocalFileIngestion
├── S3Ingestion
└── URLIngestion
```

### Proposal 8 (Framework Split) - Partially Relevant

**Text example:**
```python
BaseChunker → PandasChunker, PolarsChunker, SparkChunker
```

**Image case:**
- Image loading: PIL/OpenCV (framework-agnostic)
- Embedding: Model inference (framework-agnostic)
- Storage: Vectors only (framework-agnostic)

**Conclusion:** Proposal 8 less relevant for images. Only applies if storing intermediate results in DataFrames (Pandas vs Polars vs Spark).

---



## Ready for Implementation

### Component 1: Image Ingestion

```python
LocalFileIngestion
- Input: List of file paths
- Output: {image_id, image_path, storage_type}
- Dependencies: None (creates initial data)

S3Ingestion
- Input: S3 URIs (s3://bucket/key)
- Output: {image_id, image_path, storage_type}
- Dependencies: boto3

URLIngestion
- Input: HTTP/HTTPS URLs
- Output: {image_id, image_path, storage_type}
- Dependencies: requests or urllib
```

### Component 2: PII Redaction (Optional)

```python
YuNetPIIDetector
- Input: {image_path}
- Output: {image_path_redacted}
- Speed: Real-time capable
- Use case: Face-only, high volume
- Dependencies: opencv-python

GroundedSAMDetector
- Input: {image_path, pii_prompt}
- Output: {image_path_redacted}
- Speed: 10-100x slower than YuNet
- Use case: Flexible (faces, plates, text, phones)
- Dependencies: transformers, groundingdino, segment-anything

CloudAPIPIIDetector
- Input: {image_path}
- Output: {image_path_redacted}
- Speed: Network latency dependent
- Use case: Managed service, pay-per-use
- Dependencies: boto3 or azure-cognitiveservices-vision
```

### Component 3: Image Embedding

```python
CLIPImageEmbedder
- Input: {image_path}
- Output: {embedding: 512-dim vector}
- Critical: Enables text→image search (multimodal)
- Dependencies: open-clip-torch or transformers

DINOv2ImageEmbedder
- Input: {image_path}
- Output: {embedding: 768-dim vector}
- Use case: Accuracy-critical classification
- Limitation: Image-only (no text queries)
- Dependencies: transformers

ResNetImageEmbedder
- Input: {image_path}
- Output: {embedding: 512 or 2048-dim vector}
- Use case: Speed-critical, resource-constrained
- Limitation: Image-only
- Dependencies: torchvision
```

### Component 4: Storage

```python
PgVectorStorage
- Input: {embedding, metadata}
- Output: Stored in PostgreSQL with pgvector extension
- Query: Similarity search (cosine, L2)

FAISSStorage
- Input: {embedding, metadata}
- Output: Stored in FAISS index
- Query: Fast similarity search

DuckDBStorage
- Input: {embedding, metadata}
- Output: Stored in DuckDB with vss extension
- Query: SQL-based vector search
```

---
