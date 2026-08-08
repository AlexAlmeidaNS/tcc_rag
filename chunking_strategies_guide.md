# Chunking Strategies for RAG: Deep Dive

Chunking is arguably the **most critical design decision** in a RAG pipeline. Get it wrong and your retrieval will be poor no matter how good your embeddings or LLM are.

## 🎯 Why Chunking Matters

**The Core Problem**: Embedding models have token limits (typically 512-8192 tokens) and work best on semantically coherent text segments. Your official gazettes can be tens of thousands of tokens long.

**The Goal**: Split documents into chunks that:
* Fit within embedding model limits
* Contain complete, self-contained ideas
* Preserve enough context to be useful when retrieved
* Enable precise retrieval (not too broad, not too narrow)

---

## 📊 Chunking Strategy Taxonomy

### **1. Fixed-Size Chunking**

**How it works**: Split text every N tokens/characters, regardless of content boundaries.

```
Chunk 1: [0:512 tokens]
Chunk 2: [512:1024 tokens]
Chunk 3: [1024:1536 tokens]
```

**Pros**:
* Simple to implement
* Predictable chunk count and size
* Fast processing

**Cons**:
* **Cuts sentences/paragraphs mid-thought** (biggest issue!)
* Loses semantic coherence
* Can split tables, lists, or structured content awkwardly
* Poor for legal/government text where complete clauses matter

**When to use**: Quick prototypes, highly uniform text (logs, chat transcripts)

**Verdict for your gazettes**: ❌ **Not recommended** - government text has complex structure

---

### **2. Recursive Character Splitting (Hierarchical)**

**How it works**: Split on hierarchy of separators, trying to preserve structure:
1. First try: split on double newlines (paragraphs)
2. If chunks too large: split on single newlines
3. If still too large: split on sentences (periods)
4. Last resort: split on words or characters

**Example** (LangChain's RecursiveCharacterTextSplitter):
```python
separators = ["\n\n", "\n", ". ", " ", ""]
```

**Pros**:
* Respects document structure
* Keeps paragraphs intact when possible
* More semantically coherent than fixed-size
* Widely used (LangChain default)

**Cons**:
* Variable chunk sizes (can be inefficient for indexing)
* May not respect domain-specific structure (legal articles, numbered sections)
* Paragraph boundaries may not align with semantic boundaries

**When to use**: General-purpose text, articles, documentation

**Verdict for your gazettes**: ⚠️ **Good starting point**, but may need customization for legal structure

---

### **3. Sentence-Based Chunking**

**How it works**: Use NLP sentence segmentation, then group N sentences per chunk.

```
Chunk 1: Sentences 1-5
Chunk 2: Sentences 6-10
Chunk 3: Sentences 11-15
```

**Pros**:
* Natural semantic boundaries
* Complete thoughts preserved
* Works well with spaCy or NLTK sentence tokenizers

**Cons**:
* Sentence length varies wildly (especially in legal text with long clauses)
* May create very small or very large chunks
* Sentence detection can fail on non-standard formatting
* Computationally more expensive

**When to use**: Narrative text, news articles, books

**Verdict for your gazettes**: ⚠️ **Use with caution** - legal sentences can be 100+ words

---

### **4. Semantic Chunking (Embedding-Based)**

**How it works**: 
1. Split text into sentences
2. Generate embeddings for each sentence
3. Group consecutive sentences with similar embeddings
4. Split when semantic similarity drops below threshold

**Example**: LlamaIndex's `SemanticSplitterNodeParser`

**Pros**:
* **Best semantic coherence** - chunks stay on-topic
* Adapts to content (complex sections get more chunks)
* No manual separator tuning

**Cons**:
* **Much slower** - requires embedding every sentence
* More computationally expensive (2-3x processing time)
* Non-deterministic (depends on embedding model)
* Harder to debug and tune

**When to use**: High-value documents where retrieval quality is critical, smaller datasets

**Verdict for your gazettes**: ✅ **Premium option** if you can afford the compute cost

---

### **5. Document Structure-Aware Chunking**

**How it works**: Parse document structure (sections, articles, numbered items) and chunk by logical units.

**For official gazettes specifically**:
```
Chunk by:
- Article numbers ("Art. 1º", "Art. 2º")
- Section headers ("SEÇÃO I", "CAPÍTULO II")
- Decree/law boundaries
- Notices/announcements (often separated by horizontal rules)
```

**Pros**:
* **Perfect for legal/government documents**
* Preserves legal context (entire article = one chunk)
* Users can cite by article number
* Natural deduplication (same article across gazettes)

**Cons**:
* Requires domain-specific parsing logic
* Highly variable chunk sizes (articles can be 50 or 5000 tokens)
* May need size limits + secondary splitting for large articles
* Parsing can fail on poor OCR or formatting inconsistencies

**When to use**: Structured documents (legal, technical specs, regulations)

**Verdict for your gazettes**: ✅ **BEST OPTION** - gazette structure is your friend

---

### **6. Sliding Window Chunking (with Overlap)**

**How it works**: Fixed-size chunks with overlap between consecutive chunks.

```
Chunk 1: [0:800 tokens]
Chunk 2: [600:1400 tokens]  ← 200 token overlap with Chunk 1
Chunk 3: [1200:2000 tokens] ← 200 token overlap with Chunk 2
```

**Pros**:
* **Solves boundary problems** - context spans chunk edges
* Critical information near boundaries appears in 2 chunks (redundancy helps retrieval)
* Works with any base chunking strategy

**Cons**:
* Data duplication (overlap = storage/compute overhead)
* Can retrieve near-duplicate chunks
* Needs deduplication in retrieval results

**Overlap size guidelines**:
* 10-20% overlap: General use
* 20-30% overlap: Critical retrieval (medical, legal)
* 50%+ overlap: Rarely justified (excessive redundancy)

**When to use**: **ALWAYS add overlap** to your chosen strategy (it's a modifier, not a strategy)

**Verdict for your gazettes**: ✅ **Essential addition** - use 200-300 token overlap

---

## 🔍 Chunk Size Selection

**Token counts and their trade-offs**:

| Chunk Size | Best For | Pros | Cons |
|------------|----------|------|------|
| **128-256 tokens** | Very specific retrieval (Q&A, fact extraction) | Precise, fast search | Loses context, many chunks |
| **512-768 tokens** | **Balanced general-purpose** | Good context, manageable count | Standard choice |
| **1024-1536 tokens** | Long context models, complex docs | Rich context, fewer chunks | Less precise retrieval |
| **2000+ tokens** | Full-document context | Maximum context | Too broad, slow, may exceed limits |

**For Brazilian official gazettes, I recommend**:

**Base chunk size**: **800 tokens** (~600 words)
* Long enough for complete legal articles
* Short enough for precise retrieval
* Fits well in most embedding models (which accept 512-8192)

**With overlap**: **200 tokens** (25% overlap)
* Ensures context continuity
* Critical clauses near boundaries won't be split

---

## 🏗️ Recommended Strategy for Querido Diário

### **Primary: Hybrid Structure-Aware + Sliding Window**

```python
def chunk_gazette(text, metadata):
    """
    1. Parse gazette structure (Articles, Sections, Notices)
    2. For each structural unit:
       - If < 800 tokens: Keep as one chunk
       - If > 800 tokens: Apply sliding window (800 tokens, 200 overlap)
    3. Attach metadata to each chunk
    """
```

**Why this works**:
1. **Respects legal structure** - keeps articles/sections together when possible
2. **Handles outliers** - large articles get sub-chunked with overlap
3. **Preserves context** - overlap ensures no information loss at boundaries
4. **Citation-friendly** - chunks map back to source articles

### **Fallback: Recursive + Sliding Window**

If structure parsing is unreliable (poor OCR, inconsistent formatting):

```python
separators = [
    "\n\n\n",        # Multiple blank lines (section breaks)
    "Art. ",         # Article markers
    "Artigo ",       # Alternative article markers
    "\n\n",          # Paragraph breaks
    ". \n",          # Sentence + newline
    ".\n",           # Sentence + newline (no space)
    "\n",            # Line breaks
    ". ",            # Sentences
]
chunk_size = 800
chunk_overlap = 200
```

---

## 🎨 Metadata to Preserve

**Critical**: Each chunk must carry metadata for filtering and citation:

```python
chunk_metadata = {
    # Identification
    "chunk_id": "uuid",
    "gazette_id": "parent_document_id",
    "chunk_index": 0,  # Position in document
    
    # Source context (for filtering)
    "territory_name": "Brasília",
    "state_code": "DF",
    "publication_date": "2024-08-05",
    
    # Structural context
    "article_number": "Art. 42",  # If parsed
    "section": "SEÇÃO III",       # If parsed
    
    # Retrieval hints
    "chunk_token_count": 784,
    "has_decree": True,           # Boolean flags for key terms
    "has_financial_data": False,
    
    # Citation
    "source_url": "https://...",
}
```

---

## 📐 Implementation Decision Tree

```
START
  ↓
[Can you parse gazette structure reliably?]
  ↓                           ↓
 YES                         NO
  ↓                           ↓
Use structure-aware      Use recursive char
(Article/Section)        splitting with
chunking                 gazette-specific
  ↓                      separators
  ↓                           ↓
[Are chunks > 800 tokens?]    ↓
  ↓                           ↓
 YES                          ↓
  ↓                           ↓
Apply sliding window          ↓
(800 tokens, 200 overlap)     ↓
  ↓                           ↓
  └───────────┬───────────────┘
              ↓
    Add 200-token overlap
    between all chunks
              ↓
    Attach metadata to
    each chunk
              ↓
            DONE
```

---

## 🧪 Testing Your Chunking Strategy

**Before committing, test on sample data**:

```python
# 1. Chunk size distribution
display(chunks_df.groupBy("chunk_token_count_bucket").count())

# 2. Verify no information loss
original_text = full_gazette_text
reconstructed = "".join([chunk.text for chunk in chunks])
assert len(reconstructed) >= len(original_text) * 0.95  # Allow for overlap

# 3. Manual inspection
display(chunks_df.filter("chunk_token_count > 1000"))  # Check outliers
display(chunks_df.sample(0.01))  # Random sample for quality

# 4. Retrieval quality test
# Query: "decreto lei salário mínimo"
# Expected: Should return chunks with full decree text, not mid-sentence
```

---

## 💡 Key Takeaways

1. **No one-size-fits-all** - the "best" strategy depends on your data and use case
2. **Start simple** (recursive with overlap), then optimize based on retrieval quality
3. **Always use overlap** (20-25%) unless you have a very good reason not to
4. **Preserve metadata** - filtering by date/territory can 10x retrieval precision
5. **Test on real queries** - chunk quality shows up in end-to-end RAG performance

For your gazette project, I'd start with **recursive chunking (800 tokens, 200 overlap)** and iterate based on retrieval quality testing.

---

## 🚀 Implementation Plan for Querido Diário Gold Layer

### Phase 1: Basic Chunking (Week 1)
1. Implement recursive character splitter with gazette-specific separators
2. Set chunk_size=800, overlap=200
3. Preserve all silver layer metadata in each chunk
4. Write to gold Delta table

### Phase 2: Vector Search Setup (Week 2)
1. Create Vector Search endpoint
2. Create index on gold table
3. Test retrieval quality with sample queries
4. Tune chunk size/overlap based on results

### Phase 3: Structure-Aware Enhancement (Week 3+)
1. Add article/section parsing logic
2. Implement hybrid chunking (structure + sliding window)
3. A/B test retrieval quality vs simple recursive approach
4. Deploy best-performing strategy

---

**Document created**: August 5, 2026  
**Project**: Querido Diário RAG Pipeline  
**Author**: Databricks Assistant