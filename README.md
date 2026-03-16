# CLIRAG: CLI Retrieval-Augmented Generation

![CLIRAG](https://img.shields.io/badge/Status-Hackathon_Sprint-red) ![Hardware](https://img.shields.io/badge/Hardware-Agnostic-blue) ![Memory](https://img.shields.io/badge/Memory-Strict_Isolation-green)

**CLIRAG** is a disruptive, 100% offline, hardware-agnostic Edge AI analysis engine.  
Developed by **Shourya Varshney** (Lead) and Team **Hacktrinity** for the **AMD Ryzen Slingshot Hackathon**.

## 🌍 Thematic Context & Project Vision

CLIRAG aligns directly with the hackathon themes:
- **Future of Work & Productivity:** Facilitates private, air-gapped corporate document analysis without exposing sensitive IP to cloud APIs.
- **Sustainable AI:** Focuses on energy-efficient Edge AI processing, leveraging NPUs (like AMD XDNA) instead of power-hungry cloud GPUs.

We solve the math of traditional RAG and LLMs:
- **O(1) Context Scaling:** Using State Space Models (SSMs) to overcome the O(N^2) memory exhaustion ("Context Rot") of traditional attention mechanisms.
- **Adaptive Retrieval:** Moving beyond simple Cosine Similarity (cos(θ)) to a dynamic graph/vector retrieval model for true logical cross-document relationship understanding.

## 🏗️ Architecture: The 7 Pillars of CLIRAG

```mermaid
graph TD
    User([User CLI / Typer & Rich]) --> Router{Adaptive RAG Router}
    
    subgraph "Zero-LLM Asynchronous Ingestion"
        Ingest[Document Ingest] --> NLP[C++ NLP: GLiNER/spaCy]
        Ingest --> OCR[Tesseract OCR]
        NLP --> KG(Knowledge Graph)
    end
    
    subgraph "Disk-Bound Storage (SSD)"
        KG --> Kuzu[(KùzuDB: Graph Storage)]
        NLP --> VecMetadata[(DuckDB: Vector/Metadata)]
    end
    
    subgraph "Adaptive RAG Router"
        Router --> |Factual/Shallow| BM25[BM25 Lexical]
        Router --> |Conceptual| LightRAG[LightRAG Graph Traversal]
        Router --> |Deep/Needle| ColBERT[ColBERT Late Interaction]
    end
    
    subgraph "Reasoning Engine (RLM)"
        BM25 --> REPL[Python REPL Sandbox]
        LightRAG --> REPL
        ColBERT --> REPL
        REPL --> |Constrained Func Call| Tools[Tools: search_graph, search_text, read_chunk]
    end
    
    subgraph "Heterogeneous Inference Engine"
        REPL --> HardwareProbe{Hardware Probe}
        HardwareProbe --> |CPU Only| AVX2[Force AVX2]
        HardwareProbe --> |AMD Ryzen AI| NPU[ONNX/DirectML]
        HardwareProbe --> |Dedicated GPU| VRAM[ROCm/CUDA FP16]
        AVX2 --> LlamaCPP[llama.cpp Bindings / SSMs]
        NPU --> LlamaCPP
        VRAM --> LlamaCPP
    end
    
    LlamaCPP --> Output[TTFT Streaming < 1s]
```

## 💻 Minimum Specifications

- **RAM:** 8GB Minimum (Strict RAM Isolation enforced)
- **Storage:** NVMe/SSD required for DuckDB and KùzuDB disk-bound querying
- **CPU:** AVX2 Instruction Set Support
- **OS:** Windows (AMD Ryzen AI support) / Linux / macOS

## 🚀 Installation & C++ Dependencies

CLIRAG relies on ultra-fast C++ backends for NLP, specialized storage, and inference.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/clirag.git
   cd clirag
   ```

2. **Install core C++ dependencies (System level):**
   - **Tesseract OCR:** Required for image/scan parsing.
     - *Mac:* `brew install tesseract`
     - *Linux:* `sudo apt-get install tesseract-ocr`
     - *Windows:* Install via [UB-Mannheim installer](https://github.com/UB-Mannheim/tesseract/wiki).
   - **Build Tools:** Ensure you have `cmake` and a C++ compiler (`gcc`/`MSVC`) for `llama.cpp` bindings.

3. **Install Python Environment:**
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   pip install -r requirements.txt
   ```
   *Note: `requirements.txt` will pull in `llama-cpp-python`, `kuzu`, `duckdb`, `gliner`, `spacy`, `typer`, and `rich`.*

4. **Hardware-Specific Acceleration (Optional but Recommended):**
   - *For AMD NPUs:* Install ONNX Runtime with DirectML execution provider.
   - *For ROCm/CUDA:* Set the appropriate `CMAKE_ARGS` when installing `llama-cpp-python` (e.g., `CMAKE_ARGS="-DLLAMA_HIPBLAS=on" pip install llama-cpp-python`).

## 🕹️ Production Usage

CLIRAG is now optimized for **Sub-2s Query Responses** and **Instant CLI** performance.

### 1. One-Time Setup (Bootstrap)
Ensure all production-grade model weights are present:
```bash
python -m clirag.main bootstrap
```

### 2. Ingest Documents
Ingest high-priority files (PDF, Markdown, TXT) into the disk-bound store:
```bash
python -m clirag.main ingest "OOPS in Java.pdf"
```

### 3. Edge Engine (Performance Mode)
Keep models resident in RAM for near-instant answers:
```bash
python -m clirag.main serve
```

### 4. Smart Querying (Client)
In another terminal, query the database. Use `--doc` to filter:
```bash
python -m clirag.main ask "What is an object?" --doc "OOPS in Java.pdf"
```

### 5. Document Management
List all ingested files:
```bash
python -m clirag.main list
```

## 🛠️ Performance Metrics (Modernized)

- **CLI Response:** < 0.9s (Lazy Imports)
- **Engine Query:** < 2.0s (RAM Residency via Edge Engine)
- **Deduplication:** 100% hash-based (SHA256)

---
*Developed with ❤️ for the AMD Ryzen Slingshot Hackathon by Shourya Varshney (AI Warriors).*
