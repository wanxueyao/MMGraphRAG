<h3 align="center">🎉 MMGraphRAG</h3>

<p align="center">
  <b>✨ A Multi-Modal Knowledge Graph RAG Framework ✨</b>
</p>

<p align="center">
  <i>From documents to multi-modal knowledge graphs — an all-in-one MMGraphRAG solution</i>
</p>

<p align="center">
  <a href="README_zh.md">🇨🇳 中文文档</a>
</p>

---

## 🌟 Key Features

<table>
<tr>
<td width="50%">

### 📊 Multi-Modal Knowledge Graph
- **Text + Image** unified modeling
- YOLO-based intelligent image segmentation
- Multi-modal entity fusion (spectral clustering)

</td>
<td width="50%">

### 🔍 Intelligent RAG Retrieval
- Semantic similarity entity retrieval
- Multi-modal context-enhanced answers
- Supports chart/table-related Q&A

</td>
</tr>
<tr>
<td width="50%">

### 🖼️ Interactive Visualization
- **Built-in Web visualization server**
- Force-directed graph browsing
- Real-time search & subgraph highlighting
- Click to view entity details

</td>
<td width="50%">

### ⚡ Flexible & Easy to Use
- One-command CLI build
- Dual engine support: MinerU / PyMuPDF
- LLM caching for faster re-runs

</td>
</tr>
</table>

---

## 📖 About The Project

![MMGraphRAG Framework](examples/paper/framework.png)

This diagram illustrates the complete workflow of MMGraphRAG.

This project is based on modifications to nano-graphrag to support multi-modal inputs (community-related code removed). The image processing component uses YOLO and Multi-modal Large Language Models (MLLM) to convert images into scene graphs. The fusion component then uses spectral clustering to select candidate entities, combining the textual knowledge graph and the image knowledge graph to construct a multi-modal knowledge graph.

Our Cross-Modal Entity Linking (CMEL) dataset is available here:

https://github.com/wanxueyao/CMEL-dataset

Here is a pre-built multimodal knowledge graph for partial documents in two datasets:
- **Google Drive**: https://drive.google.com/file/d/1PJPMS-w5NPBU2PJQa-UtmrsW-ITZlTMI/view?usp=sharing
- **Baidu Netdisk**: https://pan.baidu.com/s/1uqv09zMRWibG_0xXaCDaAQ?pwd=9i6q

---

## 🔧 Environment Setup

### Dependencies Installation

#### Core Dependencies

```bash
pip install openai                    # LLM API calls
pip install sentence-transformers     # Text embeddings
pip install networkx                  # Graph storage
pip install numpy                     # Numerical computation
pip install scikit-learn              # Vector similarity calculation
pip install Pillow                    # Image processing
pip install tqdm                      # Progress bar
pip install tiktoken                  # Text chunking token calculation
pip install ultralytics               # YOLO image segmentation
pip install opencv-python             # Image processing (cv2)
```

#### Visualization Server Dependencies

```bash
pip install flask                     # Web server framework
pip install flask-cors                # Cross-origin support
```

#### PDF Parsing Dependencies

This project supports two PDF parsing options. **Install at least one**:

| Option | Installation Command | Features |
|--------|---------------------|----------|
| **MinerU** (Recommended) | `pip install -U "mineru[all]"` | Higher parsing quality, supports complex layouts, better image context extraction |
| **PyMuPDF** | `pip install pymupdf` | Lightweight, easy installation, suitable for simple PDFs |

> **Switching**: Set `USE_MINERU = True/False` in `src/parameter.py`
>
> **Fallback**: If MinerU is unavailable, the system automatically falls back to PyMuPDF

### Model Configuration

This project requires **three types of models**, all configured in `src/parameter.py`:

#### 1. Text LLM (Required)
Used for text entity extraction, relationship building, etc. Requires an OpenAI-compatible API:

```python
API_KEY = "your-api-key"
API_BASE = "https://your-api-endpoint/v1"
MODEL_NAME = "qwen3-max"  # or other text models
```

#### 2. Multi-Modal LLM (Required)
Used for image understanding, visual entity extraction, etc. Requires an API that supports image input:

```python
MM_API_KEY = "your-api-key"
MM_API_BASE = "https://your-api-endpoint/v1"
MM_MODEL_NAME = "qwen-vl-max"  # or other multi-modal models
```

#### 3. Embedding Model (Required)
Used for entity vectorization and semantic retrieval. Configure in `src/parameter.py`:

```python
EMBEDDING_MODEL_DIR = './models/all-MiniLM-L6-v2'
EMBED_MODEL = SentenceTransformer(EMBEDDING_MODEL_DIR, device="cpu")
```

> **Tip**: The embedding model can be auto-downloaded using the model name, or manually downloaded and configured with a local path.

### MinerU Configuration

If you choose to use MinerU:

1. **Install**: `pip install -U "mineru[all]"`
2. **Configure**: See [MinerU official documentation](https://github.com/opendatalab/MinerU) for model file downloads
3. **Verify**: Ensure MinerU runs independently before proceeding

---

## ⚙️ Parameter Configuration

All core parameters are defined in `src/parameter.py`:

### Directory Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `INPUT_PDF_PATH` | Input PDF file path | - |
| `CACHE_PATH` | LLM response cache directory | `cache` |
| `WORKING_DIR` | Intermediate processing files directory | `working` |
| `OUTPUT_DIR` | Final graph output directory | `output` |
| `MMKG_NAME` | Output graph name | `mmkg_timestamp` |

### Processing Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `USE_MINERU` | Whether to use MinerU for PDF preprocessing | `True` |
| `ENTITY_EXTRACT_MAX_GLEANING` | Max iterations for text entity extraction | `0` |
| `ENTITY_SUMMARY_MAX_TOKENS` | Max tokens for entity summary | `500` |
| `SUMMARY_CONTEXT_MAX_TOKENS` | Max tokens for summary context | `10000` |

### RAG Retrieval Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `QueryParam.top_k` | Number of entities to retrieve | `5` |
| `QueryParam.response_type` | Response style type | `Detailed System-like Response` |
| `QueryParam.local_max_token_for_local_context` | Max tokens for local context | `4000` |
| `QueryParam.number_of_mmentities` | Number of multi-modal entities | `3` |
| `QueryParam.local_max_token_for_text_unit` | Max tokens for text unit | `4000` |
| `RETRIEVAL_THRESHOLD` | Retrieval similarity threshold | `0.2` |

---

## 🚀 Usage

### Quick Start

```bash
# 1️⃣ Build knowledge graph
python main.py -i path/to/your/document.pdf

# 2️⃣ Query
python main.py -q "Your question"

# 3️⃣ Launch visualization ✨
python main.py -s
# 🌐 Visit http://localhost:8080 to explore the interactive graph
```

### Building Knowledge Graph

```bash
# Build graph from specified PDF file
python main.py -i path/to/your/document.pdf

# Specify working and output directories
python main.py -i document.pdf -w ./working -o ./output

# Use PyMuPDF for PDF processing (instead of MinerU)
python main.py -i document.pdf -m pymupdf

# Force rebuild (clear working directory)
python main.py -i document.pdf -f

# Show verbose debug logs
python main.py -i document.pdf -v
```

### RAG Query

```bash
# Query the built graph
python main.py -q "Your question"

# Specify retrieval parameters
python main.py -q "Your question" --top_k 10 --response_type "Concise answer"

# If graph doesn't exist, it will be built first
python main.py -i document.pdf -q "Your question"
```

### 🖼️ Visualization Server

The built-in Web visualization server lets you intuitively explore the knowledge graph:

```bash
# Start knowledge graph visualization server
python main.py -s

# Specify port and graph file
python main.py -s --port 8888 --graph path/to/graph.graphml
```

**Visualization Highlights**:
- 🔮 **Force-Directed Layout**: Automatically optimizes node positions for clear graph structure
- 🔍 **Real-Time Search**: Quickly locate entities of interest
- 🎯 **Subgraph Highlighting**: Enter a question to highlight relevant entities and connections
- 📋 **Details Panel**: Click nodes to view entity descriptions, types, and more
- 🎨 **Type Coloring**: Different entity types use different colors for easy identification

### Command Line Arguments

| Argument | Short | Description |
|----------|-------|-------------|
| `--input` | `-i` | PDF file path |
| `--working` | `-w` | Intermediate working directory |
| `--output` | `-o` | Final output directory |
| `--method` | `-m` | PDF preprocessing method (`mineru`/`pymupdf`) |
| `--force` | `-f` | Force clear working directory and rebuild |
| `--verbose` | `-v` | Show verbose debug logs |
| `--query` | `-q` | Execute RAG query |
| `--top_k` | - | Number of entities to retrieve |
| `--response_type` | - | Response style |
| `--server` | `-s` | Start visualization server |
| `--port` | - | Server port (default: 8080) |
| `--graph` | - | Specify graph file path |

---

## 📁 Example Files

The `examples/` directory contains complete usage examples, demonstrating the full workflow from PDF input to knowledge graph construction and Q&A evaluation:

### Directory Structure

```
examples/
├── example_input/          # 📥 Input files
│   ├── 2020.acl-main.45.pdf   # Sample PDF: An NLP academic paper
│   └── 13_qa.jsonl            # Q&A dataset: 13 questions (Text/Multimodal) with ground truth
│
├── example_working/        # ⚙️ Intermediate results (auto-generated)
│   ├── 2020.acl-main.45/      # PDF preprocessing output (Markdown, layout info)
│   ├── images/                # Extracted images from PDF
│   ├── graph_*.graphml        # Intermediate graphs (text graph, image graph)
│   └── kv_store_*.json        # Key-value storage (Text Chunks, Image Descriptions, etc.)
│
├── example_output/         # 📤 Final output
│   ├── example_mmkg.graphml   # Final fused multi-modal knowledge graph
│   ├── example_mmkg_emb.npy   # Graph node embeddings
│   ├── example_mmkg_report.md # Build statistics report (node count, entity distribution)
│   └── retrieval_log.md       # RAG query detailed logs
│
├── cache/                  # 💾 Cache data
│   └── *.json                 # LLM API response cache for faster re-runs
│
├── paper/                  # 📄 Project materials
│   ├── framework.png          # System architecture diagram
│   └── mmgraphrag.pdf         # Project-related paper/documentation
│
├── docqa_example.py        # 🧪 Q&A evaluation script
└── docqa_results.md        # 📊 Evaluation results report
```

### Sample Document & Evaluation

- **Sample Document** (`2020.acl-main.45.pdf`): Demonstrates the system's ability to process academic papers with rich text and charts.
- **Evaluation Script** (`docqa_example.py`): A one-click evaluation tool that:
    1. Automatically reads the sample PDF and builds a knowledge graph
    2. Loads questions from `13_qa.jsonl` (covering text-only and multi-modal chart Q&A)
    3. Performs RAG retrieval and answering using the built graph
    4. Generates a detailed evaluation report `docqa_results.md`, comparing model answers with ground truth

Run evaluation:
```bash
python examples/docqa_example.py
```

---

## 🧪 Evaluation Reference (eval_reference)

The `eval_reference/` directory contains **reference code** for document QA evaluation on two benchmark datasets:

- **DocBench**: [https://github.com/Anni-Zou/DocBench](https://github.com/Anni-Zou/DocBench)
- **MMLongBench**: [https://github.com/EdinburghNLP/MMLongBench](https://github.com/EdinburghNLP/MMLongBench)

> [!CAUTION]
> **This code is for reference only and cannot be used directly.**
>
> MMGraphRAG has undergone a major refactoring that:
> - Fixed compatibility issues caused by MinerU updates
> - Enhanced robustness for resumable execution
> - Removed redundant functionality
>
> Even reproducing results with the previous version would be quite challenging due to the more complex MinerU configuration requirements.

### Recommended Approach for Reproduction

If you wish to reproduce the evaluation results, we recommend **rewriting based on the refactored codebase**, using:

1. `eval_reference/` as a reference for evaluation logic
2. `examples/docqa_example.py` as a template for building the QA pipeline

### Directory Structure

```
eval_reference/
├── docbench_eval/              # DocBench dataset evaluation
│   ├── QA.py                      # Main QA script (MMGraphRAG, GraphRAG, LLM, MMLLM, NaiveRAG)
│   ├── evaluate.py                # Evaluation metrics calculation
│   ├── eval_llm.py                # LLM-based evaluation
│   ├── mineru_docbench.py         # MinerU preprocessing for DocBench
│   ├── naive_rag.py               # Naive RAG baseline
│   ├── check.py                   # MinerU preprocessing Result checking utilities
│   ├── result.py                  # Result aggregation
│   └── evaluation_prompt.txt      # Evaluation prompts
│
└── mmlongbench_eval/           # MMLongBench dataset evaluation
    ├── run.py                     # Main evaluation script (supports multiple methods)
    ├── eval_score.py              # Scoring functions
    ├── extract_answer.py          # Answer extraction utilities
    ├── mineru_mmlongbench.py      # MinerU preprocessing for MMLongBench
    └── prompt_for_answer_extraction.md  # Answer extraction prompts
```

### Brief Overview

| File | Purpose |
|------|---------|
| `QA.py` / `run.py` | Main entry points for running different QA methods (MMGraphRAG, GraphRAG, LLM, MMLLM, NaiveRAG) |
| `evaluate.py` / `eval_score.py` | Evaluation metrics (accuracy, F1, etc.) |
| `mineru_*.py` | MinerU-based PDF preprocessing for each dataset |

> [!NOTE]
> **Honest Disclaimer**: This evaluation code has not been polished for the research community and may appear somewhat messy. We warmly welcome contributions to improve this section of the codebase!

### Performance Notes

The refactored codebase demonstrates **improved performance** in small-scale testing (e.g., examples from the DocBench dataset). This improvement may be attributed to:

- Enhanced parsing accuracy from MinerU updates
- Performance improvements in the models used compared to the original experiments

When the paper is published, if the codebase remains unchanged, we plan to conduct a more thorough cleanup of this evaluation code.

---

<p align="center">
  <i>Letting Hues Quietly weave through knowledge graph 🎨</i><br>
  <i>a small graph with big dreams ✨</i>
</p>
