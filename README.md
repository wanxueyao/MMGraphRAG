<!-- PROJECT LOGO -->
<h3 align="center">🎉MMGraphRAG</h3>

  <p align="center">
    ✨A Multi-Modal knowledge Graph RAG framework✨
  </p>
</div>


<!-- ABOUT THE PROJECT -->
## About The Project

[![MMGraphRAG Pipeline Overview]](https://github.com/wanxueyao/MMGraphRAG/blob/main/fig1.png)

This diagram illustrates the comprehensive workflow of the MMGraphRAG pipeline.

The project is based on modifying the nano-graphrag to support multimodal inputs (removing the community-related code). The image processing part uses YOLO and multimodal LLM to convert images into scene graphs. The fusion part then uses spectral clustering to select candidate entities, combining the textual knowledge graph and the image knowledge graph to construct a multimodal knowledge graph.

Currently, the supported multimodal input formats are DOCX and PDF.

<!-- GETTING STARTED -->
## Getting Started

Note: To achieve better PDF extraction results, you need to install Mineru. The required libraries are listed in `cache/requirements.sh`. It is recommended to use Python 3.10 in the Conda environment.

### Prerequisites

The libraries that need to be installed are as follows, with `magic-pdf` being Mineru.
```sh
pip install openai
pip install sentence-transformers
pip install nano-vectordb
pip install python-docx
pip install PyMuPDF
pip install ultralytics
pip install tiktoken
pip install -U "magic-pdf[full]"
```

### Installation

1. Running this project requires at least one text model API and one multimodal model API. Alternatively, you can deploy the models locally using vLLM.
2. The code of this project currently supports local embeddings. The recommended model is all-MiniLM-L6-v2. Additionally, stella-en-1.5B-v5 has also shown excellent performance in testing. You can download the model via ModelScope:
```sh
pip install modelscope
modelscope download --model sentence-transformers/all-MiniLM-L6-v2
```
3. All parameters are specified in /mmgraphrag/parameter.py, including the storage path for the local embedding model and the model APIs.

Next, let's introduce some other configurable parameters.

⚾mineru_dir is an input directory where the files processed by Mineru can be placed. It can be used for testing when the input is a Mineru web processing result.

QueryParam:

🐵response_type: The format of the response.

🐶top_k: The maximum number of entities to retrieve.

🦊local_max_token_for_text_unit: The maximum number of tokens for entity and relation text in the retrieval results.

🐱local_max_token_for_local_context: The maximum number of tokens for the text block (i.e., context) in the retrieval results.

🦁number_of_mmentities: The maximum number of images in the retrieval results.

4. The default YOLO model is yolov8n-seg.pt, located in the `/cache` directory.

<!-- USAGE EXAMPLES -->
## Usage

Once the setup is complete, you can test the project using `mmgraphrag/mmgraphrag_test.py`. Here's how to use the script:

- `pdf_path`: The file path of the input document.
- `working_dir`: The output file path.
- `question`: The user's inquiry.

When building the knowledge graph, the `input_mode` can be set to 0, 1, or 2:
- Mode 2 is used to process PDF files when Mineru is installed.
- Mode 0 processes DOCX files.
- Mode 1 handles well-structured PDF files, though it performs poorly with complex PDF files.

When answering questions, you need to set `query_mode` to `True`.

For document-based question answering, examples of input and output are provided and stored in `/example_input` and `/example_output` directories. The `response.txt` file contains a comparison between the results from `mmgraphrag`(kimi + qwenvl) and ChatGPT 4o.
