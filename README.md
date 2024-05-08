## Multi-Agent for IELTS Writing Assistant

### This repo will use the powerful of LLM and special prompting technique to grade IELTS Essay, both Writing Task 1 and 2. You can use it on multiple LLMs.

### Model used:
- **Llama 3 8b**/ Phi-3/ Gemma-7b for text generation
- **TinyChart 3b** for chart understanding
- **Llava Phi-3** for diagram understanding

### Prompting technique:
 - 2 different Agent for WT1 and WT2
 - Self-understanding and Self-correction for generating comments and scores.
 - RAG for CoT (not effective)

Even with 3 different models at the same time, by using quantization and special technique, it can still operate on [Google Colab](https://colab.research.google.com/drive/16jO1kxdq4f5oVe6azr9Ywlaf5eiuVxyT)

### Evaluation (on-going)
#### Writing Task 2
- ChatGPT 3.5: 0.46 - 0.7
- Our: 0.25 - 0.31
