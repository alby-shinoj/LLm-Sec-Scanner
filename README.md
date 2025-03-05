# LLm-Sec-Scanner
benchmarks the security and performance of open-source large language models (LLMs) from Hugging Face

# requirements
``` sh
!pip install transformers torch detoxify pandas numpy matplotlib psutil
```
## Description
It evaluates over 50 features across 10 categories: Prompt Injection, Data Leakage, Toxicity, Adversarial Robustness, Consistency, Latency, Ethical Bias, Overgeneration, Instruction Following, and Memory Usage. The user inputs the model name, and the script dynamically adapts to causal LMs or encoder-decoder models like T5. Results are saved to Google Drive (/content/drive/MyDrive/Colab Notebooks/llm/benchmark_results.html) as an HTML file with a detailed table (color-coded red/yellow/green for vulnerability levels) and 10 bar graphs visualizing stability and risks. Features include memory tracking, bias detection, and latency measurement, making it a comprehensive tool for assessing LLM security.
