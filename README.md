# Mistral 7B Logit Difference Amplification Experiment

Testing logit difference amplification technique using open-source **Mistral 7B Instruct v0.2** model.

## Objective

Replicate the logit difference amplification research from the original OpenAI experiment using a local open-source model where we have **direct access to all logits** (unlike the OpenAI API which restricts to top-20).

**Key Advantage**: With Mistral 7B, we can access the full vocabulary logits (50,000+ tokens), allowing us to properly test the amplification technique without API restrictions.

## Quick Start

### 1. Check System Readiness (One-time)

```bash
cd /Users/kunalsingh/research_projects/mistral_7b_amplification_2026_01_27
python code/check_mistral_readiness.py
```

Verifies Python, dependencies, RAM, disk space, and GPU.

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Includes: torch, transformers, jupyter, pandas, matplotlib, etc.

### 3. Download Model (One-time, ~10-15 min)

```bash
python code/download_mistral_model.py
```

Downloads Mistral 7B Instruct (~14GB), tests it, and provides optimization tips.

### 4. Open & Run Notebook

```bash
jupyter notebook experiment_logit_amplification.ipynb
```

Then run cells in order:
1. **Setup & Imports** - Load libraries
2. **Configuration** - Set experiment parameters
3. **Model Loading** - Load Mistral 7B (keeps in memory for efficiency)
4. **Utility Functions** - Define amplification logic
5. **Test Model** - Quick sanity check
6. **Run Experiments** - Main analysis loop
7. **Visualize & Analyze** - View results with plots
8. **Save Results** - Export to JSON/CSV/PNG

**Why Jupyter for this project:**
- 🎯 Load model once, test multiple prompts (efficient)
- 📊 Inline visualizations and analysis
- 🔄 Iterate without reloading (modify cells, re-run)
- 📝 Document findings alongside code
- ⚡ Much faster than Python script reruns

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **RAM** | 12 GB (with 4-bit quantization) | 16+ GB |
| **Disk** | 20 GB | 50+ GB (for multiple experiments) |
| **GPU** | Optional (CPU works, slower) | NVIDIA/AMD/Apple Silicon |
| **Python** | 3.8+ | 3.10+ |

**Your System**: 16 GB RAM, 324 GB free disk ✅

## Directory Structure

```
mistral_7b_amplification_2026_01_27/
├── README.md                                      # This file
├── requirements.txt                               # Python dependencies (includes jupyter)
├── .env.example                                   # Configuration template
├── experiment_logit_amplification.ipynb          # Main interactive notebook ⭐
│
├── code/                                          # Setup & utility scripts
│   ├── check_mistral_readiness.py                 # System checks (run once)
│   └── download_mistral_model.py                  # Model download & verify (run once)
│
├── data/                                          # Experiment results
│   ├── amplification_results.json                 # Raw experiment output
│   ├── summary_statistics.csv                     # Aggregated metrics
│   ├── experiment_metadata.json                   # Run metadata
│   └── recommendations.txt                        # Next steps
│
├── docs/                                          # Documentation
│   ├── METHODOLOGY.md                             # Technical approach
│   ├── RESULTS_ANALYSIS.md                        # Findings & insights
│   └── ...
│
└── visualizations/                                # Generated plots
    ├── response_length_analysis.png               # Response metrics
    ├── logits_analysis.png                        # Logits distribution
    └── ...
```

## Key Differences from OpenAI Experiment

| Aspect | OpenAI API | Mistral 7B |
|--------|-----------|-----------|
| **Logits Access** | Top 20 only | Full vocabulary (50,000+) |
| **Vocabulary** | Limited to 25-35 tokens | Full access |
| **Cost** | Per-token charges | One-time download |
| **Speed** | Slower (API calls) | Faster (local inference) |
| **Reproducibility** | Depends on API | Fully reproducible locally |
| **Coherence** | Expected to collapse | Should work properly |

## Expected Outcomes

Unlike the original experiment which failed due to vocabulary bottleneck:

1. ✅ Logit amplification should produce **coherent outputs** (with full logits)
2. ✅ We can measure **actual amplification effects** on harmful content
3. ✅ Compare different α values (0.1 to 2.0) with full precision
4. ✅ Analyze semantic gap engineering more thoroughly
5. ✅ Test on multiple prompt types (jailbreaks, edge cases, etc.)

## Experiment Parameters

Default settings in `.env`:

- **Model**: Mistral 7B Instruct v0.2
- **Quantization**: 8-bit (for RAM efficiency)
- **Max tokens**: 100
- **Temperature**: 0.7
- **Top-p**: 0.9
- **Alpha values**: [0.1, 0.3, 0.5, 1.0, 1.5, 2.0]

## Implementation Notes

### Quantization Levels

The script supports three quantization approaches:

1. **Full Precision (float16)** - ~28GB GPU, high quality
2. **8-bit Quantization** - ~14GB GPU, good quality
3. **4-bit Quantization** - ~7GB GPU, acceptable quality

Choose based on available resources.

### Logging Results

All experiments log to:
- `data/amplification_results.json` - Raw results
- `data/detailed_report.txt` - Human-readable analysis
- Console output with real-time progress

### Performance Tips

For faster inference:
```python
# Use quantization
python code/logit_diff_mistral.py --quantization 4bit

# Or use vLLM for optimized serving
pip install vllm
```

## Troubleshooting

### Issue: "CUDA out of memory"
→ Use quantization: `--quantization 8bit`

### Issue: Model download interrupted
→ Model caches automatically; resume download by running again

### Issue: Very slow on CPU
→ This is normal; consider using quantization or GPU if available

### Issue: "No module named torch"
→ Run: `pip install -r requirements.txt`

## References

- [Mistral AI Official](https://www.mistral.ai/)
- [Model Card - Mistral 7B Instruct](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)
- Original research: `/Users/kunalsingh/research_projects/logit_diff_amplification_2026_01_26/`

## Status

- [ ] System readiness verified
- [ ] Dependencies installed
- [ ] Model downloaded
- [ ] Experiments running
- [ ] Results analyzed
- [ ] Documentation complete

---

**Created**: 2026-01-27
**Researcher**: Kunal Singh
