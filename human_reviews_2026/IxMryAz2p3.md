# BTZSC: A Benchmark for Zero-Shot Text Classification Across Cross-Encoders, Embedding Models, and Rerankers

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 4

## Abstract
Zero-shot text classification (ZSC) offers the promise of eliminating costly task-specific annotation by matching texts directly to human-readable label descriptions. While early approaches have predominantly relied on cross-encoder models fine-tuned for natural language inference (NLI), recent advances in text-embedding models, rerankers, and instruction-tuned large language models (LLMs) have challenged the dominance of NLI-based architectures. Yet, systematically comparing these diverse approaches remains difficult. Existing evaluations, such as MTEB, often incorporate labeled examples through supervised probes or fine-tuning, leaving genuine zero-shot capabilities underexplored. To address this, we introduce __BTZSC__, a comprehensive benchmark of $22$ public datasets spanning sentiment, topic, intent, and emotion classification, capturing diverse domains, class cardinalities, and document lengths. Leveraging BTZSC, we conduct a systematic comparison across four major model families, NLI cross-encoders, embedding models, rerankers and instruction-tuned LLMs, encompassing $38$ public and custom checkpoints. Our results show that: (i) modern rerankers, exemplified by _Qwen3-Reranker-8B_, set a new state-of-the-art with macro $F_1 = 0.72$; (ii) strong embedding models such as _GTE-large-en-v1.5_ substantially close the accuracy gap while offering the best trade-off between accuracy and latency; (iii) instruction-tuned LLMs at 4-12B parameters achieve competitive performance (macro $F_1$ up to $0.67$), excelling particularly on topic classification but trailing specialized rerankers; (iv) NLI cross-encoders plateau even as backbone size increases; and (v) scaling primarily benefits rerankers and LLMs over embedding models. BTZSC and accompanying evaluation code are publicly released to support fair and reproducible progress in zero-shot text understanding.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces BTZSC, a comprehensive benchmark for evaluating zero-shot text classification performance across three major model paradigms: NLI-based cross-encoders, embedding models, and rerankers. BTZSC unifies 22 English datasets covering sentiment, topic, intent, and emotion classification under a text-label semantic matching framework, where each label is verbalized as a natural-language sentence. Through systematic experiments with 31 models, the paper finds that rerankers achieve the highest zero-shot accuracy, while strong embedding models offer better efficiency-performance trade-offs. The benchmark further analyzes scaling trends, NLI transferability, and latency-accuracy relationships, providing new insights into how different architectures generalize under true zero-shot conditions.

### Strengths
1) The benchmark systematically evaluates three major paradigms for zero-shot text classification under a unified evaluation protocol. This breadth of comparison provides valuable insight into how different architectures trade off accuracy, scalability, and cost.
2) The benchmark integrates 22 publicly available datasets spanning multiple task types, which enables a relatively broad assessment of zero-shot performance across heterogeneous domains.

### Weaknesses
1) The paper claims to be "true zero-shot", but the 22 datasets used (such as AGNews, IMDb, AmazonPolarity, etc.) are all from public corpora that may have been seen during model pre-training. This means that the model may have indirectly learned the task distribution or category semantics, and strictly speaking, it is impossible to verify the model's migration ability under "completely new tasks".
2) The benchmark relies solely on macro-averaged F1 as the evaluation metric. While this simplifies cross-dataset comparison, it introduces several methodological limitations. F1 alone cannot reveal important trade-offs between precision and recall, nor can it account for task difficulty or class imbalance. Moreover, reranker models are evaluated only on top-1 predictions, ignoring their ranking quality. The absence of per-class variance and statistical significance testing further limits interpretability.

### Questions
1) Is the full benchmark publicly released or planned to be open-sourced? If not, could you clarify the reasons or timeline for public availability?
2) How do you ensure that the datasets used were not partially included in the pre-training corpora of the evaluated models (such as BERT, DeBERTa, GTE, E5, or Qwen)?
3) Why did you choose to rely solely on macro-averaged F1, and did you evaluate whether other metrics lead to consistent rankings across models?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces BTZSC, a benchmark comprising 22 datasets to evaluate zero-shot text classification performance across three major model families: NLI-based cross-encoders, embedding models, and rerankers. The authors conduct a broad empirical comparison across 31 models and report that rerankers achieve the highest accuracy, while embedding models offer the best efficiency-performance trade-off.

### Strengths
The paper systematically covers a wide range of datasets and model types, making it one of the most comprehensive empirical studies on zero-shot text classification to date. The release of the BTZSC benchmark and codebase supports reproducibility and could serve as a useful evaluation tool for the community.

### Weaknesses
1. The work lacks technical novelty, where no new algorithm, model, or training strategy is proposed.
2. The core contribution lies in benchmark, which is largely incremental given the existence of MTEB and similar efforts. The experimental insights are mostly intuitive or expected (e.g., rerankers scale better, embedding models are efficient), and analysis is superficial.
3. The evaluation design largely mirrors existing work, such as label verbalization and metric selection, with limited innovation in methodology.
4. The discussion around trade-offs and scaling is descriptive but lacks theoretical depth.

### Questions
Please refer to weaknesses regarding the lack of technical contributions and analytical depth.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes BTZSC, a benchmark for true zero‑shot text classification that compares three families: NLI‑tuned cross‑encoders, embedding models, and rerankers; over 22 English datasets spanning topic, sentiment, intent, and emotion. The primary metric is macro‑F1 (with micro accuracy also reported). The study evaluates 31 models (public and custom), analyzes scaling and the NLI to ZSC transfer, and reports that: (i) rerankers, notably Qwen3‑Reranker‑8B, achieve the highest overall performance (macro‑F1 = 0.72), (ii) strong embeddings (e.g., GTE‑large‑en‑v1.5) “close the gap” and tend to offer the best accuracy–latency trade‑off, and (iii) NLI cross‑encoders are plateauing while scaling primarily benefits rerankers. See Abstract; dataset suite and selection in §3, Table 1 (p. 4), Figure 1 (p. 5); model families in §3.2 (pp. 5–6) and Table 4 (p. 15); results in Table 2 (p. 9); and analyses in Figure 2 (p. 8) and Figure 3 (p. 8).

### Strengths
- Representative, diverse suite. 22 datasets across four task families; Table 1 and Figure 1 quantify class cardinalities, lengths, and lexical overlap (pp. 4–5).   
- Clean zero‑shot protocol across families. Unified scoring: NLI log‑odds entailment; embedding cosine; reranker yes/no token probability with a fixed template (§4, Appendix C.3).  
- Clear headline results. Qwen3‑Reranker‑8B tops macro‑F1 (0.72); GTE‑large‑en‑v1.5 is competitively high among embeddings; Figure 2(b) shows reranker‑favoring scaling; Figure 3(a) visualizes the speed–accuracy frontier (p. 8; p. 9).  
- NLI to ZSC analysis. Figure 3(b) offers a useful lens on transfer for cross‑encoders and contrasts to embeddings/rerankers (p. 8).

### Weaknesses
1. No zero‑shot LLM‑as‑classifier baseline. The Introduction acknowledges the approach but omits it from evaluation; 8B–32B LLMs are feasible on the stated hardware and would complete Figure 3(a) (pp. 2, 8, 10).
2. Embedding template opacity. §4 lacks per‑model instruction/template details for E5/GTE/BGE/Qwen‑Embedding; performance is often template‑sensitive (p. 6). These models can be instructed.
3. Verbalizer multiplicity untested. Single label verbalization per class; no template‑averaging/paraphrase robustness (p. 6; Appendix A notes reusing Laurer et al. 2023 verbalizers). Authors can consider zero-shot augmentations on verbalizers, with a L2-normalization of the average embedding of augmentations like CLIP, or a k-NN approach where majority voting across different augmented forms of verbalizers is used.
4. Minor inconsistencies. banking77 listed as 77 classes in text (p. 3) vs 72 in Table 1 (p. 4); and the “second‑highest overall 0.64” prose refers to accuracy after defining macro‑F1 as primary (pp. 4, 7, 9).
5. For completion, it would be interesting to verify whether the conclusions of this paper are reproducible using the same models and evaluation methodology on the classification tasks of MTEB instead. However, such an analysis is missing.

### Questions
Aren't the used reranking models cross-encoders?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a comprehensive benchmark for Zero-Shot Text Classification including 22 datasets, with a focus on comparing three types of approaches: NLI, sentence transformers, and re-rankers. The proposed benchmark is designed to cover task diversity, class granularity, diversity of text domain and length. The paper proposes a primary metric, Micro F1, and the use of NLI performance as a proxy for zero-shot generalization capabilities. Empirical results indicate reranker models achieve the highest overall accuracy, while strong embedding models offer the most favorable balance between speed and accuracy.

### Strengths
I liked the proposal of a comprehensive evaluation dataset and the clear explanations of the three zero-shot approaches: NLI, sentence transformers, and re-rankers. The evaluation is systematic and includes many model variants. I also found the analysis of scaling across model sizes informative.

### Weaknesses
1. Given this is a dataset/benchmark paper, I'm surprised by how little detail it offers about the construction of the dataset and evaluation procedure. I am having a hard time connecting the underlying datasets and how the final BTZSC evaluation procedure works.
2. The paper primarily focus on encoder-based architectures, I wonder how do generative models perform on such tasks, and would a reasonably sized generative model effectively solve these tasks? What distinct value does this benchmark provide relative to generative models?

### Questions
1. How are the datasets aggregated? Are they not directly aggregated or the labels are aggregated by domain?
2. I wonder how do generative models perform on such tasks, and would a reasonably sized generative model effectively solve these tasks? What distinct value does this benchmark provide relative to generative models?
3. Given this dataset, what are some challenges in existing methods and promising directions for future work?

### Soundness
3

### Presentation
3

### Contribution
2
