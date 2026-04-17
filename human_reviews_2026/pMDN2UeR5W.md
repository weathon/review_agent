# Better know nothing than half-know anything: A Precise and Efficient Dataset for Scientific Reasoning in Language Models

- Decision: Reject
- Scores: 4, 6, 4

## Abstract
Large Language Models (LLMs) have achieved remarkable progress in reasoning tasks, i.e., coding and mathematics. However, their ability to perform scientific reasoning remains significantly limited, probably hampered by the scarcity of high-quality scientific reasoning datasets. Existing approaches either rely on LLM-generated synthetic data (suffering from noise and hallucinations) or human-compiled documents (facing scarcity and non-standardization). In this paper, we empirically verify that integrating precise knowledge from original scientific documents with formalized questions and consistent answers can mitigate the need for large-scale data. Based on this insight, we design PreciSci, a pipeline for constructing multi-disciplinary scientific reasoning datasets. This pipeline involves extracting knowledge from reliable sources, refining questions for completeness and precision, applying multi-stage filtering to eliminate redundancy and noise, and refining answers to ensure reliable supervision. Leveraging PreciSci, we build Open-Sci, a precise and knowledge-dense scientific reasoning dataset. Experimental evaluations show that despite Open-Sci being less than one-sixth the size of state-of-the-art scientific reasoning datasets, it enables LLMs to achieve approximately $\mathbf{4.49\%}$ better performance across diverse discipline-specific benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents PreciSci, a pipeline for constructing scientific reasoning datasets, and Open-Sci, a 196k dataset covering physics, chemistry, biology, and medicine. The pipeline extracts QA pairs from authoritative sources, formalizes questions, applies multi-stage filtering, and enforces answer consistency. Despite being 16% the size of MegaScience, Open-Sci reportedly achieves 4.49% better performance on scientific benchmarks.

### Strengths
1. The five-stage pipeline provides a structured methodology for scientific data curation with clear motivations for each component: authoritative source extraction, question formalization, noise mitigation, and answer consistency enforcement.

2. Comprehensive evaluation across 12 benchmarks spanning scientific domains (MedQA, PHYSICS, ChemBench, ProteinLMBench), general reasoning (GPQA, MMLU-Pro), and mathematics (AIME2025, Math-500), with consistent improvements across Qwen and Llama models at different scales.

3. Well-balanced dataset with 47 sub-disciplines, difficulty labels, diverse question types (48% open-ended, 30% calculation), and detailed answers averaging 1,124 tokens.

### Weaknesses
1. Bias from model-based filtering. Section 3.3 uses model accuracy thresholds for cascaded filtering, and Section 3.4 states "only correct candidates consistent with references were retained." This biases the dataset toward problems current models can solve, contradicting the goal of creating challenging training data for improving model capabilities.

2. Insufficient decontamination verification. Section 3.3 mentions using "embedding similarity and LLM confirming semantic overlap" for benchmark decontamination but provides no details: embedding model, similarity threshold, LLM prompts, or how many samples were removed per benchmark. Performance gains may result from data leakage rather than quality improvements.

3. Missing critical implementation details. No prompts for question formalization (Section 3.2), no numerical thresholds for filtering (Section 3.3), no method for answer extraction (Section 3.4), and no inter-annotator agreement statistics. The work is non-reproducible.

4. Limited methodological novelty. The pipeline combines standard techniques (LLM extraction, MinHash deduplication, answer distillation) without introducing new algorithms or theoretical insights.

### Questions
1. What does "only correct candidates consistent with references were retained" (Section 3.4) specifically mean? Are you filtering based on whether Qwen3-30B generates correct answers? What metric determines correctness, and what are the complete filtering criteria?

2. Provide complete decontamination specifications: which embedding model, what similarity threshold, what LLM prompts? For each of the 12 benchmarks, how many training samples were identified as overlapping and removed? Provide concrete examples.

3. How were the 200k samples selected from each dataset in Table 6? Random sampling, stratified sampling, or another method? What ensures fair comparison given different quality distributions?

4. The paper claims 196k samples are more efficient than 1.25M, but shows no scaling analysis. Can you provide results using different data amounts (e.g., 50k, 100k, 150k, 196k) to validate this claim?

5. Table 8 shows minimal difference between base Qwen3-8B (59.97%) and Open-Sci-tuned version (60.49%) in science reasoning. Given this negligible improvement, how do you justify the claim that Open-Sci significantly improves reasoning efficiency?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces "PreciSci," a meticulous data curation pipeline designed to address the poor quality and noise prevalent in existing scientific reasoning datasets. The authors argue for a "quality over quantity" approach, starting from authoritative sources (textbooks, exams) and applying rigorous multi-stage filtering that ultimately discards over 67% of the initial raw data.
The resulting dataset, "Open-Sci," contains 196k high-precision, knowledge-dense instances. Experiments show that models fine-tuned on Open-Sci, despite its small size (1/6th of MegaScience), achieve superior performance, improving by an average of 4.49% on scientific benchmarks  and demonstrating remarkable generalization to mathematics.

### Strengths
1. Clear Motivation: The paper correctly identifies data quality, not just scale, as the primary bottleneck for scientific reasoning. The PreciSci pipeline is a robust and well-documented methodology for tackling this.
2. Exceptional Generalization (Math): The most striking result is the massive performance leap on mathematical reasoning. For example, Qwen3-8B's score on AIME2025 jumps from a baseline of 16.67% to 51.67% when trained on Open-Sci. This strongly suggests that rigorous, multi-step scientific training imparts transferable reasoning skills.
3. Thorough Ablation Studies: The authors demonstrate that each component of the PreciSci pipeline (formalization, noise mitigation, answer consistency) is critical to the final performance.

### Weaknesses
1. Analysis of Math Gains is Lacking: The paper does not adequately investigate why the AIME2025 performance gain is so exceptionally large (+30 points). A deeper analysis correlating the 30% "calculation" questions in Open-Sci to the structure of math problems is needed.
2. Unquantified Human Cost: The pipeline relies on a "hybrid AI-human process", but the human-in-the-loop effort is not quantified. This makes it difficult to assess the true cost and reproducibility of the PreciSci pipeline.
3. Potential Pipeline Bias: The pipeline itself uses Qwen and DeepSeek models for key steps like QA extraction, formalization, and answer generation. This raises a concern: Does the Open-Sci dataset inadvertently "overfit" to the style of Qwen models? This could explain why Qwen models show significantly larger gains (+8.69% for 8B) from Open-Sci than other models like Llama-3.1 (+3.11%).

### Questions
see weakness

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces PreciSci, a data curation pipeline designed to construct high-quality, compact scientific reasoning datasets, and Open-Sci, a 196k-instance dataset built using this pipeline. The core idea is that precision and quality outweigh scale in scientific reasoning. 

The pipeline involves:
1. Extracting QA pairs from authoritative sources (textbooks, exams, competitions) using a hybrid AI-human approach;
2. Formalizing questions into standard types (MCQ, fill-in, calculation, etc.) for clarity and completeness;
3. Multi-stage noise mitigation (deduplication, decontamination, filtering) to remove redundancy and low-quality samples;
4. Answer consistency refinement via distillation to ensure accurate, well-structured reasoning.

### Strengths
S1: This paper addresses a real gap—lack of high-quality scientific reasoning data—and proposes a quality-over-quantity approach, which is refreshing in the era of scaling.

S2: PreciSci is well-structured, with clear stages for extraction, formalization, filtering, and answer refinement. The ablation studies show each step contributes meaningfully.

S3: Open-Sci consistently outperforms larger datasets like MegaScience and WebInstruct across scientific, general, and math benchmarks, even on smaller models.

S4: This paper demonstrates that 196k well-curated samples can beat 1.2M+ noisy ones, supporting the paper’s central thesis.

### Weaknesses
W1: In general, STEM includes the math domain. However, Open-Sci is limited to natural sciences and excludes math, which is a major part of scientific reasoning. The claim of “scientific reasoning” is thus partially incomplete.

W2: Despite claiming human-in-the-loop, there is no quantitative human evaluation of question quality, answer correctness, or reasoning depth.

W3: The paper does not test generalization to new domains or out-of-distribution scientific problems. Take Table 4 as an example, SuperGPQA and GPQA-Diamond are also benchmarks for scientific reasoning, not general domains.

W4: This paper only compares to other datasets, not to models trained with process supervision, self-generated data, or RL-based reasoning, which are more relevant in 2025.

W5: This paper would be valuable for the community if the dataset, models, and pipeline were fully open-sourced and more examples were provided to learn about the quality of Open-Sci.

### Questions
Q1: How does Open-Sci perform on unseen/evaluated scientific domains (e.g., astronomy, geology) or interdisciplinary problems? For example, providing the result of each subject of MMLU-Pro should be better.

Q2: What is the actual human effort involved? How many hours or annotators were used, and what was the inter-annotator agreement?

Q3: How does the model perform on problems requiring creative or open-ended reasoning, not just standard QA formats?

Q4: Would Open-Sci still outperform larger datasets if both were scaled to the same size? Is the gain purely due to quality, or also due to better coverage?

Q5: How does the pipeline handle ambiguous or ill-posed questions that are common in real-world scientific inquiry?

Q6: Can the pipeline be automated further to reduce human involvement, or is it inherently labor-intensive?

### Soundness
3

### Presentation
3

### Contribution
2
