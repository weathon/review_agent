# InstructLR: A Scalable Approach to Create Instruction Dataset for Under-Resourced Languages

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4

## Abstract
Effective text generation and chat interfaces for low-resource languages (LRLs) remain a challenge for state-of-the-art large language models (LLMs) to support. This is mainly due to the difficulty of curating high-quality instruction datasets for LRLs, a limitation prevalent in the languages spoken across the African continent and other regions. Current approaches, such as automated translation and synthetic data generation, frequently yield outputs that lack fluency or even orthographic consistency. In this paper, we introduce InstructLR, a novel framework designed to generate high-quality instruction datasets for LRLs. Our approach integrates LLM-driven text generation with a dual-layer quality filtering mechanism: an automated filtering layer based on  retrieval-augmented-generation (RAG)-based n-shot prompting, and a human-in-the-loop validation layer. Drawing inspiration from benchmarks such as MMLU in task definition, InstructLR has facilitated the creation of three multi-domain instruction benchmarks: **ZarmaInstruct-50k**, **BambaraInstruct-50k**, and **FulfuldeInstruct-50k**.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a pipeline to create instruction following samples in low-resource languages. In specific, instructions in a high-resource language are first generated and input into a large language model that translates these instructions and generates corresponding response in the low-resource language. These candidate instruction-following samples are evaluated by a large language model checker with retrieval-augmented generation, where samples labeled with errors are further refined by human experts. Authors have created instruction-following datasets in three low-resource languages using the proposed pipeline. Experimental results show that models tuned on these datasets are significantly better than the ones tuned on machine-translation samples.

### Strengths
- This paper investigates an important problem: how to create a large amount of high quality instruction following samples in low-resource languages?
- The instruction following datasets generated in three low-resource languages will be helpful to the low-resource NLP community.

### Weaknesses
- **Limited Generalization**: This pipeline involves a large language model with reasonable performance on the low-resource language and some human experts for evaluation and correction, which makes it hard to scale and generalize to some low-resource languages. Given a low-resource language, this pipeline may be not applicable for all large language models performing bad or none suitable human experts. On the other hand, the number of instruction following samples is constrained by the budget to hire human experts.

- **Missing Evaluation of RAG Checker**: This method uses a RAG checker to filter out low-quality samples. However, they do not evaluate the effectiveness of this checker, which makes the quality of accepted samples questionable.

- **Missing Baseline for Comparison**: Some important new pipelines to create instruction following samples in low-resource languages are not cited or comparison [1, 2].

References

[1] Li, C., Yang, W., Zhang, J., Lu, J., Wang, S., & Zong, C. (2024, January). X-Instruction: Aligning Language Model in Low-resource Languages with Self-curated Cross-lingual Instructions. In ACL (Findings).

[2] Köksal, A., Thaler, M., Imani, A., Üstün, A., Korhonen, A., & Schütze, H. (2025). Muri: High-quality instruction tuning datasets for low-resource languages via reverse instructions. Transactions of the Association for Computational Linguistics, 13, 1032-1055.

### Questions
1. Does the RAG checker good in evaluating low-resource language instruction-following samples? Are there any problems with 85.8% samples marked "Accepted without correction"?

2. What is the advantage of your method comparing other beseline methods? How do they perform on the three low-resource languages?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a comprehensive pipeline, InstructLR, to automatically and efficiently generate high-quality instruction-tuning datasets for low-resource languages (LRLs), focusing on Zarma, Bambara, and Fulfulde.
The framework integrates:
Seed instruction generation in a high-resource language (e.g., French);
LLM-based translation and response generation directly into the target LRL;
Dual-layer quality filtering, combining automated RAG-based checking and human validation.

### Strengths
1, Tackles multilingual equity by addressing a pressing issue: lack of instruction datasets for African and other under-resourced languages.
2, The dual-layer filtering pipeline (RAG-based automatic correction + human validation) is novel and pragmatic.
3, Framework demonstrated across three distinct LRLs, showing language-agnostic and reusable properties.
4, Quantitative gains (BLEU +20–30, ROUGE, METEOR) and human preference results clearly substantiate claims.

### Weaknesses
1, Relies on Gemini and GPT-4o for initial generation; this undermines reproducibility and scalability in low-resource contexts.
2, All three LRLs are French-contact African languages, so claims of language-agnosticism remain under-tested.
3, Only five Zarma and one Bambara annotators—too few to ensure dialectal or sociolinguistic representativeness.
4, The dual-layer filtering ensures fluency but not factual correctness, leaving potential hallucination propagation unaddressed.
5, The framework is more engineering-driven than theoretically motivated; lacks a clear linguistic or data-centric theoretical foundation.
6, Evaluations are limited to BLEU/ROUGE/NER; lacks instruction-following generalization on realistic multi-turn tasks.
7, Heavy reliance on French-based seed instructions may embed Western or francophone biases into LRL outputs.
8, BLEU/ROUGE are weak proxies for instruction-following quality, especially across languages with divergent morphology.
9, 500 samples per language is not enough for statistical robustness; lacks confidence intervals for inter-annotator consistency.
10, Some pipeline components (RAG knowledge base construction, FAISS index details) are under-specified for replication.
11, The MT-Seed baseline may be too simplistic; missing comparisons to existing multilingual instruction datasets (e.g., Aya, BELLE, or Multilingual Alpaca).

### Questions
-

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
3

### Summary
The paper introduces **InstructLR**, a scalable and modular pipeline to generate high-quality instruction datasets for **low-resource languages (LRLs)**. The approach leverages large language models in high-resource languages (such as French) to generate seed instructions, translates and adapts them to the target low-resource language, and applies a **dual-layer quality filtering mechanism**—an automated RAG-based correction system followed by human validation.  
Using this pipeline, the authors produce three 50k-scale instruction datasets (Zarma, Bambara, and Fulfulde) and demonstrate through extensive automatic and human evaluations that fine-tuning open-source LLMs on these datasets substantially improves instruction-following capabilities and downstream performance (e.g., NER) in these languages.

### Strengths
- **Timely and important topic:** Addressing LLM accessibility for under-resourced languages is a highly relevant problem with social and scientific impact.  
- **Complete and scalable approach:** The paper presents an end-to-end framework, from seed instruction generation to human validation, which is reusable across languages and domains.  
- **Clarity and reproducibility:** The pipeline is clearly described and supported by well-chosen examples and figures. The authors also emphasize cost-efficiency and open licensing, which makes the work practically impactful.  
- **Empirical thoroughness:** The experiments are extensive, involving multiple models and metrics, and include both automatic and human evaluations, adding credibility to the results.  
- **Writing quality:** The manuscript is well-written and easy to read, with clear motivation and well-organized experimental sections.

### Weaknesses
- **Limited novelty:** While the framework integrates translation, RAG-based filtering, and human validation effectively, these components are individually standard. The main contribution is the *composition* of these techniques rather than a new algorithmic insight.  
- **Experimental focus:** The experiments primarily show that models fine-tuned on the resulting datasets perform better than baselines. This is known fact. However, they do not deeply analyze *the pipeline itself*—for instance, how translation quality, RAG corrections, or human validation quantitatively affect final performance. A more ablation-style study would have better demonstrated the pipeline’s internal efficacy.  
- **Applied rather than exploratory:** The paper provides solid engineering value but remains on the applied side; it does not explore new theoretical or modeling questions in instruction tuning.

### Questions
- While the paper shows that InstructLR fine-tuning improves performance in the target languages, how does this affect *related* languages (e.g., mutual benefit for typologically similar LRLs)?  
- Does fine-tuning on these new datasets lead to degradation in performance for high-resource languages such as French?  
- Can the authors provide results comparing model performance on the *original untranslated* instruction set before and after InstructLR fine-tuning? This would clarify whether the model truly improves in cross-lingual understanding or only specializes in the generated instruction style.  
- How sensitive is the overall quality to the translation step? Have the authors evaluated how errors in translation propagate through the pipeline and influence filtering success rates?

### Soundness
2

### Presentation
3

### Contribution
2
