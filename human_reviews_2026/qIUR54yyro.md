# PEARL: Differentially Private and Entropy-Aware Regulated Language Generation

- Decision: Reject
- Scores: 6, 2, 6

## Abstract
Large language models (LLMs) commonly adopt Retrieval-Augmented Generation (RAG) to improve faithfulness. However, carefully crafted extraction prompts can elicit sensitive private information. Differential Privacy (DP) has therefore been integrated into LLM inference and is widely regarded as a standard safeguard; yet most work focuses on the utility–privacy trade-off, leaving the trustworthiness of DP outputs underexplored. To assess trustworthiness, we revisit the confidence gap (CG), which quantifies an LLM’s internal knowledge conflict. We show that CG correlates with both hallucination and exposure of personally identifiable information (PII). Building on this insight, we present PEARL, a CG‑guided, entropy‑aware private decoding framework. PEARL adaptively allocates the privacy budget across tokens and sentences based on CG, concentrating protection on spans likely to contain PII while stabilizing low‑confidence, hallucination‑prone regions. In experiments, PEARL improves response trustworthiness and robustness to PII extraction attacks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies the effects of DP on hallucination, in the setting where DP is used to protect private information when using RAG in LLMs. The authors construct two RAG benchmarks in the medical and financial domains, with PII annotations. They use the Confidence Gap (CG) to detect hallucination, and they show that hallucinated sentences have lower CG than supported ones, and that sentences containing PII have higher CG. The authors propose PEARL, a differentially private entropy-aware language generation framework, where after producing an initial response with the exponential mechanism, they reallocate the privacy budget at the sentence level and selectively regenerate segments flagged by CG as hallucination-prone or containing PII. They show that PEARL enables privacy while reducing hallucination.

### Strengths
- This work studies how DP affects hallucination, which is important and under-explored.
- The results show that PEARL performs well compared to the baselines, and that it yields lower PII leakage.
- This paper contributes two RAG benchmarks in the medical and financial domains with PII annotations, which will be useful for future studies.

### Weaknesses
- The classification of sentences into supported/unsupported/uncertain categories, as well as the labeling of PII, was done by GPT-4o, and it is unclear how accurate these classifications and labels are.
- The datasets are synthetically generated using GPT-4o, and may not be reflective of real data. While the authors mention that they engaged domain experts to check the data, it would be good to give more details on what percentage of the documents were checked, and the methodology for checking.
- The main evaluation was only done on two LLMs. It might be useful to evaluate on a more diverse range of LLMs with different sizes to check that the results generalize.
- PEARL might lead to a higher computational cost and latency.

### Questions
- Could more details be given on how the datasets are checked to be similar to real data?
- What is the computational overhead and latency of using PEARL?
- How accurate are the classification into categories and PII labeling done by GPT-4o?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper tackles the challenge of applying Differential Privacy (DP) to Retrieval-Augmented Generation (RAG) models, arguing that the standard privacy-utility trade-off overlooks the critical impact of DP on trustworthiness. The authors posit that DP-injected noise exacerbates hallucinations. The paper's core contribution is identifying the Confidence Gap (CG)---the entropy difference between a base model and a context-retrieval model---as a unified signal for two distinct risks. The key insight is that this signal is bimodal: low CG indicates high knowledge conflict and correlates with hallucinations, while high CG indicates regurgitation of context and correlates with PII leakage.
Based on this, the authors propose PEARL, a multi-stage private decoding framework. PEARL first generates an initial response using a standard DP mechanism. It then privately filters this response at the sentence level using CG to identify hallucination-prone spans ($S_U$) and PII-bearing spans ($S_P$). Finally, it reallocates the remaining privacy budget to redact the PII spans ($S_P$) while selectively regenerating the hallucination-prone spans ($S_U$) using a private refilling technique. Experiments on LeakRAG, two new RAG benchmarks (medical and financial) with explicit PII annotations created by the authors, demonstrate that PEARL simultaneously reduces hallucination scores and PII leakage compared to baselines.

### Strengths
S1. The paper's primary strength is its novel and important problem formulation. By shifting the focus from a simple "privacy-utility" trade-off to a more nuanced "privacy-trustworthiness" (i.e., hallucination) analysis, it addresses a key gap in prior work.

### Weaknesses
W1. The experimental baselines are too simple and self-referential. The paper's main comparisons are against 'NOREFILL' and 'RANDOMFILL', which are ablations of the PEARL method itself, not external competitors. The related work section (Section 6) explicitly mentions other 'non-uniform privacy-budget allocation' methods (e.g., Wang et al., 2025), which are the true state-of-the-art. By omitting a comparison to these, the paper fails to demonstrate its superiority over existing, published work.

W2. The paper's definition of privacy risk is confusing due to a mismatch between its formal guarantee and its practical evaluation. The formal guarantee is document-level DP (Section 2.1), protecting against the influence of adding/removing one document. However, the evaluation (Section 5.3) uses a specific, fine-grained PII-extraction attack. This leaves the actual attack model and the practical privacy risk being addressed unclear. For instance, is the goal to protect the presence of a document (as DP implies) or to prevent the leakage of specific PII tokens (as the evaluation implies)? A model could satisfy the former but fail the latter (e.g., if PII appears in multiple documents), making it unclear what the DP guarantee actually achieves.

W3. The paper's framing of its insights can be an overclaim. The core observation that low CG correlates with hallucinations is a finding from prior work (which the paper cites as Bi et al. (2025)), making the experiment in Section 3.2 appear to be a replication of this previous finding that doesn't bring new insight, rather than a new discovery of this paper.

W4. The introduction of the new LeakRAG benchmark is presented as a major contribution, but it is not sufficiently motivated or elaborated. The paper does not situate this new benchmark within the existing literature on privacy or RAG benchmarks. Furthermore, the soundness of the new evaluation metrics (HalluScore, GoldAlign) and the precise definition of 'privacy' (which appears to be a specific list of PII types) are not well-explained or justified. Given this, presenting it as a significant benchmark seems overstated; it functions more as a new test dataset for this specific paper.

W5. The proposed PEARL framework (Algorithm 1) is a complex, multi-stage pipeline. A major confounding factor is that the refilling stage (Line 27) uses the OpenAI GPT-4o API as the "filling model", while the target LLMs being evaluated are LLaMa 3.1 8B and Qwen 2.5 7B. This introduces a massive, external, and significantly more powerful model into the loop. It is unclear how much of the performance gain (especially in reduced hallucination) is attributable to the PEARL framework versus simply getting a "second pass" from GPT-4o. The paper lacks a crucial ablation study using the same base model (e.g., LLaMa 8B) for both initial generation and refilling.

W6. The entire framework's success hinges on the Confidence Gap (CG) being a perfectly reliable and non-overlapping signal for PII and hallucinations. The paper shows a correlation (Figure 3) but does not stress-test this assumption. What happens if a sentence contains both a PII token and a hallucination? The framework's filtering (TOP-K or SVT) seems to assume these sets ($S_P$ and $S_U$) are disjoint. This potential "signal conflict" is not analyzed, and the model's behavior in this scenario is undefined.

### Questions
Q1. The epsilon = 0 baseline is labeled 'Zero-shot' in Table 1. This is confusing, as $\epsilon=0$ implies perfect privacy and infinite noise, which should result in random, useless output. However, the 'Zero-shot' model has high utility scores.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces PEARL, a novel framework for differentially private and entropy-aware regulated language generation in LLMs, specifically addressing the trade-off between factual accuracy (from Retrieval-Augmented Generation - RAG) and privacy. The authors identify that Differential Privacy (DP) mechanisms, while protecting sensitive information, can inadvertently increase "hallucination" (generation of non-factual content) in LLM outputs.

### Strengths
S1. The concept of using a "confidence gap" (CG) to guide privacy budget allocation and mitigate hallucination in DP settings is novel and well-motivated.

S2. The authors clearly articulate how DP noise can exacerbate knowledge conflicts and hallucination, laying a strong foundation for their proposed PEARL framework.

S3. new, scenario-grounded RAG benchmarks with explicit PII annotations in medical and financial domains is a significant contribution.

S4. Strong experimental results and evaluation (BERTScore, GoldAlign, and HalluScore provide a multi-faceted evaluation of the model's performance). PEARL consistenly outperforms existing baselines.

### Weaknesses
W1. A more intuitive explanation of CG or a small illustrative example in Section 2.2 could enhance understanding for readers less familiar with entropy differences in this context.

W2. While PEARL aims to mitigate the computational cost of considering privacy and hallucination at every decoding token, a more explicit discussion or quantification of PEARL's computational overhead compared to simpler DP decoding methods would be beneficial.

W3. The privacy attack involves appending a "malicious instruction." While this is a good stress test, a brief discussion on how typical prompt engineering strategies (without malicious intent) might influence PII leakage in baseline models versus PEARL could add more context.

W4. Grammar/Typos: Line 054: "DIFFERENTIALLY PRIVACY" -> "DIFFERENTIAL PRIVACY". Line 120: "output of model" -> "output of the model". Line 384: "ABLATION STUIDES" -> "ABLATION STUDIES". These are minor and easily fixable.

W5. Figure 1 Description: The text "privacy-related knowledge should not be factually disclosed, while general knowledge may be affected by DP-induced hallucination" is helpful, but consider adding a brief statement directly describing the visual elements of the diagram itself, beyond just the concept.

### Questions
See weaknesses

### Soundness
3

### Presentation
3

### Contribution
3
