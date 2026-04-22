# Recall-First Moderation via Distribution-Preserving Augmentation and Committee-Diverse Retrieval

- Avg Score: 2.50
- Decision: Reject
- Scores: 4, 2, 2, 2

## Abstract
False negatives—missed unsafe content—remain the dominant risk in safety-critical moderation. We present a novel recall-first moderation framework that integrates two complementary innovations: (i) distribution-preserving contrastiveaugmentation, which generates boundary-focused hard positives and negatives while statistically preserving corpus structure, and (ii) committee-diverse re-trieval, which combines dense, MMR, and graph-based selectors to construct label-informative, non-redundant neighborhoods at inference. Augmented corpora are validated with KL/JS divergence thresholds (≤ 0.05 globally), confirming indistinguishability from the source distribution. On a large held-out test se tof multidomain unbalanced text, vanilla retrieval-augmented pipelines expose the persistent failure mode of under-detecting FLAGGED content (recall ≈ 0.44), but also reveal a strong baseline gap: an open-source stack (FAISS + local LLaMA-3) achieves significantly higher accuracy and macro-F1 than a commercial counter-part (API embeddings + hosted LLM). Adding augmentation and committee retrieval improves sensitive-class recall by ∼ 10 points (to ≈ 0.56) while maintaining global performance, with graph-aware retrieval pushing open-source accuracy to 0.8510 and Macro-F1 to 0.7635. Ensemble experiments with DistilRoBERTa further raise recall to 0.5781 without loss of utility.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper presents a "recall-first" moderation framework for automated content moderation in safety-critical settings using LLMs with distribution-preserving contrastive data augmentation and a committee-based, diverse retrieval mechanism.

### Strengths
1. The optimization towards recall rather than accuracy alone is highly task-related.

2. The good results on both open-sourced and commercial models demonstrate the practicality of the proposed method.

### Weaknesses
1. While the paper demonstrates improvements on the DeepNLP dataset, there is little discussion of the limitations of the approach regarding domain transferability or adversarial robustness, which is crucial for safety-related works.

2. The experiments rely almost entirely on the DeepNLP dataset, which is very small, synthetic in part, and not representative of real-world moderation data such as social media. More evaluations on different large-scale datasets are recommended.

3. It seems that limited baselines are used for comparison. Could the authors compare their method with the methods in the related works section for validating their conclusion, for example CLASS-RAG and Mod-Guide?

4. The table format is not formal. Some tables lack lines to separate different rows.

### Questions
See the weakness section.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes a new recall-first moderation framework with distribution-preserving contrastive augmentation and committee-diverse retrieval, aiming to enhance the recall of harmful content.
The experiment results suggest that the new framework preserves the original data distribution, improves the recall, and maintains global performance.

### Strengths
* The authors try to integrate the strengths of multiple modules to build a better method. 

* The paper also presents detailed background knowledge about different methods, frameworks, and pipelines in Section 3.

### Weaknesses
* From my personal sense, the paper lacks a clear enough motivation to build a system that combines multiple modules together.
The author has presented some, but I don't see enough linkage between to motivation and the implementation. 
For example, 
a) ``safety-critical`` is mentioned as the focus for introducing hard positives and negatives (in the introduction), but the implementation simply considers factors as length and general semantic similarity, which do not directly correspond to safety-critical from my perspective.
b) the method ``casts votes across multiple retrieved contexts`` to improve recall, but it is unclear why recall cannot be improved by using a single retriever with parameter adjustment, e.g., a softer decision boundary.  


* The experiment scope is unclear and seems to be small. Only 1 dataset, a few combinations of models, and configurations are tested.


* The writing lacks fluency and coherence.
In terms of writing, some terms lack definition or explanation. For example, `` boundary-focused`` - what boundary is it; what are some hard cases you are targeting? Same issue for L164 ``label priors and latent structure``, L322 ``external systems``.
In terms of logic, some parts are quite disjointed. Why are more boundary-focused examples better for distribution preserving? Isn't adding more examples at the boundary as an augment shift the original distribution?

* The experiment's presentation is also very confusing. Hard to compare the enhancement of the method with the baselines. Also unsure what the purpose of Section 4.2 is to compare two baseline methods.

* The layout is highly likely to violate regulations. For example, Table 4 definitely is out in the margin. Table 1 and 2 uses aggressive /vspace{} for space saving.

### Questions
* Why are "subtle rephrases of flagged content and safe content" hard positives/negatives? For example, if one flagged content is certainly harmful, its paraphrase should also be marked as clearly harmful text. Need more explanation here.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
- The paper proposes a "recall-first" framework to address the risk of false negatives (missed unsafe content) in automated moderation.

- It introduces two primary methods: (i) a "distribution-preserving" data augmentation strategy to create hard boundary-case examples and (ii) a "committee-diverse" retrieval system that combines dense, MMR, and graph-based selectors to provide varied context to an LLM.

- The paper claims these methods significantly boost the recall of sensitive content from a baseline of $\approx0.44$ to $\approx0.56$, with further ensemble methods reaching $\approx0.58$.

### Strengths
The paper's focus on a "recall-first" approach is highly relevant and important for safety-critical applications, where false negatives are the dominant risk.

### Weaknesses
- The abstract and conclusion repeatedly claim the main contribution - the framework combining augmentation and committee retrieval - improves recall for the sensitive (FLAGGED) class from a baseline of $\approx0.44$ to $\approx0.56$.

- The paper appears to dishonestly "borrow" its headline result from a completely different experiment. The $\approx0.56-0.58$ recall numbers only appear in Section 4.4 and Table 4, which describe a separate DistilRoBERTa-based ensemble classifier - not the RAG/LLM framework that constitutes the paper's main conceptual contribution. The abstract misrepresents this ensemble result as the achievement of its RAG pipeline.

- The paper's credibility is further weakened by admitted experimental errors. In Section 4.3, the authors note that the results for the "PLUS (MMR)" configuration are invalid because "the evaluator reused $B^{+}$ outputs as $A^{+}$ placeholders," making the comparison meaningless.

### Questions
See the weakness section above.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper focuses on “recall-first” content moderation scenarios and proposes two key innovations: ① “Distribution-Preserving” contrastive text augmentation, using KL/JS thresholds to validate the consistency between augmented data and the original distribution; ② “Multi-Committee” retrieval (dense + MMR + graph) to construct neighborhoods with richer label information and reduced redundancy. The authors report that on a multi-domain imbalanced text corpus, the open-source stack outperforms commercial alternatives.

### Strengths
1. Clear problem formulation (prioritizing “missed detection” in reviews) and systematic solutions, providing end-to-end evaluation and statistical significance testing (bootstrap CI, McNemar).

2. Proposing concrete, reproducible JS/KL metrics and implementation details for “distribution-preserving” augmentation (thresholding, SVD, GMM, report generation).

3. Data splitting and evaluation emphasize leak prevention: augmented samples inherit their parent samples' splits and do not cross training/test boundaries.

### Weaknesses
The core experiment relied on the “DeepNLP” dataset from Kaggle, comprising only “therapeutic chat responses (80) + resumes (125)”, which is extremely small in scale and semantically inconsistent with real-world security review distributions. The final large-scale sample primarily originated from model-generated augmented text, raising concerns about its generalizability.


 Furthermore, the final training/test sets comprised predominantly augmented samples (e.g., Train 18,054; Test 2,151), failing to match real-world online noise and novelty.



The main text requires all global and per-cluster JS values ≤ 0.05 to establish “distribution preservation,” yet the appendix shows a maximum per-cluster character histogram JS ≈ 0.62 (Sheet 2, low-sample clusters), significantly exceeding the threshold. This anomaly is attributed to “small-sample instability,” yet it directly undermines the credibility of the core “distribution preservation” claim and subsequent comparisons.


 

Under MMR settings, authors reuse B+ outputs as placeholders for A+, resulting in identical metrics for both. While claiming this does not affect image retrieval conclusions, such execution bias weakens the persuasiveness of method comparisons.


 



The paper contrasts “commercial stacks vs. open-source stacks,” yet significant differences in embeddings, inference, and implementation details make it difficult to attribute performance variations to a single factor. Despite presenting significant results, the causal explanation for the claim “open-source significantly outperforms commercial” remains insufficient.

It includes numerous augmented samples derived from test seeds (though not split-crossed). Such “neighbor-based rewriting” differs markedly from real-world attack patterns and cross-domain drift, potentially overestimating system robustness in actual online environments.
Translated with DeepL.com (free version)

### Questions
refer to weakness

### Soundness
2

### Presentation
1

### Contribution
2
