# Guarding the Gate: ConceptGuard Battles Concept-Level Backdoors in Concept Bottleneck Models

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 6, 4

## Abstract
The increasing complexity of AI models, especially in deep learning, has raised concerns about transparency and accountability, particularly in high-stakes applications like medical diagnostics, where opaque models can undermine trust. Explainable Artificial Intelligence (XAI) aims to address these issues by providing clear, interpretable models. Among XAI techniques, Concept Bottleneck Models (CBMs) enhance transparency by using high-level semantic concepts. However, CBMs are vulnerable to concept-level backdoor attacks, which inject hidden triggers into these concepts, leading to undetectable anomalous behavior. To address this critical security gap, we introduce ConceptGuard, a novel defense framework specifically designed to protect CBMs from concept-level backdoor attacks. ConceptGuard employs a multi-stage approach, including concept clustering based on text distance measurements and a voting mechanism among classifiers trained on different concept subgroups, to isolate and mitigate potential triggers. Our contributions are threefold: (i) We present ConceptGuard as the first defense mechanism tailored for concept-level backdoor attacks in CBMs. (ii) We provide theoretical guarantees that ConceptGuard can effectively defend against such attacks within a certain trigger size threshold, ensuring robustness. (iii) We demonstrate that ConceptGuard maintains the high performance and interpretability of CBMs, crucial for trustworthiness. Through comprehensive experiments and theoretical proofs, we show that ConceptGuard significantly enhances the security and trustworthiness of CBMs, paving the way for their secure deployment in critical applications.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses a newly emerging security threat in CBMs — concept-level backdoor attacks, where an attacker poisons the concept representations instead of raw inputs. The authors propose ConceptGuard, the first certified defense framework against such attacks. ConceptGuard operates by (1) clustering concept embeddings into multiple semantic subgroups, (2) training independent classifiers for each subgroup, and (3) aggregating predictions through a majority-voting mechanism. The intuition is that backdoor triggers typically contaminate only a small subset of concept clusters, so the ensemble can suppress their influence.

### Strengths
- First comprehensive study on concept-level backdoor defense in CBMs — a novel and timely problem connecting interpretability and security.
- Experiments are carefully executed on two benchmark CBM datasets, with extensive comparisons across attack types (CAT, CAT+), cluster sizes, and ablation studies.
- Practical implications are strong — the defense is architecture-agnostic and potentially applicable to high-stakes domains such as medical CBMs or financial decision systems.

### Weaknesses
- The certification theorem assumes non-overlapping clusters and independence among base classifiers. In real concept embeddings, cluster boundaries are fuzzy and interdependent. The robustness bound may thus overestimate practical safety.
- The defense relies heavily on the initial clustering quality. The paper only uses k-means; and no adaptive cluster selection strategy is discussed.
- ConceptGuard neutralizes triggers but cannot identify which concepts or clusters are poisoned, limiting practical forensic usefulness.

### Questions
- Can you quantitatively show that ConceptGuard maintains concept-level interpretability (e.g., through fidelity or agreement metrics)?
- What is the additional computational overhead compared to a standard CBM?
- Have you tried more semantic or contrastive clustering methods?
- How does ConceptGuard handle overlapping or hierarchical concept structures (e.g., “has wings” ⊂ “can fly”)?
- How does the computational cost scale with the number of clusters m?

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
This paper aims to study how to address concept-level backdoor attacks, which inject hidden triggers into these concepts. To address this, we introduce a defense framework (ConceptGuard) that aims to protect CBMs from concept-level backdoor attacks. The authors also conduct experiments to evaluate the performance of the proposed method.

### Strengths
[+] The paper studies how to defend concept-level backdoor attacks in CBMs, which is an important problem in interpretable AI systems.

[+] The paper provides the theoretical analysis.

[+] The writing of this paper is easy to follow.

### Weaknesses
[-] The problem studied in this paper seems limited in scope. The paper focuses on defending against the concept-level backdoor attacks proposed by Lai et al. (2024), but it lacks discussion on whether other backdoor attacks could also threaten Concept Bottleneck Models (CBMs). Additionally, the paper does not examine how existing backdoor attack techniques could be adapted to CBMs. In CBMs, the concept-level information from training data is primarily represented by discrete labels rather than continuous features. Can existing backdoor attacks designed for textual or other data modalities [3,4,5] be effectively applied to CBMs? Moreover, the discussions of backdoor attacks on the input level should be provided (including existing input-level CBM backdoor attacks, and the adoption of existing backdoor attacks to input-level). Further, the study of both the input-level and concept-level backdoor attacks are not provided.

[-] The discussion of related work is insufficient. The paper primarily focuses on Concept Bottleneck Models (CBMs) within the context of deep learning but overlooks several important related models. In addition to standard deep learning-based CBMs, there exist concept bottleneck generative models [1], vision-language concept bottleneck models [2], and bottleneck diffusion models [3]. These frameworks also integrate concept-level reasoning but operate under different architectures. It is encouraged to discuss whether the attack and defense method can generalize to these broader classes of CBMs and clarify any potential limitations.  

[-] The discussion of existing defenses against backdoor attacks is insufficient. Many defense methods have been proposed in recent years (e.g., trigger detection, model pruning, input purification, and training regularization approaches [6,7,8,9]). However, the paper lacks a comprehensive analysis or positioning of the proposed method relative to these existing defenses. It would be better to discuss how current defense techniques could be adapted or generalized to CBMs and to highlight the unique challenges that distinguish CBM-specific defenses from traditional ones.

[-] The discussion of the threat model in this paper requires further clarification and justification. The paper focuses on concept-level backdoor attacks, where triggering specific concepts in the testing data leads to malicious model behavior. However, in real-world applications, for a given test sample, the CBM first predicts intermediate concept representations before producing final class labels. It remains unclear why and how the attacker can effectively inject or activate backdoors through concept-level perturbations at test time, and whether such manipulations are feasible or realistic in practice.

[-] The rationale behind the proposed defense remains unclear. First, the paper does not explain how to detect or determine whether the training dataset contains backdoored samples, and does not provide the additional computational cost analysis for the proposed method. Additionally, the proposed approach divides the dataset into m subsets, but the selection of the optimal number of subsets (m), the effective clustering criteria, and the effective distance functions are not clearly discussed. It is also unclear how the method avoids potential empty clusters during partitioning. Furthermore, since concepts may exhibit hierarchical relationships, it is also important to discuss how to address this. If k-means is adopted, how are such hierarchical dependencies handled? Finally, the majority voting strategy used for final predictions may overlook the different importance or reliability of individual base classifiers. It is encouraged to consider weighting strategies. 

[-] For the adopted AwA dataset, the paper uses GPT-4 Achiam et al. (2023) to generate full sentences to replace each original concept represented by a single word. However, the use of such a large language model may introduce hallucination issues and uncertainty in the regenerated outputs. Without addressing these, the reliability and validity of the experimental results may be undermined.

[-] In experiments, only the ResNet-50 model is adopted, which limits the evaluation generality. More evaluations on other large-scale deep learning models should be provided. Additionally, can the proposed be generalized to other CBMs (such as concept bottleneck generative models [1], vision-language concept bottleneck models [2], and bottleneck diffusion models [3])?

Reference

[1] Concept bottleneck generative models, ICLR 2023.

[2] Vision-Language Concept Bottleneck Models, ArXiv 2024.

[3] Soda: Bottleneck diffusion models for representation learning, CVPR 2024.

[4] Natural backdoor attack on text data, 2020.

[5] Backdoor attacks and countermeasures in natural language processing models: A comprehensive security review, 2025.

[6] Onion: A simple and effective defense against textual backdoor attacks, 2020.

[7] Textguard: Provable defense against backdoor attacks on text classification, 2023.

[8] Expose backdoors on the way: A feature-based efficient defense against textual backdoor attacks, 2022.

[9] A survey of recent backdoor attacks and defenses in large language models, 2024.

### Questions
[1] In experiments, the authors adopt image augmentations. However, it is unclear about why image augmentations are adopted? Do image augmentations improve the model robustness against concept backdoor attacks? More discussions on the impact of image augmentations should be provided.

[2] Could the proposed method generalize to other different data types (except the image data type)? For other different data types, are augmentations needed? How to do these augmentations?

[3] Could the authors clarify in the threat model what specific knowledge the attacker is assumed to possess and what knowledge is not available to them? Additionally, for the considered threat model, it would be helpful to discuss its feasibility in real-world scenarios. Further, how would the attack and defense perform under constrained black-box settings, such as those with query-only access to the model?

[4] Could the authors clarify whether the adopted Concept Bottleneck Models (CBMs) are trained sequentially (i.e., concept prediction followed by label prediction) or jointly (i.e., both components optimized together)? Additionally, can the proposed defense be applied to both training paradigms, and if so, are there any differences in effectiveness or implementation between the sequential and joint training settings?

[5] Typos: “ones or zeros .” in Line 162.

### Soundness
2

### Presentation
3

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
The paper addresses the problem that Concept Bottleneck Models (CBMs) in Explainable Artificial Intelligence (XAI) are vulnerable to concept-level backdoor attacks and proposes a new defense framework, ConceptGuard. The method employs semantic concept clustering and a sub-model voting mechanism to effectively isolate and mitigate concept-level triggers, with both theoretical robustness guarantees and empirical validation.

### Strengths
S1: The paper shows a degree of innovation by proposing a defense mechanism against concept-level backdoor attacks. Existing backdoor defenses operate mainly at the input or feature level, but this paper identifies that attacks can be hidden in semantic concept representations, filling a research gap in XAI security. Compared to traditional input-level defenses, ConceptGuard focuses on robustness in the concept space, opening a new security direction.

S2: The multi-stage defense framework (clustering + sub-model training + voting) is logically clear and technically solid; the theoretical section provides a “Certified Robustness” analysis (Theorem 1 and 2), proving that predictions remain unchanged under a bounded trigger size.

S3: The attack success rate (ASR) drops by 70–85%, while model accuracy is maintained or slightly improved (+1–2%), demonstrating strong empirical validity.

### Weaknesses
W1: Although the paper proposes a Certified Size analysis, the defense performance heavily depends on the quality of semantic clustering. If concept clusters are ambiguous or semantically overlapping, robustness cannot be guaranteed.

W2: Training multiple sub-models and performing majority voting significantly increases computational cost (O(m) times), with no discussion on efficiency optimization or scalability to large-scale settings.

W3: The choice of semantic distance metrics for clustering is unclear. While TF-IDF, Word2Vec, and BERT are mentioned, the paper does not explain how the method or parameters are selected, nor provide theoretical justification for the number of clusters m.

W4: Some experimental results are repetitive, and the figures and tables are overly detailed; the presentation could be more concise. For instance, there is overlapping content between Table 1 and Table 2. Table 1 (p. 7) already reports the main quantitative comparison of ConceptGuard vs. baseline (CAT / CAT+) on both datasets, while Tables 2–3 (p. 8) again list per-sub-model accuracies under the same settings. For example, Table 1 (CUB, CAT) shows 83.03 % accuracy ↑ 1.38, ASR 11.55 ↓ 74 %, and Table 2 reports each sub-model ≈ 74–78 %, ensemble = 83.03 %. The trends are identical (accuracy ↑ 1–2 %, ASR ↓ 70–85 %), and these could be summarized in one paragraph or moved to the appendix.

### Questions
The defense performance appears to rely heavily on the semantic clustering of concepts. Could the authors elaborate on how robust ConceptGuard is to suboptimal clustering (e.g., when semantically unrelated concepts are grouped together)? Have you compared different embedding strategies (TF-IDF, Word2Vec, BERT) to quantify this sensitivity?

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
4

### Summary
The paper studies concept-level backdoor attacks on Concept Bottleneck Models (CBMs), formalizing CAT/CAT+ and proposing ConceptGuard, a defense that (i) clusters concept texts into groups, (ii) trains sub-models on corresponding datasets, and (iii) aggregates predictions by majority vote. The paper provides a mathematical formulation of the attack and a threat model, describes the defense pipeline, and presents certified robustness conditions with an "improved joint certified accuracy" bound and a procedure to compute worst-case certified accuracy. Experiments are on CUB and AwA (with AwA concepts modified) with good amount of work done in evaluating various aspects of the proposed attack and defense.

### Strengths
S1. The paper is well-written and clear with nice figures.
S2. Robustness of CBM-type architectures is a critical problem and the paper's attack and defense mechanism can be a useful tool.
S3. The theoretical guarantees introduced are sound and understandable, making the work persuasive.

### Weaknesses
W1. Threat model motivation in expert-intervenable concept space: CBMs are designed for human inspection/intervention at the concept layer. CAT smotivates concept-space triggers as "hard to detect" without analyzing detectability when concept values/annotations are visible to experts (note CBMs are designed to be intervenable). The paper does not do *ANY* human-in-the-loop analysis or a discussion of how triggers evade expert audits of concept vectors. This is a very serious concern for the attack itself.

W2: Baselines compared are very basic, not acknowledging work in this field in the last few years: CAT, CAT+ have only been evaluated on CBMs and ConceptGuard train starting from CBM models. In the past couple of years, a plethora of works [1,2,3,4,5] have been released that improve specific aspects of CBMs, but they have not been compared with CAT/ConceptGuard at all. For example, [3] adversarially trains the CBM model to be robust to adversarial attack, [2] uses a VAE formulation to minimize leakage, and [1] uses concept embeddings for a smoother concept space. 

W3. Lack of more datasets for exhaustive empirical analysis: Currently, the method has only been evaluated on CUB and AWA2. Further, AWA concepts are changed with GPT into sentences to improve clustering - introducing a new confounding variable. Performance gains may hinge on LLM-enriched semantics rather than the defense alone. I would recommend that authors perform more experiments on CelebA, PascalAPY, etc. for a broader analysis.

W4. The defense relies on using the word embeddings of the concept annotations to cluster them into subspaces. However, the defense fails if the concept annotations are linked to non-generic domains, such as medical diagnosis. In the CBM paper, there is an analysis on an X-ray knee dataset (OAI), which would have perfectly demonstrated this problem. I have reservations wherein concepts like 'bone spurs' and 'bone spacing' will be clustered close together, making the defense mechanism ineffective.


[1] Concept Embedding Models. Zarlenga et al. NeurIPS 2022 \
[2] GlanceNets: Interpretable, Leak-proof Concept-based Models, NeurIPS 2022 \
[3] Understanding and enhancing robustness of concept-based models, Sinha et al, AAAI 2023  \
[4] Label-free Concept Bottleneck Models, Oikarinen et al., ICLR 2023 \
[5] Learning to Receive Help: Intervention-Aware Concept Embedding Models, Zarlenga et al., NeurIPS 2023

### Questions
Q1. Were any analyses on expert detectability of concept triggers (e.g., show concept vectors with/without triggers) done to justify why concept-space poisoning is realistic when concepts are auditable?

Q2. Were more datasets - like CelebA and OAI used? How were the performance on them?

Q3. How much does the embedding model affect ASR?

### Soundness
2

### Presentation
2

### Contribution
2
