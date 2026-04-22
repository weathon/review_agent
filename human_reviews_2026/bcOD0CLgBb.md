# Concept-Aware Privacy Mechanisms for Defending Embedding Inversion Attacks

- Avg Score: 5.20
- Decision: Accept (Poster)
- Scores: 6, 8, 2, 6, 4

## Abstract
Text embeddings enable numerous NLP applications but face severe privacy risks from embedding inversion attacks, which can expose sensitive attributes or reconstruct raw text. Existing differential privacy defenses assume uniform sensitivity across embedding dimensions, leading to excessive noise and degraded utility. We propose SPARSE, a user-centric framework for concept-specific privacy protection in text embeddings. SPARSE combines (1) differentiable mask learning to identify privacy-sensitive dimensions for user-defined concepts, and (2) the Mahalanobis mechanism that applies elliptical noise calibrated by dimension sensitivity. Unlike traditional spherical noise injection, SPARSE selectively perturbs privacy-sensitive dimensions while preserving non-sensitive semantics. Evaluated across six datasets with three embedding models and attack scenarios, SPARSE consistently reduces privacy leakage while achieving superior downstream performance compared to state-of-the-art DP methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents SPARSE, a user-centric framework for concept-specific privacy protection in text embeddings. SPARSE integrates two key components: (1) the identification of privacy-sensitive dimensions based on user-defined concepts, and (2) a Mahalanobis noise mechanism that injects elliptical noise calibrated to each dimension’s sensitivity. Through comprehensive evaluations on six datasets using three embedding models and various attack scenarios, SPARSE demonstrates significant reductions in privacy leakage while maintaining or improving downstream task performance, outperforming existing differential privacy (DP) methods.

### Strengths
This paper addresses a critical problem in privacy-preserving NLP with a novel, user-centric framework SPARSE that enables concept-specific protection via dimension-sensitive elliptical noise. The approach offers a more personalized and ethical alternative to traditional DP methods. The empirical evaluation is thorough and convincing, covering six datasets, multiple embedding models, and diverse attack scenarios. SPARSE consistently outperforms baselines in both privacy protection and downstream utility, a notable achievement.

### Weaknesses
The paper can be improved from the following aspects:

1) Lack of Formal Privacy Guarantees: The method does not provide formal differential privacy bounds (e.g., ε, δ), which limits its comparability to standard DP approaches and hinders rigorous privacy-utility analysis.

2) User Annotation Burden: Since SPARSE relies on user-defined concept privacy, the paper would benefit from a discussion or empirical analysis of the feasibility and burden of obtaining these annotations in practice.

3) Limited Applicability Across Domains: The approach assumes users can identify and define privacy-sensitive concepts, which may not hold in all contexts. This assumption could restrict the method’s usability in domains where privacy concerns are implicit or unclear.

4) Computational Overhead: There is no analysis of the runtime or resource cost introduced by SPARSE. Understanding its computational impact would be important, especially for real-time or large-scale deployment scenarios.

### Questions
The following questions can be discussed further:

1) How sensitive is SPARSE to errors in concept labeling, and what is the impact of such errors on privacy and utility? Is there any error propagation through the masking and noise injection stages?

2) How well does the method generalize to out-of-domain data, especially when user-defined concepts do not transfer or are unavailable?

3) Can SPARSE operate in zero-shot or few-shot task transfer settings settings without explicit concept annotations? If so, how is privacy managed in the absence of concept-level guidance?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper investigates text embedding privacy and proposes a novel differential privacy–based mechanism, called SPARSE, to defend against embedding inversion attacks. Unlike previous approaches, SPARSE does not apply isotropic noise; instead, it first identifies the embedding dimensions responsible for specific privacy attributes. SPARSE then applies a Mahalanobis mechanism to add noise to these dimensions, leaving the non-sensitive dimensions untouched to preserve downstream performance. The method is empirically evaluated on multiple datasets against two other defense methods and three embedding inversion attacks. A qualitative analysis of the privacy-sensitive domain and a white-box setting concludes the paper.

### Strengths
- The paper is well-motivated and addresses an important privacy-critical topic (embedding privacy). All steps are clearly described and easy to follow. Additionally, all experimental settings are thoroughly detailed, supporting reproducibility.
- The experiments demonstrate a clear advantage of SPARSE compared to related defense methods. Since the evaluation is performed on multiple datasets, the results appear reliable.
- Additional analyses of white-box attacks provide a deeper understanding of the method’s capabilities and support the assumptions made earlier in the paper. Similarly, the qualitative analysis of the identified privacy dimensions offers valuable insights.

### Weaknesses
- The paper lacks a thorough discussion of the method’s limitations. It remains unclear how dependent the method is on high-quality training data. Additionally, there is no discussion of the time required to fit a separate mask per user (or privacy setting).
- While the experimental results demonstrate an improvement over existing methods, a significant gap remains between research and practical application. For instance, an epsilon of 5 provides strong privacy but reduces utility by half. This trade-off improves for higher epsilon values, but one may still have to accept a leakage of 35%–50% to preserve utility, requiring an epsilon of 20.
- There is no analysis of how the method affects the representation of related concepts. For example, if SPARSE protects a concept like gender, how does this influence related concepts such as appearance descriptions or commonly gender-biased attributes?

Minor remarks:
- There is a missing space in L39: "T5-based embeddings.Such"
- L207: "We" should probably be capitalized.

### Questions
- Could the method, in principle, also be applied to LLM prompt extraction attacks, such as in “Extracting Prompts by Inverting LLM Outputs” (Zhang et al.)?
- How long does fitting the binary masks approximately take?
- How many attributes can the method defend against simultaneously before utility is completely compromised? For example, if we aim to prevent leakage not only of age but also of gender, ethnicity, illness, political opinion, etc., at the same time.
- L210: Is removing the private words in the negative sample set the best strategy? Why not replace the attributes with alternative values and check which dimensions change? I would expect some biases due to different input lengths of samples in the two datasets.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper studies privacy risks of text embeddings under inversion attacks and proposes SPARSE, a concept-aware defense. SPARSE first learns a differentiable neuron mask to identify embedding dimensions sensitive to a user-defined privacy concept, using a hard-concrete relaxation with an sparsity regularizer. It then injects elliptical noise via a Mahalanobis mechanism calibrated by the learned per-dimension sensitivities, claiming metric-LDP guarantees. Experiments show improved privacy–utility trade-offs over two DP baselines.

### Strengths
- Embedding inversion is relevant to deployed retrieval/RAG systems; aligning protection to user-specified concepts reflects realistic privacy needs beyond coarse PII assumptions.
- The combination of concept-conditioned sparse dimension selection and anisotropic perturbation is a step beyond spherical Laplace noise.
- SPARSE shows consistently lower leakage at comparable or better downstream metrics relative to baselines.

### Weaknesses
- The pipeline “user-defined $C$ → NER to extract tokens” inherits false positives/negatives and domain coverage limitations. The paper instantiates $C$ mostly with NER/PII tokens and acknowledges extensibility but does not quantify failure modes or robustness to imperfect concept detection
- Negative samples are built by removing tokens in $C$. This can alter syntax and semantics beyond the concept, potentially making the discrimination task easier in ways not strictly tied to $C$. The classifier may pick up distributional artifacts rather than pure concept-related differences, biasing the learned mask. No controls (e.g., semantically-preserving paraphrases) are discussed.
- May need more baselines like Gaussian mechanism and SOTA methods like truncated Laplacian mechanism.

### Questions
- Your target is preventing inference of concept tokens $C$, but the guarantee is metric-LDP in embedding space. Can you articulate a formal bridge (even approximate) from 𝜀-$||·||_M$-LDP to bounded leakage of $C$ under a class of attackers?
- How do you ensure that the classifier distinguishing $D^+$ vs $D^-$ is not exploiting grammatical breaks or topic drift created by token removal $R(s,C)$? Any controls?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes SPARSE, a concept-aware differentially private method to protect the text embeddings from inversion attacks. SPARSE selectively perturbs privacy-sensitive dimensions to apply stronger protection on protected concept. Experiments show that SPARSE has superior privacy protection and downstream performance over standard DP methods.

### Strengths
- Obfuscating sensitive concept in embeddings is a non-trivial problem, and this paper innovatively applies dimension masking and Mahalanobis mechanism to address this challenge.
- The LDP of Mahalanobis Norm can be connected with Generalized Laplace Mechanism.
- The authors conducted comprehensive experiments with promising results.

### Weaknesses
- The frameworks assumes that sensitive concept are correlated with the embedding dimension, while this might not be the case. The related dimension for each concept could change depending on the context.
- SPARSE relies on a pre-defined concept vocabulary and their corresponding masks. There could be emerging new concept in real-world, making it computation intensive to retrain the model.
- In experiment, the authors use NER to extract sensitive information, which is limited. More complex privacy concepts should be considered.

### Questions
- How to map each PII to the concept? Does each distinct PII corresponds to a single concept?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
Text embeddings used in systems like RAG can be inverted to recover sensitive attributes or content. Standard DP defenses add isotropic (“spherical”) noise to every dimension, which protects privacy but harms downstream utility and cannot target user-specified sensitive concepts. The paper proposes SPARSE, a concept-aware defense with two parts:

1. Neuron Mask Learning. Using contrastive pairs that do/do not contain a user-defined sensitive concept, it learns a sparse, differentiable mask over embedding dimensions so that “privacy-relevant” directions are identified.

2. Mahalanobis (Elliptical) Noise. Under a metric local-DP formulation, it injects anisotropic noise shaped by the learned mask (larger noise on sensitive dimensions, smaller on others), implemented as a generalized Laplace mechanism with a diagonal covariance.

### Strengths
1. The paper is well-written and well-structured.
2. The problem and methods are well-defined, especially dataset construction and learning objectives in mask learning.
3. The results look promising. SPARSE substantially reduces leakage at the same privacy budget while preserving/improving downstream accuracy compared to spherical-noise baselines across semantic similarity and retrieval tasks (e.g., STS12, FIQA) and against multiple inversion attacks (Vec2Text, GEIA, MLC. On clinical text (MIMIC-III), concept leakage (e.g., gender) drops sharply, and a white-box upper-bound variant shows the black-box method performs near that ceiling.

### Weaknesses
1. The training cost should be clarified/ compared to the baseline since the proposed method involves the additional mask learning to identify privacy sensitive dimensions.

2. The code was not released.

### Questions
Above

### Soundness
3

### Presentation
3

### Contribution
3
