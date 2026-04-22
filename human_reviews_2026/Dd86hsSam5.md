# Seeing What’s Not There: Negation Understanding Needs More Than Training

- Avg Score: 5.60
- Decision: Accept (Poster)
- Scores: 8, 6, 4, 2, 8

## Abstract
Understanding the negation in a sentence is an important part of compositional
understanding and logic in natural language. Many practical AI applications, such
as autonomous driving, include precise instruction with negations. For example,
following instruction to an AI assistant ”locate a parking spot without a vehicle”
requires the assistant to not confuse between presence and absence of vehicles. Al-
though joint embedding-based Vision Language Models (VLMs) like CLIP have
revolutionized multi-modal tasks, they struggle to interpret negation. To address
this limitation, recently many works proposed to solve the problem through a data-
centric approach by introducing additional datasets with hard-negative samples for
both image and text data. Contrary to these approaches, we present a zero-shot
approach to tackle the negation understanding problem. We probe the properties
of CLIP text embeddings and show that they follow compositional arithmetic op-
erations, which allow the addition or removal of semantic information directly in
the embedding space. We then present a rule-based approach to extract negated
text from given caption and then use it to explicitly remove corresponding se-
mantic information from original embedding, improving negation understanding
in VLMs. Our approach does not require expensive training process to induce
negation understanding into the model, and achieves the state-of-the-art perfor-
mance on popular benchmark for negation understanding. We improve baseline
CLIP model performance on NegBench from 25.5% to 67.0% for MCQ and from
50.9% to 56.1% for retrieval tasks. Even NegCLIP model which is fine-tuned on
negtion datasets, our approach boosts its MCQ accuracy from 54.03% to 66.22%
and retrieval accuracy from 59.25% to 60.1% showing strong performance.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
- This paper investigates the persistent difficulty of VLMs in handling linguistic negation. Rather than relying on large-scale negation datasets and fine-tuning, the authors propose a training-free embedding correction method that manipulates CLIP’s embeddings directly. By identifying the negated concept via rule-based parsing and removing its semantic component through a projection-based subtraction (plus a neutral anchor addition), the approach aims to improve negation understanding. 
- Experiments on NegBench and CC-Neg show large performance gains over CLIP and fine-tuned NegCLIP baselines. The authors also provide some analyses to show intuitions onto why their proposed method works.

### Strengths
- The idea of improving negation understanding through direct, post-hoc manipulation of text embeddings is novel and unconventional to the more standard data-centric strategies. The approach offers a new angle for thinking about compositional semantics in VLMs.
- The reported improvements on NegBench are substantial, even surpassing models fine-tuned on specialized negation datasets.
- The embedding-space visualizations (PCA plots) and ablation studies provide qualitative and quantitative insights into how the correction affects embedding geometry.
- The approach could inspire further work on embedding-space editing, compositional regularization, and interpretability.

### Weaknesses
- The main concern is the lack of theoretical grounding. The main motivation that negation corresponds to a linear operation in embedding space is not theoretically or empirically justified. CLIP embeddings are known to be highly entangled, and the projection subtraction used here lacks formal motivation beyond analogy to word-vector arithmetic. The choice of "neutral" words also seem arbitrary. This is totally fine as the emperical results are good, but the authors should be careful with their claims.
- The method’s success on simple caption pairs may not translate to more nuanced natural language understanding (more complex negation patterns).

### Questions
- Strange citation bug, e.g. line 143, Similarly TripletCLIP...
- Consider using an off-the-shelf tool for negation detection. In fact, your algorithm is quite similar to NegEx (https://pypi.org/project/negspacy/).
- Improvements of 40–50% absolute accuracy for a post-hoc method seem implausibly high. Did the authors reimplement the CLIP baselines and confirm that those results are comparable with previous works of using CLIP (and NegCLIP) on those benchmarks?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper points out that visual-language models such as CLIP have weak understanding of negation (no/not/without), often exhibiting affirmative bias. The authors propose a zero-training text embedding correction: rules extract the negated concept, estimate and subtract the "negation direction" from the embedding, significantly improving negation understanding without fine-tuning the model, and avoiding the decline in generality caused by fine-tuning the negation data.

### Strengths
The method requires no hard negation data collection or model fine-tuning; it only needs to correct the "negation direction" in the CLIP text embedding, resulting in extremely low deployment costs.

On NegBench, it improved the MCQ of CLIP from approximately 25.5% to approximately 67.0%, and the retrieval rate from approximately 50.9% to approximately 56.1%; it also brought considerable gains to the fine-tuned NegCLIP.

### Weaknesses
The method assumes that "negation" is approximately linear and directional in the embedding space, but the semantics of "missing" different concepts/attributes are not necessarily collinear; using the global d_global may be ineffective or even lead to incorrect offsets for fine-grained concepts.


Have you tried non-CLIP text towers (such as BLIP, SigLIP)? Even small-scale experiments should demonstrate the generality of the architecture.

How does the average overhead compare to standard CLIP text encoding? Is it easy to batch process to support large-scale retrieval? Please include a small performance table.

If λ is selected on COCO-MCQ, how would the retrieval be affected by fixing that λ to VOC/MSRVTT MCQ?

Rule-based negation scope is fragile. Negation scope extraction (Algorithm 1, page 6; Tables 6–8 in Appendix A.1, pages 12–14) has limited coverage of nested clauses, multiple negations, implicit negations (such as “few/barely”), quantifiers/comparison structures, etc. The main experiments are primarily based on NegBench patterns; generalization to real open-domain instructions remains uncertain.

### Questions
see the weaknesses

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses VLMs like CLIP’s poor negation understanding (e.g., ignoring "no"/"without"). Unlike data-centric methods using hard-negative datasets, it proposes a zero-shot, training-free embedding correction approach. In particular, the authors propose a rule-based method to extract negated concepts, then a correction formula removes their semantic info from embeddings (using directional offset and anchor embedding).

Evaluated on NegBench, it boosts CLIP’s MCQ accuracy from 25.5% to 67.0% and retrieval accuracy from 50.9% to 56.1%, outperforming fine-tuned models like NegCLIP.

### Strengths
S1. This paper is well written and is easy-to-follow.

S2. The performance improvement is clear.

S3. Sufficient visualizations are provided.

### Weaknesses
W1. The performance evaluation side is insufficient. Limited methods are compared. VLMs are mentioned but no further study is conducted.

W2. Typos should be corrected.

W3. According to Alg. 1, a set of negator words is utilized. The correctness of these words should be discussed. Besides, the quality of the generated embedding should be studied.

See questions for more details.

### Questions
Q1. Only CLIP is discussed. How about applying to SigLIP and SigLIP2? These encoders are more welcomed in practical use. 

Q2. MLLMs use CLIPs to encode features. MLLM benchmarks can also be used to evaluate the performance of VLMs. How about using your method to train a LLaVA-1.6 model. Will the general VQA performance boosted?

Q3. In Alg.1, return missing ' '; ’’ should be `` ''

Q4. Will synonyms influence your algorithm? What is the quality and diversity of corrected embedding?

Q5. What is the general retrieval, VQA, classification of the proposed method?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
his paper aims to improve negation understanding in embedding-based vision–language models such as CLIP, which struggle to correctly interpret negated textual inputs. Prior approaches have mainly relied on data-centric solutions that fine-tune models on curated negation datasets, but these tend to be resource-intensive and sometimes degrade general performance.

The authors propose a zero-shot, training-free approach that decomposes a given text query into affirmative and negated parts using a rule-based negation detection algorithm, then applies a linear correction in embedding space to form a corrected “negation-aware” text embedding. The method achieves strong performance improvements on NegBench, boosting CLIP’s retrieval accuracy from 50.9% to 56.1% and multiple-choice (MCQ) accuracy from 25.5% to 67.0%.

### Strengths
1- the proposed approach is efficient, zero-shot, and training-free, and easy to integrate into existing CLIP-based pipelines.

2- the paper shows consistent improvements on different model NegBench retrieval and MCQ tasks.

3- the paper is well organized and easy to follow.

### Weaknesses
1- the rule-based negation detection might overfit to the NegBench benchmark, which is synthetically generated using structured templates and paraphrasing; therefore, generalization to natural or free-form text is uncertain.

2- as the authors discussed in the limitation section, the method’s coverage of negation cues is incomplete. This approach misses implicit or compositional negations (e.g. “barely visible,” “few people”, "alone"), and may mis-handle nested clauses or double negation.

3- although the improvement on the MCQ task appears large (+50% relative), the absolute task structure (4 options) limits interpretability since a +25% absolute increase is effectively equivalent to eliminating one distractor option.

### Questions
**Some questions and suggestions**

1- have you considered replacing the rule-based negation scope detection with a learned or LLM-based decomposition approach? It might improve coverage and robustness to natural language variation.

2- the NLP community has proposed several deterministic negation detection methods (e.g., NegEx, DEEPEN, or syntactic-scope models) that go beyond keyword matching. Could you compare performance to some of those?

3- have you evaluated the approach on non-synthetic or human-written captions like real image descriptions to test out-of-distribution generalization?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes a training-free text-embedding edit for CLIP to handle negation: detect the negated span with rules, subtract the projection of the full caption embedding onto the negated-concept embedding, and add an “anchor” vector (Eq. 2). On NegBench, MCQ accuracy jumps from 24.7/24.3/27.5 (COCO/VOC/MSRVTT) to 72.5/78.6/50.0 for CLIP; negated-caption retrieval R-Neg@5 improves modestly (COCO +5.9; MSRVTT +5.5).

### Strengths
- Simple, training-free mechanism; clear formula and pipeline (Alg. 1, Eq. 2).

- Large MCQ gains on NegBench without changing positive retrieval (R@5).

- Geometry story (negation axis) supported by PCA visualizations (Fig. 3 and 4, Embedding space arithmetic visualization).

### Weaknesses
- Missing relevant related work: prior work [1] shows inference-time steering using CLIP geometry and visual-semantic arithmetic; it should be cited and contrasted (LM-side steering vs. text-side embedding edit). 

- Generalization beyond CLIP text towers (e.g., SigLIP/ALIGN, BLIP text encoders, MLLMs) is untested.

[1] Tewel, Yoad, et al. "Zerocap: Zero-shot image-to-text generation for visual-semantic arithmetic." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022.

### Questions
- It would be great to see if the proposed approach generalizes to similar but more recent models, such as SigLIP families. Is there any particular reason it was not tested on these?

### Soundness
3

### Presentation
3

### Contribution
3
