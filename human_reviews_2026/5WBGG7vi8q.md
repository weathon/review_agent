# LIHE: Linguistic Instance-Split Hyperbolic-Euclidean Framework for Weakly-Supervised Generalized Referring Expression Comprehension

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 4, 6

## Abstract
Existing Weakly-Supervised Referring Expression Comprehension (WREC) methods, while effective, are fundamentally limited by a one-to-one mapping assumption, hindering their ability to handle expressions corresponding to zero or multiple targets in realistic scenarios. To bridge this gap, we introduce the Weakly-Supervised Generalized Referring Expression Comprehension task (WGREC), a more practical paradigm that handles expressions with variable numbers of referents. However, extending WREC to WGREC presents two fundamental challenges: supervisory signal ambiguity, where weak image-level supervision is insufficient for training a model to infer the correct number and identity of referents, and semantic representation collapse, where standard Euclidean similarity forces hierarchically-related concepts into non-discriminative clusters, blurring categorical boundaries. To tackle these challenges, we propose a novel WGREC framework named Linguistic Instance-Split Hyperbolic-Euclidean (LIHE), which operates in two stages. The first stage, Referential Decoupling, predicts the number of target objects and decomposes the complex expression into simpler sub-expressions. The second stage, Referent Grounding, then localizes these sub-expressions using HEMix, our innovative hybrid similarity module that synergistically combines the precise alignment capabilities of Euclidean proximity with the hierarchical modeling strengths of hyperbolic geometry. This hybrid approach effectively prevents semantic collapse while preserving fine-grained distinctions between related concepts. Extensive experiments demonstrate LIHE establishes the first effective weakly supervised WGREC baseline on gRefCOCO and Ref-ZOM, while HEMix achieves consistent improvements on standard REC benchmarks, improving IoU@0.5 by up to 2.5%. The code is available at https://anonymous.4open.science/r/LIHE.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper introduces WGREC (weakly-supervised generalized REC) where an expression can match zero / one / many objects. It argues one-stage WREC pipelines can’t handle multi/no-target and “semantic collapse” from pure Euclidean similarity. The method (LIHE) has two stages:
(1) Referential Decoupling: LIHE uses a VLM to parse the original expression into K short, instance-level phrases and even decide K (possibly 0) (Appendix G prompts referenced). 
(2) Referent Grounding: Grounding each short phrase with an anchor-based detector using HEMix, a weighted mix of Euclidean and Lorentzian-hyperbolic similarities.
On results, Table 1 reports WGREC numbers; WREC tables show that plugging HEMix into prior WREC models yields small but consistent gains (≈+1–2% absolute).

### Strengths
1. Clear two-stage design that addresses cardinality: letting a VLM split multi-target expressions into single-target queries is intuitive and fits weak supervision. The decoupling stage and its prompt scaffolding are well-described.

2. HEMix is simple, differentiable, and drop-in. The paper explains why hyperbolic similarity can preserve hierarchies (general→specific) and shows mild but reliable lifts over Euclidean on both WGREC and WREC.

3. The paper is well-written and easy to understand.

### Weaknesses
1. The proposed method only shows a marginal improvement on the performance with the help of a VLM ( Qwen2.5-VL from the appendix's implementation deatils).  Furthermore, There’s no ablation across VLMs or prompting quality; thus it’s hard to separate LIHE’s contribution from the VLM’s perception strength.

2. Lack of comparison with sota methods, the latest method LIHE compared with in Table 1 is Refclip, which is a lit bit outdated. 

3. The WGREC table largely mixes fully-supervised GREC models with one weakly-supervised baseline (RefCLIP), so we don’t learn how LIHE fares vs recent weakly-supervised WREC variants adapted to GREC. 

4. Thanks the hoest of the authors, but as they also stated that the inference is slow (~2 FPS) due to the VLM, so I do not see any advantages of LIHE, as the performance gians are also limited.

### Questions
1. Please report WGREC performance comparisons across more VLMs and with no image input (text-only decoupling) to quantify reliance on VLM visual grounding. Also, which scale of the Qwen2.5-VL is used in the paper? 

2. Provide a per-category analysis: where does hyperbolic help most (highly hierarchical nouns, “people/man/woman”) vs where Euclidean suffices? Also plot α in HEMix learned/selected across datasets.

3. Since the decoupling stage is the bottleneck (~2 FPS), can batching multi-crop VLM queries or caching improve throughput? Report end-to-end latency vs a single-stage WREC baseline.

### Soundness
3

### Presentation
4

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
1

### Summary
This paper proposes a new task called Weakly-Supervised Generalized Referring Expression Comprehension (WGREC). It introduces the LIHE framework to handle variable numbers of referents through expression decomposition and hybrid similarity modeling. Experiments show that LIHE and its HEMix module achieve strong performance and set the first effective WGREC baseline.

### Strengths
strength
1. This paper is easy to follow
2. This paper introduces a large of experimental results to varify the introduced method

### Weaknesses
weakness
1. When a complex epxression is divided into several short sentences, some relationship between objects in expression will be broken. How do you handle this situation?
2. what is Hyperbolic Mapping and Mapping Euclidean? 
3. Based on your desription, The primary contribution of this paper is introduced HEMix. How does it work?

### Questions
weakness
1. When a complex epxression is divided into several short sentences, some relationship between objects in expression will be broken. How do you handle this situation?
2. what is Hyperbolic Mapping and Mapping Euclidean? 
3. Based on your desription, The primary contribution of this paper is introduced HEMix. How does it work?

### Soundness
3

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
The paper proposes the important and more challenging new paradigm of weakly-supervised generalized referential expression comprehension (WGREC), and introduces two key innovations in a two-stage framework (LIHE): VLM-based referential decoupling and hybrid geometric similarity HEMix.

### Strengths
The introduction of HEMix and its bias-variance analysis has theoretical value. The experimental results perform well on the WGREC task.

### Weaknesses
1.  **The alignment of the SimE/SimH scale is unclear.** HEMix linearly combines the Euclidean similarity and the hyperbolic similarity, but the paper does not explain the statistical distribution (such as mean, variance, and typical magnitude) of these two similarities in numerical scale, nor does it clarify whether normalization or temperature recalibration was performed before the combination. In the appendix, these two are regarded as "two biased estimators of the same true similarity", and based on this, a bias-variance analysis and the optimal α are provided. If the original scales of the two are significantly different, this theoretical assumption may not hold.

2.  **The error propagation has not been quantified.** The LIHE framework is a two-stage cascaded system. Any prediction errors of the VLM in the first stage (K prediction errors, sub-expression generation errors, or hallucinations) will inevitably be propagated downstream to the referential localization stage. The current paper lacks a systematic analysis of this error propagation.


3. **The VLM relies on insufficient robustness**, only verifying a single VLM with a fixed Prompt, lacking a systematic sensitivity assessment across the dimensions of VLM and Prompt (such as PC strictness, example quantity/order and multiple sampling voting), and also failing to quantify the hallucination rate and its impact on the final performance indicators.

### Questions
Refer to the weaknesses above.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces a novel task (WGREC) and a two-stage framework (LIHE) for grounding natural language expressions to variable numbers of visual objects under weak supervision. In the first stage, a VLM decomposes a complex expression into single-instance sub-phrases and infers the number of targets. In the second stage, the proposed hybrid similarity HEMix is used for localization. Extensive experiments show that LIHE establishes the strong weakly-supervised baseline for WGREC while also improving WREC benchmarks when HEMix is plugged in.

### Strengths
1. The paper introduces WGREC, the first weakly-supervised setting that allows zero or multiple referents. This relaxes the restrictive one-to-one mapping assumed by all prior WREC work.
2.The two-stage framework is rationally designed. The referential decoupling stage utilizes VLMs to tackle the challenging cardinality ambiguity problem in weak supervision, effectively breaking down a complex multi-target grounding problem into simpler single-target sub-tasks. The referent grounding stage then builds upon and enhances established single-stage WREC methods, making the overall framework robust.
2. The method is validated not only on the novel WGREC task but also on standard WREC benchmarks, demonstrating the generalizability and plug-and-play nature of the HEMix module. Extensive ablation studies and cross-dataset validation strongly support the necessity of each component and the robustness of the approach.

### Weaknesses
1.The entire first stage depends entirely on the output quality of the VLM. As shown in the failure cases in Fig. 4(B), the VLM can hallucinate or be biased towards frequent categories due to its training data. 
2.The referential decoupling stage relies solely on a single VLM (Qwen2.5-VL), without comparative experiments against other advanced large language models. This limitation hinders the assessment of how different model capabilities affect this stage's performance.
3.The paper suffers from notable structural flaws: critical parameters (e.g., HEMix weight α, prompt templates) are scattered between the main text and appendix, severely compromising reproducibility and reading coherence. The absence of a dedicated related work section further obscures the positioning of this research, making it difficult for readers to discern its scholarly contribution.

### Questions
1.Can the mixing weight α in HEMix be adaptively learned rather than manually set?
2.Will the capability differences of large language models directly affect the text output quality in the referential decoupling stage?

### Soundness
3

### Presentation
3

### Contribution
3
