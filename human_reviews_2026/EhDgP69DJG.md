# PMark: Towards Robust and Distortion-free Semantic-level Watermarking with Channel Constraints

- Decision: Accept (Poster)
- Scores: 6, 8, 6, 8

## Abstract
Semantic-level watermarking (SWM) for large language models (LLMs) enhances watermarking robustness against text modifications and paraphrasing attacks by treating the sentence as the fundamental unit. However, existing methods still lack strong theoretical guarantees of robustness, and reject-sampling–based generation often introduces significant distribution distortions compared with unwatermarked outputs. In this work, we introduce a new theoretical framework on SWM through the concept of proxy functions (PFs) -- functions that map sentences to scalar values. Building on this framework, we propose **PMark**, a simple yet powerful SWM method that estimates the PF median for the next sentence dynamically through sampling while enforcing multiple PF constraints (which we call channels) to strengthen watermark evidence. Equipped with solid theoretical guarantees, **PMark** achieves the desired distortion-free property and improves the robustness against paraphrasing-style attacks. We also provide an empirically optimized version that further removes the requirement for dynamical median estimation for better sampling efficiency. Experimental results show that **PMark** consistently outperforms existing SWM baselines in both text quality and robustness, offering a more effective paradigm for detecting machine-generated text. The source code is available at https://anonymous.4open.science/r/PMark.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
PMARK is a semantic-level watermarking on a new Proxy Function (PF) framework: a PF maps sentences to scalars, enabling distortion-free sampling and detection. The method extends single-channel median sampling to a multi-channel scheme to improve the robustness against modification and paraphrasing.

### Strengths
1. The idea of using multi-channel to constrain the watermark to enhance the evidence is interesting.

2. The theoretical results seem solid.

3. Experimental results show superior performance over prior works.

### Weaknesses
1. Error bar is not presented for the main results. Some values are too close to those of prior works to claim the best. 

2. Online and offline versions should be better considered as part of a general workflow and not require the user to make a decision.

3. Even though authors tried to solve the problem of watermarking short texts, which should be harder for this method, the requirements of computational resources, as well as time complexity, will be a problem.

### Questions
More analysis on the dependence of PF on the encoder. 

More guidance on the hyperparameter settings from the theory to practical deployment.

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
4

### Summary
This paper presents a unified theoretical framework for semantic-level watermarking (SWM) in natural text generation and introduces **PMARK**, a well-founded and practical method that is provably distortion-free and robust. As research on SWM remains nascent, partly due to limited theoretical guidance [1], this work could help pave the way for subsequent advances, even though several limitations warrant further study.

[1] Zhao, et al. Sok: Watermarking for ai-generated content. 2025 IEEE S&P.

### Strengths
1. Clean and solid theoretical analysis. The paper unifies prior work such as SemStamp, k-SemStamp, and SimMark under a proxy-function framework and presents a **provably distortion-free multi-channel** sampling rule furthermore.
    
2. Novel paradigm. This is the first distortion-free SWM method, whose robustness is enhanced by multi-channel constraints. The ideas in this work is elegant and effective, and may help deepen our understanding of semantic-level text watermarking.
    
3. Solid experiments. The experiments are solid, confirming the generation quality and robustness of PMARK. The ablation study is detailed and convincing.
    
4. Good writing and presentation. The paper is well organized and easy to follow; the figures and charts are clear, aside from a few minor issues (see Questions).

### Weaknesses
1. Theoretical analysis of multi-channel robustness. Although Eq. 11 characterizes the robustness of single-sentence watermarking, could the authors provide a more detailed theoretical analysis of **multi-channel** sampling under common conditions?
    
2. Discussion of token-level related work. Some token-level watermarks [2,3] use scoring functions during sampling as well. Including these works in the discussion would help contextualize the relationship to SWM.
    
3. There are also some limitations discussed in the paper including the analysis of offline method and the extension to n-shot undetectable SWM, which are leaved for future works by authors. Could the authors denote some possible directions for subsequent improvement?
    

[2] Giboulot, E. and Furon, T., 2024. WaterMax: breaking the LLM watermark detectability-robustness-quality trade-off. NeurIPS 2024.

[3] Bahri, D. and Wieting, J., 2024. A watermark for black-box language models. arXiv preprint arXiv:2410.02099.

### Questions
1. Although the authors claim that random seed generation for SWM is challenging and that prior approaches may not be suitable, could the authors suggest a possible solution or research direction for this problem?
    
2. As noted in the Weaknesses, could the authors provide a theoretical analysis of the adversarial robustness of multi-channel sampling?
    
3. (Minor) The annotations in Figure 4 are relatively small.

### Soundness
3

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
The paper introduces PMARK, a semantic-level watermarking method that aims to be both distortion-free and robust to paraphrasing. The core abstraction is a proxy function that maps a sentence to a scalar. PMARK employs rejection sampling on median-based half-partitioning of semantic space along several orthogonal channels. The authors prove distortion-freeness and demonstrate improved semantic robustness across different benchmarks.

### Strengths
1. The proxy function formalization and closed‑form distortion analysis clarify why split‑and‑reject SWMs distort and motivate a median-split alternative; the multi‑channel constraint is a simple, elegant idea that increases evidence density.
2. Strong results across two backbones, two datasets, many attacks, and against several prior methods; PMARK‑Online exceeds prior SWM baselines by large margins.

### Weaknesses
1. This work assumes the private key is drawn from a prior distribution. If the prior key is hashed from the context, the semantic robustness would break since the random seeds can't be recovered.
2. Multi-channel sampling introduces additional computation overhead and seems inherently incompatible with batch decoding.

### Questions
1. Can you provide an end‑to‑end evaluation where the detector does not query the generator (unknown M and prompts), using the offline variant?
2. Is there formal guarantees for the offline method?
3. How does PMARK work when an attacker obfuscates sentence segmentation, such as randomly changing all periods to commas?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper first introduces a new theoretical framework on semantic-level watermarking with the concept of proxy functions, which covers existing methods.
It points out that they are not robust to attacks due to distribution distortions.
Building on the framework, they propose PMark, which is distortion-free.
The experiments show that PMark improves the robustness against paraphrasing-style attacks and also outperforms in text quality.

### Strengths
* The paper's theoretical framework is accurate and clear. It is very important for the community to systematically understand the literature and the limitations of the current stage.

* The experiments are comprehensive, and the settings are timely and accurate.

* The presentation of the paper is clear and fluent.

* From my knowledge, this is the first distortion-free semantic watermark paper.

### Weaknesses
* (a minor point) Some symbols look overlapped but with different meanings. I am unsure if there is any way to prevent it. For example, $m$ denotes the size of the green list in Theorem 2; $m_v$ denotes the median of $F$ later in L213. There might be a chance the readers build a connection between, but indeed they should not.

### Questions
* (It is an open question and not related to the assessment.) Are there any insights on how the method would work (or not) if we do cross-sentence paraphrasing attacks (e.g., there will be sentence deletion, merging, etc). 

-- Will the smooth counting mechanism and the soft-count-based z-test help?

-- Why do you focus on the sentence-level instead of the n-gram chunk? The former may have a large uncertainty in each sentence's length. Maybe if a sentence only contains 2 words but has 1 word removed in attack, the sentence-level semantics will shift a lot.

### Soundness
4

### Presentation
4

### Contribution
4
