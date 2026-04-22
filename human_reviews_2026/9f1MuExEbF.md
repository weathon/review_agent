# Training-Free Spectral Fingerprints of Voice Processing in Transformers

- Avg Score: 3.33
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 6, 2

## Abstract
Different transformer architectures implement identical linguistic computations via distinct connectivity patterns, yielding model imprinted ``computational fingerprints'' detectable through spectral analysis. Using graph signal processing on attention induced token graphs, we track changes in algebraic connectivity (Fiedler value, $\Delta\lambda_2$) under voice alternation across 20 languages and three model families, with a prespecified early window (layers 2--5). Our analysis uncovers clear architectural signatures: Phi-3-Mini shows a dramatic English specific early layer disruption ($\overline{\Delta\lambda_2}_{[2,5]} \approx -0.446$) while effects in 19 other languages are minimal, consistent with public documentation that positions the model primarily for English use. Qwen2.5-7B displays small, distributed shifts that are largest for morphologically rich languages, and LLaMA-3.2-1B exhibits systematic but muted responses. These spectral signatures correlate strongly with behavioral differences (Phi-3: $r=-0.976$) and are modulated by targeted attention head ablations, linking the effect to early attention structure and confirming functional relevance. Taken together, the findings are consistent with the view that training emphasis can leave detectable computational imprints: specialized processing strategies that manifest as measurable connectivity patterns during syntactic transformations. Beyond voice alternation, the framework differentiates reasoning modes, indicating utility as a simple, training free diagnostic for revealing architectural biases and supporting model reliability analysis.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes using graph signal processing on attention-induced token graphs to detect "computational fingerprints" in transformers. The authors track changes in algebraic connectivity (Fiedler value) during voice alternation across 20 languages and 3 model families. The main finding is that Phi-3-Mini shows a large English-specific disruption, while other languages show minimal effects.

### Strengths
- **Interesting idea**: The idea of using graph spectral processing techniques for interpretability is quite natural and interesting. 
- **Systematic testing** across 20 languages and 3 model families. 
- **Statistical rigor** (bootstrap CIs, permutation tests, FDR correction, attempts to correct for differences in tokenization, etc.). I'm impressed by the authors' statistical rigor.

### Weaknesses
The work is still undermotivated, the results are weak, and it's unclear what this buys you over existing interpretability methods. The writing is needlessly technical and poorly structured.

A. Weak motivation and unclear utility

- **Why these metrics?** There is no explanation for why we should study the Fiedler value specifically. Appendix A.1 claims theoretical grounding, but the argument is hand-wavy: "models that struggle... may exhibit a breakdown in connectivity... leading to a significant drop in $\lambda_2$." This predicts the sign but not the magnitude or why this metric over alternatives. Appendix A.2 claims the Fiedler value is superior to other GSP metrics but this is based on extremely sparse and noisy data (figure 5). Moving some of this appendix into the main body would help, but the motivation provided there remains lacking. 
- **Why voice alternation?** The paper claims it "requires systematic attention reconfiguration" (lines 49–50) but doesn't explain why this particular transformation is special or what we learn from it. The same lines mention it's a "computational probe" but never cash this out.
- **What can you actually do with this?** The paper doesn't make clear whether this is:
    - A supervised method (in which case: where are comparisons to, e.g., linear probes and SAEs?). 
    - An unsupervised diagnostic (in which case: what predictions does it make? what can you discover?)
    - Just a correlation (in which case: so what?)
    In practice, the authors seem to use these metrics as a (supervised) binary classification signal, showing that these metrics make a distinction between different strategies/voice types/etc. The attempts to causally validate whether early attention structure drives spectral connectivity are suggestive but still do not answer the question of what this means in practice. 
- **Results are weak**:
    - The dramatic Phi-3 effect ($\delta \lambda_2 \approx -0.446$) means nothing without context. What's the baseline variance? What's a "large" effect in $\lambda_2$ units? Looking at the figures, I see that this is indeed larger than other languages and modesl, but it's only for one language in one model. What does that actually mean? 
    - Qwen and LLaMA show tiny, inconsistent effects (Figures 3-4). The paper spins this as "small but consistent" but they look like basically noise.

B. Poor writing and presentation

- **Unnecessarily technical**: The paper assumes prior knowledge of NLP linguistics and graph signal processing that's inappropriate for an interpretability and explainable AI audience. Examples:
    - Voice types (lines 194-195: "analytic," "periphrastic," "affixal," "non-concatenative") are never defined
    - Spectral diagnostics (Section 3.1) dumps four metrics with their mathematical definitions but no additional explanation. (Except for the Fiedler value, which is only explained in an appendix).
    - Phrases like (line 219)"tokenizer stress" and (line 209) "fragmentation entropy" appear without motivation.
- **Inconsistent terminology**:
    - "Fragmentation" refers to both a property of languages (line 217) and tokenization density (line 224)
    - "Tokenization" is used loosely throughout. Sometimes it refers to tokenization as segmentation, while other times it refers to a quantity (the token count differences). 
- **Unclear constructions**:
    - Lines 205-206: "Beyond syntactic effects, we find systematic links between spectral connectivity and tokenization, yielding a second layer of model-imprinted signatures rather than simple confounds." There are several examples like this that are incredibly difficult to parse, and where it is unclear what the authors are saying. 

**Summary**

The core idea – that attention connectivity patterns differ across models and languages and that this can be used for interpretability – is potentially interesting. But the paper needs major work:

1. Motivate why this analysis matters and what it enables
2. Compare to existing interpretability methods
3. Rewrite for clarity (assume readers know ML but not linguistics or GSP)
4. Strengthen the empirical story or acknowledge that the effects are mostly small.

Right now, this reads like a methods paper searching for an application.

### Questions
- Line 149: How do you define "passive vs. active" consistently across 20 typologically diverse languages?
- The hallucination detector (Section 7.3) feels forced in and should probably be cut.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors turn each layer’s attention map into a network of the input tokens and track “how well connected” that network is with a single number. Then they see how that number changes when you flip a sentence between active and passive voice. They run this test across 20 languages and three model families. Each model has its own 'fingerprint', e.g., Qwen2.5‑7B shows small, spread‑out changes, and LLaMA‑3.2‑1B shows moderate, systematic changes. This is a training‑free, lightweight diagnostic you can run on any model to quickly spot language coverage issues or architecture‑specific brittleness, without needing access to training data.

### Strengths
* The study is robust, covering 20-language design and three diverse families uncover architecture-imprinted patterns, rather than model-specific anecdotes

* The idea of a lightweight audit to detect language specialization/brittleness (e.g., Phi-3’s English-specific signature) and preliminary extension to hallucination detection makes the method societally and operationally relevant. This is a nice idea that the community will find interesting and possibly use for

### Weaknesses
* While head ablations help, most findings are correlational. The English-specific Phi-3 effect is interpreted as consistent with training emphasis. the paper mentions this is not definitive training-data attribution

* Building graphs from softmax attention (often noisy and not strictly causal) and then symmetrizing/aggregating heads may wash out meaningful directionality or head specialization. I don't feel too strongly about this but I think it's worth mentioning

* Focusing on voice alternation and early layers (2–5) is well-motivated, but relatively narrow. The reasoning-strategy results are preliminary and limited to one small task set; broader generalization (math, long-context, multi-turn) is not shown in this work

### Questions
How sensitive are conclusions to alternative aggregation that preserves per-head lambda_2 distributions (e.g., quantiles) rather than mass-weighted averages?

### Soundness
3

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
2

### Summary
The authors propose a new framework to uncover architectural signatures for algebraic connectivity. Primarily, the authors use the Fiedler value of a transformation of the post-softmax attention output as a proxy for understanding syntactic composition. The show results on multiple models, and many different tasks. They hypothesize that using this proxy is beneficial across several tasks such as voice alternation ( active and passive voice), hallucinations, reasoning though results on the latter are preliminary.

### Strengths
1. The paper devises a new metric to interpret model representations and linguistic processing, i.e. the Fiedler value of the transformed post-softmax attention.
2. The paper conducts rigorous statistical tests over 3 different models and several tasks to investigate evidence for generalization.
3. The hallucination results, restated from another paper are cool. Nice!

### Weaknesses
0. The figure texts are minuscule. Please increase the text sizes in all your figures. It's incredibly hard to read and interpret your figures.
1. The authors use the prespecified early-window mean $\Delta \lambda_{2}[2,5]$ as the primary endpoint (layers 2–5) -- a very important decision that the bucket in the appendix. However, the choice of layers 2-5 seems to be derived using trends observed across the three models, as opposed to being model specific. For instance, if I were to average the Fiedler value, I would pick 13-15 for Qwen, 12-15 for Phi-3-Mini and 4-7 for Llama. This is also consistent with the general findings in mechanistic interpretability that primary computation happens in the middle layers of the transformer, while early and late layers perform encoding vs decoding tasks.
2. The authors have conducted several rigorous statistical analyses, but the motivation that's tying these together is unclear to me. For eg. they state that  the spectral signatures they uncover correlate strongly with behavioral differences (Phi-3: r = −0.976), but these values are more moderate for Qwen (-0.627) and negligible for Llama (-0.14). I am finding this hard to reconcile with the claim, given that these results do not generalize, and are exhibited on a very small LLM.
3. The authors present several weak results as opposed to fully explaining one core claim, as well as have figures with very minuscule text. This paper was very hard to read/understand/motivate as a consequence. Could the authors please provide figures w enhanced text during the rebuttal process? I don't fully understand your results, and want to make sure I am judging you fairly.
4. The causal validation results are also unclear to me: How did you identify the heads to ablate? You could identify important heads using causal mediation analysis, and then repeat this experiment. Currently, the selection of the ablation sites seems arbitrary to me.
5.  The authors use fiedler connectivity to track reasoning performance: 

Line 408: 
Fiedler connectivity tracks performance: CoT (69.5%) and Standard (60.0%) yield positive Fiedler shifts, whereas CoD and ToT show negative shifts with lower accuracy. This alignment suggests that successful reasoning is associated with maintained/enhanced graph connectivity and motivates spectral-guided prompt selection based on induced $\Delta\lambda_2$

This seems to be a claim? What's your evidence? Some empirical evidence would be useful here. 

6. 7.2 is a core finding, where the Fiedler value is aligned w a strong English specific connectivity. This suggests that the Fiedler value is tied to in-distribution data, but I am not sure how these effects can be generalized to syntactic processing/interpretation of model representations
7. The authors state results wrt tokenizer fragmentation -- what is tokenizer fragmentation correlated with? Why should I be interested in this result?

### Questions
(Questions stated alongside weaknesses)

### Soundness
2

### Presentation
2

### Contribution
2
