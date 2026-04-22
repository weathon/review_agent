# Interactions between crosscoder features: a compact proofs perspective

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Dictionary learning methods like Sparse Autoencoders (SAEs) and crosscoders attempt to explain a model by decomposing its activations into independent features. Interactions between features hence induce errors in the reconstruction. We formalize this intuition via compact proofs and make five contributions. First, we show how, \textit{in principle}, a compact proof of model performance can be constructed using a crosscoder. Second, we show that an error term arising in this proof can naturally be interpreted as a measure of inteaction between crosscoder features and provide an explicit expression for the interaction term in the Multi-Layer Perceptron (MLP) layers. We then provide two applications of this new interaction measure. In our third contribution we show that the interaction term itself can be used as a differentiable loss penalty. Applying this penalty, we can achieve ``computationally sparse" crosscoders that retain $60\%$ of MLP performance when only keeping a single feature at each datapoint and neuron, compared to $10\%$ in standard crosscoders. We then show that clustering according to our interaction measure provides semantically meaningful feature clusters, and finally that sleeper agents have significant interactions. Code is available at the following anonymous repository: \url{https://anonymous.4open.science/r/anon_crosscoders-2F77/}.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a method to recursively decompose crosscoder reconstruction error into inter-layer propagation terms and "feature transition error" and derives a computable interaction metric on MLP layers to quantify the error caused by non-dominant features when interpreted by a single dominant feature. The authors also demonstrate that using this metric as a regularization term to train the crosscoder significantly improves the "dominant feature share" and conduct a series of experimental validations.

### Strengths
1、Provides a hierarchical recursive error decomposition framework and proposes a practically computable metric, bridging formal analysis and applications.

2、The interaction metric on MLPs has explicit computational steps and can be directly used as a training regularization term or analysis tool.

3、The work evaluates multiple uses of the metric，showing diverse potential applications.

### Weaknesses
1、The steps of applying a coarse upper bound to ReLU and scaling using norms may result in a loose upper bound for the final error. The lack of formalization of attention and LayerNorm limits the completeness of the theory.

2、The experimental section only presents the results without further analysis and interpretation.

3、Several symbols and assumptions ($g^l_v(u_v), \hat{e}_v, e_v, W^{l}\_{in}$) are not clearly defined, making the logical chain of equations (6)–(9) confusing to read. Clearer symbols and explanations are needed.

### Questions
1、The core decomposition is based on a single dominant feature. Are there scenarios where multiple features dominate? How should it be extended in these scenarios?

2、The method validation is mainly conducted on small/medium-sized models in the TinyStories style. How does it perform in other task scenarios?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
By bounding the layer-wise reconstruction error and decomposing the transition term into a sum of single-feature contributions plus an interaction residual, the paper provides: (1) an in-principle procedure to derive compact proofs, and (2) a closed-form metric for quantifying feature interactions in MLPs. Empirically, the paper shows that training with an interaction penalty raises the dominant-feature share from around 30\% to 60\% without increasing reconstruction error.

### Strengths
1. Using the metric as a penalty increases the dominant-feature share dramatically with modest reconstruction cost (Fig. 1a) and materially improves dominant-only ablation fidelity (Fig. 2a). This provides evidence that the interaction metric captures meaningful interactions.
1. The closed-form interaction metric is simple to compute and linear in the number of active features per token and neuron, which contrasts the exponential difficulty of calculating STII.
1. The paper examines interactions, performs layer-wise ablations, explores clustering of interacting features, and probes a sleeper-agent setting where interactions plausibly matter. The empirical narrative is cohesive.

### Weaknesses
1. As the authors admit, the in-principle procedure given in Section 3 depends on loose triangle inequality, Lipschitz bounds, and the accumulation of errors across layers. The explicit closed-form result is presently given for the MLP layer only, which limits practical generalizability; the overall procedure is broader in scope but incomplete for attention and LayerNorm.
1. Application III's anomaly detection results rely on an interaction metric derived only for MLP layers (Eq. 9), while attention and layer norm are not formally covered. Appendix C sketches a sparse Q/K feature-interaction decomposition, but OV and LN remain unresolved and a complete compact proof is not yet available. Consequently, sleeper behaviors mediated by attention pathways may be under-detected by this MLP-based signal.
1. The method assumes a single "dominant" contributor per neuron, which can be brittle under near-ties. Would you clarify whether the top-1 choice is essential or whether it can be generalized to a few-hot variant?
1. Clarity has room for improvement:
    1. The term "proof" appears to mean a verification procedure rather than a formal theorem proof. For clarity to readers outside the compact-proofs community, please define this usage explicitly early on (excluding the fixed phrase "Compact proof"). In line 119, the authors mention "the full details of the proof," which seems to refer to the full details of the reduction and bounds derivation, rather than a formal theorem proof.
    1. The derivation of the interaction metric in Section 3 contains several notational inconsistencies and undefined symbols. Please clarify: $g_v^l(u_v)$, $g^l(u_v)$, $g_v^l(u)$ all appear but seem to refer to the same object; $\varepsilon$, $\hat e_v$, $W_{out}^l$, $b_{out}^l$ are used but not defined; what is the index $i$ appearing in Eq. (9)?
    1. There are minor typos throughout the paper. Some examples are as follows:
         - line 17: "inteaction" $\rightarrow$ "interaction"
         - line 112: "fronteir" $\rightarrow$ "frontier"
         - line 195: "the the"
         - line 206: "interaction metric Eq. (6)" $\rightarrow$ "interaction metric in Eq. (9)"
         - line 319: "implements parallelizes"
         - line 351: "investige" $\rightarrow$ "investigate"
         - line 403: "Anomlay" $\rightarrow$ "Anomaly"
         - line 414: "occurences" $\rightarrow$ "occurrences"
         - line 486: "insured" $\rightarrow$ "ensured"
         - line 496: "publically" $\rightarrow$ "publicly"
         - line 499: "where" $\rightarrow$ "were"

### Questions
1. What concrete obstacles prevent a closed-form interaction metric for attention and LayerNorm?
1. How sensitive are your results to the top-1 dominant assumption?

### Soundness
3

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
2

### Summary
This paper builds on previous work showing that crosscoders can be used as a way to generate a compact "proof" of model capabilities, showing an error bound on LLM capability using a crosscoder. While this error bound is too large to be useful in practice, this is used to derive a loss term that can minimize interactions between features in crosscoder. The authors then show that this penalized cross-coder finds interpretable features using auto-interpretability, and demonstrate that the remaining interactions between features be useful to debug strange behavior in models, such as finding sleeper agent behavior from previous work.

### Strengths
- The paper offers a lot of theoretical grounding. I do not have a mathematics or theory background, so cannot judge how accurate the theory is, but it seems reasonable to me.
- The paper trains crosscoders on a real LLM showing that the technique can scale past toy models, and shows interesting interpretability benefits like uncovering sleeper agents from previous work.
- Characterizing interactions between features feels like a good intermediate step between just detecting features and locating full circuits.
- The paper provides supporting evidence on LLM crosscoders via auto-interp results and case-studies.
- It sounds like this opens up a lot of avenues for future work that builds on the findings in this paper.

### Weaknesses
This paper is a bit out of my domain as I do not have a theory background, so I struggle to follow a lot of the ideas in the paper. Where I do not understand the theory I give the authors the benefit of the doubt. That being said:

- The crosscoders trained in this work are not very big, having an expansion factor of 2 would be very tiny for an SAE, for example. Crosscoders are very expensive to train, though, so maybe there is no way around this.
- The bounds from the proof are too large to be useful, so I'm not sure what practical benefit the framing of crosscoders as providing proofs has.
- The loss proposed in the paper seems to encourage crosscoder features to map to a single LLM neuron (if I understand things correctly). I don't understand why this is an inherently beneficial property.

### Questions
- L100: "We set the decoder bias to zero to avoid assigning it to features" what does it mean to assign the decoder bias to features?
- In section 2, where are the inputs and outputs to the crosscoder coming from? I assume the input is the residual stream at layer $l$, but what is the output? Is the the MLP output or the residual steam of the next layer?
- The paper repeatedly references sections of the "SM" for "Supplementary Material". I think this is referring to the Appendix, as I do not see any supplementary material provided in the submission, but I cannot find the sections referenced in the paper in the appendix. It would help if the authors directly link to the section in the appendix that is referenced in the main body rather than just referring to  the "SM".
- It sounds like the extra loss penalty tries to force crosscoder features to align exactly with one neuron, if I'm understanding this correctly. Why is this desirable?
- The paper picks a $g(x)$ that tries to make each feature align with a single neuron, but if I understand correctly, $g(x)$ can be picked arbitrarily. What other choices of $g(x)$ are worthwhile to use? Or is the $g(x)$ used in the paper the only choice that makes sense?

### Minor issues / formatting
- L17: "inteaction" should be "interaction"
- L66: typo in "the Section 7 of theSupplementary Material"
- L73: typo "meaninfgul"
- L75: "sleeper agents Hubinger et al." seems like it's missing a word

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
3

### Summary
This work proposes the interaction metric and showcases how it can be applied to train computationally sparse crosscoders, capture semantically meaningful interactions and be leveraged to for mechanistic anomaly detection.

### Strengths
1. The paper proposes a novel measure and showcases how this metric can be used to train computationally sparse crosscoders and capture semantically meaningful interactions.

2. The work provides supporting evidence for its claim.

### Weaknesses
1. The structure of the paper and the density of the sections make it difficult to follow the contributions, both the abstract and introduction poise the compact proof as the major contribution. Then the conclusion of section 3 and the limitation section highlight that the proposed proof is unlikely to lead to non-vacuous bounds and impractical for larger models. Sections 4, 5 and 6 then focus on the Interaction Metric which is presented as a byproduct of the compact proof in section 3.

2. Section 6 is short and appears rushed, especially when compared with section 3 and 4.

3. Terms in section 3 that are not introduced, $W^{l}{in}$ $W^{l}{out}$, $\hat{e}_{v}$ when used for the first time, breaking the reading flow.

4. General typos: 319 “implements parallelizes”, 320 “to to”.

5. The analysis appears to have been conducted on a small number of models, whether the proposed metric would work under different settings remains to be shown.

### Questions
1. I could not find the fidelity metric F in Bricken 2023, which I believe is the following paper https://transformer-circuits.pub/2023/monosemantic-features/index.html, if CMD+f for Fidelity I can’t find it. Link obtained from https://www.anthropic.com/research/towards-monosemanticity-decomposing-language-models-with-dictionary-learning, for the paper Bricken 2023 2nd reference in the reviewed paper., could not find “reinsert”, “insert” either, but I could find “We often normalize this by dividing by the difference in loss between the baseline transformer’s performance and its performance after ablating the MLP layer. This gives us a fraction of the MLP’s loss contribution that is explained by our transformer. However, the performance with an abated MLP may be an especially bad baseline, so this percentage is considered an overestimate.” (Bricken 2023)]]

2. The proposal gets diluted, is the proof the main proposal? Is the interaction metric the main proposal?

3. Equation (12) what is T?

4. Is there a link between the F in equations (11) an (12), it might be clearer to use a different letter if they are unrelated?

5. Could the authors provide an example of the confusion matrix mentioned in section 4.3 loc [432,433]
[[Equation is (11) could be clarified, what is the difference between $L_0$ and $_{ablated}$?]]

### Soundness
3

### Presentation
2

### Contribution
3
