# Towards understanding multimodal in-context learning

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 4, 6

## Abstract
Multimodal large language models (MLLMs) often exhibit in-context learning (ICL) abilities, yet the conditions under which multimodal ICL emerges, and the mechanisms underlying it, remain poorly understood. In particular, how training data statistics and architectural choices jointly shape this capability is still an open question. To address this, we reverse-engineer multimodal ICL by training small transformer models on controlled synthetic classification tasks with varying data statistics and architectural choices.
We begin by revisiting core principles of unimodal ICL in modern transformers. While several prior findings replicate, our experiments yield two notable observations. First, Rotary Position Embeddings (RoPE), a standard component in contemporary LLMs, can delay the onset of ICL circuits. Second, larger models require stronger statistical cues in the training data for strong ICL to appear.
Extending our analysis to the multimodal setting reveals a fundamental learning asymmetry. Once a primary modality has learned a core ICL circuit from statistically diverse data, a secondary modality can reach comparable ICL performance with far less data complexity. In contrast to the unimodal regime, we further find that model scaling consistently improves multimodal ICL.
To understand why these patterns emerge, we turn to mechanistic analysis. Using progress measures that track circuit formation during training, we show that ICL accuracy is tightly correlated with the strength of an induction-style circuit that copies labels from in-context exemplars that match the query. Both unimodal and multimodal ICL rely on this induction mechanism, while multimodal training primarily refines and extends it across modalities.
Together, these results provide a mechanism-level account of ICL in modern multimodal transformers, offer explanations for several empirical phenomena observed in MLLMs, and introduce a controlled testbed for future work on multimodal ICL.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates multimodal in-context learning (ICL) mechanisms through controlled synthetic experiments on transformers trained with Gaussian Mixture Models. The key findings include: (1) RoPE impairs ICL by disrupting induction heads, (2) scaling unimodal models raises the data complexity threshold for ICL emergence, favoring memorization when data complexity is fixed, (3) in multimodal settings, a primary modality develops the core ICL circuit with secondary modalities then requiring minimal data complexity, and (4) scaling improves multimodal ICL unlike the unimodal case. The authors introduce "progress measurements" that track attention patterns and validate selected findings on Qwen2.5-VL and IDEFICS.

### Strengths
- The paper is well written, and the exposition is clear. 
- The observation about RoPE and its negative impacts on induction heads, is interesting and appears to be genuinely novel.
- The controlled synthetic experiments are well designed, with quantifiable progress measurements to successfully isolate factors affecting ICL emergence in the chosen regime, extending prior mechanistic work on induction heads to multimodal settings.

### Weaknesses
- The biggest weakness is that although benefits of cross-modal alignment are discussed, the paper entirely omits any analysis of how two modalities actually interact (information transfer, cross attention circuits) during ICL inference, which is often the key driver in modern LLMs (see plethora of literature on improving multimodal ICL). It remains unclear whether genuine cross-modal reasoning emerges or whether the language model simply performs ICL on features that include visual information. 
- The empirical validation on actual MLLMs is limited, with just one benchmark and minimal mechanistic probing. For the trends that are validated, these would be largely predictable even without any mechanistic analysis and as such, it is unclear if the key findings hold in practice. 
- More importantly, the findings may not generalize beyond the data-scarce regime studied. For instance, the negative effects of RoPE or scaling effects might change or even become irrelevant with appropriate data scaling, as in practice, larger models with RoPE still demonstrate excellent ICL performance. This issue is not discussed in the paper. 
- Minor typo: Line 422: "two main components" but only lists 1

Overall: While this paper makes solid strides towards understanding multimodal ICL, some claims appear to be under-substantiated due to missing cross-modal interaction analysis and limited real MLLM validation. If the authors can argue why this is not the case on either front, the reviewer can lean towards accepting.

### Questions
See weaknesses

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates the mechanisms underlying in-context learning (ICL) in both unimodal and multimodal  models. Through controlled synthetic experiments, the authors explore how architectural choices (e.g., use of RoPE) and data statistics influence the emergence of ICL. They report several key findings: (1) RoPE can hinder ICL circuit formation; (2) larger models require stronger statistical cues for ICL to emerge; and (3) in multimodal settings, once one modality learns the ICL mechanism, a secondary modality can achieve comparable behavior with much less data diversity. The work further argues that multimodal ICL relies on the same induction-style circuit as unimodal ICL, refined rather than replaced by multimodal training. Some of the findings are validated with Qwen and IDEFICs models.

### Strengths
* Understanding the mechanism of ICL, particularly in multimodal settings, is an important and underexplored area.

* he paper is generally well-written, with findings and methodology clearly presented.

* The identified role of RoPE and the modality asymmetry provide potentially valuable directions for future work.

### Weaknesses
* The claim that "Larger models consistently exhibit reduced ICL" appears to contradict evidence from large LLMs, where scaling tends to improve ICL emergence. 

* ICL and zero-shot performance should be monitored together. The paper should discuss whether the observed effects might stem from general model capability rather than ICL-specific mechanisms. For example, low/high ICL performance might be due simply to high/high model performance.

* Experimental details lacking: How many in-context examples (“shots”) were used? What modalities were considered, and how the authors define different modalities? What were the model sizes and dataset scales used in each experiment?

* The validation on Qwen and IDEFICS is insufficient to substantiate most of the claims. Showing that the model scale correlate with ICL performance (which shown in previous work, including e.g. the IDEFICS paper) and the strenfght of induction heads are minor part of the paper claims. 

* Similar findings about preimary modality bias (dominance of a “primary” modality) have been reported before [1]. The paper should better position its contributions relative to these works.

[1] Baldassini, Folco Bertini, et al. "What makes multimodal in-context learning work?." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.

### Questions
* How joint (e.g., early-fusion) multimodal training might affect the observed asymmetry?

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
3

### Summary
The paper investigates how ICL emerges and operates in MLLMs, aiming to uncover the mechanisms behind this ability at the circuit level.
Using controlled synthetic multimodal classification tasks, the authors systematically vary data statistics and architectural components.

### Strengths
1. The use of synthetic Gaussian mixture data allows precise manipulation of multimodal statistics, which strengthens causal claims.

2. Identification that RoPE suppresses induction circuits.

3. Discovery of asymmetry between primary and secondary modalities and how pretraining on one modality installs transferable ICL circuits.

### Weaknesses
1. The work largely follows existing analyses from prior studies, offering limited novelty.

2. Experimental validation under real settings is limited, reducing confidence in the result’s generalizability.

### Questions
1. How is the swapped-label implemented? 

2. It is not immediately clear why the ICL–IWL balance performs best when α₂ ≈ 1 based on the figure. Could the authors provide further justification?

3. The subgraphs in the bottom portion of Figure 6 are difficult to interpret. Can the authors clarify what each represents and how they relate to the main texts?

4. What exactly is meant by “raw high-dimensional feature”?

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
The paper systematically studies how in-context learning (ICL) emerges in multimodal transformers using a controlled synthetic setup built from Gaussian mixtures. By varying data complexity factors and introducing quantitative diagnostics—PHStrength, IndStrength, TLA, and CLA—the authors trace the formation of induction-style attention circuits. The key findings are: (1) rotary and other relative positional encodings weaken ICL formation; (2) scaling increases the data complexity threshold for unimodal ICL, promoting memorization; (3) multimodal ICL is asymmetric, with the primary modality bootstrapping learning for the secondary; and (4) pretrained encoder quality is crucial for strong multimodal ICL. Results are validated on large models like Qwen2.5-VL and IDEFICS. The work provides a clear, mechanistic view of how architecture, scaling, and representation quality interact to produce ICL behavior in multimodal transformers.

### Strengths
Careful, controlled experimental design — synthetic GMM data + control over K, ε, B, α gives clear causal evidence for how data statistics drive ICL in both uni- and multimodal regimes. This leads to mechanistic progress measurements — PHStrength, IndStrength, TLA, CLA are well-motivated, quantitatively predictive, and allow the authors to track circuit formation over training. 

Clear novel architectural insight: RoPE harms induction circuits — the paper demonstrates that RoPE (and ALiBi) consistently reduce ICL accuracy vs absolute PEs and produce more diffuse attention that weakens previous-token / induction heads. T

Important multimodal asymmetry finding — showing that a decoder pretrained on a high-diversity primary modality can bootstrap ICL such that the secondary modality needs far less diversity/burstiness is an intuitive and practically useful result for dataset and architecture design.

### Weaknesses
Synthetic → real generalization limited — while synthetic control is powerful, results hinge on idealized GMMs; the real-data validation is limited (Qwen2.5-VL analysis and a small Omniglot probe). Broader real-world tests are needed to confirm generality.

Positional encoding recommendation could be risky in practice — the paper shows RoPE/ALiBi hurt ICL in these tasks, but RoPE brings other benefits (length generalization, training stability). The manuscript does not fully quantify tradeoffs (e.g., effect on other tasks, or hybrid encodings), which limits actionable guidance. 

Scaling analysis might conflate capacity vs data regime — the unimodal result (“larger models need more complex data to show ICL”) is interesting, but the experiments use fixed data budgets. It remains unclear whether larger models trained with proportionally more data would still favor in-weight memorization. The compute/data scaling frontier isn’t fully explored.

### Questions
How might these findings inform training recipes for large production MLLMs (positional encoding choice, pretraining mix, encoder pretraining)? The paper gives implications — could you make them more prescriptive?
How sensitive are the RoPE-vs-absolute results to context length and dataset complexity? Is there a regime where RoPE still dominates (e.g., much longer contexts)?
In unimodal scaling experiments, if you scale data proportionally with model size, does the ICL threshold still increase? Please report model×data scaling curves.

### Soundness
3

### Presentation
3

### Contribution
3
