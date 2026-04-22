# AdaSCALE: Adaptive Scaling for OOD detection

- Avg Score: 4.67
- Decision: Reject
- Scores: 4, 6, 8, 2, 4, 4

## Abstract
The ability of the deep learning model to recognize when a sample falls outside its learned distribution is critical for safe and reliable deployment. Recent state-of-the-art out-of-distribution (OOD) detection methods leverage activation shaping to improve the separation between in-distribution (ID) and OOD inputs. These approaches resort to sample-specific scaling but apply a static percentile threshold across all samples regardless of their nature. In this work, we propose AdaSCALE, an adaptive scaling procedure that dynamically adjusts the percentile threshold based on a sample's estimated OOD likelihood. This estimation leverages our key observation that OOD samples exhibit significantly more pronounced activation shifts at high-magnitude activations under minor perturbation compared to ID samples. AdaSCALE enables stronger scaling for likely ID samples and weaker scaling for likely OOD samples, creating highly separable energy scores. Our approach achieves state-of-the-art OOD detection performance, outperforming the latest rival OptFS by **14.94%** in near-OOD and **21.67%** in far-OOD datasets in average FPR@95 metric in the ImageNet-1k benchmark across eight diverse architectures.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes AdaSCALE, an adaptive scaling procedure
that dynamically adjusts the percentile threshold based on a sample’s estimated
OODness. This estimation leverages a key observation: OOD samples exhibit
significantly more pronounced activation shifts at high-magnitude activations under
minor perturbation compared to ID samples. AdaSCALE achieves state-of-the-art OOD detection
performance, outperforming the latest rival OptFS by 14.94% in near-OOD and
21.67% in far-OOD datasets in average FPR@95 metric on the ImageNet-1k
benchmark across eight diverse architectures.

### Strengths
1.	Clear motivation and novel observation: OOD samples exhibit greater sensitivity of high-magnitude activations to minor perturbations, providing an intuitive and effective signal for designing an adaptive mechanism.
2.	Overall, AdaSCALE  is  a simple yet effective method.
3.	The experiments  quite extensive with significant results: across 10 architectures and 3 datasets by tuning mere one hyperparameter for a given setup.

### Weaknesses
1. AdaSCALE shares a similar design  with ATS (Krumpl et al., 2024), as both methods adaptively adjust scaling parameters during post-processing to enhance ID and OOD separability. It would be better to further clarify the differences in design rationale and implementation mechanisms between the two approaches and provide a more in-depth comparative analysis.

2. The experiments employ gradient-based perturbations, which significantly increase computational overhead and compromise practicality. It remains unclear how applicable this approach would be in real-time systems such as autonomous driving.

3. Although the paper claims that hyperparameters are transferable, p_min and p_max still require manual tuning for each model and dataset.

4. The paper lacks  visual examples (e.g., t-SNE plots or histograms of energy scores) illustrating the improved separability between ID) and OOD samples.

5. The core hypothesis of the paper that OOD samples exhibit greater instability in high-magnitude activations under minor perturbations is currently supported only by intuitive reasoning, lacking theoretical justification or references to prior work.

6. The implementation details are mostly complete, but the perturbation setup needs more clarification.

### Questions
Please refer to the Weaknesses. The overall experimental evaluation is very comprehensive; however, it would be valuable  to further address above questions to strengthen the paper’s clarity and I will take the rebuttal into consideration when revising my final score.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a novel adaptive scaling procedure for OOD detection that dynamically adjusts the detection threshold based on the likelihood of OODness for individual samples. Through extensive experiments, the superiority of the method against several baselines is established.

### Strengths
The paper is generally easy to follow, and most of the intuition for the proposed methodology and the different components of the algorithm are empirically justified.

The authors provide good figures to better convey the empirical results and justifications

The authors provide a good set of experiments and ablation studies that examine different properties of the algorithm and different sensitivities.

### Weaknesses
Despite the easy flow of the text, some parts are abstract and require some more detailed domain knowledge, which could have been better established. This mainly pertains to Fig 1, the introduction, and sec 4.1. 

Some parts of the paper seem to be stating contradicting requirements and observations. This could also be my lack of familiarity with the different terminology and their differences:
- In the preliminaries, it is mentioned that higher scores are for OOD samples. Yet the paragraph at 84 talks about giving lower scores to OODs. Same comment regarding 4.1.
- 284: "Fig 3 illustrates Q'/Q' > Q/Q", whereas the figure shows the opposite of this. Also, in Fig 3 caption.

There are some baselines that are missing. Based on my own experiments, it is critical that the authors **must** include a comparison with LINe [1] for my final decision. Other baselines can be found in [2]

Although I understand the regime of the experimental setup and its comparison with similar post-hoc methods, a comparison with methods that assume access to OOD data during training could also be useful. It is not necessary for the methodology of this paper to beat such methods, but a comparison with such work could be useful. 


[1] Ahn YH, Park GM, Kim ST. Line: Out-of-distribution detection by leveraging important neurons. In2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2023 Jun 17 (pp. 19852-19862). IEEE.

[2] https://github.com/Jingkang50/OpenOOD

### Questions
If the absolute perturbation sum over activations is indeed such a good measure of the likelihood of OODness, why not use this metric itself as the scoring function? Why do you first utilize this to acquire OODness and then utilize that information for another scoring function?  

Also see weaknesses.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes AdaSCALE, a post-hoc OOD detection method that adaptively adjusts activation scaling based on the estimated OODness of each test sample. The method is built on an empirical observation: OOD samples exhibit larger shifts in top-k activations under small input perturbations, while ID samples remain stable. Leveraging this phenomenon, the authors compute a perturbation-induced activation shift statistic, convert it into a normalized OODness score using an empirical CDF with only a few ID samples, and dynamically determine the percentile threshold for scaling. Extensive experiments across CIFAR and ImageNet-1k, eight architectures, FSOOD settings, corrupted inputs, and adversarially trained networks demonstrate consistent SOTA performance and strong generalization. The method is parameter-efficient, requires minimal ID samples, and preserves ID accuracy.

### Strengths
- The paper provides a compelling mechanistic explanation for why activation scaling works for OOD detection: semantic activations are stable for ID samples but unstable for OOD samples under perturbations. This moves scaling-based OOD detection from heuristic to a more principled footing. I consider this a significant conceptual advancement.

-Prior scaling methods use a static percentile threshold; adapting it per-sample based on estimated OODness is a meaningful and novel extension.

- Demonstrates consistent SOTA performance across: CIFAR & ImageNet-1k and 8 architectures

- only need a very small number of ID samples (as few as 10) for calibration and preserves accuracy,.

- No model retraining or gradients beyond attribution; easy to integrate into existing systems.

### Weaknesses
- Although trivial/random perturbation works, the selection of perturbation magnitude and pixel selection lacks theoretical grounding, and different perturbation strategies may influence results.

- Requires an extra forward pass + top-k operations, resulting in ~2–4× latency vs. fixed-scaling baselines, which may be limiting in latency-critical systems.

- The work discovers an important phenomenon, but lacks a deeper theoretical unification of scaling and perturbation in relation to existing methods. Such a theory would substantially enhance the impact.

### Questions
See Weakness

### Soundness
3

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
3

### Summary
The paper addresses out-of-distribution (OOD) detection in deep networks and identifies a limitation of recent post-hoc methods: they use sample-specific activation scaling with a fixed percentile threshold, which is suboptimal for separating in-distribution (ID) vs OOD samples. As models scale up, a static threshold can’t account for the varying “OOD-ness” of each sample. 
AdaSCALE is proposed as an adaptive scaling procedure that adjusts the scaling percentile per sample based on an estimated OOD likelihood. The core insight is that OOD inputs exhibit larger shifts in their top activations under small perturbations compared to ID inputs.
By quantifying this activation shift (via a gradient-based input perturbation and measuring changes in high-magnitude activations), the method assigns a higher percentile (stronger scaling) to inputs likely ID and a lower percentile (weaker scaling) to those likely OOD. This yields more separable energy scores for detection, effectively reducing OOD confidence for OOD samples while preserving ID confidence.

### Strengths
- The method demonstrates substantial performance gains. AdaSCALE achieves state-of-the-art OOD detection results on challenging benchmarks (ImageNet-1k), significantly outperforming prior post-hoc methods. For example, it dramatically lowers false positive rates (FPR@95) compared to strong baselines (e.g., vs. OptFS and SCALE) in both near-OOD and far-OOD settings.
- The paper shows robust generalization on several different network architectures (ResNets, EfficientNet-V2, ViT, etc.), where AdaSCALE outperforms or matches previous methods on each, addressing a known shortcoming of some prior methods that were tied to specific models.
- The paper is well-structured and clear. It provides a detailed algorithm, conceptual diagrams, and ablations. The authors situate AdaSCALE in context by comparing to a wide range of baselines (MSP, ODIN, ReAct, ASH, SCALE, LTS, OptFS, etc.), and perform comprehensive experiments (including metrics like AUROC/FPR, near-vs-far OOD, and even a “full-spectrum” test incorporating covariate shifts. The writing is generally easy to follow, with a solid explanation of the method and the intuition behind it.

### Weaknesses
- While the adaptive scaling idea is effective, the contribution can be seen as a relatively incremental extension of existing methods. AdaSCALE builds directly on the activation scaling paradigm introduced by ASH/SCALE/LTS. The key insight (that OOD causes unstable high activations under perturbation) extends the known observation from ReAct that OOD samples have high activation magnitudes. Thus, the method’s novelty, though useful, feels a bit additive: it combines perturbation-based OOD scoring with the existing scaling framework.
- Certain design elements in AdaSCALE appear somewhat heuristic and might lack theoretical justification. For instance, the OOD likelihood score is a composite of the top-k activation shift and a “correction” term for ID bias, weighted by a parameter $\lambda$. This was introduced to counter the high variance in OOD shift metrics, which indicates that the raw signal can be noisy or overestimate OOD likelihood without tuning. The need to hand-craft this combination (including hyperparameters $k_1$, $k_2$, $\lambda$) suggests the solution required empirically balancing factors, rather than deriving a unified criterion. The current design feels overcomplicated and may benefit from simplifications focusing on the most essential theoretical insights.
- AdaSCALE introduces additional hyperparameters (e.g., percentile range $p_{min}$/$p_{max}$, top-k sizes, perturbation size, etc.) and a more complex inference procedure. This raises two concerns:
  - The paper mentions using an automatic search (via OpenOOD) to set these, but it’s unclear how sensitive the results are to these settings. In Table 7, it is shown that AdaSCALE is somewhat insensitive to single-hyperparameter variations with ResNet-50 on ImageNet-1k. However, there is no study involving multiple-hyperparameter variations (e.g., two hyperparameters varying at the same time; considering that AdaSCALE has a lot of hyperparameters) or study on other architectures and datasets.
  - Moreover, it is unclear whether the automatic search (via OpenOOD) utilizes an OOD validation set. If it does, then the current evaluation setting may not be fair to methods that do not require any OOD validation sets. Another related concern is how sensitive AdaSCALE is to any mismatch between the OOD validation set and the actual OOD test set.
- Unlike some post-hoc methods that require no extra data, AdaSCALE does rely on a small set of ID validation samples to compute the empirical CDF for its adaptive threshold. The paper emphasizes this is a minimal requirement (only 10 samples used), but it seems that Table 9 only shows the result from a single experiment (per setting). The impact of using only a handful of samples to estimate the distribution of the OOD score isn’t deeply explored. One might wonder if this calibration could be unstable or biased if those few samples aren’t perfectly representative. This is a minor weakness given how small the requirement is, but it’s worth noting as a practical consideration.
- There is a typo in the formula for the proposed adaptive percentile threshold: a right parenthesis is missing.

### Questions
- The adaptive scaling relies on the heuristic that OOD samples’ top activations are unstable under small perturbations. Are there types of OOD inputs or shifts where this heuristic might fail or be less effective? For example, if an OOD sample is very near the ID distribution (or if the model’s features are insensitive to the chosen perturbation), would AdaSCALE risk mis-classifying it as ID? Conversely, could certain ID samples that are borderline or noisy exhibit large activation shifts and be mistaken for OOD?
- OptFS was designed for cross-architecture generalization using a piecewise constant scaling function. AdaSCALE empirically outperforms OptFS across architectures, but could the authors elaborate on *why* adaptivity gives a *significant* edge here? From Figure 2, the signal doesn’t appear to be very clear (the standard deviation regions largely overlap between ID and OOD).

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes AdaSCALE, a post-hoc method for out-of-distribution (OOD) detection that adaptively adjusts the activation scaling percentile based on the estimated OOD likelihood of each sample. The method is motivated by the observation that OOD inputs exhibit stronger top-k activation shifts under small perturbations than ID inputs. The authors design a mechanism to estimate sample OODness via activation differences and dynamically modulate the scaling strength. Extensive experiments on ImageNet-1k, CIFAR-10/100, and various architectures demonstrate consistent gains over prior post-hoc methods (e.g., SCALE, LTS, OptFS), achieving state-of-the-art performance.

### Strengths
- Motivation and observation are clear and intuitive. The link between activation stability under perturbation and OOD likelihood is conceptually appealing.

- Strong empirical performance across datasets and architectures shows that the adaptive scaling strategy generalizes better than fixed-threshold methods.

- Method simplicity.

### Weaknesses
- Incremental novelty. While the adaptive percentile idea is reasonable, it extends existing activation scaling works (e.g., ASH, SCALE, LTS) rather than introducing a fundamentally new principle.

- Limited theoretical grounding. The paper claims that activation shift reflects OODness, but lacks a formal analysis or statistical justification. The perturbation mechanism and threshold mapping are mostly heuristic.  It may disentangle this effect from other confounders (e.g., gradient norm, layer saturation, or input magnitude)

- Clarity issues. Some equations and algorithmic steps (e.g., computation of OODness score and eCDF calibration) are dense and could benefit from clearer notation or pseudo-code.

- Minor reproducibility concern. The reliance on gradient-based perturbations may introduce stochasticity; details of implementation choices (e.g., ε magnitude, percentile selection ranges) could be more transparent.

### Questions
- How sensitive is AdaSCALE to the choice of perturbation direction and strength (ε)? Could adversarial directions yield different results?

- Maybe the authors could provide more intuition or quantitative evidence that the activation shift magnitude correlates with the sample’s epistemic uncertainty rather than simply gradient magnitude?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 6

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces a post-hoc OoD detection method, AdaSCALE. The basis of the proposal is that OoD inputs exhibit higher activation instability for minor perturbations compared to ID samples. AdaSCALE proposes an adaptive scaling mechanism and computes a per sample OODness score Q'.

### Strengths
- the phenomenon according to which ID inputs activate stable features while OoD samples cause unstable high-magnitude activations under small perturbations is well highlighted and illustrated (Figure 2)

- the score mapping strategy is straightforward and effective. The authors rely on an empirical CDF built using a small ID validation set. This results in a simple and non-parametric way of translating the scores into percentile ranges.

- extensive comparison with other relevant approaches (e.g. ReAct, ASH, SCALE, LTS, OptFS) and strong overall performance, including some great results (e.g. FPR@95 of 61.17 on imagenet-1k vs 71.91 for OptFS)

### Weaknesses
- despite the generalization claims, the presented method seems to be systematically over-tuned for each test setup. The authors state that a fixed set of hyperparams may be used, and that by tuning only $p_{max}$ "near-optimal performance can be achieved", see Table 5. However, the main results (Tables 2 and 3) are obtained following a humongous level of finetuning (and implicitly computational cost), apparent in Tables 28 and 29. Presenting Tables 2/3 and claiming generalization is grossly misleading. I see two ethical ways out : either the authors claim a good generalization across all setups and finetuning goes to appendix, or the generalization claim is dropped and Table 5 goes to Appendix.

- the OODNess metric is artificially complex; the introduction of the correction term + 2 additional parameters is justified by the "high variance" of Q alone. However, Q has a FPR@95 of 59.43 vs 58.97 for the full metric.In my opinion, this shows that Q works quite well, and that the additional complexity and cost implied by the use of Q' are hard to justify.

- I also am very doubtful about the positive impact of the proposed perturbation mechanism; the "trivial" method performs a backward pass and is 2.91x heavier, while the random perturbation is only 1.56x slower. The benefit for the doubled cost is minimal (Table 22). This trade-off seems poor and is poorly described; the work should probably propose the random selection by default and introduce the other mechanism as a costlier alternative. Overall, the 3x slowdown (compared to SCALE) is not even once mentioned in the abstract / intro / conclusion; in my opinion this is a significant omission and a failure in transparency about the method's trade-offs.

### Questions
Please see the three points raised in the Weaknesses section.

### Soundness
3

### Presentation
2

### Contribution
3
