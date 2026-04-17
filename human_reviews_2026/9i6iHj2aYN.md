# A Close Look at Negative Label Guided Out-of-distribution Detection in Pre-trained Vision-Language Models

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Advances in pre-trained vision-language models have enabled zero-shot out-of-distribution (OOD) detection using only in-distribution (ID) labels. 
Recent methods in this direction expand the label space with negative labels to enhance the discrimination between ID and OOD inputs. 
Despite their promising progress, there remains a limited understanding of their empirical effectiveness in open-world scenarios, where negative labels can arbitrarily diverge from real OOD ones.
This paper bridges this research gap with the helm of a novel energy-based framework, where the energy function is built upon the margin between the similarity of an input to ID labels and that to negative labels.
Guided by this framework, we prove that the inherent tolerance of such methods to the sampling bias essentially stems from estimating the worst-case energy function over a KL-constrained set of potential distributions centered on the negative label distribution.
Furthermore, our theoretical analysis reveals that existing methods suffer from over-pessimism and consequently high sensitivity to outliers. 
Provably, we can alleviate these problems by leveraging Rényi divergence to refine potential distributions.
Extensive experiments empirically manifest that our method establishes a new state-of-the-art across a variety of OOD detection settings.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies post-hoc out-of-distribution (OOD) detection with pre-trained vision–language models (e.g., CLIP) using negative labels, and targets two limitations of prior work such as NegLabel: (i) insufficient theoretical justification for open-world effectiveness and (ii) over-pessimism induced by KL divergence. The authors propose an energy-based framework in which the energy is defined by the similarity margin between ID labels and negative labels, and they replace KL divergence with Rényi divergence (with a tunable order) to induce milder, polynomial-weighted worst-case distributions, thereby mitigating KL’s over-pessimism. Experiments on ImageNet-1K (ID) and multiple OOD datasets show state-of-the-art AUROC and FPR95, outperforming NegLabel and generalizing across CLIP architectures, input sizes, and negative-label corpora.

### Strengths
1. The paper offers a transparent energy–margin modeling and a distributionally robust optimization (DRO) perspective that interprets NegLabel as a KL-ball worst-case energy estimator, thereby characterizing robustness under negative-label/OOD mismatch. By adopting Rényi divergence, the worst-case weights transition from exponential to polynomial, theoretically alleviating KL’s over-pessimism.

2. The method surpasses NegLabel and other zero-shot baselines on ImageNet-1K and several OOD sets, and maintains strong performance across different backbones, input resolutions, and negative-label corpora, indicating good robustness and transferability.

### Weaknesses
1. Both the theory and algorithmic design explicitly rely on negative labels and are not readily extensible, in their current form, to non-negative-label OOD detection paradigms.

2. The study does not systematically adopt near-OOD or full-spectrum OOD\[1] / OpenOOD benchmark\[2], which are more representative of open-world conditions.

3. The implementation fixes L = 10,000 negatives (following NegLabel’s WordNet mining) and a single prompt template. If the negative set has limited coverage or bias, the subsequent Rényi-based optimization cannot compensate for semantic gaps. The paper lacks systematic ablations on L and on the quality of negative-label mining.

4. Although the paper states that per-input optimization of $\beta^*_x$ via 15-step SGD has negligible overhead, it does not report batch-level throughput, end-to-end latency, or side-by-side runtime comparisons with NegLabel, making practical deployability hard to assess.

\[1] full-spectrum OOD:Full-Spectrum Out-of-Distribution Detection

\[2] OpenOOD benchmark:OpenOOD v1.5: Enhanced Benchmark for Out-of-Distribution Detection

### Questions
1. The implementation appears to fix templates such as “The nice \<label>.”. Please add prompt-sensitivity ablations covering common prompt variants to validate robustness to textual prompts.

2. Incorporate near-OOD and full-spectrum OOD / OpenOOD benchmarks for a more comprehensive open-world evaluation.

3. Report sensitivity to L (e.g., 1k / 5k / 20k) and compare different mining strategies to characterize the coverage–performance trade-off and clarify dependence on corpus scale/quality.

4. Current ablations focus on small/medium backbones. Please include results on RN50x16, ViT-H, ViT-G, etc., to examine whether gains diminish when the base model already separates ID/OOD well.

5. Provide throughput and latency statistics attributable to optimizing $\beta^*_x$, with side-by-side comparisons to NegLabel—including per-sample cost and pipeline-level latency—to facilitate engineering assessment.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper addresses the problem of negative-label-guided Out-of-Distribution (OOD) detection in pre-trained Vision-Language Models (VLMs) by proposing a new energy-based theoretical framework. The authors first argue that existing methods (like NegLabel) are implicitly equivalent to estimating a worst-case energy function over a KL-divergence-constrained set of distributions.

### Strengths
1. The paper is clearly written, with solid theoretical derivations and thorough experiments.  
2. The proposed methods achieves state-of-the-art results on several OOD detection benchmarks.

### Weaknesses
1. The entire motivation of the paper rests on a critical assumption: that the original NegLabel method (Eq. 2) is "functionally equivalent" 5 to the authors' derived, KL-based worst-case energy function, $\hat{S}_{NegLabel}$ (Eq. 8). Howerer, the sole evidence provided for this equivalence is an empirical observation that their average performance in Table 1 is "on par" (93.76 vs 93.77 AUROC). This is an empirical observation, not a formal theoretical proof.  
2. The paper claims to solve the "over-pessimism" (Theorem 3) identified in the authors' own proxy model (Eq. 8), not necessarily the original NegLabel method. We cannot be certain that the original NegLabel (Eq. 2) suffers from this exact same problem, or that the authors' solution (Rényi divergence) is truly fixing an inherent flaw in NegLabel. The paper may be fixing a flawed proxy model of its own construction.  
3. Unlike the NegLabel baseline, which requires a single forward pass and softmax 11, the new method requires solving an input-specific optimization problem (Eq. 12)  for every single test sample to find the optimal $\beta_x^*$. The authors state this is done via 15 steps of SGD and claim the overhead is "negligible". However, this iterative process (the gradient of Eq. 12) requires computation over all $L=10000$ negative labels at each step. This is fundamentally more complex and slower than the baseline.
4. The paper positions itself as a "Zero-Shot Training-free" method. However, its test-time procedure, optimizing an input-specific $\beta_x^*$ via 15 SGD steps 22, makes it functionally very similar to TTA methods. The paper lacks a clear discussion of how this input-specific optimization is conceptually related to or different from other TTA OOD methods like AdaNeg. Is the proposed method a form of TTA itself? If so, how does it compare to AdaNeg as a standalone baseline (not just in combination)? This ambiguity in its positioning weakens the paper's contribution.
5. It is easy to see that the generalizability of this paper is limited. The authors only conduct experiments on ImageNet-1K Benchmarks. Following previous works, the authors should conduct extensive  near-OOD and far-OOD experiments on the OpenOOD benchmark. Besides ImageNet, the authors also should evaluate the proposed method on smaller-sized CIFAR10/100 datasets with the OpenOOD setup.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

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
This paper develops a theoretical framework to analyze CLIP-based zero-shot OOD detection with negative labels from the perspective of density estimation. The authors model the OOD score as an energy function based on the margin between similarities of test-time inputs to ID and negative labels, and demonstrate that NegLabel basically augments the negative label distribution by constructing a distribution set contained within a Kullback–Leibler (KL) ball centered on it. They also propose replacing KL divergence with Rényi divergence to mitigate the over-pessimism of NegLabel. Extensive experiments are conducted to verify the effectiveness of the proposed method.

### Strengths
1. This paper is well-motivated, focusing on the research gap in the theoretical understanding of NegLabel.
2. This paper presents a theoretical framework of OOD detection with negative labels from the perspective of energy function and density estimation, which is novel and interesting.
3. This paper conducts extensive experiments, including the conventional ImageNet-1K OOD benchmark, domain-generalizable OOD benchmarks and various ablation study, to verify the effectiveness of the proposed method.

### Weaknesses
1. **There seem to be many typing mistakes in the main text, which makes this work unclear and hard to understand.** For example, there seems to be a mismatch between index $i$ and $j$ in the $\sum_{j=1}^{K} exp[h(x,y_i)/T]$ in Eq. (2). Similar mistakes can be found in Eq. (4), Eq. (5), Eq. (6), Eq. (8) and Eq. (13). In addition, should $r=1.05$ in Line 321 Page 6 be $\gamma=1.05$?
2. **There is ambiguity about the score function** of the $\hat{S}_{NegLabel} (x;f)$ in Table 1. 

Is it just $\sum_{i=1}^{K} \frac{e^{h(x,y_i)/T}}{\sum_{j=1}^{L}e^{h(x,\hat{y}_j)/T}}$ or the $\hat{E}(x;\theta)$ in Eq. (8)? 

3. **There is a detail missing in the derivation of Theorem 1.** How is the $\delta$ cancelled out in the fifth equality in Appendix B.2?
4. **The presentation of the proposed method is unclear.** I suggest that the authors present a clear algorithmic summary of the proposed OOD detection pipeline.
5. The paper claims that the proposed method adds negligible computational overhead, but **no quantitative evidence is provided**. Could the authors evaluate and compare the average computation time of learning $\beta_x^*$ and a single model forward process for one test-time input to further demonstrate the time efficiency of the proposed method?
6. Could the authors evaluate the proposed method and other baselines on the **CIFAR-100** OOD benchmark?

### Questions
Please see the weaknesses.

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
4

### Summary
This paper presents a novel energy-based framework for zero-shot out-of-distribution (OOD) detection using pre-trained vision-language models. The method defines an energy function based on the margin between similarities to in-distribution (ID) and negative labels, providing theoretical insights into its robustness to sampling bias. The authors show that existing approaches suffer from over-pessimism and outlier sensitivity, which they address using Rényi divergence to refine potential distributions. Extensive experiments demonstrate state-of-the-art performance across diverse OOD detection scenarios.

### Strengths
1. The paper targets post-hoc OOD detection, which is more computationally efficient and practical than retraining-based methods, making it suitable for real-world deployment.

2. It introduces a novel energy-based formulation that models the margin between similarities to ID and negative labels, offering a principled multi-modal extension to prior single-modality approaches.

3. The work provides a solid theoretical foundation by interpreting negative-label methods through the lens of density estimation and KL-constrained optimization, explaining their robustness to distribution shifts.

4. The paper is well-written with clear logic and thorough experiments, making the methodology and results easy to follow and convincing.

### Weaknesses
1. The introduction of Rényi divergence and energy-based modeling increases theoretical and computational complexity, which may hinder scalability.

2. Despite improvements, the approach still relies on the quality and representativeness of sampled negative labels, which may not fully capture the diversity of real-world OOD data.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3
