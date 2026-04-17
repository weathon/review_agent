# Model Already Knows the Best Noise: Bayesian Active Noise Selection via Attention  in Video Diffusion Model

- Decision: Accept (Poster)
- Scores: 6, 4, 6

## Abstract
The choice of initial noise strongly affects quality and prompt alignment in video diffusion; different seeds for the same prompt can yield drastically different results. While recent methods use externally designed priors (e.g., frequency filtering or inter-frame smoothing), they often overlook internal model signals that indicate inherently preferable seeds.
To address this, we propose ANSE (Active Noise Selection for Generation), a model-aware framework that selects high-quality seeds by quantifying attention-based uncertainty. At its core is BANSA (Bayesian Active Noise Selection via Attention), an acquisition function that measures entropy disagreement across multiple stochastic attention samples to estimate model confidence and consistency.
For efficient inference-time deployment, we introduce a Bernoulli-masked approximation of BANSA that estimates scores from a single diffusion step and a subset of informative attention layers. Experiments across diverse text-to-video backbones demonstrate improved video quality and temporal coherence with marginal inference overhead, providing a principled and generalizable approach to noise selection in video diffusion.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents ANSE, a framework for active noise selection in video diffusion models using BANSA, an attention-based acquisition function that identifies noise seeds yielding consistent and confident generations. BANSA adapts the BALD principle to attention space with lightweight, efficient inference. Experiments across multiple text-to-video backbones show that ANSE may enhance video quality and prompt alignment with slight overhead.

### Strengths
* The paper is clearly written.

* The accompanying website includes excellent figures and animations, which are much appreciated.

* The experiments and ablation studies are detailed and thorough.

### Weaknesses
1. **Ensemble Size**  
   - In Table 7, the metrics consistently improve as the ensemble size $K$ increases.  
   - Could the authors provide results with even larger $K$ values?  

2. **Denoising Steps**  
   - What is the number of denoising steps used for each model?  
   - It appears to be 50 based on the appendix, but this is only explicitly mentioned for the CogVideoX-5B backbone (L839).  

3. **Computation Budget (NFE)**  
   - From a neural function evaluation (NFE) perspective, the total cost is effectively $M + 50$ (or the actual number of denoising steps).  
   - What would happen if the baseline methods were also given an equivalent $M + 50$ computation budget?  

4. **Ablation on Bernoulli-Masked Attention**  
   - Please include ablation results for the Bernoulli-masked attention mechanism, particularly analyzing the effect of the masking probability parameter $p = 0.2$.  

5. **Evaluation Metrics**  
   - Consider including additional motion-specific metrics, such as FVMD [1], to better assess the claimed improvements.  
   - The sample videos on the website look good and visibly outperform the baseline. A human evaluation could further validate these qualitative improvements, as VBench—while comprehensive—may have limitations in capturing perceptual quality.  

Ref:\
[1] Liu, J., Qu, Y., Yan, Q., Zeng, X., Wang, L. and Liao, R., 2024. Fr'echet Video Motion Distance: A Metric for Evaluating Motion Consistency in Videos. arXiv preprint arXiv:2407.16124.

### Questions
Please see the [Weakness] section.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper addresses noise selection in video diffusion models by introducing a metric called BANSA, which utilizes attention information to estimate model confidence and consistency. It also proposes a Bernoulli-masked approximation to efficiently compute scores from a single diffusion step.

### Strengths
+ A novel metric is defined to guide noise selection.
+ The paper is well-organized and presents the idea clearly.

### Weaknesses
- Would the method remain effective when applied to a powerful base video model (e.g., WAN 2.2/2.1 14B), or when the video model has undergone post-training procedures such as RL or DPO?
- There is a lack of visualizations showing the score distributions for different prompts and samples from different noise inputs.
- How does the Bernoulli masking affect the score distribution? Is there a significant difference between using and not using the mask?
- It would be helpful to show generated samples from the same prompt but with different scores, to help validate the effectiveness of the proposed metric.
- What is the underlying reason that using this score leads to improvements in motion, composition, and artifact reduction? More insight into the mechanism would be valuable.

### Questions
Please refer to the strengths and weaknesses

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes ANSE (Active Noise Selection for Generation), an inference-time framework that improves text-to-video diffusion by selecting the initial noise seed that minimizes a new attention-based epistemic uncertainty measure, BANSA (Bayesian Active Noise Selection via Attention). This method improves video quality and temporal coherence across various T2V backbones and is approximated efficiently via Bernoulli-masked attention and layer truncation chosen by correlation analysis to keep overhead low.

### Strengths
1. The central idea of using attention-based uncertainty as an internal signal to guide noise selection is innovative. It shifts the paradigm from relying on external, often heuristic-based priors to a principled method that is aware of the model's own confidence.
2. Through approximations (Bernoulli masking, layer truncation), authors make the consistent quality gains with a reasonable inference overhead. Demonstrates robust generalization and real-world viability.

### Weaknesses
1. Lack of statistical confidence reporting. Improvements are modest in absolute terms; without standard deviations/CI or significance tests, it’s hard to assess robustness and effect sizes across prompts/seeds.
2. Lack of Sensitivity Analysis for Key Hyperparameters. The method relies on several key hyperparameters, notably the Bernoulli masking probability p (set to 0.2) . The paper does not provide a sensitivity analysis for these values. It is unclear how the optimal choices were determined and how robust the method's performance is to variations in these parameters.

### Questions
1. The paper mentions that BANSA can be applied to cross-, self-, or temporal attention. Did the authors investigate which type of attention map is most predictive of final video quality?
2. The Classifier-Free Guidance (CFG) scale is a crucial parameter for controlling prompt alignment in diffusion models. How does the choice of noise seed, as guided by BANSA, interact with different CFG values? Is it possible that a high BANSA score could be improved by using a higher CFG scale, or does the initial noise fundamentally constrain the quality regardless of the guidance strength?

### Soundness
2

### Presentation
2

### Contribution
2
