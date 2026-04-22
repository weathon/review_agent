# HiGS: History-Guided Sampling for Plug-and-Play Enhancement of Diffusion Models

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 4, 6, 6

## Abstract
While diffusion models have made remarkable progress in image generation, their outputs can still appear unrealistic and lack fine details, especially when using fewer number of neural function evaluations (NFEs) or lower guidance scales. To address this issue, we propose a novel momentum-based sampling technique, termed history-guided sampling (HiGS), which enhances quality and efficiency of diffusion sampling by integrating recent model predictions into each inference step. Specifically, HiGS leverages the difference between the current prediction and a weighted average of past predictions to steer the sampling process toward more realistic outputs with better details and structure. Our approach introduces practically no additional computation and integrates seamlessly into existing diffusion frameworks, requiring neither extra training nor fine-tuning. Extensive experiments show that HiGS consistently improves image quality across diverse models and architectures and under varying sampling budgets and guidance scales. Moreover, using a pretrained SiT model, HiGS achieves a new state-of-the-art FID of 1.61 for unguided ImageNet generation at 256$\times$256 with only 30 sampling steps (instead of the standard 250). We thus present HiGS as a plug-and-play enhancement to standard diffusion sampling that enables faster generation with higher fidelity.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The author introduces a novel momentum-based sampling technique called History-Guided Sampling (HiGS), which improves the quality and efficiency of diffusion sampling by incorporating recent model predictions into each inference step. This approach yields more realistic outputs with finer details. Experiments across various diffusion models and datasets demonstrate promising improvements.

### Strengths
* The motivation is compelling: the authors propose that the Euler sampler used in diffusion models can be interpreted as performing stochastic gradient descent (SGD) on a time-varying energy function. Based on this perspective, they argue that the gradient estimate can be enhanced to improve both the efficiency and quality of the sampling process, incorporating a momentum-based term inspired by STORM.

* HiGS is conceptually straightforward and can be easily integrated into pre-trained diffusion models. Moreover, it introduces minimal additional computational overhead.

* The experiments are comprehensive in terms of datasets and model types, and the performance is persuasive.

### Weaknesses
* In Section 4.2, the methodology is presented in a relatively loose and unstructured manner. It would be beneficial to include clearer explanations of the design choices and a more thorough ablation study to identify which components contribute most to performance. Additionally, clarifying the rationale behind how these techniques are organized and integrated would strengthen the overall presentation.

* The tuning required for the combination of techniques in the HiGS design may pose a practical burden for users, potentially limiting its accessibility and ease of adoption.

### Questions
* What motivated the choice of Equation 5 as the formulation for HiGS over other possible alternatives?

* Could you elaborate on why "the benefits of HiGS were most evident during the early and middle sampling steps," as stated in line 226? What underlying factors contribute to this behavior?

### Soundness
4

### Presentation
3

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
This paper introduces History-Guided Sampling (HiGS), a novel, momentum-based sampling technique designed to enhance the image quality and sampling efficiency of existing diffusion models. The core insight is that integrating the history of recent model predictions into the current inference step can steer the process toward more realistic outputs with improved fine details, particularly when using a low number of neural function evaluations (NFEs) or lower classifier-free guidance (CFG) scales. The method leverages the difference between the current noise prediction and a weighted average of past predictions. Crucially, HiGS operates as a plug-and-play enhancement, requiring no additional training or fine-tuning of the base diffusion model, and introduces negligible computational overhead. The authors demonstrate HiGS's effectiveness across various models (including Stable Diffusion 3 and SiT) and achieve a new state-of-the-art FID of 1.61 using a pretrained SiT model, showcasing both strong efficiency and quality gains.

### Strengths
1. High Utility and Plug-and-Play Nature: The primary strength is that HiGS is a highly practical and post-hoc sampling enhancement. It requires no modifications to the pre-trained diffusion model or additional training, making it universally applicable and easy to integrate into existing workflows.

2. Exceptional Efficiency and Quality: The method boosts image quality even under highly constrained inference budgets (low NFEs). By leveraging a simple weighted average of history predictions, the approach introduces practically no additional computational cost.

3. Effectiveness: HiGS is shown to be effective across diverse models and architectures (SD3, SiT, SDXL-Flash), indicating that the concept of "history-guided" momentum is a fundamental enhancement to the general sampling process, rather than a solution tailored to one specific model.

### Weaknesses
1. Lack of Novelty and Differentiation from Prior Work (UCGM [1]): The approach bears significant conceptual similarity to prior work, specifically UCGM (see [1]). The paper needs to provide a rigorous comparison (both theoretical and empirical) to clearly articulate how HiGS fundamentally differs from and improves upon UCGM's methodology, particularly in its formulation of the history/momentum term.

2. Inconsistency in Theoretical Formulation and Experiments: The theoretical foundation is established using the EDM (Elucidated Diffusion Models) formulation, yet the experimental evaluation is conducted on models trained using different frameworks (DDPM for DiT/SDXL, Flow Matching for SiT/SD3). This mismatch between the theoretical background and the experimental setup must be reconciled. The authors should either generalize the theory to cover the different diffusion formulations used or provide an EDM-based experimental validation.

3. Weak or Misleading Theoretical Justification (SGD/Energy): The introduction of concepts related to Stochastic Gradient Descent (SGD) and energy functions to justify the momentum term (e.g., logic around lines 186-187) appears tenuous and possibly forced ("shoehorned"). The authors must critically re-evaluate and simplify the theoretical derivation to establish a clear, direct, and convincing link between the proposed history-guided mechanism and the underlying physics/optimization, or remove the irrelevant theoretical connections.

4. Insufficient Interaction Analysis in Ablation Study: The ablation studies (e.g., Tables 7, 8, and 9) are presented linearly and fail to adequately analyze the interaction between the various design components. It is unclear how designs are built upon each other (e.g., whether Table 8 is based on Table 7, or how $g(H_k)$ in Table 9 integrates with prior designs). The authors must clarify the baseline for each subsequent ablation and provide more comprehensive, possibly combinatorial, results to demonstrate the synergistic or interdependent effects of their design choices.

5. Theoretical Justification of the Momentum Term: While the mechanism is clearly defined, the paper should provide a deeper theoretical analysis or intuition explaining why the divergence from a historical average (the "difference between the current prediction and a weighted average of past predictions") systematically leads to higher quality, more detailed results. Is it effectively smoothing the variance of the noise prediction, or is it introducing a form of implicit guidance toward the average manifold learned during training?

6. Hyperparameter Sensitivity Analysis: As a momentum-based technique, HiGS inherently introduces a set of new parameters (e.g., the weighting factor for the history, the depth or length of the history used). The paper needs to include a thorough sensitivity analysis demonstrating how robust the results are to changes in these hyperparameters. Are the reported gains dependent on a finely tuned set of parameters?

7. Potential for Temporal Artifacts/Bias: I would like to see discussion or analysis on whether incorporating prediction history could inadvertently introduce minor temporal correlations or specific texture biases into the generated images, even if the overall FID improves. A few qualitative examples showing cases where HiGS fails or introduces unintended artifacts (if any exist) would strengthen the paper.


[1] Unified Continuous Generative Models. In page 7 of  https://arxiv.org/pdf/2505.07447, it proposes **Extrapolating the estimation**, which shares the same idea as this paper.

### Questions
See weaknesses.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes History-Guided Sampling, a training-free, momentum-style modification to diffusion sampling. It injects a correction term at each step using an EMA of past denoiser predictions. The update is scheduled over time, optionally projected to suppress the parallel component to avoid oversaturation, and high-pass filtered in the DCT domain to prevent color drift. The authors motivate HiGS by viewing Euler sampling as SGD on a time-varying energy and borrowing a STORM-like momentum term, as well as offering an error analysis showing improved local truncation error and thus better global error order for the modified update. The experiments show some improvements of HiGS compared with Stable Diffusion variants, DiT/SiT with low NFE and low CFG. It is notable that the unguided HiGS reports FID 1.61 in 30 steps on ImageNet-256.

### Strengths
1.	The paper introduces a training-free “history” term during sampling (EMA over past predictions) and combines it with two stabilizers—orthogonal projection and DCT high-pass filtering. The combination is simple in this plug-and-play form, and well-motivated by both optimization (STORM-style momentum) and numerical analysis views.
2.	The paper gives two theoretical lenses: (1) Euler sampling ≈ SGD on a time-varying energy, motivating history/momentum; (2) a truncation-error analysis showing that a properly weighted history term can improve the local error from O(h^2) to O(h^3) and thus the global error order. 
3.	The experiments show consistent gains across models including SDXL/SD3/SD3.5, DiT/SiT, tasks, and multi samplers. Quantitative tables show large preference-metric jumps and lower FID/IS steps to match baselines; ImageNet results include a new unguided FID 1.61 in 30 steps on SiT-XL+REPA-E

### Weaknesses
1.	There may be a gap between the clean theory and the implemented recipe. The error-order analysis assumes an ideal history weight $w_k$ and doesn't account for EMA history, time-windowing, orthogonal projection, or DCT filtering used in practice; the proof shows existence of a good $w_k$, not that the EMA-driven schedule approximates it. 
2.	Text-to-image improvements are primarily via HPSv2/ImageReward/Win-rate; CLIP stays flat in Table 5. There is no formal human evaluation or statistical significance tests across prompts or seeds.

### Questions
1.	The error analysis (App. B) shows an existence result for a weight $w_k=\frac{h_k}{2 h_{k-1}}$ that upgrades local error to $O(h_k^3)$. In practice the paper use an EMA history, a scheduled $w_{\text {HiGS }}(t)$, projection, and DCT filtering. Can the author quantify how closely the implemented EMA + schedule approximates the ideal $w_k$ ? A small toy ODE or linearized score model comparison would help.
2.	Section 4.1 frames Euler as SGD on a time-varying energy and connects the history term to STORM style gradient differences. Can the authors provide an empirical plot of $\left\|\nabla E_t\left(z_t\right)-\nabla E_{t-1}\left(z_{t-1}\right)\right\|$ vs. the EMA residual magnitude during sampling to validate the link?
3.	The theorem assumes smoothness and bounded derivatives. Are there empirical signs (e.g., at extreme CFG scales) that these assumptions break and the history term can destabilize?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes History-Guided Sampling (HiGS), a training-free method that improves diffusion model sampling quality by leveraging past model predictions. Inspired by momentum-based variance reduction in optimization (STORM), HiGS computes a weighted average of previous predictions and uses the difference from the current prediction as guidance. The method integrates seamlessly into existing samplers without additional forward passes, requiring only hyperparameter tuning. Extensive experiments across multiple models (SD-XL, SD3, DiT, SiT, Flux) and settings demonstrate consistent improvements, with particularly strong results in low-NFE scenarios. Using SiT-XL+REPA-E, HiGS achieves FID 1.61 on ImageNet 256×256 with just 30 steps (vs. 250 baseline), establishing new state-of-the-art for unguided generation.

### Strengths
1. **Strong practical value**: Training-free and truly plug-and-play (Algorithm 1), addressing a critical need for faster diffusion sampling. No additional forward passes required (Section D confirms identical 6.50 iter/sec with/without HiGS).
2. **Comprehensive experimental validation**: Extensive testing across architectures (U-Net, Transformer), model scales (675M-2.6B parameters), training paradigms (standard, distilled), and tasks (text-to-image, class-conditional). Table 6 demonstrates compatibility with multiple samplers (DDIM, DPM++, PLMS, UniPC).
3. **Consistent improvements**: Win rates > 0.75 across most settings (Table 2), with particularly strong results in low-NFE scenarios (Table 1: SiT-XL+REPA FID 12.08→4.86, -60%). Figure 5 shows benefits across all CFG scales (1.5-5.0) and NFE ranges (10-30 steps).
4. **State-of-the-art achievement**: ImageNet 256×256 unguided FID 1.61 with only 30 steps (Table 3) represents significant advancement in fast, high-quality generation.
5. **Well-motivated methodology**: Clear connection to optimization theory (Eqs. 3-4) and formal error analysis (Theorem B.1). Thorough ablation studies justify design choices (Tables 7-9, Figures 10-15).
6. **Compatibility with existing methods**: Works with distilled models (Table 4), guidance interval (Table 3 baselines), and can combine with APG (Figure 6).

### Weaknesses
1. **Statistical significance not established**: All experiments appear to use single runs without error bars or confidence intervals. Some improvements are small (e.g., Table 1 SD3: 27.19→26.84, only -1.3%) and may not be statistically significant. Multiple seeds and significance tests would strengthen claims, especially for marginal improvements.
2. **Hyperparameter selection guidance insufficient**: Tables 10-12 show different hyperparameters per model (w_HiGS ranges 1.0-2.5, η varies 0-1, t_min 0.3-0.4) without clear selection principles. For practitioners applying HiGS to new models, the paper lacks a practical tuning protocol or recommended defaults. Figures 12-13 show t_min sensitivity (~13% performance variation), contradicting "robust" claims.
3. **Limited theoretical-practical connection**: Theorem B.1 suggests optimal weight w_k = h_k/(2h_{k-1}), but implementation uses EMA with α=0.5-0.75. This gap is not explained. Additionally, orthogonal projection and DCT filtering lack principled justification—they appear to be heuristic solutions found through trial-and-error rather than theoretically motivated.
4. **Incomplete comparison with related work**: No direct comparison with AutoGuidance (Karras et al., 2024), which is conceptually similar but requires training. Appendix C mentions compatibility but provides no quantitative results. Comparison with other training-free fast sampling methods would contextualize the contribution better.
5. **Computational overhead not fully quantified**: While Section D reports identical iteration rates, this is for SD3 at unspecified resolution. DCT/iDCT operations have non-negligible cost at high resolutions (e.g., 1024×1024 or 2048×2048). Wall-clock time measurements across resolutions and batch sizes would be valuable.
6. **Missing failure case analysis**: Section 6 mentions "biases and limitations" but provides no concrete examples. When does HiGS not help or potentially harm? Honest reporting of failure modes would aid practical deployment decisions.

### Questions
1. **Hyperparameter selection**: For practitioners applying HiGS to a new model, can you provide: (a) A simple default configuration that works reasonably well across models? (b) A recommended tuning protocol (e.g., "start with w_HiGS=1.5, α=0.75, then adjust based on visual quality")? (c) Sensitivity analysis showing how much performance degrades with suboptimal choices?
2. **High-resolution overhead**: What is the computational overhead of DCT/iDCT at high resolutions (1024×1024, 2048×2048)? Can you provide wall-clock time comparisons showing actual latency impact in production settings?
3. **AutoGuidance comparison**: How does HiGS compare quantitatively with AutoGuidance (Karras et al., 2024) in terms of quality, speed, and applicability? Could HiGS and AutoGuidance be combined?
4. **Diversity metrics**: Table 1 shows slight Recall decrease for SiT-XL+REPA (0.73→0.70). Can you provide additional diversity metrics (e.g., Vendi score, cluster-based metrics) to quantify diversity preservation? Are the diversity benefits claimed in Figures 7-8 statistically validated?
5. **DCT filtering justification**: Why was DCT chosen specifically for frequency filtering? (a) Have you compared with other methods? (b) Figure 15 shows Rc=0 performs reasonably well—is DCT essential for metric improvements, or primarily for visual quality? (c) What is the quantitative impact of DCT on metrics beyond FID? (d) Why you choose frequency filtering as the solution of “unrealistic color compositions in generations” issue?
6. **Advanced momentum methods**: While STORM provides the initial motivation, recent advances in momentum-based optimization (e.g., Lion, SOAP) suggest promising extensions. Do you see these as promising future directions, and what challenges would they present for the training-free nature of HiGS?

### Soundness
3

### Presentation
3

### Contribution
3
