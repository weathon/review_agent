# Quantifying the noise sensitivity of the Wasserstein metric for images

- Avg Score: 4.00
- Decision: Reject
- Scores: 8, 0, 4

## Abstract
Wasserstein metrics are increasingly being used as similarity scores for images treated as discrete measures on a grid, yet their behavior under noise remains poorly understood. In this work, we consider the sensitivity of the signed Wasserstein distance with respect to pixel-wise additive noise and derive non-asymptotic upper bounds. Among other results, we prove that the error in the signed 2-Wasserstein distance scales with the square root of the noise standard deviation, whereas the Euclidean norm scales linearly. We present experiments that support our theoretical findings and point to a peculiar phenomenon where increasing the level of noise can decrease the Wasserstein distance. A case study on cryo-electron microscopy images demonstrates that the Wasserstein metric can preserve the geometric structure even when the Euclidean metric fails to do so.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper investigates the sensitivity of the Wasserstein distance to additive pixel-wise noise, providing exact, non-asymptotic bounds and theoretical insights into how noise affects the (signed) 2-Wasserstein distance. The authors show that the error scales with the square root of the noise standard deviation, while the norm scales linearly. They further validate their findings through a range of synthetic and real-data experiments, including a compelling case study on cryo-electron microscopy (cryo-EM) images.
This is a well-executed and thoughtful paper that deepens our understanding of Wasserstein metrics under noise. Its clarity, empirical grounding, and reproducibility make it a strong contribution, even if its focus is somewhat specialized.

### Strengths
- The paper is clearly written and the theoretical contributions (and limitations) are made clear.
- Interesting empirical experiments add and support the theoretical contributions nicely.
- The paper is well-motivated by application since it focusses on applications in cyro-EM, a very active research field.
	-  In particular, Figure 6 clearly outlines the usefulness of Wasserstein distances for cyro-EM
- All necessary code is in the supplementary material and is very accessible. Given that I have to review too many conference submission about new algorithms without code, I really appreciate that the authors took the effort to prepare and attach the code for a theory paper.

### Weaknesses
- Lines 39-47: I do not feel comfortable with some of the statements here given more recent research over the last five years. While Wasserstein GANs (WGANs) and autoencoders (WAEs) are inspired by OT and are important milestones in generative models, a number of papers have shown that the link between OT and these models is rather weak, e.g. (Stanczuk et al., 2021). Wasserstein GANs profit from clipping weights and controlling Lipschitz constants, whereas Wasserstein autoencoders are straghtforward models that make sense without any knowledge about Wasserstein distances. In addition to WGANs and WAEs, I would also consider OT-based flow matching as a very important milestone of OT-based generative models, see (Lipman et al., 2023; Albertgo et., 2023) for general flow matching background and (Tong et al., 2024; Chemseddine et al., 2025; Mousavi-Hosseini et al., 2025 and many more) for OT-based flow matching.

- This might be more of a ‘niche’ topic at ICLR. Nevertheless, this type of research certainly has its place.

- The proofs are interesting and added to my understanding, but switching between the statements in the main document and the proofs in the supplementary material made comprehension more difficult. I think that the supplementary proof section could benefit from repeating the statements, some additional explanations between proofs and one or two proof sketches.

- I found Figure 2 confusing because it is hard to see the differences between the lines. I would advise to consider two (fits vs. theoretical) or three  (W1 vs W2 vs L2)  side-by-side plots (with equalized axes) and to add legends for ‘fit vs. theoretical’. I think that would it make it easier to look at it.

- Similarly, I would add (fits vs. theoretical) legends to Figure 3/7. There are six lines, but one has to read the caption and the legend together to understand the meaning of each line.

- Figure 4: I would advise a larger font for the x-/y-ticks and labels and thicker lines.

- Line 648: ‘proposition 5’ should have capital ‘P’

- $\mathbb{E}m$ looks weird (line 689) – You could probably drop the expectation here, I guess.

### Questions
- My understanding is that all measures/images live on 2-Tours ($G_n$). On the other hand, your noise lives $\mathbb{R}^d$ and during various proof steps you implicitly employ that the measures live on an Euclidean space. While one can certainly extend a measure on a compact domain to the whole $\mathbb{R}^d$ via cyclic conditions, the resulting measure would not necessarily be in $P_2(\mathbb{R}^d)$. Could you please comment on the way this works, respectively, elaborate? 

- “For convenience, we again assume that n is a power-of-two.” Please specify in the main text– do you mean $n = 2^m$? Please explain why this matters in the main text. After looking through the proofs, it becomes more tangible, but it is unclear without consultation of the supplementary material.

- What is the meaning of ’≍’ (line 677, 702)? 

- The authors consider a rather particular noise model that circumvents the ‘rescaling’ problem. While it is a reasonable choice, I would think that it is more intuitive to consider an unbalanced Wasserstein divergence instead. This would also allow the authors to assume that the images $\mu, ¸\nu$ have different masses. Please comment on the decision not to consider unbalanced optimal transport. 

- I am no expert in cryo-EM, but I would be interested in a discussion and maybe an empirical comparison with ‘learned perceptual metrics’, e.g., LPIPS and FID. This would have been a nice addition to the paper. Please comment on the relevance of such neural-network-based metrics.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
This paper provides a comprehensive theoretical and empirical investigation into the noise sensitivity of the Wasserstein metric for images under pixel-wise additive Gaussian noise. The authors derive non-asymptotic upper bounds on the error of the signed $p$-Wasserstein distance between clean and noisy images, showing in particular that the error scales sublinearly (as $\sqrt{\sigma}$) for $W_2$, in contrast to the linear scaling observed with the Euclidean ($L^2$) norm. The work includes rigorous proofs, synthetic experiments, experiments on cryo-EM datasets, and a detailed analysis of the phenomenon in which increasing noise can sometimes decrease the Wasserstein distance, supported by visualizations and quantitative benchmarks.

### Strengths
- The paper rigorously derives upper bounds for the influence of additive Gaussian noise on signed Wasserstein distances, with clear and carefully stated assumptions. The approach is anchored in state-of-the-art multiscale coupling arguments and careful handling of signed measures, extending prior work.
- The experiments validate theoretical scaling laws on standard image datasets and bridge the gap between theory and practice. The authors include detailed ablation studies on noise scaling, metric behavior across image types, and practical benchmarks in high-noise cryo-EM alignment tasks.
- The analysis and illustration of the "decreasing distance under increasing noise" phenomenon (Section 4.4, Figures 7–8) are insightful and highlight subtleties in OT metrics under sparsity and mass-bridging scenarios.

### Weaknesses
- While the paper derives upper bounds for noise sensitivity (notably for $W_2$ and other $W_p$ metrics), the practical tightness of these bounds is not systematically quantified, aside from qualitative observations (e.g., Figure 3 notes the lack of tightness for $W_2$ and $W_3$). A more explicit and systematic investigation—such as plots illustrating the gap between theoretical upper bounds and empirical measurements across parameter regimes, or commentary explaining where and why the bounds become loose—would substantially clarify the practical utility of the theory.

- Although some motivation is drawn from deep learning practice (e.g., references to WGAN and WAE), the paper lacks experimental or conceptual analysis of how the theoretical findings could inform model architectures, loss function design, or optimization routines in modern machine learning. For example, discussing implications for robustness in learned deep representations would strengthen the connection between theory and practice.

- Although the reviewer appreciates the theoretical justification for analyzing effect of noisy image with the signed Wasserstein distance, the paper’s format is incorrect (it lacks the statement “Under review as a conference paper at ICLR 2026” in the header). As a result, the reviewer must assign a score of 0 for formatting compliance.

### Questions
- Can the authors provide a more systematic analysis of the practical tightness of the derived upper bounds for noise sensitivity?

- What are the concrete implications of losing the metric properties (e.g., the triangle inequality) for $W_p^{\pm}$ with $p>1$ in empirical applications? Are there known failure cases or recommended best practices for using the signed $W_2$ in image similarity or learning scenarios?

- Given the growing use of OT-based losses in deep learning, could the authors elaborate on the implications of their findings for architecture or algorithm design—for example, in selecting loss functions for noisy training environments?

- Are there practical guidelines for choosing between $W_1$, $W_2$, or $L^2$ distances in specific image analysis contexts, based on the empirical or theoretical results presented in this work?

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
The paper analyzes how the signed Wasserstein distance $W_p^\pm$ behaves when pixel-level noise is added to images. The authors derive non-asymptotic upper bounds on the deviation between noisy and clean distances. Experiments on synthetic data and cryo-EM projections validate the predicted scaling, and show that $W_2^\pm$ can preserve geometric structure better than $L_2$ in heavy noise. However, because the analysis remains confined to grid-based image histograms and does not engage with robust OT variants already in the literature, I do not see enough new insight to recommend acceptance.

### Strengths
- The authors leverage known dyadic decomposition tools (Weed \& Bach 2019) and adapt them to the signed OT setting, resulting in explicit, dimension-aware bounds that match the empirical scaling trends.
- Experiments are well aligned with the theory: the scaling plots, robustness study across different image families, and cryo-EM case study convincingly illustrate the practical message that $W_2^\pm$ can tolerate higher noise.

### Weaknesses
- All theory assumes images as discrete measures on a regular grid; there is no discussion of other data modalities or continuous supports. Many OT applications (e.g., WGANs) treat an image as part of a higher-dimensional distribution rather than as a histogram, so it is unclear how the analysis transfers.
- No lower bounds or tightness discussion. The results provide upper bounds on $\mathbb{E}\,W_p^\pm$ but do not show matching lower bounds or establish tightness beyond qualitative plots. 
- Signed OT complications. For $p>1$ the signed Wasserstein cost is not a metric and lacks triangle inequality. The text acknowledges this but theorems such as Theorem~5 still leverage triangle-like arguments; more explanation is needed to ensure the statements are well-defined.
- Instability of Wasserstein distances under noise has already motivated partial and unbalanced OT formulations [1--4]; the paper neither positions its analysis relative to this literature nor compares $W_p^\pm$ against robust alternatives in experiments. Without such context it is hard to see the novelty or practical benefits.

References:
[1] Raghvendra, S. et al.. “A New Robust Partial $p$-Wasserstein-Based Metric for Comparing Distributions,” ICML 2024. 
[2] Benamou, J.-D., et al “Iterative Bregman Projections for Regularized Transportation Problems,” SIAM Journal on Scientific Computing 2015. 
[3] Chizat, S. et al. “Scaling Algorithms for Unbalanced Optimal Transport Problems,” Mathematics of Computation 2018. 
[4] Chapel et al. "Partial Optimal Transport with Applications on Positive-Unlabeled Learning" NeurIPS 2020

### Questions
- Can you elaborate on why you inject additive zero-mean Gaussian noise that allows pixels to become negative? In practical pipelines one might clip or renormalize intensities—would your analysis or conclusions still hold under such nonnegative noise models, and what phenomena are lost if we do so?
- Theorem 5 set the optimal cost $W_p(\mu,\nu)$ with the bound terms. Could you provide a corollary specialized to the case $\mu=\nu$?
- For the alignment study, do you estimate the distances on raw values or after histogram equalization / normalization? 
- Is it possible to extend the analysis to entropic OT distances (regularized Sinkhorn costs)? 
- Can you provide empirical comparisons between $W_p^\pm$ and robust alternatives such as the robust partial $p$-Wasserstein metric [1] or unbalanced OT variants [3]?

### Soundness
2

### Presentation
2

### Contribution
2
