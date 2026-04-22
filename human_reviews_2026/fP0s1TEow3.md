# Stage-wise Dynamics of Classifier-Free Guidance in Diffusion Models

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 8, 4, 4, 4

## Abstract
Classifier-Free Guidance (CFG) is widely used to improve conditional fidelity in diffusion models, but its impact on sampling dynamics remains poorly understood. Prior studies, often restricted to unimodal conditional  distributions or simplified cases, provide only a partial picture.
We analyze CFG under multimodal conditionals and show that the sampling process unfolds in three successive stages. In the Direction Shift stage, guidance accelerates movement toward the weighted mean, introducing initialization bias and norm growth. In the Mode Separation stage, local dynamics remain largely neutral, but the inherited bias suppresses weaker modes, reducing global diversity. In the Concentration stage, guidance amplifies within-mode contraction, diminishing fine-grained variability.
This unified view explains a widely observed phenomenon: stronger guidance improves semantic alignment but inevitably reduces diversity. Experiments support these predictions, showing that early strong guidance erodes global diversity, while late strong guidance suppresses fine-grained variation. Moreover, our theory naturally suggests a time-varying guidance schedule, and empirical results confirm that it consistently improves both quality and diversity.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The main contribution of this paper is the description of three distinct phases in the application of Classifier-Free Guidance (CFG). First, in a "strong noise" phase, CFG causes an *acceleration and direction shift*, pushing the distribution toward the global, class-weighted mean of the dataset. This essentially acts as an amplification process that inflates the norms of the trajectories. The authors claim this creates an initialization effect that biases the process toward lower-diversity samples.

The second phase is *mode separation*, where trajectories begin fanning out toward the different modes in the data distribution as noise decreases, settling into their basins of attraction. The issue here, as the authors argue, is that the initial stage has already biased the results, positioning trajectories to favor certain modes over others.

Finally, the third phase is *concentration*, which occurs in the low-noise regime. Here, the dynamics are dominated by contraction, which suppresses intra-class variability. This results in samples that are sharper but less diverse.

Additional contributions of the paper include empirical validation of the theory and a time-varying guidance protocol for diffusion sampling.

### Strengths
Overall, my impression of the paper is quite positive.

* The paper is well-written and well-presented. The illustrations, especially in Figure 1, are very helpful for understanding the different stages the authors discuss.
* A significant contribution of this work is that it lends some theoretical muscle to analyzing CFG. CFG is a technique we all use, but it's essentially a hack that works without a tremendous amount of theory to motivate it. It is valuable to see an analysis from a dynamical perspective, especially as the field moves more toward ODE representations of the diffusion process. This is a genuine contribution to the literature and provides food for thought for additional theoretical development.
* The theoretical analysis itself, which models the data as a Gaussian mixture, is general enough to provide useful insight, seems sound, and provides a good foundation for the claims. However, see the caveats introduced below.

### Weaknesses
* I occasionally struggled to keep track of some of the theoretical results in the theorems. In some cases, the meaning or interpretation of certain variables wasn't entirely clear. For instance, in Proposition 3.4, a parameter $k$ is introduced that determines a radius. It's later stated that $k$ increases with the CFG weight $\omega$ (line 273), but it was not completely clear to me why that is the case. It was also not clear to me what "mild assumptions" were being made in some of the theorems. To the extent possible without introducing too much clutter, making the theoretical statements more self-contained would be helpful.
* In the experimental results, the authors' proposed method is referred to as "TV." While it can be inferred that TV means “time-varying,” the paper does not explicitly define what "TV" stands for immediately before abbreviating it (unless I missed it). So when looking at the experimental tables, I wasn't sure at first if the "TV" results belonged to the authors or someone else.
* The authors refer to the difference between "global diversity," which is negatively affected by strong early guidance, and "fine-grained diversity," which is reduced by strong late guidance. While I have an intuitive sense of what the authors mean by this terminology, I would like to see a more precise definition of these two concepts.
* The paper would benefit from discussing its findings in the context of some other relevant work:
1) The CADS method [1] uses a condition-annealing strategy during inference, where noise is added to the conditioning signal early on in sampling and is then gradually reduced to reveal an unperturbed condition. It would be interesting to see the authors reconcile their theoretical results with that technique. I would also refer to the "fine-grained detail" argument the authors make to see how it reconciles with the results reported in the CADS paper.
2) Two recent papers [2, 3] describe the *biphasic* nature of diffusion sampling, which are distinguished by the relative influence of the conditioning signal. It would be interesting to have the authors discuss their "triphasic" findings in the context of this other work.
* While I think the authors' modeling choices allow for insight into the more general conditional diffusion sampling problem, the theoretical results still rely on what is essentially a toy setting. To the authors' credit, their empirical results operate on real in-the-wild diffusion models. However, a more frank acknowledgement and discussion of the limitations of the theoretical results is needed.
* I noticed very few (but nonzero) typos in the paper (e.g. an extra period in the caption for Figure 2, using *assumption* where *assumptions* is intended in one theorem statement, etc.). One more reading pass, especially from a fresh-eyed reader, should help eliminate them. 

[1] Sadat et al. (2023). CADS: Unleashing the Diversity of Diffusion Models through Condition-Annealed Sampling

[2] Balaji et al. (2022). eDiff-I: Text-to-Image Diffusion Models with an Ensemble of Expert Denoisers

[3] Liu et al. (2024). Faster Diffusion: Rethinking the Role of Temporal Attention in Diffusion Models

### Questions
* Given the use of $\omega > 1$ in Theorem 3.2, am I correct in the interpretation that trajectories are directed toward the class-weighted mean scaled *exactly* by the CFG scale? If that's the case, it seems like a result that should be more explicitly called out, especially given the large CFG scales often used in practice. (And if that's not the conclusion here, it would be a confusing reuse of notation.)
* The authors’ theoretical analysis relies on the countable, discrete components of a GMM. This setting is well-suited to class-conditional models but less obviously applies to uncountable, continuous conditions (e.g. CLIP embeddings). What is the authors' perspective on how to extend their analysis to the continuous-condition domain?
* Please also address the issues described in the "Weaknesses" section.

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
3

### Summary
This paper presents a theoretical analysis of Classifier-Free Guidance (CFG) in diffusion models under multimodal conditional distributions. The authors identify three distinct stages in the sampling dynamics: Direction Shift (early), Mode Separation (middle), and Concentration (late), and explain how CFG leads to diversity loss through these stages. They validate their theory empirically and propose a time-varying guidance schedule.

### Strengths
1.  The paper provides a systematic analysis of CFG under multimodal conditional distributions, moving beyond restrictive unimodal assumptions prevalent in prior work.

2. The paper successfully explains the long-standing empirical observation that strong CFG improves semantic alignment but reduces diversity.

3. The experiments on Stable Diffusion v3.5 with COCO validation set appropriately test the theoretical assessments.

### Weaknesses
1.  In Related Work, the treatment of closely related concurrent papers, say, both the  theoretical or algorithmic perspectives on conditional sampling and CFG in diffusion is incomplete. 
    
2. While experiments are run on a modern diffusion model and COCO captions, the diversity of datasets and generative tasks is limited. For example, all primary experiments use a single text-to-image model/configuration (Stable Diffusion v3.5 with 5,000 COCO images).  Discussion on how these findings generalize to other modalities (e.g., audio, video) or other conditional types (e.g., class or style transfer) are missing. 
    
3. Although time-varying and interval guidance competitors are implemented (and TV-CFG is compared), the 'interval' baseline is only briefly described, and no discussion is given to recently proposed, possibly stronger or more customized adaptive schedules. Moreover, the ablation or sensitivity analysis on schedule shapes, peak timings, or trade-offs beyond the one setting is insufficient. 
    
4. The central theoretical perspective is attributed as the 'first systematic analysis' of CFG under multimodal conditionals (see Abstract, Page 2, and Conclusion). However, several recent works—some cited, some missing[1][2][3], have begun to analyze guidance dynamics in GMMs or with alternative assumptions (Li et al. ICML25/ arxiv 2025a,  Jin et al. ICML25, Wu et al. ICML24). The authors should more carefully disentangle which aspects are genuinely new (e.g., the particular stage-wise decomposition or explicit formalism) as opposed to extended or sharpened results over antecedents. 
    
5. The theory and results emphasize the effect of guidance at various sampling budgets (notably for low NFE), yet the paper lacks comprehensive ablation over NFE and guidance range. Only a select number of NFE and $\omega$ values are tested, how does the approach performs at extremely low (NFE < 5) or much higher NFE values?
    

[1] W. Deng et al., *Reflected Schrödinger Bridge for Constrained Generative Modeling. 2024.* 

[2] X. R. Li. et al., *A Simple Approach to Unifying Diffusion-based Conditional Generation. 2024.* 

[3]  G. M. Favero, et al., *Conditional Diffusion Models are Medical Image Classifiers that Provide Explainability and Uncertainty for Free. 2025.*

### Questions
refer to weaknesses.

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
This paper theoretically analyzes the effect of CFG on sampling dynamics and shows that its influence can be divided into three stages. In the Direction Shift stage (early in sampling), guidance pushes samples toward a weighted average of class means, potentially biasing the initialization toward specific generations. In the Mode Separation stage, samples are driven toward stronger modes, reducing the diversity of generated examples. In the Concentration stage, strong guidance further decreases diversity while improving alignment. Overall, the authors conclude that early strong guidance reduces global diversity, while late strong guidance suppresses fine-grained variations. This motivates the use of a time-varying guidance schedule (TV-CFG) that peaks in the middle, and the experiments demonstrate that TV-CFG improves the quality–diversity trade-off compared to standard CFG.

### Strengths
* The paper studies the influence of guidance on sampling dynamics, which is an important aspect of modern diffusion models. Given the widespread use of CFG, further intuition in this domain is essential to better understand and improve its behavior. The work aims to bridge the gap between current empirical practices using CFG and the theoretical understanding of its underlying mechanisms.

* The authors validate their findings not only on toy datasets but also on real-world text-to-image models.

* The paper is well-written and easy to follow. The examples effectively illustrate each aspect of the discussion.

### Weaknesses
In my opinion, the main weakness of the paper is that its findings may not fully justify the novelty claim. Specifically, [1] studied the dynamics of CFG in Gaussian mixture models, showing that strong guidance improves classifier confidence (i.e., better alignment) while reducing the final entropy of the generated distribution (i.e., lower diversity). They also analyzed the influence of guidance across different stages of sampling (see Section 6 and Proposition 6.1). However, this reference is described by the authors as focusing only on unimodal distributions and “overlooking the multimodal nature of real-world tasks” (L42).

Moreover, the benefits of time-varying guidance scales have already been explored in prior work from a practical standpoint, as acknowledged by the authors. Therefore, I believe the novelty claims require stronger justification to warrant a separate publication on this topic. While I appreciate the idea and analysis presented, the noted similarities with existing work affect my judgment unless the authors provide clearer differentiation. 

[1] Wu Y, Chen M, Li Z, Wang M, Wei Y. Theoretical insights for diffusion guidance: A case study for Gaussian mixture models. arXiv preprint arXiv:2403.01639. 2024 Mar 3.

### Questions
1. Have the authors considered extending their analysis to other guidance methods, such as autoguidance [1]?

2. How would the analysis differ if an SDE-based sampler were used instead of the ODE solver?

I am open to reconsidering my rating after the discussion with the authors.

[1] Karras T, Aittala M, Kynkäänniemi T, Lehtinen J, Aila T, Laine S. Guiding a diffusion model with a bad version of itself. Advances in Neural Information Processing Systems. 2024 Dec 16;37:52996-3021.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents a stage-wise theoretical analysis of Classifier-Free Guidance (CFG) in diffusion models under multimodal conditionals, identifying three distinct phases: Acceleration and Direction Shift (early), Intra-class Mode Separation (mid), and Concentration (late). It explains how CFG causes diversity loss via early-stage bias toward dominant modes and late-stage intra-mode contraction. The authors propose a time-varying guidance schedule that reduces guidance strength in early/late stages while emphasizing the mid stage, showing empirical improvements in diversity without sacrificing fidelity.

### Strengths
1.	The paper provides a principled theoretical foundation for time-varying guidance schedules—specifically, using low guidance strength in the early and late stages of sampling while applying stronger guidance during the intermediate phase—to simultaneously preserve diversity and maintain semantic fidelity.
2.	It reveals when and how Classifier-Free Guidance compromises sample diversity by analyzing the interplay between multimodal conditional structure and temporal dynamics: early-stage bias toward the global mean suppresses weaker modes before they can emerge, while late-stage over-contraction erodes intra-mode variation. This offers a unified explanation for the well-known trade-off between alignment and diversity under CFG.

### Weaknesses
1. The proposed guidance schedule—weak guidance in early/late stages and strong guidance in the middle—is not new. Prior work such as β-CFG[1] has already identified a similar three-stage behavior of CFG and explicitly designed non-uniform guidance schedules based on this insight.
2. The experimental comparison omits key adaptive guidance methods that also modulate ω over time. In particular, the paper should include comparisons against:Linear ramp-up followed by ramp-down schedules, β-CFG or other non-linear schedules.
3. The paper relies heavily on ImageReward and saturation as proxies for diversity. However, ImageReward primarily measures aesthetic or semantic alignment rather than sample variance under the same conditioning, and lower saturation does not necessarily correlate with higher diversity (e.g., outputs could be desaturated yet nearly identical). The absence of explicit, widely accepted diversity metrics—such as LPIPS variance across multiple generations per prompt—weakens the empirical support for the claim that the proposed method better preserves diversity.
4. Although the paper motivates TV-CFG through an asymmetric three-stage dynamic—comprising early direction shift, mid-stage mode separation, and late-stage concentration—the actual guidance schedule is symmetric with respect to the sampling step index and peaks precisely at the midpoint. The paper lacks ablation studies on the sensitivity of performance to the peak location (e.g., shifting the maximum guidance strength earlier or later).

[1] Malarz D, Kasymov A, Zięba M, et al. Classifier-free Guidance with Adaptive Scaling[J]. arXiv preprint arXiv:2502.10574, 2025.

### Questions
Address the weakness, especially the novelty issue.

### Soundness
3

### Presentation
2

### Contribution
2
