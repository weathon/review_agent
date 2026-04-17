# Antithetic Noise in Diffusion Models

- Decision: Accept (Poster)
- Scores: 8, 2, 6, 6

## Abstract
We systematically study antithetic initial noise in diffusion models, discovering that pairing each noise sample with its negation consistently produces strong negative correlation. This universal phenomenon holds across datasets, model architectures, conditional and unconditional sampling, and even other generative models such as VAEs and Normalizing Flows. To explain it, we combine experiments and theory and propose a \textit{symmetry conjecture} that the learned score function is approximately affine antisymmetric (odd symmetry up to a constant shift), supported by empirical evidence.
This negative correlation leads to substantially more reliable uncertainty quantification with up to $90\%$ narrower confidence intervals.  We demonstrate these gains on tasks including estimating pixel-wise statistics and evaluating diffusion inverse solvers.  We also provide extensions with randomized quasi-Monte Carlo noise designs for uncertainty quantification, and explore additional applications of the antithetic noise design to improve image editing and generation diversity. Our framework is training-free, model-agnostic, and adds no runtime overhead. Code is available at https://github.com/jjia131/Antithetic-Noise-in-Diffusion-Models-page.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper investigates the effect of sampling noise in generative image models by pairing each noise vector with its negation and analyzing the resulting generated images. The authors find that this antithetic noise approach produces samples that are strongly negatively correlated, revealing an approximate antisymmetry in the learned score function of diffusion models. They verify this symmetry statistically and propose it as a general property of generative models, extending beyond diffusion to VAEs and normalizing flows. Leveraging this finding, they show that antithetic noise can substantially improve uncertainty quantification, yielding tighter and more reliable confidence intervals.

### Strengths
1. The paper is thorough both empirically and theoretically, providing strong support for its claims through extensive experiments and proofs.

2. The results are substantial and appear to be novel, as the proposed method generalizes across different classes of generative models for image synthesis.

3. The proposed antithetic noise technique requires no retraining, making it practical and simple to apply.

4. The paper offers a compelling theoretical explanation for the observed symmetry through the antisymmetry of the score function, which deepens understanding of diffusion model behavior.

### Weaknesses
1. Some of the writing could be improved; for instance, the first sentence of the abstract (“We initiate…”) reads awkwardly.

2. The paper would benefit from an evaluation on OOD data, along with quantitative uncertainty metrics such as AUROC or AUPRC, to better demonstrate the method’s reliability in uncertainty estimation.

3. The work discusses uncertainty improvement but does not clearly distinguish between different types of uncertainty (e.g., aleatoric vs. epistemic), which would help clarify the scope and interpretation of the results.

4. Several relevant works on uncertainty in diffusion models are not cited:

[1] Berry, Lucas, Axel Brando, and David Meger. "Shedding light on large generative networks: Estimating epistemic uncertainty in diffusion models." The 40th Conference on Uncertainty in Artificial Intelligence. 2024.

[2] Berry, Lucas, et al. "Seeing the Unseen: How EMoE Unveils Bias in Text-to-Image Diffusion Models." arXiv preprint arXiv:2505.13273 (2025).

### Questions
1. Did you experiment with partial negation of the noise vector (e.g., negating only the first half or specific components of $z$) to analyze how localized symmetry affects the generated outputs?

2. Can you envision an analogous approach for LLMs, where an equivalent notion of antithetic sampling might reveal symmetry or uncertainty properties in text generation?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates the use of antithetic samples in the context of diffusion models. After presenting observational evidence in the form of pixel-wise correlation statistics of generated images across different datasets and models, the authors formulate the conjecture that the learned score network is an odd function up to a constant shift at every fixed time t.

Motivated by the negative correlation of samples generated from antithetic noise samples, the authors propose using antithetic sampling in diffusion models for various tasks such as uncertainty quantification and inverse problems.

Finally, they show that an antithetic Monte Carlo scheme significantly outperforms classical MC in terms of confidence interval width and overall sample diversity.

Overall, this paper reads as an advanced study of an expected statistical phenomenon underpinned by a well-known variance-reduction technique, rather than a fundamentally novel contribution.

### Strengths
- The paper is clearly presented and easy to follow.
- The underlying principle is simple and elegant and can be implemented easily.
- The conjecture is simple and relevant to the evidence presented.
- The experiments are thorough and convincing in terms of the superiority of antithetic vs independent sampling for uncertainty quantification and posterior sampling.

### Weaknesses
- The novelty feels quite limited. Antithetic sampling is a well-established variance-reduction technique, and the fact that fully negatively correlated samples passed into what is essentially a flow map yield back partially negatively correlated outputs feels like expected behaviour, not really a "universal discovery".

- The theoretical justification of the conjecture in the high-noise regime, although nicely presented, is fairly trivial since it mostly relies on the well-known fact that the score function of a Gaussian distribution is linear.

- The paper gives very little insight into what might be happening close to the target distribution p_0 (i.e., the data manifold). The omission feels significant, as it is clearly the region where getting the score function right matters most. Figure 3 clearly shows that the strong negative correlation breaks down faster as we approach the data distribution.

- Instead of focusing on the conjecture as a pure observational fact, the paper could benefit from either going deeper in the theoretical justification of the conjecture (in the likes of what is done using the FKG inequality in the appendix), or going deeper in the direction of using control variates to improve the efficiency of statistic computation/uncertainty quantification of diffusion models.

### Questions
- The benefits from the conjecture are not stated explicitly, e.g., how could one benefit from this underlying symmetry when designing/training a diffusion model?

- Are there ways to further validate the conjecture in the small-noise regime from a theoretical perspective?

- Could the authors propose hypotheses about the underlying mechanisms (either due to training algorithm/data/architecture) that might lead to the conjectured behaviour?

- Could other control variate schemes be explored?

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper shows that in the diffusion model sampling, pairing each noise sample with its negation induces a strong negative correlation in the generated outputs.  The authors also note that this effect is universal, observed across datasets and architectures, and even in other generative models such as VAEs and normalizing flows. They claim both empirical and theoretical support for a symmetry conjecture that the learned score function exhibits approximate affine antisymmetry. The resulting negative correlation enables substantially improved uncertainty quantification, significantly reduces the uncertainty, suggested by  90% narrower confidence intervals

### Strengths
1. The study of antithetical initial noise in diffusion models appears new to me. 
2. Although the empirical results are not perfectly aligned with the theoretical argument in Lemma 1, the characterization is interesting and could inspire future studies. 
3. The paper is well-written and easy to follow.

### Weaknesses
1. The work would be more complete if the authors could provide some discussion on why the score function admits this affine antisymmetric property.
2. While the work shows that the found fact could result in new methods to do uncertain quantification with significantly narrower CI, as we deal with the generative models, we can potentially generate an infinite number of samples, which potentially drops the significance of the proposed methods. 
3. I am a bit confused about the results in Table 1. It appears that models and datasets are mixed up.

### Questions
Please refer to the weaknesses.

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
2

### Summary
The paper discovers a universal negative correlation in outputs from diffusion models when pairing each initial Gaussian noise  with its negation (“antithetic noise”). This holds across datasets, architectures (U-Net, DiT), unconditional/conditional sampling, DDIM / DDPM, and even VAEs / Normalizing Flows. The authors explain the phenomenon with a symmetry conjecture: the learned score function is approximately affine antisymmetric (odd up to a constant), supported by temporal correlation analyses, 1D slices of score outputs, and theory in the high-noise regime via the OU process. Leveraging the strong negative correlation, they propose antithetic Monte Carlo for uncertainty quantification, yielding up to 90% tighter confidence intervals and large efficiency gains, and show complementary benefits from randomized QMC. Applications include estimating pixel-wise statistics, evaluating diffusion inverse solvers (DPS/DDS), and improving diversity, as well as training-free and model-agnostic editing.

### Strengths
(i) The paper proposes a simple, training-free, model-agnostic procedure with no runtime overhead

(ii) The paper features broad empirical validation across datasets, samplers (DDIM/DDPM), architectures, and even VAEs/flows

(iii) Antithetic Monte Carlo is well motivated and yields substantial, measurable variance reduction

(iv) Beyond the main paper, the work provides good reproducibility details and extensive appendices

### Weaknesses
(i) While theoretical support for the symmetry conjecture is partial and focused on high-noise regimes, the symmetry conjecture remains unproven in generality. The novelty lies mostly in documenting the phenomenon antithetic variance reduction and its strength in diffusion models.

(ii) Many results use pixel-level correlations and simple statistics, while semantic-level uncertainty and quality metrics (e.g., FID, CLIP alignment) are less explored or could be expanded upon more.

(iii) Conditional settings (e.g., strong CFG guidance, complex prompts) and additional methods (e.g., Consistency Models) are not exhaustively studied.

### Questions
In addition to the weaknesses outlined in points (i-iii), I present the following questions for the authors to address:

(1) How does the negative correlation behave under varying classifier-free guidance scales and different guidance strategies?

(2) Can the method improve semantics in text-image alignment uncertainty, and object counts/ positions beyond pixel statistics (a qualitative study would suffice)?

(3) How does antithetic pairing interact with distillation methods (e.g., Consistency Models) and very few-step generation (few-step generation would suffice if not applicable to consistency models)?

(4) While results seem promising, I miss a detailed discussion of limitations. Are there scenarios where antithetic pairing reduces diversity or text adherence?

### Soundness
3

### Presentation
3

### Contribution
3
