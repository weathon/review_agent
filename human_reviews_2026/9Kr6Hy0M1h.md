# Adversarial Advertisement in Text-to-Image Generative Models

- Decision: Reject
- Scores: 2, 6, 4

## Abstract
As text-to-image diffusion models (T2I DMs) gain popularity, there is a growing interest in adversarial advertisement where an attacker can compromise a T2I DM and make it generate images with the implantation of the target product brands, based on users' non-advertising input prompts. However, two challenging problems in adversarial advertisement in T2I DMs remain unsolved: imperceptible adversarial advertisement and robust adversarial advertisement. First, an estimation algorithm of multivariate continuously scaled phase-type with Lévy distribution is designed to understand the intrinsic distribution of natural sentences. By pushing non-advertising prompts to dense regions onto the estimated distribution, the perturbed prompts become indistinguishable from natural prompts with the advertisements. Theoretical analysis is conducted to validate its convergence to the empirical distribution of natural prompts with advertisements. Second, a novel masked parameter smoothing method based on mollification theory is developed to derive a smooth T2I DM with a dimension-invariant certified guarantee for adversarial-advertisement robustness against model fine-tuning in high-dimensional parameter space, while the masked smoothing can reduce the loss of model utility. Theoretical analysis shows that smooth T2I DMs can still yield adversarial advertisements against model fine-tuning within the certified radius.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper studies adversarial advertisement injection in text-to-image diffusion models. It proposes a two-part framework: (1) fitting a Multivariate Continuous Phase-type with Lévy (MCPHL) distribution over embeddings of natural advertisement prompts, and pushing normal prompts toward the high-density region of that distribution to make injected ads more imperceptible; (2) applying a masked mollification method in parameter space, inspired by randomized smoothing, to preserve the adversarial behavior after user fine-tuning. Experiments on several T2I diffusion models show higher attack success rates and some robustness compared with some baseline.

### Strengths
1. The topic, robust and stealthy adversarial advertisement in T2I models, is important but underexplored.
2. Combining heavy-tailed embedding modeling with certified smoothing is conceptually creative, even if both ideas come from existing literature.
3. The empirical section shows that the proposed method maintains attack effectiveness under partial fine-tuning.
4. Mathematical parts (MCPHL and mollification) are technically consistent at the formula level.

### Weaknesses
1. Writing quality is poor: the abstract is fully passive and fails to identify active contributions; the introduction is confusing and internally inconsistent — it first states that “The adversarial-advertising problem in T2I DMs is underexplored. To our knowledge, BAGM (Vice et al., 2024) is the **first work** to inject advertisements without using unusual triggers”, but immediately claims “this work is the **first** to study the adversarial advertisement problem in T2I DMs.” This contradiction blurs the line between background and the paper’s own contribution, leaving the novelty unclear. The “preliminary” section effectively functions as a threat model but is not labeled or structured as such. In addition, related work should not be entirely relegated to the appendix — at minimum, advertisement injection and T2I backdoor literature must be summarized in the main text to clarify how this work differs from prior methods such as BAGM or VillanDiffusion.

2. Novelty is overstated. The paper itself acknowledges that BAGM (Vice et al., 2024) already performs adversarial advertisement injection, being the first to study this problem. The present work mainly improves on imperceptibility and robustness, and would be more accurately described as pursuing a **stronger form of adversarial advertisement** rather than introducing the problem itself. However, the title “Adversarial Advertisement in T2I Models” and parts of the introduction misleadingly frame the work as the first to define this task.

3. Theoretical presentation is unclear. Although an optimization objective is provided (e.g., Eq. 6), it appears only after a long sequence of definitions and is difficult to interpret in relation to the overall method. The paper does not clearly connect this objective to the end-to-end training process—how it interacts with the alignment term and the MCPHL density estimation, and how the later smoothing procedure fits into the workflow. As a result, while the individual equations seem mathematically consistent, the overall reasoning from theory to implementation remains unclear.

4. The analogy between parameter perturbations and input perturbations is only asserted, not demonstrated. The paper borrows certified robustness theory that was originally derived for input-space, but applies it to parameter space without justification. While such an analogy might approximately hold for linear models where parameter and input perturbations are somehow equivalent, it is unclear why it should remain valid for highly nonlinear networks without additional theoretical or empirical support.

### Questions
1. Can the authors provide a single summarized loss function showing how the alignment, density, and smoothing terms combine and which parameters are optimized?

2. Is there any theoretical or empirical evidence that small parameter changes during fine-tuning behave like adversarial input perturbations?

### Soundness
2

### Presentation
1

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
This paper introduces a new adversarial paradigm called Adversarial Advertisement in T2I, which stealthily implants brand-related visual content into images generated by text-to-image diffusion models without altering user prompts. Extensive experiments on different backbones show high advertisement success rates and resistance to model perturbation, outperforming prior backdoor or concept-injection baselines.

### Strengths
1. The paper defines a new and practically significant adversarial advertisement by coverting commercial content into public T2I models.
2. The proposed MCPHL distribution is mathematically sophisticated and supported by convergence and stability proofs.
3. Experiments are extensive across datasets with ablations that validate the contribution of each component.

### Weaknesses
1. The backbone model, Stable Diffusion v1.5, is relatively outdated compared to recent architectures (e.g., SDv3). It remains unclear whether the proposed approach can generalize or maintain its effectiveness on newer, more powerful diffusion models.
2. The proposed method focuses on injecting a single advertisement concept. The scalability and interaction of multiple concurrent ads are not discussed.
3. It is unclear whether the proposed injection mechanism significantly affects the **f**idelity and diversity of the generated images. As illustrated in Figure 7, both the fruit apple and the brand Apple appear multiple times within the same image, and some results (e.g., Figure 7(e)) exhibit noticeable artifacts and lack photorealism.

### Questions
See above

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes an adversarial advertisement attack on text-to-image diffusion models, where ads are imperceptibly embedded into generated images without user intent. It models natural language distributions using a heavy-tailed phase-type Lévy distribution to keep prompts natural, and applies masked parameter smoothing to ensure robustness against fine-tuning. Experiments show the method produces realistic, hard-to-detect advertisements while maintaining model utility and certified robustness.

### Strengths
1.	The motivation and attack scenario are clearly stated appendix A.2, with illustrative comparison figures and examples.
2.	Their method combines heavy-tailed distribution modeling with masked parameter smoothing, enabling natural-looking and robust adversarial advertisements. It maintains image quality while providing theoretical robustness guarantees.
3.	They conduct thorough experiments to prove their adversarial advertisement attack is effective compared with other general attack methods

### Weaknesses
1.	Their approach relies heavily on accurate distribution estimation and high-quality branded text data, limiting generalization in diverse scenarios. Its real-world scalability and defense evaluation are not fully explored. As stated, the appendix A.6 only shows 4 brand implantations examples.
2.	The method fits a complex multi-parameter distribution in high-dimensional embedding space while jointly optimizing encoder parameters, resulting in high computational and tuning overhead. The paper also lacks any complexity or efficiency analysis of their method even they give algorithm 1, leaving scalability and practical applicability unclear. 
3.	Even though they give the examples of generated images by their method, they do not offer the comparisons using the same prompt with other baseline methods to demonstrate their effectiveness.

### Questions
please check the weakness

### Soundness
3

### Presentation
3

### Contribution
3
