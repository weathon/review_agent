# Importance Sampling for Multi-Negative Multimodal Direct Preference Optimization

- Decision: Accept (Poster)
- Scores: 6, 6, 4

## Abstract
Direct Preference Optimization (DPO) has recently been extended from text-only models to vision-language models. However, existing methods rely on oversimplified pairwise comparisons, generating a single negative image via basic perturbations or similarity-based retrieval, which fail to capture the complex nature of multimodal preferences, inducing optimization bias and hallucinations. To address this issue, we propose MISP-DPO, the first framework to incorporate \emph{multiple}, semantically \emph{diverse} negative images in multimodal DPO via the Plackett-Luce model. Our method embeds prompts and candidate images in CLIP (Contrastive Language–Image Pre-training) space and applies a sparse autoencoder to uncover semantic deviations into interpretable factors. Negative samples are selected based on reconstruction difficulty, semantic deviation from the positive, and mutual diversity, yielding broader and more informative supervision. To handle multi-negative comparisons, we adopt a Plackett–Luce objective and introduce an importance sampling strategy that improves training efficiency. Experiments across five diverse benchmarks demonstrate that MISP-DPO consistently improves multimodal alignment over prior methods, validating the effectiveness of semantic-aware, multi-negative sampling in preference-based learning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents MISP-DPO, a framework for multimodal Direct Preference Optimization that uses multiple semantically diverse negative samples instead of a single one. It combines a Plackett–Luce ranking objective with importance sampling guided by a Sparse Autoencoder trained in CLIP space to select informative negatives. Experiments on several benchmarks show improved multimodal alignment and reduced hallucination compared to existing DPO methods.

### Strengths
The paper clearly identifies a weakness in current multimodal DPO frameworks—the oversimplified single-negative setup—and proposes a principled multi-negative formulation to address it.

Experimental evaluation is extensive, including comparisons across multiple models and benchmarks, with consistent quantitative gains in hallucination reduction.

### Weaknesses
The novelty may be moderate: it mainly leverages existing models to extract multiple negative samples, without introducing substantial theoretical or methodological innovation.

The paper does not deeply analyze computational overhead or training stability when incorporating multiple negatives, which could affect scalability for larger datasets.

It remains unclear whether the improvements generalize beyond hallucination-oriented tasks (e.g., to reasoning or instruction following).

### Questions
1. How does the proposed multi-negative sampling strategy affect training efficiency and scalability when applied to larger datasets?

2. Could the authors provide a more detailed analysis of computational overhead introduced by the sparse autoencoder and importance sampling modules?

3. Beyond hallucination reduction, has the method been evaluated on reasoning or instruction-following tasks to assess generalization across multimodal objectives?

I will adjust my score based on the authors’ response.

### Soundness
2

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
This work introduces MISP-DPO, a novel framework to leverage a sparse auto encoder to identify diverse negative images for multi-negative preference optimization, building on multimodal DPO.
The authors demonstrate the efficacy of their method on multiple multimodal models across a range of benchmarks.

### Strengths
* The authors report non-trivial improvements over respective baselines across benchmarks and models. They re-implement DPO, mDPO, and CHiP for a direct and fair comparison (matching data and base models).
* The main contributions are the introduction of multi-negative preference optimization and the method proposed for selecting counterfactual images, building on CLIP retrieval, a sparse autoencoder, and a greedy algorithm to achieve diverse negatives.

### Weaknesses
* Section 4.2, describing one of the main contributions of the work, is perhaps a bit limited in detail. For example, the training (data, recipe) for the SAE is not described. And while the math presented in the negative selection may be sufficient, some discussion behind the intuition of sampling for reconstruction error and activation may make the paper more accessible, particularly to casual readers less familiar with using SAEs for interpretability.
* Another recent work [S-VCO] also argues for negative images that are substantially similar to the request image under alignment. In this work, the authors acknowledge this work and argue that this method is expensive (as [S-VCO] relies on image generation method to generate counterfactuals) while the proposed method is more efficient. However, the respective efficacy is not further discussed. An ablation comparing the retrieval + SAE based approach directly to the generative approach proposed by [S-VCO] would further enhance the contributions of this paper. ([S-VCO]’s data (MVC) appears to have been made available.)
* In table 2, the caption implies that the main difference is how negative samples are chosen, but another difference seems to be the number of negative examples being used as per the description in the text. Perhaps this could be clarified?
* For the ablations in table 2, “diffusion” and “crop+diffusion” have two or one negative images still selected by the proposed method as described in the text. I believe this may make comparison a bit more difficult? I understand that multiple negatives based on diffusion or cropping may not have enough diversity, but if that is the concern, perhaps an ablation with 1 negative sample for all methods could be made fairly, further separating the improvements achieved from the targeted selection method from the multi-negative proposal?
* Minor notes for table 2: typo “mdpo” (instead of “mDPO”); Missing average improvement as in table 1 for easier comparisons.


[S-VCO] Wu, Shengguang, et al. "Symmetrical visual contrastive optimization: Aligning vision-language models with minimal contrastive images." arXiv preprint arXiv:2502.13928 (2025).

### Questions
* Considering that $d_i$ is dependent on $m_p$, $x$ and $m_n$, how scalable is the retrieval of negatives at training time, if one assumes potentially scaling up the distractor pool and the data used for alignment?
* In the reproduction of mDPO, section 5.4 mentions “mDPO, which relies on a single diffusion-generated negative”. But mDPO constructs the negative image via random cropping (0-20%). Is this a typo?
* It is not clear to me why selecting more than 3 negatives would be detrimental to performance as presented in figure 2 and briefly discussed in 5.4. The authors propose this may be “due to noise introduced by redundant or low-quality samples”, but then redundancy may be directly addressed through the diversity-promoting selection and COCO may not have substantial amounts of “low-quality samples”?
* The reported numbers for MMHalBench for at least LLaVA 1.5 7B seem surprisingly strong, even for reported baselines? Earlier works such as [MDPO] has baseline LLaVA 1.5 7B at 2.19 (in this paper: 2.78) and with their method they achieve “only” 2.39, whereas the “mDPO” reproduction in this paper reports 2.99. Are the evaluation protocols comparable?


[MDPO] Wang, Fei, et al. "mdpo: Conditional preference optimization for multimodal large language models." arXiv preprint arXiv:2406.11839 (2024).

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes MISP-DPO, a listwise preference-optimization framework for multimodal LLMs that combines (i) a Plackett–Luce (PL) objective over a *winner + multiple negative* candidates, (ii) importance sampling with a learned proposal (q_\phi) to reduce training cost and variance, and (iii) a semantically diverse negative-construction pipeline built in CLIP embedding space using sparse autoencoders (SAE) and feature-level perturbations. The method is applied to both *visual-preference* (image conditioned) and *textual-preference* (response conditioned) settings. Experiments report consistent gains over DPO/mDPO/CHiP-style baselines on datasets such as MMHal-Bench, HallusionBench.

### Strengths
- **Clear problem framing & practical motivation.** The paper demonstrates clear problem formulation and presentation. The PL-based listwise loss directly optimizes rankings rather than isolated pairs. 
- **Good objective with scalable estimation.** Using **importance sampling** and a learned proposal (q_\phi) to approximate the PL gradient is a sound strategy to keep many-negative training tractable while emphasizing informative (“hard”) negatives. 
- **Semantically diverse negatives grounded in CLIP space.** The SAE-based feature editing and “mix-and-match” construction pipeline plausibly increases negative diversity without requiring extra human labels. 
- **Broad evaluation.** The method is tested across multiple hallucination/factuality benchmarks frequently used for MLLMs (MMHal-Bench, HallusionBench, POPE, WildVision, MMVP) and with a modern evaluation toolkit (VLMEvalKit).

### Weaknesses
- **Novelty concern.** Beyond pairwise (1 chosen, 1 rejected) multimodal DPO, there already exists listwise optimization in the context of multimodal DPO, such as LPOI[1]. Therefore, there may be certain novelty concern, **especially when the authors claim** their paper is 

  - "the **first** framework to incorporate *multiple*, semantically *diverse* negative images in multimodal DPO" (line 16, in abstract)
  - "the **first** framework to incorporate multi-negative supervision into multimodal DPO" (line 89), 
  - "However, such techniques **remain underexplored** in vision-language models" (line 117). 

  **So please double-check your claims in the submission.** Also, the use of Plackett–Luce objective is also not really a novelty for listwise DPO, as there are already some prior work such as PLPO[2].

- **Insufficient experiments against former baselines.** As there are already many works published on multimodal DPO, it's not enough to just incorporate mDPO and CHiP as baselines apart from Random and basic DPO in experiments. Please at least incorporate and run experiments for latest methods OPA-DPO[3] and SymMPO[4]. Any more baselines are also welcome. 

- **Estimator properties insufficiently analyzed.** The text would benefit from formal statements or empirical diagnostics of *bias/variance* under finite negative sampling, any **weight clipping** or self-normalization, and the stability of (q_\phi) training (e.g., divergence from target leading to high-variance importance weights). (I did not see explicit guarantees/ablation in the provided pages.) 

- **Ablation depth.** While the framework has several moving parts (PL listwise loss, IS with (q_\phi), SAE-based negatives, textual-preference branch), the paper would benefit from *systematic ablations* that isolate each contribution and report uncertainty (std/CI over seeds).

References:

[1] Fatemeh Pesaran zadeh et al. "LPOI: Listwise Preference Optimization for Vision Language Models" In ACL 2025 Main Conference.

[2] "Plackett–Luce Preference Optimization (PLPO): Listwise Ranking for Preference Optimization" Preprint 2024.

[3] Yang et al. "Mitigating Hallucinations in Large Vision-Language Models via DPO: On-Policy Data Hold the Key" In CVPR 2025.

[4] Liu et al. "Mitigating Hallucination Through Theory-Consistent Symmetric Multimodal Preference Optimization" In NeurIPS 2025.

### Questions
1. **Cost accounting:** Please add wall-clock/GPU-hour comparisons vs. DPO, mDPO, CHiP to demonstrate the promised efficiency gains of importance sampling.
2. **Negative construction controls:** How do you ensure that CLIP-SAE-driven negatives are not trivially separable (e.g., distributional artifacts), and that they **stress visual grounding** rather than language priors? Any human spot-checks? 
3. **Unbiasedness & variance:** Is the IS gradient strictly unbiased under your training scheme? Do you apply **weight clipping** or **self-normalized IS**? Please report effective sample sizes or variance diagnostics across training.

### Soundness
2

### Presentation
3

### Contribution
2
