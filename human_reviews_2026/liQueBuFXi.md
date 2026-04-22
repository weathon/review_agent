# Transferable and Stealthy Adversarial Attacks on Large Vision-Language Models

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 6, 2, 6

## Abstract
Existing adversarial attacks on Large Vision-Language Models (LVLMs) often struggle with limited transferability to black-box models or produce perceptible artifacts that are easily detected. This paper presents Progressive Semantic Infusion (PSI), a diffusion-based attack that progressively aligns and infuses natural target semantics. To improve transferability, PSI leverages diffusion priors to better align adversarial examples with the natural image distribution and employs progressive alignment to mitigate overfitting on a single fixed surrogate objective. To enhance stealthiness, PSI embeds source-aware cues during denoising to preserve visual fidelity and avoid detectable artifacts. Experiments show that PSI effectively attacks open-source, adversarially trained, and commercial VLMs, including GPT-5 and Grok-4, surpassing existing methods in both transferability and stealthiness. Our findings highlight a critical vulnerability in modern vision-language systems and offer valuable insights towards building more robust and trustworthy multimodal models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes PSI, an  adversarial attack method targeting large Vision-Language Models (VLMs) that achieves strong transferability, naturalness, and stealthiness. To enhance transferability and naturalness, PSI leverages diffusion priors to better align adversarial examples with the natural image distribution, and adopts a progressive alignment strategy to alleviate overfitting to a single fixed surrogate objective. To further improve stealthiness, PSI incorporates source-aware cues during the denoising process, thereby preserving visual fidelity and avoiding easily detectable artifacts. Experimental results demonstrate that PSI produces adversarial examples with superior imperceptibility to humans and reduced detectability by VLMs, outperforming existing attack methods.

### Strengths
1. The progressive alignment objectives in PSI, constructed through co-evolving selection on localized regions, appear to be a novel and interesting contribution.
2. The incorporation of source-aware denoising is also an interesting aspect of PSI, as it helps preserve the visual consistency of the original image while enhancing stealthiness.
3. The experimental results demonstrate promising performance across various VLMs.

### Weaknesses
1. Although the experiments demonstrate that PSI achieves strong transferability, the concern raised in Eq. (3) remains only partially addressed, as the proposed joint objective is still constructed based on the surrogate model  𝐹.
2. The contribution of the source-aware denoising component is not explicitly evaluated or ablated in the experimental section, making it difficult to assess its individual impact.
3. The defense experiments are not comprehensive. The authors are encouraged to consider additional defense techniques, such as DISCO [1], to provide a more complete evaluation of PSI’s robustness.
4. Minor Comment: In line 261, DDIM should be corrected to DDPM.

Reference

 [1] C.-H. Ho and N. Vasconcelos, “DISCO: Adversarial defense with local implicit functions,” in Proc. Adv. Neural Inf. Process. Syst., vol. 35, Jan. 2022, pp. 23818–23837.

### Questions
1. The authors are encouraged to include a theoretical analysis to support the effectiveness of PSI, instead of relying exclusively on experimental validation.
2. The authors are advised to report and compare the computational cost of PSI with other baseline methods, to better illustrate its efficiency and practicality.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes **Progressive Semantic Infusion (PSI)**, a diffusion-based adversarial attack designed to generate adversarial examples that are both **highly transferable** and **visually imperceptible**. By combining diffusion priors, progressive semantic alignment, and source-aware denoising, PSI ensures both naturalness and strong attack capability. Experiments conducted on multiple open-source and commercial VLMs demonstrate that PSI achieves superior performance compared to FOA, M-Attack, and AdvDiffVLM in terms of transferability and stealthiness.

### Strengths
* **Important and realistic problem.**
 The paper addresses one of the most critical and practical challenges in current VLM security — achieving black-box adversarial robustness with both transferability and visual imperceptibility.  
* **Well-motivated and coherent design.**
 The method leverages diffusion priors to maintain naturalness and progressive alignment to enhance cross-model consistency, forming a conceptually clear and technically sound framework.  
* **Comprehensive experiments.**
 The paper includes extensive experiments across multiple datasets and models, with strong baselines, ablation studies, and hyperparameter analyses that substantiate the effectiveness of PSI.

### Weaknesses
* **Unclear boundary of contributions.** The method can be seen as a combination of diffusion-based purification and FOA-like alignment. The novelty is somewhat incremental, and the paper should include a dedicated section clarifying the key theoretical and methodological contributions beyond prior works such as FOA and AdvDiffVLM.  
* **Limited validation of the naturalness term.** The joint objective includes a data distribution term \( p
_D(x) \), which is only indirectly approximated via the diffusion prior. The paper lacks analysis or evidence showing how this term contributes to improved transferability or robustness.  
* **Unjustified choice of scale parameter.** The subregion scale \( s \in [0.4, 0.9] \) is used without ablation or justification. In contrast, FOA and M-Attack adopt \( s \in [0.5, 0.9] \). The rationale for this difference should be explained, ideally supported by quantitative sensitivity analysis.  
* **Insufficient theoretical support for the “degenerate case” statement.** The claim that *“The random cropping techniques used in M-Attack and FOA can be viewed as a degenerated case of the progressive alignment”* is conceptually intuitive but lacks formal theoretical justification or mathematical derivation.

### Questions
1. **Clarify the main contributions.** Add a short subsection explicitly summarizing PSI’s conceptual and theoretical advances over FOA, AdvDiffVLM, and M-Attack.  
2. **Provide empirical or theoretical evidence for the naturalness term.** Analyze how the naturalness loss \( \mathcal{L}_{nat} \) correlates with attack transferability or perceptual quality, supported by visual or statistical comparisons.  
3. **Include a hyperparameter ablation on the scale range.** Report results under different \( s \) intervals (e.g., [0.2, 0.8], [0.5, 0.9]) to justify the chosen values and their influence on ASR and perceptual metrics (LPIPS, BRISQUE).  
4. **Strengthen theoretical grounding.** Provide a formal explanation or appendix derivation showing how random cropping in M-Attack and FOA can be mathematically reduced to a special case of progressive alignment (e.g., when temporal dynamics or semantic references are removed).  
5. **Enhance clarity and reproducibility.** Include detailed pseudocode, hyperparameter tables, and mask-update procedures for local alignment in the appendix or supplementary materials to facilitate reproducibility.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper aims to improve the effectiveness and imperceptibility of adversarial examples against large vision–language models (VLMs) by leveraging diffusion models with progressive alignment. Experiments demonstrate that the proposed method outperforms existing approaches.

### Strengths
1. Enhancing the stealthiness of adversarial attacks is an important and timely research direction.
2. The idea of aligning the distribution between adversarial and natural examples is reasonable and intuitive.
3. Experiments on multiple models validate the effectiveness of the proposed approach.

### Weaknesses
1. The paper appears to focus on targeted attacks, but this is not clearly stated by the authors. This lack of clarity may mislead readers regarding the nature of the adversarial setting.
2. The literature review is limited. Many existing adversarial attack methods are relevant, beyond those specifically developed for VLMs. Their core ideas could also apply to this context. The authors should discuss these works and include comparative results.
3. The motivation behind the Joint Objective (e.g., Eq. (4)) is not well explained. The paper should clarify its intuition and necessity.
4.  The proposed method only considers attacks on the image modality while ignoring the text modality. Since a key characteristic of VLMs is their cross-modal alignment, it is not meaningful to attack or evaluate the model solely through the image modality.
5. The authors propose to select the source model that is similar to the target model, which is not reasonable.

### Questions
Please refer to Weaknesses.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
- This paper proposes PSI, a transferable adversarial attack against large VLMs. 
   - PSI leverages diffusion models to progressively align and infuse natural target semantics during the denoising process. 
- To enhance transferability, diffusion priors are utilized to align adversarial examples more closely with the natural image distribution, while progressive alignment mitigates overfitting to a single surrogate objective. 
- In addition, PSI introduces source-aware cues via DDPM inversion to preserve visual fidelity and avoid perceptible artifacts. 
- Experimental results demonstrate that PSI effectively attacks open-source, adversarially trained, and commercial VLMs, including GPT-5 and Grok-4, achieving superior performance in both transferability and stealthiness compared to existing methods.

### Strengths
- The paper introduces a transferable adversarial attack tailored for large Vision-Language Models (VLMs), revealing a critical semantic-level vulnerability in modern multimodal systems and providing valuable insights toward building more robust and trustworthy models.
- The proposed PSI method is conceptually simple yet effective, achieving superior attack performance across various open-source and commercial models. It consistently outperforms multiple state-of-the-art approaches in both transferability and stealthiness.
- The paper offers a clear discussion of related work and experimental results, with comprehensive implementation details and prompts provided in the appendix, thereby strengthening reproducibility and transparency.

### Weaknesses
- The source-aware denoising module primarily builds upon the existing DDPM inversion technique [1], which is not a novel contribution of this work and therefore should not be highlighted as a core innovation.

> [1] An Edit-Friendly DDPM Noise Space: Inversion and Manipulations, CVPR 2023.

- There is some inconsistency in the experimental results. For example, the reported ASR on Claude-3.5 differs considerably from that in M-Attack, even though both studies use the same perturbation budget $\epsilon=16/255$ and GPTScore threshold (0.3). The paper would benefit from a clearer explanation of these discrepancies to ensure comparability and reproducibility.

- The main methodological difference from prior approaches such as M-Attack, FOA, and AdvDiffVLM lies in the design of a distinct localized alignment strategy combined with a co-evolving selection mechanism within the diffusion framework. While these modifications lead to performance gains, they appear incremental in nature, and the paper provides limited analysis or ablation evidence to justify the effectiveness of each proposed component.

### Questions
- It would be beneficial for the paper to include a clearly structured Threat Model section that explicitly defines the attack assumptions, adversary capabilities, and experimental settings. This addition would help readers better understand the scope and applicability of the proposed method.

- The evaluation of defenses could be further strengthened. Presenting results on a broader range of Large Vision-Language Models (LVLMs) would provide stronger evidence of the generality and robustness of the proposed PSI attack.

- Could the authors clarify what specific modifications or innovations were introduced beyond the original inversion framework? For example, was any new modeling applied to noise injection or update dynamics during inversion?

> An Edit-Friendly DDPM Noise Space: Inversion and Manipulations, CVPR 2023.

- The paper highlights progressive alignment and co-evolving selection as the core innovations of PSI. While an ablation study is provided (Table 3), it is limited to a single model, which may not be sufficient to fully demonstrate the general effectiveness of these components. Could the authors consider expanding the ablation analysis or providing additional quantitative evidence to better validate their impact on transferability and stealthiness?

- The reported ASR on Claude-3.5 is notably higher than in M-Attack, despite using the same perturbation budget $\epsilon=16/255$ and GPTScore threshold (0.3). Were there any differences in prompts, sampling steps, or surrogate model ensembles? How sensitive are your results to these settings?

- PSI leverages diffusion priors to enhance naturalness, but stronger adherence to the natural image manifold might reduce controllability over the target semantics. Did the authors observe such a trade-off, and if so, is there a mechanism in PSI to balance naturalness and semantic fidelity?

- DiffPure or semantic-consistency detection) perform against PSI, and what potential directions exist for developing defense strategies tailored to such diffusion-based attacks?

### Soundness
3

### Presentation
3

### Contribution
3
