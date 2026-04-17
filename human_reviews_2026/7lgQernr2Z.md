# Unleashing Guidance Without Classifiers for Human-Object Interaction Animation

- Decision: Accept (Poster)
- Scores: 2, 6, 4, 6

## Abstract
Generating realistic human-object interaction (HOI) animations remains challenging because it requires jointly modeling dynamic human actions and diverse object geometries. Prior diffusion-based approaches often rely on handcrafted contact priors or human-imposed kinematic constraints to improve contact quality. We propose a data-driven alternative in which guidance emerges from the denoising pace itself, reducing dependence on manually designed priors. Building on diffusion forcing, we factor the representation into modality-specific components and assign individualized noise levels with asynchronous denoising schedules. In this paradigm, cleaner components guide noisier ones through cross-attention, yielding guidance without auxiliary classifiers. We find that this data-driven guidance is inherently contact-aware, and can be further enhanced when training is augmented with a broad spectrum of synthetic object geometries, encouraging invariance of contact semantics to geometric diversity. Extensive experiments show that pace-induced guidance more effectively mirrors the benefits of contact priors than conventional classifier-free guidance, while achieving higher contact fidelity, more realistic HOI generation, and stronger generalization to unseen objects and tasks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a new method for text-driven HOI generation. The main technical contributions are two fold. First, it porposes a new guidance mechanism where asynchronous denoising induces guidance without using an external classifier/objective function. Second, a contact-aware shape-spectrum data augmentation strategy is proposed that preserves contact semantics while varying object geometry. Experimental results on InterAct, BEHAVE, and OMOMO are reported.

### Strengths
1. The proposed data augmentation technique is promising. Generalization of the interaction semantics to similar objects is a fundamental challenge, which is not only useful for the HOI generation task studied in this paper, but may also be useful say other domains, say robotics.

2. State-of-the-art results are reported in this paper.

### Weaknesses
1. While the autors have explained **how** the proposed asynchronous denoising of different parts works for HOI generation, they didn't explain **why** it works. In a high-level sense, it works in a way analogous to CFG, but it lacks detailed analysis and investigation. Especially considering the proposed approach is inspired by (or connected to) the duffions forcing mechanism, which is a generic framework (not tailored for HOI generation), showing **insights** why it is useful for HOI generation is critical as future work may better understand when it works and when it may not.

2. The contact-aware shape-spectrum augmentation is a big contribution of this paper. But it lacks sufficient details of how it works (e.g., statistics of number of objects before and after augmentation) and no visual examples are shown in the main paper. Only limited illustrations are provided in the appendix and supplementary video. It makes readers hard to gauge the effectiveness of this part and reproduce it.

### Questions
1. $\textbf{x}_S$ is not defined in the paper. Could you please explain it?

2. In Eq. (4), for $\textbf{x}_S'$, it it a concatenation of $\textbf{x}_U^{m_1}$ and $\textbf{x}_S^{M_2}$? By the way, the symbol of $\textbf{x}$ in $\textbf{x}_S'$ is different from that in $\textbf{x}_U^{m_1}$ and $\textbf{x}_S^{M_2}$ in the paper.

3. Is ithe pace-induced guidance used in the inference only?

### Soundness
3

### Presentation
2

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
1. This paper presents LIGHT, an asynchronous-denoising–based guidance mechanism that achieves soft, flexible conditioning without relying on external classifiers.

2. In addition, the paper introduces contact-aware shape-spectrum augmentation, which maintains contact semantics while altering object geometry, enhancing robustness and generalization.

3. Extensive experiments further verify the approach, showing consistent improvements over existing baselines and enabling the generation of vivid, realistic interactions.

### Strengths
1. It proposes a pace-induced guidance mechanism to generate more realistic and plausible human-object interactions.

2. Through extensive experiments, it analyzes the effects of pace-induced guidance, token separation, augmented data, guidance intensity and denoising lag, and guidance direction.

3. The authors also provide fair comparisons by re-implementing and modifying prior baselines (e.g., InterDiff, Text2HOI).

### Weaknesses
1. There is no reference to Figure 5 in the section Impact of Guidance Intensity and Denoising Lagging. Please explicitly link the analysis to the figure.

2. In the Impact of Guidance Intensity and Denoising Lagging experiment, the paper states that the best value of δ is 300, but Figure 5 seems to suggest that 200 performs best. Could the authors clarify which value is correct?

3. I am not fully clear about the settings in Table 2. For the case where hand-body separation is ✓ and human-object separation is -, is the object token concatenated with the hand stream or with the body stream? In other words, which grouping is correct: {b, ho} or {bo, h}?

4. Additionally, if there are no separator tokens, meaning all tokens are combined, then it seems that the staged schedule cannot be applied. Could the authors clarify how the model behaves in this case?

### Questions
1. Will the authors release the code and the augmented data used for the Contact-Aware Shape-Spectrum Augmentation?

2. In Section 4, how do you compute x′_S from x^{m1}_U and x^{m2}_S? Is it a simple concatenation or another operation?

3. In the supplementary video at timestamp 1:06, the motions for InterDiff and Text2HOI appear very similar to each other but different from the proposed method. Were different text prompts used for these baseline results?
Please check it.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes a novel classifier-free guidance framework for diffusion-based HOI animation. Unlike prior methods relying on external contact classifiers or kinematic constraints, LIGHT achieves guidance through asynchronous denoising schedules. The cleaner (less noisy) modalities guide noisier ones, producing contact-aware behavior. Additionally, the paper introduces contact-aware shape-spectrum augmentation using ShapeNet and Objaverse objects to improve geometric generalization. Extensive experiments on the InterAct, BEHAVE, and OMOMO datasets demonstrate that LIGHT outperforms baselines such as HOI-Diff, InterDiff, CHOIS, and Text2HOI in FID, contact quality, and text-motion alignment, while maintaining realism and generalization across unseen object categories

### Strengths
- The idea of pace-induced guidance (asynchronous denoising between modalities) is innovative, extending diffusion forcing into a practical HOI setting
- The quantitative results are comprehensive, with thorough comparisons against existing baselines and well-conducted ablation studies.

### Weaknesses
- From the final visual results, the proposed strategy indeed demonstrates the ability to effectively leverage human priors to generate plausible interaction poses. However, the method still struggles with fine-grained contact modeling, and noticeable artifacts remain at the contact level.
 - The evaluation metrics seem somewhat questionable — the R-Precision scores are all within a similar range, and several other metrics also show minor differences across methods. It is unclear whether these metrics are sensitive enough to effectively distinguish the quality of human–object interactions.
- Since the video format is the most effective medium for evaluating the quality of human–object interactions, here the visual results are quite limited, with only two baseline comparisons, one augmented-training example, and one ablation study shown. This makes it hard to visually assess the claimed improvements or the model’s generalization capability.

### Questions
- Just curious about why does the prediction-based method Interdiff with modification (text-conditioning) achieve such competitive results?
- Some large-scale subsets of the InterAct dataset, such as BEHAVE and OMOMO, lack explicit hand motion (showing the mean hand pose). In this case, is the proposed hand–object separation strategy still meaningful or effective? Additionally, does the data augmentation stage include any hand-related synthesis to compensate for the missing hand motion?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents LIGHT, a novel framework for generating 3D human-object interaction (HOI) motions from textual descriptions. At the core of the network, the authors propose a pace-induced guidance that avoid the usage of handcrafted contact priors for good contact quality and realism. Instead, LIGHT is formulated with two paths: a uniform pass denopises all modalities (body, hands, object), while a staged pass that uses the outputs from the first pass to guide the noise components. Additionally, a contact-aware shape-spectrum augmentation strategy is proposed to improve the generalization on unseen objects. Extensive experiments have been conducted to present its superior performance over existing techniques.

### Strengths
+ The proposed pace-induced guidance and contact-aware shape-spectrum augmentation are novel. Specifically, (1) The pace-induced guidance is proven to provide a data-driven altenative, which is even more effective than the priors used in previous methods. (2) The augmentation builds a informative invariance directly into the training data, leading to improved generatlization.

+ The method demonstrates clear quantitative and qualitative improvements over strong baselines (HOI-Diff, CHOIS, InterDiff) across multiple datasets (InterAct, BEHAVE, OMOMO) and metrics.

### Weaknesses
- The method is currently designed and evaluated for interacting with only one single object. Interactions with multiple and complex objects would be more beneficial. 

- Comparisons with zero-shot HOI generation methods, such as InterDreamer and ZeroHSI, may also be useful.

- Minor issues: The inference process requires around 72 seconds, which is higher than HOI-DIff and InterDiff (non-guided baselines).

### Questions
Besides the weaknesses listed above, the reviewer may have some additional questions:

+ Could the authors illustrate more about the failure mode, such as types of interactions and objects? 

+ How's the physical quality, such as penetration issues, of the generated HOI sequences? It might also be better to evaluate for the penetration issues?

### Soundness
3

### Presentation
3

### Contribution
3
