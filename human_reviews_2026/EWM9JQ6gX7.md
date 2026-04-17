# SDErasure: Concept-Specific Trajectory Shifting for Concept Erasure via Adaptive Diffusion Classifier

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
Concept erasure methods have proven effective in mitigating the potential for text‑to‑image diffusion models to produce harmful content. Nevertheless, prevailing methods based on post fine-tuning introduce substantial disruption to the original model’s parameter distribution and suffer from excessive model intrusiveness in two dimensions. (1) Images generated under erased concepts are perceptually aberrant. (2) Images generated under unrelated concepts exhibit pronounced quality degradation. We attribute these limitations to applying a uniform strategy to erase diverse concepts, failing to account for concept-specific generative mechanisms. Through rigorous experimentation and analysis, we identify that the generative process of each concept hinges on a narrow subset of critical timesteps. This insight motivates a targeted intervention strategy that enables precise and minimally invasive concept erasure. Therefore, we introduce $\textbf{SDErasure}$, a novel training framework for concept-specific erasure via adaptive trajectory shifting. First, a Step Selection algorithm that utilizes a diffusion classifier is proposed to guide the model in pinpointing the key timesteps associated with the undesired concept’s generation. Second, a Score Rematching loss is introduced to align the model’s predicted score function with that of anchor concepts, extending its applicability to both anchor-free erasing and anchor-based altering. Third, a Quality Regulation consisting of early-preserve loss and concept-retain loss is introduced to maintain the model's generative quality along two dimensions. Empirical results demonstrate that SDErasure achieves state-of-the-art concept erasure performance, reducing FID from 9.51 to 6.74 while effectively eliminating the target concept.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper observes that different concepts show distinct generative processes, with critical timesteps varying across concepts. Building on this insight, the paper introduces SDErasure, a novel concept erasure framework for text-to-image diffusion models. First, it proposes a step selection strategy to identify the critical timesteps most relevant to the target concept. It then introduces a score rematching loss to steer the generative process away from the target concept and toward an anchor concept.
To preserve the model’s generative capability, two additional regularization losses are proposed: the early-preserve loss, which enforces consistent noise predictions for target concepts at early steps, and the concept-retain loss, which maintains prediction consistency for non-target concepts after erasure.
Comparisons with prior concept erasure methods across four different domains (objects, celebrities, artistic styles, and sensitive content) demonstrate the effectiveness of SDErasure.

### Strengths
- The idea of adaptively selecting critical timesteps for concept erasure is novel and well-motivated.
- The proposed framework is efficient and effective, achieving strong performance in erasing target concepts while preserving non-target ones in all settings.
- The paper is well-written and easy to follow.

### Weaknesses
- The paper lacks an ablation study on the timestep selection strategy. It is unclear how the identified timesteps are verified to be optimal. For example, in Figure 6, the critical timesteps for erasing the Van-Gogh-style concept are found in the mid-to-late stages, but the truly optimal timesteps might be in the early-to-middle stages.
- The generalizability of the step selection strategy remains uncertain. If the total number of timesteps changes (e.g., from 50 to 200), can the same strategy still identify the correct critical timesteps?
- Since SSScore plays a crucial role in SDErasure, it would be valuable to include additional analyses similar to Figure 6. For example, what are the SSScore distributions for different concept categories, such as objects and celebrities? Do they follow consistent patterns, e.g., critical timesteps for objects appearing in mid stages and for celebrities in mid-to-late stages?
- In the multi-concept erasure experiments (Table 10), several established methods, such as UCE, MACE, and RECE, are omitted. Including these baselines would provide a more complete evaluation.
- (minor) In Appendix E, what do the hyper-parameters $\alpha$ and $\beta$ represent? Are they typos for $\beta_1$​ and $\beta_2$​ in Equation (14)?
- (minor) What values are used for the hyper-parameter $\eta$ in Equation (11) and for the iteration count $K$ in the pseudocode?

### Questions
See the weaknesses above.

### Soundness
2

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
4

### Summary
This paper presents SDErasure, a new framework for concept-specific erasure in text-to-image (T2I) diffusion models. Rather than modifying the full model, SDErasure focuses on selectively fine-tuning critical denoising timesteps, which are identified using a Step Selection algorithm. This approach allows the model to effectively erase target concepts with minimal disruption to unrelated content. By introducing two key technique: Score Rematching Loss and Quality Regulation, the method achieves a strong balance between effective concept erasure and preserving generative quality. Experimental results show that SDErasure outperforms existing methods in various scenarios, especially when erasing semantically meaningful or fine-grained concepts.

### Strengths
1. The paper makes a key observation: only a subset of denoising timesteps are critical for concept representation in diffusion models. By fine-tuning these specific steps instead of the whole model, the method is both efficient and targeted, reducing unnecessary side effects.
2. Quality Regulation is an important contribution. It explicitly addresses the trade-off between removing unwanted concepts and maintaining high-quality generations. This helps prevent issues like over-erasure or degradation of unrelated content.
3. The use of Score Rematching Loss aligns the denoising trajectory after erasure with the expected distribution.

### Weaknesses
1. The method’s effectiveness appears to be sensitive to anchor concept selection. This dependence might limit the method's generalizability across concepts that lack well-defined or semantically close anchors. A more robust or anchor-independent strategy would improve usability.

2. While the Step Selection algorithm is central to the method, the paper does not provide a theoretical justification for why the selected timesteps are optimal or what properties make certain steps more important. This makes it harder to understand or predict the method's behavior.

3. Although the method performs well on many tasks, performance gains are inconsistent. In some cases, improvements are small or even absent. The paper could benefit from a more in-depth analysis of failure cases, including explanations of when and why the method struggles.

4. Several related works on concept erasure and safety in T2I models are not cited or discussed. Notably missing are:

[1] Eraseanything: Enabling concept erasure in rectified flow transformers

[2] Dark miner: Defend against unsafe generation for text-to-image diffusion models

[3] One Image is Worth a Thousand Words: A Usability Preservable Text-Image Collaborative Erasing Framework

[4] Erasing More Than Intended? How Concept Erasure Degrades the Generation of Non-Target Concepts

### Questions
1. How sensitive is the method to the choice of threshold in the Step Selection algorithm? Does this threshold need to be manually tuned for each concept, or can it be set universally? Is there any theoretical or empirical basis for choosing an optimal value?

2. Structural concepts (e.g., human pose or object layout) are often deeply embedded in the generation process. Could the Quality Regulation mechanism interfere with the goal of erasing such structural concepts, especially if they strongly influence generation quality?

3. For multi-concept erasure, if both structural and fine-grained concepts need to be erased simultaneously, will the method still work effectively?

4. SDErasure effectively preserves the generative quality of unrelated concepts while removing the target concept. Can you provide a more detailed analysis of how Quality Regulation and Score Rematching loss impact generative quality, respectively?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a fine-grained loss for concept erasure. It targets key timesteps critical for each concept, which are automatically identified using a diffusion classifier. By incorporating constraints to preserve untargeted concepts, the method achieves effective concept erasure in experiments. The authors evaluate the approach against multiple baselines.

### Strengths
1. The fine-grained loss across different timesteps for each concept is noteworthy, and the corresponding early-preserve loss mitigates the instability issues of fine-tuning.
2. The selection of critical timesteps demonstrates potential for effective concept erasure while preserving untargeted concepts.
3. The authors compare their method with recent attacks, such as SPEED and ANT, showing that it achieves superior performance.

### Weaknesses
1. Appendix G.3 presents the performance of the proposed method in multi-concept settings. However, comparisons with other methods in these settings should be included to better demonstrate the scalability of the approach.

2. Does the step selection lead to error prediction? This could help determine whether all concepts follow the same principles during step selection.

3. The SSScore is calculated at each timestep. How does this impact computational efficiency?

### Questions
1. Comparison with other methods in multi-concept settings.
2. Analysis of error predictions in step selection.
3. Assessment of computational efficiency.

### Soundness
2

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
5

### Summary
This paper introduces SDErasure, a concept erasure method for diffusion models that adaptively targets concept-specific timesteps based on a diffusion classifier. It proposes a Step Separability Score (SSScore) to select critical steps, and introduces Score Rematching and Quality Regulation to balance erasure efficacy and generation quality. The method outperforms prior work across multiple erasure tasks.

### Strengths
1. The paper is built on clear observations that different types of concepts emerge at different stages of the denoising process, and addresses this with a principled, targeted erasure strategy.
2. The proposed SSScore + Step Selection mechanism allows the model to automatically identify key timesteps for each concept, improving erasure precision without requiring heuristic or manual selection. Specifically, I praise the usage of the diffusion classifier to guide the training process.
3. The experiments are extensive, covering multiple concept types, metrics (efficacy, specificity, generality), and include ablations, multi-concept erasure, and comparisons with strong baselines.

### Weaknesses
1. The SSScore must be computed at every timestep via multiple forward passes using a diffusion classifier. Although this is only done once per concept, the cost is still high, especially when scaling to many concepts or large batches.

2. Although anchor-free erasure is supported, in many cases selecting a suitable anchor is important for quality preservation. The method relies on heuristic selection rules, and the anchor choice significantly affects performance. However, there seems no effective anchor-selection method now (though mentioned in the limitations, but I believe this is really important in this method).

### Questions
1. Since the method evaluates SSScore at every single timestep using multiple forward passes, this may become a bottleneck for large-scale or multi-concept erasure tasks. Can this process be approximated or accelerated further?
2. How robust is the method when anchors are suboptimal, or when no clear anchor exists? Can the method adaptively learn anchors instead of relying on heuristic rules? 

I expect the authors to response to these concerns. I will raise my rating if they are well-addressed.

### Soundness
3

### Presentation
3

### Contribution
3
