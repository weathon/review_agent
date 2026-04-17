# FreeFuse: Multi-Subject LoRA Fusion via Auto Masking at Test Time

- Decision: Reject
- Scores: 4, 2, 6, 4

## Abstract
This paper proposes FreeFuse, a novel training-free approach for multi-subject text-to-image generation through automatic fusion of multiple subject LoRAs. In contrast to existing methods that either focus on pre-inference LoRA weight merging or rely on segmentation models and complex techniques like noise blending to isolate LoRA outputs, our key insight is that context-aware dynamic subject masks can be automatically derived from cross-attention layer weights. Our analysis shows that constraining each LoRA’s influence to its corresponding subject region via these masks effectively mitigates feature conflicts between LoRAs. FreeFuse demonstrates superior practicality and efficiency as it requires no additional training, no modification to LoRAs, no auxiliary models, and no user-defined prompt templates or region specifications. Alternatively, it only requires users to provide the LoRA activation words for seamless integration into standard workflows. Extensive experiments validate that FreeFuse outperforms existing approaches in both generation quality and usability under the multi-subject generation tasks. We will release the source code upon the official publication of the paper.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents a method for personalized multi-subject generation via LoRA fusion. Given several LoRAs, each trained on a specific subject separately, the goal is to generate a single image containing multiple personalized subjects.
The proposed method consists of two main stages. In the first stage, a mask is generated for each subject, indicating the image regions where the corresponding LoRA should be applied. In the second stage, these masks are used to fuse the predictions from the different LoRAs by multiplying each LoRA's output with its corresponding mask.

Merging multiple LoRAs using masks is a well-known technique in the community (https://github.com/lifeisboringsoprogramming/sd-webui-lora-masks?tab=readme-ov-file). Therefore, the main contribution of the paper lies in its automatic mask generation approach, which requires no user input or external model.

The method is evaluated both qualitatively and quantitatively against several approaches for personalized multi-subject generation.

### Strengths
- The results presented in the paper are of high quality, demonstrating good identity preservation and in most cases plausible prompt adherence.
- The paper provides relevant background on competing methods and clearly differentiates itself from the baselines.
- A large gallery of qualitative results is presented, effectively showing the method's performance.
- The paper includes multiple quantitative metrics and shows consistent advantages across all of them.

### Weaknesses
- Section 3.1 lacks clarity in both reasoning and presentation:
    - If the goal of this subsection is to demonstrate that Equation 4 holds, it would be more direct to compute each term in the equation across multiple images, timesteps, and layers, and then show their similarity. The discussion about the queries, keys, values, and feed-forward layers seems unnecessary for this purpose.
    - In Figure 4, the average MSE loss when disabling to_q and to_k appears comparable to the loss when disabling to_v. While the qualitative example shows a difference, this is based on a single sample. Therefore, I am not fully convinced by the conclusion supporting Equation 2.
    - Moreover, it is unclear why, if the LoRA outputs are typically 1-2 orders of magnitude smaller than the base model's outputs, they would affect the Q,K layers differently from the V,FF layers. This further raises doubts about the validity of Equation 2.
    - The computation in Figure 3a is not entirely clear.

- Given that the method is easy to implement (this is an advantage), it would strengthen the paper to include experiments on additional base models (e.g., SD, SDXL) for a fairer comparison with existing baselines.

- Since the main contribution lies in the mask extraction, this part should be evaluated more thoroughly and compared with other techniques. For instance, how does it perform relative to a simple average of attention maps across timesteps and layers, where the queries correspond to image pixels and the keys correspond to subject-related prompt tokens (as in Prompt-to-Prompt)? There are also related approaches such as Readout-Guidance and Self-Guidance that could serve as baselines for mask extraction.

- The paper only demonstrates results for two subjects, whereas other works (e.g., Mix-of-Show) show examples with more subjects. 

- In some examples, prompt adherence appears weaker than in competing methods. For example, on page 15, in the top example, the faces are not dusted with flour, and in the 7th row, the faces are not smeared with color as specified in the prompt.

### Questions
See the questions in the Weaknesses Section.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper addresses the critical challenge of feature conflicts in multi-subject text-to-image generation when fusing multiple subject LoRAs during joint inference. Existing methods often require retraining, auxiliary segmentation models, user-defined prompts/regions, or pre-inference LoRA weight merging—limiting their practicality. To solve this, the authors propose FreeFuse, a training-free framework tailored for Diffusion Transformers (DiTs) that enables seamless multi-LoRA fusion via automatically derived subject masks.
FreeFuse requires no additional training, no modifications to existing LoRAs, no auxiliary models, and only needs users to provide LoRA activation words. Experiments on FLUX.1-dev show it outperforms baselines (LoRA Merge, ZipLoRA, OMG, Mix-of-Show, CLoRA) across metrics: e.g., achieving a VLM score of 74.03 (vs. 57.74 for Mix-of-Show) and a 10-Pass LVFace score of 0.4685 (vs. 0.4417 for Mix-of-Show). It also excels in complex subject interactions (e.g., hugging, whispering) that prior methods struggle with.

### Strengths
- Unlike methods requiring retraining (Mix-of-Show) or auxiliary segmentation models (OMG), FreeFuse operates entirely at test time. It needs no LoRA modifications, no user-defined region prompts, and only requires LoRA activation words—enabling seamless integration into standard text-to-image workflows.
- FreeFuse addresses key flaws of attention-based mask extraction (e.g., attention sink, noisy pixel-wise maps) via heuristic filtering, self-attention locality exploitation, and superpixel voting. This ensures masks are accurate and spatially coherent without human intervention.
- Most existing multi-LoRA fusion methods (e.g., ZipLoRA, CLoRA) are designed for UNet-based models. FreeFuse targets DiTs (e.g., FLUX.1-dev), filling a critical gap in supporting state-of-the-art transformer-based diffusion models.
- Experiments use diverse metrics to evaluate identity preservation (LVFace), feature similarity (DINOv3), human preference (DreamSim, HPSv3), and prompt adherence (VLM). It compares against 5 strong baselines and validates on complex interaction scenarios—strengthening the credibility of its effectiveness.

### Weaknesses
- FreeFuse’s core application scenario (generating multi-subject interaction images, e.g., hugging, face-to-face talking) overlaps heavily with methods designed for character relationship synthesis (e.g., DreamRelation). However, the paper fails to cite or compare with such works, leaving its novelty relative to state-of-the-art interaction-focused generation methods unclear.
- All experiments rely on a small, fixed set of subjects (e.g., daiyu_lin, haoran_liu, Harry Potter, Rihanna). There is no validation on more diverse identities—such as subjects of different ethnicities, ages, artistic styles (e.g., cartoon vs. photorealistic), or non-human subjects (e.g., animals, fictional creatures). This limits the demonstration of FreeFuse’s generalizability.
- The paper excludes recent training-free multi-LoRA fusion methods beyond K-LoRA (e.g., latest variants of LoRA merging or dynamic gating approaches). This incomplete comparison may overstate FreeFuse’s advantages by ignoring competing methods with similar practicality.
- The authors explicitly acknowledge that FreeFuse degrades when the number of subject LoRAs increases. As each LoRA’s masked region shrinks, features from other LoRAs are more likely to intrude—making the method ineffective for scenarios with 5+ subjects.
- In scenes with heavy subject overlap (e.g., two people embracing with intertwined limbs), the attention-based masks may fail to accurately separate individual subjects. This leads to residual feature conflicts that FreeFuse cannot resolve.

### Questions
- Why were methods for character relationship synthesis (e.g., DreamRelation) not compared? How does FreeFuse’s performance on multi-subject interaction tasks differ from these methods, especially in terms of interaction naturalness and identity preservation?
- What is the reason for using only a small set of fixed identities in experiments? If tested on more diverse subjects (e.g., elderly individuals, non-Western ethnicities, cartoon characters), would FreeFuse’s metrics (e.g., LVFace similarity, mask accuracy) remain stable, or would performance degrade?
- The paper notes performance issues with many LoRAs, but no potential solutions are proposed. Could dynamic mask resizing, multi-scale attention fusion, or adaptive LoRA activation weights address this limitation?
- Can FreeFuse be adapted to other DiT models (e.g., Stable Diffusion 3) or UNet-based models? If so, would the mask generation step (e.g., layer selection, denoising step) require significant adjustments?
- How would FreeFuse handle scenes with extreme subject overlap (e.g., a group hug with 3+ people)? Is there a strategy to improve mask accuracy in such cases, such as integrating lightweight geometric cues?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents FreeFuse, a training-free and segmentation-free framework for multi-subject text-to-image generation by fusing multiple subject LoRAs directly at inference. Instead of retraining or merging LoRAs, FreeFuse derives context-aware subject masks from cross-attention maps and applies them to LoRA outputs, mitigating feature conflicts during joint inference. The key insight is that context-aware dynamic subject masks can be automatically derived from cross-attention layer weights, which well approximate the case where each LoRA is integrated into the diffusion model and used individually. FreeFuse extracts these masks from a single attention block and denoising step, achieving efficiency advantages over prior methods such as CLoRA, OMG, and Mix-of-Show. FreeFuse outperforms several baselines on DINOv3, DreamSim, HPSv3, and Gemini-2.5 VLM metrics, with notably higher VLM score (74.03 vs 57.74 for Mix-of-Show).

### Strengths
1. Clear problem motivation and theoretical grounding: This paper identifies intense competition among LoRAs in key subject regions as the source of failures in joint inference, supporting it with cosine-similarity visualizations of latent interference.

2. Elegant and efficient formulation: The core mathematical argument formally justifies why masking LoRA outputs approximates isolated inference, showing that locality of attention ensures near-identical representations inside the mask.

3. Attention-based automatic mask extraction: The pipeline introduces attention-sink filtering and superpixel-level voting to ensure spatial coherence. Importantly, it requires no retraining, no LoRA modification, and no external segmentation.

4. Strong empirical evaluation: Evaluation uses five complementary metrics: DINOv3, DreamSim, LVFace, HPSv3, and Gemini-2.5 VLM, covering both perceptual similarity and human preference alignment. FreeFuse achieves the highest VLM score (74.03) and DreamSim 10-pass (0.8052), showing superior realism and consistency.

5. Comprehensive ablation studies: Fig. 7 clearly isolates the effect of attention-sink handling, self-attention maps, and block-level voting; omitting any step causes visible artifacts, reinforcing the necessity of each design.

### Weaknesses
1. Lack of runtime and resource benchmarks: Implementation details mention 37s per image on a single L20 GPU, but omit comparisons with CLoRA/OMG or multi-step variants, making it hard to quantify efficiency gains.

2. Potential bias toward photorealistic scenarios. Evaluation prompts emphasize intimate, realistic human interactions. It remains unclear whether FreeFuse generalizes to style LoRAs, cartoons, or abstract concepts.

3. Scalability is unclear with more subjects. Yet no quantitative evidence is provided for how performance degrades beyond two subjects.

### Questions
1. Generalization beyond human characters — All evaluation prompts involve human pairs. Have you tested FreeFuse on object+character or style+subject LoRA fusion (e.g., “anime character + van Gogh style”)? If so, do auto-masks still localize meaningfully?

2. Attention-sink filtering parameters — In Eq. 6, you fix p = 1 %. Did you tune p for different resolutions or datasets? How sensitive are final masks and performance to this threshold?

3. Scalability with subject count — Could you report quantitative degradation (e.g., VLM / LVFace scores) for 3-, 4-, and 5-subject scenes to illustrate the scaling limit?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces FreeFuse, a training-free framework for multi-subject text-to-image generation that enables fusing multiple LoRA modules without retraining, external models, or manual prompt engineering. The key insight is that cross-attention maps in diffusion transformer (DiT) models contain sufficient spatial locality to derive subject-specific masks automatically. These masks are then applied to constrain each LoRA’s effect to its relevant region, mitigating feature conflicts during joint inference.
The method is implemented in two stages: (1) mask extraction via filtered attention maps and superpixel-based voting, and (2) mask-guided inference where LoRA outputs are masked at each denoising step. Extensive experiments on FLUX.1-dev demonstrate that FreeFuse outperforms prior works such as OMG, CLoRA, Mix-of-Show, and ZipLoRA across quantitative metrics (DINOv3, DreamSim, LVFace, HPSv3, and VLM scoring). The approach is practical, efficient, and requires no modification to existing LoRA modules

### Strengths
The method is entirely training-free, does not modify LoRAs or base diffusion models, and integrates seamlessly with existing workflows. This makes it extremely relevant for real-world adoption. The primary strength of this work lies in its "plug-and-play" nature. By eliminating the need for any additional training, external models, or manual user intervention (like region specifications), FreeFuse presents a highly practical solution to a common problem. Using cross- and self-attention maps for automated subject mask generation is a simple yet powerful concept. Theoretical analysis showing that masked LoRA outputs approximate isolated inference adds credibility.

### Weaknesses
1. The most significant weakness is the method's unproven scalability. All qualitative examples in the paper demonstrate a fusion of exactly two subject LoRAs. The authors explicitly concede this limitation in the conclusion, stating that the method's core premise gradually becomes invalid as the number of subject-LoRAs increases. For a paper on multi-subject fusion, the lack of any experimental validation (even as a failure case analysis) for three or more subjects is a major omission.
2. The mathematical justification (Eq. 4) is based on empirical assumptions about attention locality and LoRA perturbation magnitude. While reasonable, it remains an approximation rather than a rigorous derivation.
3. Most baselines (OMG, Mix-of-Show, CLoRA) were implemented on U-Net diffusion models. Evaluating FreeFuse against more recent DiT-native multi-LoRA methods would strengthen the claim of DiT superiority.

### Questions
1.Does FreeFuse perform equally well on non-human or abstract LoRAs (e.g., style or object-based ones)? Are the masks still meaningful in such cases?
2. The masks are computed once (step 6) and reused. How stable are these spatial associations across later denoising steps? Would re-computation improve consistency?
3. Can the authors report actual runtime and memory overhead compared to baseline inference? The claim of “one-step extraction” suggests efficiency, but quantitative figures would help.
4. Have the authors explored hierarchical or recursive masking strategies for cases with more than three subjects? Such an extension could broaden the method’s applicability.

### Soundness
3

### Presentation
3

### Contribution
2
