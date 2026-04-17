# CTCal: Rethinking Text-to-Image Diffusion Models via Cross-Timestep Self-Calibration

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 6, 4

## Abstract
Recent advancements in text-to-image synthesis have been largely propelled by diffusion-based models, yet achieving precise alignment between text prompts and generated images remains a persistent challenge. We find that this difficulty arises primarily from the limitations of conventional diffusion loss, which provides only implicit supervision for modeling fine-grained text-image correspondence. In this paper, we introduce Cross-Timestep Self-Calibration (CTCal), founded on the supporting observation that establishing accurate text-image alignment within diffusion models becomes progressively more difficult as the timestep increases. CTCal leverages the reliable text-image alignment (i.e., cross-attention maps) formed at smaller timesteps with less noise to calibrate the representation learning at larger timesteps with more noise, thereby providing explicit supervision during training. We further propose a timestep-aware adaptive weighting to achieve a harmonious integration of CTCal and diffusion loss. CTCal is model-agnostic and can be seamlessly integrated into existing text-to-image diffusion models, encompassing both diffusion-based (e.g., Stable Diffusion 2.1) and flow-based approaches (e.g., Stable Diffusion 3). Extensive experiments on T2I-Compbench++ and GenEval benchmarks demonstrate the effectiveness and generalizability of the proposed CTCal.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper addresses the challenge of imprecise text-image alignment in text-to-image diffusion models, proposing Cross-Timestep Self-Calibration (CTCAL) based on the observation that alignment degrades with increasing (more noisy) timesteps .
CTCAL uses reliable cross-attention maps from smaller (less noisy) timesteps to calibrate larger-timestep learning, with components like noun-prioritized attention selection, pixel-semantic joint optimization, and subject response regularization, plus timestep-aware weighting for integrating with diffusion loss .
Model-agnostic CTCAL works with diffusion-based (e.g., Stable Diffusion 2.1) and flow-based (e.g., Stable Diffusion 3) models .
Experiments on T2I-CompBench++ and GenEval show CTCAL improves alignment in attributes, object relationships, and compositions, with user studies confirming better visual and semantic fidelity .

### Strengths
1. Targeting a validated issue: It addresses the measurable degradation of text-image alignment with increasing diffusion timesteps, supported by cross-attention map visualizations .

2. Model agnosticism: It seamlessly integrates into diverse text-to-image diffusion models, including diffusion-based (e.g., SD 2.1) and flow-based (e.g., SD 3) approaches .

3. Comprehensive validation: It is rigorously tested on T2I-CompBench++/GenEval benchmarks and user studies, with no trade-offs in image diversity or aesthetic quality .

### Weaknesses
1. Limited novelty. It is essentially an integration of existing techniques rather than a breakthrough: using cross-attention for alignment, filtering non-semantic tokens, and combining multi-loss terms are all well-explored in prior diffusion model optimization works. The token mapping in the attention map has been well-explored in previous works, either during inference process or training process. For example, , Dreamo[1] explores routing constraints in DiT structure to distinguish multiple subjects  and Anystory[2] explores multiple subjects injection on SDXL with attention maps restrictions. 


2.  Fragile Noun Selection undermines core supervision. CTCAL’s performance depends on its POS-based filter, which relies on Stanza to extract "spatially meaningful" nouns. However, the paper admits this filter is flawed: it incorrectly includes non-spatial nouns (e.g., directional terms like "top" or "left") that lack semantic-spatial correspondence. The proposed workaround—an ad-hoc blacklist—is not generalizable, and the authors only mention "using LLMs for semantic filtering" as a future direction.

Reference:

[1] DreamO: A Unified Framework for Image Customization

[2]AnyStory: Towards Unified Single and Multiple Subject Personalization in Text-to-Image Generation

### Questions
please check the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a  fine-tuning method that leverages the text-image alignment (cross-attention maps) formed at smaller timesteps to explicitly calibrate the learning at larger timesteps. The goal of the paper is to improve the persistent challenge of poor alignment between the text prompt and generated images.

### Strengths
1. While a lot of prior works focus on improving image-text alignment during inference, its interesting to see this paper talk about providing explicit supervision for modeling fine-grained text-image correspondence during training instead.
2. The main idea of Cross-Timestep Self-Calibration is novel. It moves beyond conventional losses by introducing a self-supervised signal derived internally from the model's behavior at different levels of noise.

### Weaknesses
1. The method seems to add computational complexity, and the qualitative results do not seem strong enough to suggest utility of the proposed approach. For example, in the first half of Figure 4, the jar in the 5th column is just floating in the air, the banana in the 6th column looks unnatural and there is leakage of green to the banana.
2. The authors choose $t_{tea}=0$ in the final setup, but used t_{tea}=1 while motivating the overall approach in figure 1. I wonder whether t_{tea}=0 would give meaningful differentiation across attention maps for various objects unless I may be missing something.

### Questions
1. How reliable is the Part-of-Speech Tagger for complex prompts?
2. The authors mention in L203-204 that nouns (denoting objects or entities) are extracted and their attention maps are considered. I wonder whether the attributes of objects e.g. "yellow" for a yellow should be considered as well.
3. The real role of the autoencoder introduced is unclear to me. While a reconstruction task is used to prevent mode collapse, why would the alignment in the compressed space is better than pure pixel-level alignment?

### Soundness
2

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
The paper tackles the gap that text–image correspondence tends to be strong at small timesteps (low noise) but weak at large timesteps. It proposes a training-time self-calibration scheme (CTCAL): for each sample, a "teacher" small-timestep cross-attention map supervises the "student" large-timestep map. The loss combines pixel-level and semantic-level attention alignment, a subject-response balancing term, and a timestep-aware weighting that emphasizes harder (noisier) steps. On SD 2.1 and SD 3, the method improves compositional alignment and prompt faithfulness on standard benchmarks.

### Strengths
1. **Clear problem framing:** Directly targets cross-timestep misalignment.
2. **Simple supervision signal:** Reuses model-internal cross-attention as a self-supervised ``teacher".
3. **Comprehensive design:** Pixel + semantic attention alignment and subject balancing are sensible; timestep-aware weighting matches the difficulty profile.
4. **Empirical gains:** Consistent improvements on compositional/prompt-following metrics.

### Weaknesses
1. **Diversity risk from attention supervision.**  
   Using small-timestep attention to shape large-timestep behavior might bias the model toward more deterministic layouts and reduce output diversity. I wonder is there a drop on diversity or mode collapse in generation. An metric analysis or visualization result may help.

2. **Dataset construction and generalization.**  
   Training data is curated from T2I-CompBench-like prompts via reward-driven selection. This raises concerns about overfitting to that prompt style or metric. Please evaluate on broader benchmarks or metrics (e.g., FID, CLIPScore, HPS, ImageReward) to demonstrate the generalization.

3. **Positioning vs reward-based post-training (ReFL / GRPO family).**  
   While CTCAL focuses on improving cross-timestep consistency during training, recent post-training methods such as ReFL or GRPO also enhance text–image alignment by optimizing explicit reward signals. It would strengthen the paper to clarify how CTCAL compares or complements these approaches—both conceptually and empirically. A short discussion or a compute-matched comparison (e.g., ReFL vs. CTCAL under similar reward setups) would help readers understand whether CTCAL offers distinct benefits or can be combined with reward-based fine-tuning.

4. **Complexity and scalability of the training recipe.**  
   The approach combines several components (noun-focused maps, pixel+semantic alignment, subject regularization, timestep-aware weighting). It’s not obvious how robust this recipe is when scaling to larger/faster backbones. More evidence of training stability and a brief report of training cost or efficiency would make the method’s practicality more convincing.

### Questions
Please check the weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
