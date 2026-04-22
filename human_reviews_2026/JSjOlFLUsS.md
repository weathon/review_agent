# Localized Concept Erasure in Text-to-Image Diffusion Models via High-Level Representation Misdirection

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 4

## Abstract
Recent advances in text-to-image (T2I) diffusion models have seen rapid and widespread adoption. However, their powerful generative capabilities raise concerns about potential misuse for synthesizing harmful, private, or copyrighted content. To mitigate such risks, concept erasure techniques have emerged as a promising solution. Prior works have primarily focused on fine-tuning the denoising component (e.g., the U-Net backbone). However, recent causal tracing studies suggest that visual attribute information is localized in the early self-attention layers of the text encoder, indicating a potential alternative for concept erasing. Building on this insight, we conduct preliminary experiments and find that directly fine-tuning early layers can suppress target concepts but often degrades the generation quality of non-target concepts. To overcome this limitation, we propose High-Level Representation Misdirection (HiRM), which misdirects high-level semantic representations of target concepts in the text encoder toward designated vectors such as random directions or semantically defined directions (e.g., super-categories), while updating only early layers that contain causal states of visual attributes. Our decoupling strategy enables precise concept removal with minimal impact on unrelated concepts, as demonstrated by strong results on UnlearnCanvas and NSFW benchmarks across diverse targets (e.g., objects, styles, nudity). HiRM also preserves generative utility at low training cost, transfers to state-of-the-art architectures such as Flux without additional training, and shows synergistic effects with denoiser-based concept erasing methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes HiRM, a concept erasure method that edits only the text encoder with a decoupled strategy: it sets the erasure target on the final text-encoder layer but updates only the first layer. There are two variants: HiRM-R (random target) and HiRM-S (semantic/safety target). Results on UnlearnCanvas and several adversarial benchmarks show a balanced trade-off: reasonable erasure, small utility loss, low compute, and transferable between model variants (e.g., SD/FLUX).

However, gains over Diff-Q on UnlearnCanvas are small, and there is a clear gap to SOTA on some adversarial benchmarks. The NSFW pipeline relies on templates and keyword gates, which may not generalize to other abstract concepts or compositional prompts.

Overall, the method is simple and practical, but the novelty is not enough and the robustness is not fully established.

### Strengths
- Decoupled supervision: HiRM applies the erasure objective at the final text-encoder block and only update the first block, avoiding low-level "representation shattering" and yielding a better erasure–utility trade-off. 

- Efficiency and modularity: Only editing the text encoder makes the approach easy to train and transferable across pretrained T2I models (e.g., SD and FLUX variants) without modifying the backbone (UNets and Transformers).

- Balanced empirical profile: On UnlearnCanvas, HiRM-S matches or slightly improves over strong text-side baselines with very low cost rather than heavy retraining.

### Weaknesses
- The paper is an incremental iteration on Diff-Q: it still modifies the first transformer block of the text encoder but shifts the optimization target to the final layer, which carries higher-level representations. Conceptually sound, but not a paradigm shift.

- Gains over Diff-Q are small. In Table 2, IRA (object erasure) drops from 98.37 to 98.18 (−0.19 pts). The gap to SOTA remains large: in Table 3 the method underperforms on UnlearnDiffAtk (19.01 vs. 9.80) and MMA-Diffusion (3.30 vs. 0.40).

- For NSFW, HiRM-S constructs a safety vector by differencing prompt embeddings, making it dependent on template and vocabulary. This may risk brittleness under template changes, across languages, or at nuanced concept boundaries.

### Questions
1. Your method uses two pipelines for *nudity* vs. *non-nudity* (keyword/template gated). Why is this split necessary? Could a single pipeline (no keyword gate) achieve similar or better results? 

2. For abstract attributes (e.g., *violent scene* vs. *scene*), the difference vector seems to capture a theme (e.g., war) rather than the attribute(violence). How do you ensure the vector encodes the attribute itself, not a topic proxy? Show stress tests across domains (e.g., domestic/school violence).

3. How does the method handle compositional prompts with scope and negation (e.g., "… but without weapons")? Can you provide some targeted evaluations (AND/OR/NOT, "topic vs. attribute" disentanglement).

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
3

### Summary
The paper proposed a concept erasure method that operates solely on the text encoder, called High-Level Representation Misdirection (HiRM). Specifically, HiRM guides the token representations output by the last layer of the text encoder in an incorrect direction and updates the weights of the first layer accordingly. Extensive experiments demonstrate the effectiveness of the proposed method.

### Strengths
* The writing is fluent and logically coherent, exhibiting strong readability.
* The proposed method is highly modular, requiring only a modification to the first layer of the text encoder to achieve concept erasure, which demonstrates strong practical applicability.  
* The experimental design is thorough, and the results yield insights with meaningful implications for the research community.

### Weaknesses
* The proposed method is relatively empirical and experimental, lacking solid theoretical support. It would be more beneficial to the community if the interpretability of the erased concept could be analyzed from the perspective of the distribution of activated neurons.
* The core idea of HiRM lies in computing the loss based on the output of the last layer of the text encoder, thereby enhancing its ability to erase high-level concepts. However, in Figure 2, the elimination of the high-level concept *Fauvism* appears to be less effective than that of the more concrete concept *Tree*. Why is this the case?
* Although the authors provide additional examples in the Appendix, the reviewer finds that the visualizations still lack sufficient diversity. Including more prompts and results across a wider range of backbone models would make the work more convincing.
* The reviewer observes that both HiRM-R and HiRM-S may, to some extent, affect the similarity between non-target concepts and the original model outputs — as shown in Figures 9 and 10. Could the authors please provide an explanation for this issue?

### Questions
See 'Weaknesses'.

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
3

### Summary
This paper leverages a fine-tuned text encoder for concept erasure. The authors propose fine-tuning the early layers of the text encoder by directing the high-level semantic representations of target concepts toward random or super-category directions. Experimental results on single-concept erasure demonstrate that the method achieves comparable performance while requiring minimal time and memory.

### Strengths
1. Rather than training the diffusion model parameters, the authors fine-tune the text encoder parameters to improve efficiency.
2. The proposed method is straightforward and easy to implement.
3. The idea of using high-level semantic representations to guide updates in the early layers is interesting.

### Weaknesses
1. Related work is missing. SPEED [A] leverages null-space constraints to achieve rapid concept erasure and can be extended to multi-concept scenarios. The authors should include SPEED as a baseline and compare efficiency.

2. Although the authors mention plans to extend the proposed method to multi-concept erasure, its tuning-based nature limits scalability. As more concepts are introduced, optimizing the early layers becomes increasingly difficult. Therefore, the authors should include a multi-concept erasure setting to demonstrate the method’s potential.

3. Table 2 shows that Diff-Q achieves comparable results on UnlearnCanvas, and unlike training-based methods, it attains higher performance via a closed-form solution. It shows limited improvement introduced by HiRM. Besides, Table 1 reports the low performance of Diff-Q in preserving untargeted concepts. Does this suggest that the metrics used in Table 2 may not effectively reflect the ability to preserve untargeted concepts, given that IRA and CRA scores are near 100?

[A] Li, Ouxiang, et al. "Speed: Scalable, precise, and efficient concept erasure for diffusion models." arXiv preprint arXiv:2503.07392 (2025).

### Questions
1. Comparison with SPEED.
2. Support for multi-concept erasure.
3. Choice and justification of evaluation metrics.
4. Would introducing additional constraints on the higher layers (e.g., the last two layers) improve performance?

### Soundness
1

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
4

### Summary
The paper is motivated by the observation that Diff-QuickFix, which modifies only the first self-attention layer, performs poorly when addressing high-level abstract concepts such as nudity or NSFW content. Moreover, when modifying the entire first transformer block toward random vectors (inspired by the RMU approach in LLMs), it tends to over-unlearn, negatively affecting unrelated concepts.

To address this, the paper proposes targeting high-level concept representations in the final encoder block (rather than the first layer) while applying updates only to the early-layer weights. This decoupling strategy enables more precise concept removal with minimal impact on unrelated concepts.

### Strengths
•  The paper is clearly written and easy to follow.

•  The proposed method is simple and intuitively reasonable.

•  The experiments appear comprehensive, and the results look promising.

### Weaknesses
-	In diffusion model training, the text encoder is typically a pre-trained model (e.g., CLIP text encoder) and remains frozen throughout the training process. This means that if unlearning is applied only to the text encoder, malicious users could easily replace the sanitized text encoder with the original one to recover all unlearned concepts. Therefore, in open-source settings, fine-tuning the core denoising model (i.e., the U-Net) makes more sense and is a more robust approach. This can be seen in Table 4, where HiRM-R does not perform as well as ESD, CA on Flux architectures (which use two text encoders rather than one, as in Stable Diffusion) 
-	The proposed method appears to be a simple extension of RMU (a popular unlearning method for LLMs) to diffusion models, with the additional trick of fine-tuning only the early layers instead of all layers. This could be viewed as the result of a simple ablation study on layer selection rather than a fundamentally new approach. 
-	While the paper emphasizes its efficiency (faster than training-based methods), this seems somewhat expected — since it only updates weights without requiring output generation (as in ESD or other training-based methods), it is naturally much faster.

### Questions
•  Does the method employ any losses to preserve unrelated concepts, similar to UCE or SHS? If yes, how are the to-be-retained concepts chosen?

•  How does the method perform against the Random Probe recovery attack proposed in [1], which adds noise to the text encoder to confuse generation and recover unlearned concepts?

•  How are the super-categories determined? Are they predefined manually, or can they be learned end-to-end as in [2]?

[1] Lu, Kevin, et al. "When Are Concepts Erased From Diffusion Models?." NeurIPS 2025 

[2] Bui, Anh, et al. "Fantastic targets for concept erasure in diffusion models and where to find them." ICLR 2025

### Soundness
2

### Presentation
2

### Contribution
2
