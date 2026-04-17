# Mitigating Semantic Collapse in Generative Personalization with Test-Time Embedding Adjustment

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 4

## Abstract
In this paper, we investigate the semantic collapsing problem in generative personalization, an under-explored topic where the learned visual concept ($V$) gradually shifts from its original textual meaning and comes to dominate other concepts in multi-concept input prompts. This issue not only reduces the semantic richness of complex input prompts like "a photo of $V$ wearing glasses and playing guitar" into simpler, less contextually rich forms such as "a photo of $V$" but also leads to simplified output images that fail to capture the intended concept. We identify the root cause as unconstrained optimisation, which allows the learned embedding $V$ to drift arbitrarily in the embedding space, both in direction and magnitude. To address this, we propose a simple yet effective training-free method that adjusts the magnitude and direction of pre-trained embedding at inference time, effectively mitigating the semantic collapsing problem. Our method is broadly applicable across different personalization methods and demonstrates significant improvements in text-image alignment in diverse use cases. Our code is published at \url{https://github.com/tuananhbui89/Embedding-Adjustment}.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The author define semantic collapse problem and propose a method TEA to mitigate it in test-time without re-training or altering embedding weights. Experiements show that TEA perform well on top of multiple personalizated models.

### Strengths
1. This paper provide a in-deep analysis about the root cause of SCP, claiming it's beacuse unconstrained optimisation, which make senses to this field.
2. The method is simple yet effective, though the idea that using the concept token to regularize the V* token is not novel, simply using SLERP at test-time is somehow interesting.
3. Comprehensive qualitative experiments shows the effectiveness of proposed TEA.

### Weaknesses
1. Novelty of SCP's definition: SCP is not an problem. that has not been discussed  What's the difference between SCP and semantic drift phenomenon discussed in ClassDiffusion[1]? According to the author's claim, they share a same phenomenon: can only generate "a photo of V*" when using the prompt "a photo of V* with other concepts". CoRe[2] also discuss similar problem. Also, some negative impace are somehow discussed in their paper.
2. Insufficient evaluation: as CLIP-T and CLIP-I often fail to provide fair evaluation about fine-grained results, could author provie other quantitative results about TEA's performance? Like using user study or VLMs as judge.


[1] Huang, Jiannan, et al. "ClassDiffusion: More Aligned Personalization Tuning with Explicit Class Guidance." The Thirteenth International Conference on Learning Representations.
[2] Wu, Feize, et al. "Core: Context-regularized text embedding learning for text-to-image personalization." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 39. No. 8. 2025.

### Questions
- What's CLIP$^{p}_T$ stands for? This metric is mentioned in Table 1, but not described in the experiment section. 
- Do TEA's result align with author's discuss about SCP? Could author provide similar visulization/chart on models w/ TEA?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the ​​Semantic Collapsing Problem in generative personalization, where personalized token embeddings drift significantly in both magnitude and direction during optimization, causing them to dominate multi-concept prompts and degrade compositional generation. The authors propose ​​Test-time Embedding Adjustment (TEA)​​, a training-free method that recalibrates the embedding’s norm and direction toward its original semantic concept during inference.

### Strengths
The identification and empirical analysis of SCP are interesting. 

TEA is lightweight, requires no retraining, and is compatible with numerous existing frameworks.

The paper is well-structured and easy to follow.

### Weaknesses
While the Test-time Embedding Adjustment (TEA) method is practical and easy to deploy, its technical contribution is relatively modest. The core mechanism—adjusting the magnitude and direction of an embedding vector—is a straightforward application of existing vector space operations, lacking the novelty of a more transformative technique. The approach does not introduce new learning paradigms or architectural innovations, but rather applies a post-hoc correction to the outputs of existing models. This simplicity, though beneficial for accessibility, may limit the method's impact and long-term relevance in a fast-evolving field that often rewards more complex or foundational advancements.

The paper heavily relies on quantitative metrics like CLIP and DINO scores to demonstrate improvements, but provides limited qualitative evidence, especially for subject-driven customization tasks. For instance, while quantitative gains are reported, side-by-side visual comparisons showcasing TEA's enhancement over base methods (e.g., DreamBooth, Textual Inversion) across diverse concepts are sparse. More juxtaposed examples illustrating how TEA better preserves contextual details in complex prompts (e.g., "V* wearing glasses and writing in a red notebook") would strengthen the empirical claims. A broader set of qualitative results is necessary to help readers appreciate the practical improvements in output quality and alignment.

A thorough discussion of scenarios where TEA underperforms or introduces new artifacts is absent. The paper does not systematically analyze failure cases, such as conditions where the embedding adjustment might over-regularize and weaken the distinctiveness of the personalized concept. Furthermore, the limitations of only adjusting the embedding at inference time—potentially being a "shallow" fix rather than addressing optimization issues during training—are not critically examined. A dedicated section exploring these boundaries, possibly with examples of prompts where TEA fails to mitigate semantic collapse or even degrades output, would provide a more balanced and credible evaluation.

Although the effect of hyperparameters α (rotation factor) and β (scaling factor) is briefly studied, the analysis is conducted with a fixed base model and a limited set of prompts. The paper does not explore how these parameters interact with different model architectures (e.g., varying layers of U-Net or transformer-based text encoders) or diverse concept complexities. This narrow scope leaves open questions about the robustness and generalizability of the chosen default values across a wider range of real-world use cases.

### Questions
The core operations of TEA (normalization and SLERP) are geometrically intuitive but relatively simple. Were more complex adjustment strategies explored, such as a learnable linear transformation or a small neural network? What was the rationale for choosing this specific, non-adaptive formulation? The simplicity is a strength for deployment, but it raises questions about the exhaustiveness of the exploration.

The analysis in Appendix B on multi-concept personalization is intriguing but preliminary. When two personalized concepts, V_man and V_dog, compete in a prompt, TEA is applied to which concept? Both? Does applying it to both concepts restore a better balance, or does one still dominate? This is a critical scenario for assessing SCP and TEA's utility.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces the "Semantic Collapsing Problem" in generative personalization, where a model over-focuses on a learned subject (e.g., a specific person) and neglects other contextual details in a prompt. The authors propose a simple, training-free solution called Test-time Embedding Adjustment (TEA) that corrects the subject's text embedding at inference time. This method significantly improves text-image alignment across various personalization techniques and surprisingly reveals a vulnerability in current anti-personalization defenses.

### Strengths
1.	High quality of writing and presentation.
2.	The introduction of TEA, a method that is training-free and easily transferable.
3.	An insightful analysis of the "Semantic Collapsing Problem" within generative personalization and the mechanics of anti-dreambooth methods.

### Weaknesses
1.	The paper lacks comparisons to recent, mainstream works, particularly those in the in-context generation paradigm (e.g., OminiControl[1], FLUX.1 Kontext[2], and Diffusion Self-Distillation[3]).
2.	The evaluation is insufficient. It should be strengthened by including MLLM-based benchmarks, such as DreamBench++[4].
3.	Table 1 shows a performance decrease in Reference and Image alignment scores. This is presumably because TEA's objective focuses exclusively on fidelity between the learned and original subjects, potentially neglecting other alignment aspects.
4.	The choice of SLERP is not justified. The paper provides no evidence or ablation study to demonstrate that SLERP is superior to other interpolation methods.

[1] Tan, Z., Liu, S., Yang, X., Xue, Q., & Wang, X. (2025). Ominicontrol: Minimal and universal control for diffusion transformer. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 14940-14950).
[2] Batifol, S., Blattmann, A., Boesel, F., Consul, S., Diagne, C., Dockhorn, T., ... & Smith, L. (2025). FLUX. 1 Kontext: Flow Matching for In-Context Image Generation and Editing in Latent Space. arXiv e-prints, arXiv-2506.
[3] Cai, S., Chan, E. R., Zhang, Y., Guibas, L., Wu, J., & Wetzstein, G. (2025). Diffusion self-distillation for zero-shot customized image generation. In Proceedings of the Computer Vision and Pattern Recognition Conference (pp. 18434-18443).
[4] Peng, Y., Cui, Y., Tang, H., Qi, Z., Dong, R., Bai, J., ... & Xia, S. T. (2024). Dreambench++: A human-aligned benchmark for personalized image generation. arXiv preprint arXiv:2406.16855.

### Questions
1.	How does the paper demonstrate that the method genuinely improves performance, rather than simply acting as a trade-off between instruction alignment and reference image alignment?
2.	To better understand the model's behavior, could the authors provide results for prompts that include the subject class (e.g., "a photo of V* dog"), not just the identifier "V*" alone?
3.	What is the specific effect of the prior preservation loss on the semantic collapsing phenomenon?

### Soundness
2

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
This paper pioneers the definition of the Semantic Collapsing Problem (SCP)—a long-overlooked issue where personalized tokens lose their original textual semantics and dominate complex prompts, with unconstrained optimization identified as its root cause. It then proposes Test-time Embedding Adjustment (TEA), a lightweight, training-free solution that aligns the magnitude and direction of personalized embeddings with their original semantic concepts at inference, effectively mitigating SCP. TEA demonstrates strong generality, enhancing text-image alignment across diverse personalization frameworks (e.g., Textual Inversion, DreamBooth).

### Strengths
The work proposes the Semantic Collapsing Problem (SCP) in generative personalization—an under-explored issue—and rigorously identifying unconstrained optimization as its root cause, with solid empirical evidence across textual and image spaces.
The proposed TEA method is lightweight and practical: it requires no additional training, avoids modifying model weights, and generalizes well across diverse frameworks (e.g., Textual Inversion, DreamBooth) and architectures (Stable Diffusion, Flux), making it easy to integrate into existing pipelines.
The unexpected finding that TEA mitigates Anti-DreamBooth’s adversarial corruption adds novel insights to privacy-utility tradeoffs in personalization, uncovering vulnerabilities in current anti-personalization defenses.

### Weaknesses
TEA relies on fixed hyperparameters (α=0.2, β=1.5) across all prompts, which may not be optimal for diverse scenario.

### Questions
How to determine the  rotation factor for each customization process?
Why TI+TEA and DB+TEA will degrade the customized image's alignment with reference？
why the author choose SLERP interpolation to adjust the embedding?

### Soundness
2

### Presentation
2

### Contribution
3
