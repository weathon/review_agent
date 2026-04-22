# FARI: Robust One-Step Inversion for Watermarking in Diffusion Models

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 6, 2, 6

## Abstract
Inversion-based watermarking is a promising approach to authenticate diffusion-generated images, yet practical use is bottlenecked by inversion that is both slow and error-prone. While the primary challenge in the watermarking setting is robustness against external distortions, existing approaches over-optimize internal truncation error, and because that error scales with the sampler step size, they are inherently confined to high-NFE (number of function evaluations) regimes that cannot meet the dual demands of speed and robustness. In this work, we have two key observations: (i) the inversion trajectory has markedly lower curvature than the forward generation path does, making it highly compressible and amenable to low-NFE approximation; and (ii) in inversion for watermark verification, the trade-off between speed and truncation error is less critical, since external distortions dominate the error. A faster inverter provides a dual benefit: it is not only more efficient, but it also enables end-to-end adversarial training to directly target robustness, a task that is computationally prohibitive for the original, lengthy inversion trajectories. Building on this, we propose **FARI** (**F**ast **A**symmetric **R**obust **I**nversion), a one-step inversion framework paired with lightweight adversarial LoRA fine-tuning of the denoiser for watermark extraction. While consolidation slightly increases internal error, FARI delivers large gains in both speed and robustness: with ~20 minutes of fine-tuning on a single NVIDIA RTX A6000 GPU, it surpasses 50-step DDIM inversion on watermark-verification robustness while dramatically reducing inference time.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a method for one-step robust inversion in the context of diffusion model-generated images. This method leverages LoRA fine-tuning with the insight that the inversion trajectory has lower curvature than the forward step, making it easier to reproduce with fewer steps. They specifically speed up the inversion process with minimal loss of quality for watermark extraction, demonstrating that their method outperforms the existing baseline of DDIM in both speed and quality when used with the Tree-Ring and Gaussian Shading watermarks. They also show that this improvement is maintained in adversarial settings where the images have been perturbed naturally. Finally, they study the robustness of their method to varying attack and guidance parameters and perform an ablation study over some of their hyperparameters.

### Strengths
While the process itself is not particularly original, the curvature observation appears novel. The use case of faster inversion for watermarking and how it enables better adversarial training is significant.
The results showcase a tangible improvement over the existing baseline and other adapted methods. 
The paper is well-written.

### Weaknesses
1. While the paper is well-written, it suffers from a lack of clarity around key points, such as the exact purpose of the LoRA branch, and it glosses over some important substitutions in the equations. 

2. The experimentation could benefit from an evaluation of FARI against stronger watermark removal methods [1,2], rather than only natural distortions.

3. The experiments use the metric of TPR at $10^6$ FPR, but they only evaluate with $10^3$ samples. This appears to be a mismatch, either the FPR is extrapolated or the sample size needs further description. Likewise, they are using four significant digits, even though the measurement is based on an evaluation with only 1000 samples. This would be dependent on how many bits were used per-key, but the number is not specified. Was each prompt sample used with different watermarking keys repeatedly? If so, how many? Either the FPR is meaningless, or the exact number of samples used needs clarification. These issues cast a shadow of doubt on the credibility of your results. The exact experimentation and evaluation setup needs to be clarified to justify that your results are meaningful.

[1] Zhao, Xuandong, et al. "Invisible image watermarks are provably removable using generative ai." Advances in neural information processing systems 37 (2024): 8643-8672.

[2] Lukas, Nils, et al. "Leveraging optimization for adaptive attacks on image watermarks." arXiv preprint arXiv:2309.16952 (2023).

### Questions
1. It should be clarified whether the LoRA branch is purely for robustness or is necessary for the one-step inversion process as a whole. What this means is, if we remove the LoRA parameters and their associated fine-tuning from Equation (10), does the inversion process still work? If it does not work, clarifying that LoRA is necessary for the entire process, rather than just for robustness, could be valuable.

2. For Equations (9) and (10), I have a few questions and concerns:

2.a. It is specified that $t$ is set to 0 instead of $t=T$ as one would normally do for inversion when following the piecewise linearity assumption. Rightfully, you highlight that for a one-step process, the piecewise linearity assumption does not hold, but that does not really justify your choice of $t=0$ rather than any other $t$ value, like $t=T$ or $t=1$. Also, from my understanding, $t=0$ is never used when training the denoising network (since $t=0$ is the last step), so it would be out-of-distribution. Also, you then mention that empirically, you use a small $t \sim 0$ value, but never specify what it is, how it is chosen, and why it is not just zero as you specified in Equations (9) and (10). Perhaps an ablation study over different values of $t$ could help clarify the choice.

2.b. You mention that the piecewise linearity assumption is broken, and you mention that Equation (10) is equivalent to Equation (2). Yet, in Equation (2), the $\epsilon$ component is dependent on $z_t$, but for Equation (10), you substituted it with $z_0$. This would only be possible if the piecewise linearity assumption held, allowing you to substitute with minimal error. This substitution is not explained nor justified. My intuition is that the LoRA training helps mitigate the error caused by this substitution.

3. Please clarify, either in the text or the figure caption in Figure (1), both what the unit of measurement is for curvature and how this curvature was measured. Has this been computed over one sample, one hundred samples, 1000 samples? Since this is your motivating result, it should be clear how it was derived. Showing curvature statistics across multiple seeds, model architectures, and datasets could help solidify your observation as a general one.

4. This is minor, but the figure text can sometimes be quite small to read, especially for small figures like Figure 3.

### Soundness
2

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
3

### Summary
This paper proposes a solution to the bottleneck of inversion-based watermarking in diffusion models: the diffusion inversion process is slow, unreliable, and particularly lacks robustness against external distortions. The authors' core observation is that the curvature of the DDIM inversion trajectory is significantly lower than its forward generation trajectory, making the inversion path highly compressible and suitable for approximation with very few NFEs. Based on this, the paper proposes a one-step inversion framework named FARI. The method achieves this one-step inversion through lightweight adversarial LORA fine-tuning, which unlocks end-to-end robustness training. Experiments show that with only about 20 minutes of fine-tuning, FARI's one-step inversion surpasses the robustness of the 50-step DDIM inversion baseline while dramatically increasing speed.

### Strengths
1. Novel Insight about Trajectory Curvature: The paper's main contribution is its core observation—that the DDIM inversion trajectory's curvature is significantly lower than the generation trajectory. The authors correctly identify this geometric property as a key "enabler," showing that the inversion path is inherently highly compressible and thus well-suited for one-step approximation via techniques like distillation. This is a non-obvious and highly original insight.

2. Practical Implementation: The use of lightweight LoRA for fine-tuning is a very practical and elegant solution. It requires minimal training time (i.e., ~20 minutes) and allows the "plug-in" robustness enhancements to be applied during the inversion, without affecting the original model's weights, thus perfectly preserving generation quality. This effective design is highly practical for real-world usage.

### Weaknesses
Weakness

1. Technical Novelty: While the geometric insight is novel, the methods used (e.g., trajectory distillation, adversarial training, LoRA fine-tuning) are all established techniques. The paper's contribution lies more in the clever application and combination of these existing tools rather than a fundamental breakthrough in inversion or distillation methodology itself.

2. Reliance on Unconditional Inversion: The FARI inverter is trained and executed in an unconditional setting (i.e., guidance scale = 1.0) to avoid the irreversibility issues of CFG. However, most images are generated using a much larger CFG. This mismatch between guided generation and unguided inversion is a potential source of error and a limitation on the method's generality, even if adversarial training empirically compensates for it.

3. Limited Applicability: The method and the inversion-based watermarks it relies on are fundamentally tied to ODE-based deterministic samplers (e.g., DDIM). The authors acknowledge this in the limitations section. It is incompatible with SDE samplers, which limits its universality in the broader diffusion model applications.

### Questions
Questions

1.  Regarding unconditional inversion (refer to the Weakness): If the inversion is performed unconditionally (CFG=1.0) while generation uses strong guidance (e.g., CFG=7.5). How does this unguided inversion robustly map back to the correct $z_T$? Does the adversarial training implicitly learn to "undo" the CFG effect, and have the authors explored alternatives?

2.  Regarding inversion curvature: The paper's core claim is the lower curvature of the inversion trajectory, which is attributed to "low-frequency information retained in the accumulated error." This explanation feels more like an intuitive observation than a formal analysis. Could the authors provide a more in-depth, formal theoretical analysis explaining why the inversion trajectory systematically exhibits lower curvature?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper implements a learning-based inversion for Diffusion Models using LoRA, and names the method FARI. Also, by using the fact that the inversion trajectory has a small curvature, it reduces NFE to 1. This allows performing inversion with very low runtime after a single training. The paper applies this to the Tree-ring watermark, a famous application of diffusion inversion, and achieves faster inference time and good accuracy compared to many baselines.

### Strengths
1. This work seems quite original, as this tries learning-based inversion of diffusion models, and such approches are seldom found in the existing works. (But I have a concern for this; which will be discussed in Weakness)

2. Table 1 shows that the proposed method works well on Tree-Ring Watermarks (Wen et al., 2023).

### Weaknesses
1. The proposed method uses a learning-based inversion with LoRA. I personally do not think learning-based inversion methods basically give good fidelity. This is because of the basic information difference between calculating an image from (image + noise) and calculating (image + noise) from an image. This is why many cited inversion papers use mathematical methods instead of training a new one (Wallace et al., 2023; Pan et al., 2023; Hong et al., 2024; and so on). Again, I basically think the paper's approach makes accurate noise estimation difficult. However, in the main experiment in Section 4.2, the current Tree-Ring watermark might be just too strong. This could be why FARI showed good watermark detection performance (TPR and bit accuracy) in Table 1 (I also suspect the experiment is just too easy, because all other baselines also have very high bit accuracy). But, reconstructing the noise $z_T$ well is a more fundamental goal and is more flexible for applications, but such experiments were not conducted in the manuscript. For example, if we want to put more information into the watermark than the current Tree-ring (e.g., a tree-ring watermark with high information density, like a circular barcode), then good noise reconstruction will be more important. To show this, many papers compare MSE or NMSE for noise reconstruction. Without this comparison, and based on my research experience and personal belief, it is hard to believe that FARI got good results in Table 1 because it is good at noise reconstruction. I suggest adding another table, like Table 1, that shows the round-trip MSE loss for noise reconstruction and image reconstruction. Also, it would be good to add qualitative results, not just quantitative ones. For example, pass the estimated noise $\hat{z_T}$ through the Decoder and compare it to the original $z_T$, and add error maps. Honestly, I think this new Table and Figure showing the inversion error directly is more important than the current Table 1.

2. The models used for experiments are not the latest. Rectified Flow is a more advanced diffusion model with reduced curvature, and I understand that SD3 (the successor to SD 1.5 and SD 2.1) is a Rectified Flow model. For practical use, it would be good to see experiments on models like SD3.

3. L231 states it is memory-efficient because of LoRA. But, it will use the same amount of RAM even if you train a separate network. It should be called disk-efficient because it saves some neural network weights. But saving a little disk space on SD weights is not a big advantage.

4. Overall, the font size in the figures is small. Also, some subsections are just one big paragraph, which makes it hard to read. Figure 2 could be made larger by increasing its width. For tables, there are better formats like https://github.com/jonbarron/tabilize/blob/main/tabilize.ipynb. This is used often and would be good to consider.

### Questions
If there are any changes from the original settings proposed in the Tree-rings watermark paper, please summarize them.

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
This work builds off the literature on inversion-based watermarking that use diffusion forward and backward processes to encode and then recover watermarks. The authors find a critical weakness in many watermarking approaches: the inversion process which is both weak and slow. They then propose a methodology that uses LoRA and an adversarial suite to not only make their watermarking more robust but much faster.

### Strengths
1. The approach identifies a key weakness in existing watermarking literature.
2. I believe that the approach is new to my knowledge, particularly the insights about curvature and low-NFE approximation. 
3. One step inversion is commonly done (DDIM) but the authors note that these approaches have limitations.
4. Empirical results seem promising

### Weaknesses
1. Existing works commonly use models such as SDXL-Turbo (one-step generation) to achieve faster watermarking speeds [1]. I believe that these methods may be faster and have potentially better quality.
2. We should compare against some more SOTA attacks (regeneration, combination of many attacks)
3. The concept of using an adversarial testing suite + introducing trainable parameters is not that new.
4. Requires training (weakness compared to DDIM which seems very good in its base version looking at Table 1). Given the simplicity and existing speed benefit (of DDIM single step) I find it a bit harder to motivate the use of this method.


[1] Lu, Shilin, et al. "Robust watermarking using generative priors against image editing: From benchmarking to advances." arXiv preprint arXiv:2410.18775 (2024).

### Questions
1. What is the performance relative to VINE?
2. Please try attaching Gaussian Shading + Tree-Ring with SDXL-Turbo backbone. I wonder about the robustness/speed/image quality comparison.
3. I wonder about the generalizability as we seem to test/train on a singular dataset. I think that there is a need to explore other kinds/styles of images.

### Soundness
3

### Presentation
3

### Contribution
3
