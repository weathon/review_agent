# Latent Wavelet Diffusion For Ultra High-Resolution Image Synthesis

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 4

## Abstract
High-resolution image synthesis remains a core challenge in generative modeling, particularly in balancing computational efficiency with the preservation of fine-grained visual detail. We present $\textit{Latent Wavelet Diffusion (LWD)}$, a lightweight training framework that significantly improves detail and texture fidelity in ultra-high-resolution (2K-4K) image synthesis. LWD introduces a novel, frequency-aware masking strategy derived from wavelet energy maps, which dynamically focuses the training process on detail-rich regions of the latent space. This is complemented by a scale-consistent VAE objective to ensure high spectral fidelity. The primary advantage of our approach is its efficiency: LWD requires no architectural modifications and adds zero additional cost during inference, making it a practical solution for scaling existing models. Across multiple strong baselines, LWD consistently improves perceptual quality and FID scores, demonstrating the power of signal-driven supervision as a principled and efficient path toward high-resolution generative modeling.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Latent Wavelet Diffusion (LWD), a novel training framework designed to enhance the quality of ultra-high-resolution (UHR) image synthesis (2K-4K) from latent diffusion models. The authors identify a key limitation in existing methods: a uniform training process that fails to distinguish between low-detail and high-detail regions, leading to wasted computation and a loss of fine-grained texture.

LWD addresses this by introducing a signal-driven, frequency-aware supervision strategy. The framework consists of two main stages. First, a pre-trained Variational Autoencoder (VAE) is fine-tuned with a scale-consistent spectral objective to produce a more stable and spectrally regular latent space. Second, during the fine-tuning of a latent diffusion model, LWD computes wavelet energy maps from the latent codes at each step. These maps are used to create a time-dependent spatial mask, which dynamically modulates the training loss to focus more intensely on detail-rich (high-frequency) regions.

### Strengths
LWD is a training-only strategy, requiring no architectural changes to the diffusion model and adding zero computational overhead during inference.

### Weaknesses
1 The paper called the proposed VAE is a spectrally-aware VAE, where no spectral objective is involved. In my opinion, it is somewhat overclaimed.

2 Figure 3 is hard to read. For example, what does DCT mean?

3 The title is Latent Wavelet Diffusion. However, in the paper, the wavelet only proposes a mask for RGB-based diffusion training. I do not 
agree that this model is a wavelet diffusion model, and worry about the contribution&novelty of this paper.

4 The improvements in Table 1&2 are limited.

### Questions
See the weakness above.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Latent Wavelet Diffusion (LWD), a method aimed at improving ultra-high-resolution image synthesis (2K-4K) using latent diffusion models. The authors propose a frequency-aware masking strategy based on wavelet energy maps, which helps focus the training process on detail-rich regions in the latent space. The approach also incorporates a scale-consistent VAE objective to ensure high spectral fidelity. The key advantage of LWD is that it enhances image quality and detail without requiring architectural modifications or additional inference costs. The paper provides experimental results across multiple datasets and compares LWD with several state-of-the-art models.

### Strengths
1. The method is notably efficient, both in training and inference, as it does not require any changes to the underlying model architecture or introduce additional computational cost during inference.
2. The approach results in noticeable improvements in key image fidelity metrics such as FID and LPIPS, with a demonstrated enhancement in texture and detail preservation.
3. The frequency-aware saliency map and time-dependent masking strategy offer a interpretable way to focus model attention on high-frequency regions.

### Weaknesses
1. While the method shows improvements in several fidelity metrics, there is a noticeable performance drop on certain tasks, such as PickScore and HPSv2.1. These drops suggest potential degradation in text-image alignment, which could impact the quality of the generated images, as seen in the generated Eiffel Tower in Figure 4. The seasonal inconsistency in the image (the LWD + URAE version of the Eiffel Tower showing a different season than the original) further indicates this issue.
2.  The paper discusses the VAE fine-tuning with scale-consistency loss, but the contribution of this step to the overall performance is not adequately ablated or isolated. Without a clear comparison, it is difficult to assess how much this component adds to the method’s success.
3. The paper introduces a saliency map generated using wavelet decomposition, but it does not provide an ablation study to verify whether this saliency map is actually contributing to the improvements in performance. An experiment comparing results with and without the saliency map would help clarify whether this component is functioning as intended and contributing meaningfully to the enhancement of fine details.

### Questions
See above.

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
This paper proposes Latent Wavelet Diffusion (LWD), a frequency-aware modification of diffusion training that enables existing architectures to generate high-resolution samples without architectural changes or excessive sampling cost. The authors use a multiscale VAE training objective together with a wavelet-based diffusion training loss, extending both the latent space and the diffusion process to better address the unique challenges of high-resolution generation. The VAE training ensures that the latent space exhibits well-structured frequency characteristics, while the frequency-aware loss encourages the model to allocate more training capacity to high-frequency regions of the input. Experiments demonstrate that LWD achieves higher-quality generation at high resolutions compared to baseline methods.

### Strengths
- This paper leverages existing architectures for high-resolution generation with minimal training overhead. Given the importance of high-resolution generation in many applications, the proposed method could have significant practical impact.

- The use of wavelets to modulate the loss function during diffusion model training is, in my opinion, both novel and interesting.

- The method also fine-tunes the VAE component, which appears crucial for maintaining a well-structured latent space at higher resolutions.

- The experimental results demonstrate the benefits of the proposed approach, at least to some extent.

- Overall, the paper is well written and easy to follow.

### Weaknesses
- The quantitative and qualitative results presented in the paper are somewhat confusing, in my opinion. While the qualitative results clearly show advantages of LWD + URAE, the quantitative metrics—particularly HPSv2, which I believe aligns better with human perception than FID—indicate worse performance. Conducting a user study and providing more qualitative comparisons between LWD and existing baselines would strengthen the paper’s contributions.

- The claim regarding identical inference time is somewhat misleading. The inference cost of diffusion models also depends on the input resolution. As discussed in the HiDiffusion paper, generating high-resolution images with models like SDXL is increasingly challenging due to the computational demands at higher resolutions. The authors should clarify this point and perform a more balanced evaluation that demonstrates the inference-time limitations of running large models (e.g., Flux) at higher resolutions.

- The proposed multiscale VAE loss is not entirely novel, as similar formulations have been introduced in EQ-VAE and other prior works. The authors should revise the claim suggesting this is their contribution.

**Minor**:
- There appears to be a typo in Equation (1).

### Questions
1. Can you explain Figure 3? To me, SD3-Med has the closest spectrum to the RGB image as far as the plot shows. How does SE tuning help in this case?

2. Can you report HPS scores for Table 2? It might be more reliable compared to other metrics such as FID for text-to-image generation

3. Have you tried training the model only on low-frequencies first, and then shifting the attention more and more toward high-frequency details as the training progresses? The current version has high-frequency training for all steps.

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
This paper proposes to mask the diffusion loss to favor areas where details lie.
A masking weight matrix $A_\text{wavelet}$ is initially computed from the input image using a 2x2 wavelet decomposition (only Haar wavelets appears to be mentioned).
A mask $M_t$ is computed at every training by setting $1$ at every location where $A_\text{wavelet} + l \geq t/T$, here $t$ is current training step out of a total of $T$.
A masked loss is then computed by multiplying the score error with $M_t$, effectively zero-ing gradients in low-detail areas.

In addition the paper makes use of existing methods for scale-consistency VAE fine-tuning to further improve results and help the conditioning of the latents provided to their core method.

### Strengths
1. The idea is simple and easy to implement, it's reminding me of past ideas for super-resolution where people used edge detection maps as an extra signal for learning. 
2. The paper is well written and easy to follow.
3. The method does not incur any extra training costs or inference cost other than compute the wavelet matrix which is insignificant.

### Weaknesses
1. Notations can be confusing: in some settings $t$ is the current training step (eq.6) while in other settings $t$ is the noise level / diffusion time step conditioner (eq 4, 5). 
2. What happens with other wavelets, or even DCT or FFT which can fulfill similar roles as Haar wavelets? (you mention in the appendix that Haar are the best suited, yet I'd still want experimental confirmation).
3. Your method has two component (1) scale-consistency VAE fine-tuning (from existing works) and (2) frequential-energy masked loss, what's the effect on the evaluation metrics of each component?
4. The $l$ ablation and other ablations such as scale-consistency should be in the main paper, not in the appendix.
5. Some big images such as Fig.1 could be moved to the appendix to make space for scientific content like the aforementioned the ablations and wavelets comparisons that are currently missing.
6. Experimental results feel inconclusive: sometimes there is improvement, sometimes degradation. There are lot of metrics being reported in tables 1 and 2, it's unclear whether some are more important than others. Just to explain further my point of view: 2.5 FID points reduction from 35.25 to 32.88 does not feel significant because the FID is still very far from 0.

### Questions
1. From Fig.2 I understand the map gets normalized globally, can that impact areas of details with less contrast?
2. The mask as described progressively vanishes, having more and more 0 as training progresses. What mechanism ensures that no (catastrophic) forgetting is happening for the masked-out points? Does it rely entirely on $l$ being large enough?
3. In Eq.3 what's the purpose of $1/C$ if $E$ is min-max normalized after?
4. Have you tried the RMS amplitude $\sqrt{E(i,j)}$ instead of $E$, given your curriculum is a linear thresholding?
5. In tables 1 and 2, when comparing to the baselines, do these baselines benefit from scale-consistency VAE fine-tuning?
6. The scale-consistency VAE fine-tuning seems to originate from previous papers, so I assume it would make sense to make baselines benefit from it unless you consider the scale-consistency fine-tuning part to be your own contribution. Please clarify this.

### Soundness
3

### Presentation
3

### Contribution
2
