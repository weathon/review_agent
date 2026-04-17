# Diff-SSR: Diffusion Model with Structure-Modulated for Image Super-Resolution

- Decision: Reject
- Scores: 8, 6, 2, 2

## Abstract
Recent advances in diffusion models, like Stable Diffusion, have been shown to significantly improve performance in image super-resolution (SR) tasks. However, existing diffusion techniques often sample noise from just one distribution, which limits their effectiveness when dealing with complex scenes or intricate textures in different semantic areas. With the advent of the segment anything model (SAM), it has become possible to create highly detailed region masks that can improve the recovery of fine details in diffusion SR models. Despite this, incorporating SAM directly into SR models significantly increases computational demands. In this paper, we propose the Diff-SSR model, which can utilize the fine-grained structure information from SAM in the process of sampling noise to improve the image quality without additional computational cost during inference. In the process of training, we encode structural position information into the segmentation mask from SAM. Then the encoded mask is integrated into the forward diffusion process by modulating it to the sampled noise. This adjustment allows us to independently adapt the noise mean within each corresponding segmentation area. The diffusion model is trained to estimate this modulated noise. Crucially, our proposed framework does NOT change the reverse diffusion process and does NOT require SAM at inference. Experimental results demonstrate the effectiveness of our proposed method, which exhibits the fewest artifacts compared to other generated models.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The work presents Diff-SSR, a diffusion-based image SR framework that injects semantic structural guidance from SAM (Segment Anything Model) during training (only). The core idea evolves around modulating the mean of the forward noise within segmentation regions, thereby encouraging the denoising model to better respect object boundaries and structural integrity. During inference, Diff-SSR functions like one of the earliest models, SRDiff, so without SAM and no extra overhead.

### Strengths
- The idea is a conceptually clean modification to integrate semantic priors without architectural changes or test-time cost
- Consistent improvements in quantitative metrics across diverse benchmarks
- Comprehensive ablations explore the role of SAM variants
- The paper identifies a well-known gap (diffusion models often ignore semantic boundaries) and provides a plausible training-time fix

### Weaknesses
- The idea of injecting segmentation-based priors during training has precedents (e.g., StableSR, ControlNet-SR). However, the novelty lies mostly in where the SAM information is injected (noise modulation) and not what (Diff-SR does not introduce an additional conditioning).
- The claim that forward noise modulation leaves the reverse process unaffected is intuitive but needs rigorous justification.

### Questions
- Could the authors formalize why modulating the forward noise mean does not alter the statistical validity of the reverse process?
- Have the authors tried extending this to non-bicubic or real-world degradations to confirm generalization?
- Could the same benefit be obtained by simply concatenating mask features to the latent input, i.e., is the forward noise modulation itself essential?

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
5

### Summary
Previous diffusion-based image SR models perform noise sampling from a single distribution, which limits their ability to handle real-world scenes and complex textures across semantic regions. To address this challenge, the authors propose Diff-SSR, which introduces SAM-based structural positional information into the diffusion process to enhance structure-level detail restoration in SR process. The proposed model leverages fine-grained structural information from SAM during the noise sampling process to improve image quality, without incurring additional computational cost during inference. In addition, the authors evaluate and analyze the effectiveness of the proposed method across multiple datasets.

### Strengths
1. The authors ingeniously introduce SAM into the diffusion process to explore the influence of structural information on SR quality, which has been rarely considered in previous works.
2. The proposed method introduces no additional computational cost during inference and can restore super-resolved images without accessing SAM.
3. The proposed method achieves competitive results in the provided experimental results.

### Weaknesses
1. How is the coefficient $\varphi_t$ of $E_{\text{SAM}}$ in Equation (2) determined, why does its value decrease as t becomes smaller? The authors should provide the theoretical explanation of this design and provide a reasonable analysis of how varying the value of this coefficient affects the super-resolution performance. In addition, the are encouraged to provide experimental results with this coefficient set to zero to verify the performance gain of the introduction of SAM information on the reconstruction results. 
2. Is the proposed algorithm trained from scratch, or does it start from pretrained weights? The authors should clearly emphasize this point in the paper. 
3. Since all datasets used in the experimental evaluation are synthetic, the authors should provide comparative analyses on real-world SR datasets to better demonstrate the robustness and generalization capability of the proposed method.
4. Can the proposed strategy be easily transferred to other SR models to improve their performance? Will it introduce significant additional training costs?
5. Minor correction:  In Table 2, for the General100 dataset, the best FID result should be 23.56 (PiSA-SR).

### Questions
See weaknesses.

### Soundness
3

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
4

### Summary
This paper proposes Diff-SSR, a diffusion-based super-resolution framework that integrates fine-grained structure priors derived from the Segment Anything Model (SAM) into the diffusion process. The key idea is to modulate the mean of the added noise during the forward diffusion process based on structural masks encoded via a Structural Position Encoding (SPE) module. Unlike directly integrating SAM during inference, Diff-SSR aims to inject structure-level information only during training, avoiding extra inference cost. Extensive experiments on standard SR benchmarks (DIV2K, Urban100, BSDS100, Manga109, etc.) demonstrate improved structural fidelity and reduced artifacts compared to prior GAN- and diffusion-based baselines.

### Strengths
The authors re-design the diffusion model by introducing structural information. Both the training and inference process are efficient compared with the naive solution of "SAM+SRDiff", particularly of the inference process.

### Weaknesses
1. The paper is difficult to follow, primarily because the motivation for introducing structural noise is unclear. The authors argue that “different local areas of an image may exhibit distinct data distributions,” which is a reasonable observation. However, this alone does not justify why adding structural noise to the diffusion process is necessary or beneficial. A clearer theoretical or empirical explanation of how structural noise contributes to learning better representations or improving reconstruction fidelity is needed to establish the method’s motivation.

2. The proposed forward process converges to a non-zero-mean Gaussian distribution influenced by the structural mask. However, the reverse sampling process still begins from a zero-mean Gaussian distribution, which appears inconsistent and counterintuitive. The authors should provide both theoretical justification and empirical evidence to clarify this discrepancy.

3. As for the quantitative comparison, more perceptural metrics, such as LPIPS, MUSIQ, and CLIPIQA, should be reported. 

4. The paper mentions an “Artifact” metric, but its definition or calculation procedure is not provided. If this metric is computed from the error map between the predicted and ground-truth images, it might correlate strongly with PSNR and thus offer limited new insight.

5. The illustration of the forward and reverse processes in Figure 3(b) appears inconsistent with the corresponding mathematical formulations described in Section 4.1. Consistent notation and a clear visual explanation would greatly improve the paper’s readability.

6. I recommend including experiments on real-world low-resolution images to further validate the practical effectiveness of the proposed method.

### Questions
See weakness.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes Diff-SSR, a diffusion-based framework for single-image super-resolution (SR) that integrates structure-aware noise modulation guided by segmentation information from the Segment Anything Model (SAM). The authors introduce a Structural Position Encoding module that encodes positional information into segmentation masks and modulates the mean of the diffused noise in each region. This process allows the denoising network to learn structure-dependent priors without requiring SAM at inference.

### Strengths
1. The paper introduces a novel way to incorporate segmentation priors into diffusion-based SR — not by concatenating masks as conditions, but by modulating the noise distribution itself.

2. The method cleverly balances performance and efficiency by using pre-computed SAM masks only during training.

3. The paper is clearly written with helpful figures illustrating the conceptual difference between baseline diffusion, SAM integration, and the proposed method.

### Weaknesses
1. The paper lacks deeper analysis or ablation on why modulating the mean of the noise (as opposed to variance or conditional concatenation) is optimal.

2. There is no sensitivity study showing robustness to mask errors or alternative segmentation sources.

3. How the model behaves on real-world degraded inputs (like RealSR) beyond standard ×4 SR is missing.

4. In Table 2. the performance of more recent work like StructSR and PiSA-SR are significantly worse than other methods, but there is no explain about which this happen.

5. Table 2 is too small.

### Questions
1. About noise modulation design, have you compared modulating the variance or combining mean + variance instead of only the mean? Would this improve expressiveness or stability?

2. Why was Rotary Position Embedding (RoPE) chosen instead of sinusoidal or learned positional encoding? Does it significantly influence restoration quality?

3. In Section 4.3, you initialize ( x_T ) with a mean of 0 instead of ( \phi_T \cdot E_{\text{SAM}} ). I am curious whether sampling from a distribution with mean ( \phi_T \cdot E_{\text{SAM}} ) could further enhance the reconstruction quality, and if so, how much improvement it might bring.

4. In Figure 4(a), is the modulated noise ( \epsilon' ) a single-channel signal? If so, when it is added to the image as described in Equation (3), does it broadcast the same noise value across all channels?

### Soundness
2

### Presentation
3

### Contribution
2
