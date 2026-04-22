# GenDR: Lighten Generative Detail Restoration

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 8

## Abstract
Although recent research applying text-to-image (T2I) diffusion models to real-world super-resolution (SR) has achieved remarkable progress, the misalignment of their targets leads to a suboptimal trade-off between inference speed and detail fidelity. Specifically, the T2I task requires multiple inference steps to synthesize images matching to prompts and reduces the latent dimension to lower generating difficulty. Contrariwise, SR can restore high-frequency details in fewer inference steps, but it necessitates a more reliable variational auto-encoder (VAE) to preserve input information. However, most diffusion-based SRs are multistep and use 4-channel VAEs, while existing models with 16-channel VAEs are overqualified diffusion transformers, e.g., FLUX (12B). To align the target, we present a one-step diffusion model for generative detail restoration, GenDR, distilled from a tailored diffusion model with a larger latent space. In detail, we train a new SD2.1-VAE16 (0.9B) via representation alignment to expand the latent space without increasing the model size. Regarding step distillation, we propose consistent score identity distillation (CiD) that incorporates SR task-specific loss into score distillation to leverage more SR priors and align the training target. Furthermore, we extend CiD with adversarial learning and representation alignment (CiDA) to enhance perceptual quality and accelerate training. We also polish the pipeline to achieve a more efficient inference. Experimental results demonstrate that GenDR achieves state-of-the-art performance in both quantitative metrics and visual fidelity.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes GenDR for better tradeoff between detail enhancement and inference efficiency. By utilizing a pre-trained 16-channel VAE, it expands the restoration capacity of the proposed SR model with a 0.9B UNet backbone. To reduce computational overhead, it introduces consistent score identity distillation technique to effectively train a one-step model while preserving its ability to generate vivid details. Extensive experiments demonstrate its ability on diverse scenarios.

### Strengths
* By utilizing a pre-trained 16-channel VAE, GenDR effectively expands the capacity of the proposed SR model.
* It proposes a SR-tailored step distillation technique CiDA that restore vivid details and stabilize training.
* Extensive experiments demonstrate its effectiveness on various benchmark.

### Weaknesses
* Since one of the main contributions of this paper is expanding SR capacity by a large channel VAE, more analysis about this component should be added, including the choice of the latent channel, training difficulty and reconstruction ability. 
* Results of multi-step GenDR model are absent. Although authors provide comparison between various distillation methods, comparing with the teacher model can further validate the performance of the proposed CiDA.
* While replacing the text encoder by a fixed positive embedding reduces inference overhead with slight performance degradation, results on Tab. 7 shows that slightly increasing CFG scale leads to worse performance. I am concerned that the weak textual guidance might come from using a fixed embedding. The authors should consider testing on input images that require high semantic control, such as portraits or text-containing inputs.

### Questions
* The proposed method utilizes a generator and two trainable score networks to perform the score distillation process. Thus, quantitative comparison of memory usage should be provided.
* The authors should consider providing more image results as supplemental material.

### Soundness
3

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
This paper presents GenDR, a method for efficient super-resolution (SR) that combines VAE-16 and small latent diffusion models (LDMs) using score distillation. The authors claim their approach enhances restoration quality while maintaining speed, achieving superior performance over existing methods. The paper’s main focus appears to be on improving the balance between fidelity and inference time in diffusion-based SR models.

### Strengths
1. The combination of VAE-16 with latent diffusion models, coupled with score distillation, is an interesting approach for super-resolution.
2. The experimental results demonstrate that GenDR outperforms existing models in terms of both quality and efficiency, with strong quantitative and qualitative performance across multiple benchmarks.
3. The method is designed to be efficient, offering improved restoration speed without compromising visual fidelity, which is a significant advantage in real-world applications.

### Weaknesses
1. The paper struggles with clarity and coherence, especially in how it connects its contributions. While the abstract and introduction spend significant time discussing the advantages of using VAE-16 for SR tasks, the method section shifts focus entirely to the proposed CiDA loss. This abrupt shift makes it difficult to understand the relationship between the two contributions, as they don’t seem to be closely tied. 
2. While the paper introduces CiDA as a novel technique for distilling scores, its relevance to the task of detail restoration in SR is not sufficiently discussed. The paper does not clearly explain why this loss function is needed for SR or how it helps improve the restoration of fine details. The overall discussion of CiDA feels more aligned with general diffusion model training rather than specifically addressing the SR challenge.
3. The methodology section is dense and difficult to follow. The authors jump between various components like VAE-16, CiD, and adversarial learning without fully explaining their relationships or how they work together to address the SR problem. Additionally, the notations used in the equations make the explanation harder to follow (e.g, $f(\cdot)$ and $\epsilon(\cdot)$ seem to both refer to the scores).

### Questions
See above.

### Soundness
2

### Presentation
1

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
This paper proposes GenDR, a diffusion-based model for real-world image super-resolution (SR) that aims to address the long-standing trade-off between detail fidelity and inference efficiency. The authors identify that most text-to-image (T2I) diffusion models are suboptimal for SR tasks because (1) they use low-dimensional latent spaces (typically 4-channel VAEs), and (2) they require multi-step inference to synthesize images. To overcome this, the authors develop an SD2.1 model with a 16-channel VAE backbone and a consistent score identity distillation strategy.  Empirically, GenDR achieves competitive or superior performance over state-of-the-art (SOTA) models such as DiffBIR, OSEDiff, and DreamClear, with notable improvements in LIQE, MUSIQ, and Q-Align scores and a 77 ms runtime.

### Strengths
I agree with the claim on the disadvantages of the 16-channel VAE, which probably limits the performance upper bound of diffusion-based SR methods. As illustrated in Table 2, such a modification indeed improves the SR results.

### Weaknesses
1. My main concern mainly focuses on the experimental performance. The authors claim that the proposed method is able to enhance the detail fidelity. The quantitative comparison results in Table 1 cannot support such a claim, in which GenDR does not show obvious improvements regarding the fidelity metric, such as PSNR, LPIPS. 

2. As for the visual results, I personally don't think GenDR is better than other SoTA methods, particularly in the first example in Fig. 1 and the second example in Fig. 5.

3. As for the efficiency, GenDR shows slight advantages compared with InvSR and OSEDiff. I guess such improvement is due to the removal of the text encoder. However,  this trick is also suitable for other methods (e.g., InvSR) that do not rely on dynamic textual information.

### Questions
I wonder about the effectiveness of RAPE regularization in SR task.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper introduces GenDR, a framework designed to convert a pre-trained text-to-image diffusion model (such as Stable Diffusion) into a compact model for real-world image super-resolution. The approach uses a 16-channel VAE to retain more detailed image representations and applies CiDA  to align the training objective with the super-resolution task. GenDR produces results in a single inference step and shows improved performance over several baseline methods in terms of perceptual quality and runtime efficiency on benchmark datasets.

### Strengths
- The paper addresses a practical limitation of the commonly used f8c4 VAE in SR tasks by expanding it to 16 channels, which helps preserve more details. 

- The proposed CiDA framework is a well-engineered solution that enables stable training and efficient inference even in a higher-dimensional latent space.

- The experiments are thorough, covering model size, inference speed, ablations on CiDA, prompt simplification, and performance after SR-specific fine-tuning. These results support both the effectiveness and practical relevance of the method.

### Weaknesses
- In Table 2, the paper only reports changes in no-reference metrics, which do reflect perceptual quality improvements, but it lacks reference-based metrics like LPIPS. Including trends in LPIPS (even if degraded) would provide a more complete picture, especially considering the known trade-offs between no-reference and reference-based measures. Visualization comparisons under different loss functions would also strengthen the argument.

- It remains unclear why the model, after being fine-tuned for SR on a 512-resolution UNet, can still generate coherent 1024-resolution T2I results without typical artifacts (e.g., extra limbs). Some clarification on how the SR tuning impacts generalization to higher-resolution T2I tasks would be helpful.

### Questions
I suggest the authors include a brief comparison with TVTSR [1] (ICCV 2025) in the main text. While the motivation is aligned, the two methods adopt different and complementary strategies: the proposed approach keeps the spatial compression ratio fixed but increases the channel dimension by 4×, whereas TVTSR maintains the channel size and compresses the spatial resolution by 2×. A short discussion of these orthogonal design choices would help clarify the novelty and positioning of the proposed method.

[1] https://arxiv.org/pdf/2507.20291

### Soundness
3

### Presentation
3

### Contribution
3
