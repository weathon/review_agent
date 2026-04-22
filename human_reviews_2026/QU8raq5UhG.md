# Fast Text-to-Audio Generation with One-Step Sampling via Energy-Scoring and Auxiliary Contextual Representation Distillation

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Autoregressive (AR) models with diffusion heads have recently achieved strong text-to-audio performance, yet their iterative decoding and multi-step sampling process introduce high-latency issues. To address this bottleneck, we propose a one-step sampling framework that combines an energy-distance training objective with representation-level distillation. An energy-scoring head maps Gaussian noise directly to audio latents in one step, eliminating the need for a costly recursive diffusion sampling process, while distillation from a masked autoregressive (MAR) text-to-audio model preserves the strong conditioning learned during diffusion training. On the AudioCaps benchmark, our method consistently outperforms prior one-step baselines on both objective and subjective metrics while substantially narrowing the quality gap to AR diffusion systems with multi-step sampling. Compared to the state-of-the-art AR diffusion system, IMPACT, our approach achieves up to $25$× faster inference with highly competitive audio quality. These results demonstrate that combining energy-distance training with representation-level distillation provides an effective recipe for fast, high-quality text-to-audio synthesis.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a one-step text-to-audio generation framework that builds upon the IMPACT model. The main contribution lies in replacing the diffusion-based loss with an energy-distance objective and introducing a representation-level distillation process to enhance one-step generation quality. While the idea of simplifying generation through distillation is relevant and aligns with current trends in efficient diffusion or flow-matching models, the overall contribution appears incremental. Moreover, the paper’s presentation lacks clarity, making it difficult to follow the relationship between the proposed loss, model inputs and outputs, and overall architecture.

### Strengths
•  The work explores a distillation-based strategy to enhance the performance of a one-step generation framework, which could reduce computational cost and inference time.

•  The paper includes a wide range of experiments examining the influence of different distillation weights and classifier-free guidance strengths.

•  The idea of leveraging an energy-distance objective as an alternative to diffusion loss is conceptually interesting and potentially valuable for fast generation research.

### Weaknesses
•  The writing and structure of the paper need substantial improvement. The current format makes it difficult to follow the key ideas; for instance, the introduction includes too many technical details that belong to the related work or methodology sections.

•  The method section is overly complicated and lacks a clear explanation of the model architecture—no structural figure is provided, and the correspondence between equations and model components is unclear.

•  The dataset processing description is confusing. In particular, the approach to handling audio longer than 10 seconds and the captioning process for AudioSet (which contains only label data) are not well explained.

•  The subjective quality of the generated audio in the demos does not align with the quantitative results reported in the tables, and the improvements over state-of-the-art consistency-based models are not convincing.

•  The ablation study on sampling methods seems unconvincing, as existing flow-matching or consistency models should achieve comparable results with minimal sampling steps.

### Questions
•  How are captions handled when audio clips longer than 10 seconds are randomly cropped into 10-second segments?

•  How were captions generated for the 500-hour AudioSet subset, given that AudioSet only provides label-level annotations?

•  How many participants were involved in the subjective evaluation on the 90 generated samples?

•  Which specific flow-matching approach or implementation was used for the experiments reported in Table 4?

### Soundness
3

### Presentation
1

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
This paper proposes AudioDEAR, a energy scoring masked auto-regressive text-to-audio generation model.
The authors show that AudioDEAR can significantly accelerate a diffusion MAR method like IMPACT by removing the iterative de-noising procedure of IMPACT's diffusion head.
A distillation process, which enhances the learning by aligning it with the IMPACT teacher model in the transformer hidden space, further strengthens the result.
On the AudioCaps dataset, AudioDEAR balances inference speed and generation quality.

### Strengths
The proposed idea is novel, and the method is sound. The ablation study on loss function weight, whether to freeze the transformer, and number of examples for energy distance is thorough.

### Weaknesses
In the introduction, the authors claim that they are the first to apply the energy-distance objective in TTA generation, enabling one-step
latent synthesis with low latency. Based on my understanding, this statement may be misleading -- while the diffusion head is replaced with a single-step energy distance head, the overall generation process is still iterative, with 64 decoding iterations. Hence, I would disagree that the proposed method is truly "one-step", and invite the authors to clarify. That said, I agree that with a small model size, the total computation to generate each example is quite low, on par with truly single-step methods, as reflected in Figure 3.

Similarly, Table 1 and Figure 1 also refers to AudioDEAR as "one-step". They should be clarified.

Minor -- Figure 1 would be clearer if the x axis is in log scale.

### Questions
- The authors mention in Section 5.2 that the setting “freeze” denotes that the transformer backbone is initialized from IMPACT and kept frozen during training, while only the lightweight energy-scoring head is optimized. For distillation runs for which the transformer and the diffusion head are NOT freezed, are they similarly initialized with IMPACT?

- The IMPACT teacher in this paper seems to use less non-captioned training data than the original IMPACT paper, which used 5500 hours if I remembered correctly. Is there a reason for this modification?

- CFG is performed within the transformer of AudioDEAR according to section 3.2. If I remembered correctly, this contrasts with the IMPACT paper's implementation, where CFG is performed within the diffusion head. How much difference does this change bring? Does the IMPACT teacher/baseline in this paper also use transformer CFG?

- The single-diffusion-step performance of AudioDEAR is quite impressive, balancing inference speed and quality. Although one key advantage of energy-distance training is that the teacher is optional, it is unclear whether energy-distance approach can outperform consistency distillation in a distillation setting. It would be too much to ask for an experimental comparison between these two approaches, but I would appreciate some discussions on this matter.

### Soundness
3

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
3

### Summary
The paper proposes a one-step text-to-audio (TTA) generation framework that replaces multi-step diffusion with energy-distance training and representation-level distillation from a diffusion-based teacher. This approach achieves up to 25× faster inference than state-of-the-art models (IMPACT).

### Strengths
**1. Effective Integration of Energy-Distance Loss and Distillation**

The paper successfully combines energy-distance loss with representation-level distillation from a diffusion teacher, enabling one-step text-to-audio generation while maintaining strong alignment between audio fidelity and text semantics.

**2. Significant Improvement in Inference Speed**

The proposed model achieves over 25× faster inference compared to diffusion-based baselines (e.g., IMPACT), demonstrating strong efficiency.

### Weaknesses
**1. Performance Gap with Teacher Model**

Unlike AudioLCM (Make-An-Audio as teacher), the proposed method shows a noticeable performance degradation compared to its diffusion-based teacher model (IMPACT), indicating that the distillation and energy-distance training did not fully preserve the teacher’s capability.

**2. Limited Inference Advantage in Small Batches**

Although the model achieves large speedups overall, its inference latency is slower than AudioLCM when batch size = 1 or 2, suggesting that the efficiency gain primarily appears in large-batch scenarios and may not generalize well to real-time or low-latency applications.

### Questions
**1. On Distillation and Teacher Model Fidelity**

Do the authors believe that the proposed distillation and energy-distance loss can fully replicate the teacher model’s performance? If not, what aspects of the teacher are hardest to transfer through this training scheme?

**2. On Inference Efficiency in Small-Batch Settings**

The proposed model shows slower inference than AudioLCM when batch size is 1–2. Is there a way to further optimize for improving small-batch efficiency? Alternatively, is this limitation inherent to the IMPACT-based autoregressive backbone used in the model?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes replacing the diffusion head in IMPACT, which is a masked autoregressive diffusion-based text-to-audio model, with a one-step MLP generator trained by minimizing energy distance. The proposed AudioDEAR model is initialized with IMPACT, and then fine-tuned with energy distance loss and a feature matching loss between the student AudioDEAR model and the teacher IMPACT model.

### Strengths
The paper is well written, and the theoretical derivations are easy to understand. The proposed method is novel, at least for text-to-audio models.

The authors included many comparison results, including comparisons with MeanFlow and Shortcut. Although including more diffusion distillation would make it better. For example, variational score distillation methods such as DMD and adversarial distillation methods (GAN).

### Weaknesses
The audio quality sounds much worse compared to the demo audios from the IMPACT website. You should also provide a comparison between the IMPACT demo and your reproduction on the demo page.

Conducting CFG as described in Section 3.2 lacks theoretical grounds. Interpolating encoder output does not provide any guarantee on the sampled distribution. I saw [1] also introduced a different CFG variant for one-step generators. How well does it work compared to the proposed method?

MAR [2] reported that "during inference, the diffusion sampler has a decent cost of about 10% overall running time." So this raises concerns about whether eliminating the diffusion sampling from IMPACT really matters that much.

All comparisons between IMPACT and AudioDEAR are evaluated with 64 encoder steps and 100 diffusion steps for IMPACT. Is the trade-off between latency and sample quality still appealing with fewer encoder/diffusion sampling steps in IMPACT? Can you provide more analysis of the inference FLOPs as well?

[1] Efficient Speech Language Modeling via Energy Distance in Continuous Latent Space, URL: https://arxiv.org/abs/2505.13181

[2] Autoregressive Image Generation without Vector Quantization, URL: https://arxiv.org/abs/2406.11838

[3] https://audio-impact.github.io/

### Questions
When it comes to inference latency, engineering details matter. What kind of implementation are you using for testing inference latency? Did you use technologies such as FlashAttention and CUDA Graphs for accelerating inference? Is the acceleration still significant on more recent NVIDIA GPU architectures? I only see V100 evaluation results in the paper.

What is the current status of research in accelerating specifically MAR-like models?

What do you think is the main cause of existing diffusion distillation methods that work on image generation not working on masked diffusion autoregressive models such as MAR?

### Soundness
3

### Presentation
4

### Contribution
2
