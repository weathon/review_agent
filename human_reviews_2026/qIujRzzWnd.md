# Generation then Reconstruction: Accelerating Masked Autoregressive Models via Two-Stage Sampling

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 8, 4, 6

## Abstract
Masked Autoregressive (MAR) models promise better efficiency in visual generation than continuous autoregressive (AR) models for the ability of parallel generation, yet their acceleration potential remains constrained by the modeling complexity of spatially correlated visual tokens in a single step. To address this limitation, we introduce Generation then Reconstruction (GtR), a training-free hierarchical sampling strategy that decomposes generation into two stages: structure generation establishing global semantic scaffolding, followed by detail reconstruction efficiently completing remaining tokens. Assuming that it is more difficult to create an image from scratch than to complement images based on a basic image framework, GtR is designed to achieve acceleration by computing the reconstruction stage quickly while maintaining the generation quality by computing the generation stage slowly. Moreover, observing that tokens on the details of an image often carry more semantic information than tokens in the salient regions, we further propose Frequency-Weighted Token Selection (FTS) to offer more computation budget to tokens on image details, which are localized based on the energy of high frequency information. Extensive experiments on ImageNet class-conditional and text-to-image generation demonstrate 3.72X speedup on MAR-H while maintaining comparable quality (e.g., FID: 1.59, IS: 304.4 vs. original 1.59, 299.1), substantially outperforming existing acceleration methods across various model scales and generation tasks.
Our codes will be released in https://github.com/feihongyan1/GtR.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Based on the two observations: (1) the spatially adjacent tokens tend to influence each other; and (2) covering more spatial locations indicates creating more information, this paper proposes Generation-then-Reconstruction (GtR), a novel sampling strategy to accelerate image generation in Masked Autoregressive (MAR) models, which decomposes generation into two stages: structure generation establishing global semantic scaffolding, followed by detail reconstruction efficiently completing remaining tokens. It also proposes Frequency Weighted Token Selection (FTS) to offer more computation budget to tokens on image details. Extensive experiments on ImageNet class-conditional generation and text-to-image synthesis demonstrate that GtR achieves significant speedups while maintaining competitive or superior image quality.

### Strengths
1.	The "generation-then-reconstruction" paradigm is conceptually novel, which is a training-free hierarchical sampling strategy and expands the mainstream random/sequential generation mode.
2.	The proposed Frequency-Weighted Token Selection (FTS), a training-free strategy that could allocate more diffusion steps to the tokens with more details.
3.	Compared with other accelerated model, the proposed method achieves faster inference speedup for MAR while maintaining comparable quality, which better balance between generation speed and generation quality.

### Weaknesses
1.	Some of the organization and arguments in the paper are problematic: a). In line 013, the MAR is the Masked Autoregressive (MAR) models from [1] , but it is not clear whether AR specifically refers to a discrete autoregressive model or a continuous autoregressive model. If it is discrete autoregressive model, the inference efficiency of MAR is lower than AR; b). In line 134, the xAR should not be classified as Next-token autoregressive visual generation, because more than one token is generated at each step.
2.	Although the specific division stages and the average generation rates for each stage are described in section 4.1, the number of inference steps of each stage is unclear. And it is not clear why it is set up like this.
3.	The generated tokens in the generation stage satisfied the condition (i+j) mod 2 = 0 in all experiments. Lack the ablation study about setting the condition (i+j) mod 2 = 1. In addition, lack the ablation study about β.
4.	Inconsistent writing format in section 3.1. “Image Generation as Next-Token Prediction.” and “Image generation as next-set prediction” .

[1] Autoregressive image generation without vector quantization.

### Questions
1.	The paper uses K=3 for MAR and K=4 for LightGen. Is this choice empirical or theoretically motivated? How does K affect the quality-speed trade-off?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes Generation-then-Reconstruction (GtR), a training-free, two-stage sampling strategy to accelerate Masked Autoregressive (MAR) visual generation. GtR first generates a spatially non-adjacent “skeleton” of tokens (checkerboard / conservative generation stage) to establish global semantic structure, then reconstructs the remaining tokens in very few highly-parallel steps. The authors also introduce Frequency-Weighted Token Selection (FTS) to allocate extra diffusion steps to tokens with high high-frequency energy (i.e., fine details). On ImageNet class-conditional and text-to-image setups the paper reports ~3.72× speedup for MAR-H with comparable FID/IS. The method is presented as training-free and applicable to existing MAR pipelines.

### Strengths
- **Simple, practical idea with strong engineering value.** Splitting generation into “creation” (structure) and “reconstruction” (detail) is intuitive and requires no model retraining, which makes it immediately usable in deployed MAR systems.
- **Clear empirical speed/quality benefit.** The reported ≈3.72× acceleration on MAR-H while maintaining generation quality is compelling evidence of practical value.
- **Ablations and comparisons.** The paper contains ablations across sampling orders and token selection strategies (Raster/Subsample/Random/GtR and High-Freq/Random/Full-Enhanced), which strengthens the empirical claims.

### Weaknesses
- **Limited analysis of failure modes and generality.** The paper demonstrates gains on ImageNet and a text→image generator, but it is unclear how GtR behaves with different tokenizers (VQ variants, continuous latents), non-square layouts, or with very detail-dense images where global structure is weak. The paper makes the empirical claim that reconstruction is “much easier than creation” but does not deeply analyze where this breaks down.
- **Limited theoretical justification for checkerboard schedule.** The checkerboard / stage partition is intuitively motivated (reduce adjacent simultaneous predictions) but lacks theoretical analysis or simple metrics demonstrating why it is near-optimal vs. other structured partitions. The ablations show GtR > Random, but more insight would help adoption.

### Questions
- The paper report “3.72× speedup”, but it is not clear whether this is measured as (a) total wall-clock on standard hardware, (b) encoder FLOPs only, (c) decoder/diffusion FLOPs, or (d) a mix. A breakdown (encoder vs diffusion vs IO) and experiments on multiple hardware targets would increase reproducibility and confidence.
- I recommend conducting a Hyperparameter sensitivity study, including experiments that sweep K, Trec, β (FTS percentile), and Tdetail to show how sensitive quality and speed are. A small table/heatmap showing FID vs speed for several settings would help practitioners choose defaults.

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
4

### Summary
This paper proposed a two-stage sampling strategy (Generation then Reconstruction, GtR) for MAR. The goal is to sample images with fewer autoregressive steps while maintaining comparable quality. Specifically, the generation stage samples image tokens in a checkerboard sequence, using more steps to establish global semantic scaffolding and structure. The reconstruction stage then samples the remaining tokens with fewer steps to fill in the details. Additionally, the paper introduces a Frequency-Weighted Token Selection strategy for choosing diffusion steps. Experimental results demonstrate that this GtR strategy can accelerate sampling speed.

### Strengths
1. The proposed sampling strategy is effective and intuitively easy to understand.

2. The paper is clearly written.

3. The performance in ImageNet-256x256 is impressive.

### Weaknesses
1. Limited Generalizability: The method's generalizability is a significant concern. While it performs well on ImageNet, its performance drops considerably on the T2I task (as shown in Table 2). This suggests that the proposed sampling strategy might be an overfitted, heavily-tuned solution for the ImageNet dataset and may not generalize well to other datasets or tasks. Moreover, the proposed strategy's applicability seems limited, as it is designed specifically for the MAR.

2. The novelty of the work is limited. The core concept of establishing a global structure before refining details ("global-to-detail") has been explored in prior work like MUSE[1] and Hi-MAR[2] (missing reference and comparison). The paper's main contribution appears to be adapting this concept specifically to the sampling process of MAR.

3. Insufficient Experiments: The paper only reports results for a very sparse set of sampling steps (e.g., steps 8 and 16 for C2I; steps 12 and 16 for T2I). It is unclear why step 8 was omitted for the T2I task, and the performance at other step counts remains unknown. The authors should provide the curves (e.g., FID vs. Sampling Steps) to show the full performance trend. Moreover, there are no trade-off curves to analyze the FID vs. speed when: (a) varying the step of the generation and reconstruction stages, (b) varying the autoregressive steps and the diffusion steps.

4. Do not provide an introduction to the baseline methods (Halton, DiSA, LazyMAR).

I am open to raising my score if the authors can address these concerns in their rebuttal.

[1]Chang, Huiwen, et al. "Muse: Text-to-image generation via masked generative transformers." ICML 2023.

[2]Zheng, Guangting, et al. "Hierarchical Masked Autoregressive Models with Low-Resolution Token Pivots." ICML 2025.

### Questions
Please refer to weaknesses.

### Soundness
3

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
4

### Summary
The authors identify some issues with MAR and related frameworks, particularly related to modeling potentially correlated tokens as independent when denoising them parallely. On this hypothesis, they suggest a two step generation methodology - first generate tokens in a checkerboard like pattern so as to avoid adjacent tokens being unmasked parallely, and once this provides sufficient context, reconstruct the rest of the tokens quickly in 1-2 steps, conditioned on the previously generated tokens. This leads to inference time accelerations as suggested by the empirical results, with minimal or no degradation.

### Strengths
- The authors present a comprehensive review of the literature in the context of visual generation, and identify shortcomings, proposing a sound method to address these limitations
- The two stage generation pipeline is a fairly novel contribution, utilizing ideas from previous works on token ordering
- The inference time speedups in terms of real time latency is promising
- The authors perform a detailed set of ablations, explaining their design choices

### Weaknesses
- The key hypothesis of this method is that the checkerboard-like pattern would lead to good global context and therefore more coherent generation. However, it is possible for images to have structures that extend to larger regions of the image, thus still potentially producing inconsistencies in generation. A deeper discussion on this point would be helpful
- Adding on to the previous point, since the checkerboard pattern followed by reconstruction is meant to produce more coherent generations, this approach should have led to generations with better fidelity, alongside latency speedups. However, the empirical results suggest neutral quality and only latency speedups when compared to corresponding baselines. It would be helpful to see the generations with highest FID to inspect and analyse the shortcomings of the proposed method

### Questions
Please see the weaknesses section above

### Soundness
3

### Presentation
3

### Contribution
2
