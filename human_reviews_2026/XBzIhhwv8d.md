# Neodragon: Mobile Video Generation Using Diffusion Transformer

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 6, 4, 6

## Abstract
We propose Neodragon, a video DiT (Diffusion Transformer) designed to run on a low-power NPU present in devices such as phones and laptop computers. We demonstrate that, despite video transformers' huge memory and compute cost, mobile devices can run these models when carefully optimised for efficiency. To achieve this level of efficiency, i) we replace the original large Text-Encoder with a much smaller one with minimal quality loss through our novel distillation framework which doesn’t require any image or video data. ii) We propose an Asymmetric Decoder distillation approach which allows us to replace the native codec-latent-VAE decoder with a more efficient one, without disturbing the generative latent-space of the video generation pipeline. iii) With our Block Pruning strategy, we remove entire blocks from the MMDiT denoiser based on their relative importance and recover original performance through a two-stage distillation process. iv) We reduce the diffusion sampling cost using our novel extended version of DMD (Distribution Matching Distillation) for the Pyramidal Flow-Matching objective. Neodragon generates 49 frames of [640$\times$1024] resolution within 6.7 seconds on the Qualcomm Hexagon NPU with a VBench total score of 81.61, setting a new state-of-the-art for mobile video generation.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes Neodragon, an efficient on-device video diffusion transformer designed to run entirely on a mobile NPU. The method starts from the Pyramidal-Flow framework and introduces four complementary optimizations:
1. Text-Encoder Distillation: Compresses the large T5-XXL encoder into a lightweight “DT5 + ContextAdapter” model with minimal quality loss.
2. Asymmetric Decoder Distillation: Replaces the original heavy VAE decoder with a much smaller decoder transferred from other models, keeping the encoder and DiT frozen.
3. MMDiT Block Pruning: Removes less important blocks from the diffusion transformer while maintaining fidelity through two-stage fine-tuning.
4. Step Distillation for Pyramidal-Flow: Reduces the number of denoising steps from 480 to 21 using an extended Distribution Matching Distillation (DMD) strategy.

These combined optimizations enable real-time text-to-video generation on the Qualcomm Hexagon NPU, setting a new state of the art for mobile video generation.

### Strengths
1. **Well-structured and practical problem analysis:** I appreciate the way the paper systematically analyzes bottlenecks in the video diffusion pipeline. It first identifies that memory (not compute) is the primary constraint, and addresses this through text encoder and decoder distillation. Once memory becomes manageable, the authors shift focus to speed via block pruning and denoising step distillation. This staged approach shows deep understanding of real deployment challenges and practical prioritization.
2. **Targeted and effective design choices.** Each optimization directly addresses a critical performance bottleneck—encoder size, latent reconstruction cost, model depth, and step count—without losing model compatibility. The work shows excellent engineering sense in solving exactly the key part at each stage.
3. **Comprehensive ablation studies.** Nearly every design component is supported by specific ablations. For example, Figure 1 (b) compares different text encoder distilltation method, Table 1 compares multiple decoder architectures under asymmetric fine-tuning; Table 2 analyzes MMDiT pruning ratios; and Table 3 reports performance across step distillation configurations. This makes the conclusions well supported and convincing.
4. **Rich visual evidence.** The paper provides abundant qualitative results, both in the main paper and in the appendix, including frame-by-frame comparisons and long video samples. These visuals clearly demonstrate that the model preserves temporal consistency and visual quality even after aggressive compression.

### Weaknesses
1. Although the experiment is rich and supportive, the setting of experiment is sometimes not clearly stated, For example, I do not realize what base model the paper use before a second read. Second example is that I was confused for a while about what "Neodragon E2E" and "T2V Multi-Step" are, and later find in line 470 and 471 what these mean. The author is encouraged to add a specific section to clarify the setting of experiment conducted throughout the paper and I believe this would greatly increase the clarity of the paper. Possibly this is due to limited spaces. Also, some table/figure (like table 4) is not referred in the text. 
2. Lack of NPU Implementation Detail. While the paper reports on-device latency and memory numbers, it omits critical information such as hardware numbers (memory, flops), runtime environment, or hardware-specific optimizations.

### Questions
1. The paper perform full-stack, end-to-end optimization on pyramid-flow model. I wonder whether the similar techiques like can be applied--or have been tried--on other video diffusion models such as Wan or CogVideoX, which target consumer-level GPUs. Would the authors expect comparable performance gains or face new challenges due to architectural differences?
2. I am experienced in CUDA programming and are curious about NPU programming. Could the author compare the differences or GPU and NPU programming and provide some implementation insight?
3. The step distillation process seems to offer much greater acceleration than block pruning. Can the author provide some insights in how to combine step distillation and block pruning to achieve best speed accuracy tradeoff? Does the speed gain of block pruning worh the accuracy loss?

### Soundness
4

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
4

### Summary
This paper proposed Neogradon, a video DiT designed to run on a low-power NPU present in devices. It designed a lightweight text encoder, video decoder, pruned diffusion transformer, and step distillation method for applying Distribution Matching Distillation for the Pyramidal Flow-Matching objective. Experiments shown that Neodragon generates 49 frames of 640×1024 resolution videos within 7.6 seconds on the Qualcomm Hexagon NPU with a VBench total score of 81.61.

### Strengths
1. The motivation is clear and meaningful as a on-device design of video generation model are important for phones or laptops.

2. The experiments are thorough, providing a detailed exploration of various compression techniques.

3. The paper has adapted various components in the video generation model on the edge-device, which is a very systematic project.

4. The proposed model has good performance and can achieve advanced generation performance on the edge-device.

### Weaknesses
1. The display of some relational data in the paper is not intuitive, such as some key metrics for end-to-end deployment such as memory consumption and latency, which are only explained in text. Providing some tables or charts would be more obvious.

2. The paper should provide clearer references to some of the baseline choices made in the appendix regarding the baseline method, such as why Pyramidal Flow was chosen.

3. More efficiency should be reported, such as memory consumption and latency of other comparison methods.

4. The balance between the performance and efficiency of different proposed components on the final model should be reported.

### Questions
Please see above weaknesses.

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
The paper proposes Neogradon, a video DiT (Diffusion Transformer) designed to run on a low-power NPU present in devices such as phones and laptop computers. It reduces the diffusion sampling cost using their novel extended version of DMD (Distribution Matching Distillation) for the Pyramidal Flow-Matching objective.

### Strengths
1. The paper successfully transforms a large, server-side Diffusion Transformer into a highly efficient, deployable mobile solution.
2. The paper is well-structured, featuring a data-driven abstract and logical presentation of the methodology. 
3. The paper provides a finding that the Cosine Distance loss is indispensable for stabilizing the text encoder distillation process. This highlights the crucial role of preserving the directional coherence of text embeddings for the downstream attention mechanisms.

### Weaknesses
1. The final performance claim (VBench 81.61) is achieved by a hybrid pipeline that uses external models (SSD-IB for high-quality first-frame initialization and QuickSRNet for super-resolution) to compensate for artifacts caused by aggressive step distillation in the core model.
2. The key assumptions underpinning the core compression strategies, such as the "universality of compressed video latent space" and the concept of "similarly shallow semantic demands" for large language models, are supported primarily by empirical results rather than formal theoretical proofs.
3. The paper lacks critical quantitative data, such as comparisons of NPU peak memory usage and real-time decoding latency, to fully justify the final selection of certain components (e.g., choosing the TinyAEHV decoder despite its lower PSNR).
4. The necessity of the two-stage curriculum used for MMDiT block pruning is only proven empirically (i.e., direct Stage 2 fails), but a definitive theoretical or optimization-landscape explanation for why this two-step curriculum is mandatory is missing.
5. The block importance metric used for MMDiT pruning is insufficient on its own and requires the inclusion of subjective visual impact assessment to determine the final blocks to remove, suggesting the need for a more comprehensive, objective metric.

### Questions
For specific issues, please refer to the points listed in the Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper proposes a practical recipe for on-device text-to-video generation built on an MMDiT backbone, targeting Qualcomm Hexagon NPU. The framework includes four techniques: 1) Text encoder distillation: replace the original 5B T5-XXL text encoder with a tiny 130M distillT5 + context adapter. 2) Asymmetric VAE decoder distillation with only a 10M parameter model for efficient latent decoding. 3) block pruning for MMDiT by analyzing block importance to reduce parameters for MMDiT, and 4) extend the distribution matching distillation to the pyramidal flow-matching setting.

### Strengths
- The paper presents a clear, step-by-step framework for on-device video generation with competitive Vbench.
- Text-only distillation for the text-encoder is well-explained and practically motivated.
- The step-distillation for pyramidal model adaptation is novel and empirically effective under 4‑4‑4 Tab.3, with a dqualitative result of tradeoff.

### Weaknesses
- The paper does not include a user study or human evaluation of the generated videos, which would strengthen the perceptual quality claims.
- The hardware/deployment setup is under-specified — key details such as peak memory, per-module latency, data types/quantization, and runtime environment are missing.
- The claim that the visual and textual importance of a block are uncorrelated is interesting, but currently under-supported; additional evidence/analysis would make this argument more convincing.

### Questions
- Do you have a latency breakdown for each module of the pipeline?
- Have you conducted a user study or human evaluation to compare the base model with the smaller/pruned model?
- Is there an ablation study showing the effectiveness of block pruning based on importance scores?
- typos:
  - `neogradon` -> `neodragon` in the abstract.

### Soundness
3

### Presentation
2

### Contribution
3
