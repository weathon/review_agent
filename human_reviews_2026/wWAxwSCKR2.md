# Lumos-1: On Autoregressive Video Generation with Discrete Diffusion from a Unified Model Perspective

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4

## Abstract
Autoregressive large language models (LLMs) have unified a vast range of language tasks, inspiring preliminary efforts in autoregressive (AR) video generation. Existing AR video generators either diverge from standard LLM architectures, depend on bulky external text encoders, or incur prohibitive latency due to next-token decoding. In this paper, we introduce Lumos-1, an LLM-based unified model for AR video generation with efficient discrete diffusion. Firstly, to fit videos with LLMs, we identify that 1D RoPE is ill-suited for visual spatiotemporal correlation modeling, and while demonstrated to be useful, naive 3D RoPE exhibits imbalanced frequency spectra. Therefore, we propose MM‑RoPE, which preserves the original textual RoPE while seamlessly accommodating video data with comprehensive frequency spectra and scaled 3D positions. Secondly, to fit the video data's nature and overcome the inefficiency of next-token decoding, we adopt a parallel and mask-based discrete diffusion with the intra-frame bidirectional and inter-frame causal attention masks. Based on this attention mask, we uncover the frame‑wise loss imbalance issue caused by spatial information redundancy and propose Autoregressive Discrete Diffusion Forcing, which introduces temporal tube masking during training with a compatible inference‑time masking policy to avoid quality degradation. Despite using only 48 GPUs for pre-training, limited data and a discrete tokenizer, Lumos-1 achieves results surpassing those of Show-o2 on GenEval, COSMOS-Video2World on VBench-I2V, and OpenSoraPlan on VBench-T2V.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces Lumos-1, a unified, LLM-based autoregressive video generator that addresses the central challenge of spatiotemporal positional encoding. To align with the dual nature of video (i.e., spatial bidirectionality and temporal causality), the authors propose two key contributions: 1) MM-RoPE: a distributed, scaled 3D Rotary Position Embedding that preserves standard textual RoPE while interleaving temporal, height, and width channels in repeated "meta" groups to broaden the frequency spectrum. It further scales latent 3D positions by the tokenizer's compression ratio to harmonize visual–text rotary rates, correcting the frequency imbalance inherent in naïve 3D RoPE. AND 2) AR-DF: a mask-based discrete diffusion scheme that replaces next-token decoding under an intra-frame bidirectional and inter-frame causal attention mask. During training, temporal tube masking mitigates late-frame loss collapse caused by spatial redundancy. At inference, consistent partial-context masking combined with KV caching stabilizes quality and motion, outperforming random masking, loss reweighting, and diffusion-forcing baselines.

Lumos-1 is trained on 60M images and 10M videos using 48 GPUs, which achieves competitive text-to-image performance on GenEval compared to diffusion and AR models, and produces image-to-video and text-to-video results on VBench comparable to COSMOS-Video2World and OpenSoraPlan/LTX-Video, despite substantially lower compute and data budgets. The paper also presents extensive, thorough ablation studies.

### Strengths
1. The introduction of MM-RoPE is both clear and rigorous. It lays out a coherent, step-by-step narrative: from RoPE fundamentals to naive 3D extensions, diagnosing frequency-spectrum imbalance, and culminating in MM-RoPE, so readers see exactly why each design choice is made. The claims are substantiated by targeted experiments and diagnostics, including validation-loss comparisons across vanilla RoPE, Scheme 1, Scheme 2, and M-RoPE, as well as visual analyses of frequency allocation and rotary speeds that reveal the imbalance.

2. Extensive and convincing ablations: 1) Efficacy of MM-RoPE (showing distribution is primary and scaling complementary); 2) Effectiveness of temporal-tube masks during AR-DF training; 3) Effect of AR-DF inference masks. The "Inference time analysis" suggests substantial speedups for mask-based discrete diffusion with KV caching over next-token decoding.

3. By retaining standard text-side RoPE and an autoregressive transformer, and augmenting them with plug-and-play MM-RoPE for visual tokens and AR-DF for training–inference alignment, the system integrates seamlessly with pretrained LLMs. This composability makes it straightforward to scale to larger multimodal LLMs, leveraging broader pretraining and alignment to improve both understanding and generation.

### Weaknesses
1. Despite the title emphasizing "from a unified model perspective", the paper offers relatively little analysis or evidence on the understanding side of a unified system (e.g., text capability, multimodal understanding). While the architecture remains LLM-compatible and preserves text-side RoPE, the work does not demonstrate positive transfer to or from understanding tasks. Given page limits, a full "generation + understanding" suite isn’t necessary, but minimal support would help: Expanding this section with small-scale evaluations or a concrete discussion/roadmap would better align the title with the evidence.

2. Both the fourth and fifth paragraphs of Introduction begin with "To account for the nature of videos," which disrupts the logical flow. I recommend inserting a paragraph before them that explicitly outlines the key considerations for modeling the spatiotemporal visual space. Then, follow with separate paragraphs that elaborate on each aspect, resulting in a clearer and more coherent structure.

3. The current comparisons do not include 2025 state-of-the-art models, which weakens the empirical case. Please add or reference 2025 baselines (both AR and diffusion/hybrid) with matched settings where possible.

### Questions
1. Does the design in AR-DF affect long-form video generation, especially complex scenarios like shot transitions, and metrics related to dynamics?

2. I recommend providing playable example videos on the project page to enable more effective qualitative visual comparisons.

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
3

### Summary
This paper introduces Lumos-1, an autoregressive video generator based on an LLM architecture with minimal modifications, which aims to create a unified multimodal model. A key achievement is the demonstration of video generation using LLM architecture, which paves the way for a truly unified foundation model and eliminates the need for external text encoders. 

Additionally, Lumos-1 incorporates two innovations:
1.  It proposes MM-ROPE to address imbalanced frequency spectrums in 3D RoPE, enhancing spatiotemporal correlation modeling via distributed channel allocation and scaled 3D positions.
2. Lumos-1 employs autoregressive discrete diffusion forcing (AR-DF) to mitigate frame-wise loss imbalance and spatial information redundancy in autoregressive video generation training.

The paper provides detailed descriptions of the methods employed, thorough experimental evaluations against robust benchmarks, and insightful ablation studies that effectively demonstrate the merits of the innovations. However, Lumos-1 is trained from scratch despite a high structural similarity with Llama, so the model needs to learn both language and vision simultaneously, which can introduce instability and inefficiency during training. 

In summary, this is an innovative work with significant community impact, and I appreciate its architectural exploration; however, I remain concerned about its training efficiency and stability, and I am curious about its performance with LLM initialization.

### Strengths
1. This work proposes Lumios, which demonstrates the effectiveness of video generation using LLM architecture, paving the way for a truly unified foundation model and eliminating the need for external text encoders. 
2. It proposes MM-ROPE to address imbalanced frequency spectrums in 3D RoPE, enhancing spatiotemporal correlation modeling via distributed channel allocation and scaled 3D positions.
3. Lumos-1 employs autoregressive discrete diffusion forcing (AR-DF) to mitigate frame-wise loss imbalance and spatial information redundancy in autoregressive video generation training.
4. Comprehensive experiments and complete ablation studies validate the advantage of MM-ROPE and AR-DF, and the effectiveness of LLM-based video generation.

### Weaknesses
1. Although structurally similar to Llama, Lumos-1 is trained from scratch, requiring simultaneous learning of language and vision, which may lead to training instability and inefficiency. The validation curve in Figure 7(a) supports concerns about the instability. Furthermore, as a foundation model, the full cost of the training, particularly in terms of time, is not disclosed, which is a cause for concern in terms of the inefficiency of the training.
2. The paper claims minimal structural modification from LLM for a unified text-visual model, but it lacks a comprehensive structural comparison with other established unified generative models or LLM-based autoregressive models, making the claimed advantage less substantiated.

### Questions
1. Since the model structure is highly similar to LLM, why not initialize the parameters with pretrained Llama?
2. Based on the experimental results, the unified architecture improves semantic consistency significantly, yet the visual quality still lags behind some baseline methods. Have you attempted to explain this trade-off or conducted a deeper analysis of its causes?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper presents Lumos-1, a unified autoregressive video generation model built upon a LLM architecture, designed to address key limitations of existing autoregressive video generators. To overcome decoding inefficiency, the authors integrate masked diffusion with causal attention. They further introduce MM‑RoPE that preserves standard 1D RoPE for text while effectively capturing spatiotemporal correlations in video. Additionally, the model mitigates error accumulation through a specialized form of diffusion forcing that shares both timestep and mask perturbation across frames. Lumos-1 achieves performance comparable to models such as Emu3, Cosmos1, and OpenSoraPlan v1.3.

### Strengths
- Lumos-1 demonstrates that a pure LLM architecture can be employed for autoregressive video generation.
- Lumos-1 addresses the frequency limitations of the original M-RoPE.
- Lumos-1 proposes AR-DF scheme to mitigate the training-inference inconsistency.

### Weaknesses
- **Reconsidering the Masking Strategy of AR-DF.** Diffusion forcing[1] models the conditional probabilities between frames by adding independent noise to each frame. However, due to the issue of information leakage in mask-based approaches, AR-DF can only apply the same noise to all frames, which deviates from the core idea of diffusion forcing. In my view, AR-DF is more like a form of data augmentation or an exploration of better masking strategies for video generation. I believe the authors need to reconsider and clarify their masking strategy.
- **Regarding the novelty of MM-RoPE.** I consider that interleaved frequency allocation (IL-RoPE) was first introduced in Mogao[2], the authors should cite this work and provide a comparative analysis between MM-RoPE and IL-RoPE.
- **Suboptimal performance of unified architechture.** Although Lumos-1 employs architecture without a text encoder, its text-to-image and text-to-video performance remains suboptimal.  The authors could substantially improve performance to better align with the promise of its unified architecture. Moreover, some of the reported baselines appear outdated or unfair, for instance, Show-o[3]’s GenEval score is 0.68 without rewritting, and NOVA[4] achieves a VBench score of 80.12 using long prompts.

- [1] Boyuan Chen, Diego Marti Monso, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Diffusion forcing: Next-token prediction meets full-sequence diffusion. arXiv preprint arXiv:2407.01392, 2024a.
- [2] Chao Liao, Liyang Liu, Xun Wang, Zhengxiong Luo, Xinyu Zhang, Wenliang Zhao, Jie Wu, Liang Li, Zhi Tian, and Weilin Huang. Mogao: An omni foundation model for interleaved multi-modal generation. arXiv preprint    arXiv:2505.05472, 2025. 
- [3] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. In ICLR, 2025.
- [4] Haoge Deng, Ting Pan, Haiwen Diao, Zhengxiong Luo, Yufeng Cui, Huchuan Lu, Shiguang Shan, Yonggang Qi, and Xinlong Wang. Autoregressive video generation without vector quantization. In ICLR, 2025.

### Questions
- See weaknesses.
- In short, the work presents Lumos-1 model for autoregressive video generation,  the contributions of this are work are not sufficiently substantiated by the experimental results: 
1. A unified architecture for visual generation with strong performance.

2. Asynchronous AR-DF for mitigating the training-inference inconsistency.

3. The difference between IL-RoPE and MM-RoPE on interleaved frequency allocation.

If the authors provide stronger model performance or a more thorough analysis of the weaknesses, I would be happy to raise my score.

### Soundness
3

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
5

### Summary
The paper presents Lumos-1, a unified autoregressive model for video generation using a discrete diffusion process. It combines text and video vocabularies and proposes two main techniques: MM-RoPE for better spatiotemporal position encoding, and AR-DF (Autoregressive Discrete Diffusion Forcing) to handle loss imbalance during masked diffusion training. The model is tested on T2I, I2V, and T2V tasks, showing comparable results to larger diffusion-based systems despite using fewer resources.

### Strengths
1. The study of RoPE frequency allocation is genuinely interesting. The proposed MM-RoPE makes sense — it fixes some imbalance in 3D RoPE and doesn’t really cost extra compute.
2. The setup is efficient: 60M images and 10M videos, trained on 48 H20 GPUs.
3. The experiments are complete, covering main benchmarks (T2I, T2V, I2V) and solid ablations.

### Weaknesses
1. The biggest weakness is the performance. Even though the paper claims efficiency, Lumos-1 doesn’t really beat diffusion or existing AR models on any benchmark. The visuals still look blurry and the motion often feels weird or distorted. It’s not convincing that this setup improves things beyond simplicity.
2. The experimental part only uses validation loss to validate different component like AR-DF and MM-RoPE. Including benchmark metrics would better demonstrate effectiveness since validation loss does not always aligns with down stream performance, even for AR models.
3. The frequency allocation for MRoPE is actually getting more and more attention nowadays. For example, [1] proposes to use U-RoPE. A concurrent work [2] gives a much more in-depth discussion about this, including HoPE and IL-RoPE. This paper only compares with VideoRoPE, weaken its contribution.
4. AR-DF looks like a combination of discrete diffusion and diffusion forcing, only with tube masking. But tube masking actually breaks the unidirectional dependency in AR models. This could hurt the scalability of the model.

[1] UniViT: Unifying Image and Video Understanding in One Vision Encoder
[2] Revisiting Multimodal Positional Encoding in Vision–Language Models

### Questions
Please refer to the weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
