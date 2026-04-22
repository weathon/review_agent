# OneFlow: Concurrent Mixed-Modal and Interleaved Generation with Edit Flows

- Avg Score: 5.50
- Decision: Reject
- Scores: 6, 4, 6, 6

## Abstract
We present OneFlow, the first non-autoregressive multimodal model that enables variable-length and concurrent mixed-modal generation. Unlike autoregressive models that enforce rigid causal ordering between text and image generation, OneFlow combines a insertion-based Edit Flows for discrete text tokens and Flow Matching for image latents. OneFlow enables concurrent text-image synthesis with hierarchical sampling that prioritizes content over grammar. Through controlled experiments across model sizes from 1B to 8B, we demonstrate that OneFlow outperforms autoregressive baselines on both generation and understanding tasks while using up to 50% fewer training FLOPs. OneFlow surpasses both autoregressive and diffusion-based approaches while unlocking new capabilities for concurrent generation, iterative refinement, and natural reasoning-like generation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
OneFlow is a non-autoregressive multimodal generation framework that jointly handles variable-length text and an arbitrary number of images by combining (i) an insertion-based Edit Flows procedure for discrete tokens and (ii) Flow Matching for image latents. A novel interleaved time schedule couples per-image generation times to the text insertion process so images can be inserted and denoised concurrently with text rather than waiting until an image is completed.

### Strengths
Novel Integration: The work integrates Edit Flows (insertion operations) with continuous Flow Matching for image latents in a single backbone so both modalities are predicted by the same model and can be denoised jointly.

Rigorous Evaluation: The paper conducts rigorous experiments using a diverse set of benchmarks, metrics, and baselines to thoroughly evaluate the effectiveness of its proposed method. Also the appendix is really rich.

Emergent Reasoning: The observation that OneFlow develops implicit reasoning chains (Figure 5, 19) is a fascinating finding from my point of view.

Writing: The paper is well-written, and the methodology is explained with remarkable clarity and it was so easy for me to follow.

### Weaknesses
Ablation clarity for interleaved schedule: The interleaved schedule is central. I would like to see ablations for it. The paper describes κ(t)=t (linear) and claims it works well, but sensitivity analysis is missing.

Potential mode of failure for complex interleavings: Examples show 2 images interleaved with text. It is unclear how the model behaves when the number of inserted images is large, or when images must be heavily conditioned on earlier generated text.
Limited operation diversity in Edit Flows: The model only supports insertion operations, omitting deletion and substitution — both fundamental to edit-based generative modeling. (they are even present in the Original Edit Flow Paper)

### Questions
1- It would be insightful to further evaluate the model’s ability to handle compositional generation and compositional visual question answering, and to benchmark its performance against comparable multimodal baselines in these challenging settings. (e.g. attribute binding, missing entities etc.)

2- What modifications would be needed to extend OneFlow to other modalities like audio or video, given its reliance on Edit Flows for discrete elements and Flow Matching for continuous ones?

3- Beyond the linear κ_t scheduler, what alternatives were tested, and why did linear perform best? Could adaptive schedules further reduce FLOPs while maintaining performance?

4- The paper shows that CFG for text increases detail but also the chance of hallucinations (Figure 11). A more systematic analysis of this trade-off (e.g., a plot of detail vs. hallucination rate across CFG scales) would be beneficial.

5- Please include qualitative examples of failed generations (misplaced images, inconsistent text-image references, incoherent insertions).

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents OneFlow, the first non-autoregressive (NAR) multimodal model designed for variable-length and concurrent mixed-modal (text-image) generation. It addresses key limitations of existing autoregressive (AR) and diffusion-based multimodal models: AR enforces rigid sequential generation (preventing cross-modal refinement), while diffusion models only support fixed-length, pre-specified text-image pairs.

### Strengths
1. OneFlow fills a critical gap in multimodal generation by pioneering an NAR framework for concurrent, variable-length interleaved text-image generation. The integration of Edit Flow and Flow Matching into a unified transformer backbone is non-trivial: it avoids modality-specific silos and enables cross-modal dependency modeling during generation.

2. The model’s training efficiency is a standout advantage: by using a linear deletion scheduler that retains only 50% of tokens during training, OneFlow reduces training FLOPs by up to 50% compared to AR baselines.

3. OneFlow enables practical, previously unachievable use cases: Unlike AR models that append images to the end of text, OneFlow inserts images dynamically within text (via <|image|> tokens) and refines them simultaneously; Without CoT prompting or RL post-training, OneFlow generates reasoning chains for visual questions (e.g., object counting, visual search), demonstrating that NAR architectures can support complex reasoning.

### Weaknesses
1. High Inference Cost: OneFlow lacks key-value caching (due to bidirectional attention), leading to higher inference latency and memory usage than AR models. This limits its applicability to low-latency scenarios (e.g., real-time multimodal chatbots). No preliminary optimizations (e.g., semi-autoregressive decoding, sparse attention) are proposed to mitigate this.

2. Incomplete Comparisons to SOTA Models: While the paper compares OneFlow to VQA baselines like Show-O (1.3B), Janus-Pro (1.5B/7B), and MMaDA (8B), it omits critical recent SOTA models that have set new benchmarks in visual question answering. Specifically, there is no comparison to Show-O2, Mogao, or BAGEL; For image generation (Table 1), the paper evaluates OneFlow against AR models (e.g., Transfusion, Janus-Flow) and diffusion models (e.g., MMaDA, FUDOKI) but lacks comparisons to some SOTA works: Show-O and DreamLLM.

### Questions
1. For the VQA task, you omit comparisons to recent SOTA models like Show-O2, Mogao, and BAGEL. Do you have preliminary results comparing OneFlow to these models on key VQA benchmarks (e.g., VQAv2, GQA, DocVQA)? 

2. You note that OneFlow’s lack of KV caching increases latency. Have you explored lightweight optimizations (e.g., semi-autoregressive block decoding, sparse bidirectional attention, or model distillation) to reduce inference cost?

3. You finetune on 512×512 images, but many SOTA models (e.g., SD3) support higher resolutions (1024×1024). Has OneFlow been tested on higher-resolution image generation, and if so, how does performance (e.g., FID, detail) scale with resolution?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes OneFlow, the first non-autoregressive multimodal model that enables concurrent and variable-length text-image generation by combining an insertion-based "Edit Flow" for text with "Flow Matching" for images, which outperforms autoregressive baselines with greater efficiency.

### Strengths
- This is a very solid work from both a technical and an engineering perspective. It skillfully combines the recent, popular diffusion text generation techniques with the consistently effective diffusion image generation methods within a single Transformer. I believe this is a strong contribution to both the DLMs and the unified model communities.

- The paper presents comprehensive and rigorous experiments. This includes scaling experiments from 1B to 8B parameters, extensive comparisons against baseline models (for both multimodal understanding and generation), and supplementary ablation studies in Appendix F. This significantly increases the credibility of the paper.

### Weaknesses
This paper frequently mentions two terms: "interleaved mixed-modal generation" and "concurrent mixed-modal generation." My concerns are primarily centered on these two points.

- Regarding Interleaved Generation: In the introduction, the authors highlight OneFlow's ability to perform interleaved generation with a variable number of outputs as a major contribution. However, in the experiments, the paper only evaluates OneFlow's performance on multimodal understanding and image generation. It notably lacks evaluations on interleaved generation benchmarks, such as OpenING. I believe this experimental omission fails to substantiate the claims made in the introduction.

- Regarding Concurrent Generation: Similarly, the introduction categorizes concurrent mixed-modal generation as a novel capability for unified models. However, after reading the paper, I still do not understand why we need concurrent generation. Is it simply because it leads to better performance (as suggested in Figure 3)? Is there a deeper insight or analysis behind this capability that I missed?

### Questions
- Will the OneFlow model weights and modeling files be open-sourced? If subsequent researchers cannot build upon this work, it would be detrimental to the unified model community.
- It appears the authors used closed-source fine-tuning data (Line 267). This could potentially impact the reproducibility of the method.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces OneFlow, a novel non-autoregressive multimodal model that unifies text and image generation within a single framework and overcomes key limitations of existing autoregressive and diffusion models by combining Edit Flows for variable-length discrete text generation and Flow Matching for continuous image generation. This allows for the concurrent and interleaved generation of variable-length text and a variable number of images, enabling capabilities like simultaneous cross-modal refinement.

### Strengths
- The proposed OneFlow model requires fewer training FLOPs than compared to AR models, thanks to its non-autoregressive, insertion-based training which only predicts missing tokens.
- The paper provides extensive experiments showing that OneFlow is competitive with or superior to state-of-the-art AR and diffusion models across a diverse range of benchmarks for both image generation and understanding.
- The paper convincingly demonstrates new emergent capabilities, such as implicit reasoning chains without Chain-of-Thought prompting, the application of classifier-free guidance to improve text detail, and the dynamic insertion and denoising of images within a text sequence.

### Weaknesses
- While the integration of Edit Flows with Flow Matching is innovative and effective, the core architectural contribution is the combination of these existing techniques for a new multimodal task. The paper's impressive results are therefore heavily reliant on the pre-established Edit Flows framework, and the fundamental methodological novelties introduced beyond this combination are more incremental.
- As stated by the authors, the bidirectional attention required for non-autoregressive generation prevents the use of key-value caching, making OneFlow's inference slower and more memory-intensive than cached AR sampling, despite requiring fewer steps.
- The 20% mixed-generation probability is used without justification. The performance sensitivity to this key hyperparameter is also unknown, making it hard to determine the optimal data mixture.
- Certain details about hyperparameters are omitted and could use ablations, see Q1 (Ablation on Scheduler) and Q2 (t-Independent Parameterization).

Formatting Concerns:
- Table captions must precede the tables.
- Although clear from the context, the abbreviation VQA is not explicitly clarified.

### Questions
- Can you elaborate more on the different candidates for κt (line 112) and provide more information?
- The decision to use a t-independent model is noted to work better in practice despite the theoretical justification for t-dependence. Could you provide an ablation quantifying the performance gap between these two parameterizations? Were there specific tasks where the t-dependent model performed better?
- As shown in Figure 11 and discussed in Sections 3.5 and 4, higher CFG weights consistently lead to longer, more detailed captions but also to an increased chance of hallucinations. Can you discuss this trade-off in more depth?
- The emergent reasoning chains (line 367) are a fascinating finding. Can you elaborate on the conditions or training data that you believe led to this behavior? Is it a general property of the Edit Flow text generation, or is it specific to the multimodal pretraining mixture used?

### Soundness
4

### Presentation
4

### Contribution
3
