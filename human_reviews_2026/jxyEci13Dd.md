# Long-Text-to-Image Generation via Compositional Prompt Decomposition

- Decision: Accept (Poster)
- Scores: 4, 4, 4, 6

## Abstract
While modern text-to-image (T2I) models excel at generating images from intricate prompts, they struggle to capture the key details when the inputs are descriptive paragraphs. This limitation stems from the prevalence of concise captions that shape their training distributions. Existing methods attempt to bridge this gap by either fine-tuning T2I models on long prompts, which generalizes poorly to longer lengths; or by projecting the oversize inputs into normal-prompt space and compromising fidelity. We propose \textbf{P}rompt \textbf{R}efraction for \textbf{I}ntricate \textbf{S}cene \textbf{M}odeling (\textit{PRISM}), a compositional approach that enables pre-trained T2I models to process long sequence inputs. PRISM uses a lightweight module to extract constituent representations from the long prompts. The T2I model makes independent noise predictions for each component, and their outputs are merged into a single denoising step using energy-based conjunction. We evaluate PRISM across a wide range of model architectures, showing comparable performances to models fine-tuned on the same training data. Furthermore, PRISM demonstrates superior generalization, outperforming baseline models by \textbf{7.4\%} on prompts over 500 tokens in a challenging public benchmark.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the limitation of modern text-to-image models in capturing fine-grained details from long text inputs. The authors propose a trainable PromptDecomposer module that decomposes lengthy text prompts into multiple semantically coherent sub-prompts. Experimental results demonstrate that the proposed method achieves strong performance on the challenging DetailMaster benchmark.

### Strengths
A key strength of this work lies in the latent-space prompt decomposition strategy. By decomposing prompts in the latent space, the method avoids the semantic fragmentation issues often observed when splitting raw text. This leads to more coherent sub-prompt representations and enables the model to balance local detail preservation with global semantic consistency, contributing to improved text-image alignment.

### Weaknesses
1. Limited Adaptability to Different Backbones:
As acknowledged by the authors, the proposed method exhibits scalability issues when applied to larger diffusion backbones. Specifically, both the training and inference costs grow significantly with model size, which raises concerns about its practicality on modern large-scale models such as SD3, FLUX, SD3.5, or Qwen-Image. This limitation restricts the method’s usability in real-world or production-level scenarios.

2. Insufficient Experimental Coverage:
To ensure fair and comprehensive evaluation, the method should also be tested on the Evaluation Dataset proposed in LongAlign [1]. Without such comparison, it is difficult to judge whether the performance gains are consistent across different long-text benchmarks.

3. Lack of Verification on Short Prompts:
The paper focuses primarily on long-text generation, but does not evaluate whether the proposed approach compromises the model’s ability to handle short prompts. Experiments on standard short-prompt benchmarks such as GenEval [2] and T2I-CompBench++ [3] are necessary to demonstrate the generalization and robustness of the proposed method.

4. Missing Inference Efficiency Comparison:
A comparison of inference memory consumption and latency between this method and LongAlign [1] would help clarify the trade-offs between alignment improvement and computational overhead, providing a more complete understanding of its practical value.

[1] Improving Long-Text Alignment for Text-to-Image Diffusion Models.

[2] GenEval: An Object-Focused Framework for Evaluating Text-to-Image Alignment.

[3] T2I-CompBench++: An Enhanced and Comprehensive Benchmark for Compositional Text-to-image Generation.

### Questions
1. The training data used in this paper differs from that used in LongAlign [1], which may lead to unfair comparison and make it difficult to attribute performance improvements solely to the proposed method.

2. The PromptDecomposer module appears to have limited contribution when considering the scaling behavior across different model backbones. How does the proposed framework adapt or generalize to models with varying capacities?

3. In [2], a training-free method for processing long texts at the sentence-level can be added to the baseline for reference, which can make the experiment more comprehensive.

[1] Improving Long-Text Alignment for Text-to-Image Diffusion Models.

[2] Hybrid Layout Control for Diffusion Transformer: Fewer Annotations, Superior Aesthetics.

### Soundness
2

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
Compositional approach for long text-to-image generation using learnable queries to decompose long-prompt representations into sub-prompts in representation space, processed in parallel through frozen pre-trained T2I models and merged via concept conjunction. 
Achieves 7.4% better generalization on 500+ token prompts on SD 1.5-based baselines.

### Strengths
1) Interesting compositional framework grounded in energy-based models.

2) Efficient for legacy models: trains only PromptDecomposer (~20hrs on 4 A100s), avoiding large fine-tuning costs.

3) Strong generalization results. 7.4% improvement on 500+ tokens is substantial.

4) Comprehensive evaluation on DetailMaster. With thorough ablation studies and multiple metrics.

4) Clear presentation with good visualizations, honest limitation discussion.

### Weaknesses
1. **Questionable practical motivation with outdated baselines**: Focuses on SD 1.5 (2022) while modern models (SD 3.5, Flux) with T5-XXL encoders likely handle long prompts well. 
> Authors must demonstrate that these SOTA models actually fail on long prompts to justify the compositional approach, and provide comprehensive DetailMaster benchmark comparisons showing where SD 1.5-based methods fall short relative to modern baselines. At least when I see Table 4 result, Vanilla SD 3.5 already shows pretty good result in numbers compared to PromptDecomposer+SD1.5+Tuned or all the other baselines. I cannot understand why we should use PromptDecomposer, if the use of SD 3.5 (or Flux -- I want this added as baseline comparsion too.) is already better in numbers.


2. **Poor scalability to modern architectures**: SD 3.5 experiments (Table 4) show minimal improvements despite requiring 1.2B parameters for PromptDecomposer-SD3.5; moreover, Table 4 reveals SD 3.5 already achieves strong scores (CLIPScore 34.97, DenScore 22.37) suggesting the problem may not exist for modern models. No experiments with Flux/SDXL provided. The approach appears limited to weak text encoders (CLIP), and generalization across different text-encoder architectures (CLIP vs T5 vs T5-XXL) needs systematic demonstration.
> Please add text-encoder comparison experiments for clear demonstration of your contribution.

3. **Evaluation Metrics** : In Tab. 2, you reported HPSv3, but in Tab. 4, you reported HPSv2. Is it Typo? If it is typo, I see that PromptDecomposer or other baselines in Tab. 2 stays at around 6.7 (ELLA) to 13.26 (LongAlign), but using vanilla SD 3.5 shows 28.86. Pickscore and Denscore shows better numbers in PromptDecomposer or on some baselines, but difference is not as big compared to HPS difference.
> I want more explanations on this typo and number differences. Also, please add SD 3.5 / Flux vanilla (w/o tuning or adding PromptDecomposer component) to the main table for clear comparison. Also for these SOTA models I think adding qualitative comparison might be helpful if the actual result of PromptDecomposer demonstrating the long-prompt is better than those SOTA models.

### Questions
**Check weaknesses section above for details.**

The major weakness of this work is the **questionable motivation** and **lack of evaluation/demonstration on state-of-the-art models**. The paper focuses on SD 1.5 (2022) while modern models with stronger text encoders (SD 3.5, Flux with T5-XXL) likely already handle long prompts effectively, yet no comprehensive comparison is provided. Table 4 shows SD 3.5 already achieves strong performance, and the minimal improvement from PromptDecomposer-SD3.5 raises fundamental questions about whether this problem still exists for current models.

**To increase my score, the authors should address:**
1. Demonstrate that SOTA models (SD 3.5, Flux) actually fail on long prompts with DetailMaster benchmark results (or at least on qualitative results.)
2. Include these models as baselines in main comparison tables
3. Provide ablations showing improvements come from decomposition rather than just better text encoding (e.g., T5-XXL with SD 1.5 without decomposition)
4. Add user studies comparing generation quality against modern models

If these additional experiments convincingly show the compositional approach provides value beyond simply using better text encoders, I will reconsider my evaluation.

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
5

### Summary
This paper addresses the challenge of generating high-quality images from lengthy text prompts by proposing PromptDecomposer, a trainable module that decomposes long prompts into manageable sub-prompts processed in parallel by pre-trained T2I models, with outputs fused via concept conjunction. The method achieves competitive performance on DetailMaster benchmark and demonstrates superior generalization on prompts exceeding 500 tokens, improving performance by 7.4% over existing approaches.

### Strengths
1. The paper proposes compositional long-text-to-image generation and unsupervised long-prompt decomposition methods to enable models to better perceive lengthy text inputs.

2. Experimental validation effectively demonstrates the effectiveness of the proposed approach.

### Weaknesses
1. While decoupling long text representations is a reasonable idea, the proposed modules bear strong resemblance to existing architectures (e.g., Q-Former), lacking sufficient novelty. The approach is heavily data-driven without explicit representation loss guidance, which raises concerns about the generalization capability of learned parameters. Additionally, the acquisition of image captions is critical for training but not thoroughly discussed.

2. In Table 1, SDXL-based models outperform the proposed method on Character Presence and Object metrics. Why were experiments not conducted on SDXL? SD-1.5 is outdated. As acknowledged in the limitations, transferring to SD3.5 does not yield significant improvements. Given the emergence of Flux, Qwen-Image, and similar models, exploring complex prompt generation on these newer architectures would be more valuable.

### Questions
How should this method be adapted to state-of-the-art models like Qwen-Image (with Qwen2.5-VL as encoder) or MetaQuery-type architectures? What modifications are necessary for effective transfer?

### Soundness
3

### Presentation
3

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
The paper aims to enhance image generation on long, paragraph-length prompts. It proposes a compositional pipeline with a trainable PromptDecomposer that splits a long prompt into distinct sub-prompts. A pre-trained T2I model processes these sub-prompts in parallel, and outputs are merged via concept conjunction. The method shows good gain on >500-token prompts in the DetailMaster benchmark.

### Strengths
- Good motivation and problem formulation: it’s a well-established research question that the state-of-the-art image generation models are usually trained on limited-length captions, which generates lower-quality images on long prompts. 
 - Intuitive modules and good results: the proposed modules in PromptDecomposer are well motivated. The experiments show PromptDecomposer clearly outperformed baselines on long prompts.

### Weaknesses
- Limited novelty: many modules in this paper can be found in references. For example, cross-attention, T5, CLIP are off-the-shelf modules. It’ll be great if the authors could explain more about what’s the unique contribution and novelty in this paper. 
 - Risk of losing global coherence: when merging independently generated components,  global coherence (such as lighting, perspective, style) might be lost, or might be conflicting with other components (e.g. day vs night, mountain vs sea).

### Questions
- The authors ablated the number of learnable queries in the paper. But they are still fixed. I wonder if we should make the number of learnable queries adaptive or dependent on the prompts?

### Soundness
3

### Presentation
3

### Contribution
3
