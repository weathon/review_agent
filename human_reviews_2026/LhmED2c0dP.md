# STAR: STacked AutoRegressive Scheme for Unified Multimodal Learning

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 8, 4, 2

## Abstract
Multimodal large language models (MLLMs) play a pivotal role in advancing the quest for general artificial intelligence. However, achieving unified target for multimodal understanding and generation remains challenging due to optimization conflicts and performance trade-offs. To effectively enhance generative performance while preserving existing comprehension capabilities, we introduce STAR: Stacked AutoRegressive scheme for task-progressive unified multimodal learning. This approach decomposes multimodal learning into multiple stages: understanding, generation, and editing. By freezing the parameters of the fundamental autoregressive (AR) model and progressively stacking isomorphic AR modules, it avoids cross-task interference while expanding the model's capabilities. Concurrently, we introduce a high-capacity VQ to enhance the granularity of image representations and employ an implicit reasoning mechanism to improve generation quality under complex conditions. Experiments demonstrate that STAR achieves state-of-the-art performance on GenEval (0.91), DPG-Bench (87.44), and ImgEdit (4.34), validating its efficacy for unified multimodal learning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper introduces STAR (Stacked AutoRegressive Scheme) — a unified multimodal learning framework that aims to integrate image understanding, text-to-image generation, and image editing within a single model. The authors propose a task-progressive training paradigm, where the base autoregressive (AR) transformer is frozen and extended by stacking isomorphic AR modules. They also introduce a high-capacity STAR-VQ tokenizer for fine-grained visual tokenization and an implicit reasoning mechanism to handle complex compositional prompts. Experimental results show that STAR achieves state-of-the-art performance on multiple benchmarks including GenEval (0.91), DPG-Bench (87.44), and ImgEdit (4.34), while maintaining strong multimodal understanding capabilities.

### Strengths
1.**Strong Empirical Results.**
The model achieves state-of-the-art or competitive performance on a wide range of benchmarks across understanding, generation, and editing tasks. This broad validation suggests that the proposed approach is robust and effective.

2.**Comprehensive Evaluation.**
The paper includes extensive quantitative and qualitative experiments, along with meaningful ablation studies on VQ design, stacked layer depth, and diffusion decoder strategies.

3.**Clarity and Structure.**
The paper is generally well-organized, with clear explanations of each component (VQ encoder, stacked AR model, diffusion decoder) and training stages (Figure 3).

### Weaknesses
1. **Lack of deeper insight.**
   Conceptually, the stacked-AR design still functions as a *feature transformation* module (as mentioned around line 043). The integration of a diffusion decoder is largely a *common practice* in recent unified multimodal models. The true novelty lies in the proposed multi-stage training paradigm; however, the paper lacks ablation studies or quantitative analysis that isolate the benefits of this multi-stage scheme.

2. **Possible gains due to increased parameter count.**
   The improvements brought by the stacked-AR module may primarily stem from the increase in trainable parameters, rather than from architectural novelty itself—especially when compared with strong baselines such as BLIP3-o-next.

3. **Understanding benchmarks not a core contribution.**
   The scores reported in Table 1 for the understanding benchmarks appear to be almost identical to those of Qwen2.5-VL, indicating that these results are inherited rather than achieved through STAR’s innovations. Therefore, this section could be presented as secondary, with more emphasis placed on the generative benchmarks and results, which form the true core contribution of the paper.

### Questions
1. **Effect of multi-stage training:**
   Could the authors provide quantitative evidence or ablation studies showing the benefit of the multi-stage training pipeline? How does it compare to training the stacked-AR and diffusion decoder jointly in a single stage?

2. **Design choice for the decoder:**
   Since the diffusion decoder ultimately replaces the VQ decoder, did the authors experiment with using a *VQ encoder + diffusion decoder* setup from the beginning (e.g., in Stage 1)?

3. **Image editing fidelity with diffusion decoders:**
   Prior work (e.g., EMU2) suggests that diffusion decoders struggle to accurately reconstruct fine visual details, which is critical for image editing tasks. Given that STAR also employs a diffusion decoder, how does it mitigate this limitation?

4. **Impact of parameter count in stacked-AR:**
   Is the improvement of stacked-AR primarily due to the increased number of trainable parameters? How would the performance compare if a *shallow stacked-AR* or a *simple MLP connector* were used instead? Clarifying this point would make the architectural contribution of stacked-AR more convincing.

### Soundness
4

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
4

### Summary
This paper introduces STAR, a new training and architectural paradigm for building unified MLLMs that can perform image understanding, generation, and editing without degrading existing capabilities. It addresses the critical challenge in unified MLLMs: optimization conflict between understanding and generation tasks.

The STAR framework proposes a stacked autoregressive scheme that decomposes learning into task-progressive stages. It preserves previously learned capabilities (e.g., image-text understanding) by freezing the fundamental AR backbone and progressively stacking isomorphic AR modules for new capabilities like generation and editing.

The method also introduces STAR-VQ, a high-capacity vector quantizer for discretized image tokenization, and an implicit reasoning mechanism that enhances generation quality under complex prompts without requiring new parameters.

Experiments show STAR achieves state-of-the-art results across several benchmarks:

- GenEval (0.91), DPG-Bench (87.44), ImgEdit (4.34)

- Competitive on understanding benchmarks like MMStar, SEED, OCRBench

- Strong performance on reasoning benchmark WISE (0.66)

### Strengths
The paper proposes a novel stacked autoregressive (STAR) architecture for unified multimodal learning that allows progressive expansion from understanding to text-to-image generation and image editing without retraining or catastrophic forgetting. The idea of stacking isomorphic AR modules as “frozen base + appended heads” represents a creative rethinking of multimodal model scaling. The overall technical design is sound and well-motivated. The proposed STAR-VQ tokenizer, the modular training curriculum (four well-separated stages), and the decoupled optimization scheme demonstrate careful engineering and empirical validation. The experiments are comprehensive across understanding, generation, and editing benchmarks, and ablation studies provide insights into how stacking and the implicit reasoning mechanism contribute to the performance. The method is shown to mitigate optimization interference that often plagues unified generative–understanding models.

### Weaknesses
1. Missing comparison. Some methods[1,2,3] are missing in experiments. Inclusion of such baselines would clarify STAR’s relative advantage and limitations

2. Limited theoretical justification. The paper provides strong empirical validation but offers little theoretical analysis explaining why the stacked autoregressive (AR) expansion avoids optimization interference. A more formal discussion of gradient isolation or representational decoupling between frozen and stacked modules would strengthen the conceptual foundation.

3. Limited discussion of data scaling and domain coverage. STAR’s improvements may partly stem from larger or higher-quality pretraining data, but dataset scale and composition are underreported. A clearer breakdown of data sources and training schedule would help assess fairness in comparison and potential data bias.

4. Qualitative analysis and failure cases missing. The paper shows limited visual examples and no systematic failure analysis. Understanding where implicit reasoning or the diffusion decoder fails (e.g., under abstract or multi-object prompts) would provide valuable insight for future improvements.

[1] UnifiedMLLM: Enabling Unified Representation for Multi-modal Multi-tasks With Large Language Model

[2] OmniGen2: Exploration to Advanced Multimodal Generation

[3] VisionLLM v2: An End-to-End Generalist Multimodal Large Language Model for Hundreds of Vision-Language Tasks

### Questions
Please weaknesses.

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
This paper introduces STAR (STacked AutoRegressive scheme), a framework for creating unified multimodal large language models (MLLMs) that are capable in understanding, generation, and editing tasks. The core problem addressed is the inherent conflict and performance trade-offs that arise when training a single model for both comprehension and generation. STAR's key contribution is a task-progressive training strategy where a pre-trained multimodal understanding model is frozen, and its capabilities are extended by stacking isomorphic autoregressive (AR) layers on top of it. The framework is further enhanced by a high-capacity vector quantizer (STAR-VQ) for finer image representation and an implicit reasoning mechanism at inference time.

### Strengths
1. Strong Empirical Performance: The paper provides comprehensive experimental validation. STAR achieves state-of-the-art (SOTA) results on multiple generation benchmarks, including GenEval, DPG-Bench, and ImgEdit. As shown in Table 1, the model also retains competitive performance on a wide array of understanding benchmarks (MMStar, SEED, MME, OCRBench), demonstrating that the generative extensions did not lead to a significant degradation of the base model's comprehension abilities.

2. High-Capacity Vector Quantizer (STAR-VQ): The development of a 1B-parameter vector quantizer is a notable engineering effort. This component directly addresses the need for high-fidelity discrete visual representations, which can be a limiting factor for the quality of images produced by autoregressive generative models. The ablation in Table 6a suggests this component contributes positively to the final generation quality.

3. Implicit Reasoning Mechanism: The inference-time strategy of using the frozen base model to first generate a latent semantic representation before the stacked layers generate the image is a zero-parameter-cost method for improving prompt alignment. This two-step process is designed to better handle prompts that require world knowledge or compositional reasoning, and the qualitative example in Figure 4(b) suggests its potential benefit.

### Weaknesses
1. Limited Technical Novelty: The core technical proposal—stacking additional, isomorphic layers onto a frozen backbone—is a straightforward and well-established technique in transfer learning. While its application to unified MLLMs is shown to be effective, the underlying mechanism lacks significant technical novelty and could be viewed as an incremental engineering contribution rather than a fundamental advance in model architecture or training paradigms.

2. Unanalyzed Computational Cost and Parameter Overhead: The method incurs a substantial increase in parameters. The STAR-7B model adds a 3B parameter AR stack on top of the base model, and both variants rely on a very large 1B parameter STAR-VQ. The paper does not provide an analysis of the training and inference costs (e.g., FLOPs, memory, latency) associated with this overhead. This makes it difficult to assess the efficiency of the method and raises the question of whether the performance gains are primarily a result of the proposed strategy or simply due to the massive increase in model size allocated to the generative task.

3. Insufficient Ablation of the Implicit Reasoning Mechanism: The paper claims that the implicit reasoning mechanism "yields substantial gains" on knowledge-intensive benchmarks. However, this claim is not supported by any quantitative ablation studies. The only evidence is a single qualitative example in Figure 4. A proper ablation showing performance on a benchmark like WISE with and without this mechanism is necessary to validate its effectiveness and justify its inclusion as a key contribution.

4. Rigidity of the Frozen Backbone Assumption: The core design choice is to keep the entire base model frozen to prevent catastrophic forgetting. While effective for this purpose, this rigid constraint may limit the model's ability to learn fine-grained alignments between the pre-trained understanding features and the new generative task. An exploration of alternatives, such as partially fine-tuning the top layers of the base model or employing parameter-efficient fine-tuning (e.g., LoRA), would strengthen the paper's claims by showing that a fully frozen backbone is indeed the optimal choice.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

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
The paper introduces STAR(STacked AutoRegressive scheme), a unified multimodal large language model for image understanding, generation and editing. It works by freezing the parameters of a fundamental autoregressive (AR) model and then progressively stacking isomorphic AR modules on top that are trained for generation and editing with the standard next-token objective. This method enables the new modules to learn generative capabilities (like text-to-image image generation) without degrading the base model's existing comprehension skills. STAR also introduces a high-capacity Vector Quantizer (STAR-VQ) for highly precise image representations and employs an implicit reasoning mechanism to improve generation quality under complex, knowledge-intensive conditions.

### Strengths
1. The manuscript includes targeted ablations for stack depth, initialization strategy, VQ type, diffusion vs VQ decoder, input strategies for DiT conditioning, which help explain why each design choice was made and where gains come from.

2. Various experiments, comprehensive comparison with existing works.

3. A wide array of experiments is conducted to showcase the effectiveness of STAR.

### Weaknesses
1. Table 2 and Table 3 are misleading, as bold text should highlight the best model. The authors may want to reconsider whether Table 1 is necessary, as the image understanding capabilities are entirely inherited from the frozen VLM.

2. There are a few typos: issues with plural and non-plural forms, and “autoregressive (AR)” should be used as an adjective, not a noun.

3. Key claimed novelties are combinations of known ingredients.
While the paper proposes the “stacked isomorphic AR layers + STAR-VQ + diffusion refinement” recipe, the core ideas are largely incremental relative to recent work.
Stacking layers is a structural variant of warm-start/adaptor approaches (LMFusion’s parallel modules and MetaQueries’ learnable queries). Dual-tokenization + diffusion refinement is explicitly done by ILLUME+/Janus-style papers. Scaling VQ codebooks has independent prior art (e.g., VQGAN-LC). STAR’s contribution reads as an engineering recipe (combination/scale/tuning) rather than a new conceptual mechanism.

3. The method itself fails to address the questions raised in the abstract regarding how it reduces training complexity.

### Questions
1. STAR attributes gains partly to the very large codebook (65k×512). Could you provide some experimental analysis on whether performance would hold with a smaller codebook paired with the stacked AR, or whether gains are mostly from codebook scale? 

2. From my understanding, the stacked autoregressive (AR) layers in STAR functionally act as a connector between the frozen vision-language model (VLM) and the diffusion decoder. Can you explain how this particular structure is better?
3. Can you make some comparisons between STAR and other related works like UniWorld, Unifusion?

I am more than happy to raise the score if the authors could provide sufficient reasons.

### Soundness
3

### Presentation
2

### Contribution
2
