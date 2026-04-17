# IPGO: Indirect Prompt Gradient Optimization for Text-to-Image Model Prompt Finetuning

- Decision: Reject
- Scores: 4, 6, 4, 2

## Abstract
Text-to-Image (T2I) Diffusion models have become the state-of-the-art for image generation, yet they often fail to align with specific reward criteria such as aesthetics or human preference.  We propose Indirect Prompt Gradient Optimization (IPGO), a novel model-independent framework that enhances prompt embeddings by injecting learnable text embeddings as prefix and suffix around the original prompt embeddings. IPGO leverages reduced-rank approximation and rotation, while enforcing range and orthonormality constraints and a conformity penalty to ensure stability and mitigate reward hacking. We evaluate IPGO against six baseline methods under single-prompt optimization with three reward models targeting image-text alignment, image aesthetics, and human preferences across three datasets of varying prompt complexity. The results show that IPGO consistently outperforms all baselines over strong competitors such as DRaFT-1 and TextCraftor. We investigate and mitigate reward hacking, in particular for aesthetics. Ablation studies further highlight the contributions of each IPGO component, while additional experiments demonstrate IPGO's application to various T2I diffusion models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a prompt tunning methods for human or aesthetics preference alignment. The main idea of this paper is to finetun the prefix and suffix of the given prompt in the embedding space. Therefore, the trainable parameter can be greatly reduced. In addition, the authors introduced low-rank approximation and rotation transform on the trainable embedding space, which also demonstrate its improvement in the experiment section.

### Strengths
- Strengths

1) Similar to the prefix prompt tunning, this paper finetung prefix and suffix embedding for aligning the human preference using reward model.

2) In addition, the authors introduced rotation and low-rank approximation (I would like to say that it is not true) for improving the expressiveness of the trainable embedding.

### Weaknesses
- Weaknesses

1) In the related works, the authors claimed that they do not require to access the original image compared to PEZ and Textual Inversion. However, I think this is not true as it is related to the task. PEZ requires original images since they consider the prompt discovery tasks.

2) In equation (3), the dimension of Z and E are not aligned, they can not be multiplied directly.

3) The reason for decompose the prefix or suffix embedding into low-rank matrices are not clear. In addition, according to the Table 8, m is set to 300 while the prefix or suffix is 10, thus it is totally not the low-rank approximation or parameter-efficient strategy, but the over-complete dictionary learning. The authors should thoroughly correct this claim. 

4) Given a reward loss function, how to optimize the embedding? Can we directly optimize the embedding through via BP? More details can also be introduced in the paper.

### Questions
See the weakness section for details.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduce Indirect Prompt Gradient Optimization (IPGO), a parameter-efficient method for aligning text-to-image (T2I) diffusion models with various reward objectives including aesthetic, CLIP alignment, and human preference. Instead of modifying diffusion model’s backbone or text-encoder parameters, IPGO optimizes a few learnable prefix and suffix embeddings added to the original prompt’s text embeddings. These embeddings are optimized via a constrained, gradient-based procedure incorporating low-rank approximation, rotation parameterization. The framework is training-efficient (0.47M parameters) and evaluated under multiple datasets and showcased consistent improvements of 1–3% over strong baselines.

### Strengths
1. **Clear Writing:** The paper is clearly written and well-structured. The motivation and methodology are presented in an organized and accessible manner, making it easy to follow and pleasant to read.
2. **Novel and Effective Method:** The proposed approach introduces a novel optimization-based framework for prompt refinement, which effectively enhances text-to-image alignment and image quality. Extensive experiments showcased the effectiveness of the IGPO method.

### Weaknesses
1. **Limited Exploration on Large-Scale Models:** While the method targets parameter-efficient learning, most results are on backbones with relatively small text encoders (e.g., SD). It remains unclear how well the approach scales to larger systems and richer text stacks (Flux). Evaluating on modern large diffusion models—adding both quality gains and compute/latency/memory trade-offs to Table 6—would strengthen the claim of broad effectiveness and practical scalability.
2. **Weak Justification for Using Both Prefix and Suffix:** The paper claims that using both prefix and suffix embeddings improves performance. However, as shown in Table 5, configurations with only prefix or only suffix embeddings (e.g., (10, 0) or (0, 10)) achieve nearly identical CLIP scores (0.286 vs. 0.289). This marginal difference does not strongly justify the necessity of employing both prefix and suffix components simultaneously. Authors may need to compare (10,10) with (20,0) or (0,20) to ensure a fair comparison with the same learnable parameters.

### Questions
see weakness

### Soundness
3

### Presentation
4

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
This paper introduces Indirect Prompt Gradient Optimization (IPGO), a parameter-efficient framework for prompt-level finetuning in text-to-image (T2I) diffusion models. IPGO enhances prompt embeddings by injecting learnable prefix and suffix embeddings, optimized via gradient-based methods with low-rank approximation, rotation, and stability constraints (orthonormality, range, and conformity). Unlike prior approaches, IPGO requires no modification to the diffusion model or text encoder and operates with far fewer parameters, enabling efficient, prompt-wise optimization at inference. Experiments across three datasets and three reward models (aesthetics, image-text alignment, and human preference) show that IPGO consistently outperforms baselines, such as TextCraftor and DRaFT-1, while using fewer parameters.

### Strengths
+ The proposed method is efficient and it achieves better results with fewer parameters and lower hardware requirements than baselines such as TextCraftor and DRaFT-1.

+ Looks like the proposed method has potential to be applied to existing T2I diffusion models and reward functions without modifying the underlying model or text encoder.

+ The proposed outperforms several baselines across multiple datasets and reward types. It also provides ablation studies to validate the contribution of each design choice and constraint.

### Weaknesses
- Native image generation models (e.g. VAR [a] BAGEL [a], ) are pretty popular recently which does not include a text encoder. It is not clear how the proposed method can be applied or generalized to these recent SOTA methods. 

[a] Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction, 2024
[b] Emerging properties in unified multimodal pretraining, 2025

- It is not clear how well the proposed method can handle the case where inference prompts are long and detailed (e.g. > 150 words). As for recent methods, LLM prompt rewrite/expansion is a commonly used strategy which will transfer short input prompt to detailed long prompt as input to the model. It is not clear how good ADSS can be applied to those cases with detailed prompt.

- SOTA performances are claimed by the submission, while the most recent methods included for comparisons are TextCraftor (year 2024) and DRaFT-1 (year 2023) which is a bit old.

### Questions
Please refer to the detailed questions raised in Weakness section above.

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
This paper proposes IPGO (Indirect Prompt Gradient Optimization), a method for aligning text-to-image diffusion models with reward functions through optimization of prompt embeddings. The approach adds learnable prefix and suffix embeddings to the original prompt embeddings, parameterized through rotated low-rank approximations. Instead of using explicit KL regularization like traditional RL-based alignment methods, IPGO relies on three embedding-space constraints to prevent reward hacking: orthonormality of the embedding basis, range constraints on coefficients ([-1,1]), and conformity (mean preservation with original prompt embeddings). The method operates as a test-time optimization approach, optimizing each individual prompt separately for multiple epochs. Experiments are conducted on prompts from COCO, DiffusionDB, and Pick-a-Pic datasets using Stable Diffusion v1.5 and three reward models (CLIP alignment, LAION aesthetics, HPSv2 human preference). IPGO is compared against six baselines, including training-based methods (TextCraftor, DRaFT-1, DDPO) and training-free methods (DPO-Diff, Promptist), showing improvements over competing methods when evaluating on the reward that is optimized.

### Strengths
- **Novel parameterization approach for prompt embedding optimization**: The method combines prefix-suffix embeddings with rotated low-rank parameterization and three constraints (orthonormality, range, conformity) to keep optimization within a meaningful embedding region. The linguistic motivation for prefix-suffix design is intuitive, and the approach allows preserving original prompt semantics while adding learnable content.

- **Ablation studies**  The paper provides valuable ablations showing task-dependent importance of constraints (e.g., range crucial for aesthetics, orthonormality for alignment) and demonstrating that the parameterization significantly outperforms naive unconstrained optimization. Ablations on prefix/suffix lengths reveal that equal lengths work best and longer isn't necessarily better.

### Weaknesses
Major weaknesses, experimental design: (I would be open to reconsidering my score given that these concerns are adressed.)
- **No evaluation on general T2I benchmarks**: The evaluation is limited to the same reward models used for optimization, which provides no evidence that IPGO improves general image quality beyond simply overfitting to the metric. Without testing on standard T2I benchmarks that assess compositionality and attribute binding (e.g., GenEval, T2I-CompBench), or conducting human studies, or minimally cross-validating rewards,  the reported gains could reflect reward hacking rather than genuine improvements. The risk of exploiting reward model biases is substantial, especially given the multi-epoch per-prompt optimization and the absence of explicit KL regularization to maintain fidelity to the original model distribution.
- **Experimental design does not align with the test-time optimization paradigm**: The experimental design is not fully convincing to me, IPGO requires multiple optimization epochs per generated image, against training-based baselines (e.g., TextCraftor, DRaFT-1) that offer instant inference after a one-time training cost. While this comparison is interesting, I would argue the main comparison should be against other test-time techniques, including wall-clock time as one axis. Currently, it's unclear what the compute vs performance trade-off looks like. While the main comparison would be Promptist, other test-time optimization techniques are in my view the main competing methods to IPGO (e.g., noise selection [1] (Best-of-N, or over paths) or noise optimization [2,3]).
- In my opinion, reporting wall-clock time (and GPU memory) and performance on some metric that is disentangled from the optimized metric (e.g. GenEval) compared to more test-time techinques, is needed to accurately assess the performance of IPGO, which the current evaluation doesn't sufficiently cover.

Minor weaknesses:
- **Limited justification for specific design choices**: Many of the method's core components lack rigorous justification and appear arbitrary. The rotation parameterization is motivated by a simplified 2D analysis that does not transfer to the high-dimensional setting and is empirically contradicted by results where it harms alignment scores. Likewise, the constraint formulations, such as the `[-1,1]` range and the preservation of the mean, are presented as heuristics without theoretical motivation or ablation against alternatives. While the paper shows these components contribute to performance, it fails to provide a clear rationale for why these specific choices are optimal.
- The majority of experimental evaluations are only on SD1.5, which is far w.r.t. performance from the models currently used in practice (SD3, FLUX, ...).

[1] Ma et al. "Inference-Time Scaling for Diffusion Models beyond Scaling Denoising Steps". CVPR 2025.

[2] Tang et al. "Inference-Time Alignment of Diffusion Models with Direct Noise Optimization". ICML 2025

[3] Eyring et al. "ReNO: Enhancing One-step Text-to-Image Models through Reward-based Noise Optimization". NeurIPS 2024.

### Questions
- **Generalization across noises**: Does an optimized prefix/suffix for a given prompt generalize to different initial noise seeds, or must the optimization be re-run for every new image generation? What is the typical wall-clock time for this inference-time optimization?  
- **Interpretability of Embeddings**: Have you attempted to project the learned prefix/suffix embeddings back into the vocabulary space to see if they correspond to interpretable words or concepts? What does IPGO learn to "say" to improve aesthetics or human preference?  
- **Rotation's Role**: The rotation component appears to have a task-dependent effect (improving aesthetics but not alignment in the ablation). Do you have an intuition for why this is the case? Could the rotation be made adaptive based on the reward function?

### Soundness
1

### Presentation
2

### Contribution
3
