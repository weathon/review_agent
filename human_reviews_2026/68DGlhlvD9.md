# UltraLLaDA: Scaling the Context Length to 128K for Diffusion Large Language Models

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Diffusion LLMs have attracted growing interest, with plenty of recent work emphasizing their great potential in various downstream tasks; yet the long‑context behavior of diffusion LLMs remains largely uncharted. We present a case study of post‑training techniques for extending the context window of diffusion LLMs (i.e., LLaDA) without retraining from scratch. We show that a simple modification to the standard Rotary Positional Embeddings (RoPE) extension effectively accommodates the probabilistic modeling inherent in the diffusion process, enabling stable scaling to longer context ranges. We further compare masking strategies used during post‑training and analyze their impact on optimization stability and long‑range recall. Instantiating these insights, we introduce UltraLLaDA, a diffusion LLM with a 128K‑token context window that, in our empirical evaluation on long‑context tasks, significantly outperforms training‑free baselines. Our experimental results highlight the special positional extension as a key lever for scaling diffusion LLMs to extended contexts and offer practical guidance for practitioners seeking 128K‑scale context via efficient post‑training.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces UltraLLaDA, a method for extending the context window of diffusion-based Large Language Models (LLMs) to 128,000 tokens through an efficient post-training process. The research addresses a critical and largely unexplored area, as the long-context capabilities of diffusion LLMs have not been systematically studied.   

The authors propose two main contributions:

Diffusion-aware NTK: A novel adaptation of the Neural Tangent Kernel (NTK) method for scaling Rotary Positional Embeddings (RoPE). The key insight is that diffusion models, with their bidirectional attention mechanism, learn a much wider range of relative positions during pre-training (approximately twice the context length) compared to auto-regressive models. By accounting for this property, the authors develop a more suitable RoPE scaling factor that enables stable extrapolation to very long contexts.   

Masking Strategy Analysis: The paper systematically investigates data packing and attention masking strategies to mitigate "cross-document interference" during long-context fine-tuning—a significant challenge for models with global bidirectional attention. It compares adaptive attention masking (which blocks attention between concatenated documents) and end-of-document (EOD) token concatenation against a naive direct concatenation baseline.   

Empirically, UltraLLaDA demonstrates remarkable performance. It achieves 100% accuracy on the "Needle-in-a-Haystack" (NIAH) retrieval task at the full 128K context length. Across a suite of benchmarks, including Perplexity, LongBench, and RULER, UltraLLaDA consistently and significantly outperforms the base LLaDA model and a training-free extension baseline (LongLLaDA), with the performance gap widening as context length increases. Ablation studies confirm that both the Diffusion-aware NTK and the use of boundary-aware masking strategies are essential for achieving these results.

### Strengths
Addresses a novel and important problem: long-context extension for diffusion LLMs. As diffusion models gain traction, understanding how to scale their context window is crucial for their practical application and competitiveness.   

The model's performance is a standout feature. Achieving perfect 100% accuracy on the 128K NIAH task is a powerful demonstration of the method's effectiveness in long-range information retrieval. The consistent and significant improvements over baselines across multiple diverse benchmarks (PPL, LongBench, RULER) provide robust evidence supporting the authors' claims.   

The core technical contributions are simple, intuitive, and clearly justified. The adaptation of NTK scaling is based on a clear-eyed observation of the architectural differences between diffusion and auto-regressive models. Furthermore, the systematic study of masking strategies provides valuable, practical insights for training such models.

The paper includes thorough ablation studies that successfully isolate the impact of each key component (the NTK variant and the masking strategy). This experimental rigor greatly strengthens the validity of the conclusions and clearly demonstrates that both proposed techniques are necessary for the final performance.

### Weaknesses
The narrow set of baselines, all comparisons are internal to the diffusion model LLaDA and LongLLaDA. Considering that this is relatively new in diffusion LLM, this is understandable, but perhaps could consider migrating other methods commonly used in auto-regression models for comparison, such as PI and YARN, to provide more insights.

The appendix reveals that UltraLLaDA's performance on standard short-context benchmarks degrades after long-context fine-tuning. This is a critical trade-off common in context extension methods but is not addressed in the main body of the paper. Acknowledging and analyzing this limitation in the main text would provide a more balanced and complete picture of the method's characteristics.   

The evaluation could be strengthened by incorporating more challenging long-context reasoning benchmarks to provide a more comprehensive assessment of the model's capabilities beyond information retrieval.

### Questions
Q1: In Section 3.2, the explanation provided for $T_{\text{cap}}$ and $T_{\text{Ecap}}$ being twice as large in diffusion LLMs compared to auto-regressive LLMs is intuitive. However, the argument would be significantly strengthened by a more detailed theoretical derivation or formal proof to rigorously support this conclusion.

Q2: The paper compares the model's training-free performance at critical dim = 64 versus 70. It would be insightful to see an analysis of performance at other proximal values (e.g., 69 or 71). Investigating whether 70 represents an optimal or near-optimal setting would provide stronger validation for the optimization process of the Diffusion-aware NTK method.

Q3: The evaluation of long-context capabilities currently concentrates primarily on retrieval tasks. To provide a more comprehensive assessment, please consider including benchmarks that evaluate long-context reasoning abilities, such as the relevant tasks in LongBench v2.

Q4: The experiments cap the maximum sequence length extension at 128k. Could the authors clarify the rationale for this specific limit? It would be beneficial to understand the method's performance at even greater lengths or, alternatively, to discuss the method's ultimate scaling limit (e.g., what is the maximum extension factor this approach can effectively achieve?).

### Soundness
2

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
This paper presents UltraLLaDA, a method for extending the context length of diffusion-based large language models to 128K tokens. The approach introduces a diffusion-aware NTK scaling technique and explores various masking strategies for handling long-context documents. Experimental results demonstrate the effectiveness of the proposed method.

### Strengths
1. The paper is well-written and easy to follow.

2. The proposed diffusion-aware NTK scaling method is simple yet effective, showing a certain level of novelty.

3. The paper conducts extensive experiments that clearly demonstrate the effectiveness of the proposed approach.

### Weaknesses
1. The proposed diffusion-aware NTK scaling is primarily based on an empirical assumption that a diffusion LLM can naturally handle a wider range of relative positions, with $T_{cap} \sim 2T_{train}$ and $T_{Ecap} \sim 2T_{target}$. It would be more convincing if the paper provided deeper theoretical justification or analysis for this assumption.

2. The investigation of different masking techniques for long documents shows limited novelty, as similar findings have already been discussed in prior studies on large language models[1,2].

[1] LongRoPE2: Near-Lossless LLM Context Window Scaling

[2]  The Llama 3 Herd of Models, https://arxiv.org/abs/2407.21783

### Questions
See the weaknesses section

### Soundness
3

### Presentation
3

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
This paper introduces UltraLLaDA, a post-training approach designed to extend the context window of diffusion-based large language models (specifically LLaDA) from 4K to 128K tokens. The authors make two main contributions: first, they develop a diffusion-aware NTK scaling method that refines standard RoPE extrapolation to better suit the bidirectional attention mechanisms of diffusion models, using a context cap of approximately twice the training length rather than the single-length limit typical of autoregressive models. Second, they explore several masking strategies for managing multi-document concatenation during training, comparing adaptive masking, the use of explicit end-of-document (EOD) tokens, and direct concatenation. UltraLLaDAdemonstrates strong performance on long-context benchmarks such as NIAH, LongBench, and RULER, outperforming the training-free baseline, LongLLaDA.

### Strengths
1. Systematic study of post-training methods for extending context in diffusion LLMs, addressing an underexplored area.
2. Well-motivated technical approach. The diffusion-aware NTK modification is intuitive, accounting for bidirectional attention means the model sees roughly 2x the relative position range during training.
3. Comprehensive evaluation. Multiple benchmarks (PPL, NIAH, LongBench, RULER) across context lengths up to 128k tokens
4. Lightweight post-training (600 steps) makes the approach accessible.

### Weaknesses
1. Single base model. All experiments use only LLaDA-8b as the base. Generalization to other diffusion LLMs or different model sizes is unclear.
2. LongLLaDA baseline cannot be evaluated beyond 32k, making comparisons incomplete at the longest contexts.
3. Masking strategy conclusions unclear. Tables 4-5 show adaptive masking and EOD concatenation trading advantages at different lengths, but the paper doesn't provide clear guidance on which to use when. The difference between all methods are very small and could be attributed to noise.
4. No comparison against YaRN, a more widely used RoPE interpolation method.

### Questions
1. Can you provide theoretical or empirical analysis showing that diffusion models actually learn relative positions in the range [-2T_train, 2T_train] during pre-training?
2. Why was NTK-aware/ABF RoPE scaling used instead YaRN? Is there a intrinsic problem with YaRN that prevents it from being adapted to diffusion models? SoTA models are mostly always using YaRN instead of NTK scaling.
3. Have you tested this approach on other diffusion LLMs or other model sizes? Would it work without finetuning on larger model sizes? What about other architectures?

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
3

### Summary
This paper introduces UltraLLaDA, a diffusion LLM post-trained from LLaDA with a modified RoPE scaling method to extend the context length to 128K. UltraLLaDA proposes a diffusion-aware NTK/RoPE scaling method. The authors argue that standard NTK-aware RoPE scaling (as used in autoregressive long-context extension) is suboptimal for diffusion LLMs because diffusion models use bidirectional attention rather than causal attention. They propose a modified scaling that better reflects the positional distance statistics of bidirectional denoising. They then conduct long-context post-training on 64K length packed sequences, exploring several strategies for handling multiple concatenated documents: naive concatenation, EOD-token concatenation, and adaptive attention masking. The experimental results show that UltraLLaDA maintains near-perfect Needle-in-a-Haystack retrieval up to 128K and keeps perplexity stable out to 128K, whereas both the original LLaDA and the training-free method LongLLaDA collapse much earlier.

### Strengths
1. **Good practical significance for diffusion LLMs.**  
   Ultra-long context capability in diffusion LLMs has not been well explored. This work provides a realistic recipe to reach 128K with stable retrieval and usable perplexity, which is a meaningful capability milestone for dLLMs.

2. **Comprehensive study of the proposed diffusion-aware NTK.**  
   The paper presents a detailed analysis of the proposed diffusion-aware NTK, comparing it with LongLLaDA's baseline NTK, and shows the empirical improvement of diffusion-aware NTK in the training-free setting. This analysis is convincing.

3. **Comprehensive experimental results and ablations.**  
   This work conducts experiments on various long-context tasks, including Needle-in-a-Haystack, LongBench, and RULER, for all baselines (LLaDA, LongLLaDA) and UltraLLaDA trained with different sentence packing strategies. This thorough comparison supports the proposed method.

4. **Clear writing.**  
   The paper is clearly written and easy to follow.

### Weaknesses
1. **Core novelty feels incremental.**  
   The main contribution of this work is diffusion-aware NTK. While the motivation (bidirectional vs. causal attention) is reasonable, the scaling rule itself shows only a moderate improvement over LongLLaDA’s baseline scaling when evaluated without post-training. From Table 4 and Table 5, even with post-training, the improvement for 4K–16K context lengths still seems small. The three long-context post-training sentence-packing strategies explored in this paper are also existing methods in autoregressive post-training. Applying and comparing them in the diffusion LLM setting is empirically valuable but not conceptually new.

2. **Limited analysis of non-retrieval reasoning at longer context lengths.**  
   Most ultra-long-context evaluations are on NIAH and perplexity stability. The paper does evaluate on LongBench with 16K context length and RULER at 32K, but there is less evidence for complex multi-document synthesis or instruction following at 64K–128K.

3. **Lack of evaluation on more models.**  
   The paper only conducts experiments on LLaDA model. There are multiple kinds of diffusion LLMs: models trained from scratch in diffusion style like LLaDA, models converted from AR like Dream, and block-diffusion-style models. This work should apply the proposed method to more diffusion LLMs to demonstrate generalization.

### Questions
1. Could you evaluate UltraLLaDA on more kinds of tasks beyond PPL and NIAH at longer context lengths? Are there any failure cases at 128K context length (because the current NIAH evaluation results seem perfect)?

2. Could you apply the proposed method to models like Dream (which is trained from AR model) and SDAR-series models (which are also trained from AR models with block diffusion style)?

### Soundness
3

### Presentation
3

### Contribution
2
