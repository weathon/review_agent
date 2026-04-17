# BlockSpec: Blockwise Speculative Decoding for Diffusion LLMs

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
In diffusion-based Large Language Models (dLLMs), parallel decoding is usually realized through threshold-based or top-k strategies. While effective in high-confidence tokens, these strategies often collapse on low-confidence tokens, forcing the model into inefficient single-token decoding. To address this limitation, we propose Block Speculation (BlockSpec), a novel training-free blockwise speculative decoding method that explores multiple future decoding trajectories in parallel. Our method introduces a new tree-based trajectory generation strategy and a blockwise parallel verification module, where decoding tokens are organized into tree exploration paths and then multiple decoding trajectories can be simultaneously verified. Unlike traditional speculative decoding that focuses only on fixed-order left-to-right token speculation, our approach is the first attempt to introduce block-level speculation, which jointly explores both token choices and decoding trajectories for dLLMs. We also design two complementary speculation formulations—intra-block and inter-block speculation—that jointly accelerate dLLMs within and across blocks. Extensive experiments show that the proposed BlockSpec model reduces iteration steps by up to 40\%, accelerating over 80\% of decoding steps. As a result, our model achieves up to 7–14× speedup over vanilla dLLMs, together with an additional 1.3× improvement over state-of-the-art methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces BlockSpec, a novel, training-free speculative decoding framework designed to mitigate the "low-confidence degradation" problem in diffusion-based Large Language Models (dLLMs), where parallel decoding efficiency collapses under uncertainty. BlockSpec addresses this by exploring multiple decoding trajectories simultaneously using a new tree-based generation strategy to create candidate token blocks in parallel. These blocks are then efficiently validated through a corresponding blockwise verification mechanism tailored for the any-order, bidirectional nature of dLLMs, with performance further enhanced by intra- and inter-block speculation. Experimental results demonstrate that this approach significantly reduces decoding steps, achieving up to a 14x speedup over vanilla dLLMs and a 1.3x improvement over strong baselines, establishing a more robust and efficient inference paradigm.

### Strengths
The proposed BlockSpec in the paper is a novel training-free decoding acceleration strategy that attempts to consider more branches to alleviate the degradation problem in parallel decoding. Experimental results show that this method is effective on different datasets, different models, and different hardware.

### Weaknesses
1. In Section 3.3, the authors state that “…allows key–value caching on the prompt…”. However, the rest of the paper (including the experimental settings) does not clarify whether KV caching was actually used. If KV cache was indeed applied, the authors should provide its design details and implementation description.

2. The stated motivation of the paper is to address the “low-confidence degradation problem of parallel decoding.” However, intuitively, the proposed method should lead to better performance, which is not reflected in the experimental results. I suggest that the authors revise the Introduction to better align it with the presented method and findings.

3. In Figure 3(3), for the current block, the subsequent masked tokens (the rightmost three blocks with ellipses) should be invisible. Is this a plotting error? This visualization seems to indicate that the model is aware that this is the final block to be generated, which could substantially influence the effective generation length and thereby degrade model performance.

### Questions
1-2. See weakness 1 and 3.
3. Although Section 3.4 introduces the concept of inter-block speculation, it still lacks sufficient implementation details. For example, it does not appear in any of the provided pseudocode segments. My question is: if both the current block and the next block are generated simultaneously, and each block is supposed to attend to the prompt as well as all preceding blocks, then which branch of the current block should the next block attend to?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes BlockSpec, a training-free, blockwise speculative decoding method for diffusion language models (dLLMs). Through tree-structured trajectory generation and block-level parallel verification, the method achieves efficient decoding with high parallelism even under low-confidence conditions. The paper further introduces intra-block and inter-block speculation mechanisms, evaluates performance on reasoning and code-generation benchmarks, and analyzes computational overhead and limitations, reporting up to a 14× speedup over a vanilla dLLM.

### Strengths
1. The proposed BlockSpec framework delivers substantial decoding acceleration, reducing iteration steps by up to 40% and achieving 7–14× speedups over vanilla diffusion LLMs, while maintaining comparable output quality on reasoning and code-generation benchmarks.

2. By combining tree-based trajectory generation with block-level parallel verification, the method effectively mitigates low-confidence degradation in diffusion LLMs, sustaining parallelism without compromising accuracy.

### Weaknesses
1. According to Fig. 3(3), all draft blocks are invisible to subsequent masked tokens. Thus, when unmasking a block, the process essentially reduces to dLLM-style generation with a generation length equal to the block size, which can lead to issues such as prematurely forcing an answer or emitting eos early. This suggests that the current blockwise attention mask is problematic.

2. Based on Tables 1 and 2, BlockSpec’s high throughput (TPS) speedup seems attributable not only to fewer forward passes but also to reduced matrix computation induced by the specialized attention mechanism. However, the experiments and appendix do not clearly isolate or quantify this contribution.

### Questions
1. See Weakness 1.

2. What is the structure of the blockwise attention mask in inter-block speculation.

3. The experiments appear to show that a blockwise attention mask that breaks the bidirectional attention mechanism can still yield generation that is nearly lossless relative to full bidirectional attention. Have the authors investigated the underlying reasons for this?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
Common unmasking strategies, such as 'confidence threshold' and 'top-k', are slower on low-confidence tokens since only a single token can be sampled per step. The authors argue that low confidence occurs when dLLMs consider multiple possible trajectories. The paper, therefore, proposes a novel self-speculative decoding method called BlockSpec, which simultaneously searches for and verifies possible trajectories to select multiple tokens. BlockSpec achieves much higher throughput than the baseline and is twice as fast as concurrent work.

### Strengths
- BlockSpec is a customised, self-speculative decoding method designed specifically for dLLMs. It effectively addresses issues such as multi-token verification and changing the hidden state prefix.
- The paper also proposes a novel inter-block speculation method that substantially increases block efficiency.
- BlockSpec achieves a much higher TPS speedup compared to the baselines.
- BlockSpec is twice as fast as concurrent work.

### Weaknesses
- W1: There is no empirical evidence to support the argument that the low confidence tokens are due to plausible candidates. For example, the model would display low confidence if the problem was too complex for it to handle.
- W2: Since dLLMs use bidirectional attention, the hidden state of the prefix tokens varies as the model explores different nodes in the search tree. Therefore, the method is based on Fast-dLLM, which fixes the prefix token during the denoising process. However, since prefix caching leads to performance degradation, dependency on Fast-dLLM must be a weakness.
- W3: Despite its fast speed, BlockSpec shows similar performance to Fast-dLLM (dual cache). Formal Speculative Decoding methods maintain the original performance of the target models.
- W4: There has been no ablation study on the role of 'medium-confidence tokens'. 
- W5: Short generation length. The generation lengths are fixed to 512, which is relatively short.

### Questions
- Typo in section 2 "When facing everal..."
- Is there any way to find an optimal tree configuration for arbitrary settings, such as model size, dataset, GPU FLOPS, and GPU memory bandwidth?
- W1: Could you show the results for more complex tasks, such as AIME?
- W3: Where does the performance degradation come from? I believe that the Fast-dLLM's performance would be the upper bound.

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
4

### Summary
The paper proposes Speculative Parallel Decoding (SPD) for diffusion-based LLMs, a topic gaining traction in the community. The method is training-free, maintains near-lossless performance on studied benchmarks, and introduces a simple draft tree generation algorithm with verification. The speed-up gains are good

### Strengths
- Proposes SPD for diffusion LLMs, which is an emerging and relevant area.
- Training-free approach, reducing complexity and resource requirements.
- Maintains almost lossless performance across evaluated benchmarks.
- Introduces a simple draft tree generation algorithm combined with verification.
- Works effectively with multi-token unmasking while maintaining accuracy.

### Weaknesses
- Draft tree generation algorithm lacks clarity in certain cases (e.g., W3D3(6) configuration).
- Missing citations for speculative decoding and diffusion-related prior work.
- Some results and configurations are not fully explained (e.g., latency trade-offs, inter-block setup).

Please see questions for more details

### Questions
Draft Tree Generation:
For W3D3(6), why is there no [1,4] node at step (or level) 2, given that the likelihood of [1,4] > [1,3,4] always?


Main Results:
Could the authors clarify that W2D2(3) was chosen because it offers lower latency compared to W3D3(6) on A800 early on in the results section?


Figure 5:
What does “base” refer to, and on which dataset are these results generated? Also, what is the difference between Fast-dLLM (dual cache) and the proposed method on H800, given that Fast-dLLM performs second best on A800?


Table 1:
HumanEval(0) shows BlockSpec outperforming baseline methods, while in other benchmarks BlockSpec performs slightly worse (as expected due to lack of lossless generation). Is there any justification for this anomaly?


Table 3:
Inter-block has the highest average tokens per step—what about its latency? Is the inter-block setup used only for 2 blocks? If extended to more than 2 blocks, does it start to hit TPS limits? Also, when doing inter-block, are draft trees for respective blocks generated independently?


Missing Citations:
Citations for speculative decoding and diffusion-based approaches are missing. This idea essentially extends speculative decoding for images as discussed in prior works.

- Diffusion Speculation


1. Diffusion Models are Secretly Exchangeable: Parallelizing DDPMs via Autospeculation
Hengyuan Hu, Aniket Das, Dorsa Sadigh, Nima Anari
arXiv:2505.03983


2. Accelerated Diffusion Models via Speculative Sampling
Valentin De Bortoli, Alexandre Galashov, Arthur Gretton, Arnaud Doucet
arXiv:2501.05370


- Speculative Decoding


1. Direct Alignment of Draft Model for Speculative Decoding with Chat-Fine-Tuned LLMs
Raghavv Goel, Mukul Gagrani, Wonseok Jeon, Junyoung Park, Mingu Lee, Christopher Lott
arXiv:2403.00858


2. Recursive Speculative Decoding: Accelerating LLM Inference via Sampling Without Replacement
Wonseok Jeon, Mukul Gagrani, Raghavv Goel, Junyoung Park, Mingu Lee, Christopher Lott
arXiv:2402.14160

### Soundness
2

### Presentation
2

### Contribution
3
