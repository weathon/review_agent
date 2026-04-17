# IVC-Prune: Revealing the Implicit Visual Coordinates in LVLMs for Vision Token Pruning

- Decision: Accept (Poster)
- Scores: 6, 4, 8, 6

## Abstract
Large Vision-Language Models (LVLMs) achieve impressive performance across multiple tasks. A significant challenge, however, is their prohibitive inference cost when processing high-resolution visual inputs. While visual token pruning has emerged as a promising solution, existing methods that primarily focus on semantic relevance often discard tokens that are crucial for spatial reasoning. We address this gap through a novel insight into how LVLMs process spatial reasoning. Specifically, we reveal that LVLMs implicitly establish visual coordinate systems through Rotary Position Embeddings (RoPE), where specific token positions serve as implicit visual coordinates (IVC tokens) that are essential for spatial reasoning. Based on this insight, we propose IVC-Prune, a training-free, prompt-aware pruning strategy that retains both IVC tokens and semantically relevant foreground tokens. IVC tokens are identified by theoretically analyzing the mathematical properties of RoPE, targeting positions at which its rotation matrices approximate identity matrix or the $90^\circ$ rotation matrix. Foreground tokens are identified through a robust two-stage process: semantic seed discovery followed by contextual refinement via value-vector similarity.  Extensive evaluations across four representative LVLMs and twenty diverse benchmarks show that IVC-Prune reduces visual tokens by approximately 50\% while maintaining $\geq$ 99\% of the original performance and even achieving improvements on several benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This work presents a novel method for visual token pruning for VLMs. Analyses reveal that there are specific token positions as implicit visual coordinates (IVC tokens) that are essential for spatial reasoning. The proposed method, IVC-prune, is based on this insight and retains both IVC tokens and semantically-related tokens. IVC tokens are identified by theoretically analyzing ROPE, while semantic tokens to keep are identified through a two-stage process of discovery and refinement. Experiments with 4 VLMs and 20 benchmarks illustrate the effectiveness of the proposed method.

### Strengths
- The proposed method tackles an important problem in VLMs to improve their inference efficiency.
- The proposed approach is shown effective on a variety of models and benchmarks.

### Weaknesses
- The introduction of IVC tokens brings the configurations of extra hyper-parameters such as setting token preservation ratios. It seems that the best settings depend on the underlying models, and it is unclear whether there can be a principled way to select these hyper-parameters.
- It would better if there can be more analyses and insights on the benefits of the proposed method; why they work better than others?
- The IVC seems to be derived by only considering Rm and Rn individually, but in ROPE, they are multiplied together for relative positional encoding. I'm not quite sure about the motivation here; there should be more illustration on this point (S3.2).

### Questions
(Please refer to the weakness part.)

### Soundness
2

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
This paper introduces IVC-Prune, a training-free, text-aware pruning approach for LVLM inference accelaration. Specifically, IVC-Prune identifies some reference tokens from implicit visual coordinates, and presearves those IVC tokens and foreground tokens for effective reasoning. Experimental results on standard benchmakes across four widely used LVLMs validate the effectiveness of the proposed approach.

### Strengths
1. The proposed approach seems to be simple yet effective and can be applied to LVLMs with different architectures.
2. Extensive experimental results validate the effectiveness of the proposed approach.
3. The paper is generally well-written and the stucture is clear.

### Weaknesses
1. While the empirical experiments demonstrate the significant impact of the IVC tokens, I would be more convinced if a more detailed analysis were provided to explain *why* these tokens are important.
2. Relevant baselines employing window-based token selection approaches should be included in the main experiments, as they also aim to preserve spatial information along with the foreground tokens. Additionally, a discussion on novelty is needed to better differentiate this work from those related methods.
3. The foreground token selection strategy seems to primarily combine existing methods rather than introducing fundamentally new ideas.
4. I would like to see an evaluation of the proposed approach under extremely low token budgets (e.g., 11.1%–33.3%), as reported in prior work [3], to better understand its robustness.

[1] Token Pruning in Multimodal Large Language Models: Are We Solving the Right Problem? https://www.arxiv.org/abs/2502.11501

[2] VScan: Rethinking Visual Token Reduction for Efficient Large Vision-Language Models. https://arxiv.org/abs/2505.22654

[3] PyramidDrop: Accelerating Your Large Vision-Language Models via Pyramid Visual Redundancy Reduction. https://arxiv.org/abs/2410.17247

### Questions
See the weaknesses mentioned above. I am open to updating my rating if they are properly addressed.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposed a new method for visual token pruning in VLMs. Starting with a theoretical analysis of ROPE embeddings, the authors pointed out that ROPE implicitly encode spatial information already when processing visual tokens. In particular, at positions where the rotation matrix is close to an identity matrix or 90 degree rotation matrix, the model can learn to use those positions as anchors to better understand the visual inputs. Based on this observation, they proposed always keep visual tokens at these positions during pruning, and further keep semantically salient tokens to maximize task performance (that are computed based on similarity between value vectors instead of attention scores, to minimize the impact of positional bias).  The actual pruning is determined at an intermediate layer in VLM, and once the tokens are decides, the entire layers’ KV cache for those pruned tokens are removed. The experiments with 4 different VLMs on a wide range of visual understanding tasks showcase the effectiveness of the proposed approach.

### Strengths
Unlike most previous methods on this area, this paper actually conducted a nice theoretically analysis of the working mechanism in VLMs, and the strong empirical results further supported the analysis. 

The pruning strategy is well designed, once the tokens for pruning are decided in the first forward pass, all layers KV cache can be cleaned to maximize the saving in compute and memory. 

The experiments with 4 different VLMs with different architectures, image handling strategies all show very strong performance across a wide range of tasks (including visual grounding, fine-grained perception tasks, which are often overlooked), clearly showing the advantage of the proposed method. 

The ablations are well done to verify the effectiveness of IVC tokens, and explains well why previous works might have different findings → the IVC tokens are pruned in the earlier layers.

### Weaknesses
I don’t have any major concerns. One minor issue is that the methods depends on the property of RoPE, thus the generalizability to other model architectures with different position embeddings is unknown.

### Questions
None

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
4

### Summary
The paper proposes a novel token pruning strategy, called IVC-Prune. The central idea of the paper is that LVLMs implicitly establish visual coordinate systems through RoPE, and certain token positions act as implicit visual coordinates (IVC tokens) that are crucial for spatial reasoning tasks. The authors propose a training-free, prompt-aware pruning strategy that retains both IVC tokens and semantically relevant foreground tokens. This method significantly reduces the number of visual tokens while preserving or even improving performance on various vision tasks.

### Strengths
1. The paper introduces a new perspective on token pruning by focusing on the IVC tokens for spatial reasoning in LVLMs. The idea of implicit visual coordinate from RoPE is novel.
2. The experiment results are impressive. The performance can approach or even surpass the vanilla model under 50% compression.

### Weaknesses
1. The paper lacks comparison with more recent baselines. FastV is already an weaker baseline, and it would be beneficial to include a comparison with SparseVLM.
2. The paper does not clarify how IVC tokens and foreground tokens should be allocated under a 50% total budget. Ablation experiments should be included to investigate this.

### Questions
1 & 2: Please see the weakness.
3. Tables 3 and 5 do not include data for PDrop. 
4. It is recommended to add results on high-resolution benchmarks, such as DocVQA and OCRBench, as these benchmarks involve a larger number of visual tokens.

### Soundness
3

### Presentation
3

### Contribution
3
