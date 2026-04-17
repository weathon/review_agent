# AdaptInfer: Adaptive Token Pruning for Vision-Language Model Inference with Dynamical Text Guidance

- Decision: Reject
- Scores: 4, 2, 4, 4

## Abstract
Vision-language models (VLMs) have achieved impressive performance on multimodal reasoning tasks such as visual question answering, image captioning and so on, but their inference cost remains a significant challenge due to the large number of vision tokens processed during the prefill stage. Existing pruning methods often rely on directly using the attention patterns or static text prompt guidance, failing to exploit the dynamic internal signals generated during inference. To address these issues, we propose AdaptInfer, a plug-and-play framework for adaptive vision token pruning in VLMs. First, we introduce a fine-grained, dynamic text-guided pruning mechanism that reuses layer-wise text-to-text attention maps to construct soft priors over text-token importance, allowing more informed scoring of vision tokens at each stage. Second, we perform an offline analysis of cross-modal attention shifts and identify consistent inflection locations in inference, which inspire us to propose a more principled and efficient pruning schedule. Our method is lightweight and plug-and-play, also generalizable across multi-modal tasks. Experimental results have verified the effectiveness of the proposed method. For example, it reduces CUDA latency by 61.3\% while maintaining an average accuracy of 93.1\% on vanilla LLaVA-1.5-7B. Under the same token budget, AdaptInfer surpasses SOTA in accuracy.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a training-free token pruning method. It directly leverages the attention computation mechanism of the base model for pruning. Furthermore, the paper provides insightful guidance on which layers are suitable for pruning and which are best left alone. The experiments in the paper are relatively comprehensive and provide strong support for the authors' claims.

### Strengths
The author uses experiments to heuristically verify which layers are more suitable for pruning and which are not. For example, the attention shift is more obvious in the 1st, 10th, and 20th layers.

### Weaknesses
1. The claims in 3.1.3 are based on experiments, but the authors tested them on only one model. They provide a discussion of generalizability, but there are limitations on whether these claims can be generalized to different models.
2. Tables 1 and 2 are too far removed from the analysis of the results to be easily readable.

### Questions
1. Based on the experiments in the author's methodology, I am curious whether this claim can be generalized to different tasks? For example, do classification tasks and QA tasks perform consistently across different layers? In addition, intuitively, classification tasks should be able to tolerate more aggressive pruning, so is it acceptable to prune layers where attention shift is obvious?
2. The method proposed by the author can prevent important information from being drop. Does this method ensure that the knowledge of the model remains consistent? As far as I know, token pruning methods can ensure that overall performance does not drop significantly. However, they may introduce knowledge boundary drift, meaning that answers that were previously correct may now be incorrect, and vice versa [1]. This possibility could have negative consequences in scenarios where critical questions must be answered correctly. If possible, could the authors conduct a simple analysis to see whether the proposed method leads to changes in individual case performance?

[1] Sun, Yizheng, et al. "Does Acceleration Cause Hidden Instability in Vision Language Models? Uncovering Instance-Level Divergence Through a Large-Scale Empirical Study." arXiv preprint arXiv:2503.06794 (2025).

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents AdapterInfer, a plug-and-play training-free adaptive token pruning approach for VLMs. First, they use layer-wise text-to-text attention map to enhance the text-based vision-token pruning, use important text tokens to mine important vision tokens. Second, they identify consistent inflexion locations in inference via analysis of cross-modal attention in LLaVA, and improve the pruning schedule.

### Strengths
1. **Reasonable Design.** Using important text tokens to mine important vision tokens is reasonable.

### Weaknesses
1. **Weak Baseline.** This paper uses LLaVA-1.5 as a baseline, which is an open-source VLM released 2 years ago. It's too weak and completely outdated. For a training-free method, the evaluation should be conducted on more recent, competitive and widely used VLMs, such as Qwen2.5-VL.

2. **Lack Generalizability.** The consistent attention inflection point found in this paper is based on LLaVA-1.5. The author should check other base VLMs like Qwen2-VL and Qwen2.5-VL for a similar situation.

3. **Lack Comparison.** This paper should compare with newer baselines, like VisionZip[1], a widely used efficient VLM approach accepted in CVPR25.

4. **Cannot work with flash-attention.** AdapterInfer based on t2t and t2v attention score, which cannot be obtained when inference with flash-attention.

[1] VisionZip: Longer is Better but Not Necessary in Vision Language Models

### Questions
1. **Reasoning VLMs.** Reasoning VLMs also face the problem of token redundancy. The author should discuss whether this method can be applied to Efficient Reasoning VLMs[1].

2. **Resize.** In [1], the researchers discussed that in most general scenarios, even simple resizing can achieve strong performance. How do the authors view this issue? I look forward to the author discussing this point.

[1] VisionThink: Smart and Efficient Vision Language Model via Reinforcement Learning

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
This paper proposes an adaptive token pruning method for VLMs, of which the core design is a plug-and-play framework to reuse the text-to-text attention maps for a soft prior of text-token importance. In the experiments, The proposed method outperforms FastV, SparseVLM, and other baselines on the average accuracy of 5 VQA benchmarks, showing the effectiveness of the AdaptInfer module.

### Strengths
- The proposed dynamic cross-attention guided visual token pruning is enhanced by reusing text-token attantion to put a higher focus on important text tokens.
- The proposed AdaptInfer leverage the observation of attention distribution shift to guide choices of hyperparameters like the insertion location of the pruning layer.
- The proposed method provides notable savings in inference latency.

### Weaknesses
- The proposed method is only verified on 5 different VQA datasets, whereas the baseline methods are usually evaluated on much more diverse benchmarks. For example the PyramidDrop is evaluated on 16 different benchmarks.
- Some benchmarks are also evaluated on video VQA benchmarks, including PyramidDrop and SparseVLM. A more well-rounded comparison will be more convincing especially when the improvement over SparseVLM is not consistent.
- The newer SOTA benchmark VisionZip [1] is not included in the comparison.
[1] Yang, Senqiao, Yukang Chen, Zhuotao Tian, et al. n.d. VisionZip: Longer Is Better but Not Necessary in Vision Language Models.

### Questions
See weaknesses.

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
3

### Summary
The authors propose "AdaptInfer," a plug-and-play, training-free framework designed to address these two specific problems.

First, the core technical novelty is a dynamic text-guided pruning mechanism. Instead of using a static set of "key" text tokens, AdaptInfer re-computes text-token importance at each designated pruning layer. It cleverly reuses the existing text-to-text (t2t) attention maps (which are already computed) to create a "soft prior" distribution over the text tokens. This prior is then used to reweight the text-to-vision (t2v) attention, yielding a more informed and context-aware ranking of vision tokens for pruning. This is an elegant solution, as it introduces minimal computational overhead.

Second, to replace heuristic pruning schedules, the authors conduct an offline analysis to identify "attention inflection points." By applying change-point detection to the cumulative t2v attention trajectories, they discover consistent, data-driven locations in the architecture (e.g., layers 1, 10, and 20 for LLaVA-1.5-7B) where the model's utilization of visual information significantly shifts. These inflection points are then used as principled locations for applying the pruning.

### Strengths
- The method is both training-free and, critically, reuses attention maps that are already computed during the forward pass. This means it introduces almost no additional overhead (Sec 3.2.2), making it an extremely practical solution for real-world inference acceleration.

- The paper shows clear and consistent performance gains over its closest SOTA-level competitor (SparseVLM) across multiple benchmarks and token budgets (Table 1, Fig 3). The latency test (Table 3) confirms that these theoretical gains translate to real-world speedups.

- The primary contribution—using dynamic text-to-text attention as a proxy for text-token importance—is a novel and elegant solution to the limitations of static guidance. It's theoretically well-motivated by the (correct) observation that information importance evolves by layer.

### Weaknesses
- The specific schedule (layers 1, 10, 20) is architecture-dependent (LLaMA-7B). While the method for finding the schedule is general, it requires a new, non-trivial offline analysis (running 1000+ samples through the model and performing change-point detection) for every new backbone

- The main paper emphasizes the locations of pruning, but the amount to prune at each location is also a critical hyperparameter. Appendix G.3 mentions that these "pruning ratios" are also selected, following a method from prior work. The sensitivity of the model to these ratios, and how they interact with the pruning locations, is not fully explored, adding a layer of tuning that is under-discussed.

- In Sec 3.1.3, the authors posit that an attention shift can mean a token is either becoming critical or redundant. The logic in Sec 3.2.3 for building the schedule around these points ("pruning should avoid the regions with intense attention shifts... pruning just before or after the dense regions") feels slightly post-hoc and could be articulated more clearly. For instance, why is pruning at layer 1 (an inflection point) and at layer 10 (the start of a dense region) the optimal strategy?

### Questions
The questions are mainly covered by the weakness above.

### Soundness
3

### Presentation
3

### Contribution
3
