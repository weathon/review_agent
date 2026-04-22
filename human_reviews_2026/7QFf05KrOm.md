# Enhancing Multi-Image Understanding through Delimiter Token Scaling

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
Large Vision-Language Models (LVLMs) achieve strong performance on single-image tasks, but their performance declines when multiple images are provided as input.
One major reason is the cross-image information leakage, where the model struggles to distinguish information across different images.
Existing LVLMs already employ delimiter tokens to mark the start and end of each image, yet our analysis reveals that these tokens fail to effectively block cross-image information leakage.
To enhance their effectiveness, we propose a method that scales the hidden states of delimiter tokens.
This enhances the model’s ability to preserve image-specific information by reinforcing intra-image interaction and limiting undesired cross-image interactions.
Consequently, the model is better able to distinguish between images and reason over them more accurately.
Experiments show performance gains on multi-image benchmarks such as Mantis, MuirBench, MIRB and QBench2. 
We further evaluate our method on text-only tasks that require clear distinction. 
The method improves performance on multi-document and multi-table understanding benchmarks, including TQABench, MultiNews and WCEP-10. 
Notably, our method requires no additional training or inference cost.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The method is evaluated on a wide range of models and benchmarks, demonstrating consistent performance improvements on multi-image understanding tasks (Mantis, MuirBench, etc.). Crucially, the authors show the method's generality by applying it to text-only multi-document and multi-table tasks, again achieving performance gains. The proposed approach requires no additional training or inference costs.

### Strengths
The paper's primary strength is its clear and insightful analysis of how delimiter tokens function and where they fall short. The "image-wise tagging" concept provides a strong theoretical motivation for the proposed solution.

The method is remarkably simple (a single scaling operation) and highly efficient (no training, no inference overhead, compatible with optimizations like FlashAttention). This makes it a very appealing and practical technique.

The effectiveness of the method is demonstrated across a wide variety of models (Qwen2.5-VL, InternVL3, LLaVA-OV), model sizes (0.5B to 32B), and task domains (multi-image, multi-document, multi-table). This shows it is a general principle for improving multi-instance understanding, not a model-specific trick.

### Weaknesses
The paper states that the scaling layer and factor λ are tuned for each model. The appendix mentions the scaling layer is fixed for each model, but the process for choosing this layer is not detailed. A more principled explanation or heuristic for selecting the optimal layer(s) to apply scaling would make the method even more robust and easier to adopt. The current approach feels slightly post-hoc.

The method involves scaling a single chosen layer's hidden states. A deeper analysis on why a particular layer is optimal or an exploration of scaling multiple layers could provide further insights into the model's internal workings.

### Questions
please see the weaknesses.

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
This paper addresses a critical limitation of LVLMs, i.e., cross-image information leakage in multi-image tasks, by scaling delimiter token hidden states, a simple yet effective training/inference-cost-free method. Experiments across multi-image (Mantis, MuirBench) and text-only (MultiNews, TQABench) benchmarks show consistent gains, validating its robustness and generality. The analysis of delimiter tokens’ two key properties also provides valuable insights into LVLM attention mechanisms.

### Strengths
- Solves cross-image leakage in LVLMs via delimiter scaling, with gains in multi-image/text tasks, no extra cost.
- Analyzes delimiter tokens’ key properties, offering clear theoretical basis for the method.
- Generalizes to text multi-instance tasks, works across models (0.5B–32B), and fits optimized kernels.

### Weaknesses
- Though claiming minimal impact on text-image interaction, it only mentions a 10% drop in text-to-image attention scores without detailing how this drop affects downstream cross-modal tasks (e.g., image-text retrieval), leaving uncertainty about real-world cross-modal performance

### Questions
- Your work shares similarities with parallel context encoding [1] and attention sink, both focusing on context separation/attention regulation. How does your delimiter scaling method relate to attention entropy (a core factor in Zhang et al.’s work)? Does scaling delimiter states reduce cross-image attention entropy, and if so, how? 
- Is there a run-time analysis to show the exact additional inference cost? 


[1] Attention Entropy is a Key Factor: An Analysis of Parallel Context Encoding with Full-attention-based Pre-trained Language Models
, Z Zhang et al, ACL 2025.

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
This paper presents a training-free method for limiting cross-image interactions in LVLMs by scaling the hidden states of delimiter tokens across multiple images. Comprehensive evaluation results on multi-image, multi-document, and multi-table benchmarks are provided to demonstrate the effectiveness of the proposed approach. Both qualitative and quantitative analyses are included to illustrate the role and impact of image delimiter tokens.

### Strengths
1. The problem of cross-image information leakage in multi-image LVLM settings is important and worth investigating.
2. The prior analysis on delimiter tokens and the characterization of their key properties is clear and insightful, helping readers better understand the mechanism.
3. The experiments are extensive, covering multiple LVLM families and sizes, four multi-image benchmarks, two multi-document benchmarks, and one multi-table benchmark.

### Weaknesses
1. The concept of “sink tokens” has been studied in prior works, and there are also existing methods addressing cross-image leakage. Thus, the novelty and significance of the current findings appear limited.
2. The technical contribution of the proposed method is relatively weak. Scaling the hidden states of delimiter tokens provides only marginal performance gains. For instance, when applied to larger LVLMs such as InternVL3-14B or Qwen2.5-VL-32B, the improvements are minimal (e.g., 42.42 → 42.58).

### Questions
1. Could the authors propose a more technically substantial method or extend the current approach with additional mechanisms? Further ablation studies building on the discovered delimiter token properties may help strengthen the contribution and highlight the novelty of the work.
2. The writing could be further polished for clarity, and several figures and tables could be improved for readability. Enhancing the presentation quality would significantly improve the paper.

### Soundness
3

### Presentation
2

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
This work focuses on interleaved inputs for multi-modal large language models. 

It is motivated by the hypothesis that **adjacent frames carry stronger contextual relevance.** 

Unlike previous interleaved approaches that rely on special textual tokens, this work introduces a delimiter token reweighting mechanism.

### Strengths
The interleaved image–text processing still lacks clarity in terms of how information flows across modalities. The explanation within MLLMs remains underexplored, though this work makes a valuable attempt to highlight the importance of key regions.

The motivation is clear.

The core idea is **very simple yet interesting**, and the method section is clearly presented. The method do not bring additional training cost and just reweight the hidden states.

### Weaknesses
1. The main concern is the **limited scope of evaluation**. The paper focuses primarily on math and multi-view benchmarks, whereas multi-image input represents a special case of __interleaved data__ that can be applied to a broader range of scenarios. The performance under few-shot settings, where multiple instances are concatenated together, remains unclear and differs from the explored benchmarks.

2. The performance improvements are sometimes marginal, suggesting limited generalization.

3. The proposed reweighting operation introduces inconvenience during inference, as it requires modification of hidden states and is therefore applicable only to open-source models.

### Questions
1. In Appendix A.1, several details require clarification. For instance, what are the four images and the corresponding text shown? How many samples were used to construct this figure? If the images share visual similarity, would the corresponding results change?

2. Can the proposed method be extended to interleaved inputs (e.g., alternating image–text sequences)?

3. For multi-image inputs, could there be potential attention sink issues similar to those discussed in “When Attention Sink Emerges in Language Models: An Empirical View”?

I may reconsider my evaluation after reading the authors’ rebuttal.

### Soundness
3

### Presentation
3

### Contribution
3
