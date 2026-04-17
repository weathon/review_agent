# Distilling to Hybrid Attention Models via KL-Guided Layer Selection

- Decision: Accept (Poster)
- Scores: 6, 4, 8, 6

## Abstract
Distilling pretrained softmax attention Transformers into more efficient hybrid architectures that interleave softmax and linear attention layers is a promising approach for improving the inference efficiency of LLMs  without requiring expensive pretraining from scratch. A critical factor in the conversion process is  layer selection, i.e., deciding on which layers to convert to linear attention variants. This paper describes a simple and efficient recipe for layer selection that uses layer importance scores derived from a small amount of training on generic text data.  Once the layers have been selected we use a  recent pipeline for the distillation process itself \citep[RADLADS;][]{goldstein2025radlads}, which consists of attention weight transfer, hidden state alignment, KL-based distribution matching, followed by a small amount of finetuning. We find that this approach is more effective than existing  approaches for layer selection, including heuristics that uniformly interleave linear attentions based on a fixed ratio, as well as more involved approaches that rely on specialized diagnostic datasets.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a distillation and layer selection method that allows to derive hybrid attention LLMs from classical Transformers-based ones. Their method allows them to turn computationally intensive models into more efficient alternatives without training from scratch, while retaining strong performance on in-context retrieval tasks, where previous methods were less performant.

The proposed method is based on a 2-step distillation process (first at layer-wise attention representation level, and then at logit level), and on a greedy layer switching exploration of possible hybrid configurations. The best identified configuration is then distilled again.

The authors conduct extensive evaluations on the RULER and SWDE benchmarks with 3B Llama and Qwen models, showing the efficacy of their approach.

### Strengths
Overall, this paper clearly states its problem and proposes a simple but efficient method as a solution.
- **The method is rather straightforward and yields benefits**: the proposed greedy approach is easy to grasp and seems to provide good results on the two benchmarks used, especially for Qwen.
- **Several variants and combinations are tested**: I appreciate that the authors have tried sensible variants of the Greedy Addition method in Table 1. This provides valuable insights and helps underline the reason for the success of their method. It is also nice to see the interplay between different linear attention variants in section 5.2, and the discussion about early stopping in section 5.3.

### Weaknesses
- **Limited scaling potential**: The method presented in this paper has compute requirements that scale linearly in the number of layers. As model grow larger, they will tend to have more layers and will thus be doubly more expensive (both from the parameter and data viewpoints) to distil into hybrid variants. Given that the authors do not provide a data point at a different size (either smaller or larger than 3B), it is difficult to assess the relevance of this method at a larger scale where it will be more costly to run.
- **Lack of discussion about selected layers**: There is no discussion or analysis about potential patterns (or the absence thereof) for the selected layers, and the behavior of $\mathcal{I}(l)$. This would be a valuable insight for subsequent work.
- **Limited evaluation scenarios**: Although the authors report successful results on RULER and SWDE, it remains unclear whether their approach recovers the same long-context abilities than softmax-based models for every sequence lengths, or whether the hybrid distilled models solely match the performance of the teacher models on the context lengths used in these benchmarks (<10k tokens). A perplexity-based analysis or a needle-in-a-haystack could help provide insights on that question.

### Questions
- Why is a stage-1 MSE step necessary in the final distillation stage? Would re-using the distilled layers from the previous step make sense?
- How did the distilled models behave in terms of validation perplexity on the DCLM benchmark? Is the distillation training process stable?
- Although the Jaccard similarity seems to provide valuable insights on the transferability and similarity of selected layers, I am wondering if a correlation/distillation metric would be better-suited for this analysis. The Jaccard metric fails to differentiate the ranking of the layers by the $\mathcal{I}$ score, which would be captured by e.g. a Spearman correlation, and seems like a meaningful information to compare two selections. Did you run such analysis?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper tackles the problem of converting pretrained softmax attention Transformers into more efficient hybrid architectures that combine softmax and linear attention layers. While prior works have largely used fixed interleaving ratios (e.g., 1 softmax per 3 linear layers), this paper argues that such heuristics are suboptimal for distillation settings.

The authors propose a simple yet effective KL-guided layer selection strategy: Each layer’s importance is determined by the reduction in teacher–student KL divergence when that layer alone is restored to softmax attention in an otherwise all-linear student. Layers with the highest KL importance scores are retained as global attention layers in the final hybrid model.

Extensive experiments with Qwen2.5-3B-Instruct and Llama-3.2-3B-Instruct show that this method consistently outperforms heuristic or NAS-based baselines (e.g., UNIFORM, SMART, PostNAS), particularly on long-context recall benchmarks such as RULER and SWDE.
The approach requires only ~25–30B tokens, far fewer than PostNAS (400B).

### Strengths
Despite its simplicity, the method achieves large gains—especially in the low-softmax regime (e.g., 12.5% ratio).
The KL-based layer importance metric is conceptually elegant and empirically grounded.

### Weaknesses
All experiments focus on 3B-class decoder-only models; no results for encoder–decoder or smaller-scale models.

While the paper emphasizes efficiency, explicit measurements of inference latency or memory savings are missing.

The accuracy of layer importance relies heavily on the stability of Stage-2 KL distillation. If the teacher–student mismatch is large, the ranking may become unreliable.

### Questions
Validate the layer-selection behavior across scales (e.g., 1B, 7B, or 14B) to demonstrate scalability and consistency.

Provide intuition or gradient-based analysis showing why KL reduction correlates with information retention or attention selectivity.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes a learned method to choose which layers should be softmax attention layers and which should be linear attention layers in a student model to distil from a teacher model with only softmax attention layers. To achieve this, the authors use distillation and KL-loss to guide which layers to select. They start by distilling a fully linear attention student model. Then they used that distilled model to calculate the importance of each layer being a softmax layer, by distilling the previous model, with its k-th layer replaced by a softmax layer. They then use the KL-loss to the teacher model to calculate the K most important layers, which they switch to softmax layers before doing a final distillation. Their method requires few training tokens (600M unique, 20B total) and outperforms other techiniques.

### Strengths
- The paper is very well structured and written, the sections flow naturally from one to another, and the motivation is very clear.
- The experiments are thorough, and the authors compare their method to various baselines and show that their method outperforms the others in settings where the vast majority (75% or more) of the layers are linear layers rather than softmax layers.
- The method is simple and effective, and does not require much computing (compared to the total pre-training cost) to create much more efficient models with little to no performance cost. This is especially important because of the reduction in inference cost for such models.
- The authors do multiple ablations to show that the choice of stage at which to calculate importance and to do addition of softmax layers rather than removal of softmax layers is justified. In addition, they show that their method works for different linear attentions and that calculating importance using one type of linear layer can be translated to another type of linear attention

### Weaknesses
- On line 339, the authors state that their method is iterative; however, this is not totally correct, their method calculates the importance of each softmax layer independently, and then selects the top-K best. This ignores how adding one softmax layer can affect the importance of other softmax layers, which an iterative process (add one, recalculate importance, add another, etc.) would take into account.
- It would be interesting and helpful for the paper to know which layers were chosen. Whether the layers with the highest importance were near each other or if they were spread out through the model (in other words, checking whether there exist clusters of important layers).

### Questions
- For Figure 2, it would be nice to have the performance when all the layers are linear.
- Did you consider the existence of clusters of important layers, which could indicate that only one of those layers is important, rather than all of them being important?
- Did you try having a penalty for important layers that are close to each other (to avoid selecting them, in case their importance was due to the same factors)?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a method for distilling pretrained softmax attention Transformers into more efficient hybrid architectures that interleave softmax and linear attention layers. The key aspect of this process is layer selection, i.e., deciding which layers to convert to linear attention variants. 

The authors present a simple and efficient approach for layer selection based on layer importance scores derived from a small amount of training on generic text data. Once the layers have been selected, the authors use a recent distillation pipeline (RADLADS) consisting of attention weight transfer, hidden state alignment, KL-based distribution matching, and a small amount of finetuning. The authors find this approach to be more effective than existing methods for layer selection.

### Strengths
* **Addresses a Practical**. Tackles the challenge of converting pretrained softmax attention Transformers into more efficient hybrid architectures without expensive pretraining from scratch. Focuses on improving inference efficiency of LLMs, which is a critical concern in practical deployments
* **Intuitive Layer Selection Approach**. Proposes a KL-guided layer selection criterion that is both simple and theoretically motivated. The intuition is clear: layers that are more critical for maintaining performance will show larger KL divergence reduction when kept as global attention layers. This represents an improvement over fixed interleaving strategies used in previous work.
* **Comprehensive Empirical Analysis and Ablations**. Provides detailed analysis showing the trade-offs between different attention mechanisms on various task types. Demonstrates that sliding window attention works well for common-sense reasoning but struggles with in-context recall tasks (Figure 1). Shows empirical evidence that their approach outperforms existing layer selection methods
* **Clear Problem Formulation**.
Clearly identifies the two key decisions in distillation: student architecture selection and distillation recipe optimization
Focuses on the less-explored but equally important architecture selection problem

### Weaknesses
* How does this method compare to a baseline that just randomly selects the layers to replace with linear ones i.e instead of Uniform or any of the fancy methods of selecting the layers to linearize if we just randomly chose K, layers and linearized them, how would that affect the performance. 
* The number of layers to be linearized K seems to be a heuristic or dataset dependent ? It makes the method feel somewhat brittle. Would this automatically transfer to some other task or would you need to do a linear transfer of layers again to get it working?
* The goal of linearization etc., is to speed up the performance. How does the memory, latency/throughput required to match or be close to the quality of full attention, compare to that of using full attention?

### Questions
* Is there any clear pattern or intuition of which layers get selected for linearization ?

### Soundness
3

### Presentation
3

### Contribution
3
