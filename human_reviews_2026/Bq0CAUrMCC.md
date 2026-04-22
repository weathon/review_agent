# Structured Transformer Circuits Pruning

- Avg Score: 3.50
- Decision: Reject
- Scores: 6, 2, 4, 2

## Abstract
Transformers become ubiquitous across vision and language tasks, but their depth and parameter count often far exceed what is needed for a given downstream application, leading to unnecessary compute and memory overhead. Existing layer‐pruning techniques either require multiple retraining cycles, rely on continuous relaxations that never fully deactivate blocks, or depend on architecture‐specific analyses. We introduce STCP, a model‐agnostic, single‐pass pruning framework that learns binary gates over each block’s multi-head self-attention (MHSA) and MLP sub-layers in a pretrained transformer.  We optimize gates while also injecting noise and introducing an $L_1$ penalty: this allows us to escape from local minima, and to find sparser circuits. We validate STCP on both image classification and NLP tasks with large pretrained models, showing good trade-offs in terms of complexity and performance. The code will be made publicly available upon acceptance of the article.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Structured Transformer Circuits Pruning (STCP), a structured pruning method to improve the efficiency of Transformer models by removing entire Multi-Head Self-Attention (MHSA) and MLP sub-layers within each block. They apply a binary gate to each sub-layer to decide whether to turn it on or off. This gate is implemented using the non-differentiable Heaviside step function, and it employs the Straight-Through Estimator (STE) trick to allow gradients to flow during backpropagation. To help the gates effectively explore on and off states during fine-tuning, Gaussian noise is injected. Simultaneously, an L1 regularization penalty is applied to encourage the gate values to move toward zero. After training, gate parameters $g$ are ranked, and sub-layers are removed sequentially until performance drops just below a threshold. Experiments across various Vision (CLIP, ViT) and NLP (BERT, RoBERTa) tasks demonstrated that the method is highly robust in terms of its performance-to-compression ratio compared to existing methods, validating the effectiveness of the proposed methodology.

### Strengths
1. The method is well-motivated by applying a binary gating mechanism to structured pruning. This directly targets real-world computational and memory overheads, which is often not achieved by less-structured pruning.

2. STCP is a single-pass pruning framework. This makes it significantly simpler and more practical than prior approaches that typically rely on complex, multi-stage, iterative prune-and-finetune cycles.

3. The framework demonstrates broad applicability. It is model-agnostic and shows strong, consistent performance across diverse architectures (e.g., CLIP, ViT, BERT, ROBERTa) and different domains, including both Vision and NLP tasks.

4. The method proves to be highly robust. Extensive experiments show that it can prune a significant number of sub-layers while maintaining performance very close to, or in some cases even slightly exceeding, the original dense model.

### Weaknesses
1. The final pruning stage is a greedy approach that ranks gate values and removes sub-layers sequentially. This method does not guarantee finding the optimal combination of remaining sub-layers.
2. The process requires manual intervention. The user must define an acceptable performance drop threshold to determine when to stop removing layers, which is not an automated part of the framework.
3. The paper lacks sufficient references and comparisons to other established differentiable gating mechanisms used for pruning (e.g., Gumbel-Softmax).
4. The overall writing, structure, and presentation require further polish and refinement. This includes insufficient detail in the Method section (Sec 3), redundant sentences (e.g., “The gates’ values and the possibility for each gate to flip after training for the CLIP model trained on the CIFAR-10 are presented in Appendix B.” is duplicated on pages 7 and 8), and imprecise statements (e.g., “Thus, any gate with $|g|$ small has roughly a 50% chance to be on or off, …” on page 4).

### Questions
Refer to Weaknesses

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This manuscript proposes and evaluates a procedure for removing entire components of transformer models. The scheme adds "gates" to the model and then jointly optimizes parameters of the gates and model weights to identify components that can be removed. Numerical experiments are provided to illustrate the efficacy of the scheme.

### Strengths
The manuscript provides an interesting method for shrinking the size of transformer models while retaining performance. Moreover, in the realm of methods that remove entire components it seems to outperform other methods when no fine-tuning is allowed.

### Weaknesses
The main weakness of this work is its narrowness in terms of how it presents the comparative landscape. While I agree with the manuscript that there are good computational reasons to preference removal of entire components within a model vs. unstructured pruning methods (in terms of practical speedup), it does seem that the competitive landscape is ultimately larger than just methods that can remove entire components. For example structured pruning methods (i.e., those that remove entire neurons) can be effective for MLPs since they retain the computational structure of the layer just at a smaller size (as opposed to, e.g., introducing unstructured sparsity). 

The literature is too numerous to name, but, e.g., [Kwon, Woosuk, Sehoon Kim, Michael W. Mahoney, Joseph Hassoun, Kurt Keutzer, and Amir Gholami. "A fast post-training pruning framework for transformers." Advances in Neural Information Processing Systems 35 (2022): 24101-24116.] and [van der Ouderaa, Tycho FA, Markus Nagel, Mart Van Baalen, and Tijmen Blankevoort. "The LLM Surgeon." In The Twelfth International Conference on Learning Representations.] 

One reason to bring this up is that a cursory comparison of results with Kwon et al. above and https://arxiv.org/pdf/2406.00061 suggests that, e.g., BERT models can have a substantial fraction ~40% of the parameters removed in a structured matter while retaining performance similar to the proposed method when only 2 layers are removed. (Thought it is hard to make an explicit comparison across papers because it seems the pretrained baseline may differ?)

In addition, quantization is also a very effective strategy for reducing both storage and computational time. While, presumably, a pruned model could then be quantized, it is not clear that these procedures are orthogonal: perhaps an aggressively pruned model resists quantization in a way that makes it better to quantize the larger model than prune then quantize. While it is unreasonable to burden this manuscript with answering that rather broad question, it does illustrate that a broader range of experiments are likely needed to help situate the proposed procedure in the broader landscape. I consider comparisons with structured pruning somewhat essential, quantization less so.

Lastly, while I am sympathetic to the ease with which one ask for "more experiments," I do think that newer, larger models (particularly for NLP tasks) should be considered to understand both the generality and scalability of the method. Perhaps (smaller) Llama models or similar would be useful here to provide at least a few points of comparison with other modern methods (e.g., LLM surgeon or similar)

Some assorted minor comments/weaknesses:

- While the proposed method does not have an explicit "fine tuning" step, the weights are updated with the gates (assuming the pseudo code is followed). (In fact, presumably this is why removing a layer can increase performance.) As such, it seems a bit unfair to completely omit fine tuning from other methods; better would be to control for compute and balance that (or, report compute to make it easier to assess the relative expense of methods).

### Questions
- While the method is articulated as "single-pass," presumably once a certain layer is removed that could change the optimal $g$ parameters for the other layers significantly, is there any value to thinking about the method via sweeps in which at each step 1 or a few layers are removed and then the gates are retrained? 

- The $\ell_1$ penalty seemingly drives the $g$ towards 0, but that corresponds to at 50/50 gate (if I interpreted things correctly) and not an "off" gate, were other regularization strategies or penalties explored?

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
The paper proposes Structured Transformer Circuits Pruning (STCP), a single-pass pruning method that inserts learnable binary gates into each MHA and MLP sub-layer of a pretrained transformer. During fine-tuning, Gaussian noise and an L1 regularization encourage some gates to turn off, allowing the model to identify unimportant sub-layers. The method is evaluated on both vision and language transformers (CLIP, ViT-H14, BERT, RoBERTa) and reports competitive results compared with Shortened-Taylor and Joint Layer Drop.

### Strengths
1. The paper is well-written and easy to follow. All training details are included, and the experimental setup is consistent across datasets.
2. The experiments span both image and text domains and include ablation studies, FLOP/memory/time tables, and comparisons to multiple baselines.
3. STCP achieves competitive or slightly better results than dense baselines in several small-scale benchmarks, which is pretty impressive.

### Weaknesses
1. The proposed gating+L1 framework and mask-training approach have been well-studied before (e.g., SNIP, GraSP, MaskLLM, LLM-Eraser). The contribution on this side feels incremental and more engineer-like.
2. Because each sub-layer requires storing gates, masks, and gradients throughout training, the memory cost can easily exceed that of normal fine-tuning. The paper shies away from this overhead and never reports peak GPU memory. For large-scale models, such methods quickly become infeasible.
3. The pruning criterion relies on validation accuracy to decide the pruning threshold, implying that the whole fine-tuning and ranking process must be repeated for each new task, which is far from the claimed "model-agnostic" in practice.
4. All results are on CIFAR-10, Tiny-ImageNet, SST-2, and QNLI with medium-size transformers. There is no evidence that STCP works for modern-scale LLMs or vision foundation models.
5. While FLOPs and latency are reported for the inference stage, memory usage and training cost vs. accuracy trade-offs are not included, which are often the true bottlenecks for structured pruning.

### Questions
1. Could the authors provide the training cost, including Memory usage, Training Time, and FLOPs for baseline methods and STCP? If this is too time-consuming, at least I want to see the comparison between STCP and the dense model, i.e., regular SFT.
2. Could the authors show that their approach is really "model-agnostic"? Modern Structured pruning methods like LLM-Pruner evaluate their pruned model on several tasks instead of one.
3. Could the authors try to perform STCP on a larger-scale model like Llama3-1B/Qwen2.5-0.5B? They are not so large and are on the same model size compared to what the authors have used in their paper (ViT-H14/Roberta, etc.)

I would be happy to raise my scores if the author could provide these details.

### Soundness
3

### Presentation
2

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
This paper proposes a pruning framework named "Structured Transformer Circuits Pruning" (STCP). The method aims to learn binary gates for MHSA and MLP sub-layers within Transformer blocks through a single fine-tuning pass. Its core mechanism is claimed to combine Gaussian noise injection and L1 regularization to remove redundant sub-layers without iterative retraining. The authors report that this method outperforms SOTA (Shortened-Taylor, Joint Layer Drop) on models like CLIP, ViT, and BERT.

### Strengths
The paper well written, the presentation of the idea is clear for most part. 
The results discussions in main text and appendix seems quite multi-dimensional and comprehensive with a lot of datasets, ablation and hyperparam discussions.
There is also a limitation section which worths encouraging.

### Weaknesses
1. The evaluation setup seems to have unfairness issue. One of the key advantages of this paper seems to be very cheap in finetuning compared to similar structured pruning methods like LLM-Pruner and RECAP and only requires single pass. But it is confusing that experimental detail also mentioned you still require 10000 steps to train, and there is no training cost comparison with those pruning methods. So I feel the main selling point of this paper becomes questionable. 

2. Although the authors put efforts to include a lot of dataset and tasks in experiment, a key problem is that there are no any SOTA pruning methods to compare with in your main results, only a few vanilla strategies like weight pruning and gradient pruning. not sure how the performances position in the latest landscape. 

3. The compression rates achieved are also underwhelming for models in main results (CLIP and ViT), i.e. up to 17/64=26.56% total pruning rate, which is quite conservative compared to SOTAs like LLM-Pruner, OBC [1]. 


[1] Frantar, Elias, and Dan Alistarh. "Optimal brain compression: A framework for accurate post-training quantization and pruning." Advances in Neural Information Processing Systems 35 (2022): 4475-4488.

### Questions
1. Does the single-pass referred to pruning steps or finetuning steps? If it's pruning stage, then it kind of defeats the whole motivation claimed in this paper, because SOTA methods like LLM-pruner also only requires single-pass for pruning stage if using taylor criterion to collect gradient. If it's finetuning step, what's the 10000 training steps mentioned in Section 4.1 refers to?
2. Related works mentioned some SOTA pruning baselines like LLM-Pruner, but none were compared in the experiment results. It is concerning if this meets ICLR standard.

### Soundness
3

### Presentation
3

### Contribution
2
