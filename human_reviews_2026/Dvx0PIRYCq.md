# Where to Begin: Efficient Pretraining via Sub-network Selection and Distillation

- Decision: Reject
- Scores: 4, 8, 4, 6

## Abstract
Small Language models (SLMs) offer an efficient and accessible alternative to Large Language Models (LLMs), delivering strong performance while using far fewer resources. We introduce a simple and effective framework for pretraining SLMs that brings together three complementary ideas. First, we identify structurally sparse **sub-network initializations** that consistently outperform randomly initialized models of similar size under the same compute budget. Second, we use **evolutionary search** to automatically discover high-quality sub-network initializations, providing better starting points for pretraining. Third, we apply **knowledge distillation** from larger teacher models to speed up training and improve generalization. Together, these components make SLM pretraining substantially more efficient: our best model, discovered using evolutionary search and initialized with LLM weights, matches the validation perplexity of a comparable Pythia SLM while requiring **5.16x** and **1.26x** fewer floating point operations for token budgets of 10B and 100B, respectively. We release all code publicly, offering a practical and reproducible path toward cost-efficient small language model development at scale.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the high cost of pretraining Small Language Models by proposing a framework that combines three synergistic techniques: 1) warm-starting the SLM by extracting an initial sub-network from a large, pretrained teacher LLM; 2) using evolutionary search to find the optimal sub-network architecture within a parameter budget; and 3) applying knowledge distillation from the teacher during pretraining. The authors demonstrate that this framework is highly effective, with their best model achieving the same validation perplexity as 9.2× fewer pretraining tokens.

### Strengths
1. Significant Practical Impact: The 9.2× token-efficiency saving is a major achievement, making high-performance SLM development accessible with limited compute.
2. Contribution to Reproducibility: By releasing the whittle library, the authors provide a clear, generalizable pipeline, addressing the closed-source limitations of prior art and contributing significantly to the community.

### Weaknesses
1. Incomplete Discussion of Related Work: The paper is actually very similar to methods that prune and then continue pretraining to recover performance, which is an active area of research. However, the authors appear to have not discussed any related papers, and the performance improvements they achieve are also similar. The authors should compare their work to Sheared LLaMA, DRPruning and other recent methods to clarify their unique contribution. Therefore, I believe this paper's contribution to the community is relatively limited.

[1] Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning. ICLR 2024.

[2] DRPruning: Efficient Large Language Model Pruning through Distributionally Robust Optimization. ACL 2025.

2. Unaccounted Cost of Evolutionary Search: The paper's core claim is "efficiency." However, the evolutionary search itself consumes significant compute. The 9.2× token saving figure does not account for this search cost. If not, the true end-to-end efficiency gain is overstated.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors introduce a novel method to train efficiently SLM's leveraging a LLM for initialization and distillation.
They provide a method to search the subnetwork space of a LLM corresponding to a space of possible SLM architectures constrained by a smaller parameter budget. They use an evolutionary algorithm to search this space and retrieve the most promising subnetwork, which is then pre-trained with a distillation objective using the same LLM as teacher. They accelerate the SLM pre-training by up-to 9.2X compared with a SLM with the same architecture but randomly initialized. They open-source a library to reproduce and extend the work.

### Strengths
* The idea to use an evolutionary algorithm to extract a subnetwork as initialization is novel and convincing
* They evaluate their strategy thoroughly and obtain impressive speedups to train SLM
* They open source a library to reproduce and extend their work

### Weaknesses
Not seeing any significant weakness

### Questions
* The method seems generic. What are the constrained for a neural network architecture so that the method can be applied to any type of network (e.g. ResNets and image classification)? 

*  If the comparison between training from scratch a SLM with random initialization and pre-training it from the retrieved LLM subnetwork also include the LLM subnetwork search budget, will the 9.2x speedup persist? How does this change the comparison?

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
This paper introduces a framework to make the pretraining of Small Language Models (SLMs) more efficient. The core idea is to first select a smaller sub-network from a large, pretrained teacher model and use its weights as a warm start. This sub-network is then further trained, primarily using knowledge distillation from the teacher model. The authors employ an evolutionary search algorithm to find optimal sub-network architectures within different parameter budgets. They also release an open-source library called `whittle` to facilitate this process. The results show that this approach can significantly reduce the number of pretraining tokens required to reach a target performance level compared to training a standard model of a similar size from scratch.

### Strengths
1. The paper provides some useful insights into the process of knowledge distillation for SLMs.

2. The paper explores four types of search spaces, from coarse to fine-grained and from uniform to layer-wise configurations.

### Weaknesses
1. The main contribution of the paper feels incremental. The general strategy of first pruning a large model and then using distillation to train the smaller model has been explored in previous works, such as Sheared LLaMA [1] and Minitron [2].

2. The proposed sub-network selection method, particularly in the fine-grained search spaces, seems to lack sophistication. The paper states that for selecting components like attention heads or neurons within a layer, it samples indices from the teacher's components. This appears to be a random selection process that does not consider weight importance, activation patterns, or any other semantic information, which are common considerations in modern structured pruning methods.

3. The performance comparison is not convincing enough. To better demonstrate the effectiveness of the proposed method, it should be benchmarked against strong, relevant baselines like Sheared LLaMA [1], which also focuses on efficient pre-training through structured pruning. Without this comparison, it is difficult to assess the true advantage of this framework over the state-of-the-art.

**References**

[1] Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning.

[2] LLM Pruning and Distillation in Practice: The Minitron Approach.

### Questions
1.  The experiments are conducted on the Pythia model family. Would this method work as effectively on more powerful base models, such as the Qwen family? It is possible that stronger, more optimized models are inherently harder to compress and might not yield the same efficiency gains.

2.  Could you provide more detailed statistics on the overhead and efficiency of the search phase? The paper mentions a budget of 16 hours per parameter bin. It would be helpful to see more granular data to better understand the costs of this initial search step.

3.  How is the crossover operation performed in the fine-grained search space?

### Soundness
2

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
The paper presents a methodology to initialize language models from larger pretrained models using evolutionary search and subsequently train them with distillation to achieve high-quality small language models. In particular, the authors present the search space they consider and then an evolutionary search algorithm to identify good candidates in that search space. They show in their experiments that the networks found using their search algorithm perform significantly better than random initialization or random points in that search space.

### Strengths
- The paper is very well written with clearly defined goal and methodology as well as a comprehensive appendix.
- The evolutionary search algorithm is clearly presented and intuitive.
- The results clearly show that starting from a model found using the evolutionary search is a significantly better choice compared to random initialization and a random point in the search space.

### Weaknesses
The main weakness of the paper is in its evaluation section. In more detail:

1. When comparing initialization performance of different methods (superset, random init, etc) none of the methods are actually FLOP normalized. Namely, the evolutionary search required some FLOPS to identify the best model. However, random initializations didn't so they should be allowed more optimization to account for that discrepancy. More importantly, the evolutionary search should be compared to random search ie sample different points from the search space and keep the best.
2. It is slightly worrying that the bigger the model the coarser the search space needs to be. The more fine-grained search spaces are supersets of the coarse ones so they should perform at least as well. The fact that they don't points towards the search not being effective which increases the importance of point 1.

### Questions
I have laid out my concerns in the weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
3
