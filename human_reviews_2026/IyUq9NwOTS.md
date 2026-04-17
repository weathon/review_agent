# Little By Little: Continual Learning via Self-Activated Sparse Mixture-of-Rank Adaptive Learning

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Continual learning (CL) with large pre-trained models is challenged by catastrophic forgetting and task interference. Existing LoRA-based Mixture-of-Experts (MoE) approaches mitigate forgetting by assigning and freezing task-specific adapters, but suffer from interference, redundancy, and ambiguous routing due to coarse adapter-level selection. However, this design introduces three key challenges: 1) *Interference*: Activating full LoRA experts per input leads to subspace interference and prevents selective reuse of useful components across tasks. 2) *Redundancy*: Newly added experts often duplicate or contradict existing knowledge due to unnecessary activation of unrelated ranks and insufficient reuse of relevant ones. 3) *Ambiguity*: Overlapping features across tasks confuse the router, resulting in unstable expert assignments. As more experts accumulate, earlier task routing degrades, accelerating forgetting. We propose ***MoRA***, a **M**ixture-**o**f-**R**ank **A**daptive learning approaches with self-activated and sparse rank activation for CL. Unlike mixing multiple low-rank matrices, MoRA decomposes each rank-r update into r rank-one components, each treated as an independent expert, enabling fine-grained rank-one expert utilization while mitigating interference and redundancy. To avoid ambiguous routing, we propose that each rank-one expert can infer its own relevance via intermediate activations. Coupled with our proposed rank pruning and activation budgets, MoRA adaptively selects a sparse mixture of ranks per input. We validate MoRA on continual learning benchmarks using CLIP and language models, analyzing both in-domain learning and out-of-domain forgetting/generalization during fine-tuning. MoRA shows significant effectiveness on enhancing CL with PTMs, and improving generalization while mitigating forgetting.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper looks at a concrete pain point in continual learning with PEFT: adapter-level selection is too coarse and causes interference/forgetting when tasks pull in different directions. The authors analyze LoRA through the lens of rank composition and argue that a rank-r update entangles several directional “atoms,” only some of which are task-relevant at a given step. They then design MoRA: factorize LoRA into r rank-1 experts, let them be self-activated with sparse routing under a budget, and prune. I like how the narrative moves from “what’s going wrong” to “what structure inside LoRA can be exploited,” and the final recipe is simple enough to adopt. Empirically, on CLIP/LLM streams, MoRA consistently reduces interference at modest cost, and the ablations make the design believable.

### Strengths
What I like most is the research arc: the paper identifies that “choose an adapter” is the wrong granularity, connects that to rank structure, and then implements a mechanism that lets the model choose directions rather than whole adapters. The self-activation + sparsity budget feels natural and yields stable improvements across settings. Implementation details are concrete enough that I could reproduce this without guesswork.

### Weaknesses
Where I feel the paper is thin is evidence hardness. I was hoping to see equal-compute, equal-time comparisons to the strongest recent CL-PEFT baselines (multi-adapter/multi-expert routing) on longer sequences, with 3–5 seeds and CIs. It’s also a pity there’s little about routing interpretability and failure modes—which rank-1 experts fire where, when does the router collapse, what happens under strong domain shift? Finally, inference trade-offs are underspecified: I’d like quantitative curves of sparsity vs. latency/VRAM and sensitivity to the number of experts and temperature/thresholds.

### Questions
Could you provide 
(i) equal-budget head-to-heads with the strongest CL-PEFT methods on long streams, 
(ii) 3–5-seed statistics, 
(iii) sparsity→(accuracy, latency, VRAM) trade-offs and sensitivity to expert count/temperature, and 
(iv) a short visualization of routing dynamics with a couple of failure cases? 

If these arrive and look strong, I’m inclined to raise my score.

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
To address the challenges of (1) interference across experts and limited knowledge reuse, (2) redundancy across experts, and (3) forgetting caused by routing, in Mixture-of-Expert LoRA, the paper proposes a Mixture-of-Rank Adaptive (MoRA) learning method with self-activated and sparse rank activation for continual learning. MoRA treats each rank-one vector pair as a rank-one expert and uses expert activation for effective routing, achieving interference mitigation and parameter redundancy. Experiments on continual learning benchmarks using CLIP and language models have shown significant effectiveness in both in-domain learning and out-of-domain forgetting/generalization, improving generalization while mitigating forgetting.

### Strengths
1. Since current coarse-grained experts with routing cause redundancy and conflicts, this paper utilizes fine-grained experts which treats each rank-one LoRA as an expert, avoiding interference from whole LoRA blocks.

2. Instead of using external routers, MoRA uses each rank-one expert’s activation for routing, achieving self-activated selection.

### Weaknesses
1. The motivation analyzed in this paper, especially in the introduction section, mostly discusses the original problem of the current mixture-of-expert framework itself. For example, redundancy caused by coarse-grained experts is a fundamental problem of MoE. The three challenges mentioned in the introduction lack strong connections to continual learning. 

2. For most LoRA-based continual learning, LoRA is added in attention blocks, while I found that MoRA also adds LoRA in the MLP layer, as shown in Figure 2. It makes it confusing about this choice, and also blurs the main goal, whether it is for MoE or for continual learning.

3. The learning of a new task is not clearly clarified in the paper, for example, it does not mention how previous tasks’ experts participate in the forward process during training and inference.

4. MoRA is not memory-efficient since the number of LoRA experts grows linearly with the number of tasks. Also, since it uses rank-one experts for reducing redundancy, the trained LoRA experts would be sparse matrices, which wastes memory space.

5. The writing of this paper is not in good shape. Paragraphs lack connections, especially from lines 093 to 095, which suddenly and directly introduce a self-activation routing mechanism without any transition from the rank-one expert. It seems like lines 088 to 107 should be in one paragraph.

### Questions
1. In MoRA, for keeping previous knowledge, I found the only operation is freezing previous LoRA experts without any constraints, so how to make sure MoRA maintains previous knowledge, and how to make sure newly added experts update in an orthogonal direction from previous experts’ subspaces.

2. For self-activated rank-one experts routing mechanism, especially in Eq.(8), why choose A to compute the routing score? What’s the difference between A and B? Why not first use SVD on BA to obtain a more orthogonal matrix and then use that orthogonal matrix to compute the routing score?

3. From Table 1, it seems like MoRA is not robust to all tasks. Can we have the conclusion that MoRA is sensitive to different tasks? Can authors explain the specific order of the tasks used in Table 1? Since the ordering of tasks is an important factor in continual learning.

4. Can authors compare the performance of MoRA with SAPT [1], and analyze the difference from SAPT?

[1] SAPT: A Shared Attention Framework for Parameter-Efficient Continual Learning of Large Language Models, ACL2024.

5. Equations and Figures:

5.1 In Eq(8), it should be $A_{i,:}$ and $A_{j,:}$.

5.2 In line 151, what’s the dimension of $y_i^{t}$?

5.3 What’s the mathematical objective or loss for MoRA?

5.4 In Figure 2, $R(x)$ may need to be mentioned.

6. Can authors evaluate the performance of MoRA on another common metric, OPD [1]? 

[1] Scalable and Order-robust Continual Learning with Additive Parameter Decomposition, ICLR2020.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes MoRA, a novel method for continual learning (CL) with large pre-trained models. The authors identify key limitations in existing LoRA-based Mixture-of-Experts (MoE) approaches for CL, namely: (1) interference from coarse-grained expert activation, (2) redundancy in newly learned parameters, and (3) routing ambiguity as the number of tasks grows. To address these, MoRA decomposes each rank-r LoRA adapter into r rank-1 components, treating each as a fine-grained expert. The core innovation is a self-activation mechanism where the key vector of each rank-one expert scores its own relevance to the input, eliminating the need for a separate router. This is coupled with sparsity enforcement via top-k selection, temperature scaling, and test-time thresholding. The method is evaluated extensively on CL benchmarks for both vision-language (CLIP) and language models (T5, LLaMA), demonstrating strong performance in mitigating forgetting, improving generalization, and reducing parameter activation compared to prior state-of-the-art methods.

### Strengths
+ The paper is a model of clear scientific writing. The problem is well-motivated, the method is explained step-by-step with helpful formulations and figures, and the results are presented logically.
+ MoRA provides a more parameter- and compute-efficient path for continual learning with large models, a problem of great practical importance. The demonstrated ability to reduce forgetting while improving generalization on unseen tasks is a meaningful

### Weaknesses
+ The language models used for experiment are outdated. The experiments with state-of-the-art LLMs are needed
+ For the mixture-of-lora, if the lora cannot be merged into the pre-trained weights, then it will have additional computation overhead. But for baselines inflora and sdlora, their loras can be merged into the pre-trained weights in test time.

### Questions
See the weakness.

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
This paper introduces MoRA, a continual learning (CL) approach for pre-trained models, which mitigates parameter redundancy and task interference by decomposing LoRA updates into a set of fine-grained rank-one experts. The method incorporates a self-gated sparse mixture mechanism, where each rank-one component functions as an independent expert and is selectively activated in an input-dependent manner. Experiments on CLIP and language models show that MoRA improves downstream task performance and reduces forgetting, while using fewer parameters than MoE-LoRA baselines. The study offers valuable insights into fine-grained expert design and efficient knowledge reuse, though the theoretical analysis could be further deepened. Moreover, while intriguing, the empirical gains over established benchmarks remain modest and would benefit from more in-depth validation.

### Strengths
1. This paper provides a rank-level decomposition that could reduce interference and redundancy compared to coarse-grained MoE-LoRA, enabling precise expert specialization.
2. The authors present extensive results. The method is evaluated on representative benchmarks with diverse backbones, showing improved results in standard CL metrics.
3. The proposed method supports parameter efficiency and scalability, whose sparse activation of ranks lowers memory usage and computational cost, with efficient scaling to large models like LLaMA.

### Weaknesses
1. The visualization of task-specific semantics of rank activation in Sec. 4.4 and A.8 rely on a few examples from limited projections and images without providing statistical summaries to support generalizability.
2. The paper lacks a theoretical justification for why self-activated sparse experts inherently address interference or redundancy, relying solely on empirical observations.
3. CL experiments on LlaMA-7B only include one baseline O-LoRA for comparison (Tab. 9), limiting the robustness of related conclusions.
4. Performance improvement of the proposed method in the primary benchmark X-TAIL (Tab.1) is modest compared to the recent approaches, making it less clear whether gains are meaningful or marginal.
5. MoRA appears to combine well-established techniques and tricks like self-attention and sparse MoE, without clearly articulating a unique methodological or theoretical contribution.

### Questions
1. Since the top-k rank selection is generally applied in the training phase, how could the algorithm guarantee that the newly introduced experts can be activated for incoming tasks? How does the initialization of those new rank-one experts (e.g., zero-value, random, or warm-start from previous experts) impact knowledge acquisition and avoidance of trivial routing to old experts? The manuscript mentions rank freezing but not initialization details. Also, how does the $k$'s value in the top-k selection get selected?
2. Can the authors provide more statistical summaries (e.g., activation distributions) for the semantic rank visualizations like Fig. 3, Fig. 4, and Fig. 6, to ensure they are representative beyond cherry-picked examples?
3. What theoretical guarantees ensure that self-activation + top-k prevents interference, particularly as tasks increase?
4. Typo in line 151, the formula representation of the dataset is missing curly braces.

### Soundness
3

### Presentation
2

### Contribution
2
