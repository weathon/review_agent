# MoDr: Mixture-of-Depth-Recurrent Transformers for Test-Time Reasoning

- Decision: Accept (Poster)
- Scores: 4, 8, 6, 2

## Abstract
Large Language Models have demonstrated superior reasoning capabilities by generating step-by-step reasoning in natural language before deriving the final answer. Recently, Geiping et al. introduced 3.5B-Huginn as an alternative to this paradigm, a depth-recurrent Transformer that increases computational depth per token by reusing a recurrent block in latent space. Despite its performance gains with increasing recurrences, this approach is inadequate for tasks demanding exploration and adaptivity, a limitation arising from its single, chain-like propagation mechanism. To address this, we propose a novel dynamic multi-branches routing approach for Huginn, termed as Mixture-of-Depth-Recurrent (MoDr) Transformer, which enables effective exploration of the solution space by shifting linear latent reasoning into a LoRA-based multi-branch dynamic relay mode with a learnable hard-gate routing. Meanwhile, we introduce an auxiliary-loss-free load balancing strategy to mitigate the potential routing collapse. Our empirical results reveal that MoDr achieves average accuracy improvements of +7.2% and +2.48% over the original Huginn model and its fine-tuned variant, respectively, across various mathematical reasoning benchmarks and improvements of +21.21% and +1.52% on commonsense reasoning benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
In this work, authors propose the Mixture-of-Depth-Recurrent Transformer (MoDr), which replaces the single recurrent loop in the Huginn model with multiple LoRA-based branches. To switch between branches authors propose a hard-gated dynamic routing mechanism that adaptively selects the best branch for each reasoning step. This allows the model to explore diverse reasoning paths in latent space efficiently. A loss-free load balancing mechanism ensures that routing remains stable and evenly distributed across branches.
Authors test their model on several math and commonsense reasoning tasks and show improvements over vanilla model and LoRA finetuning.

### Strengths
1. The paper in well organized and easy to follow
2. Authors propose a novel architecture, combining MoE with dynamic routing, LoRA-based adapters, and recurrent nature of the Huginn model. 
3. Authors explore an interesting paradigm for “dynamic latent exploration” during test-time reasoning, bridging the gap between latent reasoning and mixture-of-experts routing.
4. Authors show improvements on six mathematical and six commonsense reasoning datasets, covering both in-domain and out-of-domain settings.

### Weaknesses
1. While overall work is novel, all its components - Depth recurrence, MoE, and auxiliary-loss-free load balancing, - are assembled from existing paradigms rather than introducing a fundamentally new principle
2. The comparison set is limited mostly to Huginn and LoRA-Huginn models:
2a. For fair comparison, authors should specify additional compute budget, specifically number of additional parameters, total number of trainable parameters, spent FLOPS, and change in inference time, if any. Since MoDr introduces routing and multiple LoRA branches, empirical runtime and memory cost comparisons with Huginn are essential.
2b. Simple Chain-of-Thought prompting, majority voting, and full-SFT baseline are missing, while all being popular and cheap techniques widely used in literature and on practice.
3. The paper only reports Top-1 routing; Top-k routing or soft gating could reveal trade-offs between performance and stability.
4. The paper’s claim that routing improves reasoning exploration is plausible but largely untested. While the authors include a qualitative case study, no quantitative evidence shows that different branches develop distinct reasoning skills or that routing adapts to task complexity.

### Questions
See "Weaknesses" section

### Soundness
2

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
3

### Summary
This paper proposes Mixture-of-Depth-Recurrent (MoDr) Transformer, a novel extension of the depth-recurrent Huginn model. The key idea is to enhance the adaptability and exploration capability of latent-space reasoning by transforming Huginn’s single recurrent reasoning path into a multi-branch, dynamically routed system. Specifically, the authors:

Introduce multiple LoRA-based recurrent branches that share the Huginn loop weights, providing lightweight diversity in reasoning behavior.

Employ a hard-gate dynamic router to select the most suitable branch per token based on hidden-state context.

Design an auxiliary-loss-free load balancing strategy to prevent routing collapse during training.

### Strengths
Novel extension of depth-recurrent architectures — The work builds directly on Huginn and introduces a meaningful architectural innovation: turning the single-chain recurrent loop into a mixture-of-recurrent-paths framework.

Dynamic routing in latent reasoning — The proposed hard-gate router for token-level branch selection is conceptually elegant and empirically validated, pushing the field toward adaptive latent reasoning rather than verbal CoT.

Parameter-efficient design — The use of LoRA branches preserves the efficiency of the original model while expanding reasoning capacity.

Comprehensive experiments — Results on both mathematical and commonsense reasoning tasks are convincing, and ablations (router removal, branch count, load balancing) are thorough and insightful.

Strong empirical gains with negligible overhead, making the approach practical for future LLM test-time reasoning research.

### Weaknesses
I am not very familiar with the previous work (i.e., Huginn), but I find this paper to be quite intuitive and well-motivated. The Huginn model repeatedly reuses a single core module for latent reasoning, which inherently limits the model’s exploration depth. In contrast, this paper introduces a gated routing mechanism combined with LoRA adapters to merge multiple parallel recurrent branches, thereby enhancing the reasoning capability of LLMs. Overall, this is an elegant and valuable piece of work.

However, I still have some critical questions:

The paper only reports results on Qwen/Qwen2.5-Math-7B-Instruct, which is a relatively small model. Have the authors attempted to test larger LLMs (e.g., Qwen3, Qwen2.5-72B)? Since the proposed approach seems computationally lightweight, restricting the experiments to a single small model weakens the empirical support for the paper’s claims.

The evaluation benchmark is not sufficiently comprehensive. Some challenging reasoning benchmarks, especially in mathematics and science (e.g., AIME, GPQA, Super-GPQA), are missing, which limits the generality of the results.

### Questions
None

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes MoDr, a Mixture-of-Depth-Recurrent Transformer that improves test-time reasoning by replacing Huginn's single recurrent block with multiple LoRA-based branches and a hard-gate router.

### Strengths
1. The proposed idea is both novel and compelling. The authors introduce an innovative multi-branch dynamic routing mechanism for recurrent transformers, leveraging LoRA adapters to create lightweight and diverse reasoning branches, all while sharing the backbone weights for efficiency.

2. The method demonstrates strong performance, outperforming baseline models not only on in-domain tasks but also showing robust generalization to out-of-domain benchmarks.

3. The paper presents a comprehensive analysis, including detailed ablation studies, router effectiveness evaluations, branch specialization insights, and load balancing assessments. These provide robust empirical evidence supporting the model’s effectiveness, adaptability, and practical value.

### Weaknesses
1. It would be helpful to include more details about the trained model, such as training duration, detailed parameter count, etc.

2. A comparison with a standard Transformer model of similar size would strengthen the evaluation and better highlight the effectiveness of the proposed approach.

3. Including memory usage and FLOPs would provide a clearer picture of the model’s efficiency and practical deployment cost.

### Questions
NA

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper studies a modification to the recurrent-depth model Huginn aimed at improving performance on reasoning tasks. Specifically, it introduces MoDr: a set of multiple LoRA parameter groups and a routing mechanism that determines which LoRA parameter set will be active for a given token. After supervised finetuning on reasoning traces, MoDr outperforms the baseline model and the model with a single LoRA parameter set on mathematical and commonsense reasoning tasks.

### Strengths
**Originality**  The submission shows that the multiple branches of a MoDr model specialize in different areas, and the routing mechanism appropriately sends data to the right specialist to obtain high performance (Table 3). This analysis of MoDr's LoRA+routing approach supports this general direction's potential for impact. Moreover, the performance benefits on reasoning tasks are notable. 

**Quality**  The baselines facilitate testing of the proposed approach’s components (e.g., the contribution of the router). The method's generalization to OOD tasks is well established.

**Clarity**  The paper is mostly well written and clearly communicates key points. 

**Significance**  The overall idea to equip latent reasoning with different modes that a router can send tokens through is very interesting. As the authors point out, the potential for more efficient reasoning models makes this area of study significant.

### Weaknesses
I would be happy to raise my score if the following can be addressed:
- There might be a weakness regarding the baselines' training durations. Please see "Questions" for details.

- Important related work is not discussed in the main text and appears to be missing in the appendix. Clear discussion of prior work that uses MoE/routing + LoRA would help clarify the methodological novelty of the submission. Applying such an approach to Huginn-like models seems new, and the submission should clarify other novel aspects through such discussion.

A minor weakness is that some points are made multiple times (e.g. chain-like reasoning and loss-free balancing are mentioned >5 times each in the first few pages), and more efficient writing could make space for clearer discussion of related work in the main text. Currently, related work is primarily discussed in the appendix.

### Questions
- Please find and discuss related works to address the contextualization weakness above. Some examples might include StructMoE or Lottery Ticket Adaptation.

- Figure 1: Could this be modified to illustrate the proposed approach more clearly? In contrast, Figure 2 is very helpful and clear. 

- Line 246: The “relay race” analogy is unclear. Could you please clarify what is meant? Modifying Figure 1 could help address this point, too.

- How is the introduced load balancing different from the approach proposed in Wang et al. (2024)?

- Line 316: If the number of training epochs is greater for models with recurrent branches than it is for baseline models, should you have an ablation study that trains baseline models for more epochs? Relatedly, Figure 5 shows that performance improves with the number of branches, but the number of epochs is also changing, so it's unclear how much of the benefit is from branch count (as opposed to epoch count).

### Soundness
2

### Presentation
2

### Contribution
3
