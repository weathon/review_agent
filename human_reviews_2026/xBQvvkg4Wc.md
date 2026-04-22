# Training Large Language Models To Reason In Parallel With Global Forking Tokens

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Although LLMs have demonstrated improved performance by scaling parallel test-time compute, doing so relies on generating reasoning paths that are both diverse and accurate. For challenging problems, the forking tokens that trigger diverse yet correct reasoning modes are typically deep in the sampling tree. Consequently, common strategies to encourage diversity, such as temperature scaling, encounter a worsened trade-off between diversity and accuracy. Motivated by this challenge, we treat parallel reasoning as a set-of-next-token-prediction problem and incorporate a set-based global loss into Supervised Fine-Tuning (SFT) using bipartite matching between global forking tokens and unique reasoning traces. We observe that whereas naive fine-tuning with multiple reasoning traces collapses these unique reasoning modes, our proposed method, Set Supervised Fine-Tuning (SSFT), preserves these modes and produces emergent global forking tokens. Global Forking Policy Optimization (GFPO) leverages these maximally steerable tokens to incentivize complex reasoning, and the resulting models consistently outperform their SFT counterparts with GRPO on both math reasoning and execution-based code generation benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper tackles "mode collapse" in LLMs trained on diverse reasoning traces. It introduces Set Supervised Fine-Tuning (SSFT), which frames parallel reasoning as a set prediction problem. SSFT uses a set of global forking tokens $\{g^{(i)}\}$ and multiple reasoning traces $\{r^{(j)}\}$. At each step, it finds an optimal bipartite matching $\hat{\sigma}$ between tokens and traces by minimizing the NTP loss. The model only optimizes this set-based loss, $\mathcal{L}_{Hungarian}(\theta)$. This forces tokens to learn distinct reasoning modes, improving both Pass@1 (single) and Cons@k (voting) accuracy on reasoning benchmarks.

### Strengths
1. This paper introduces a novel problem formulation, treating parallel reasoning as a "set-of-next-token-prediction" task. It applies a set-based loss via bipartite matching (inspired by DETR in vision) to LLM finetuning.
2. The empirical evaluation is rigorous. By comparing SSFT against baselines trained on the exact same data (SFT-mixed) and an ablation (SSFT-random), the paper proves the method's core components are responsible for its success. Visualizations (Fig. 4 vs 5) powerfully confirm that SSFT prevents the "mode collapse" that plagues naive finetuning.
3. The paper is clearly written, and its core, complex mechanism is made simple and intuitive by the visualization in Figure 1, which illustrates the entire training step.
4. This work provides the first effective method for multi-teacher distillation, allowing a single model to learn diverse reasoning strategies. It significantly improves the sample efficiency of parallel decoding (Cons@k) and introduces "global forking tokens" as a new, powerful form of steerable generation.

### Weaknesses
1. Computational Scalability: The SSFT algorithm requires $\mathcal{O}(N \times M)$ forward passes (where $N$=tokens, $M$=traces) at each training step to build the cost matrix. This is a potential bottleneck that may not scale efficiently if researchers want to use many more tokens or traces.
2. Reliance on Matching Heuristic: The optimal matching is computed using only the first $T_L=1000$ tokens of each trace for efficiency. The paper lacks an ablation study on this critical $T_L$ hyperparameter, making it unclear if the method is robust for reasoning paths that only diverge after this prefix.
3. Limited Task Scope: The experiments are confined to mathematical and structured-knowledge reasoning. It is not demonstrated how SSFT would perform on more open-ended, creative, or argumentative tasks where "correctness" and "diversity" are less clearly defined.
4. Weak Justification for Pass@1 Token: The heuristic for choosing the best token for Pass@1 inference (the one with the "largest coverage") is not well-justified. The paper does not provide strong evidence for why this "flexibility" heuristic should correlate with the highest single-attempt accuracy.

### Questions
1. The caption of the table should be placed above.
2. See Weakness.

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
2

### Summary
The paper treats parallel reasoning as a set-of-next-token-prediction problems. The paper proposes Set Supervised Fine-Turning (SSFT). They incorporate a set-based global loss and use the loss for Supervised Fine-Turning (SFT). Experiments on multiple reasoning benchmarks show our SSFT method consistently outperforms SFT.

### Strengths
1. I think the paper treats parallel reasoning as a set-of-next-token-prediction problem is an interesting and useful idea.
2. The paper focuses on an important question.
3. The motivation and intuition under the paper is pretty clear and easy to follow.

### Weaknesses
1. The experiments are only conducted with a single base model. But I think it will be helpful is models with different size and from different model families can also be used.
2. I think the comparison to baseline is a bit unfair. For (MultiTarget), they use one <think> token for all four traces and treating them as four individual data points. This can make all traces mixed together and confuse the model. I think a fairer comparison is to use different tokens for different traces. As you do for SSFT. (Single-Target △) models trained with one trace per question and (Multi-Target ⋆) models trained with four traces per question. Does it mean that the total number of training tokens is different?
3. It's unclear to me how robust the algo is, especially the matching part. I think there will be probamatic if the generated trace of the base model is very similar to one if the teacher model. For example, if the base model is deepseek-distilled, and deepseek is one of the teacher model.

### Questions
1. "To find the optimal bipartite matching for each input prompt, we consider only the first 1,000 tokens when computing the matching cost in Equation 2 for computational efficiency." Why did you only choose the first 1,000 tokens? Is it enough for computing the matching cost? How accurate is it? 
2. Have you ever compared the methods to the RL method? [1,2] can also generate different length of the outputs guided by instructions/tokens.
3. When generate Training Dataset, what to do if a teacher model can't generate correct traces? 
 [1] Zhang, Xuechen, et al. "Making small language models efficient reasoners: Intervention, supervision, reinforcement." arXiv preprint arXiv:2505.07961 (2025).

[2] Aggarwal, Pranjal, and Sean Welleck. "L1: Controlling how long a reasoning model thinks with reinforcement learning." arXiv preprint arXiv:2503.04697 (2025).

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes Set Supervised Fine-Tuning (SSFT), which introduces global forking tokens prior to generating parallel reasoning traces and frames parallel reasoning as a set prediction problem. This method expands the diversity of generated traces and keeps the accuracy of the generation at the same time.

### Strengths
1. The motivation of this work is reasonable, which points out the shortcomings of the current parallel scaling methods and the trade-off between diversity and accuracy.
2. Based on the motivation in (1), the logic of the design of the framework and matching algorithm makes sense.
3. The experiment part shows the effectiveness of this method compared to other baselines in both mathematical and STEM reasoning tasks. 
4. This work also introduces a way to deploy the method efficiently, which is a good engineering contribution, although this might not be the main focus of this paper.

### Weaknesses
1. Some obvious and important typos, such as Fine-tuning not Fine-turninig.
2. The main result section might need more diversity on model families and model sizes. 
3. Maybe an analysis of the generalization ability of this method is needed.

### Questions
1. Could you please include an analysis of out-of-distribution performance after training?
2. Could you please include 1 or 2 more different sizes of models in the main experiment part?
3. Could you please include 1 or 2 other model families to show the generalization of this method, such as Qwen3, Llama, or Phi?

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
This paper introduces **Set Supervised Fine-Tuning (SSFT)**, a novel framework designed to make Large Language Models (LLMs) generate multiple, diverse, and correct reasoning paths simultaneously. SSFT reframes parallel reasoning as a "set-of-next-token-prediction" task. It employs a set of learnable **"global forking tokens"** and a **bipartite matching** loss function to associate these tokens with different reasoning solutions during training. This approach enables the model to produce varied reasoning patterns when prompted with different forking tokens, effectively overcoming the typical trade-off between diversity and accuracy. Experiments show that SSFT-trained models significantly outperform standard Supervised Fine-Tuning (SFT) on challenging reasoning benchmarks.

### Strengths
1.  **High Innovation**: The SSFT framework and the concept of "global forking tokens" are highly original. Formalizing parallel reasoning as a set prediction problem with bipartite matching is a significant and clever departure from standard fine-tuning paradigms, effectively mitigating "mode collapse."
2.  **Effective and Elegant Method**: The use of the Hungarian algorithm for optimal matching allows the model to end-to-end learn associations between forking tokens and different reasoning styles without explicit labels. This design is both elegant and highly effective.
3.  **Strong Empirical Results**: The SSFT model demonstrates superior performance on difficult reasoning benchmarks like AIME and MATH-500. The ablation studies are thorough, clearly proving the necessity of the optimal matching component and showing a better diversity-accuracy balance compared to baselines.
4.  **High-Quality Presentation**: The paper is well-written and logically structured. The visualizations provide clear, intuitive evidence for the method's effectiveness and interpretability.

### Weaknesses
1.  **Implementation Complexity**: SSFT's training process, which requires solving a bipartite matching problem at each step, is more complex than standard SFT. This could pose a barrier for application on larger-scale models or datasets.
2.  **Data Dependency**: The method's success hinges on access to a high-quality, diverse set of reasoning paths, which can be costly to obtain. The performance might degrade if such data is unavailable. But considering that this work explicitly aims to preserve the model’s diversity under such settings, this weakness is not particularly limiting.
3.  **Heuristic for Single-Path Inference**: The strategy for choosing a forking token in single-pass (Pass@1) inference is based on a heuristic (the token with the most connections). This choice lacks strong theoretical justification and could be explored further.

### Questions
- What is the expected performance of SSFT on much smaller (<7B) or larger (>70B) models?
- Regarding the number of global forking tokens, N, I would like to explore the potential impact of its fixed setting. The inherent diversity of reasoning paths differs for any given problem. Is it possible that a uniform N value could create redundancy for simpler problems while failing to fully capture the diversity of more complex ones? I am curious about the authors' perspective on the feasibility and potential of dynamically adjusting N for each problem. For instance, could performance be improved by a mechanism that predicts or assigns an optimal N' value tailored to each specific question?
- This paper brilliantly demonstrates how SSFT addresses mode collapse in SFT. An interesting parallel occurs in the reinforcement learning (RL) domain, where "entropy collapse" is also a core challenge—the model overfits to a few high-reward trajectories, thereby losing its exploratory capacity. I would like to ask the authors if they believe SSFT's core mechanism (i.e., using forking tokens and set matching to "protect" diversity) could offer new insights for solving entropy collapse in RL. Specifically, how might the ideas from SSFT be combined with or complement existing RL techniques aimed at maintaining policy entropy, such as entropy regularization?
- I've noticed that the experiments in this paper are primarily focused on mathematical reasoning datasets. A notable characteristic of these datasets is that a single problem often has multiple, distinct, and structurally varied correct solution paths. SSFT has achieved great results by using N explicit forking tokens to capture this diversity. However, my question is whether this advantage can naturally generalize to other, non-mathematical domains. In many general-purpose scenarios (such as open-ended question answering, creative writing, or code generation), "diversity" may manifest in a more implicit and subtle manner, rather than as the several clearly distinct "solutions" seen in math problems. In these contexts, is forcing a match between N discrete forking tokens and this more continuous or ambiguous diversity still the optimal strategy? Could the performance of SSFT in these general-purpose domains be limited by this "explicit forking" design?

### Soundness
3

### Presentation
3

### Contribution
3
