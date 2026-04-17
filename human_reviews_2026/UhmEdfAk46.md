# Multi-objective Large Language Model Alignment with Hierarchical Experts

- Decision: Accept (Poster)
- Scores: 4, 6, 6

## Abstract
Aligning large language models (LLMs) to simultaneously satisfy multiple objectives remains a significant challenge, especially given the diverse and often conflicting nature of human preferences. Existing alignment methods struggle to balance trade-offs effectively, often requiring costly retraining or yielding suboptimal results across the Pareto frontier of preferences. In this paper, we introduce HoE (Hierarchical Mixture-of-Experts), a lightweight, parameter-efficient, and plug-and-play approach that eliminates the need for model retraining, while enabling LLMs to adapt across the entire Pareto frontier and accommodate diverse user preferences. In particular, HoE consists of three hierarchical components: LoRA Experts, Router Experts and Weighting Router, reaching optimal Pareto frontiers and achieving a trade-off between parameter size, training cost, and performance. We evaluate HoE across various tasks on 16 objectives and 200 different preferences among 8 benchmarks, demonstrating superior performance over 15 recent baselines.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes HoE, a hierarchical mixture-of-experts framework for multi-objective alignment of large language models, enabling them to adapt to diverse and conflicting human preferences without costly retraining. HoE integrates LoRA experts, router experts, and preference routing to efficiently cover the Pareto frontier, providing scalable and fine-grained control over model behaviour.

### Strengths
The paper propose a novel alignment approach named HOE. There are some strengths:
* **Methodology**: This paper introduces a hierarchical expert-model framework (HOE) to handle multi-objective alignment, and incorporates Pareto-optimality concepts to provide a theoretical grounding for the approach.
* **Scalable and extensible**: leverages model fusion techniques and a lightweight routing module to enable efficient training with lower resource costs.
* **Experiments**: The evaluation covers multiple datasets and baseline methods, yielding a comprehensive analysis of the results.

### Weaknesses
However, there are some weakness of this paper:
* **Methodology**: multi-objective LoRA experts are expected to learn different preferences, and the experimental results also validate it. However, the design of multi-objective router expert seems to be redundant. The ablation study only discusses a single router and the role and necessity of a multi-expert router are not clearly demonstrated.
* **Reproducibility**: The results appear to rely on the pretrained model, yet the manuscript does not specify how the pre-trained model was chosen. The method mainly fine-tunes the routing layer while freezing LoRA parameters, but there is no detail on where the LoRA parameters come from or whether they need to be trained. Taken together, these points suggest certain reproducibility gaps in the code and experimental setup.
* **Evaluation**: The evaluation in this paper primarily relies on reward-model scores, which may limit the ability to capture objective, user-centred aspects of quality and user needs. It would be more persuasive if the authors could explain the rationale behind the chosen metrics.

### Questions
* Q1: The HOE integrates the PPO paradigm to optimise models. I wondering that if the reward model in PPO is the same with the reward model in evaluation?
* Q2: In Line 1233, there may be citation error "(??)". Please correct it.
* Q3: The method mainly fine-tunes the routing layer while freezing LoRA parameters, but there is no detail on where the LoRA parameters come from or whether they need to be trained.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper tackles the important and challenging problem of multi-objective alignment for LLMs. The core idea of using a hierarchical, decomposition-based MoE framework (HOE) is novel and parameter-efficient. The paper's primary strength lies in its comprehensive and very strong empirical results, demonstrating state-of-the-art performance across numerous benchmarks by dominating 15 baselines.

### Strengths
1. The core architectural idea of a hierarchical Mixture-of-Experts for MOA, inspired by decomposition methods, is novel. 

2. HOE achieves state-of-the-art performance, consistently dominating the Pareto frontiers of 15 competitive baselines (including RS, MOD, and RiC) in 2-objective settings. This is a very strong empirical contribution.

3. The paper features high-quality ablation studies that provide clear insights into the model's components.

### Weaknesses
1. The paper's central claim is the achievement of "optimal Pareto frontiers" and "superior Pareto-optimal results". However, the paper provides no evidence that the proposed HOE method actually converges to the true, global Pareto optimal frontier. The theoretical analysis in Appendix G relies on strong assumptions, such as the convexity of the objective functions (Assumption G.1), which are well-known to not hold in the non-convex landscape of LLM optimization. While the use of Tchebycheff (TCH) scalarization is appropriate (as it can find non-convex frontiers), it does not guarantee that the frontier it finds is the optimal one. Therefore, all of the paper's claims of "optimality" are purely empirical and lack the foundational theoretical guarantees that the term "Pareto optimal" implies.

2. The framework's design is confusing, as it introduces two distinct types of multi-objective components: "Multi-Objective LoRA Experts"  and "Router Experts". Both components appear to serve the exact same purpose: covering intermediate points on the Pareto frontier. The paper fails to provide a clear justification for why both are necessary.

3. This paper lacks of criticial details. A core component, the Merge function used to create multi-objective experts ($\tau_{\lambda}=Merge(...) $), is never defined in the paper.  The "task-SVD" compression process is also vague. The example in Appendix E.1  suggests it may involve manual, per-objective hyperparameter tuning, which would severely undermine the "plug-and-play" and "lightweight" claims. Furthermore, the mathematical formulation for the router optimization is inconsistent between the main text and the appendix, confusing any attempt at re-implementation. 

4. There are several issues for the Proof in App. G as well. 

First, the proof's strongest claims of convergence rely entirely on Assumption G.1 (Convexity). The parameters $\theta$ being optimized belong to the Router Experts, which are neural network layers within a Transformer. The optimization landscape for LLMs and deep neural networks is well-known to be highly non-convex.

Second, even if local stationary point convergence is established, a local stationary point is not, by any means, equivalent to a Pareto optimal solution.

Third, Lines 1480-1494 are very confusing. For example: “If none of these conditions hold but assumptions 1, 2, 3, and 4 remain valid..."

5. Please clarify some mathematical formulations. The main text in Section 3.2 (Eq. (6)-(8)) introduces an Online Mirror Descent (OMD) method for the Tchebycheff (TCH) objective . However, Appendix E.3 (Eq. (12)-(18)) shows a different-looking formulation that is supposedly based on the same TCH and OMD principles. The relationship between Eq. (8)  and Eq. (18)  is unclear, even though both appear to describe the same PPO gradient

### Questions
Please see the Weaknesses above.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors studied the problem of multi-objective alignment problem in LLMs. To address the problem, they decompose the alignment problem into a number of single-preference subproblems, each of which handled by specialized experts. They combined LoRA experts, router experts, and preference routing to address the problem in their hierarchical MOE framework.

### Strengths
The paper is well-written and easy to follow.
A number of different NLP tasks were taken to evaluate the performance of the proposed framework.
A number of datasets were used to conduct the experiments.

### Weaknesses
See the below questions.

### Questions
Motivations of why the authors combine LoRA experts, router experts, and preference for the multi-objective alignment problem are not convincing, as there are many methods/strategies to address such tasks. 

Because the proposed framework is combined by LoRA experts, router experts, and preference, there are a large number of parameters applied in the framework, although the authors tried to reduce the size of the LLM by their lightweight, parameter-efficient, and plug-and-play approach. In fact, such combination may not be appropriate to address the multi-objective alignment ask due to the size of the LLMs after the combination.

Not sure if there are other ways to eliminate the need to train the proposed model beside the propose HoE? 

Are there any additional baselines that were published in this year to be taken as baselines in the experiments?

### Soundness
3

### Presentation
3

### Contribution
3
