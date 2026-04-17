# Multi-objective Hyperparameter Optimization in the Age of Deep Learning

- Decision: Reject
- Scores: 2, 2, 8, 6

## Abstract
While Deep Learning (DL) experts often have prior knowledge about which hyperparameter settings yield strong performance, only few Hyperparameter Optimization (HPO) algorithms can leverage such prior knowledge and none incorporate priors over multiple objectives. As DL practitioners often need to optimize not just one but many objectives, this is a blind spot in the algorithmic landscape of HPO. To address this shortcoming, we introduce PriMO, the first HPO algorithm that can integrate multi-objective user beliefs. We show PriMO achieves state-of-the-art performance across 8 DL benchmarks in the multi-objective _and_ single-objective setting, clearly positioning itself as the new go-to HPO algorithm for DL practitioners.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a multi-objective hyperparameter optimization method named PriMO that leverages expert prior knowledge. The approach integrates multi-objective expert priors into Bayesian optimization and utilizes cheap approximate surrogate models for initial design. Experimental results demonstrate that the method outperforms baseline approaches in both multi-objective and single-objective settings, while maintaining robustness to different prior strengths.

### Strengths
- The integration of expert prior knowledge with multi-objective optimization in hyperparameter tuning, as presented in this paper, represents a beneficial endeavor.  
- The proposed method, PriMO, demonstrates superior performance in experiments, outperforming baseline methods in both multi-objective and single-objective settings.  
- PriMO exhibits strong robustness to different prior strengths, and ablation studies confirm the effectiveness of each component of the framework.

### Weaknesses
- The description of the proposed method, PriMO, is relatively brief and lacks sufficient detail.  
- The baseline methods used for comparison in the experiments are somewhat outdated (ranging from 2006 to 2021).  
- There is a lack of case studies in real-world scenarios.

### Questions
*  Can the proposed method be applied to the fine-tuning or training of current popular LLMs?
*  How does the runtime efficiency of the proposed method compare to other approaches?
*  How is the expert knowledge introduced in this method defined, and can it be generalized to broader domains?

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper addresses a specific issue in HPO, how to utilize prior knowledge in multi-objective HPO. It proposes a Bayesian optimization algorithm, PriMO, that integrates an initial design strategy and prior weights in BO steps. Experiments show that PriMO performs well in different cases, including all-priors-good, mixed-priors, and all-priors-bad.

### Strengths
- Utilizing prior knowledge in multi-objective HPO is a good, under-explored topic. 

- Results exhibit good performance, whether the priors are good or bad.

### Weaknesses
- The title is too exaggerated in my eyes. HPO for deep learning faces numerous challenges, while the topic in this paper is only a very small one. Besides, it is not clear how the work addresses specific issues for deep learning.

- In practice, prior knowledge should be scarce and diverse. There is a lack of assumptions about the priors that this paper considers.

- The paper claims that priors can be good or bad. I wonder if it is a rigorous problem definition. How can you differentiate which ones are good or bad? If you cannot, how do you handle them differently?

- Figure 2 can not explicitly exhibit the motivation. First, it is not clear how to add prior on MOASHA or RS. Second, the advantages of adding prior in all-priors-good and MOASHA in all-priors-bad demonstrate the importance of how to differentiate good or bad priors, instead of the weakness of the naive method for adding priors.

- The main contribution comes from Eq. 4 and Algorithm 1. However, there is doubt that they are addressing issues related to the priors. 

- Experiments are limited due to the diversity of the benchmark.

### Questions
See Weaknesses.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper addresses the lack of HPO algorithms capable of incorporating multi-objective expert priors. While single-objective prior-informed optimization has received attention, extending this to multi-objective settings is both conceptually and technically nontrivial due to the need to reason over Pareto frontiers and conflicting objectives. The authors approach this issue starting from practical considerations in deep learning, looking at tradeoffs between accuracy, latency, cost, and fairness are common. The proposed PriMO framework provides a unified approach to integrate prior beliefs and cheap approximations while retaining robustness to misspecified priors.

### Strengths
1. The experimental section is definitive. The authors benchmark PriMO against a wide spectrum of baselines, ranging from classical multi-objective evolutionary algorithms to multi-fidelity optimizers (MOASHA, Hyperband) to Bayesian approaches. They also construct custom baselines (e.g., MOASHA+Prior, πBO+RW) to isolate the benefits of priors in the multi-objective context. PriMO consistently outperforms across eight deep learning benchmarks (image classification, translation, and language modeling) in both anytime and final performance metrics. It is rare in optimization to see such unilateral gains, which goes to show how underdeveloped the multi-objective HPO literature is and adds to the impact of this paper.

2. The authors keep practical considerations close to heart throughout the paper, which leads to very thorough investigation of relevant quantities like training cost and hypervolumes. This is very different from other approaches to multi-objective optimization, which tend to be either very theoretical and/or very complicated and engineered.

### Weaknesses
1. While the authors' acquisition function in Equation (4) works well empirically, the paper lacks a theoretical analysis of its properties. Ideal results would describe under what conditions we get convergence to the true Pareto frontier, or how the exploration parameter interacts with uncertainty estimation in BO. It is hard to know what the *secret sauce* of this choice is. I think this work would be stronger if there were some clear and simple example to have in mind that demonstrates the issue in multi-objective bilevel optimization which your approach solves/mitigates. I see that your algorithm looks reasonable and appears to do well, but in my opinion the most convincing results (and the ones that continue to hold at scale) are the ones with a clear "we unlock the ability to solve something other approaches completely fail at". Without knowing where to expect improvement to come from, it can make it very hard to refine and scale things.

2. It would be nice to have a more clear runtime analysis to understand exactly where the computation goes and how hyperparameters affect it. There is a lot of emphasis on wall-clock time improvement; the authors do well to demonstrate the improvement here, but in general it is good to have a theoretical asymptotic behavior to expect and aim to match/improve on.

### Questions
1. How do you envision practitioners specifying multi-objective priors? Would you consider incorporating structured or hierarchical priors for related tasks and subtasks?
2. I am naturally curious in all the theoretical properties: convergence to optimality, rates, robustness to noise, robustness to a bad prior, generalization to related problems, etc.. What do the authors expect based on the empirical anecdotal observations? What are the apparent strengths, weaknesses, and so on?

### Soundness
4

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
This paper presents PriMO, a prior-informed multi-objective hyperparameter optimization algorithm that extends Bayesian optimization with expert priors and a multi-fidelity initial design. The authors clearly motivate the gap that, while prior-based HPO methods exist for single objectives, no existing method supports multi-objective settings that are common in deep learning. PriMO introduces an ε-greedy acquisition strategy that balances prior guidance and exploration, and integrates a MOASHA-based warm-start to exploit cheap approximations. Extensive experiments across eight deep-learning benchmarks show strong anytime and final performance, outperforming state-of-the-art multi-objective and prior-based baselines.

### Strengths
The paper is very well written: definitions, algorithms, and experiments are presented cleanly and logically, making the work easy to follow. Multi-objective HPO is an important and practical topic for modern deep-learning workflows; addressing the lack of prior-aware solutions fills a real methodological gap. The proposed combination of prior-weighted acquisition with ε-greedy scheduling and a multi-fidelity initialization is conceptually coherent and empirically justified. The evaluation covers multiple domains, both surrogate and realistic, with clear ablation and robustness analyses that support the authors’ claims. Component-wise behaviors (e.g., prior strength, noise robustness, and early-stage acceleration) are analyzed in detail, giving the paper strong empirical credibility.

### Weaknesses
1. Beyond the Pareto-front visualizations, the paper could include more case-level examples or qualitative comparisons to help readers connect the optimization behavior with real task utility and model performance trade-offs.

2. In Algorithm 2 (the BO step), the parameter η is listed but seems unused—clarifying whether it affects fidelity scheduling or is inherited from the initialization stage would improve completeness.

3. A brief theoretical or intuitive discussion about how PriMO behaves when priors are highly correlated or partially redundant could further strengthen the understanding of its robustness.

### Questions
See weakness

### Soundness
4

### Presentation
3

### Contribution
3
