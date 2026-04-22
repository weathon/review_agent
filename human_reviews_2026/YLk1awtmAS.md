# On Training Mixture-of-Experts: A Social Choice Perspective

- Avg Score: 3.60
- Decision: Reject
- Scores: 2, 4, 6, 4, 2

## Abstract
Mixture-of-Experts (MoE) training faces a dilemma between expert specialization and balanced computation. We recast this problem through the lens of social choice theory, attributing training difficulties to Arrow's Impossibility Theorem. Inspired by this, we propose Regulated Mixture-of-Experts (RMoE), comprising a phased curriculum for load-balancing and stateful fusion for expert weighting. Experiments on GLUE and DomainBed show RMoE significantly outperforms standard MoE and dynamic routing baselines. Furthermore, RMoE demonstrates strong scalability on large-scale reasoning tasks with Qwen3 and Mixtral architectures. Our code is available at https://anonymous.4open.science/r/R-MoE-E3DC.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper reframes MoE training through the lens of social choice theory, arguing that the difficulty in balancing task performance and load balancing can be attributed to Arrow's Impossibility Theorem. The authors propose RMoE, which combines a phased curriculum for the load-balancing loss weight and stateful fusion using momentum for expert weighting. They demonstrate improvements over baselines on GLUE and domain generalization benchmarks.

### Strengths
1. The paper attempts an interesting interdisciplinary connection between MoE training and social choice theory, which offers new insights into understanding routing collapse. 

2. The experimental results show consistent improvements over baselines across multiple benchmarks. 

3.  the paper includes extensive ablation studies and analysis of expert specialization patterns.

### Weaknesses
1. The social choice framing feels more like a loose analogy than a rigorous theoretical foundation. While the paper invoke Arrow's Impossibility Theorem, they don't provide a formal proof of how it applies to MoE training - the mapping between voting systems and routing is imprecise. The actual solutions proposed  are fairly standard techniques that don't really emerge from social choice principles. For instance, curriculum-based approaches for MoE have been explored in [1] and progressive training strategies in [2].

2. The technical contributions are incremental. The phased curriculum is essentially scheduling the auxiliary loss weight, which has been explored in various forms (e.g., gradual unfreezing, warm-up schedules). The stateful fusion mechanism is just applying EMA to routing scores, similar to momentum-based methods in optimization. Neither innovation requires the social choice perspective to motivate or understand.

3. The experimental setup has significant limitations. All experiments use relatively small models, making it unclear if the approach scales to production-scale MoE models like Mixtral-8x7B [3] where routing collapse is more problematic. 

[1] Lewis M, Bhosale S, Dettmers T, et al. Base layers: Simplifying training of large, sparse models[C]//International Conference on Machine Learning. PMLR, 2021: 6265-6274.
[2] Zhou Y, Lei T, Liu H, et al. Mixture-of-experts with expert choice routing[J]. Advances in Neural Information Processing Systems, 2022, 35: 7103-7114.
[3] Jiang A Q, Sablayrolles A, Roux A, et al. Mixtral of experts[J]. arXiv preprint arXiv:2401.04088, 2024.

### Questions
1. Can you provide a formal proof showing how Arrow's theorem applies to MoE routing? 

2. How does RMoE perform on larger-scale MoE models like [1]? 

[1] Jiang A Q, Sablayrolles A, Roux A, et al. Mixtral of experts[J]. arXiv preprint arXiv:2401.04088, 2024.

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper connects the training of MoE to the social choice theory, where there is a conflict between task efficiency (utilitarian) and load balancing (fairness). It proposes two strategies that help to train MoEs better: a) phased curriculum where the coefficient of load balancing loss is decayed linearly b) momentum-based weight fusion of expert outputs. It is shown that both strategies help to achieve better performance and specialization of experts.

### Strengths
- The proposed method is simple to implement and shows better results compared to baseline like DynMoE 
- The problem of MoE routing is relevant to the community, especially for training at scale. 
- Experiments are conducted in both language and vision domains

### Weaknesses
- The connection to social choice theory seems weaker. Impossibility theorem mentioned in abstract is not elaborated elsewhere in the paper and it is harder to make connection to MoE training 
- Second-best seems like a theoretical concept that applies when all objectives cannot be satisfied. It feels like inspiration rather than a technical solution for a given problem. Could you tell why it is relevant here? 
-  Are these results using multitask training from GLUE tasks? The method would be convincing if trained in multitask fashion. One baseline to beat here would be training a single expert on all data. 
- The proposed method should be shown at scale for it to be practical. Eg. pretraining on 1B tokens with model size of 1B or lower. If the focus is on downstream tasks, then approaches where experts are parameter efficient modules (https://arxiv.org/abs/2306.03745) should be compared. 
-   Add more baselines like SoftMoE (https://arxiv.org/abs/2308.00951), DeepSeek MoE (https://arxiv.org/pdf/2401.06066), auxiliary free load balancing (https://arxiv.org/pdf/2408.15664v1) to make the work more comprehensive.

### Questions
- How is the moving average of routing scores done? Is it across the same tokenID appearing through training? Or are you averaging as you move across sequence length? 
- How do you handle moving averages at inference time? The text in lines 314 and 315 is incomplete. 
- Figure 3 doesn’t provide information about stable convergence of the proposed method over the baseline. Could you elaborate more?

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
4

### Summary
This paper redefines the routing problem in MoE as a social choice problem: the input token is regarded as the "agent", the expert as the "candidate", and the router as a social welfare function that aggregates preferences. The author points out that the trade-off between task loss and load balancing loss in MoE training is similar to the conflict between efficiency and fairness in social choice, and borrows Arrow's impossibility theorem to explain the theoretical root cause of training difficulties. Based on this, they proposed the RMoE framework, which consists of two mechanisms: phased courses (gradually increasing the weight of load balancing losses) and state fusion (using EMA to smooth the expert fusion weights). Experiments were conducted on GLUE and DomainBed, demonstrating that RMoE outperformed multiple baseline

### Strengths
1.  Providing code for reproducing the experiments is commendable.

2.  Linking the routing problem of Mixture-of-Experts (MoE) with social choice theory offers a brand-new theoretical perspective for understanding routing crashes and training instability.

3.  The two proposed mechanisms (phased training and state fusion) are elaborated in detail.

### Weaknesses
1. The experiments were based on BERT-base and ViT-Small, and their scalability was not verified on larger-scale models such as MoE with undreds of billions of parameters.

2. The essence of phased learning is to dynamically adjust the loss weights, which has been widely applied in multi-objective optimization and phased learning. EMA smoothing in state fusion is also very common in time series models.

### Questions
1. Could you provide a rigorous argument for directly mapping the MoE training problem to the conditions of Arrow's impossibility theorem (such as independence and non - dictatorship)?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Regulated Mixture-of-Experts (RMoE), a framework for improving the training stability of MoE models. The authors frame the trade-off between task performance and load balancing as a social choice problem, attributing the difficulty to Arrow's Impossibility Theorem. They propose two main components: a "Phased Curriculum" for scheduling the load-balancing loss and Stateful Fusion which uses an EMA to smooth expert weights. Experiments on GLUE and DomainBed show improvements over baseline MoE and DynMoE models.

### Strengths
The high-level perspective of connecting MoE training to social choice theory is creative and provides an interesting narrative.

### Weaknesses
Limited Technical Novelty: The core technical contributions are essentially reinterpretations of well-established techniques.

The "Phased Curriculum" is a simple linear annealing schedule for an auxiliary loss weight, a common practice in machine learning (e.g., β-annealing in VAEs).

Stateful Fusion is an application of Exponential Moving Average (EMA) to introduce momentum and stabilize training, a concept that is neither new nor unique to this work. Its conceptual overlap with existing momentum-based methods is significant.

Superficial Theoretical Grounding: The connection to Arrow's Impossibility Theorem is presented as a high-level analogy rather than a rigorous, formal framework. The paper fails to formally map the components of MoE training (tokens, experts, router) to the axioms required by the theorem (e.g., non-dictatorship, independence of irrelevant alternatives). Consequently, the theoretical framing feels like a post-hoc justification for heuristic design choices, rather than a principled foundation that guides the method's development.

### Questions
Q1 Can you better articulate the novelty of the proposed mechanisms beyond being applications of loss scheduling and EMA smoothing? What distinguishes them fundamentally from prior work using these concepts?

Q2 Can you provide a more formal proof of how MoE training violates the axioms of Arrow's theorem? Following that, how do the specific designs of Phased Curriculum and Stateful Fusion mathematically address or relax these axioms to "escape" the impossibility result?

Q3 Could you justify the omission of critical baselines like "Loss-Free Balancing"? Furthermore, can you provide a complexity analysis of RMoE to demonstrate its viability for models with thousands of experts?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This study designed a two-stage mechanism to regulate the expert selection problem in MoE, improving model performance, and attempted to explain it using social choice theory, which is quite an interesting approach.

### Strengths
This work tries to capture the dynamic process in global and local optimization in MoE expert selection by considering the two stage mechanism. Such two process sounds common in other domains, especially in optimization field. I think this idea is worthy of further study.

### Weaknesses
1, The author's discussion in the text is misleading. From the perspective of our sequential social choice framework, the MoE training objective represents a classic social dilemma, characterised by a conflict between two competing social welfare objectives: utilitarianism (efficiency) and egalitarianism (fairness) (Sen, 1977; 1986). The author has inserted their own viewpoint into the references, misleading readers into thinking that it is the original author's opinion, which is inappropriate.

2, The proposed controlled mixture-of-experts RMoE has a certain novelty, but the working mechanism of its training method needs to be further clarified. Perhaps this process is closer to simulated annealing, for example, the interaction mechanism between the first stage Phased Curriculum and the second stage Stateful Fusion in the training process.

3, The improvement in experimental performance is limited. The performance improvement compared to the baseline model is relatively limited.

### Questions
I personally strongly disagree with using social choice theory to explain the expert selection problem in MoE, because these are two completely different issues with clearly distinct mechanisms. In social sciences, we need to emphasise efficiency and fairness, as fairness can affect efficiency. However, in MoE, if efficiency can be guaranteed, that is, if performance can be ensured, why emphasise fairness? Could the author explain how fairness in MoE affects efficiency? To be more specific, if fairness can guarantee efficiency, then simply distributing tokens equally to each expert would suffice, so why don't we do this?

### Soundness
2

### Presentation
3

### Contribution
2
