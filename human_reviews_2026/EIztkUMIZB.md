# What is an Optimal Growth Schedule for Large Language Models? A Theoretical Study

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 2

## Abstract
Existing training methods for Transformer-based large language models (LLMs) rely on massive amounts of data training from scratch, which requires a high cost in terms of computation and time. One promising research direction has developed effective lifelong learning pipelines for efficient LLM pre-training by growing from small pre-trained models to large ones—a technique known as model growth. There are two main research problems associated with model growth: growth schedule and growth operators. Existing research focuses on growth operators, detailing specific manipulations of potential dimensions to expand Transformer parameters. Few studies have investigated the optimal growth schedule, which involves integrating all possible growth operators to create an optimal multi-staged growth path. This work gives a theoretical study regarding what is an optimal growth schedule for multi-stage growth of LLMs by introducing a Schedule Learning methodology that uses an Optimal Path requiring minimal experimental training, referred to as SLOP. SLOP utilizes marginal utility as an appropriate measure for an optimal schedule that balances training costs and model performance after multi-stage growth. With this measurement, the objective of determining the optimal growth schedule is converted into a dynamic programming problem, which is then solved mathematically in polynomial time. Experiments with up to 7B target LLM show SLOP's theoretical validity as well as its efficiency, outperforming alternative schedules in a range of settings.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces SLOP, a methodology designed to find the optimal schedule (i.e., the sequence of operations) for the multi-stage growth of LLMs. The authors define four types of growth operators (layer, mha, ffn, hidden) and frame the problem as finding the optimal sequence of applying these operators. The paper proposes the "Marginal Utility of Schedule" (MUS) as a guiding metric. The central claim is that this optimization problem could be theoretically transformed into a dynamic programming problem, specifically minimizing the product of parameter increases at each stage (Eq. 10), thereby identifying an optimal schedule efficiently without exhaustive trial training.

### Strengths
Problem Relevance: The paper addresses a highly relevant and practical problem, scheduling of model growth.

Novel Formulation: The conceptual idea of framing the growth schedule as a sequential decision process that can be solved via dynamic programming or shortest path algorithms may be a novel approach.

### Weaknesses
Flawed Theoretical Derivation: The core weakness lies in the mathematical derivation in Section 3.4. It contains several steps that appear mathematically invalid, most notably the transformation in Eq. 7 (claiming $\arg\max \sum \ln(\cdot) \iff \arg\max \ln(\sum \cdot)$), which is incorrect and invalidates subsequent steps. The paper presents conflicting information.

Unjustified Assumptions and Objective Degeneration: The derivation relies on strong, unsupported assumptions, such as training time being proportional to parameter increase (Eq. 3) and final performance being independent of the growth path (Eq. 6). These assumptions lead to the degeneration of the final objective (Eq. 10) into a cost-only minimization problem ($\min \prod \Delta params$), which ignores the performance gains ($\Delta ppl$) central to the initial MUS definition.

### Questions
1.  Could the authors please provide a rigorous justification for the mathematical transformation in Eq. 7? The equivalence $\arg\max \sum \ln(u_k) \iff \arg\max \ln(\sum u_k)$ does not hold in general.
2.  What is the justification for the equivalence ($\iff$) in Eq. 3, which substitutes training time ($\Delta t$) with parameter increase ($\Delta params$)?
3.  Why does the final objective (Eq. 10) only depend on minimizing $\Delta params$ and not on $\Delta ppl$? This seems to contradict the initial motivation of maximizing the MUS (performance-per-cost).
4.  Could the authors clarify the contradiction regarding the MHA operator? Does it (Sec 4.1) or does it not (Appendix I) increase the model's parameter count?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies the optimal growth schedule for gradually expanding a small LLM to a target size along depth, width, and hidden dimensions. The authors propose SLOP, a dynamic-programming framework that maximizes the marginal utility of perplexity reduction per GPU hour. Polynomial-time optimization outputs the exact sequence of single-dimension growth stages. Experiments up to 7 B parameters show SLOP schedules reach equal or better perplexity and downstream scores while using consistently less wall-clock training time than hand-designed or one-shot alternatives, confirming the theoretical prediction.

### Strengths
This paper propses SLOP, which first defines and theoretically solves the multi-stage LLM growth-schedule problem by introducing an economic marginal-utility metric that makes the objective both quantifiable and optimizable. 

The authors prove the problem can be cast as dynamic programming and present a polynomial-time algorithm, eliminating the exponential cost of brute-force path search and offering excellent scalability.

Comprehensive experiments covering 1 B and 7 B target models and downstream benchmarks show that SLOP schedules achieve lower perplexity and higher task scores, validating the theoretical findings and demonstrating broad practicality.

### Weaknesses
1. Oversimplified theoretical assumption limits generalizability. The method assumes each dimension expands only once in a fixed sequence, which does not reflect the flexible, multi-dimensional growth strategies often needed in modern LLM development. 

2. Heavy reliance on standard growth operators without novelty or ablation. The performance depends heavily on existing operators like random initialization, yet no ablation studies confirm whether better operators would diminish the importance of the schedule itself.

3. Lack of comparison with latest methods: The baselines are mostly from 2022–2023; it is unclear if any new growth-schedule work exists and how SLOP would fare against it.

### Questions
As described in Weakness.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper describes a dynamic-programming solution for the optimal growth schedule for training LLMs, starting from a small, fixed structures, ending with a model with target parameter numbers.
The algorithm, namely SLOP (Schedule Learning via Optimal Path), optimizes marginal utility, defined as  the derivation of the reduction of ppl to training time, in each stage. There are strong assumptions on the operator space, growth stage numbers, and the objective function.
Experimental results show that SLOP results in improved wall-time and perplexity.

### Strengths
1. This is among the first researches that seriously consider the optimization of growth schedule, as most related work focus on growth operators. The SLOP algorithm provides an interesting solution under its assumptions.
2. The paper provides experimental results with multiple settings, partially supporting their claim that SLOP finds decent growth schedules. The perspective has some novelty.

### Weaknesses
There are a noticeable amount of strong assumptions which significantly impacts the soundness of this work. 

Examples:
(1) as claimed in Eq.2, delta_t means the training time from M_k to M_k+1, then why is it converted to delta_params in Eq.3? As far as I understand, it should directly be params(M_k+1), as the full substage is trained with M_k+1. 

(2) why is Eq.2 equivalent to maximizing the overall delta(ppl) / delta(t) from stage 1 to 4?

(3) the relaxation from Eq.4 to Eq.5 is as overpowered as Eq.2

(4) alleviating part.1 in Eq.6 with Chinchilla law is confusing, as Chinchilla law (and most other scaling laws) is achieved with **full-model** training with **full data**, we can not assume it still holds for multi-staged growth. Moreover, as presented in this paper's experimental results, with the same data D, different structures and different growth strategies obviously result in different perplexity, which can not be considered as constant when defining the optimization objective.

(5) The solution is demonstrated in a fixed 4-stage growth, which is far from answering ``when to grow''. The number of steps of each stage seems not considered. I do not feel there is actually a large search space under these settings.

### Questions
Please clarify the motivation and generalization scope of the strong assumptions I mentioned above.

### Soundness
2

### Presentation
2

### Contribution
1
