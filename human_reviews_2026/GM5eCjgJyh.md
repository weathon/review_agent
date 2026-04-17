# M$^2$GenCO: Multi-task Meta Learning for Generative Combinatorial Optimization

- Decision: Reject
- Scores: 2, 2, 6, 4

## Abstract
Despite the fast progress, especially recent diffusion-based models for solving Combinatorial Optimization Problems (COPs) on graphs, existing neural solvers mainly learns a narrow task (e.g., uniform TSP) at a time and hardly handle instances with diverse distributions. To fill such gaps, this paper proposes M$^2$GenCO, a multi-task learning paradigm that pioneers the incorporation of generative CO solving into the meta-learning mechanism for graph-based COPs, first formulating "tasks" in meta-learning as distinct problem types instead of instances of the same problem. Additionally, a lightweight graph neural network with a hybrid of task-specific and shared encoding blocks is tailored to instantiate the framework, performing effective joint pre-training on a variety of problem types and efficient fine-tuning to adapt for out-of-distribution scenarios. Further, we establish a comprehensive benchmark comprising 5 classic graph-based COPs with varying scales and multiple distributions, forming 38 distinct test datasets that facilitate standard evaluation of generalizability and adaptability for neural CO solvers, which has not been well developed in literature. Empirically, M$^2$GenCO with only greedy decoders yields an overall 9.16% performance gain with an average 95.6$\times$ acceleration for inference, and achieves concrete state-of-the-arts on all test sets with simple local searchers, maintaining superior solving time against previous neural methods. Meanwhile, the resource and time consumption for training are saved by up to 82\% and 91\%, respectively.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes M²GenCO, claiming to be the first multi-task meta-learning framework for graph-based combinatorial optimization (CO). The core idea is to use MAML to pretrain across different CO problem types (TSP, ATSP, MIS, MCl, MCut), instantiated with a consistency model as the backbone. The model is pretrained on certain distributions (RB graphs for MIS/MCl, BA for MCut, uniform for TSP/ATSP), then finetuned on new distributions (ER, HK, WS, Gaussian, etc.) in a few-shot manner. The authors also contribute a benchmark with 38 test datasets across 5 COP types and multiple distributions.

### Strengths
1. **Benchmark contribution**: The proposed multi-distribution benchmark (38 datasets across 5 COPs) addresses a gap in the literature. Most prior work only evaluates on uniform/in-distribution instances, so having systematic out-of-distribution test sets is valuable for the community.

2. **Strong reported performance**: According to the experimental results, the method shows significant improvements over SOTA baselines across the benchmark, along with substantial speedup. **If these results hold under fair comparison settings**, they would represent meaningful progress for the field.

### Weaknesses
### Critical Issues

#### 1. The "meta-learning" framing is fundamentally misleading

The paper claims to do meta-learning across different COP types, but there's a fatal flaw: **pretrain and test use completely different distributions**. 

- **Pretrain**: RB (MIS/MCl), BA (MCut), Uniform (TSP/ATSP)
- **Test**: ER, HK, WS, Gaussian, Cluster, HCP, SAT, etc.

This violates the core assumption of meta-learning, where train and test tasks should come from the same task distribution $p(\mathcal{T})$. What you're actually doing is **transfer learning** - pretraining on one set of distributions and adapting to completely different ones. This is a well-established paradigm (think ImageNet pretraining), not a novel meta-learning approach. The entire conceptual framing needs to be reconsidered.

#### 2. Minimal algorithmic innovation

Looking at the technical components:
- MAML: Finn et al. 2017
- Consistency models: Song et al. 2023
- Multi-task learning: decades-old paradigm
- GCN for COP: Joshi et al. 2019

The paper essentially combines existing techniques without introducing new algorithms or theoretical insights. The "innovation" of treating different COP types as meta-tasks is actually quite natural in the MTL literature - it's standard practice to have diverse task types, not a conceptual breakthrough.

#### 3. Naive MAML implementation ignores known problems

The paper adopts vanilla MAML without addressing well-documented issues:

- **Task conflict**: Different COPs (TSP needs connectivity, MIS needs sparsity, MCut needs balanced cuts) have fundamentally different structures. How do shared parameters simultaneously optimize for these conflicting objectives? No discussion.

- **Negative transfer**: When tasks are very different, meta-learning can hurt rather than help. No analysis of when/why negative transfer might occur.

- **One-step inner loop**: You do $\theta' = \theta - \alpha \nabla L$ (Eq. 1) for ONE step. But consistency models typically need thousands of steps to converge. How can one gradient step meaningfully adapt the model? This seems fundamentally incompatible with the CM backbone.

- **Gradient pathology**: Second-order gradients in MAML are notoriously unstable, especially with deep networks (your 6-layer GCN). The L2 normalization (Eq. 3) is a band-aid, not a solution. Why not use ANIL, BOIL, or other improved meta-learning methods?

#### 4. Training setup for baselines is completely opaque

This is extremely problematic. You detail your own training (Algorithm 1, full architecture spec) but say **nothing** about how baselines were trained:

- Were baselines also pretrained on RB/BA/Uniform then finetuned on test distributions?
- Or were they trained from scratch on your train or test distributions?
- If the latter, you're comparing pretrained (yours) vs. non-pretrained (baselines) - fundamentally different settings

Without this information, I cannot assess whether the performance gains come from (a) the meta-learning mechanism, (b) multi-distribution pretraining, or (c) simply using more diverse training data. This is a critical omission.

#### 5. Testing protocol appears biased

The statement "We set batch_size=1 and use greedy decoders for all tests unless otherwise stated" is concerning:

- **Greedy decoding severely handicaps sampling-based methods**: diffusion models all rely on multiple samples or search algorithms. Forcing greedy can cause 5-10× performance drops for these methods.

- **Double standard**: Table 2 shows M²GenCO with ‡ marks (using MCTS), but baselines don't get this. DIMES also uses MCTS in its original paper - why is it denied here?

- **Vague definition**: What exactly does "greedy" mean? Does it prohibit sampling? Augmentation? MCTS? The ambiguity is suspicious.

If all methods must use greedy but only yours gets additional post-processing, that's not a fair comparison.

### Major Issues

#### 6. No statistical significance testing

Every single result in Tables 2-3 is a point estimate with no error bars, standard deviations, or significance tests. How many runs did you do? Is the 9.16% improvement statistically significant or within noise? This is unacceptable for an ML venue in 2025.

#### 7. Code not available during review

You promise to release code "upon publication" but provide nothing for reviewers to verify. This makes it impossible to check:
- Whether results are reproducible
- How baselines were actually implemented
- Whether there's cherry-picking
- The exact definition of "greedy decoder"

### Minor Issues

#### 8. Incomplete ablations

Table 5 shows with/without finetuning and Fig 2 shows with/without diffusion/meta-learning, but missing:
- Effect of task pool size $m$ (what if $m=10$ instead of 2-3?)
- Effect of inner learning rate $\alpha$, outer learning rate $\beta$
- Effect of number of meta-tasks $k$ per iteration
- Ablation of gradient normalization (Eq. 3)

#### 9. Hyperparameter sensitivity not analyzed

MAML is notoriously sensitive to learning rates, yet you don't show how performance varies with $\alpha$ and $\beta$. This makes it hard to know if results are robust or require careful tuning.

**Minor note**: Table 1 has a reference error - MAB-MTL cites Liu et al., 2024b, which is a mismatch on this paper.

### Questions
**Q1**: Can you clarify the baseline training protocol?
- Exactly what distributions were each baseline trained on?
- Were baselines pretrained on RB/BA/Uniform and then finetuned on test distributions, matching your setup?
- If not, why not? You're comparing pretrained (yours) vs. non-pretrained (theirs), which seems unfair.

**Q2**: What does "greedy decoder" mean precisely?
- Does it prohibit sampling? How many samples?
- Does it prohibit augmentation (like POMO's multiple starting points)?
- Does it prohibit MCTS/local search?
- If MCTS is allowed, why do only some methods (yours) have ‡ marks in Table 2?

**Q3**: Can you provide statistical significance?
- How many random seeds did you run?
- What are the standard deviations?
- Which results are statistically significant ($p < 0.05$)?

**Q4**: Can you explain the distribution mismatch?
- You pretrain on RB/BA/Uniform but test on ER/HK/Gaussian/etc.
- This violates standard meta-learning assumptions (train/test tasks from same distribution)
- Why is this still "meta-learning" rather than transfer learning?

**Q5**: Have you tried the fair baselines?
- **Baseline 1**: Multi-task pretraining WITHOUT meta-learning (no MAML, just standard MTL)
- **Baseline 2**: Your method trained from scratch on test distributions (no pretraining)
- This would isolate the contribution of the meta-learning mechanism

**Q6**: How does one-step inner loop help?
- Consistency models typically need thousands of training steps
- How can $\theta' = \theta - \alpha \nabla L$ (one step) meaningfully adapt the model?
- Did you try multi-step inner loops?

**Q7**: Do you have evidence of negative transfer or task conflict?
- What happens when you train on conflicting tasks (e.g., TSP + MIS)?
- Is there ever negative transfer where multi-task hurts compared to single-task?

**Q8**: Why are training costs lower?
- CM training is slow, MAML adds overhead (multiple forward/backward passes)
- Why is your training faster than single-task methods?
- Are you sure baselines were trained to convergence?

**Q9**: What's the performance with sampling?
- If you allow sampling (e.g., 100 samples) for all methods including yours, what happens?
- This would be a fairer comparison for diffusion/RL baselines

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper presents “M2GenCO" a Multi-Task Meta Learning for Generative Combinatorial Optimization. M2GenCO defines each problem type as a meta-task, enabling cross-problem generalization and few-shot adaptation to unseen distributions. The model employs a lightweight graph neural network with shared and task-specific layers and a supervised diffusion process to efficiently learn instance-wise solution distributions. The authors also introduce a new benchmark of 38 datasets for evaluating generalization across problem types and distributions. Experiments demonstrate that M2GenCO achieves state-of-the-art performance while reducing training cost.

### Strengths
- The paper formulates meta-learning tasks across different combinatorial problem types (rather than instances of one), aiming to bridge diffusion-based generative solvers with multi-task meta-learning for broad generalization.
- The authors construct a large-scale benchmark of 5 COP types and 38 datasets, enabling systematic assessment of out-of-distribution and cross-problem adaptability. The dataset can be of good value to the community.
- The authors reported state-of-the-art results on all tested benchmarks.

### Weaknesses
- The paper can be greatly improved in its structure and presentation. In its current state, it is very difficult to follow and understand even for expert readers.
- The paper claimed as the first to (1) define meta-learning tasks across different COP types and (2) integrate a supervised diffusion backbone into a multi-task meta-solver, I feel this statement may be an over-claim, for example, unified frameworks e.g., GOAL, UniCO, MVMoE/MAB-MTL and diffusion-based CO solvers, are all relevant prior works, without clear differentiation, this claim is not grounded, and the contribution and novelty isn't justified. I suggest the authors to rephrase the claim and clearly articulate the novelty and contributions of their work in the context of existing approaches.
- The proposed benchmark, though diverse in distributions, primarily uses synthetic datasets rather than real-world industrial or logistics data. Also although five classic COPs are covered, all are graph-based problems; extending to non-graph combinatorial domains may be needed.
- The experiment section focuses heavily on quantitative metrics but provides little insight into what is the source of success

### Questions
- Would the code and dataset be released for public access
- What're the sources of the gains? Would recommend the ablation study to include carefully designed experiments to provide understanding on this so future work can be built on top of such knowledge

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes M2GenCO, a multi-task meta-learning paradigm that treats different COP types as meta-tasks. It couples a supervised diffusion (consistency) backbone and a lightweight GCN with hybrid task-specific / shared blocks to enable fast few-shot finetuning across graph-based COPs. The proposed method achieves strong empirical performance.

### Strengths
S1 (strong performance): M2GenCO reports consistent gains across many OOD benchmarks and an overall improvement index while achieving dramatic inference speedups compared to previous generative solvers.

S2 (novel framework): Treating problem types as the meta-task distribution and combining that with a diffusion/consistency generative backbone seems to be a new and well-motivated formulation that bridges CO and meta-learning.

S3 (efficiency): The lightweight GCN design is suitable for multi-task sharing and enables the enables efficient adaptation compared to larger backbones.

S4 (thorough analysis): The paper reports thorough cross-distribution, cross-scale and ablation studies.

### Weaknesses
W1 (finetuning protocol): The proposed adaptation relies on offline few-shot finetuning using a support set. It is not fully clear how sensitive the method is to to distributional mismatch between support and query or to scenarios where no labeled support set is available.

W2 (design choices): Many design choices such as the task-sequence length k, the gradient normalization in Eq. (3), and sensitivity to inner/outer learning rates lack deeper ablation studies or theoretical justifications.

W3 (scope of tasks): The suite covers only 5 graph COPs, but the claim toward “general-purpose” neural solvers would be stronger if the method were tested for non-graph COPs (knapsack, integer programs, etc.) or if limitations were emphasized more in the limitations section.

### Questions
See weaknesses.

### Soundness
3

### Presentation
4

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
This paper introduces M2GenCO, a multi-task meta-learning framework for generative combinatorial optimization on graphs. The key innovation is reformulating meta-learning tasks as different problem types (TSP, ATSP, MIS, MCl, MCut) rather than instances of the same problem, integrating this with diffusion-based generative modeling. The framework consists of multi-task meta-pretraining across problem types, followed by few-shot finetuning for specific distributions. Additionally, the authors construct a comprehensive benchmark with 38 datasets spanning 5 COPs across diverse distributions. Experiments demonstrate approximately 50% improvement in solution quality and up to 4× faster inference compared to prior SOTA, while achieving 82% reduction in computational resources and 91% reduction in training time.

### Strengths
--Ambitious vision: The paper tackles a challenging problem - learning across structurally different COPs - which pushes the boundaries of current NCO methods


--Comprehensive evaluation: The experimental evaluation is impressively thorough, covering 38 datasets with multiple baselines and extensive ablations


--Valuable benchmark: The multi-distribution benchmark addresses a critical need in the NCO community for standardized OOD evaluation

### Weaknesses
-- Theoretical foundation needs development: While the empirical results are encouraging, the paper would benefit from deeper analysis of why cross-problem learning helps. Even an intuitive explanation based on shared graph structures or optimization patterns would strengthen the work.


-- Task pool selection could be more systematic: The current selection appears somewhat arbitrary. An ablation study varying task combinations would provide valuable insights into which problems benefit from joint training.


-- Integration could be tighter: The diffusion and meta-learning components, while both valuable, feel somewhat independent. Exploring their synergies more explicitly would strengthen the narrative.


-- Baseline comparisons need clarification:


  * It would be helpful to clarify whether Meta-EGN's sampling strategy difference affects fairness
  * Comparing against standard multi-task learning (without meta-learning) would isolate the meta-learning benefit
  * Training data amounts should be standardized where possible

--Scalability discussion: The paper would benefit from more explicit discussion of scaling limits and potential solutions for larger instances (>1000 nodes).

### Questions
-- Shared structure hypothesis: Could you elaborate on what common structures or patterns across COPs might enable positive transfer? For instance, do all problems benefit from learning graph connectivity patterns?


-- Task pool sensitivity: How robust is performance to different task combinations in pretraining? Does removing any single task significantly impact results?


-- Ablation depth: Could you provide results for standard multi-task learning without the meta-learning machinery to isolate its contribution?


-- Component necessity: Given Table 15's results, have you explored whether simpler architectures might achieve similar performance with less complexity?

### Soundness
3

### Presentation
2

### Contribution
2
