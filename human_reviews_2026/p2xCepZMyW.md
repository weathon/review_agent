# Configuring Parallel Training of Neural Networks using Bayesian Optimization

- Decision: Reject
- Scores: 4, 4, 6

## Abstract
Training of modern large neural networks (NNs) is often done in parallel across multiple GPUs. While there are existing parallel training frameworks which easily allow NN training using multi-dimensional parallelism, the challenge remains in optimizing the balance between size of each parallelism dimensions, and in tuning the hyperparameters within these parallelism dimensions. Due to a large number of possible parallelism configurations (PCs) for a given training scenario, it is infeasible to perform exhaustive search over all candidates. Existing PC optimization methods typically either require conducting training trials on a large number of PCs, each of which can be expensive to perform, or rely on an approximate cost model which may be inaccurate and hardware-specific. To overcome these issues, we present OPPA, which combines constrained Bayesian optimization methods with prior knowledge in the form of a parallelism-informed prior belief, to obtain an optimal PC using a minimal number of NN training trials. We also propose a framework for early termination of trails involving suboptimal PCs, whose efficiency gains can be theoretically justified. We show that OPPA finds an optimal PC more efficiently for training transformers on various multi-GPU systems compared to the methods used in existing parallel training frameworks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work introduces a Baysian optimization method to find suitable hardware parallelization configurations using a minimal amount of execution time. Notably, the surrogate Matern Gaussian Process can integrate prior guesses about the throughput and memory use that are refined during the subsequent optimization. The optimizer is experimentally evaluated on Transfomer architectures for multi-GPU settings, demonstrating reduced optimization time compared to simpler baselines.

### Strengths
- The work tackles an important practical problem is distributed training with a focus on overall optimization time
- The ability to integrate prior domain knowledge is likely beneficial in this context as it is plausible that good guesses about memory use and execution time are available for a wide variety of hardware and models.
- The manuscript includes an introductary overview of techniques and related prior work, making it accessible and self-contained

### Weaknesses
- The novelty of the method is limited as the employed techniques for Bayesian surrogate optimization are well established
- The memory component of the optimization problem is modelled as part of the constraint and is thus only implicitly optimized for. It seems more suitable to formulate the problem as an multi-objective optimization problem to allow the optimizer to explicitly optimize the memory. 
- The idea of early termination for likely suboptimal configuration warrants further investigation since a search sample's usefulness is not necessarily determined by its closeness to an 'optimal' value. Since the suboptimal value was suggested by the surrogate model it represents a point in the search space that the surrogate model predicts to be better than it actually is. This means that this suboptimal sample will likely be very useful for improving the surrogate model. Of course, there is a practical trade-off between taking time to generate a sub-optimal sample and taking time to generate many samples but it seems that the early termination throws away information that could be useful to improve the surrogate model (and thus speed up overall convergence). The ablation study in Figure 7 suggests that both Early Termination and Prior Belief are required so it could be that Early Termination is helpful if the Prior Belief already insures that the surrogate model has a good idea of the promising regions in the search space. The presented experiments do not seem to delineate between 'time saved through convergence speed' and 'time saved through termination'.
- The search method itself seems to rely on hyperparameters (e.g. IQR range for outlier detection) but it unclear how they are chosen in practice

### Questions
- L237 assumes "All-Reduce operations involved in DP and TP, and point-to-point communications involved in PP. We consider a canonical ring/tree scaling with hierarchical aggregation such that intra- and interhost connections are separately modeled" -- do these assumptions matter for the ability to provide domain knowledge priors or do you expect OpPa to generalize to other settings?
- L294 "we assume the additional contribution is expected to have no additional bias on the estimates of the quantities" - Can you please elaborate on why this is justified.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes OPPA, a Bayesian-optimization–based tuner for multi-dimensional parallel training configurations. OPPA combines a parallelism-informed prior, a GP residual to capture unmodeled effects, a constrained UCB to penalizes OOM, and a early-termination rule to stop clearly suboptimal trials. Experiments on BERT/Qwen/LLaMA across single- and multi-host clusters show higher achieved throughput over baselines.

### Strengths
- **Practical problem and clear formulation.** Treating parallelism configuration (PC) search as constrained black-box optimization with real hardware feedback is well motivated and useful for practitioners.
- **Principled surrogate design.** The additive “prior + GP residual” captures domain knowledge while remaining adaptable; the GP uncertainty naturally supports exploration.
- **Evidence across setups.** Results on 8/16/32-GPU settings (single/multi-host) and different transformer sizes show consistent throughput gains vs baselines; ablations isolate the value of the prior and early termination.

### Weaknesses
- **Incremental over prior autotuners.** Combining a hand-crafted prior with BO and early stopping is reasonable, but the conceptual advance over existing autotuners (e.g., FlexFlow/Alpa cost-model/simulator planners) appears modest. The paper should articulate why **BO is uniquely appropriate** for PC search—beyond empirical wins.
- **MoE/expert parallel not evaluated.** Given current practice, omitting expert-parallel (EP) MoE is a major gap. The prior does not model all-to-all traffic, routing imbalance, or token-dependent heteroscedasticity typical in MoE; early termination could be fragile under bursty load imbalance.
- **Early-stop sensitivity.** The theorem guarantees existence of ${βᵢ, τ_q}$, but practical tuning of $q_{min}, q_{max}, τ_q, βᵢ$ is nontrivial. Warm-up effects (graph capture, kernel autotuning, cache fills) can make early steps unrepresentative, risking premature termination and biased selection. The paper does not quantify false-negative stop rates.

### Questions
- The method introduces many hyperparameters. How are they tuned, and how sensitive are results to each? I recommend the authors to include a table listing all hyperparameters with definitions, default values, and search ranges.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper considers the design of parallel configurations for training neural nets with many GPUs.
It uses Bayesian optimization to maximize training throughput (training examples processed per unit time) subject to a constraint on GPU memory.   A UCB-style regret guarantee is given and experimental results are presented.

### Strengths
My background is in Bayesian optimization but not in AutoML / neural architecture search.

(Weakness) From a BO perspective, this paper is a straightforward application paper without a great deal of novelty.  A constrained BO problem is proposed --- maximize training throughput subject to a constraint on GPU memory use.  Informative priors are proposed that consist of carefully-crafted application-specific mean functions plus a mean-zero Gaussian process (this is sometimes called "universal kriging" in the old literature).  A UCB acquisition function is proposed and a regret guarantee is provided.  The regret guarantee appears to be fairly standard though the authors should inform me if there is something novel in the proof.  So from a BO perspective, the paper does not rise above the publication bar for ICLR.

(Strength) But from an AutoML perspective, the contribution seems valuable.  The method seems to outperform several benchmarks that seem reasonable to me by a practically meaningful margin in an evaluation that seems realistic.  I would be interested to hear from someone who has more experience with problems in this space who can comment on the selection of benchmark methods and the evaluation methodology.

### Weaknesses
See the text above under Strengths

### Questions
When evaluating parallel configurations, training instances are processed.  Are the results used or thrown away?  

The paper is written as if we have some budget for choosing the parallel configuration, after which we get some reward. But it seems that this can be formulated as an online problem --- my goal is to process all of the training data as quickly as possible and I can switch parallel configurations during training.  Presumably there is some large overhead for switching parallel  configurations, so from a practical perspective any good algorithm will eventually pick one configuration and finish a large fraction of the training with this. Yet, it doesn't seem that the budget before you settle on one must be fixed. In this sense, the model in the paper seems a bit like it could be improved.

 Also, if the cost for switching configurations is large, then we this should be reflected as an additional cost during optimization of the parallel configuration.

### Soundness
4

### Presentation
3

### Contribution
2
