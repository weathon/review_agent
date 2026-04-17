# OFMU: OPTIMIZATION-DRIVEN FRAMEWORK FOR MACHINE UNLEARNING

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
Large language models deployed in sensitive applications increasingly require the
ability to unlearn specific knowledge, such as user requests, copyrighted materi-
als, or outdated information, without retraining from scratch to ensure regulatory
compliance, user privacy, and safety. This task, known as machine unlearning,
aims to remove the influence of targeted data (forgetting) while maintaining per-
formance on the remaining data (retention). A common approach is to formu-
late this as a multi-objective problem and reduce it to a single-objective prob-
lem via scalarization, where forgetting and retention losses are combined using
a weighted sum. However, this often results in unstable training dynamics and
degraded model utility due to conflicting gradient directions. To address these
challenges, we propose OFMU, a penalty-based bi-level optimization framework
that explicitly prioritizes forgetting while preserving retention through a hierar-
chical structure. Our method enforces forgetting via an inner maximization step
that incorporates a similarity-aware penalty to decorrelate the gradients of the for-
get and retention objectives, and restores utility through an outer minimization
step. To ensure scalability, we develop a two-loop algorithm with provable conver-
gence guarantees under both convex and non-convex regimes. We further provide
a rigorous theoretical analysis of convergence rates and show that our approach
achieves better trade-offs between forgetting efficacy and model utility compared
to prior methods. Extensive experiments across vision and language benchmarks
demonstrate that OFMU consistently outperforms existing unlearning methods in
both forgetting efficacy and retained utility.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes OFMU, a bi-level optimization framework for machine unlearning in large language models. The key idea is to formulate unlearning as a hierarchical optimization problem where the inner loop maximizes forgetting (with a similarity-aware gradient decorrelation penalty) and the outer loop minimizes retain loss. The authors develop a penalty-based reformulation to avoid expensive nested optimization, provide convergence analysis for both convex and non-convex settings, and validate their approach on TOFU, WMDP (language) and CIFAR-10/100 (vision) benchmarks.

The method shows consistent but modest improvements over existing baselines like Gradient Ascent, NPO, and RMU, particularly in maintaining stability on hard-to-forget samples and achieving better balance between forgetting quality and model utility.

### Strengths
S1: Well-motivated framework

The hierarchical formulation clearly captures the asymmetry between forgetting (must succeed) and retention (restore afterward). The similarity-aware penalty that decorrelates forget and retain gradients addresses a real problem in existing scalarization approaches.

S2: Solid theoretical grounding

Lemmas 1-3 provide rigorous convergence analysis for both convex and non-convex settings. The penalty reformulation (Equation 7) is theoretically justified and avoids the computational burden of fully solving the inner problem at each iteration.

S3: Consistent empirical performance

OFMU shows stable improvements across multiple benchmarks and scenarios. I particularly like Figure 1's analysis showing reduced coupling between sample difficulty and utility loss (though this might be from appendix content—focusing on main text results). The method avoids the catastrophic utility collapse of Gradient Ascent while achieving better forgetting than utility-focused methods like RMU.

S4: Comprehensive evaluation

The paper evaluates across both language (TOFU, WMDP) and vision (CIFAR-10) tasks with multiple metrics capturing different aspects of unlearning quality.

### Weaknesses
W1: Modest and potentially insignificant improvements

Looking at Table 1, the gains are small and inconsistent. For forget05 on LLaMA-2, OFMU gets FQ=0.13 vs NPO's 0.09, but NPO actually beats OFMU on forget10 (0.42 vs 0.41). More critically, the paper provides no error bars or significance tests in main results tables. Without confidence intervals, it's impossible to judge whether these differences are meaningful or just noise.

W2: Computational cost completely unaddressed

The method requires Hessian-vector products (Equation 9) at each outer iteration, plus T inner gradient ascent steps. This is clearly more expensive than single-loop baselines, yet the paper provides zero analysis of wall-clock time, memory usage, or FLOPs. For a method claiming "scalability" (abstract), this is a major omission that undermines practical applicability.

W3: Limited scale and missing ablations

The largest model tested is LLaMA-2-7B. Given that the method involves second-order computations, scalability to 70B+ models is questionable but unexplored. Additionally, key design choices lack justification—why T inner steps specifically? How sensitive is performance to β and ρ schedules? These are mentioned but not systematically studied in the main paper.

### Questions
Q1: Statistical significance

Can you provide confidence intervals or statistical tests for the main results in Tables 1-3? Given the small effect sizes, this is essential to determine if improvements are real or within noise margins.

Q2: Computational overhead

What is the actual training time and memory consumption of OFMU compared to baselines like GA and NPO? How many additional gradient computations does the Hessian-vector product require per iteration?

Q3: Scalability validation

Have you tested on models larger than 7B parameters? If not, what are the expected computational bottlenecks that would prevent scaling to 70B+ models used in production?

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
3

### Summary
`This paper proposes a bi-level optimization formulation for LLM unlearning, where the inner level optimizes the LLM paramater to unlearn forget data while  decorrelating forget data and retain data gradient and the outer level restores the unlearned model performance on retain data. To avoid heavy bi-level optimization, this paper proposes an iternating optimization scheme on mini-batch for stable training. This paper presents theoratical analysis and also shows performance improvement on LLM unlearning and image classifier unlearning.

### Strengths
* The bi-level optimization framework is reasonable and easy to follow.
* The involvement of decorrelating penalty term is well motivated and promising.

### Weaknesses
* Potential retain data robustness issue. The similarity-aware penalty relies on the cosine similarity between $\nabla_\theta L_f$ and $\nabla_\theta L_r$, which assumes meaningful gradient directions from the retain data. What's the effect of different retain data on the unlearning training?
* Concerning performance on WMDP. Table 3 in Appendix presents the performance for WMDP. From my understanding on WMDP dataset, Bio Acc. and Cyber Acc. are measuring how the unlearned LLM performs on the desired knowledge to remove (Bio/Cyber), but Table mentions higher the better, is this really correct?

### Questions
* Is the outer level optimization always fine-tuned to divergence? Algorithm 1 suggests only 1-step for outer loop.
* How is the penalty schedule determined in this work?
* Given that the framework operates in bi-level optimization, involving runtime information can better allow users to estimate its cost.

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
This paper proposes OFMU, an LLM unlearning method based on bi-level optimization. Specifically, the authors formulate unlearning as a bi-level problem, where the lower level maximizes the forget loss and negative cosine similarity between the forget and retain gradients. The upper level minimizes the retain loss. For efficient optimization, the authors propose to convert the original problem into a single-level problem and leverage a two-loop optimization scheme to update the model in an efficient and stable way. The authors also provide theoretical analyses to show the convergence guarantee of the proposed method. Finally, experiments on both language and vision benchmarks demonstrate the effectiveness of the proposed method.

### Strengths
1. The paper formulates unlearning as a bi-level optimization problem, which addresses the issues in prior works where the forget and retain objectives are difficult to balance. The authors further propose a practical algorithm to solve the optimization problem.
2. Experiments on both language and vision tasks show that the proposed method outperforms the strong baselines.
3. The paper is well-written. The method is motivated and explained well.

### Weaknesses
1. The method introduces additional costs due to the two-loop optimization and the computation of the Hessian-vector product. How's the speed of the method compared to simple baselines such as GA or NPO?
2. The method introduces additional hyperparameters such as the penalty parameter and the regularization parameter $\beta$. How robust is the performance with respect to these hyperparameters?

### Questions
Please see weaknesses

### Soundness
3

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
This work proposes a bi-level method to tackle the unlearning problem, treating forgetting as inner maximization and retention as outer minimization. A brief convergence theory is provided. Two models and 4 datasets are tested empirically.

### Strengths
This work tackles an important problem with a relatively new angle. The motivation is clear. The experiments are extensive. I think the 2-level approach makes sense, and the theoretical analysis supports it.

### Weaknesses
Some parts of this paper is not self-consistent and makes it hard to trust.

1. RMU (Risk Minimization Unlearning) is mis-cited.
RMU an important baseline throughout this work. However, it is attributed to Bu et al in Section 7.4.11 but also attributed to Li et al in Section 5.2. Both citations are wrong. Bu's method is termed NGDiff and has nothing to do with risk. Li's paper is actually about WMDP benchmark, not an optimization method. Also in Section 5.1, WMDP is again mis-cited to Bu et al. 

In the end, what is RMU? Did you really implement NGDiff?

2. Tables are not trust-worthy.

I am highly suspicious of Table 1 to 4. First of all, the bold texts seem to suggest OFMU is the best, but this is simply misleading. In Table 5, **bold texts clearly mean the best performing method**. However, in Table 1 to 4, the authors always highlight OFMU even if it is not the best under many metrics. There are so many false emphasis so I will only point out a few: Table 1, LLaMA-3.2-1B-Instruct, FTR (RMU is better but OFMU is bolded); Table 2, Random Forgetting, RA and TA columns (FT and GA and IU are much better but OFMU is bolded). 

Secondly and more importantly, Table 1 didn't not use fair baselines for comparison. The FQ column is almost all zeros and MU column is very low except OFMU, which is highly unlikely in the normal setting. I cannot list all entries in this big table that are suspicious to me, but I require clarity before I finalize my review score. What did I miss?

3. Learning rate concerns

There is only one learning rate given in Section 7.4.12. But your 2-level algorithm needs two learning rates, inner and outer. Why not give both learning rates? Did you discuss the extra cost of tuning learning rates? If one sweeps 5 values for one learning rate, one will need 5*5=25 values to sweep for two learning rates, which can be very inefficient and impractical.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
