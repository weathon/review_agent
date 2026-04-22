# Multi-Policy Pareto Front Tracking Based Multi-Objective Reinforcement Learning

- Avg Score: 3.00
- Decision: Reject
- Scores: 2, 2, 4, 4

## Abstract
Multi-objective reinforcement learning (MORL) plays a pivotal role in addressing multi-criteria decision-making problems in the real world. The multi-policy
(MP)-based approaches are widely used to obtain high-quality Pareto front approximations for the MORL problems. Relying primarily on the online reinforcement learning (RL), the traditional MP approaches usually adopt the evolutionary
framework that requires maintaining a large policy population. In practice, however, this often leads to sample inefficiency and/or excessive agent-environment
interactions. To address these issues, we propose the novel Multi-policy Pareto
Front Tracking (MPFT) framework that eliminates the need to maintain any policy population, compatible with both online and offline MORL algorithms. The
proposed MPFT framework comprises four stages: Stage 1 approximates all the
Pareto-vertex policies whose mappings to the objective space lie on the vertices
of the Pareto front; Stage 2 proposes a new Pareto tracking mechanism that starts
from each Pareto-vertex policy to track the Pareto front, where a proof of its exponential convergence is provided; Stage 3 identifies the sparse regions in the tracked
Pareto front, and then newly designs an objective weight adjustment method to facilitate the policy tracking for filling these regions; Finally, by combining all the
policies tracked in Stages 2 and 3, Stage 4 approximates the complete Pareto front.
Experiments are conducted on seven continuous-action robotic control tasks using
both online and offline MORL algorithms. Results demonstrate that our proposed
MPFT approach outperforms state-of-the-art benchmarks in terms of hypervolume and expected utility performances, while significantly reducing the agentenvironment interactions.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work presents a Multi-Policy Pareto Front Tracking (MPFT) framework intended to be compatible with existing reinforcement learning (RL) algorithms. The approach identifies “Pareto vertex” policies, then tracks additional policies to form a “Pareto-edge” policy set, fills in sparse regions, and constructs a dense Pareto policy set.

### Strengths
1. The framework presents a reasonable design for two-objective cases. It first identifies the extreme “vertex” policies, then tracks intermediate “edge” solutions, and finally considers sparse regions, which is a logical progression for covering the Pareto front.
2. The empirical results are better than baselines like PGMORL and PA2D-MORL.

### Weaknesses
1. The framework does not appear to extend naturally to problems with more than three objectives, where additional structures such as “facets” (beyond edges) would need to be tracked.
2. The claimed advantage over evolutionary methods such as PGMORL is limited and not clearly articulated. The main difference seems to lie in the training curriculum rather than in a fundamentally new mechanism. Moreover, Alegre et al. (2022) also presents a non-evolutionary curriculum but suffers from exponential cost growth with the number of objectives. It is unclear how the proposed MPFT framework avoids a similar issue. Although MPFT claims linear time complexity, but it is for parallel training, which could be misleading because the total training cost still grows exponentially.
3. The paper lacks sufficient discussion and comparison with similarity-based methods such as PD-MORL [1]. The proposed approach for filling sparse regions could also be achieved through similarity constraints, but this is overlooked.
4. The use of the term “vertex” for single-objective extreme solutions is not rigorous. In discrete spaces, even for a 2D Pareto front, multiple vertices may exist between the two extremes, making this terminology and formulation imprecise.


[1] PD-MORL: Preference-Driven Multi-Objective Reinforcement Learning Algorithm
Toygun Basaklar, Suat Gumussoy, Umit Y. Ogras

### Questions
1. How does the method scale to more than 3 objectives where “facets” (beyond edges) would need to be tracked?
2. Is there comparison against Alegre et al. (2022)？
3. How many policy models need to be maintained after training?

### Soundness
2

### Presentation
3

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
This paper proposes MPFT, a multi-policy multi-objective reinforcement learning (MORL) method for approximating the Pareto front in multi-criteria decision-making problems. The framework consists of four stages, including a Pareto-tracking mechanism that locally follows the Pareto front and a subsequent stage that detects and fills sparse regions of the front. The method is evaluated on MuJoCo environments with two and three objectives. However, the work lacks comparison with closely related approaches, and several methodological and experimental details remain unclear. The evaluation is limited to low-dimensional settings (mostly two-objective tasks and only one three-objective case), making it uncertain whether the approach can effectively scale to higher-dimensional or more complex environments.

### Strengths
1. The paper addresses an important problem of approximating the Pareto front in multi-objective reinforcement learning.
2. The proposed framework is conceptually clear and provides a theoretical stability analysis (Appendix C) for the Pareto-tracking mechanism.
3. The experimental setup includes both on-policy and off-policy settings, showing that the framework can be applied under different training paradigms.

### Weaknesses
1. Misleading claim about population-free training.
The claim that the method “eliminates the need for maintaining any population” is somewhat misleading. In practice, both Stage 2 (Pareto tracking) and Stage 3 (Pareto filling) continuously add newly generated policies and remove dominated ones, thereby maintaining an evolving set of non-dominated policies. The difference lies mainly in how this population is updated—sequentially rather than in parallel. Moreover, Stage 3 involves multiple rounds of objective-weight adjustment, which essentially follows an evolutionary process.
2. Limited theoretical guarantee of the tracking mechanism.
Appendix C only proves that the proposed tracking dynamics keep policies within a neighborhood of the Pareto-stationary set, ensuring local stability but not controlled tracking the entire Pareto front. The mechanism cannot determine the precise direction of movement on the Pareto front, particularly when the number of objectives exceeds two, and therefore cannot guarantee full coverage of the front. This limitation should be clearly stated when describing the claimed “exponential convergence.”
3. Unclear theoretical basis of the Stage 3 weight-adjustment method.
The appendix B.2 states that previous weight-adjustment/selection techniques are heuristic, yet the proposed approach is not theoretically analyzed either. The absence of theoretical justification makes it difficult to assess how the new adjustment differs from existing heuristic methods.
4. Strong similarity to C-MORL [1] without comparison.
The proposed method is highly similar to C-MORL, yet the authors provide no discussion or experimental comparison with it. Both frameworks initialize a few extreme policies, locally improve other objectives while maintaining one (Stage 2 tracking), and fill sparse regions of the Pareto front (Stage 3 filling). Appendix B.2 even suggests using PCA-based reduction or the crowding-distance rule from C-MORL when the number of objectives m>3, but no experiments are shown in such settings. Since C-MORL already demonstrated scalability up to nine objectives while this work only evaluates two- and three-objective tasks, the claimed novelty of the “Pareto-tracking mechanism” appears incremental, and the absence of a C-MORL baseline makes the claimed advantages unclear.
5. Unspecified size of the maintained Pareto set.
The number of non-dominated policies maintained (often called the policy-buffer size in multi-policy MORL) is not reported. This number directly affects both hypervolume and computational cost, including the claimed CPU-efficiency improvement. Without controlling or reporting this factor, the computational comparison is incomplete.
6. Limited experimental scope and missing ablations.
Experiments are restricted to Mujoco tasks with two or three objectives, without tests on higher-dimensional or discrete environments. Limited ablation is provided for key hyperparameters, such as the number of training episodes in Stage 2 (tracking) and Stage 3 (filling), which are likely to affect coverage and performance.

### Questions
1. Regarding the “population-free” claim:

(a) Could the authors clarify in what sense the method “eliminates the need for maintaining any population,” given that Stages 2 and 3 still add and remove policies within an evolving archive?

(b) In Stage 3, multiple episodes of objective-weight adjustment are used—does this process not constitute an evolutionary search?

2. Regarding the theoretical guarantee (Appendix C):

(a) The proof shows convergence to a neighborhood of the Pareto-stationary set, but not guaranteed traversal or full coverage of the Pareto front. Could the authors clarify the intended scope of the “exponential convergence” statement in the main text?
(b) How does the proposed mechanism behave when the number of objectives exceeds two, where the direction along the front is not uniquely defined?

3. Regarding the weight-adjustment method in Stage 3:

(a) The appendix suggests prior weight-adjustment methods as heuristic; is the proposed one also heuristic, or is there any theoretical justification for its design?

(b) How sensitive is the method to the number of weight-adjustment episodes or the choice of adjustment method?

4. Regarding comparison with C-MORL:

(a) Why is C-MORL not included as a baseline, given the clear methodological similarity and its demonstrated scalability up to nine objectives?

(b) If the proposed method reuses the crowding-distance rule from C-MORL when m>3, what specific difference or advantage remains?

(c) Would C-MORL, with only two initial extreme policies already, reproduce the same Stage 1–3 behaviour?

5. Regarding computational efficiency and policy-set size:

(a) What is the number of policies maintained in the Pareto set during training in all multi-policy baselines and MPFT? Is this number fixed or dynamically adjusted?

(b) Since this directly affects both CPU usage and hypervolume, can the authors provide a fair comparison under equal policy-set sizes?

(c) The claimed CPU-efficiency advantage may depend heavily on the chosen level of parallelization. In many multi-policy MORL baselines (e.g., PG-MORL or C-MORL), the number of policies trained or stored in parallel, for example, during the warm-up or stage, can be flexibly reduced or run sequentially to lower computational cost. Was the level of parallelization (i.e., the number of simultaneously trained or stored policies) matched across methods? 

6. Regarding experimental scope and ablations:

(a) Why are experiments limited to Mujoco tasks with two or three objectives?

(b) Can authors also evaluate the expected utility (EU) metric?

(c) Can the method scale to other domains in mo-gymnasium [2]?

(d) Could the authors provide ablation studies on key hyperparameters, such as the number of episodes or update ratios in Stages 2 and 3?

7. The claimed benefit of reducing CPU usage seems overemphasized, while the more meaningful contribution is actually adapting the multi-policy Pareto-tracking framework to offline settings. The paper would be stronger and distinguished from previous multi-policy methods if it focused on analyzing and validating the offline performance and stability, rather than framing CPU efficiency as the main advantage.

[1] Ruohong Liu, Yuxin Pan, Linjie Xu, Lei Song, Pengcheng You, Yize Chen, and Jiang Bian. Efficient discovery of Pareto front for multi-objective reinforcement learning. In The Thirteenth International Conference on Learning Representations, 2025b.

[2] Felten, Florian, et al. "A toolkit for reliable benchmarking and research in multi-objective reinforcement learning." Advances in Neural Information Processing Systems 36 (2023): 23671-23700.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a decomposition-based Multi-Policy Pareto Front Tracking (MPFT) framework for MORL. MPFT introduces a Pareto-tracking mechanism which alternates between Pareto-ascent and Pareto-reverse updates to progressively approximate the Pareto front, and an objective weight adjustment procedure to identify and fill sparse regions. Experimental results are reported for seven continuous-action problems, but discrete problems or discrete Pareto fronts were not considered.

### Strengths
The framework is clearly structured and easy to follow.

Introduces a tracking mechanism combining Pareto-ascent and Pareto-reverse updates. Theoretical analysis, including the exponential convergence proof, adds rigor to the method. 

The framework’s compatibility with both online and offline RL algorithms which may increase applicability.

The experiments show that the proposed framework can be combined with multiple RL algorithms, suggesting it could serve as a flexible plug-in mechanism for various MORL settings.

### Weaknesses
The paper claims to focus on efficiency for real-world applications, but the experiments still use tens of millions of environment steps on standard MuJoCo tasks. This raises concerns about the claimed efficiency. How can the authors justify this efficiency claim or demonstrate effectiveness under shorter training budgets? How would the framework perform when the training steps are significantly reduced?  

Although the theoretical time complexity is linear in the number of objectives, the Pareto-tracking process itself (alternating ascent and reverse updates) seems to naturally require many update cycles to tracking the front. In particular, this does not take into account the potentially complex topology of Pareto fronts of several objectives, so that it is practically risky to rely on the individual objectives and leave the (exponential) complexity of the Pareto front unexplored.

The proposed framework seems closely related to previous Pareto-ascent–based work: both use Pareto-ascent principles, and MPFT extends them with a reverse-tracking component. Explaining on the connection, difference and conceptual novelty more clearly between those two methods would help appreciate the distinct contribution. 

If a Done_t variable is used, why also a maximal time T is needed in theory (i.e. why not T=infinity)? Or what would happen if the job isn't finished before time T? To me it seems these are aspects of the particular implementation, so, if at all, it can be discussed in the appendix (where actually T=infinity is used!), while here a less redundant formulation could be preferable.

Definitions 1 and 2, are not suitable to deal with the case of non-connected Pareto fronts (whether this is because not solutions exists in a certain region of the objective space or whether the best solutions in a certain direction are Pareto-suboptimal.

256: There is no need to use metaphorical expression like '“continuous”', instead an suitable (topological) definition of continuity should be used. Also, in a continuous problem the idea to "differ by only one state-action pair" is not well-defined (at least not in measure-theoretic sense), so that a better formulation is needed here anyway.

288: Use an algorithm environment to represent this part.

The experimental validation is limited. The experimental comparison mainly includes two external baselines, while the other algorithms are variations within the proposed framework.

While this setup helps illustrate MPFT’s internal consistency, the empirical advantage would be clearer if the authors could discuss how MPFT relates to more recent GPI-based, preference-conditioned, or other decomposition-based MORL methods.

### Questions
How does the Pareto-tracking (alternating ascent–reverse update) process influence computational cost in practice? 

How would it work with problem that has a non-connected or discrete Pareto front? 

Is MPFT robust when the training steps or tracking iterations are reduced?

From a conceptual perspective, what is the main new idea behind MPFT compared with prior Pareto-ascent–based framework?

How does MPFT compare with more recent MORL paradigms such as GPI-based, preference-conditioned, or other decomposition-based approaches in terms of sample efficiency and Pareto-front coverage?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a multi-stage MORL algorithm. Stage1 approximates the Pareto-vertex policies by solving a single objective optimization problem for each objective; Stage2 initializes thePareto-approximation policy set by parallelly tracking the Pareto front from each approximate Pareto-vertex policy; Stage3 fills the top-K sparse regions; Stage4 completes the Pareto-approximation policy set, by combining the tracked policies from Stages2 and 3.

### Strengths
This paper introduces a Multi-Policy Pareto Front Optimization by jointly optimizing multiple policies to cover different regions of the Pareto front. It uses a diversity-promoting objective to ensure that policies specialize toward distinct trade-offs. A mutual-learning mechanism is also adopted where policies share gradient information to stabilize learning and avoid redundant convergence.

The paper precisely identifies the limitations of existing methods. Preference-conditioned single-policy models often fail to generalize smoothly across the entire preference simplex. Independent policy training leads to poor sample efficiency and redundant exploration.

### Weaknesses
The step of "Approximate all Pareto-vertex policies" may take a lot of computation, and it is practically not very efficient to track all their performances.

The sparse regions are not well defined. Are they mathematical sets?

The paper could benefit from deeper discussion linking its multi-policy optimization to areas such as policy ensembles in meta-RL, and mixture-of-experts or multi-head architectures.

### Questions
What is the computation time for each stage? The paper shall clearly clarify and discuss them.

### Soundness
2

### Presentation
2

### Contribution
2
