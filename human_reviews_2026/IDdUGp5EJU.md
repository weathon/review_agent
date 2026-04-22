# SkillWrapper: Generative Predicate Invention for Skill Abstraction

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 2, 6, 6

## Abstract
Generalizing from individual skill executions to solving long-horizon tasks remains a core challenge in building autonomous agents. A promising direction is learning high-level, symbolic abstractions of the low-level skills of the agents, enabling reasoning and planning independent of the low-level state space. Among possible high-level representations, object-centric skill abstraction with symbolic predicates has been proven to be efficient because of its compatibility with domain-independent planners. Recent advances in foundation models have made it possible to generate symbolic predicates that operate on raw sensory inputs—a process we call generative predicate invention—to facilitate downstream abstraction learning. However, it remains unclear which formal properties the learned representations must satisfy, and how they can be learned to guarantee these properties. In this paper, we address both questions by presenting a formal theory of generative predicate invention for skill abstraction, resulting in symbolic operators that can be used for provably sound and complete planning. Within this framework, we propose SkillWrapper, a method that leverages foundation models to actively collect robot data and learn human-interpretable, plannable representations of black-box skills, using only RGB image observations. Our extensive empirical evaluation in simulation and on real robots shows that SkillWrapper learns abstract representations that enable solving unseen, long-horizon tasks in the real world with black-box skills.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper tackles the problem of learning plannable, high-level models of black-box skills from raw RGB observations by inventing predicates and composing them into PDDL operators that can be used with off-the-shelf planners. The core idea is a formal framework for generative predicate invention with explicit target properties (soundness, completeness, “suitability”), and an algorithm, SKILLWRAPPER, that alternates between (i) actively collecting data by proposing skill sequences, (ii) inventing predicates when current abstractions cannot explain observed success/failure or effects, and (iii) learning operators by clustering transitions on lifted effects and intersecting preconditions. The system uses a foundation model both to generate predicate candidates and to classify their truth values from images; only RGB images are assumed. Theoretical results show operators are supported by data (soundness) and that the learned model is probabilistically complete under finite-hypothesis assumptions. Empirically, the method is evaluated in Robotouille (a simulated kitchen domain) and on two real setups (Franka Panda; bimanual Kuka). In simulation, SKILLWRAPPER achieves 73.3% solved on “Easy” with PB=2.9, 38.3% on “Hard” with PB=6.1, and 100% correct detection of impossible tasks, outperforming ViLa and random exploration and competitive with expert operators on some splits. On real robots, it generalizes predicates learned in restricted settings to a larger test setting (e.g., 60.0% solved in generalization split; PB=4.0). Iterative runs show performance improves as more predicates are invented and data collected.

### Strengths
Originality.

 • Provides a formal target for predicate invention in the context of skill abstraction, not merely state abstraction, and uses it to drive algorithmic design (two concrete invention triggers based on executability and effect inconsistencies).
 • Uses a foundation model as a relational classifier (truth assignment for grounded predicates) and not just a planner, enabling learning from RGB images without pre-defined state factors.

Quality (theory).

 • Clear statements of soundness (operators backed by observed transitions) and probabilistic completeness w.r.t. a distribution over transitions, with proofs/sketches and finite-hypothesis assumptions.

Quality (empirics).

 • Evaluations span simulation and two real robot platforms; the setup probes both generalization across environments (Franka) and iterative improvement under irreversible actions (bimanual Kuka). The Robotouille benchmark comparison includes expert, system predicates, ViLa, and random exploration. Metrics include solved rate and a planning budget proxy (PB) tied to completeness.

Clarity.

 • The paper is organized around a tight loop: data → predicates → operators, with pseudocode, invention conditions, and scoring functions, plus prompt details in the appendix. Examples of learned predicates/operators (with natural-language semantics) help assess interpretability.

Significance.

 • Demonstrates that language/VLM priors can yield plannable, interpretable abstractions from raw images, and that these abstractions can scale to longer-horizon problems than open-loop LLM planning—an important step for integrating FM-driven perception with symbolic planning in robotics.

### Weaknesses
Theoretical scope and assumptions.
 • The soundness guarantee is empirical (“supported by at least one observed transition”) and does not formalize robustness to classification noise from the VLM. In practice, VLM truth assignments will be imperfect; the theory currently does not propagate uncertainty through operator learning or planning, nor does it bound error from misclassifications.
 • The probabilistic completeness bound relies on a finite hypothesis class H and i.i.d. sampling; it is unclear how H is instantiated in practice when predicate proposals are open-ended (FM-generated) and when pruning/reevaluation can expand or contract the model class over time. The bound risks being vacuous without a concrete characterization of |H| or sample complexity as a function of invented predicates. 
 • The method assumes deterministic skills and that skills affect only the bound objects. Many real skills are stochastic and produce side effects; the invention conditions and operator learning rules may need adaptation to handle such cases. 

Algorithmic design choices.
 • Operator learning computes preconditions via intersection of initial abstract states within a cluster; this is conservative and can produce spurious preconditions when data are sparse—acknowledged by the authors—but the paper offers limited guidance on when predicate re-evaluation suffices versus when additional data are required. A quantitative analysis of false-positive preconditions over iterations would strengthen the claim.
 • Predicate selection hinges on thresholds in score functions (Algorithm 6). The paper does not study sensitivity to the threshold h, nor how choices trade off compactness vs. coverage (e.g., learned operator sparsity, PB, solved rate). 
 • The active data collection relies on LLM-proposed sequences and heuristic scores (coverage/chainability). While well-motivated, there is no ablation isolating the gain from these heuristics vs. random or simpler curricula.

Empirical evaluation.
 • Baselines: “System Predicates” lacks invention and has privileged state access; by construction it will underperform on tasks requiring new predicates, making it a weak comparator. Missing are stronger learned-predicate/action-model baselines (e.g., neurosymbolic predicate learners or prior predicate-invention techniques) to more precisely attribute gains to the proposed invention logic.
 • Scale and variance: Many results are averaged over three runs; the real-robot evaluation uses small problem sets, limiting statistical confidence. Reporting confidence intervals and significance tests would help.
 • VLM-as-classifier reliability: Since abstract states are inferred directly from a VLM on RGB images, experiments should report truth-assignment accuracy against labeled ground truth (even on a subset) and robustness to viewpoint changes/occlusion; currently, the paper assumes full observability from images.

### Questions
Questions
1. Hypothesis class & bounds. How do you instantiate the finite hypothesis class H used in Theorem 2 when predicates are FM-generated and can be reevaluated/removed? Can you provide a practical upper bound on |H| (e.g., as a function of max invented predicates and arity) and a sample-complexity estimate in transitions to achieve a target \epsilon?
2. Noise-aware guarantees. Do you foresee modifying the framework to incorporate noisy predicate truth values (e.g., via probabilistic predicates or confidence-weighted effects), and can the soundness/completeness results be extended to this setting? Empirically, what is the observed misclassification rate of the VLM over your predicates?
3. Ablations. Please provide ablations for (a) coverage/chainability in skill-sequence proposal, (b) predicate re-evaluation (on/off), and (c) threshold h in Algorithm 6. How do these affect solved %, PB, number of predicates, and operator sparsity?
4. Stochastic or side-effectful skills. How does the effect-inconsistency trigger behave if a skill succeeds but produces variable effects (or mid-execution failure) due to stochasticity? Do you anticipate inventing context predicates vs. effect predicates, and how are these disambiguated empirically?
5. Portability across embodiments. In the real-robot experiments, to what extent were the same predicates reused across Panda and Kuka (vs. reinvented)? Could you report a transfer study where operators learned on one platform are applied to the other with minimal additional data?
6. VLM classifier robustness. Have you measured sensitivity to camera viewpoint/lighting/occlusion or to minor domain shift (e.g., new mugs/utensils not seen during learning)? Even a small held-out labeled set would be informative.
7. Metrics. Beyond solved% and PB, could you report plan optimality gaps, planning time, interaction budget, and predicate/operator counts over iterations (with variance), to better illuminate data efficiency and model compactness?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors propose a formal theory to characterize the necessary conditions for generative predicate invention, and present the SKILLWRAPPER framework, which leverages the capability of LLMs to perform operator learning and skill abstraction. Both simulation and real-world robot experiments are conducted to validate the framework.

### Strengths
1.The paper provides a theoretical proof of completeness for skill learning. Although I did not go through the detailed derivation, I believe this is likely one of the main innovations of the paper.

2.The paper includes real-robot experiments, which convincingly demonstrate that the SKILLWRAPPER framework can be effectively deployed in real-world settings.

### Weaknesses
1. In Table 1, the experimental results of SKILLWRAPPER do not consistently outperform expert-designed predicates and operators. This raises concerns about whether the proposed framework offers a real advantage over manually defined predicates and operators.
2. The experimental section lacks sufficient task descriptions. It is difficult to understand how task difficulty is defined or differentiated. If such details exist, please indicate where they are presented.
3. The authors only evaluate on the Robotouille simulated task. Why not conduct experiments on more well-known benchmarks such as IsaacGym (https://github.com/isaac-sim/IsaacGymEnvs), MetaWorld(https://github.com/Farama-Foundation/Metaworld)? Please explain the rationale behind this choice.
4. The paper’s main contributions are not clearly highlighted. Both skill abstraction and predicate generation using LLMs are not entirely new techniques. The paper lacks comparisons with these baselines, and the overall presentation fails to make the core novelty of the work clear.

### Questions
1. Why does ViLA not include the Planning Budget metric? Please provide a rough explanation.
2. How are the conclusions derived from the formal theory used to guide the design of the SKILLWRAPPER framework?
3. Has any ablation study been conducted—for example, how would removing active data collection affect the skill abstraction process? What would happen if expert-designed operators and SKILLWRAPPER-learned skills were combined?
4. Other questions are mentioned in the Weaknesses section.

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
This paper discusses how to learn predicates and operators for given skills. It builds a system like VisualPredicctor that asks LLMs to propose and ground predicates and uses symbolic methods to evaluate and manage the proposed predicates and operators. It also asks LLMs, together with symbolic heuristics, to propose sequences of actions for exploration and data collection. The method is evaluated in several simulated and real-robot environments.

### Strengths
This paper shows that generative predicate invention works in real-robot settings. 

The paper is generally well-written and easy to understand. 

The LLM exploration heuristics are more interesting to me and deserve more space in the main paper, in my understanding. It would be great to study the exploration effectiveness given each combination of the heuristics in practice.

### Weaknesses
* The theories are either trivial or missing important strong assumptions in the main context. For example, Theorem 2 relies on i.i.d. samples which is a strong assumption (never true in online-exploration or LLM-exploration settings in practice) and is **not** stated in Theorem 2. With i.i.d. samples, Theorem 2 is true for any method (such as VisualPredicator) that satisifies $\hat Err(\hat M_n) = 0$.
* The predicate-invention method is very similar to VisualPredicator.
* Missing baseline such as VisualPredicator

### Questions
* Are there results to compare with VisualPredicator? What's the difference and why? 
* Are there results analyzing the effectiveness of various exploration strategies?

### Soundness
2

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
3

### Summary
The paper introduces SKILLWRAPPER, a framework for generative predicate invention that learns human-interpretable symbolic models of black-box skills for long-horizon planning from RGB images only. The method formalizes when and how to invent new predicates so that the resulting abstraction yields provably sound and (probabilistically) complete PDDL operators for planning. Practically, the system (i) actively gathers data by proposing skill sequences, (ii) invents predicates when failures/successes are indistinguishable under the current vocabulary, and (iii) learns operators via associative model learning with type hierarchies and periodic predicate re-evaluation (Algs. 1–2). Experiments in the Robotouille grid-world kitchen and on two real robot setups (Franka Panda; bimanual Kuka) show higher solve rates and lower planning budgets than VLM prompting and random exploration, and competitive performance against expert-authored operators; results include generalization to richer environments and handling “impossible” tasks. The novelty lies in a formal theory of predicate invention tailored to skill abstraction with guarantees, plus a concrete system that uses a foundation model both to propose predicates and to classify their truth values directly from images.

### Strengths
1. Originality: Provides a formal theory for generative predicate invention specifically for skill abstraction, with explicit conditions tied to precondition/effect indistinguishability (Sec. 3.2), addressing a gap in prior ad-hoc predicate generation.
2. Real-world evaluation: Includes two real-robot settings (Franka; bimanual Kuka), with generalization across object/skill subsets and learning curves that surpass baselines as predicates accumulate (Figs. 3–5, Table 2).
3. Proves soundness of learned operators and probabilistic completeness relative to a finite hypothesis class (Theorems 1–2), linking learning criteria directly to planning guarantees (Sec. 3.4).

### Weaknesses
1. The approach assumes accurate truth-value predictions from a foundation model (Sec. 4.1), which might not always work in real world.
2. Results are averaged over three runs, with no error bars or significance tests (Sec. 4). 
3. Real-robot sections assume deterministic skills and fully observable states (Sec. 4), which may not hold in cluttered, partially observable settings.
4. Lack of compute budgets (GPU/CPU hours), inference costs for predicate evaluations (the cost of GPT-5).

### Questions
1. What is the per-predicate truth-value accuracy of the VLM classifier, and how does plan success degrade under controlled label noise?
2. Can you report mean and std (or confidence intervals) over >= 5 seeds for Table 1 and Table 2?
3. What are the compute budgets (GPU/CPU hours)? How much do you spend on GPT-5 for evaluation and predicate generation?
4. Could you list some scenarios of failure as case studies?
5. In the real-robot setups, how often did image viewpoint changes alter predicate truth judgements? Any mitigations (multi-view, temporal smoothing)?

### Soundness
3

### Presentation
3

### Contribution
3
