# Knowledgeable Language Models as Black-Box Optimizers for Personalized Medicine

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 8

## Abstract
The goal of personalized medicine is to discover a treatment regimen that optimizes a patient's clinical outcome based on their personal genetic and environmental factors. However, candidate treatments cannot be arbitrarily administered to the patient to assess their efficacy; we often instead have access to an *in silico* surrogate model that approximates the true fitness of a proposed treatment. Unfortunately, such surrogate models have been shown to fail to generalize to previously unseen patient-treatment combinations. We hypothesize that domain-specific prior knowledge—such as medical textbooks and biomedical knowledge graphs—can provide a meaningful alternative signal of the fitness of proposed treatments. To this end, we introduce **L**LM-based **E**ntropy-guided **O**ptimization with k**N**owledgeable priors (**LEON**), a mathematically principled approach to leverage large language models (LLMs) as black-box optimizers without any task-specific fine-tuning, taking advantage of their ability to contextualize unstructured domain knowledge to propose personalized treatment plans in natural language. In practice, we implement LEON via 'optimization by prompting,' which uses LLMs as stochastic engines for proposing treatment designs. Experiments on real-world optimization tasks show LEON outperforms both traditional and LLM-based methods in proposing individualized treatments for patients.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces LEON, an LLM based entropy guided optimizer for conditional black box optimization in personalized medicine. The method steers an LLM with two principled constraints, a Wasserstein bound enforced by a source critic to limit shift from the training distribution and a coarse entropy control defined over equivalence classes of candidate treatments. The algorithm alternates proposing treatments, scoring them with a learned surrogate that is regularized by the critic, and updating dual variables for the two constraints. Prior medical knowledge is injected through tool use over textbooks, knowledge graphs, and a specialist model, and the system caches reflections across iterations. Experiments on five tasks under explicit distribution shift report consistent improvements over classical baselines such as Bayesian optimization and gradient methods as well as OPRO style prompting, with ablations on equivalence relations, embeddings, budgets, temperature, and knowledge quality. The writing is clear overall and the approach is theoretically motivated, though some key engineering details are pushed to the appendix and several evaluations rely on learned or simulated outcome functions.

### Strengths
1. A principled formulation of patient level decision-making as constrained conditional optimization that combines a Wasserstein distribution shift control with an entropy based exploration control, yielding a practical and interpretable loop.
2. Broad empirical evaluation across multiple biomedical tasks with ablations that illuminate sensitivity to embeddings, equivalence relations, budget, temperature, and knowledge quality.
3. Thoughtful use of knowledge tools and reflection rather than vague claims, with explicit tests of helpful versus irrelevant knowledge.
4. Clear problem setup, readable pseudocode, and helpful diagrams that separate the roles of the critic, the entropy control, and the knowledge components.
5. Responsible positioning with discussion of clinical caveats and an intention to release code for public tasks.

### Weaknesses
1. Several evaluations use a learned oracle for outcomes rather than real clinical endpoints, which risks baking evaluation bias into the target function and limits external validity.
2. The choice of equivalence relation and embedding strongly affects entropy estimates and search behavior, yet selection guidance is limited and some details are deferred to the appendix.
3. The source critic training and dual updates introduce stability and tuning concerns, and the paper does not fully characterize sensitivity to clipping, step ratios, or learning rates.
4. The pipeline is complex, with interdependent modules for prompting, clustering, critic training, tool access, and dual updates. This challenges reproducibility and may require significant engineering effort.
5. Computational cost is likely high due to repeated LLM calls, clustering, and critic retraining. Wall clock and token cost reporting is incomplete for the main text.
6. Presentation is dense in places and certain implementation choices for knowledge tool wiring and safety controls are difficult to locate in the main body.
7. Baseline tuning details and budget parity need to be fully specified to rule out configuration gaps, especially for OPRO style prompts and human or gradient baselines.
8. Practical safety and deployment are not validated with clinician in the loop studies. The method can degrade with irrelevant or adversarial knowledge and the current safeguards appear limited.
9. The paper is difficult to understand and follow.
10. A lot of materials and supporting evidence for reproducibility, yet no working link to the code.

### Questions
1. For each task, what model families were used to construct the evaluation oracle, and how do conclusions change under alternate architectures or calibrated versions of the oracle?
2. Can you report performance as a function of shift magnitude and describe failure modes when the target distribution materially departs from the source?
3. How is the constraint level chosen in practice? Provide a principled rule or adaptive schedule and an ablation in the main text.
4. Did you try gradient penalty or spectral normalization rather than weight clipping?
5. How many samples per batch are required for a stable estimate of the entropy multiplier and how does its variance depend on temperature and batch size?
6. Beyond k-means and Leiden, can the equivalence relation be learned end-to-end with contrastive or outcome-aware objectives?
7. Precisely how are textbooks, graphs, and the specialist model connected to the LLM context, for example through retrieval, tool calling, or structured APIs?
8. Can you add pre-filters based on citation consistency or entailment and report their effect on robustness?
9. Can you report token counts, wall clock times, and GPU hours per task?
10. Can you publish the exact prompt templates, budgets, and tuning for OPRO style and human baselines, and confirm identical oracle query budgets across methods?
11. How does performance change with smaller or open LLMs, and what minimum capability is required to retain the observed gains?
12. How is patient safety enforced when suggestions deviate from standard practice, including any hard constraints, verification, or human in the loop controls?
13. Can you provide a minimal working example with synthetic data that exercises all components including critic updates, clustering, and knowledge tools, along with seed handling and version pins?

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
This paper introduces LEON (LLM-based Entropy-guided Optimization with kNowledgeable priors), a framework that leverages large language models as conditional black-box optimizers for personalized medicine under distribution shift. The authors formulate personalized treatment design as a conditional optimization problem, where the LLM iteratively generates candidate treatment strategies that are evaluated and refined using the LEON objective across multiple rounds. Evaluated on five real-world personalized medicine tasks, LEON consistently outperforms both traditional and LLM-based baselines, achieving an average rank of 1.2 without any model fine-tuning.

### Strengths
- The work addresses an important and emerging problem—how to leverage LLMs for safe, knowledge-guided optimization in personalized medicine under distribution shift.

- LEON consistently outperforms both traditional and LLM-based baselines across multiple real-world personalized medicine tasks.

- The paper is well-structured and easy to follow, with the key ideas and methodological steps clearly highlighted.

### Weaknesses
See questions.

### Questions
1. How does the computational and monetary cost of LEON scale with the length of the LLM context? Since LEON relies on iterative prompting and maintains a growing history of prior designs and knowledge within the prompt, long contexts could substantially increase inference cost. 

2. The paper does not clearly specify the modality of the treatment design. Are treatment regimens represented as structured variables (e.g., dosage vectors, categorical drug combinations) or as natural-language descriptions? Could the authors elaborate on how treatment designs are represented, constrained, and parsed across the five experimental tasks, and whether different modalities were handled uniformly?

3. How does the LLM perform without the optimization procedure (i.e., when generating treatment designs directly from prior knowledge without iterative scoring and refinement)? Could the authors include an ablation comparing this zero-iteration baseline to the full LEON process? Additionally, it would be helpful to visualize performance as a function of iteration number to illustrate convergence behavior and the marginal gains achieved through iterative optimization.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose LEON, a framework that formulates personalized medicine as a conditional black-box optimization problem guided by two constraints: a Wasserstein distance bound to reduce out-of-distribution risk, and a coarse-grained entropy regularization to promote deterministic sampling across semantically similar treatments. LEON iteratively prompts an LLM to generate treatments using patient context and external medical knowledge, updating dual variables to balance exploration and prior alignment. Experiments on five clinical tasks demonstrate strong performance across multiple baselines.

### Strengths
- Introduces learnable determinism via entropy and Wasserstein constraints, moving beyond static penalties or temperature tuning.
- Lemma 4.2 simplifies the constrained optimization problem, making implementation tractable. The dual gradient update for lambda is intuitive.
- The algorithmic steps are well explained, including the role of prior knowledge and reflective prompting.
- Outperforms strong baselines on several clinical proxy tasks without fine-tuning, highlighting the promise of uncertainty-aware, knowledge-driven generation.

### Weaknesses
1. Clustering is heuristic (embedding + k-means), with no expert verification or sensitivity study on class count/distance metrics. Efficiently finding within-class optima $x_i^*$ in high-dimensional/discrete spaces is not thoroughly discussed.

2.  Lipschitz constraint via $\ell_\infty \leq 0.01$ clipping may reduce capacity or cause instability; no comparison to spectral norms or gradient penalties.  Assumes access to source domain $D_{\text{src}}$, limiting applicability in private or black-box settings; no fallback strategy is provided.

3. Some data (e.g., cancer/ADR) are proprietary; no prospective or human-in-the-loop evaluations are conducted.  "Human baseline" may overestimate human quality by ignoring real-world constraints (drug access, comorbidities, etc.).

4. Fairness and safety not evaluated: No subgroup analysis (e.g., gender/race) on Warfarin/HIV tasks. No safeguards against unsafe treatment suggestions or bias propagation.

### Questions
- How is class count $N$ chosen? Were domain-specific structures (e.g., drug targets) considered? Any expert annotations to validate class quality?
- Can you provide $\mu_t, \lambda_t$ trajectories per task and analyze their link to performance? Any divergence observed?
- Why use $\ell_\infty$ clipping over other Lipschitz enforcement methods? Might $c^*$ wrongly reject rare but valid samples?
- How robust is $\mu$ to outdated/noisy/conflicting knowledge? Any causal evidence that knowledge improves determinism?
- Did all baselines (e.g., OPRO, Eureka, LLAMBO) have equal access to external knowledge? If not, is the comparison fair?
- Does the final treatment recommendation account for toxicity, availability, or interactions? Can the method accommodate such constraints?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes using LLM as a black box optimizer to propose stochastic treatment designs in the context of personalized medicine. The paper experiments on real-world optimization tasks to show that the method outperforms other existing methods.

### Strengths
I like the paper a lot. Even though I am not quite familiar with the relevant literature, it is a pleasure to read the paper and learn what it is trying to convey. 

Originality: the use of LLM as black box in solving optimization problems in personalized medicine is quite new. 
Quality: The paper has good theoretical derivations as well as real-world examples. 
Clarity: It is easy to follow for someone like me who does not know relevant literature.

### Weaknesses
Some minor weaknesses: 

1. The idea of using LLM as a black box tool in optimization is studied before. The being said, the paper is still studying a quite original and significant problem.  
2. Some of the really interesting experiments are in the appendix.

### Questions
None

### Soundness
3

### Presentation
4

### Contribution
3
