# Forward Chaining Neural Network for Rule Induction

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 2, 6

## Abstract
Inductive Logic Programming (ILP) learns logical rules from data, forming an interpretable machine learning model.
Early-stage symbolic ILP systems perform outstandingly on small-scale tasks but suffer from combinatorial explosion.
Emerging neuro-symbolic ILP methods demonstrate a certain degree of scalability and are more robust to noisy data.
However, existing neuro-symbolic ILP methods are limited to constrained language biases, hampering further scalability.
In this work, we propose Forward Chaining Neural Network (FCNN), a stochastic neural network that can learn logical rules under any language bias.
FCNN relaxes all syntactically correct rules into continuous spaces and searches for the semantically correct solutions via gradient-based optimization.
Experiments on standard evaluation tasks and recently proposed large-scale tasks show that FCNN outperforms existing methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes the Forward Chaining Neural Network (FCNN), a neuro-symbolic model for Inductive Logic Programming (ILP). Unlike earlier neural ILP approaches (e.g., ∂ILP, LRI, HRI, DFORL), FCNN introduces a universal meta-rule framework that allows learning Horn rules of arbitrary arity and body length, under both closed-world and open-world assumptions. The method relaxes symbolic unification into a continuous probabilistic framework, parameterizing head and body atoms with embeddings and Bernoulli random variables. Optimization is performed using nested REINFORCE estimators and entropy regularization. Experiments on both classic small-scale ILP benchmarks and the new large-scale GeoILP dataset show that FCNN outperforms previous neuro-symbolic systems and even recent reasoning LLMs on most tasks.

### Strengths
The idea of fully relaxing Horn rule unification into a differentiable stochastic process is elegant and theoretically sound. The probabilistic modeling of atom and variable unification with Bernoulli and categorical distributions provides a flexible parameterization.

The paper is rigorous, with clear mathematical definitions, probabilistic modeling, and proofs (e.g., unbiased gradient estimator, completeness theorem). Algorithmic details for differentiable subset sampling are also well described.

The paper provides a concrete procedure (Algorithm 2) for extracting interpretable symbolic rules from the learned stochastic representations — an important feature for ILP research.

### Weaknesses
Presentation and readability.

Lack of conceptual comparisons.

Limited ablations on modeling choices.


i will detail these points in the section below.

### Questions
**Presentation and readability.** The paper’s exposition is overly dense and bottom-up. The intuition behind the model (e.g. idea of reparametrizing unifications) could have benefited from a clearer top-down narrative — motivating ideas first, then details. A graphical illustration could also have helped.

**Limited ablations on modeling choices.** Although prior neuro-symbolic systems like ∂ILP, LRI, HRI, and DFORL are discussed, comparisons to alternative paradigms that feel closer in paradigm misses. For example, DiffLog [1] and AlphaILP[2] exploited templates and forward chaining for ILP. How does it differ? Also, DeepSoftLog[3] introduces the idea of soft unification into probabilisti logic programming. Soft unification is actually a different parameterization (kernel-like) of a probabilisticu unification. Also in that paper there was some ILP on automata. What are the links?

**Limited ablations on modeling choices.** The ablations mainly address sample sizes and OWA vs CWA modes. Missing are experiments probing the necessity of specific design choices (e.g., embeddings vs discrete parameters, REINFORCE vs relaxation-based training, as many other models do).


[1] Si, Xujie, et al. "Synthesizing datalog programs using numerical relaxation." arXiv preprint arXiv:1906.00163 (2019).
[2] Shindo, Hikaru, et al. "Learning differentiable logic programs for abstract visual reasoning." Machine Learning 113.11 (2024): 8533-8584.
[3] Maene, Jaron, and Luc De Raedt. "Soft-unification in deep probabilistic logic." Advances in Neural Information Processing Systems 36 (2023): 60804-60820.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This submission investigates neural-symbolic inductive logic programming (ILP) with the goal of learning logical rules from data. The core mechanism employed is a forward-chaining neural network based on so-called meta-rules (in the form of Horn clauses). Its main advantage lies in relaxing syntactic constraints on rules—for example, allowing the arity of predicates to increase—and thereby supporting, to some extent, the open-world assumption.

### Strengths
- The paper demonstrates some progress compared to prior work. In particular, while previous approaches typically impose strong restrictions on atoms, the current method can, in principle, handle general Horn clauses.

- The paper is written in a formal and structured manner, which contributes to its clarity and readability.

### Weaknesses
-  Limited novelty. The work follows a fairly standard approach to tackling ILP in a neuro-symbolic manner—namely, by neuralizing logic rules (in this case, via forward-chaining neural networks) and softening symbols through distributions, thereby effectively continuizing traditional discrete objects. The use of REINFORCE is also conventional. In this respect, the contribution appears to lie primarily in adapting existing techniques to the ILP setting, rather than introducing fundamentally new ideas.

-  Overstated contributions. The framework remains template-based, albeit with a more relaxed form than prior work. Consequently, its ability to address the open-world assumption (OWA) is still limited. For example, it is unclear whether the proposed method can discover new predicates or formulate new symbolic concepts, rather than relying on predefined templates.

- Limited impact. The experimental evaluation is restricted to ILP benchmark datasets, which are relatively small and arguably toy problems. Given that ILP represents a niche research area, the scope of the current paper appears narrow, as reflected in the limited breadth of related work. It remains uncertain—though IMHO unlikely—whether the proposed method can generalize to real-world applications.

### Questions
- Line 059, without **strong** language bias. What does strong mean here? 
- Line 205-206, I do not really understand “ i.e., unifying the head atom according to a probability”, “Such a unification corresponds to the property that the Horn rule only allows one head atom.” 
- Line 295 “After optimizing, the distributions are supposed to collapse to deterministic distributions, […]” why?

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
3

### Summary
This paper proposes FCNN, a stochastic neural network that can learn logic rule. They introduce a universal meta-rule that serves as a general template for Horn rules, removing the strong language bias and manual variable assignment present in prior neural ILP systems. FCNN performs probabilistic forward-chaining reasoning and optimizes the expected reward via REINFORCE. The paper evaluates the method on both classical tasks and large-scale ILP tasks with open-world setting, where previous neural ILP systems fail to scale, and shows that FCNN outperforms prior neural ILP systems and LLMs.

### Strengths
1. This paper addresses the limited expressivity and strong language bias of prior neural ILP systems.
2. This method connects inductive logic learning with stochastic gradient optimization via probabilistic rule sampling and the REINFORCE estimator.
3. The paper demonstrates scalability and interpretability on both small and large-scale ILP tasks, outperforming prior neural ILP systems and LLMs.

### Weaknesses
1. The overall algorithm is not clearly presented, algorithm 1 and 2 describe partial components (rule sampling and body-atom extraction), but the complete training loop is missing, which makes the method section difficult to follow.
2. The theoretical part on probabilistic relaxation equivalence is intuitive but lacks formal assumptions (e.g., boundedness, convergence) and a detailed proof.
3. The efficiency and scalability of the method are not discussed, while a simple complexity estimate is given for body-atom sampling, this method could be more computationally expensive than traditional neural ILP. Its efficiency should therefore be analyzed or at least empirically compared with other neural ILP systems and LLMs.

### Questions
1. It is impressive that FCNN solves all tasks and outperforms other neural ILP systems, but the discussion is limited. Could the authors clarify which components (e.g., probabilistic rule sampling, variable unification, or optimization scheme) contribute most to the improvements?

2. What is its sample and computational efficiency compared to other neural ILP or LLM-based reasoning systems? How does runtime scale with the number of predicates or body-atom length?

### Soundness
2

### Presentation
1

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
This paper introduces FCNN, which addresses a fundamental limitation in neuro-symbolic ILP, namely that most existing methods are restricted to unary/binary predicates and ≤2 body atoms. The paper proposes universal meta-rules that probabilistically unify with candidate atoms and variables through learned embeddings, optimized via nested REINFORCE. Key contributions: (1) direct learning of variable-argument unification vs. manual specification, (2) linear-time constrained sampling algorithm, (3) support for arbitrary predicate arities. Results: 100% success on standard small benchmarks including the historically difficult Fizz/Buzz tasks; outperforms LLM on GeoILP with high-arity predicates (up to 8 arguments). Theorem 4.3 proves completeness.

### Strengths
Originality: Universal meta-rule framework removes restrictive templates; direct variable-argument learning vs. manual specification is novel approach. Symbol randomization for fair LLM comparison is methodologically sophisticated.

Quality: Completeness theorem provides theoretical foundation; ablation studies examine key design choices; 100% success on standard benchmarks; handles 8-arity predicates where baselines fail.

Clarity: Motivation clear; problem formulation well-justified; experimental setup rigorous.

Significance: Improves on fundamental scalability barrier in neuro-symbolic ILP; enables learning of complex rules with arbitrary arities; GeoILP benchmark specifically designed to test claimed contributions.

### Weaknesses
1. Figure 1 (Sensitivity of sample size) shows larger sample sizes increase variance with no explanation provided. This raises questions about RLOO baseline effectiveness and practical deployability.

2. No time/space complexity analysis, runtime comparisons, or convergence speed discussion. Critical for assessing practical scalability beyond synthetic benchmarks.

3. Symbolic hyperparameters (|U|, B, V, auxiliary predicates) replace template design bias with hyperparameter bias. No guidance for setting these; ablations only cover Na, Nv. "Sufficiently large" appears frequently without bounds.

4. CWA/OWA comparison on only 4 tasks; missing ablations for embedding dimensions, number of meta-rules, entropy decay schedule.

5. Extension from equality (Ahmed et al.) to inequality constraint claimed "almost the same" but is non-trivial technical step deserving explicit derivation.

6. GeoILP limited to "basic" level despite scalability claims; no failure mode analysis; learned rules not shown/analyzed for interpretability.

### Questions
How were symbolic hyperparameters chosen for GeoILP? Was domain knowledge used? What happens with significantly oversized B, V—does optimization fail or does entropy regularization compensate? 

Briefly elaborate the key technical step extending equality constraint (s=k) to inequality (s≤B) beyond "almost the same" citation.

What are time/memory complexities? How do training times compare to HRI/DFORL on matched tasks?

Can you show examples from GeoILP? Are they interpretable and geometrically meaningful?

Did you test LLM with original (non-randomized) GeoILP predicates to quantify the "reasoning gap" closed by semantic knowledge?

### Soundness
3

### Presentation
3

### Contribution
3
