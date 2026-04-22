# Bayesian Symbolic Regression with Entropic Reinforcement Learning

- Avg Score: 3.33
- Decision: Reject
- Scores: 6, 2, 2

## Abstract
Symbolic regression is the problem of finding an algebraic expression describing a stochastic dependence of a target variable on a set of inputs. Unlike forms of regression that fit parameters assuming a fixed model structure, symbolic regression is a search problem over the space of expressions, represented, for example, as abstract syntax trees using a library of operators. Symbolic regression is typically used in settings with limited, noisy data in the natural sciences. However, searching for a single best-fitting expression fails to capture the epistemic uncertainty about the expression, which motivates a Bayesian perspective that enables uncertainty quantification and specification of natural priors to constrain the search space. In this work, we propose ERRLESS (Entropy-Regularised Reinforcement Learning for Expression Structure Sampling), a scalable approach for sampling the posterior distribution over expressions given data using maximum-entropy reinforcement learning. ERRLESS learns a neural policy that constructs expressions sequentially by building up their abstract syntax trees. At convergence, the policy samples expressions from the posterior. At test time, expressions can be sampled by rollouts of this policy. We demonstrate that ERRLESS achieves near state-of-the-art exact symbolic recovery on the AI Feynman benchmark. Beyond exact recovery, we demonstrate that the mean of the posterior predictive approximated by ERRLESS achieves a coefficient of determination ($R^2$) of $0.98$, highlighting the benefits of the Bayesian perspective in symbolic regression.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper study the a Bayesian symbolic regression (SR) framework and proposes  Entropy-Regularized Reinforcement Learning for Expression Structure Sampling 

ERRLESS models a posterior distribution over expressions using maximum-entropy reinforcement learning (RL). The core idea is to amortize posterior sampling of symbolic expressions using a neural policy trained via trajectory balance (a GFlowNet-style objective).

### Strengths
- Solid mathematical definition. The paper provides formal definitions of priors, likelihoods, unit constraints, and posterior training objectives.
- Novel idea on Translating Bayesian symbolic regression into a maximum-entropy RL setting.

### Weaknesses
1. **Organization and Focus of the Paper**

   * The paper’s organization is not well-balanced. A large portion of the text focuses on the details of post-order traversal and unit constraints, which could be greatly simplified or moved to the appendix.
   * Instead, the section on translating Bayesian symbolic regression into a **maximum-entropy reinforcement learning (RL)** framework should be expanded and clearly explained.
   * The motivation and benefits of using **maximum-entropy RL** are not adequately discussed — the authors should explicitly explain why this formulation is preferable compared to standard RL or other probabilistic inference methods.

2. **Comparison to Deep Symbolic Regression (DSR)**

   * The authors should more clearly differentiate their approach from **Deep Symbolic Regression**, which also employs RL and includes an entropy regularizer. It is unclear what the theoretical or practical advantage of the proposed method is relative to DSR.

3. **Clarification of Prior Work Description**

   * The statement that existing methods “have the goal of finding a single best expression” is inaccurate. Many symbolic regression methods explicitly search for a **Top-K set** or a **Pareto front** of optimal expressions under different objectives (e.g., accuracy and simplicity). The description of prior work should be corrected accordingly.

4. **Comparison with Monte Carlo Tree Search-based Methods is missing.**

5. **Improvement of Figure 1**

   * Figure 1 currently illustrates only the internal mechanism of the proposed approach. It should also highlight the **advantages or improvements** of this method compared to existing baselines — for example, how it better models uncertainty, enforces constraints more effectively, or improves sampling efficiency.

### Questions
- Could you clarify the conceptual and practical differences between deep reinforcement learning with entropy regularization and maximum-entropy reinforcement learning (MaxEnt RL)? 
- What concrete advantages did you observe in your experiments when using MaxEnt RL compared to standard RL with entropy regularization
- What underlying factors contribute to this improvement?

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
5

### Summary
The paper introduces ERRLESS, a Bayesian symbolic regression framework that learns to sample both expression structures and parameters through an entropy-regularized reinforcement learning policy. Instead of finding a single best-fit formula, the method models a posterior distribution over symbolic expressions, aiming to capture uncertainty and improve robustness. It also proposes a bottom-up tree-generation process to enforce dimensional consistency of physical units. Experiments on synthetic and Feynman datasets show competitive symbolic recovery and strong predictive performance under noise.

### Strengths
Originality

The paper provides a coherent Bayesian framing of symbolic regression, formulating it as joint inference over expression structures and parameters and linking entropy-regularized reinforcement learning with posterior sampling.

While the use of a neural policy and RL training is standard, the integration of a Bayesian interpretation for uncertainty-aware symbolic regression offers a conceptually unified perspective that may help reframe how probabilistic ideas are applied in interpretable model discovery.

The paper also emphasizes dimensional consistency and structured priors in the generation process, aligning symbolic regression more closely with physical reasoning tasks.

Quality

The methodology is technically consistent and implemented carefully, combining entropy-regularized RL, structural constraints, and parameter sampling in a logically sound way.

The empirical section is comprehensive, evaluating ERRLESS on both synthetic and Feynman datasets with appropriate metrics (symbolic recovery, predictive accuracy, and uncertainty calibration).

The results show competitive performance and credible uncertainty estimates, demonstrating that the framework is functional and not merely theoretical.

Clarity

The paper is clearly written and well organized, with intuitive explanations, clean mathematical notation, and helpful figures.

The connection between the Bayesian posterior objective, the entropy-regularized RL formulation, and the sampling procedure is explained transparently, making the framework accessible to both symbolic regression and probabilistic ML audiences.

### Weaknesses
Originality and Positioning

The claimed methodological contributions are not sufficiently novel relative to existing symbolic regression frameworks. Using a neural policy trained via reinforcement learning to generate symbolic expressions has been extensively explored in prior works such as Deep Symbolic Regression (Petersen et al., 2019), the Finite Expression Method (Liang et al., 2022), PhySO (Tenachi et al., 2023), Neural-Guided Genetic Programming (Li et al., 2023). ERRLESS largely follows this paradigm, differing mainly in applying a Bayesian interpretation.

The paper could improve by clearly distinguishing what aspects of the Bayesian formulation are algorithmically new (e.g., how entropy regularization meaningfully approximates posterior inference beyond standard policy exploration) and by acknowledging that probabilistic sampling of expressions is already a common practice in symbolic regression.

The introduction and related work should also discuss Parsing the Language of Expression (Huang et al., 2025), which incorporates domain-aware symbolic priors, and other grammar-based Bayesian approaches (e.g., Probabilistic Regular Tree Priors, Grammar-Guided GP). This would provide a more accurate contextual positioning.

Bayesian Formulation and Computational Feasibility

The proposed Bayesian approach introduces significant computational overhead without evidence of efficiency or scalability benefits.

Modeling a posterior over both expression structures and parameters entails sampling from a high-dimensional, multimodal distribution, which is computationally intractable for expressions with many constants. In practice, the method collapses this to a simple Gaussian approximation, undermining the Bayesian claim.

The paper could improve by (a) including runtime and complexity analyses, (b) providing ablation studies showing how posterior sampling scales with model size, and (c) discussing approximations (e.g., variational or factorized posteriors) that might make the Bayesian framework practical.

Bottom-Up Generation Justification

The claimed advantages of bottom-up tree generation—namely producing valid intermediate expressions and easier dimensional checks—are not unique to this design.

Top-down generation frameworks can also ensure dimensional compatibility through type constraints or operator masking (as in PhySO) and can extract valid subexpressions from partial trees.

To strengthen this component, the authors should provide comparative experiments or ablation results demonstrating concrete benefits (e.g., reduced invalid expressions or faster convergence) of bottom-up versus top-down generation.

Experimental Scope and Analysis

The quantitative results show comparable or slightly worse performance than existing methods (e.g., 44% recovery on Feynman versus ~59% for PhySO and 50–60% for DSR/PySR). Yet, the discussion emphasizes conceptual rather than numerical advantages.

The paper would be stronger with statistical analyses (variance, confidence intervals) and runtime comparisons across methods, which would help clarify whether ERRLESS provides any tradeoff between accuracy, uncertainty, and efficiency.

Additionally, more real-world or scientific case studies (beyond synthetic and Feynman benchmarks) would help demonstrate the claimed benefits in uncertainty quantification or robustness.

Use of Structural and Physical Priors

The inclusion of structural priors and physical constraints is not new. Prior work such as Parsing the Language of Expression (Huang et al., 2025), SciMED (Nature 2024), ParFam (OpenReview 2025), and Grammar-Guided GP already incorporate these ideas.

The authors could strengthen their contribution by proposing a more formal or learnable prior mechanism (e.g., data-driven estimation of structural likelihoods or adaptive physical-unit embeddings) rather than manually specified constraints.

Presentation and Scope of Claims

While the writing is generally clear, several claims (e.g., “amortizes sampling of both structures and parameters efficiently” and “bottom-up generation enables better physical reasoning”) are overstated relative to the evidence.

The authors should moderate such claims or substantiate them with direct quantitative or ablation evidence. Clarifying the limitations of the Bayesian approximation would improve credibility.

### Questions
Clarification on Bayesian Inference Approximation

The paper frames ERRLESS as performing Bayesian inference over both structures and parameters. Could the authors clarify how the high-dimensional posterior over constants is represented and sampled in practice?

Is the policy sampling parameters from a fixed Gaussian, or is there an adaptive approximation (e.g., variational or amortized posterior)?

A more detailed description of the Bayesian approximation would help assess whether ERRLESS meaningfully captures parameter uncertainty or simply adds stochasticity to RL sampling.

Computational Complexity and Runtime Evidence

The paper claims that ERRLESS amortizes parameter fitting into policy training, implying computational advantages. Could the authors provide runtime comparisons (e.g., training time or sample efficiency) versus existing methods such as DSR, PySR, PhySO, or FEX?

How does the method scale with expression depth, number of constants, or dataset size? Concrete complexity or scaling curves would make the efficiency claim more convincing.

Bottom-Up Generation vs. Top-Down Approaches

The authors argue that bottom-up tree construction ensures valid intermediate expressions and easier enforcement of dimensional consistency.

Could they provide an empirical comparison or ablation study showing how bottom-up generation affects search efficiency, validity rates, or symbolic recovery compared to a top-down baseline?

If both strategies are possible, under what conditions is bottom-up generation more advantageous?

Quantitative Analysis of Uncertainty Quality

The paper highlights uncertainty quantification as a strength. Could the authors provide quantitative metrics for uncertainty calibration (e.g., negative log-likelihood, coverage probability) and compare them with ensemble-based or bootstrapped SR methods?

Demonstrating that the posterior uncertainty is useful (e.g., for model selection or active learning) would significantly strengthen the paper’s claims.

Evaluation Breadth and Reproducibility

The benchmarks are limited to synthetic and Feynman datasets. Are there plans to test ERRLESS on real-world scientific or engineering problems, where uncertainty estimation might be critical?

Providing open-source code and pre-trained policies would also enhance the paper’s impact and reproducibility.

Treatment of Structural and Physical Priors

The use of structural and dimensional priors resembles earlier approaches such as Parsing the Language of Expression (Huang et al., 2025), PhySO (Tenachi et al., 2023), and ParFam (2025).

Could the authors clarify what is new in their prior formulation—does it introduce learnable or data-driven components beyond manually specified grammar rules?

If possible, an ablation comparing ERRLESS with and without these priors would help quantify their contribution.

Interpretation of “Amortized Sampling”

The phrase “amortizing sampling of both expression structures and parameters” is central to the paper’s narrative. Could the authors more precisely define this term?

Is the amortization referring to shared policy parameters across different expressions (as in DSR) or to efficient reuse of parameter inference results? A clearer explanation would clarify the claimed efficiency benefits.

Positioning Relative to Prior Work

The paper would benefit from a clearer comparison to Bayesian symbolic regression literature (e.g., Probabilistic Regular Tree Priors, Grammar-Guided GP, and Neural-Guided GP).

Could the authors explain how ERRLESS differs conceptually or computationally from these methods and whether the probabilistic formulation offers tangible new capabilities?

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
5

### Summary
The paper proposes a Bayesian formulation for reinforcement learning of symbolic regression. The idea is to construct a joint probability distribution over the expression tree and the measurement data. The logarithm of the joint probability is used as the reward, and then an entropy regularization term is added to conduct reinforcement learning. The experiments on a small manually constructed synthetic dataset and Feynman SR database demonstrate the effectiveness of the method.

### Strengths
1. The writing is clear.
2. the method formulation is correct.

### Weaknesses
1. Novelty is limited. Entropy regularization is a common technique for RL used to encourage exploration. This has already been used in the recent most related work, DSR. It is not the unique contribution of this work. 

2.  The motivation is not reasonable enough. In fact, all RL based symbolic regression is learning a probabilistic expression generator. For example DSR uses an RNN to implement Eq (3) (based on the preoder traversal of the tree). The recent CADSR [1] uses transformer to implement Eq(3) based on breadth-first search.  Via RL, all these methods are learning a posterior expression tree sampler --- given the measurement data. The proposed method mainly differs in using bottom-up order for tree generation.  It does not seem clear about the advantage.  

3. The empirical results are limited. At least, the proposed method should be tested on the more comprehensive SR bench database, which includes multiple datasets (including Feynman) and many black-box problems. 


[1] Bastiani Z, Kirby R M, Hochhalter J, et al. Complexity-Aware Deep Symbolic Regression with Robust Risk-Seeking Policy Gradients[J]. arXiv preprint arXiv:2406.06751, 2024.

### Questions
see above

### Soundness
2

### Presentation
3

### Contribution
2
