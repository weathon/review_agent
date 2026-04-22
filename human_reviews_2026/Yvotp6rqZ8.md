# Risk-Optimal Prediction under Unseen Causal Perturbations

- Avg Score: 5.50
- Decision: Reject
- Scores: 4, 6, 6, 6

## Abstract
Predicting intervention effects is important in various scientific fields, including biomedicine. Classical methods depend on fully specified causal graphs and extensive observational data, while recent invariance-based approaches typically assume access to the state of perturbed features.
These assumptions may not hold in practical settings with unknown causal relationships, partial and limited interventional data, and the need to consider novel, untested interventions/perturbations. 
We propose a novel framework for causal effect estimation under such conditions that uses interventional embeddings to capture perturbation-specific information. Leveraging ideas from causality and robust learning, we propose a predictor that 
targets a form of interventional regime-specific risk-optimality but that does so using transformations of available data and hence does not require access to interventional data from the target regime.
We put forward an end-to-end attention-based model that jointly learns embedding transformations and similarity-based weighting, enabling scalable prediction of causal effects even when no features are observed under intervention. Experiments on synthetic and real-world datasets show that our framework generalizes effectively to unseen interventions, hence addressing a critical challenge in prediction of causal effects in complex, real-world settings.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper presents a framework for predicting causal effects under previously unseen interventions, addressing scenarios where causal structures are unknown and interventional data from target regimes are unavailable. The authors introduce the concept of risk-optimal prediction within causal environments, supported by two key assumptions: the Implicit Causal System (ICS), which models data as responses of a latent causal system to interventions, and the Invariant Embedding Transformation (IET), which enables consistent mapping of interventional embeddings across regimes. Building on these assumptions, the work derives a formulation expressing test-environment risk as a weighted combination of training-environment risks and proposes practical algorithms to predict responses in test-environment. Extensive evaluations on both synthetic and biological datasets demonstrate good generalization performances.

### Strengths
- The paper addresses an important problem of *risk-optimal prediction* under unseen causal interventions, effectively integrating ideas from causal inference and robust learning.  
- It develops both feature-dependent and feature-free implementations, thereby extending the framework’s applicability to feature-agnostic data settings.  
- The experimental results demonstrate strong performance across diverse biomedical applications.

### Weaknesses
- The overall framework lacks sufficient clarity, motivation, and comparative analysis. It is recommended to reformulate the approach within a potential outcome or structural equation framework, or at least clarify its relationship to these established causal paradigms to better align with current literature.  
- Under the assumptions proposed, the paper appears to address an overly simplified scenario in which embeddings and coefficients are treated as equivalent for both feature and response prediction—two inherently distinct tasks.  
- The approach relies on linear latent causal structures and orthogonality of interventions, assumptions that may not hold in complex or nonlinear real-world systems.  
- The paper includes limited ablation studies despite relying on strong assumptions and aiming to address DRO in general environments. Furthermore, it lacks detailed discussion of the experimental settings and the extent to which the assumptions are satisfied in each dataset.

### Questions
- In line 169, the paper states that relevant noise terms are included in $g^\*$, $\theta_g^\*$, and $\psi_e^\*$. I am confused about how noise can be incorporated, given that $g^\*$ is a deterministic function, $\theta_g^\*$ represents fixed parameters, and $\psi_e^\*$ denotes fixed information.  
- Could the authors provide concrete examples illustrating how the key assumptions, particularly Assumption 3 and the orthogonality condition, are satisfied in practice?  
- Case II is somewhat unclear, especially regarding how the weights are estimated without any feature observations. A more rigorous mathematical formulation or an explicit algorithmic description would be helpful.  
- It is recommended to include ablation studies, particularly evaluating the robustness of the proposed method under violations of the stated assumptions.  
- Minor issue: brackets are missing on lines 64 and 103. Theorem 3.1 is proved under near-orthogonality assumption.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses predicting causal effects of interventions not seen during training, particularly for applications where testing all possible perturbations is infeasible. The authors introduce the Latent Expressivity Condition (LEC), which assumes test environment representations can be expressed as linear combinations of training environment representations, and derive a theorem showing that test risk decomposes into a weighted sum of training risks. They develop algorithms including an attention-based approach that learns embedding transformations and environment-specific weights jointly, enabling prediction when only intervention embeddings (e.g., chemical structures, gene identifiers) are available rather than direct feature measurements. Experiments on synthetic data and biological datasets show improvements over existing methods like IRM and Anchor Regression.

### Strengths
* The paper tackles a problem with clear practical relevance that could have real impact on prediction tasks - a refreshing contrast to much of the causality literature which often remains quite theoretical and disconnected from applications.

* The authors make a commendable effort to minimize causal assumptions, particularly avoiding the need for precise causal graph specification. This brings their causal prediction framework much closer to real-world applicability.

* The presentation is exceptionally clear and well-organized throughout the paper.

* The use of asterisk notation to denote latent/inaccessible variables and functions is a helpful convention that aids readability.

### Weaknesses
* L232: The authors claim their latent causal system can be viewed as a transformation h that renders interventions additive. However, they don't address what conditions would make this transformation possible. After transformation, the system needs to satisfy both (i) additive interventions and (ii) linear causal relationships in the transformed space. Unless the original system already has these properties (making h trivial), it's unclear which non-linear systems could actually be transformed this way - my impression is this class might be quite restrictive.

* While the paper references invariant risk minimization, it misses important connections to related causal literature:
  - The goal of predicting effects of unseen interventions from observed ones relates directly to intervention generalization work. This includes both general identifiability on pure causal structure [1] and work under parametric assumptions [2, 3, 4].
  - The specific structure in Equation 2 (linear relations, additive shifts, downstream target) has appeared before in causal abstraction identifiability work [5].
  - The multi-environment setup is standard in causal representation learning - see [6] and references therein for context.

**Minor:**

* L247: The term "Expressivity" feels misplaced here. Would it be more accurate to say the train dataset embeddings span the space of possible embeddings?

References:

[1] Lee, Sanghack, Juan D. Correa, and Elias Bareinboim. "General identifiability with arbitrary surrogate experiments." Uncertainty in artificial intelligence. PMLR, 2020.

[2] Saengkyongam, Sorawit, and Ricardo Silva. "Learning joint nonlinear effects from single-variable interventions in the presence of hidden confounders." Conference on Uncertainty in Artificial Intelligence. PMLR, 2020.

[3] Bravo-Hermsdorff, Gecia, et al. "Intervention generalization: A view from factor graph models." Advances in Neural Information Processing Systems 36 (2023)

[4] Kekic et al. "Learning Joint Interventional Effects from Single-Variable Interventions in Additive Models." arXiv preprint arXiv:2506.04945 (2025).

[5] Kekic et al. "Targeted reduction of causal models." (2023).

[6] von Kügelgen, Julius. "Identifiable causal representation learning: Unsupervised, multi-view, and multi-environment." (2024).

### Questions
* Could you clarify what conditions on the original causal system would allow finding a transformation h that satisfies Equation 3?

**Minor:**

* L271: Should this be "exact orthogonality" rather than "near-orthogonality"?
* L472: I'm having trouble parsing "First the linear..." - could you clarify what this sentence means?

### Soundness
3

### Presentation
3

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
This paper tackles a fundamental problem at the intersection of causal inference and robust generalization, how to perform risk-optimal prediction under unseen causal interventions, especially when interventional data or full causal graphs are unavailable. The authors propose a framework grounded in a set of structured assumptions, the Implicit Causal System (ICS) and the Invariant Embedding Transformation (IET), and develop both feature-based and embedding-based learning schemes, including an attention-based model for predicting responses in new interventional regimes.

### Strengths
1. Novel problem framing. The idea of risk-optimal prediction under unseen interventions is highly original and bridges causal inference with distributional robustness.
2. The authors derive both linear-analytic (ridge/lasso) and deep-learning (attention) realizations of the same principle, showing broad applicability.
3. Applications to gene knockout and chemical perturbation datasets demonstrate that this framework has real biomedical impact.

### Weaknesses
1. The ICS and IET assumptions, though conceptually appealing, are not verifiable in real applications. They presuppose that embeddings  preserve causal invariances—this is not guaranteed.
2. Despite the causal framing, the model does not estimate identifiable causal effects or provide counterfactual reasoning. The approach is closer to causality-inspired representation learning rather than formal causal effect estimation.
3. The linear additive form (Eq. 2) is restrictive and may not capture complex non-linear causal dependencies prevalent in biological systems.

### Questions
1. How sensitive is your approach to the choice and quality of embeddings u_e? Would low-quality embeddings (e.g., from unrelated pre-training) invalidate IET?
2. Could your risk-optimal framework be extended to non-linear or non-additive latent causal systems?
3. How do you ensure that learned attention weights alpha_e correspond to meaningful causal similarities between interventions?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a method to predict the response under unseen causal interventional shifts that can be expressed using the interventions/perturbations available during training under certain assumptions. There are two variants for the proposed approach -- one that can work with test-time features, and another that does not require test-time features (it uses some embeddings). The method is validated on both synthetic and real (biological and chemical) datasets.

### Strengths
The idea is very interesting, and the paper is largely well-written. The theoretical results are also strong. I believe these results have practical applications in learning from multiple environments.

### Weaknesses
I have a major (W1) and a few minor (W2-3) concerns about this work, in addition to several questions about some parts of the text (see "questions").

**W1. Setting of this work**: The setting of this work is not clear. There are several terms used, but looking at the math, they seem to be the same thing under the hood. For instance, $x^e$ denotes "features" from an environment $e$. Using these features, we may predict the response variable $y^e$. The inference environment, $v$, is not observed during training. My understanding is that we may or may not have access to the features $x^e$ during training. Instead of $x^e$, we have access to "embeddings" $u^e$ during training (please see my related comment about embeddings in W3). Lines 182-183 say these embeddings "contain relevant information about the intervention in question." Lines 293-295 say that "we can in general write the feature embeddings $h^\*(x^e)$" as a function of only the intervention embeddings $u^e$." From these sentences, I gather that the embeddings $u^e$ are some kind of super-informative variable that can predict both the interventional description $\psi^e$ and the latent embedding $h^*(x^e)$ (which is complex enough to model the intervention as a mean-shift operation).

**W1. (a) Are the two settings different?** At that point, is there any difference in terms of the two considered settings (with and without features)? Having access to only the embeddings, and not the features, does not make the setting any more challenging.

**W1. (b) Do we need the asterisk terms?** If we can always express the unobserved, but informative, variables such as $h^\*$ and $\psi_e^\*$ in terms of observable embeddings $u^e$, why do we need to write anything in terms of $h^\*$ or $\psi_e^\*$? Can't all the findings in this paper be expressed in some learned latent space (like $h$) where eq. (2) holds?

**W2. Comparison to related works**: I would appreciate it if the findings could be contrasted with works such as [A1-2] (and the references therein) that also consider modeling interventional effects under the process in eq. (2). (Shen et al., 2023) and (Rothenhausler et al., 2021), referenced in the paper, also use this interventional model. Specifically, are $\alpha$'s in this work related to anchors in anchor regression (Rothenhausler et al., 2021)?

**W3. Difference between embeddings and features**: How are embeddings $u^e$ different from features $x^e$? The descriptions for $u^e$ in lines 129-130 and 377-390 are not helpful.

### Questions
While I agree that this is not an easy paper to write, the writing could be improved a bit, especially in the introduction. For instance, in the setting description, a more rigorous description of "information on the nature of the intervention" (lines 040-041) will be useful to the readers. Minor questions/comments on writing:

**Q1.** In lines 96, does "features are available for learning..." mean features from causal regime $v$ are available during training?

**Q2.** Why is $c^*$ invariant under IET assumption?

**Q3.** Is it $\epsilon$ or $\varepsilon$ in eq. (2)?

**Q4.** There is some confusion in lines 217-219. Does "the effects are causally downstream" mean the response $y^e$ is an effect of the feature embeddings? In a related note, line 106 said the learned function was "not constrained to use only features that [were] direct causes of $y$". Does that mean the function can use successors of $y$, or that it can use non-direct ancestors of $y$? Does $(\delta_e)_{q+1}=0$ in lines 217-219 mean the changes in $y^e$ due to the perturbation in environment $e$ are purely a downstream effect of the changes in $x^e$?

**Q5.** Line 140 says that * denotes aspects that are not accessible in practice. Why does $h$ that combines $h^\*$, $g^\*$, and $c^\*$ in eq. (7) not have $\*$? Line 392 says $h$ "will be learned from data in an end-to-end fashion." But in algorithm 1, $\hat{h}(x^e) = \phi(u^e)$, where $\phi$ is "a generic regression model" (lines 362-363). How is $\phi$ obtained? And, how is $h$ "invariant across environments" (line 303)?

**Q6.** $k(b, \alpha)$ is defined differently in Theorem 3.1 and in its proof in Appendix B.1. Also, is it supposed to be $\delta_v^\*$ instead of $\delta_e^\*$ in line 651 (and in the equations following that line)?

**References**

[A1] Dominik Rothenhäusler, Peter Bühlmann, Nicolai Meinshausen, "Causal Dantzig: fast inference in linear structural equation models with hidden variables under additive interventions", Annals of Statistics, 2019.

[A2] Julius von Kügelgen, Jakob Ketterer, Xinwei Shen, Nicolai Meinshausen, Jonas Peters, "Representation Learning for Distributional Perturbation Extrapolation", ICLR Workshop on Learning Meaning Representations of Life, 2025.

### Soundness
4

### Presentation
3

### Contribution
4
