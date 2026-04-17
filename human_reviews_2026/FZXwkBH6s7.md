# Learning Human Habits with Rule-Guided Active Inference

- Decision: Accept (Poster)
- Scores: 4, 6, 4, 6

## Abstract
Humans navigate daily life by combining two modes of behavior: deliberate planning in novel situations and fast, automatic responses in familiar ones. Modeling human decision-making therefore requires capturing how people switch between these modes. We present a framework for learning human habits with rule-guided active inference, extending the view of the brain as a prediction machine that minimizes mismatches between expectations and observations, and computationally modeling of human(-like) behavior and habits. In our approach, habits emerge as symbolic rules that serve as compact, interpretable shortcuts for action. To learn these rules alongside the human models, we design a biologically inspired wake--sleep algorithm. In the wake phase, the agent engages in active inference on real trajectories: reconstructing states, updating beliefs, and harvesting candidate rules that reliably reduce free energy. In the sleep phase, the agent performs generative replay with its world model, refining parameters and consolidating or pruning rules by minimizing joint free energy. This alternating rule–model consolidation lets the agent build a reusable habit library while preserving the flexibility to plan. Experiments on basketball player movements, car-following behavior, medical diagnosis, and visual game strategy demonstrate that our framework improves predictive accuracy and efficiency compared to logic-based, deep learning, LLM-based, model-based RL, and prior active inference baselines, while producing interpretable rules that mirror human-like habits.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors have sought to extend the ActInf framework by introducing a trainable library of symbolic rules that model habitual behavior. This happens via an alternation between wake and sleep phases whereby in the wake phase, candidate rules are collected and in the sleep phase the rules are consolidated and pruned using generative replay. The hybrid policy can then choose between the symbolic fast rules, and the slower deliverative ActInf planning using the full expected free energy minimisation over trajectories.
Experiments are run across four very distinct domains, showing improved accuracy (5–12 %) and reduced latency (2–10×) over baselines such as DreamerV2, STLR, and prior AIF variants.

### Strengths
There is a good biological plausibility to the whole architecture, which is certainly interesting, and the integration of the symbolic with the neural is elegant and theoretically well founded. The alternating wake-sleep mechanism is again well biologically motivated, and the fast and slow thinking paradigms are again seemingly consistent with real planning/action. The results show that the framework can apply to pretty distinct sequential decision tasks, which shows good coverage and potential generality.
The authors present quantitative analyses showing how rule-bank size mediates accuracy and latency, giving practical insight into how to balance interpretability and efficiency.

Overall, the presentation is well put together and clear

### Weaknesses
The main issue to me is the lack of statistical rigour. Although the reported accuracy and latency improvements appear consistent, the authors provide no confidence intervals, variance estimates, or multiple-seed averages. Because the wake–sleep rule extraction process and generative replay are stochastic, these missing statistics make it impossible to assess whether the gains are robust or simply artifacts of single-run variability. This detracts a lot from the results' overall message.

The system’s interpretability is not actually intrinsic. Rules are defined over latent states, meaning that human readability depends on the decoder’s ability to preserve semantic structure. If the latent space drifts or becomes entangled, rules lose meaning. Because of this, it seems that Interpretability is contingent on the quality of the learned representations rather than guaranteed by the rule architecture itself.

From what I can see, all datasets also rely on LLM-derived feature engineering and mental-state initialization (as discussed in appendix D.3). This external semantic bootstrapping partially determines rule quality and seems to go against claims of generality. No ablation quantifies how performance changes without these priors. Demonstrating that rules can emerge from raw or unsupervised features would really strengthen the contribution.

The rule activation depends on hard matching of m_MAP to m_f^* which forces a winner-takes all intent selection. This design precludes probabilistic blending of intentions and this could cause brittle switching near category boundaries. I think that one ought to be able to implement a soft gating mechanism which may make it more robust.

Key parameters seem to vary dramatically across datasets which produces very large accuracy swings. There doesn't seem to be any principled selection rule, which seems to indicate that the system is heavily domain-tuned.

I think that there is also some literature which has not been cited, which is relevant.

Tschantz, A., Baltieri, M., Seth, A. K., & Buckley, C. L. (2020). Learning action-oriented representations through active inference. 
Dolan, R. J., & Dayan, P. (2013). Goals and habits in the brain.
Cushman, F., & Morris, A. (2015). Habitual control of goal selection in humans. 
Bacon, P.-L., Harb, J., & Precup, D. (2017). The option-critic architecture. 
d’Avila Garcez, A., et al. (2019). Neural-symbolic learning and reasoning: A survey and interpretation.

Currently the paper is marginally below the acceptance threshold, in large part because of the lack of statistical rigour in the results which weakens the arguments significantly.

### Questions
In addition to questions regarding the weaknesses above, can the system be used for continuous control and/or hierarchical action spaces?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a framework for learning human habits within rule-guided (deep) active inference, modelling both planning and rapid, habitual responses. Such habitual responses (the main contribution) use a kind of symbolic rule extraction with generative world models, learned via a sleep-awake algorithm. Empirically, the authors provide both interpretability and strong predictive performance across diverse domains, including human sports, driving, and clinical diagnosis tasks.

### Strengths
This work tackles an interesting problem, and manages to scale up active inference models to "large" scale tasks (large for the active inference community, not so much for the DL one). Also, empirical results demonstrate high accuracy and low inference latency compared to both deep learning and logic-based baselines, across multiple domains.

The wake-sleep algorithm is a nice contribution, and it seems to work well. 

Well written paper, precise, and to the point.

### Weaknesses
What is the reason for not testing on the whole atari26? Have you optimized your architecture for specific games only, or is there another underlying reason? It would be a great results to show performance on a popular benchmark lile the Atari 100k challenge using such a model. 



Minors:

- Weird phrasing at the end of the abstract (typo)

- Suboptimal choice of citations: When you introduce active inference, you only cite (the very nice work of) P.Mazzaglia, 2022. However, that should not be the first citation: the main citation for active inference today is the official book (T. Parr et al., Active Inference: The Free Energy Principle in Mind, Brain, and Behavior (2022)). It is missing here. More generally, all the main papers that have introduce the theory of active inference are missing. I'd suggest the authors to go through the main K.Friston's papers on active inference/free energy principle, and address that. To conclude, there is also a work that introduces a more "habitual" kind of planning, that is inductive inference (https://direct.mit.edu/neco/article-abstract/37/4/666/128203/Active-Inference-and-Intentional-Behavior?redirectedFrom=fulltext). It is very different from what you propose, but still a related work. 

- Some figures are basically unreadable given the tiny font.

### Questions
Do you think this method can scale up to more complex tasks? I'm thinking of Atari26, but also others environments typical of the model based RL literature.

Why only Berserk, and not others? Is there a specific reason for picking this game only? Am i correct in thinking that some additional problems may arise if we plan to implement your proposed method to a more complex/general setup? If yes, how do you think such problems will have to be addressed in the future?




I would be happy to raise my score if my concerns/comments are addressed.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a hybrid active‑inference (AIF) framework in which symbolic “rules” are learned and consolidated via a wake–sleep procedure and then fused with expected free energy (EFE) planning for action selection. Concretely, the latent state is split into a continuous external state (S_t) and a discrete “mental state” (m_t). Candidate rules of the form ((S_f^\star, m_f^\star) -> a_f) are harvested during a wake phase and consolidated with generative replay in sleep. At decision time, if a rule “hits” (via a kernel match to the MAP estimate of ((S_t,m_t))) it supplies a prior over actions; otherwise the agent falls back to EFE‑based planning. Experiments on NBA player trajectories, car‑following, DDXPlus (URTI), and Atari–Berzerk report higher Acc@k and lower latency than logic‑based, deep learning, AIF, and DreamerV2 baselines, while yielding interpretable rules. (Table 1; figs. on pp. 8–9.)

  The key problem that is tackled is learning habit‑like, interpretable shortcuts that allow an AIF agent to arbitrate between fast habitual actions and slower planning, and applying this to modeling human(-like) behavior across domains.

 The motivation (habits vs. planning) is timely and clear, and the ability to flexibly and context-sensitively tradeoff between habit-based action vs more costly-model based planning is a very valuable one. A small nitpick is that the paper under‑positions itself relative to existing methods in computational cognitive neuroscience, psychophysics and computational psychiatry and the large literature on fitting RL/decision‑making models to human data (e.g., model‑based RL, active inference, DDMs). The Related Work emphasizes neuro‑symbolic rules and AIF agents, but not human model‑fitting traditions.  Reported gains in Acc@k/latency appear consistent across tasks (Table 1) and ablations indicate that rules and (m_t) matter. However, several specification and correctness issues (see below) make it hard to assess whether improvements stem from principled inference or from ad‑hoc fusion/thresholding. 

The idea of rule‑guided AIF and the attempt to compress planning into reusable symbolic habits is interesting and potentially impactful for both interpretable agents and human behavior modeling. In its current form, however, conceptual and notational gaps limit the work’s reliability and portability.

### Strengths
Compelling objective: compressing repeated planning successes into interpretable rules and using them to shortcut EFE search is elegant and practical. 

Clear engineering benefit: latency vs. accuracy curves and ablations demonstrate a Pareto knee where small rule banks reduce inference time while preserving accuracy. 

Visualization & interpretability: rule envelopes in latent space and trajectory overlays help the reader understand what the rules encode. 

Cross‑domain experiments spanning continuous control/trajectories, clinical dialog, and pixels.

### Weaknesses
EFE definition and preference distribution:  In AIF the EFE uses a biased preference likelihood (encoding utilities/preferences) that is not the same as the observation model used for inference. Here EFE is written with the same (p_\phi(O_{t+\tau}\mid Z_{t+\tau})) used by VFE (Eq. 3), and the text does not explain how preferences are encoded. This obscures how “goals” enter planning and undermines the conceptual link to the risk/ambiguity decomposition. 

Ad‑hoc policy fusion and temperature dependence: The hybrid policy adds a rule prior to a temperature‑scaled EFE term (Eq. 4) and gates it with a binary “rule hit” indicator based on a similarity kernel and threshold in ((S_t^\text{MAP},m_t^\text{MAP})). This is anti‑Bayesian in spirit and creates several tuning knobs (kernel, threshold (\tau_r), temperature). A principled alternative is to cast rule invocation as inference under a mixture model (e.g., infer (q(m_t)) and mixture weights over rule‑conditioned policies), letting uncertainty in EFE naturally down‑weight unreliable rollouts without extra temperatures. 

Objective couples VFE and EFE without theoretical justification:  The “total free energy” in Eq. (5) simply sums per‑step VFE, rollout EFE (scaled by (\eta)), and a KL regularizer on successive (q(m_t)) (scaled by (\gamma)). The inclusion ofc (\operatorname{KL}[q(m_{t-1})\Vert q(m_t)]) feels ad‑hoc (shouldn’t that KL come out naturally of the generative model when minimizing VFE), and the paper does not justify why this combination is a coherent bound or how it relates to generalised free energy or control‑as‑inference objectives. (§4.2, Eq. 5.) 

Rule growth is heuristic:  Rules are created/updated when a triplet ((S_t^\text{MAP},m_t^\text{MAP},a_t)) recurs and “reliably reduces free energy,” with centroids updated by exp(−VFE) weights. This is reasonable as an engineering heuristic, but it could be formalised as variational learning in a mixture model with inference over (q(m_t)) and parameters of (p(S_t\mid m_t)) (and optionally a nonparametric prior for rule birth/pruning). 

Typos and clarity 
Minor: Ambiguity of contribution (human modeling vs. model-based RL agent):  The abstract and introduction oscillate between “learning human habits alongside human models” and benchmarking a better agent (“improves predictive accuracy and efficiency”). Please clarify whether the primary goal is computational cognitive neuroscience (fitting human behavior) or agent performance. As written, expectations are set for both. I assume it’s the former, otherwise you would have ben benchmarking on metrics more like reward than model-fit like (the number of accurately predicted actions, rather than whether those actions were “correct” in a maximizing-reward sense). 
Minor: Generative model misspecification in Eq. (1):

No explicit prior over initial hidden states (e.g., (p_\phi(Z_1))).

The factorization includes (p_\phi(a_t\mid Z_t)) inside the generative model while the LHS lists (a_{1:T-1}), creating an indexing mismatch (a term for (a_T) is included on the RHS but not in the LHS).

There is no prior over (a_{t-1}), yet the state transition conditions on it; the dependency graph is inconsistent. (p. 3, Eq. 1.) 
Minor: Latent split factorization also misses initial priors and has indexing mismatches. In the section(Latent State Representation), the joint (p_\phi(O_{1:T},S_{1:T},m_{1:T},a_{1:T-1})=\prod_t p(O_t \mid S_t) p(S_t \mid S_{t-1},a_{t-1}) p(m_t \mid m_{t-1},S_t) p_{\phi^\pi}(a_t \mid S_t,m_t)) lacks priors for (S_1) and (m_1), and again indexes (a_t) up to (T) while the LHS says (a_{1:T-1}). (p. 4.)

“neurosciencetau accounts…” (p. 5, l. ~262) → “neuroscientific accounts.”

Positioning with neuroscience evidence:  The claim that seamless habit–planning switching is a hallmark of human intelligence contradicts years of evidence from the behavioral neuroscience literature and animal studies (in rodents and to some extent monkeys) on the transition between goal-directed and habitual behavior. Classic devaluation and contingency-degradation paradigms demonstrate context-dependent switching early (goal-directed) vs. after overtraining (habit). In terms of its biological basis, evidence suggests that dorsomedial striatum and prelimbic PFC support goal-directed control, while dorsolateral striatum and infralimbic PFC promote habits. Flexibly toggling between cached habits and on-the-fly planning is a general mammalian capability, not a human hallmark. 

Overall recommendation
Key reasons: (i) policy fusion relies on ad‑hoc kernel thresholds and temperatures rather than principled inference; (ii) Generative model and EFE formulation contain gaps/mismatches that need fixing; (iii) contribution framing (human modeling vs. agent) is unclear. The core idea—rule‑guided AIF that amortises planning into reusable, interpretable habits—is promising and, with the above issues addressed, could become a strong contribution.

### Questions
Preferences in EFE: How are preferences represented in Eq. (3)? Is (p_\phi(O_{t+\tau}\mid Z_{t+\tau})) re‑used from the observation model or replaced with a preference distribution? Please spell this out and reconcile with the standard EFE decomposition. (§3.)  

Rule fusion as inference: Could the rule‑vs‑planning arbitration be formulated as posterior inference (e.g., a gating variable) instead of kernel‑plus‑temperature? What prevents learning the weights of this mixture directly via variational inference under your joint objective? (§4.1–§4.2.) 

Objective design: What is the principled justification for VFE + (\eta)·EFE + (\gamma)·KL? Is there a derivation tying this to a bound on expected log‑evidence or a generalised free energy objective? (§4.2.) 

Baselines and fairness: How was DreamerV2 adapted (offline? behaviour‑cloned?) to these supervised prediction tasks and tuned to parity (esp. on DDXPlus/Atari)? Please clarify compute budgets, hyperparameter search, and early‑stopping criteria across methods. (Table 1.)

### Soundness
2

### Presentation
2

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The work introduces a new method within the active inference framework. The methods is made of mainly two novel contributions: (i) a set of symbolic rules, which are learned as part of the world models and allow the agent to decide between deliberate and intuitive prediction, (ii) a wake-sleep training algorithm, alternating phases where the actions are selected and the world model is provided as opposed to phases where acting rules are consolidated. The method is evaluated in "offline" settings, on (small) predictive benchmarks.

### Strengths
* **Novelty**: while there are previous works focussed on learning habitual policies in active inference (see Fountas et al), the proposed approach, learning/distilling rules as part of the world model's representation is, to the best of my knowledge, novel and compelling
* **Interpretability**: the method, using active-inference and a set of rules that are learned by the world models and used by the policy enables stronger interpretability of behaviors. This is an important features, especially compared to more blackbox approaches, such as reinforcement learning and imitation learning.

### Weaknesses
* **Scalability**: if we compare with the current state of AI, the datasets employed are tiny. Even CIFAR 10, which is considered a very small dataset these days, is 3x larger any training dataset used in the paper.
* **Baselines**: while the proposed approach is thoroughly analyzed, it is unclear what are the difficulties of the baselines in solving the given, small problems. Given the nature of the predictive tasks, it would also be useful to consider at least one LLM-based baseline (with a very small model, such as Qwen 0.5B)

### Questions
* What is the parameter count of the model?

Typos:

-> in the abstract "Experiments on basketball player movements, car-following behavior ~demonstrate~, medical diagnosis, and visual game strategy **demonstrate** that our framework"

-> neurosciencetau -> neuroscience

### Soundness
4

### Presentation
4

### Contribution
3
