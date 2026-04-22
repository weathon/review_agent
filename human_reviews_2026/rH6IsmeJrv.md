# PRISM: Festina Lente Proactivity—Risk-Sensitive, Uncertainty-Aware Deliberation for Proactive Agents

- Avg Score: 6.67
- Decision: Accept (Poster)
- Scores: 4, 10, 6

## Abstract
Proactive agents must decide not only what to say but also whether and when to intervene. Many current systems rely on brittle heuristics or indiscriminate long reasoning, which offers little control over the benefit-burden tradeoff. We formulate the problem as cost-sensitive selective intervention and present PRISM, a novel framework that couples a decision-theoretic gate with a dual-process reasoning architecture. At inference time, the agent intervenes only when a calibrated probability of user acceptance exceeds a threshold derived from asymmetric costs of missed help and false alarms. Inspired by festina lente (Latin: "make haste slowly"), we gate by an acceptance-calibrated, cost-derived threshold and invoke a resource-intensive Slow mode with counterfactual checks only near the decision boundary, concentrating computation on ambiguous and high-stakes cases. Training uses gate-aligned, schema-locked distillation: a teacher running the full PRISM pipeline provides dense, executable supervision on unlabeled interaction traces, while the student learns a response policy that is explicitly decoupled from the intervention gate to enable tunable and auditable control. On ProactiveBench, PRISM reduces false alarms by 22.78% and improves F1 by 20.14% over strong baselines. These results show that principled decision-theoretic gating, paired with selective slow reasoning and aligned distillation, yields proactive agents that are precise, computationally efficient, and controllable. To facilitate reproducibility, we release our code, models, and resources at https://prism-festinalente.github.io/; all experiments use the open-source ProactiveBench benchmark.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces PRISM, a decision-theoretic framework for cost-sensitive, selective intervention in proactive language agents—systems that must decide whether and when to offer help. Instead of always reasoning deeply or using fixed heuristics, PRISM dynamically balances benefit and burden by modeling both need and acceptance probabilities and gating interventions through an adaptive, cost-derived threshold.

### Strengths
The paper offers a fresh formulation of proactivity as cost-sensitive selective intervention, introducing a principled gating mechanism that jointly models user need and acceptance under asymmetric costs. The dual-process “fast vs. slow” reasoning and the festina lentedesign elegantly merge decision theory, calibration, and selective computation.


PRISM provides a practical and general framework for controllable, efficient proactive agents. Its decision-theoretic approach and selective reasoning strategy have broad relevance to risk-sensitive and resource-aware AI.

### Weaknesses
Real user studies or human acceptance feedback is not conducted, which is important in order to show that PRISM’s gating truly improves user experience and perceived helpfulness. 

The framework assumes stable calibration of $p_{need}$ and $p_{accept}$, but no experiments examine how PRISM performs under domain shift or calibration drift.

In the ablation study, the precise effect of threshold width and cost ratios are not systematically explored.

The paper could engage more deeply with related concepts such as metareasoning and risk-sensitive RL to situate PRISM’s novelty.

### Questions
1. Have the authors validated PRISM’s benefits with real user judgments or pilot human studies? Since proactive timing and acceptance depend on user perception, it would be valuable to know how well the LLM-judge proxy aligns with human acceptance and satisfaction. Could the authors provide evidence of correlation between judge labels and human ratings?

2. The approach relies heavily on calibrated $p_{need}$ and $p_{accept}$. How does PRISM handle calibration drift in deployment or across domains? Would an online or adaptive recalibration mechanism (e.g., temperature scaling on recent feedback) help sustain performance?

3. How sensitive are the results to the chosen false-alarm and missed-help cost ratios and slow-margin width?

4. Could the authors provide explicit measurements of runtime latency and token usage compared to always-slow or baseline proactive agents?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
This paper presents PRISM, a framework for proactive agent intervention that combines a cost-derived acceptance gate with margin-gated slow reasoning and calibration-aware distillation. The central contribution is formulating proactive assistance as a cost-sensitive selective decision problem where the agent estimates two calibrated probabilities, $p_{\text{need}}$ (whether help is needed) and $p_{\text{accept}}$ (whether help will be accepted), and intervenes only when $p_{\text{accept}}$ exceeds a dynamic threshold derived from asymmetric costs.

PRISM invokes resource-intensive, slow reasoning only near the decision boundary through a margin parameter $\delta_{\text{slow}}$, concentrating computation where it is most likely to change the outcome. Training employs Risk-Decision Consistent distillation (RDC-SFT) that aligns supervision with the deployment gate and costs. On ProactiveBench, PRISM achieves F1 of 86.61\% (vs.\ 83.28\% for DeepSeek-R1), precision of 77.05\% (vs.\ 72.35\%), and reduces false-alarm rate to 22.94\% (vs.\ 27.64\%), all at near-saturated recall (98.88\%).

### Strengths
1. The decision-theoretic gate is properly derived from Bayes-risk minimization under asymmetric costs (Appendix, Proposition 1). The comparative statics are correct: the threshold tightens with higher false-alarm cost and loosens with higher need probability. This is the *right* way to control the benefit-burden tradeoff, and it connects cleanly to the classical selective prediction and reject-option literatures (Elkan 2001; Chow 2003; Geifman & El-Yaniv 2017). The paper appropriately positions its novelty: while cost-sensitive thresholding is well-established in classification, the *synthesis* (i.e., decomposing intervention into calibrated need/acceptance probabilities, deriving an adaptive threshold, and coupling it with margin-gated slow reasoning) is novel for proactive LLM agents and goes beyond heuristic triggers in prior systems (Lu et al. 2024). The false-alarm cost modeling is particularly well-motivated: recent HCI studies document that unsolicited AI assistance can “backfire,” threatening users’ self-perception and eroding trust (Liao et al. 2016; Zhang & Zhu 2025). PRISM’s risk-sensitive gate directly addresses this documented failure mode.

2. Invoking slow reasoning only operationalizes Russell & Wefald’s (1991) value-of-computation framework; compute where it can change the action. This echoes recent dual-process architectures (e.g., Dualformer’s auto-mode that adaptively switches between fast and slow reasoning, Su et al. 2025) but tailors it to the intervention decision through counterfactual checks. Empirically, this improves F1 by 3.66 points over fast-only and 2.59 points over always-slow while achieving the highest AUDBC (87.73 vs. 79.84 and 76.63). This is consistent with selective classification theory and the need for well-calibrated probabilities.

3. The RDC-SFT objective explicitly rewards calibrated estimates and penalizes squared errors on ( (p_{\text{need}}, p_{\text{accept}}) ), then filters training data by a ranking score (Eq. 3.2). This approach aligns with recent advances in training for calibration (e.g., Damani et al. 2025’s RLCR, which augments rewards with proper scoring rules like Brier score to jointly improve accuracy and confidence reliability). The schema-locked distillation that decouples the response policy from the intervention gate is a practical innovation that addresses deployment concerns: it enables post-training threshold tuning without retraining, yielding auditability. Ablations (Table 4) confirm that RDC-SFT dominates vanilla SFT (F1 86.61 vs. 76.09), weighted-SFT (86.61 vs. 80.59), and DFT (86.61 vs. 76.09), and that the policy improvement depends on calibration (Table 2: dynamic thresholds on uncalibrated probabilities underperform fixed thresholds). Temperature scaling (Guo et al. 2017) is acknowledged as a simple fix for miscalibration.

4.  PRISM achieves large margins over public baselines: +20.14 F1 points and −22.78 % relative reduction in false alarms vs. strong proactive baselines. Crucially, the distilled student *beats* its teacher (DeepSeek-R1) on precision and false-alarm rate at similar recall, which is non-trivial and highlights the value of structured distillation and gating beyond raw reasoning capacity. The evaluation uses a consistent LLM-as-judge protocol with majority voting, and the paper commits to open-sourcing code and models on a public benchmark (ProactiveBench), supporting reproducibility.

5. The framework exposes a compact set of knobs that move the benefit-burden frontier in predictable ways without retraining. This addresses a practical gap: in deployed systems, model outputs and product-level rules are often entangled, making it hard to audit or tune behavior post-deployment. By decoupling the intervention gate from the response policy, PRISM enables independent threshold adjustment—a design principle that echoes earlier mixed-initiative systems (e.g., Horvitz’s work on modeling interruption costs and user attention, 1999–2003) but brings it into the LLM era. This is valuable for product deployment and aligns with best practices from calibration and abstention literature.

### Weaknesses
The main weakness is using LLM-as-judge evaluation, because it's fragile. Labels for $(y_{\text{need}}, y_{\text{accept}})$ rely on DeepSeek-R1 with majority voting over three judges (including GPT-4o, Claude 3.5-Sonnet). This is vulnerable to bias, variance, and rubric drift. A recent comprehensive survey (Ye et al.\ 2024) documents that LLM judges exhibit position bias, inconsistency, and can give overly assertive assessments with unjustified confidence; they are ``not yet ready to fully replace human evaluators, especially for complex, high-stakes judgments.'' While the paper acknowledges this limitation (Section 6) and notes that baselines face the same constraint—making PRISM's gains meaningful \emph{within} the competitive evaluation context—the issue remains: PRISM's gains might partially reflect overfitting to the judge's biases rather than genuine alignment with user preferences. The paper would be substantially strengthened by robustness checks (see Questions below). This is a field-wide challenge, and the requested checks align with emerging best practices (multi-LLM consensus, human-in-the-loop validation).

### Questions
1. Can PRISM maintain its advantage over baselines across *all* judge pools if you re-score 100–200 events with two different judge pools (swap in/out frontier or open models) and report inter-judge agreement (Cohen kappa or Matthews correlation) plus F1 deltas?

2. On a small human-labeled subset (50–100 events) with labels for y_need and y_accept, what do calibration metrics (ECE, Brier) for (p_need, p_accept) and task metrics (F1/precision/false-alarm rate) show? If miscalibration appears, after temperature scaling, how do those metrics change?

3. If you sweep (C_FA, C_FN) over a 2D grid, how do AUDBC, false-alarm rate, and expected latency vary as a function of the slow-margin delta_slow, and does this trace a clear benefit–burden Pareto frontier?

4. How much does each signal contribute under these ablations: (a) gate on p_accept only (set p_need = 1), (b) gate on p_need only (set p_accept = tau(p_need)), and (c) uncalibrated vs calibrated (p_need, p_accept) (pre vs post temperature scaling)?

5. If you re-run the ProactiveAgent pipeline using PRISM’s gate and margin-gated slow reasoning but *without* RDC-SFT (i.e., keep their original model, replace only the decision policy), how much of the gain comes from the gate versus the distillation?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper presents PRISM (Proactive Risk Sensitive Intervention with Slow mode Margin), a framework for training proactive agents, with the goal of minimizing false alarms. The method introduces two key features: 

(1) The probability that the user *needs* help, and that this help will be *accepted* are estimated separately and combined into a score to decide whether to act or not. Action is only allowed when the score is higher than a certain threshold.

(2) These estimations are first computed with a *fast* model. If the score is close to the threshold, then the estimations are recomputed with a *slow* model, making decisions "more thoughtful" near the decision boundaries.

The paper evaluates PRISM against proprietary and open source LLMs without specific proactive training, and against agents from the ProactiveBench paper (ICLR 2024). In the experimental evaluation, the PRISM agents show improved performance with respect to the baselines.

### Strengths
S1. Principled solution to a relevant problem: separating need and acceptance probability, and allowing more compute budget for difficult decisions makes sense and is shown to empirically work. It also makes the decision more easily interpretable.

S2. Ablation studies help see the contribution of each of the critical optimizations introduced by PRISM.

### Weaknesses
W1. Using LLM as judge is an important limitation, especially when measuring inherently human responses, as are the usefulness and acceptance of the agent's help.

W2. The empirical evaluation lacks a discussion on statistical significance. Some of the reported small improvements may be explained by the inherent randomness of the problem. For a clear example, see lines 300-304.

W3. Both in learning and deployment, considering a fixed cost of false alarm and false negative may be overly simplistic. In real situations, these costs are different for each scenario, and may also evolve over time. These nuances are not captured by the presented framework.

W4. The paper does not include the code and data to reproduce the experiments.

W5. The paper needs an editorial pass to polish certain presentation aspects. From more to less important:
- The citation of (Barto 2021) is wrong in two different ways. The reference in SIAM Review 6(2):423 is not by Andrew Barto, but by Volker H. Schulz. Secondly, the reference in question is a review of the book "Reinforcement Learning: An Introduction", written by Richard Sutton and Andrew Barto. I guess what the author really intended originally is to cite the book, and not the review of the book.
- It is a good practice to, whenever available, cite the peer-review version of a paper. This applies to Jimenez et al. 2023 [ICLR 2024], Lu et al. 2024 [ICLR 2024], Wang et al. 2022 [ACL 2023], Zhang et al. 2024 [EMNLP 2024].
- There are missing references (indicated as ??) in line 542
- Title of the appendix is "EDeriving" instead of "Deriving" (line 540)
- Missing space after period in line 206.

### Questions
Q1. In Table 3 you show the ablation on slow-mode triggering. To my understanding, the expected trend should be that the more the slow-mode is used, the better the performance. However, this is not the case. Can you explain why?

Q2. This is a request rather than a question: in lines 072 - 073, you mention that you will release the code and protocols upon acceptance through an anonymous repository. Once accepted, there is no need (or point) for anonymity, but an anonymous repository is the perfect tool to ensure reproducibility during the review process. Please release them now so they can be part of the review process.

### Soundness
2

### Presentation
3

### Contribution
3
