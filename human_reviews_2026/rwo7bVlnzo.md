# Let's Think in Two Steps: Mitigating Agreement Bias in MLLMs with Self-Grounded Verification

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 8, 4

## Abstract
Verifiers—functions assigning rewards to agent behavior—have been key to AI progress in domains such as math, code, and games. However, extending these gains to domains without clear-cut success criteria (e.g., computer use) remains a challenge: while humans can recognize desired outcomes, translating this intuition into scalable rules is nontrivial. Multimodal LLMs (MLLMs) emerge as a promising solution, given vast world knowledge, human-preference alignment, and reasoning capabilities. We evaluate MLLMs as verifiers across web navigation, computer use, and robotics, spanning 13+ model families, 28+ evaluation templates, curated trajectories from diverse agents and of varying lengths, and distinct verifier applications. We identify a critical limitation: a strong tendency for MLLMs to over-validate agent behavior—a phenomenon we term agreement bias. This bias is pervasive across models, resilient to test-time scaling, and can harm methods relying on MLLM evaluations, such as filtered behavior cloning and self-improvement. We provide guidance on the design and evaluation of MLLM verifiers, and introduce Self-Grounded Verification (SGV), a lightweight method that harnesses MLLMs' own sampling mechanisms by modulating (un)conditional generation to better leverage their knowledge, alignment, and reasoning. SGV operates in two steps: first, the MLLM is elicited to generate broad priors about desired behavior, independent of the data under evaluation. Then, conditioned on self-generated priors, it reasons over and evaluates a candidate trajectory. Our methods yield gains across models and environments, improving failure detection by up to 25pp and accuracy by 14pp, with benefits extending to downstream applications. In self-improvement and online supervision, SGV boosts task completion of a GUI specialist in OSWorld, a diffusion policy in robomimic, and a ReAct agent in VisualWebArena—setting a new state of the art, surpassing the previous best by 20pp. Finally, we release an updated version of VisualWebArena featuring strong agent baselines, more human-aligned evaluators, high-fidelity environment parallelism, runtime speedups exceeding 10x, and VisualWebArena-Lite, a 1/3-scale subset with comparable evaluation fidelity. Our code, models, and data are publicly available at [our project page](https://mshalimay.github.io/agreement-bias-sgv/).

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work investigates the issue of agreement bias in vLLMs when used as evaluators. Specifically, vLLMs are employed to assess intermediate steps in multimodal agent tasks, serving as a form of reward or signal that can facilitate test-time scaling. However, the authors find that vLLMs may generate chains of thought to rationalize potentially flawed actions, even when their evaluative knowledge is aligned with human judgment. To mitigate this issue, the authors propose having the vLLM first generate a plausible prediction of the next state before conducting evaluation, which effectively reduces agreement bias. The problem is studied across domains such as web agents, GUI agents, and robotics.

### Strengths
The research question is both interesting and important, though the phenomenon of agreement bias and the proposed solution might appear somewhat intuitive or straightforward. The experiments are comprehensive and the results are solid. The paper is also very well written — I appreciate that the abstract clearly conveys the main takeaways.

### Weaknesses
A critical problem remains: I’m particularly curious whether you have any further experiments and analysis on the “generating chains of thought to rationalize flawed behavior” aspect. You also claim that MLLMs exhibit strong, human-aligned priors on desired behavior — so at which reasoning step exactly does the failure occur? Is this bias intrinsic to the model itself, and beyond prompting strategies, are there training or sampling techniques that could help mitigate this issue?

### Questions
Please check the weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper 1) identifies "agreement bias", a tendency for MLLMs to inappropriately favor agent trajectories in their context window, as a critical limitation for MLLM-based verifiers, and 2) proposes Self-Grounded Verification (SGV), a two-step method that retrieves task priors first then evaluates trajectories against them, achieving up to 20pp gains in verification accuracy and setting a new SOTA on VisualWebArena.

### Strengths
1. Originality: Framing "agreement bias" as a distinct limitation from self-bias and targeting it via self-generated priors is a novel angle for MLLM verifiers.
2. Quality: Experiments use diverse benchmarks (1,200+ tasks) and models, ensuring results are generalizable rather than model-specific.
3. Clarity: The SGV method is described simply, with step-by-step breakdowns and concrete examples (e.g., Figure 5) making it easy to follow.
4. Significance: Improving MLLM verifier reliability directly benefits downstream tasks like agent training, data filtering, and real-time supervision, a key for deploying AI agents safely.

### Weaknesses
1. SGV does not address underlying vision-language flaws (e.g., Figure 7’s counting error), and the paper lacks discussion on combining SGV with specialist models for fine-grained perception.
2. Current studies primarily focus on moderate-length trajectories (e.g., "We set the maximum number of steps to 30"). However, the scalability of SGV to extremely long sequences remains unclear. Such sequences are common in computer usage scenarios, and the context window pressure under extremely long sequences may cause biases to reemerge.
3. The ablation on SGV’s prompt design (Appendix B.6) is limited. more tests on prior generation diversity (e.g., multiple priors vs. single) would strengthen claims about SGV’s mechanism.

### Questions
Address the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper identifies “agreement bias,” i.e., a preference of verifiers to return positive results, as a key issue for agent tasks. They then propose a simple method in which a verifier first proposes a task-specific rubric and scores trajectories according to this rubric in order to mitigate agreement bias. Finally, the paper shows that these improved verifiers can be used to improve model performance on VisualWebArena, OSWorld, and the robotic manipulation task robomimic.

### Strengths
1. This paper proposes a straightforward extension of the idea in Pan, et al. (2024) which shows that automatic evaluators can be used to improve the performance of web navigation and device control agents at training or inference time

2. The paper makes a compelling case that models exhibit agreement bias when evaluating agent trajectories, i.e., a bias toward positive labels (Table 1a). The paper also clearly shows that its method leads to a reduction in agreement bias (Table 1b), and that this method works across a wide range of verifier models

3. Most importantly, the paper shows that stronger verifiers are more useful, e.g., at improving agents at inference time using methods like Reflexion (Figure 2)

### Weaknesses
1. It would be nice to see some comparison of the proposed method with simpler strategies, e.g., different prompts to the verifier model, or prompting models to generate confidences and applying Platt scaling

2. The paper could benefit from an additional round of proofreading. For example:
     
   - Line 101: missing a period
   - Line 221: “Table Table 7” -> “Table 7”
   - Line 263: broken reference (“??”)
   - Line 1234: “AgentRewarBench” -> “AgentRewardBench”
   - \citet should be replaced with \citep in many places

### Questions
N/A

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
3

### Summary
This paper identifies "agreement bias" as a critical failure mode in Multimodal Large Language Models (MLLMs) when they are used as verifiers for agent trajectories. The authors find that MLLMs tend to favorably evaluate flawed agent behavior, even generating rationalizations for it, despite possessing strong, human-aligned priors on correct task execution. They attribute this to a retrieval bottleneck. To address this, the paper introduces Self-Grounded Verification (SGV), a two-step prompting method. SGV first elicits the MLLM's broad priors about task completion independent of the agent's trajectory, and then conditions the MLLM on these self-generated priors to evaluate the candidate trajectory. Experiments across web navigation, computer use, and robotics benchmarks show that SGV significantly improves failure detection and accuracy, boosting the performance of agents in online supervision settings.

### Strengths
1. This work identifies a significant and practical problem, the "agreement bias" of MLLM verifiers. This is an important contribution as these verifiers are increasingly proposed for data filtering, self-refinement, and online agent guidance.
2. The paper demonstrates strong empirical results, particularly in improving the True Negative Rate (failure detection). This is a crucial metric that is more informative than overall accuracy for this problem, as the primary goal of a verifier is to catch flawed behavior.
3. The method is validated across a diverse and challenging set of multimodal environments (VisualWebArena, OSWorld, robomimic) and application settings (offline evaluation, self-refinement, and online supervision), which strengthens the generality of the claims.

### Weaknesses
1. The core mechanism of SGV—generating "broad priors" in Step 1 independent of the agent's trajectory —may be a significant flaw. By being ungrounded from the specific context of the agent's current state, these priors may be overly generic or common sense hallucinations that are irrelevant to the task at hand. This could lead the verifier to be "overly strict," unfairly penalizing valid or creative solutions that deviate from the generic script, a failure mode the authors acknowledge. The paper lacks a sufficient analysis of when these broad priors are helpful versus when they are harmful.
2. The proposed solution is a heuristic-driven prompting technique. While it shows good results, it is not clear how the method will scale or interact with future model development. The paper claims the issue is a "retrieval bottleneck", but it is equally plausible that agreement bias is an artifact of current alignment techniques or context-window management. The paper does not provide evidence to suggest whether SGV is a durable solution or a temporary patch for a flaw that might be solved more fundamentally by future models, rendering the heuristic obsolete.
3. The evaluation of the online supervision setting seems arbitrary in its implementation. For instance, in OSWorld, the verifier is called "every 5 steps". There is no justification for this hyperparameter, and it glosses over the significant trade-off between verification frequency (and thus, token/compute cost) and the ability to catch errors in real-time.

### Questions
How does the performance benefit of SGV change with model scale and capability? The paper shows it helps both weaker and stronger "reasoning" models, but does the relative gain (SGV vs. baseline) shrink as models become more capable? This would provide insight into whether SGV is fixing a fundamental reasoning flaw or a specific weakness of current models.

### Soundness
2

### Presentation
3

### Contribution
2
