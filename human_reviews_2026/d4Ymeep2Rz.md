# Beyond Worst-Case: Efficient Robust RL via In-Context Generalization

- Avg Score: 5.00
- Decision: Reject
- Scores: 2, 6, 6, 6

## Abstract
Robustness and generalization are central challenges in reinforcement learning (RL). Classical robust RL handles perturbations with worst-case minimax optimization, which is hard to solve in practice and often yields pessimistic policies with suboptimal performance during deployment. In contrast, in-context RL  (ICRL), where pretrained transformers adapt to new tasks without parameter updates, is designed for generalization.
Rather than treating robustness and generalization as orthogonal, we demonstrate an interesting link that robustness can emerge as a consequence of generalization in ICRL, and that generalization can be systematically leveraged to improve robustness. Specifically, we apply ICRL models for robust RL and observe that even without explicit robust training, in-context models perform strongly on robustness benchmarks. 
Motivated by this, we investigate the robustness of ICRL models and identify significant performance degrade under disturbances. To address this limitation, building on the insight that we can transform robustness challenges to generalization problems, we propose in-context adversarial pretraining, which augments pretraining tasks with environment perturbations to expand diversity without solving the minimax game. We also introduce an adaptive rollout variant that uses the pretrained transformer to generate high-value trajectories online, improving coverage and sample efficiency. 
Across navigation and continuous-control benchmarks, these strategies consistently improve robustness and nominal return, offering a scalable path to robust performance without worst-case optimization or significant performance sacrifice.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
In this work, the authors try to link robustness and generalization in in-context RL (ICRL). The authors propose a training procedure that augments the pretraining dataset with action labels obtained from robust policies, trained on diverse environment disturbances, so the transformer learns to generalize across tasks and perturbations.  The authors being by explaining the general idea of solving first a minmax game to obtain robust policies, and then use these robust policies to obtain action labels that are used as targets for training a DPT-like model. The authors then propose a series of relaxation to improve computational efficiency. They show   numerical results across different environments,  Dark Room, Meta-World, and MuJoCo tasks.

### Strengths
- I appreciate the efforts of the author to study the link between in-context learning and robustness
- The idea of reframing robust training as data/task augmentation over adversaries, avoiding possible computational burdens, can improve computational efficiency.
- Numerical evidence is provided on a range of environments, comparing to  DT-family and robust-RL baselines.

### Weaknesses
At the moment, there are 2 main weaknesses for me in the manuscript. First, the readability is quite low. I found it particularly slow and difficult to read the paper. There is redundant content, and the main ideas of the paper should be explained more clearly. Secondly, I am not convinced by the modeling, numerical results and the lack of theoretical results showing the robustness properties of the policy learned by the authors.

Regarding the presentation, the authors use a narrative style that probably hurts more than it helps. For example, the method part starts by explaining the general idea of solving first a minmax game to obtain robust policies, and then use these robust policies to obtain action labels to be used as targets train a DPT-like model. Then they try to alleviate the computational burden of the method by proposing a series of changes, and describe these changes in the subsequent 2 sections. However, the presentation is quite verbose, and mostly provided in textual form, making it hard to read, and by the time I reached the final part, I was confused about what the algorithm is really doing.

Here's a list of weaknesses


- I believe sentences of this type“By contrast, in-context RL (ICRL) has recently attracted considerable attention for its ability to generalize to novel tasks. In ICRL, we pretrain transformer models (TMs) on a diverse suite of RL tasks and then deploy them to novel ones.” are  a bit misleading. To my understanding, ICRL is a form of meta-learning, based on training a model on a prior set of tasks. Prior literature, to the best of my knowledge, does not necessarily show the capability of generalizing to task that do not belong to this prior set.
-	The introduction is quite verbose, and a reader gets lost easily. After reading it I was left more confused. The authors  try to motivate/describe what they do, but it cannot be precise because they haven’t introduced the proper notation/methodology.
-	Line 135: in in-context learning, where the task is drawn from a prior over tasks, the expected cumulative reward is averaged over the prior. The agent usually has no knowledge of the task faced. In eq. 1  the agent is conditioned on data from the data generated from $tau$, but not on $tau$ itself.
-	Some assumptions are missing. What assumptions do we have on the set of tasks $\mathcal{M}$ and on the prior $p_\tau$.
-	Line 180: it is a bit confusing to denote both policies by $\pi$, although the adversary’s policy is parametrized by $\phi$ (but the agent policy is not?)
-	I’m not convinced that solving eq. 3 solves for a “robust policy” as the authors state. It is indeed minimax with respect to opponents playing this Markov game, but then the kind of policy we get depends on the range of things the adversary is allowed to do. Hence, what capabilities does the adversary have? Does this framework capture problems where task-parameter are allowed to vary? (e.g., the gravity parameter in robotic tasks, friction coefficients, etc.). How is this related to robustness in the classical sense that “performance should not vary across a range of parameters”? Why should we assume that the adversary has the same information as the agent, and not more? (Or less?)
-	Section 3.2: here the authors introduce the concept of training robust policies for each task in the set of tasks so that they can then train another policy, in a supervised manner, by using as targets the robust actions predicted the policy trained in the first step. However, in this two-steps scenario I would expect to split the dataset of tasks between the first and the second phases to improve generalization. Moreover, it is also not clear why by doing so we obtain a policy that is robust. What is the theoretical grounding? (as a comparison, imitation learning in RL is not necessarily the best thing to do).
-	Section 4.1: the same concerns mentioned in the previous point also apply here. The authors however do not solve a min-max problem anymore, but fix a prior over adversaries. This seems effectively like domain randomization, but a policy here is trained on a (environment, adversary)-pair, so it’s not even averaged over the prior of adversaries, making it not scalable (if we have M environments and K adversaries per environment, we need to train MK policies). Why not apply directly a domain randomization approach?
-	Section 4.2: At the current state, the section is hardly readable, and I did not understand what the authors are trying to do. This was possibly the least clear part of the manuscript for me.
-	Formal theoretical results are missing. Informal ones are provided in the form of text. Moreover, the author state:
  - “Our first results reveal that well-pretrained ICAG models perform implicit Posterior Sampling “ This is known from prior literature that in-context learning is performing a sort of posterior sampling, the result is not  novel.
  - “By implicitly performing PS during deployment over both the deployment environment and potential adversarial perturbations, transformer models pretrained by ICAG are able to act optimally in the presence of an adversary capable of modifying the environment.” This is not clear that ICAG acts optimally. It’s also not clear why it should be robust. How is this proved? And does an adversary that modifies an environment fit the Markov game model proposed by the authors?
-	Regarding the numerical results there are some issues:
  - A comparison with CVaR-PPO is missing. Furthemore, why is M2TD3 not used also in Mujoco tasks?
  - The plots at the moment are barely readable
  - It’s not clear how robustness is showed from the results. For example, the authors could show how the performance varies for different parameters. Ideally, this performance should remain stable. clearly show how the performance varies

### Questions
- "deployment parity” is not well defined
-	Line 149, the terms offline/online are a bit confusing, especially if we compare to offline/online RL
-	Line 250: what is meant by parameter set here?
-	What is concretely the space $\Phi$

### Soundness
1

### Presentation
1

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
This paper studies robustness in in-context RL and proposes ICAG/ICAA as scalable alternatives to explicit max–min robust training.
The idea is to turn robustness into a generalization problem: instead of labeling with robust actions from a per-task max–min solve (IC2RL), the authors augment each task with K adversarial/perturbed variants and label with the optimal action in each variant, then pretrain a transformer that learns to generalize across these disturbances.
Concretely, the adversary set is constructed per benchmark: in Dark Room, robustness is probed by action overrides at probability p and by switching among several OOD priors; in Meta-World, robustness is tested under state disturbances; in MuJoCo, the authors instantiate K fixed adversary policies, each a distinct neural network, and train SAC to convergence under each adversary to collect trajectories; yielding variation environments $(\tau_i,\phi_{i,k})$ used for supervised pretraining; robustness is then evaluated both against adversarial disturbances and under environmental parameter shifts (mass/friction).
Across navigation, manipulation, and continuous control, ICAG/ICAA report higher nominal and disturbed performance than strong ICRL and robust-RL baselines, while avoiding explicit worst-case optimization.

### Strengths
* Conceptual reframing: Treating robustness as in-context generalization is elegant and practical; ICAG avoids the computational and pessimism issues of explicit max–min solves. 
* Experiments: Results span Dark Room, Meta-World, and MuJoCo; ICAG/ICAA consistently outperform DPT/DT variants and are competitive with or better than robust-RL baselines (RARL, QARL) in disturbed settings. 
* Positioning vs. robust DT: The paper situates itself against ARDT, a worst-case-aware DT that learns minimax returns via expectile regression; this contextualization clarifies the difference between learning worst-case labels (ARDT) and generalizing across adversaries (ICAG). 

* No max–min in label generation + parallelizability. ICAG labels come from per-variant optimal policies $\pi^*_{\tau,\phi}$ (Eq. 5), not a minimax solve (except for MuJoCo); training these $K$ variants is embarrassingly parallel, which is a pragmatic advantage over robust max–min pipelines. 

* "Null adversary" design prevents over-pessimism. By including $\phi_0$ (no disturbance) in $\Phi(\tau)$, the augmented space $M_v$ strictly contains the original task space $M$, so robustness training doubles as principled data augmentation without sacrificing nominal performance, which is the main problem in robust reinforcement learning. 

* The method connects ICRL pretraining to posterior sampling  and gives a monotonic improvement result for ICAA, grounding the “generalize-via-variation” design in a principled lens familiar from DPT/ICRL theory.  

* The construction naturally covers agent disturbance (fixed adversaries for worst-case testing) and environmental change (mass/friction grids), aligning the training data recipe with the evaluation metrics

### Weaknesses
* IC2RL label quality not evaluated: While it’s clear in principle that IC2RL’s performance is bounded by the quality of the robust policy used to label, there is no ablation varying robust-policy quality (e.g., early-stopped RARL vs well-converged, or RARL vs stronger solvers) to measure downstream IC2RL sensitivity. 
* MuJoCo baselines miss an important robust method: The MuJoCo section compares against SAC/RARL/QARL and DT baselines, but does not include M2TD3  even though M2TD3 is used elsewhere in the paper and in recent robust-RL benchmarks; this weakens claims on continuous-control robustness. 
* Main text not self-contained: Important setup and full results are pushed to appendices (C–E), making it harder to assess choices like adversary architectures, K, compute budget, and per-task outcomes directly from the main body. 
* Compute cost for MuJoCo data: Even without explicit max–min, training SAC to convergence for each of K adversaries per task is expensive (and scales linearly with K×tasks), which may limit practicality in larger suites; the paper would benefit from clearer accounting of this cost.

### Questions
1. IC2RL sensitivity to labeler quality: Can you report an ablation that varies the robust labeler’s quality (e.g., early vs late RARL/QARL checkpoints, or alternative robust solvers) and measures the downstream IC2RL performance? This would directly test the hypothesized dependence. 
2. Why no M2TD3 on MuJoCo? You include M2TD3 in Dark Room/Meta-World figures but omit it for MuJoCo locomotion/adversarial results, where it is a strong and less over-conservative robust baseline. Could you add M2TD3 (e.g., via RRLS-style uncertainty sets) to your MuJoCo comparisons? 
3. Adversary-set design details: How should practitioners choose K and the adversary architectures (capacity, action spaces, force bounds) across tasks? Do results saturate with K (e.g., 4/8/16), and what is the failure mode if adversaries are too weak/strong? Some guidance beyond the current appendix pointers would help. 
4. Could you provide wall-clock/time-per-task and total pretraining cost for SAC×K adversary training and dataset generation, alongside baselines (RARL/QARL, ARDT)? This would clarify “efficiency” claims. 
5. Worst-case metric definition. For MuJoCo, are “with disturbance” scores a min over adversary policies or an average? A per-task table reporting mean and worst-case (min) across fixed adversaries would make the robustness metric explicit.

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
This paper proposes a novel perspective on robust reinforcement learning by linking robustness to generalization in in-context reinforcement learning (ICRL). Rather than relying on computationally expensive worst-case minimax optimization, the authors demonstrate that robustness can emerge naturally from ICRL's generalization capabilities. The paper makes two key observations: (1) pretrained ICRL models (without explicit robustness training) already outperform dedicated robust RL baselines in classical robust RL settings, yet (2) these same models still exhibit significant performance degradation under deployment disturbances. To address this gap, the authors propose two complementary methods: In-Context Adversarial Generalization (ICAG), which augments pretraining with adversarial task variations to enable the transformer to generalize across disturbances without solving max-min problems, and In-Context Adversarial Adaptation (ICAA), which efficiently bootstraps high-quality action labels for continued pretraining using online deployment trajectories. Theoretical analysis shows that ICAG implicitly performs Posterior Sampling over both environments and adversaries (Theorem F.3), while ICAA provably improves action label quality across iterations (Theorem F.5). Empirical evaluation across sparse-reward navigation (Dark Room), manipulation (Meta-World), and continuous control (MuJoCo) demonstrates consistent improvements in both robustness and nominal performance compared to robust RL baselines and competing ICRL methods.

### Strengths
**Novel and well-motivated link between robustness and generalization.** The paper articulates a compelling insight that robustness can emerge from generalization rather than requiring explicit robust optimization. The observation in Figure 1 that ICRL methods substantially outperform dedicated robust RL methods (Domain Randomization, M2TD3, RARL, QARL) on classical robust RL benchmarks is striking and non-obvious, motivating the entire research direction.

**Principled theoretical analysis.** Theorems F.3 and F.5 provide formal guarantees connecting ICAG to Posterior Sampling and characterizing ICAA's iterative improvement. The connection to PS is particularly valuable, as PS is known to be sample-efficient and provides a theoretical justification for ICAG's effectiveness beyond heuristic intuition. The theoretical framework grounding the approach enhances credibility.

**Thoughtful treatment of deployment requirements and limitations.** Section 7 candidly acknowledges that ICRL requires context trajectories at deployment, which may be impractical in data-scarce safety-critical domains, and suggests when classical robust RL remains preferable.

### Weaknesses
**Limited novelty of core components.** While the link between robustness and generalization is novel, the individual techniques are not: augmenting environments with perturbations is standard data augmentation, and leveraging pretrained models to generate action labels for continued pretraining is established practice. The contribution is more one of combination and insight than technical innovation.

**Restrictive theoretical assumptions.** The theoretical analysis in Appendix G imposes strong conditions: (1) the joint distribution in Equation 6 assumes optimal action labels are available for each variation environment which is unrealistic in practice, (2) Assumptions F.1-F.4 in Appendix F require conditions like invertible Hessians and strict local maxima that may not hold for neural network policies. The gap between these assumptions and practical deep RL is significant, limiting theoretical guarantees' applicability.

**Disconnect between theory and practice.** ICAG pretraining uses the ideal supervised objective in Equation 1, but in practice, obtaining optimal action labels for every variation environment (τ, ϕ) via training policies to convergence is computationally expensive. While ICAA partially addresses this by using online-collected labels, the method introduces additional approximation that weakens theoretical guarantees. The practical pretraining cost is not thoroughly analyzed.

### Questions
**Q1:** What explains the remaining performance gap between ICAA and ICAG? Since ICAA uses non-optimal action labels, how much performance is sacrificed? Can tighter upper bounds be established?

**Q2:** How do ICAG and ICAA perform in sparse-action environments or with discrete action spaces? Experiments focus on continuous control; scalability to discrete domains is unclear.

### Soundness
3

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
This paper exposes an interesting connection between robustness and generalization in in-context reinforcement learning (ICRL) that these two qualities are not orthogonal. In the first experiment, the authors show that ICRL without robustness objective can outperform classical robust RL baselines, indicating that ICRL is a preferable method since it avoids performance sacrifices---a typical challenge in robust RL.

### Strengths
The empirical observation of the connection between generalization and robustness is interesting. The experiments are conducted thoroughly. The proposed method is also simple, which is easy to apply.

### Weaknesses
1. **Overclaims**: In Fig 1(b), the authors draw the conclusion of ICRL better than classical RL. The authors should demonstrate these with more environments, not only just Dark Room. Otherwise, the claim that ICRL methods substantially outperform robust-RL baselines without robustness objective is overly strong. To me, addressing the robustness of ICRL is already a good motivation, even without claiming ICRL is better than classical robust RL in the first place.

2. **Presentation**: I think the authors should put the pseudo-code (short version) in the main text to help understand your algorithm. The texts from 316 to 330 are really hard to follow. The removable/reducible part to me is the theoretical analysis in 4.2 and some of the analysis in 4.1.

### Questions
Q: This question could be due to the unclear presentation, but from my understanding ICAA requires extensive generations from ICRL model. Will this computational cost be concerning?

### Soundness
2

### Presentation
2

### Contribution
3
