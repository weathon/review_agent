# Guided Flow Policy: Learning from High-Value Actions in Offline Reinforcement Learning

- Decision: Accept (Poster)
- Scores: 6, 2, 6, 6

## Abstract
Offline reinforcement learning often relies on behavior regularization that enforces policies to remain close to the dataset distribution.  However, such approaches fail to distinguish between high-value and low-value actions in their regularization components. We introduce Guided Flow Policy (GFP), which couples a multi-step flow-matching policy with a distilled one-step actor. The actor directs the flow policy through weighted behavior cloning to focus on cloning high-value actions from the dataset rather than indiscriminately imitating all state-action pairs. In turn, the flow policy constrains the actor to remain aligned with the dataset's best transitions while maximizing the critic. This mutual guidance enables GFP to achieve state-of-the-art performance across 144 state and pixel-based tasks from the OGBench, Minari, and D4RL benchmarks, with substantial gains on suboptimal datasets and challenging tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes Guided Flow Policy (GFP), an offline RL algorithm that trains a flow policy. The method is built on top of the behavioral regularized actor-critic framework, and more specifically, flow Q-learning (FQL). The main differences between GFP and FQL are that (1) GFP uses a weighted BC flow policy instead of a vanilla BC policy, and (2) it uses a more conservative Q target by mixing the behavioral action and the policy action. The authors evaluate their method on a large number (129) of tasks across OGBench, Minari, and D4RL, showing that GFP outperforms FQL and other strong baselines.

### Strengths
* The paper is well-written and easy to understand.
* The empirical evaluation is notably thorough. Across more than 100 tasks, GFP consistently achieves better performance than other baselines (IQL, FQL, and ReBRAC). The authors also mainly use 8 seeds for each task.
* The paper provides several ablation studies, including analyses of the effects of conservative Q targets, weighted behavioral cloning, etc. They demonstrate the importance of these techniques.

### Weaknesses
I don't see major weaknesses in this work. Two minor ones are as follows:
* One weakness is its relatively limited novelty and contribution. Essentially, the paper can be summarized as AWR + FQL (or broadly AWR + BRAC), which is a rather straightforward extension of existing methods. While I do appreciate the simplicity of this combination, given how straightforward the method is, the paper could further be strengthened by either having more large-scale experiments (e.g., more challenging/long-horizon tasks, more realistic benchmarks (or even real robot experiments), pixel-based tasks, etc.) and/or providing additional insights beyond simply applying existing techniques to the flow RL setting.
* Another weakness is that this method requires an additional parameter ($\eta$) that needs to be tuned per task, compared to FQL. Generally, introducing more hyperparameters would naturally only make the method better, so the real question is how large the gap is compared to the closest baseline (FQL) with this extra degree of freedom. From Figure 4 and Table 2, the difference doesn't seem unreasonably small, but it would have been even more convincing if the proposed method had achieved an even larger performance boost.

### Questions
Have you tried vanilla AWR for learning a weighted BC policy? Why do we want to have the additional normalization term in the denominator (compared to AWR) in Eq. (10)?

### Soundness
3

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
4

### Summary
The paper proposes guided flow policy (GFP), an offline RL method that leverages flow-matching to train expressive policies that can tackle tasks with suboptimal datasets. The proposed method trains the base flow-matching policy  $\pi_\omega$ via weighted behavior cloning to match $\pi(a\mid s) \propto_a \pi_\beta \exp(\tau Q(s, a))$ (where $\pi_\beta$ is the behavior distribution described by the dataset). Then, it applies a recently proposed technique, FQL, on top of $\pi_\omega$ where it trains a 1-step distilled policy to stay close to the base flow policy $\pi_\omega$ while maximizing its value under the critic. In practice, the authors adopt a slightly different/more conservative Q-target backup by using a mixture of actions from both the more aggressive 1-step distilled policy and the more conservative multi-step base flow policy. Across multiple benchmarks (OGBench, D4RL, Minari), the proposed method outperforms FQL and Gaussian policy offline RL baselines (ReBRAC, IQL) especially on noisier offline datasets.

### Strengths
- The paper considers a comprehensive set of benchmark tasks which make the comparison between the proposed method and the baselines very convincing. I also appreciate that the authors seem to properly tune the hyperparameters for the baselines, which make the comparisons fair.

- The proposed method is presented in a concise and clear manner and the algorithmic design has no technical issues.

### Weaknesses
The paper make some factually questionable claims
- Table 1: all prior methods (IQL, TD3+BC, ReBRAC, FQL) are listed under "handles suboptimal data (x)" which implies that these methods cannot handle suboptimal data. This is misleading because most offline RL methods listed and in fact most offline RL methods in general can handle suboptimal data. It would be good to further clarify and be precise about what "handles suboptimal data (x)" actually means here.
- The authors introduce Value-aware behavior cloning (VaBC) as a novelty in the algorithm and claim that "this is the first integration of value-aware behavior cloning", but as how it is currently presented it seems to be me that it is exactly equivalent to advantage weighted regression applied to flow-matching, which has been explored in many recent flow-matching/diffusion papers (e.g., FAWAC baseline in FQL[1], energy-weighted flow matching loss in QIPO [2]).

The comparison to prior work is lacking:
- The paper only compares with a very limited number of baselines (FQL is the only baseline for expressive policies). For the similarity to prior work as mentioned above, I would encourage the authors to consider comparing to some of them such that the reviewers could be more informed about where GFP's performance stands in the literature.
- The related work section discusses very little prior work despite a rich literature in diffusion/flow-matching in reinforcement learning and it is not clear from the section how the proposed method is different from these prior work.

Small typos:
   - Table 4: in the value column, second row: "50,0000" should be "500,000"
   - L248: $Q_{\bar \phi}(s', \mu_\omega(s', z)$ is missing a right bracket. 

[1] Park, Seohong, Qiyang Li, and Sergey Levine. "Flow q-learning." arXiv preprint arXiv:2502.02538 (2025).

[2] Zhang, Shiyuan, Weitong Zhang, and Quanquan Gu. "Energy-weighted flow matching for offline reinforcement learning." arXiv preprint arXiv:2503.04975 (2025).

### Questions
Related to my concern above -- how is the proposed `value-aware behavior cloning' different from prior methods? (e.g., QIPO and FAWAC)

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a framework for performing offline RL using flow policies. Compared to prior such as FQL, which first trains a BC policy as a flow network, then uses this policy to distill a one-step policy which is then trained via Q-learning, this work also connects the Q network back to the flow network. Specifically, an AWR-style weighting is used on the loss, weighting higher-advantage actions. In essence this creates *two* feedback mechanisms for the learned Q function -- it affects the flow network (through advantage-weighting) and also the resulting one-step policy (through the DDPG loss). Experiments show that this method reliably outperforms prior work on a wide range of experiments.

### Strengths
This paper presents a clean framework with empirically sound results. The idea of using the AWR weighting to train the flow network is a neat way to achieve policy improvement wrt a learned Q function. It is useful computationally that the TD errors for Q-learning can be calculated using the distilled one-step policy, avoiding the need for an expensive ODE integration during training. 

In terms of clarity and quality, the paper reads well and is easy to follow. See weaknesses section for additional points here.

The significance of this paper is boosted by its strong empirical performance and wide evaluation. The paragraph describing how the results in this work were re-run for previous baselines, and in fact improved upon them, lends credibility to the evaluation. In addition, the scaling curves with relation to temperature in Figure 3 clearly show that the intuition for the temperature term is correct in practice.

### Weaknesses
As this framework in essence uses two methods of feedback from the Q function, the paper would greatly benefit from a clear analysis on the effects of both of these terms. For example, two clear baselines are the same method where either the flow policy is not weighted (or if this is exactly FQL, some text making this clear) and where the one-step policy does not utilize the DDPG loss. 

Building on the above, an analysis as to why the two feedback methods are complimentary would take this paper one step forwards. As of the current submission, it is shown that empirically it is useful to use both, but it is unclear on a deeper level why this might be true. The paper hints that "there is no fundamental justification for preferring one divergence or distance metric over another", and more investigation into this would be potentially fruitful.

### Questions
In the experiments, which version of the TD learning objective is used? Equation 7 shows that actions can be taken from either the distilled policy or the flow policy. Presumably taking actions from the flow policy is computationally expensive as an ODE integration is required.

In equation 6, the notation to use pi_c is somewhat confusing.

Does the optimal temperature change if the BC weighting (lambda) is adjusted, and vice-versa? What does the landscape of these hyperparameters look like, and how sensitive is the method to both of them?

It is interesting that both the one-step policy and the VaBC policy can be used for evaluation, as mentioned in table 2. What do the average scores for the VaBC policy look like?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes Guided Flow Policy (GFP), an offline RL method that couples (i) a multi-step flow-matching policy trained by value-aware behavior cloning (VaBC) and (ii) a distilled one-step actor guided by a critic. The key idea is a bidirectional guidance: the actor and critic guide the flow model to preferentially clone high-value dataset actions (rather than all actions), while the flow model regularizes the actor to remain within the support of high-value dataset transitions. A temperature-controlled guidance term g_η modulates how selective the cloning is. Extensive experiments across 129 tasks spanning OGBench, Minari, and D4RL show that GFP consistently outperforms pervious methods.

### Strengths
The method is elegant and easy to follow. It cleanly integrates flow matching into the BRAC framework: a value-aware flow policy shapes the action distribution, and a distilled one-step actor provides fast inference. The bidirectional guidance narrative is intuitive, and the overall training loop is straightforward to implement.
The empirical evaluation is thorough. The paper reports results on 129 tasks spanning OGBench, Minari, and D4RL, providing broad coverage and reducing the risk of cherry-picking. Across these diverse settings, GFP demonstrates consistently strong performance.
Finally, the paper offers a useful sensitivity analysis of the temperature η. The study clarifies how guidance sharpness trades off selectivity versus diversity, giving practitioners a clear knob to choose stable, high-performing settings.

### Weaknesses
First, the novelty is moderate. The method still fits squarely within the BRAC family: it combines an expressive flow-matching policy with value-aware cloning and then distills into a one-step actor. While the synthesis is thoughtful and practically useful, the conceptual step beyond prior behavior-regularization plus expressive policy models feels incremental.

Second, the paper’s structure could be clearer, especially in the Experiments section. Main results and analysis are interleaved, which makes it harder to extract the headline takeaways before diving into diagnostics. I recommend separating “Main Results” (per benchmark, with concise aggregates) from “Analyses & Ablations.” In particular, if my understanding is correct, Section 4.1 “Suboptimal datasets” functions primarily as a temperature sensitivity study on η; the current title is confusing. Retitling to something like “Sensitivity to Temperature on Noisy/Suboptimal Data” would improve readability.

Third, the role of distillation isn’t entirely clear in the current narrative. Table 3 reports VaBC-only results, but the training pipeline still relies on a distillation phase, which makes it hard to tease apart how much of the gain comes from the value-aware flow versus the student actor. A clean ablation that removes distillation end-to-end (i.e., deploy the flow directly), paired with a short intuition for when distillation helps would clarify its necessity and practical benefit.

### Questions
See weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3
