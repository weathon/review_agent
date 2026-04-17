# Scalable Supervising Software Agents with Patch Reasoner

- Decision: Reject
- Scores: 6, 8, 6, 6

## Abstract
While large language model agents have advanced software engineering tasks, the unscalable nature of existing test-based supervision is limiting the potential improvement of data scaling. The reason is twofold: (1) building and running test sandbox is rather heavy and fragile, and (2) data with high-coverage tests is naturally rare and threatened by test hacking via edge cases. In this paper, we propose R4P, a patch verifier model to provide scalable rewards for training and testing SWE agents via reasoning. We consider that patch verification is fundamentally a reasoning task, mirroring how human repository maintainers review patches without writing and running new reproduction tests. To obtain sufficient reference and reduce the risk of reward hacking, R4P uses a group-wise objective for RL training, enabling it to verify multiple patches against each other's modification and gain a dense reward for stable training. R4P achieves 72.2\% Acc. for verifying patches from SWE-bench-verified, surpassing OpenAI o3. To demonstrate R4P's practicality, we design and train a lite scaffold, Mini-SE, with pure reinforcement learning where all rewards are derived from R4P. As a result, Mini-SE achieves 26.2\% Pass@1 on SWE-bench-verified, showing a 10.0\% improvement over the original Qwen3-32B.
This can be further improved to 33.8\% with R4P for test-time scaling. The stable scaling curves in both RL test rewards and test-time accuracy reflect R4P's practical utility for scalable supervision on software agents.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper studies an interesting problem of learning "execution free patch verifiers" for SWE agents. Currnent test time scaling agents fall into 3 categories: execution-free verifiers, execution-based verifiers and hybrid verifiers. Current execution-free verifiers require access to agent trajectory which can bias the final response of the patch verifier. The paper simply considers learning better "exec ution free patch verifiers" as a reasoning problem which can be solved with RL. The proposed approac R4P achieves 72% "verification accuracy" while also help improve TTS (test time scaling) performance by ~10% on SWE-Bench-Verified. (Its important to note that 72% is verification accuracy, not accuracy after TTS)

### Strengths
* The idea of using RL for training patch verifiers is interesting.

* "the binary outcome reward is very easy to hack, making the training unstable." This is a very insightful observation especially for training patch verification agents with RL.

* The final results show improvements for patch verification and TTS performance.

### Weaknesses
* The paper mentions that "The Pass@1 resolution rate on SWE-bench-verified steadily improves with more training data and finally reaches 26.2%, outperforming Lingma Agent +
Lingma SWE-GPT-72B (Ma et al., 2024)." Have the authors also tried their approach on closed source models like Claude 4.5 Sonnet and best open source models like R2E-Agent, SWE-Smith, DeepSWE, etc.?

* The paper considers group verification for training the RL model (sec. 3). Have the authors tried training a patch verifier using RL for individual patches given the input problem statement and repository sandbox?

* The authors mention that for training they use "2,438 issue instances". Have the authors explored impact of number of issues on the performance of the patch verifier? Is 2500 issues enough for RL training?


* For fig. 2, I believe r2e-gym also has an execution based verifier. I will be curious on how the proposed verifier compares to execution-based verifiers?

* Also for Tab. 1, are all values reported for execution-free verification given just the problem statement and final patch?

* Using RL for training the patch verifier is interesting. Can the authors please also share some outputs from the patch verifier to help understand how RL shapes the reasoning process as compared to closed-source patch verifiers like o3, gpt-5 etc?

* Finally, while not a major concern, have the authors tried comparing R4P with recent pretrained closed-source patch verifiers like Claude 4.5 Sonnet, gpt-5 etc?

### Questions
Please see the weaknesses section for some additional questions.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces R4P (Reasoning-for-Patch) — a scalable, test-free reward model for supervising software engineering (SWE) agents via reasoning rather than traditional test execution.
It addresses the scalability bottleneck of test-based verification, which is heavy, fragile, and limited by test coverage.
R4P formulates patch verification as a group-wise reasoning task, comparing multiple candidate patches to produce dense, stable rewards during reinforcement learning (RL).
Experiments show that R4P achieves 72.2% patch verification accuracy, surpassing proprietary models like OpenAI o3. The authors further train a lightweight agent, Mini-SE, purely under R4P supervision, which achieves 26.2% Pass@1 on SWE-bench-verified—+10% over Qwen3-32B—and 33.8% when combined with R4P at test time.
The paper argues that R4P enables scalable supervision for SWE agents without dependence on sandbox testing

### Strengths
- The group-wise reasoning objective transforms sparse binary verification into a dense, stable reward signal
- The paper provides ample evidence showing the advantage of R4P, along with nice ablation studies to analyze the behavior of R4P model

### Weaknesses
- The reward model is fixed post-training, leading to potential reward drift as agents improve. It will be interesting to understand the RL behavior when you overtrain the model with such a static reward model model

- In Fig. 9, it will be good to draw the confidence interval to see if the trend is significant. The bins to the right have too few samples, which makes the conclusion that "verification accuracy positively correlates with/ number of edited lines" a bit ungrounded

- Despite the two challenges of applying R4P directly to existing agent scaffolds via RL, it'd be interesting to demonstrate R4P's ability to provide supervision for training models to work on general agent scaffolds. e.g., you can use R4P to re-rank patches/trajectories generated on training datasets like SWE-Gym+OpenHands and SFT on the top 10% trajectories and measure performance improvements on OpenHands vs. random sampling. This could demonstrate R4P's ability to generalize across scaffolds.

### Questions
> As the agent’s policy improves, the static reward model may become misaligned with true answer quality 
I wonder whether the authors have tried to overtrain the policy (i.e., training the model longer in Figure 3a). I'd be interested in understanding the R4P approach's bottleneck, e.g., whether it will saturate at a fixed performance on SWE-Bench or degrade performance if overtrained.

- I would be helpful if the authors could share more details about how the Acc/F1/EM was calculated, as well as the exact reward function that was used to perform RL on Mini-SE LM.

### Soundness
2

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
This work proposes a reasoning-based framework for supervising large language model (LLM) agents in software engineering tasks without relying on computationally expensive or fragile testing environments.
To overcome these challenges, the authors introduce R4P, a reasoning reward model that performs group-wise patch verification—evaluating multiple code patches for a given software issue to determine correctness via reasoning rather than execution. This design produces dense, stable rewards and mitigates reward hacking. Built upon Qwen2.5-Coder-32B-Instruct, R4P achieves 72.2% accuracy on the SWE-bench-verified dataset, outperforming OpenAI’s o3 model.

### Strengths
1. Innovative Test-Free Supervision Paradigm.
R4P redefines software agent supervision as a reasoning task, eliminating the dependency on sandbox testing. This shift addresses scalability, cost, and fragility in existing solutions.
2. The reward model design is novel.

### Weaknesses
1. Scope of this work can be better elaborated.
2. The evaluation can be more comprehensive.

My major concen of this work is clarity and evaluation, I believe these shortcomings can be overcame before submitting the camera-ready version.

Why the reward design is technically sound and how it affects the learning? I think this is important when designing a reward function for reinforcement learning, and maybe it is better to elaborate that it is aligned with your objective to avoid reward hacking.
Equation (3) can be better explained, and similar problem happened in other places, please define and explain each symbol carefully.

Maybe it is better to include some real data if possible, the current tests are completely on sythetic data.
I'm curious about the results if the correctness ratio is imbalanced, seems this is more natural in real-world problems.

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
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes to automatically evaluate software repository patches proposed by SWE agents via a verifier model that reasons explicitly about the proposed patches. They propose this as an explicit alternative to simple reward modeling that provides a sparse scalar estimate of quality of proposed patches. The proposed model, R4P, can be used to evaluate SWE agents and to train them in cases where a repository doesn't have ground-truth test cases. Experiments focus on evaluating R4P's quality as a verifier, and on the application of R4P for training SWE agents via RL and improving performance via test-time scaling. As far as I can tell, the main contribution is that R4P is trained to not only provide judgments over patch quality, but to also provide reasoning about its judgments. Augmenting models with reasoning capabilities has been a popular approach for improving model outcomes in difficult tasks, so it makes sense that it works well here too.

### Strengths
* The analysis in 5.3 is really comprehensive
* Experiments prove the efficacy of R4P on a simple agent scaffold, and strong performance as a verifier when compared with existing non-reasoning verifiers

### Weaknesses
* Figure 3 shows rewards for test data. Test data should be used in experiments very sparingly, to avoid compromising integrity of conclusions about model generalization
* Evaluation is only performed on the mini-SE scaffold, rather than other scaffolds that achieve stronger base performance on SWE-bench-verified. Would R4P still be useful in these other scaffolds? Do these scaffolds make R4P more difficult because they include much longer trajectories, which are more difficult to reason about (although R4P is trained on OpenHands trajectories, so it should be in-distribution to apply it to the OpenHands scaffold)?

### Questions
* What is the difference between DeepSWE-Verifier and DeepSWE-Test?

### Soundness
3

### Presentation
3

### Contribution
3
