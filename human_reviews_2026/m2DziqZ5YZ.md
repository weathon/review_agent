# ReST-RL: Reinforcing LLM Reasoning through Self-Training and Value-Guided Decoding

- Decision: Reject
- Scores: 2, 2, 6, 6

## Abstract
With respect to improving the reasoning accuracy of LLMs, the representative reinforcement learning (RL) method — Group Relative Policy Optimization (GRPO) — has achieved critical success, yet it still suffers from the issue of insignificant reward variance. This paper introduces ReST-RL, a unified LLM RL paradigm that combines an improved GRPO algorithm with a meticulously designed test-time decoding method to improve LLM’s code reasoning ability. As the first stage of policy reinforcement, ReST-GRPO adopts an optimized ReST algorithm to increase the reward variance of GRPO sampling, thereby improving training effectiveness. Building on this foundation, we further introduce a test-time decoding optimization method, VM-MCTS, which employs an adapted Monte-Carlo Tree Search (MCTS) guided by a trained Value Model (VM) to provide precise process signals and verification scores, further enhancing LLM reasoning accuracy. We validate our RL paradigm on multiple coding benchmarks (e.g., APPS, BigCodeBench, and HumanEval), where it significantly outperforms other reinforcement training baselines (e.g., naive GRPO and ReST-DPO), as well as decoding and verification baselines (e.g., PRM-BoN and ORM-MCTS), indicating its power to strengthen LLM’s reasoning capability. We further examine ReST-RL on out-of-domain math reasoning tasks, demonstrating that ReST-RL and the VM have strong transferability and generalizability across unseen reasoning domains and policy checkpoints, confirming that it extends beyond coding. Notably, our approach achieves strong performance with limited data, showcasing its effectiveness, efficiency, and generalizability.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces ReST-RL, a unified RL paradigm that enhances LLM reasoning for code generation. It operates in two stages: ReST-GRPO, a training algorithm that increases reward variance via self-training, and VM-MCTS, a decoding method that uses a Value Model to guide the LLM. The main contributions are as follows:
1. It proposes the ReST-RL paradigm, which effectively integrates the advantages of offline self-training, online policy optimization, and value-guided decoding to achieve a balance between effectiveness, efficiency, and data cost.
2. It introduces two key technical components: the ReST-GRPO algorithm for more effective policy training and the VM-MCTS method for enhanced decoding, both of which are shown to outperform strong baselines.
3. It demonstrates significant performance gains over existing methods across multiple LLMs and challenging code benchmarks , showcasing the approach's power to strengthen LLM reasoning.

### Strengths
1. The work shows strong originality by creatively unifying offline self-training, online RL, and value-guided tree search into a novel, two-stage paradigm. This synthesis effectively addresses key limitations of prior methods, namely low reward variance and high annotation costs for process supervision.
2. The paper's quality is demonstrated through extensive, rigorous experiments across multiple code benchmarks and base LLMs. Results consistently show significant improvements over strong baselines, with convincing ablation studies and efficiency analysis, providing robust validation for the proposed method.
3. The presentation is exceptionally clear, with a well-motivated narrative and detailed algorithms. The work is significant for delivering a practical framework that balances performance, efficiency, and data cost, offering a valuable blueprint for enhancing LLM reasoning in complex domains like code generation.

### Weaknesses
1. The paper omits comparisons with recent, highly relevant state-of-the-art methods. Specifically, there is no comparison with R1-style process-based reinforcement learning or the closely related PSPO framework, which also combines process reward with policy optimization. Demonstrating superiority over these established strong baselines is crucial to fully validate the claimed contributions.
2. While the method claims efficiency, it lacks a detailed analysis of the computational overhead. The combined cost of iterative ReST-GRPO training plus the VM-MCTS decoding pipeline is likely substantial. A breakdown of training time, inference latency, and GPU memory footprint compared to baselines is needed to assess the practical trade-offs of the proposed approach.
3. The Value Model is trained and tested on distributions from the same policy and dataset. Its performance and reliability when guiding the LLM on out-of-domain problems or after significant policy updates are not investigated. This raises questions about the long-term applicability and robustness of the VM-MCTS component without frequent, costly retraining.
4. The reliance on a purely rule-based, test-case-passing reward, while avoiding reward hacking, ignores richer signals of code quality. This may limit the model's ability to learn truly generalizable coding principles. Incorporating a learned reward model, even with its risks, could be a necessary evolution for more advanced reasoning.

### Questions
1. Could you provide a concrete breakdown of the computational overhead for both ReST-GRPO training and VM-MCTS decoding compared to key baselines? This is crucial for assessing the practical deployability of your method.
2. Why were comparisons with strong, relevant baselines like R1 or PSPO omitted? Including these would more convincingly demonstrate the superiority of your proposed approach against the current state-of-the-art.
3. How does the performance of your Value Model degrade when the underlying policy is updated or when applied to out-of-domain problems? An analysis of its robustness to such distribution shifts is needed to assess long-term viability.

### Soundness
2

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
3

### Summary
The paper proposes a two-stage framework to improve code-reasoning LLMs: (1) ReST-GRPO, which filters prompts by reward variance and bootstraps training with partial-trace starting states before GRPO updates; and (2) VM-MCTS, which trains a value model with MCTS-collected targets and uses it for value-guided search/verification at test time. Experiments on HumanEval/MBPP/APPS-500/BCB suggest consistent gains over naive GRPO/ReST-DPO and over ORM/PRM-based verification, with additional curves for train-step efficiency and budgeted verification.

### Strengths
The paper investigates an interesting area and proposes a combination of existing methods. The proposed partial-state solution is particularly compelling and intuitively sound. Experimental results show that ReST-RL and VM-MCTS achieve superior performance across various coding benchmarks.

### Weaknesses
1. The paper proposes to combine existing methods, ReST, an offline self-training paradigm with GRPO, an on-policy RL algorithm (stated by the authors at line 261). I have concerns with the distribution shifting issue in the loss update. The original ReST investigates multiple offline RL methods, the ReST-MCTS* and ReST$_{EM}$ are using SFT. Given the authors' design of the partial state completion, the training data distribution is shifted comparing to the test prompt if they use common GRPO objective as they stated at line 262. I acknowledge that p.m.f at equation (3) might partially mitigate it as it gives probability that the data fall back to original GRPO. I think it worths to investigate and discuss why ReST-GRPO performs so well and how is the distribution shifting coped. It would be great to add ablation experiments for different alpha in p.m.f might give more insights.
2. VM-MCTS and ReST-GRPO seems to be distinct components. VM-MCTS can help test time computation is not novel.  In PPO-MCTS, the value function is a byproduct of the PPO training while GRPO does not have a value model. Using a smaller value network to help the LLM in the test time has been proposed by "Reasoning with Language Model is Planning with World Model" before and the comparing between VM-MCTS and ORM and PRM with Best-of-N seems ignoring other strong tree search baselines. 

Other minor issues:
1.Over-claiming the general reasoning benefits in the introduction but all experiments are code benchmark. I think either be specific about the benefit of reasoning capability in coding or add experiments with math benchmark will be more convincing.
2. The caption of the figure and table is not stand along explainable. For instance, in Table 1, what does the 1st iter and 2nd iter mean. I think it would add clarity by introduce it in the caption.

### Questions
1. Does the authors use any way to address the distribution shifting in the training set? If not, why?
2. What is the purpose of VM-MCTS? Is it just to show that it can improve the test time performance? In experiments section, does the experiments with ReST-GRPO apply the VM-MCTS in the test time?

### Soundness
2

### Presentation
2

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
The paper introduces ReST-RL, a two-stage reinforcement learning framework for improving code reasoning in large language models. The method unifies GRPO with a value-guided decoding strategy, addressing the low reward variance in GRPO and the annotation cost of process reward models:
- ReST-GRPO: Builds on GRPO and self-training by sampling multiple solutions per prompt, filtering data by reward variance and maximum reward, and assembling partial prompts from high-reward traces.
- VM-MCTS: Trains a Value Model (VM) using rollouts collected from MCTS with rule-based rewards. At inference, an adapted MCTS leverages VM predictions both as process signals and for solution verification (Best-of-N over VM scores).

Experiments on HumanEval, MBPP, APPS-500, and BigCodeBench using several 6-8B LLMs demonstrate that 1) ReST-GRPO outperforms naive GRPO and ReST-DPO in all benchmarks. 2) VM-MCTS surpasses PRM, ORM, and ORM-MCTS under identical 100-sample budgets 3) ReST-GRPO trains faster and sustains improvement over longer steps.

### Strengths
1. The framework harmonizes online RL, self-training, and MCTS-based decoding in a coherent way, showing the benefit of connecting these previously separate threads.
2. Strong empirical improvements: Both ReST-GRPO and VM-MCTS deliver measurable gains across multiple code LLMs and benchmarks. The improvements are robust under both sampling and greedy decoding.
3. Data efficiency: The entire pipeline uses ~7k prompts and no human-annotated rewards, demonstrating a favorable cost–performance trade-off.

### Weaknesses
1. Baseline mismatch for PRM/ORM: VM is trained on code with unit‑test targets; ORM/PRM baselines are generic (Skywork ORM; Qwen2.5 PRM).
2. Ablations on ReST‑GRPO design: No sensitivity analyses for $\sigma_0, r_0, \alpha, \beta, N$; unclear contributions of filtering vs. partial‑state restarts.
3. Reward‑shaping: GRPO training uses Eq. (8) (substring bonus, redundancy penalty) with non‑negligible weights; it’s not shown how much these influence outcomes, nor whether DAPO/GRPO baselines used identical shaping.

### Questions
1. How were BCB and APPS splits handled to prevent overlap between training and evaluation?
2. Could you provide results using PRM/ORM models trained on the same code dataset to ensure domain parity?
3. How sensitive is ReST-GRPO to $\sigma_0, r_0, \alpha, \beta$? Which component contributes most? variance filtering or partial-state sampling?
4. How do results change if you remove the essential‑substring and redundancy terms in Eq. (8)? Did DAPO/GRPO baselines use the exact same shaping?
5. Appendix A.8 assumes unbiased and independent estimators with equal variance. What happens empirically when the VM is biased/correlated with rewards?

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
4

### Summary
The paper proposes ReST-RL, a two-stage reinforcement learning (RL) framework to improve code reasoning in large language models (LLMs). Stage 1, ReST-GRPO, augments Group Relative Policy Optimization (GRPO) with an improved ReST-style self-training loop that filters and assembles high-reward partial solutions to increase reward variance and training efficiency. Stage 2, VM-MCTS, trains a Value Model (VM) via Monte-Carlo Tree Search (MCTS) without extra annotations, then uses the VM to guide MCTS decoding at test time. Extensive experiments on HumanEval, MBPP, APPS-500 and BigCodeBench show that ReST-RL outperforms baselines, while using limited training data.

### Strengths
- Introduces a principled combination of self-training (ReST sampling/assembly) with GRPO and a value-model guided MCTS at test time. The combination and the particular sampling/filtering heuristics (σ0, r0, exponential sampling with α) appear novel in this exact form.
- Solid empirical coverage: 6 benchmarks, 4 base code-LLMs, plus general LLMs (Llama-3, Qwen3) with both pass@1 and test-case-average metrics.
- Mathematical justification (Appendix A.8) proves that value-based MC rollouts produce lower-variance value estimates than full-trace rollouts and supports design choice.
- Algorithms are presented via clear pseudocode boxes (Alg. 1–3).
- Demonstrates that nontrivial improvements are achievable with few training prompts, suggesting data-efficient RL for code is possible.
- Provides a plug-in decoding module (VM-MCTS) that can be applied on top of any base policy without re-training.

### Weaknesses
- Missing statistical significance tests. Reported gains are single-run (no std-dev). With pass@1 often changing by ≤3 points between iterations (Table 1), observed differences may be within random noise. For RL-style training this is critical.
- The paper gives some VM-MCTS hyperparameters (T, n, c) for experiments but lacks a centralized hyperparameter table and justification of choices.
- Value model generalization unclear: VM is trained on the same coding prompts; no out-of-domain (OOD) evaluation (e.g., math reasoning) or scaling curve with more prompts.

### Questions
- Can you supply an ablation showing ReST-GRPO without the partial-state assembly (β=0 or α→0) to demonstrate the added value of partial state sampling?
- Does the value model retain accuracy when deployed on coding domains not seen during training?
- What is the computational overhead of VM-MCTS at test time (rollouts, latency, memory) compared to Best-of-N baselines?
- How does ReST-RL perform with non-code reasoning tasks (e.g., GSM8K) to support broader applicability?
- performance vs. GPU-hours for ReST-GRPO vs naive GRPO / DAPO to validate efficiency claims.(Fig.3 a claims but lacks compute axis)

### Soundness
3

### Presentation
3

### Contribution
2
