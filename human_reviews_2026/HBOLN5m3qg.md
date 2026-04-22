# $\textbf{Re}^{2}$: Unlocking LLM Reasoning via Reinforcement Learning with Re-solving

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
Reinforcement learning with verifiable rewards (RLVR) has shown promise in enhancing the reasoning performance of large language models (LLMs) by increasing test-time compute. 
However, even after extensive RLVR training, such models still tend to generate unnecessary and low-quality steps in their chain-of-thought (CoT), leading to inefficient overthinking and lower answer quality.
We show that when the initial direction or quality of the CoT is suboptimal, the model often fails to reach the correct answer, even after generating several times more tokens than when the initial CoT is well-initialized.
To this end, we introduce $\textit{\textbf{Re}inforcement Learning with \textbf{Re}-solving}$ (Re$^2$), in which LLMs learn to flexibly abandon unproductive reasoning paths and restart the solution process when necessary, rather than always committing to a final answer.
Re$^2$ applies pure reinforcement learning without any preliminary supervised fine-tuning, successfully amplifying the rare redo behavior in vanilla models from only 0.5\% to over 30\%.
This leads to substantial performance gains over standard RLVR under the same training compute budget, and also demonstrates notable improvements in test-time performance as the number of samples increases.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors argue that suboptimal initial reasoning often leads LLMs to inefficient overthinking. To address this, they propose Re^2, a RL framework that enables the model to restart the reasoning from scratch in the middle. Trained purely via RL without supervised fine-tuning, Re^2 significantly improves performance across diverse reasoning benchmarks compared to DAPO, under the same training compute.

### Strengths
- The paper presents a novel reinforcement learning framework that enables large language models to restart the reasoning process when the current trajectory is likely to lead to an incorrect solution.
- Re^2 consistently outperforms strong baselines across five diverse reasoning benchmarks (AIME24/25, AMC23, GSM8K, GPQA), covering both in-domain and out-of-domain settings.
- The reward design for the re-solving action is mathematically grounded, relying on estimated future success probabilities, and is formally derived in the appendix.
- Models trained with Re^2 exhibit favorable scaling with test-time compute, in contrast to DAPO-trained models whose performance tends to plateau or even degrade under increased sampling.

### Weaknesses
- The paper only compares Re^2 with DAPO, omitting other competitive baselines. Additionally, the paper only reports test-time scaling results using majority voting.
- Re^2 introduces an additional cost at inference time due to repeated re-solving, and there is no mechanism to regulate its frequency. This may be problematic in latency-sensitive settings.
- Under majority voting with a small number of samples, Re^2 underperforms DAPO. This is counterintuitive, given that Re^2 outperforms DAPO when the number of samples is 1 (see Table 1).
- The approach assumes access to verifiable reward signals, which limits its applicability to tasks with clear correctness criteria and makes generalisation to open-ended domains uncertain.

### Questions
- In the experiment described in Section 3.2 ('Start generating from x% of incorrect responses'), can we really assume that all the prefixes are incorrect? Even if the final response is incorrect, the first x% of the response may contain some correct reasoning or statements.
- Some recent papers (2025) use DPO to help LLMs check and correct their own answers without the need for an external model or supervision. Citing these papers would clarify the contribution of Re^2.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces a new reinforcement learning framework that allows large language models to restart reasoning when current solution paths become unproductive. Unlike standard RL with verifiable rewards (RLVR), this methods adds a re-solve action that lets the model abandon low-quality chains of thought and reinitiate problem solving. Using a novel three-way reward scheme (correct, incorrect, re-solve), Experiments across five reasoning benchmarks show consistent 4–6% accuracy gains and superior test-time scaling efficiency compared to prior RLVR methods like DAPO.

### Strengths
1. The paper proposes a novel re-solving paradigm that lets LLMs restart reasoning when current paths fail, breaking from the one-shot chain-of-thought assumption in prior RLVR work.
2. The approach is rigorously implemented, with a clear algorithm, fair baselines, and strong empirical validation across multiple benchmarks.

### Weaknesses
While the paper presents an interesting framework, the underlying mechanisms behind the method's effectiveness are not sufficiently analyzed. It remains unclear why the re-solving behavior leads to higher reasoning accuracy or a higher performance ceiling. For example, does restarting primarily improve exploration diversity, mitigate early trajectory bias, or simply increase the effective number of rollouts? A deeper behavioral or theoretical analysis would make the contribution more convincing.

Moreover, the paper lacks ablation studies to identify which components truly contribute to the gains. The new “redo” prompt, three-way reward, and prefix grouping may each affect learning in different ways. Notably, $Re^2$ and DAPO are trained with different prompts—the former includes explicit self-evaluation and restart instructions—so part of the improvement may arise from prompt design rather than algorithmic changes. Controlled ablations such as removing the “redo” instruction, disabling the re-solve reward, or turning off prefix grouping would better clarify which parts of the method are essential to its success.


If the authors could clarify these questions, it would really help me understand the mechanism better, and I will be happy to raise my score.

### Questions
1. I'm curious whether reflection is actually essential for reasoning, or if the ability to restart and resample is more critical. Does $Re^2$’s improvement suggest that reflection (continue generating from a given prefix is not a kind of reflect in my mind) within a single chain is less valuable than exploration through re-solving?

2. How sensitive is $Re^2$ to the number of rollouts during training or inference? Would smaller rollout budgets still yield similar benefits?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a reinforcement-learning–based training paradigm, Reinforcement Learning with Re-solving (Re²), that teaches language models to abandon unproductive partial chains of thought and restart problem solving when the current trajectory appears misguided. The method frames each query as a distribution over truncated prefixes sampled from full responses; for every prefix, the policy generates multiple continuations that can either commit to a final answer, produce an incorrect answer, or explicitly choose a resolve action that restarts from scratch. Rewards are assigned in three ways: 1 for correct final answers, 0 for incorrect ones, and for resolve actions the expected accuracy of solving from scratch, estimated from out-of-group completions and capped by at most R redo rounds via a closed-form geometric term. Advantages are computed group-wise within each prefix’s continuations and optimized with clipped policy updates, without any preliminary supervised fine-tuning. Across five benchmarks (AIME 2024/2025, AMC 2023, GSM8K, GPQA-Diamond) and five models from 3B to 14B parameters, Re² improves average accuracy over a strong RLVR baseline (DAPO) under matched training token budgets, and it shows stronger test-time scaling as the number of samples increases. The authors also provide analyses indicating that errors in early reasoning correlate with longer, lower-accuracy chains, and they report that Re² amplifies the otherwise rare redo behavior from roughly 0.5% to over 30% during training.

### Strengths
The core idea is original in the RLVR context: instead of privileging a single forward chain, the policy is explicitly trained to reconsider and restart when early evidence suggests confusion. This operationalizes a common human strategy and formalizes it with a clear three-way action space and a principled reward for resolve based on out-of-group success rates with a finite-round geometric expectation, yielding an intuitively aligned decision rule between “continue” and “restart.” 

Methodologically, the paper is careful to control training token budgets against DAPO, computes advantages within prefix groups to stabilize learning, and filters degenerate groups to avoid vanishing gradients. The empirical study spans multiple datasets and model types, including an RL-trained reasoning model, and consistently shows nontrivial gains in average accuracy, especially on the most challenging math tasks; furthermore, the test-time scaling plots indicate that Re² keeps improving with more samples where standard RLVR saturates, suggesting a better compute–performance frontier at inference. 

The presentation is clear about the training pipeline, reward derivation, and hyperparameters, and it includes qualitative traces that concretely illustrate how restarting avoids early misdirections that standard methods tend to entrench.

### Weaknesses
Althouth the truncate-then-continuate method is shown to be effecitve and better than DAPO, it is much slower in rollout procedure. I would be interested to see how much computaional cost overhead is added for this continuation step.

Weaknesses

Comparisons emphasize DAPO, but the space of modern RLVR baselines is broader (e.g., GRPO-style group objectives, critique-assisted RL, efficient RL variants), and decoding-time strategies that terminate low-confidence chains or backtrack could provide competitive non-RL baselines under similar token budgets; the absence of these ablations weakens the claim of superiority of the paradigm rather than the specific implementation. 

On evaluation, final answers are integers to ease verification, which simplifies reward parsing but narrows scope; it remains unclear whether the approach retains its advantage with symbolic or multi-step numeric answers that require more complex verifiers. 

The test-time comparison that counts redo attempts against the same sample budget is reasonable, yet when the sample count is small the method underperforms because many draws are resolve rather than final answers; guidelines or controllers for resolve frequency at inference are not yet provided, which limits practical deployment where token budgets are tight.

### Questions
How sensitive are the gains to the resolve reward estimator? Please report variance and bias when computing out-of-group accuracies with different n and m, and compare against a baseline that uses a moving average or a separate value head so we can see whether batch-coupled estimates are essential or merely convenient. 

Beyond DAPO, could you include GRPO-style objectives, recent efficient RLVR methods, and a decoding-only backtracking strategy under matched compute to clarify whether the benefit stems from the re-solve action space or other training details?

 For benchmarks with non-integer answers or programmatic verification (e.g., algebraic expressions or code), do you observe similar trends, and what verifier failure rates do you tolerate before resolve reward becomes noisy? 

aLso, please provide wall-clock training cost, instability metrics (e.g., degenerate-group rate over time), and an ablation on the template that triggers redo, since the current detection mechanism might entangle prompt engineering with policy learning.

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
3

### Summary
This paper proposes to allow the model to restart the reasoning path from scratch, preventing the  previous low-quality reasoning steps from hindering the model's reasoning.

### Strengths
- The method is well-motivated. I appreciate that the author talks about the problem in section 3 before introducing the proposed method.
- The idea of allowing the model to restart the reasoning makes a lot of sense to me.
- The paper is well-written and easy to follow.

### Weaknesses
- The performance gain is limited to me in Table 1. I understand that it may be subjective on whether +5.8 is significant or just noise. I suggest including confidence interval when the results are this close.

### Questions
- The finding in Section 3 about the correlation between the response length and accuracy seems to be contradictory to the finding in Deepseek-R1's paper. Does the author have any insight on this?

### Soundness
2

### Presentation
3

### Contribution
3
