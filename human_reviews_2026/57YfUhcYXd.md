# Eliminating Inductive Bias in Reward Models with Information-Theoretic Guidance

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Reward models (RMs) are essential in reinforcement learning from human feedback (RLHF) to align large language models (LLMs) with human values. However, RM training data is commonly recognized as low-quality, containing inductive biases that can easily lead to overfitting and reward hacking. For example, more detailed and comprehensive responses are usually human-preferred but with more words, leading response length to become one of the inevitable inductive biases. A limited number of prior RM debiasing approaches either target a single specific type of bias or model the problem with only simple linear correlations, *e.g.*,  Pearson coefficients. To mitigate more complex and diverse inductive biases in reward modeling, we introduce a novel information-theoretic debiasing method called **D**ebiasing via **I**nformation optimization for **R**M (DIR). Inspired by the information bottleneck (IB), we maximize the mutual information (MI) between RM scores and human preference pairs, while minimizing the MI between RM outputs and biased attributes of preference inputs. With theoretical justification from information theory, DIR can handle more sophisticated types of biases with non-linear correlations, broadly extending the real-world application scenarios for RM debiasing methods. In experiments, we verify the effectiveness of DIR with three types of inductive biases: *response length*, *sycophancy*, and *format*.  We discover that DIR not only effectively mitigates target inductive biases but also enhances RLHF performance across diverse benchmarks, yielding better generalization abilities. The code and training recipes are available at https://github.com/Qwen-Applications/DIR.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper targets to the issue of inductive bias of reward models. This paper proposes an information-theoretic debiasing method, called DIR, which maximizes the mutual information between preference prediction and input pairs, while minimizing the mutual information between outputs and predefined bias attributes to debiase these biases. Experimental results demonstrate its effectiveness in debiasing biases of length, sycophancy and format, while achieving competitive alignment performance.

### Strengths
- The paper is well written and clear to read.
- Denoising reward model is very important for RLHF to improve alignment performance of LLMs.
- The experimental results are solid, covering three types of biases.

### Weaknesses
- In the experimental setup, the authors address the biases of length, sycophancy, and format individually. However, in practice, we would expect a reward model to debias all these biases simultaneously. It seems that the paper lacks an experimental setup that assesses and discusses the ability of DIR to mitigate all these biases simultaneously.
- There are two commonly used benchmarks for preference alignment, i.e., AlpacaEval [1] and MT-Bench [2], which should be included to evaluate the model ability of open-ended and multi-turn generations.
- I think that a figure to illustrate your methods will help readers intuitively understand your proposed method.
- The proposed method DIR relies on predefined inductive biases, such as length, sycophancy and format. However, if the reward model has some potential biases that are not intuitive such as length perceived by human, can the proposed DIR handle such scenario?

[1] Length-controlled alpacaeval: A simple way to debias automatic evaluators.  
[2] Judging llm-as-a-judge with mt-bench and chatbot arena.

### Questions
- Do you compare performance with DPO directly trained using the same preference data?
- What is your implementation details of the variational network $\psi$?

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
The paper introduces a method called DIR (Debiasing via Information Optimization for Reward Models) aimed at mitigating inductive biases, such as response length, sycophancy, and format, that plague reward models in RLHF settings. It frames training as an information-theoretic problem: the method maximises the mutual information (MI) between model preferences and input response pairs, while simultaneously minimising the MI between the model’s internal representation and specified bias attributes. The debiasing objective is realised via an adversarial variational network applying the CLUB MI-estimator to ensure the learned representation is less correlated with bias attributes.

### Strengths
1. The paper propose a new method and tackles well-documented biases in reward models
2. the approach provides a  structure that could extend to multiple bias types.
3. The proposed DIR method shows improvements over several baselines

### Weaknesses
1. The paper presents itself as introducing a “novel information-theoretic framework,” but its core components are repurposed versions of existing methods. The preference loss is simply the standard Bradley-Terry ranking loss, reinterpreted post hoc as a mutual-information maximization objective. The debiasing term also relies on a conventional adversarial setup using the CLUB estimator [1], a technique already established in prior work. Although the implementation is sound and practically useful, it does not represent a genuinely new theoretical contribution.
2. The paper’s analysis of the key hyperparameter, $\lambda$, is self-contradictory. Figure 4 claims that performance peaks at $\lambda = 1$ and drops sharply at $\lambda = 10$ , calling the latter an “over-correction.” However, Table 5 shows the opposite: the $\lambda = 10$  model achieves the highest overall and “Hard” subset scores. These results directly conflict, undermining the credibility of the authors’ claims about the optimal debiasing strength and the rigor of their tuning methodology.
3. Although the results are generally strong, they are not consistently superior across benchmarks. In the Length Bias test, the proposed method’s correlation (0.468) only slightly outperforms the Skywork baseline (0.498), showing minimal real improvement. In the Sycophancy Bias test, the InfoRM baseline even outperforms the proposed method in the 20%/70% adversarial setting (86.6 vs. 85.1). These mixed outcomes weaken the claim of “most consistent and robust performance” and suggest the improvements may be context-dependent rather than universal.
4. The method’s dependence on a pre-defined and labeled bias attribute ($b$) is a practical limitation. It requires the user to already know what the bias is (e.g., length, sycophancy) and be able to label it for every data pair. This makes it useless against unknown or hard-to-quantify biases, a limitation that more "indirect" methods like InfoRM (which the paper critiques) [2] would not have.

[1] Club: A contrastive log-ratio upper bound of mutual information.
[2] Inform: Mitigating reward hacking in rlhf via information-theoretic reward modeling.

### Questions
1. The DIR framework's primary practical limitation is that it requires a "pre-defined bias attribute b". This means a human must identify and label the specific bias (length, format, etc.) they want to remove. How does this method fare against unknown or unlabeled biases, which are common in real-world data?

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
3

### Summary
The paper proposes an information-theoretical viewpoint on reward modeling and the Bradley-Terry model. Specifically, the proposed method, DIR, focuses on mitigating inductive biases, such as verbosity bias and stylistic biases, by maximizing the mutual information between preference prediction and input-response pairs. By demonstrating the performance of the reward model itself and as a preference proxy in RLHF training, the paper shows that DIR could be an effective debiasing objective for reward modeling.

### Strengths
1. The paper presents an intuitive yet theoretically reasonable scope of understanding reward modeling as aligning the preference distribution and preference prediction from the reward model.
2. The benchmark analysis on the biases in the reward model benchmark, RM-Bench, comes before the actual debiasing evaluation of the proposed method, which strengthens the experimental rigor of the paper.
3. Alongside the well-known length bias, the paper studies multiple types of biases and demonstrates that DIR can be an effective learning objective across different biases.

### Weaknesses
The main weakness of the paper is in the clarity of writing. The clarity of mathematical notations and experimental details in the paper can be improved. Other points that could either be clarified or stated as weaknesses are listed in the questions. Overall, the clarity in Sections 2 and 3 should be improved for better clarity. While there are multiple cases where the notational consistency/clarity is lacking, these are a few examples:
- Section 3.1 starts by saying that $\mathcal{L}\_\text{total}$ consists $\mathcal{L}\_\text{pref}$ and $\mathcal{L}\_\text{debias}$, while Equation (12) uses $\mathcal{L}\_\text{reward}$ instead.
- $\mathcal{L}_\text{debias}$ is not explicitly defined in the paper and appears in Equation (12).

These inconsistencies and missing definitions prevent a clear understanding of the paper, even though the paper's theoretical soundness should be highlighted as its strength.

### Questions
- Can DIR be expanded to the implicit reward models like direct alignment algorithms?
- On the official RM-Bench leaderboard, “Skywork-Reward-Llama-3.1-8B-v0.2” (“BT” in Table 5) has mostly higher numbers compared to the results stated in Table 5. For example, “Hard” score for Skywork-Reward-Llama-3.1-8B-v0.2 is 52.6 and 69.3 for “Chat”, while the reported scores in the paper are 42.76 and 64.69, respectively. Given that the numbers on the leaderboard could change the trend in Table 5 (e.g., the overall score of Skywork-Reward-Llama-3.1-8B-v0.2 on the leaderboard is higher than “Ours-10.0”), this part needs clarification.
- On the RM-Bench scores, why does the “Ours” model experience a notable drop in the “Easy” accuracy? By debiasing, is the model experiencing a trade-off in easy stylistic differentiation?

### Soundness
2

### Presentation
2

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
This paper proposes DIR, an information-theoretic framework for debiasing reward models in RLHF. The method maximizes MI between predictions and true preferences while minimizing MI between internal representations and predefined bias attributes. Using data processing inequality and variational bounds (BA and CLUB), the theoretical objective becomes a tractable loss. Experiments on length, sycophancy, and format biases show improvements in both RM metrics and downstream RLHF performance.

### Strengths
1. The explicit use of data processing inequality to justify representation-level debiasing, combined with dual variational bounds (BA for information retention, CLUB for bias suppression), provides an elegant and principled solution.

2. Experiments cover three diverse bias types with end-to-end assessment (RM performance + downstream PPO policies). Strong ablations on representation choice (Table 6) and hyperparameter $\lambda$ (Figure 4) validate design decisions.

3. Zero inference overhead and moderate training cost (~33% increase). Demonstrated versatility across bias types suggests broad applicability.

### Weaknesses
1. Method requires knowing bias types *a priori* and labeling $b_{\mathrm{rel}}$ for every pair. No mechanism for unsupervised bias discovery limits real-world applicability.
2. Experiments isolate single biases. Real datasets likely contain concurrent biases (e.g., lengthy + sycophantic responses). Unclear how to extend DIR—multiple debiasing terms with separate $\lambda$ values? Potential optimization conflicts?
3. Sycophancy evaluation uses fixed prefix injection (“*Yes, you are right.*”). Real biases are more subtle and contextually integrated, raising generalization concerns.
4. Unexplored Representation Alternatives: Exclusively uses final hidden state without comparing alternatives (e.g., mean-pooling). Global representations may better capture stylistic/format biases.

### Questions
1. How would DIR handle concurrent biases? Would you use $\mathcal{L}\_{\text{reward}} + \sum\_i \lambda\_i \mathcal{L}\_{\text{debias}}^{i}$? What optimization challenges arise from negative interactions between debiasing signals?
2. Have you studied the architecture sensitivity of $q_{\psi}$? Could an overly powerful $q_{\psi}$ discard legitimate preference-informative correlations along with spurious biases?
3. How do debiased RMs interact with direct alignment methods like DPO? Could altered reward landscapes complicate implicit differentiation?

### Soundness
3

### Presentation
3

### Contribution
3
