# Certifiable Safe RLHF: Fixed-Penalty Constraint Optimization for Safer Language Models

- Decision: Reject
- Scores: 4, 4, 2

## Abstract
Ensuring safety is a foundational requirement for large language models (LLMs). Achieving an appropriate balance between enhancing the utility of model outputs and mitigating their potential for harm is a complex and persistent challenge. Contemporary approaches frequently formalize this problem within the framework of Constrained Markov Decision Processes (CMDPs) and employ established CMDP optimization techniques. However, these methods exhibit two notable limitations. First, their reliance on reward and cost functions renders performance highly sensitive to the underlying scoring mechanism, which must capture semantic meaning rather than being triggered by superficial keywords. Second, CMDP-based training entails tuning dual-variable, a process that is both computationally expensive and does not provide any provable safety guarantee for a fixed dual variable that can be exploitable through adversarial jailbreaks. To overcome these limitations, we introduce Certifiable Safe-RLHF (CS-RLHF) that introduces a cost model trained on a large-scale corpus to assign semantically grounded safety scores. In contrast to the lagrangian-based approach, CS-RLHF adopts a rectified penalty-based formulation. This design draws on the theory of exact penalty functions in constrained optimization, wherein constraint satisfaction is enforced directly through a suitably chosen penalty term. With an appropriately scaled penalty, feasibility of the safety constraints can be guaranteed at the optimizer, eliminating the need for dual-variable updates. Empirical evaluation demonstrates that CS-RLHF outperforms state-of-the-art LLM model responses rendering at-least 5$\times$ efficient against nominal and jail-breaking prompts.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes a constrained alignment framework (Eq. 5) with a corresponding optimality result (Theorem 1). Compared to a standard Lagrangian approach, the proposed method yields an optimal policy with higher expected return, while the dual variable $\lambda$ is bounded and linked to constraint violation without requiring the averaging procedures used in prior work. The authors also introduce a cost model and present preliminary experiments.

### Strengths
1. $\textbf{Theoretical advantage:}$ Under the formulation in Eq. 5 and Theorem 1, the optimal policy enjoys higher expected return than the Lagrangian baseline, and the dual parameter $\lambda$ is bounded and tied to constraint violations. This is the paper’s primary contribution.

2. $\textbf{Clear objective reformulation:}$ Moving from the Lagrangian setup to the proposed framework is well motivated and theoretically characterized.

### Weaknesses
1. $\textbf{Cost model claim vs. Safe-RLHF needs a fair test.}$ The paper claims the proposed cost model is superior to Safe-RLHF. However, Safe-RLHF does not simply rely on keyword heuristics. It has metadata and content-aware harmful categories, and assigns the same safety label to each (prompt, response) pair. To substantiate the claim, please perform a matched comparison on the Safe-RLHF dataset: train with your Eq. 10 on the Safe-RLHF train split and report label correctness (or a comparable metric) on the Safe-RLHF test split, versus optimizing Eq. 9 as in the original paper [1]. If the core contribution is actually the new labeling scheme/dataset, then (i) describe the labeling protocol in detail, (ii) justify why it is better than Safe-RLHF, and (iii) provide dataset statistics (sources, examples).

2. $\textbf{Experimental scale and baselines in Table 2 are insufficient.}$ Evaluating on 100 randomly selected prompts is too small and risks cherry-picking. Please scale to a size comparable to the Safe-RLHF test split (≈1k prompts). In addition to Safe-RLHF, include stronger baselines that address the same constrained alignment goal without PPO-style training, e.g., Stepwise/DPO-style methods [2] and constrained DPO [3]. A broader set of baselines is necessary to make the empirical case convincing.

3. $\textbf{Comparative win rates / Elo.}$ Include win-rate or Elo-style evaluations against reference systems (e.g., Alpaca-7B, Safe-RLHF, CS-RLHF) in addition to Figure 2, to round out the results and make the findings more interpretable.


[1] Dai et al., Safe RLHF: Safe Reinforcement Learning from Human Feedback, 2023.

[2] Wachi et al., Stepwise Alignment for Constrained Language Model Policy Optimization, NeurIPS 2024.

[3] Liu, Sun, Zheng, Enhancing LLM Safety via Constrained Direct Preference Optimization, 2024.

### Questions
Please see my weaknesses: my primary concerns are the cost-model claim and the limited experiments. I will consider increasing my score if both are resolved.

### Soundness
2

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
The paper tackles LLM safety as a constrained optimization problem and argues that standard CMDP/Lagrangian RLHF is fragile: performance hinges on noisy reward/cost scoring and expensive dual-variable tuning that lacks fixed-λ guarantees and can be jailbroken. It proposes Certifiable Safe-RLHF (CS-RLHF), which (i) trains a cost model on large harmful/harmless labels to deliver semantically grounded safety scores, and (ii) replaces the Lagrangian with a rectified exact-penalty objective that activates only when the expected cost exceeds a threshold. (iii) The authors also add a BoN decode-time guarantee.

### Strengths
The formalization of safety alignment as a constrained optimization problem is highly novel, making the safety alignment process itself more reliable and trustworthy.

Highlighting that an overly conservative cost model can negatively impact the final performance of aligned LLMs is crucial. While the issue of overly conservative models has been well-studied in the RL community, it remains underexplored in the LLM space. This paper makes an important contribution by addressing this gap.

Additionally, the introduction of the new Best-of-N (BoN) sampler provides a decode-time safety guarantee, further enhancing the trustworthiness of the safety alignment process.

### Weaknesses
Please refer to the Questions.

### Questions
1) **Expectation vs. pointwise constraint (Eq. 4).**  
   The rectified penalty appears to be applied to an *expected* cost. How exactly is ReLU composed with the expectation—\(\mathrm{ReLU}(\mathbb{E}[C]-d)\) vs. \(\mathbb{E}[\mathrm{ReLU}(C-d)]\)? These are not equivalent; please clarify the chosen form and its implications for per-sample violations.

2) **Penalty scaling and “non-negotiable” safety (Thm. 1 & line 233).**  
   The guarantee seems to require \(\lambda\) to increase as the target \(\epsilon\) decreases, potentially affecting the reward–cost trade-off. How is \(\lambda\) set in practice? If feasibility loosens for moderate \(\lambda\), how does this align with the claim that “harmlessness is non-negotiable”?

3) **Dataset comparability and fairness.**  
   How does your dataset differ from Safe-RLHF’s? Since Safe-RLHF also provides binary safe/unsafe labels, why not train/evaluate on their data for comparisons? Using different datasets risks confounding the empirical gains.

4) **Behavior on jailbreaks (Fig. 2).**  
   Figure 2 seems to show the cost model assigning low cost to certain jailbreak prompts (classifying them as safe). Please explain this behavior, provide failure cases, and quantify its frequency.

5) **Cost-model design vs. simpler alternatives.**  
   You fine-tune LLaMA-2-7B-chat as the cost model. A simpler baseline—e.g., using layer-26 activations with an unsupervised classifier to output \(\{0,1\}\)—could test whether heavy fine-tuning is necessary. If such a baseline performs similarly, Eq. (10) may be unnecessarily complex.

6) **Loss in Eq. (10).**  
   Why not use a standard classification loss (e.g., logistic cross-entropy or margin losses)? Please justify the chosen objective or show ablations against standard losses.

7) **Semantics over keywords: evaluation gaps (line 309).**  
   To support the claim that the cost model captures semantics rather than triggers, evaluatation on over-refusal sets such as *XSTest*[1], *CoCoNot*[2], and *OR-Bench*[3], where benign prompts share toxic tokens with unsafe prompts. This would more directly test the stated research question.

[1] Röttger, Paul, et al. "Xstest: A test suite for identifying exaggerated safety behaviours in large language models." arXiv preprint arXiv:2308.01263 (2023).

[2] Brahman, Faeze, et al. "The art of saying no: Contextual noncompliance in language models." Advances in Neural Information Processing Systems 37 (2024): 49706-49748.

[3] Cui, Justin, et al. "Or-bench: An over-refusal benchmark for large language models." arXiv preprint arXiv:2405.20947 (2024).


I would like to raise my score if part of the questions are answered appropriately.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Their method modifies SafeRLHF by using a fixed penalty term instead of the lagrangian to encourage safety. They also use ReLU to only take positive penalties into account instead of negative ones. They show that using their penalty during best of N inference guarantees that the safe response is selected if any of the responses are safe. They also create a dataset of labeled safe and unsafe prompt response pairs.

### Strengths
Novelty is sufficient but mixed.
Their core contribution is a small change to an existing method. However, I think the other contributions add up to sufficient novelty.

The experimental results are solid and show improvement over a reasonable baseline.

### Weaknesses
The guarantee that “if any of the N candidates is safe, the selected response is safe” seems trivial to achieve. The following method does the trivially: Pick the safe response which gets the highest reward. Let me know if I’m missing something, since I didn’t understand this part very well (see below).

Equation 10 isn’t well motivated. The paper doesn’t explain the difference between equation 10, and training the classifier using SFT with sigmoid cross entropy, or the advantage of their approach over sigmoid cross entropy.


Lack of ablation studies

There aren’t any ablation studies so it’s hard to tell which of the contributions improved the results over the baseline. I would like to see ablation studies testing the impact of each contribution individually. Examples of possible ablation studies:
Using their classifier with the baseline optimization method.
Training a classifier on their dataset with sigmoid cross entropy instead of their objective function.
Comparing training a classifier on their dataset of safe and unsafe responses with training on an existing dataset.


Clarity

The inference time safety section isn’t well explained. Specifically: How is equation 11 analogous to BoN? Note: The explanation of the BoN guarantee in the introduction is much more clear.

Tables and figures are hard to understand.
Figure 1 is hard to read because the text is too small, and it’s hard to tell what is a different method and what’s a different part of the same method.
Table 2 confusing. It’s hard to tell what subset of the data each of the numbers is computed over. It would be more straightforward to have one table for evaluating the safety of each method's responses according to humans, and a separate table for how often each of the classifiers agree with humans. For example, “Reward Evaluation Helpful” is a subset of “Human Verdict Safe” but that’s not immediately obvious without doing math, or reading the caption in detail.
The text in figure 2 is too small to read. Also, the description for each of the subplots should be included in the figure itself, not just in the caption for ease of reading. Each subplot should use the same color for ‘regular prompts’ so the legend can be shared between all of them.

In the Cost Model section, it’s not explained why the baseline heavily penalizes keywords.

Minor: ReLU is defined multiple times in 4.1

### Questions
See weaknesses. Specifically, my questions around BoN, and equation 10.

### Soundness
2

### Presentation
2

### Contribution
2
