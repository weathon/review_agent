# SFT-GO: Supervised Fine-Tuning with Group Optimization for Large Language Models

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 4, 6, 6

## Abstract
Supervised fine-tuning (SFT) has become an essential step in tailoring large language models (LLMs) to align with human expectations and specific downstream tasks. However, existing SFT methods typically treat each training instance as a uniform sequence, giving equal importance to all tokens regardless of their relevance. This overlooks the fact that only a subset of tokens often contains critical, task-specific information. To address this limitation, we introduce Supervised Fine-Tuning with Group Optimization (SFT-GO), a novel approach that treats groups of tokens differently based on their importance. SFT-GO groups tokens in each sample based on their importance values and optimizes the LLM using a weighted combination of the worst-group loss and the standard cross-entropy loss. This mechanism adaptively emphasizes the most challenging token groups and guides the model to better handle different group distributions, thereby improving overall learning dynamics. We provide a theoretical analysis of SFT-GO's convergence rate, demonstrating its efficiency. Empirically, we apply SFT-GO with three different token grouping strategies and show that models trained with SFT-GO consistently outperform baseline approaches across popular LLM benchmarks. These improvements hold across various datasets and base models, demonstrating the robustness and the effectiveness of our method.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper studied the token importance for supervised fine-tuning in large language models. It was motivated by the observation that important and unimportant tokens had different average CE losses during fine-tuning. To improve supervised fine-tuning, this work introduced a novel SFT-GO approach by grouping tokens based on their importance values and then optimizing LLMs using a weighted combination of the worst-group loss and the standard cross-entropy loss. Experimental results demonstrated that the proposed approach outperformed standard supervised fine-tuning in large language models.

### Strengths
**Originality:** The paper was motivated by the empirical results in Figure 1 that important and unimportant tokens had different average CE losses during supervised fine-tuning in LLMs. It then introduced a novel SFT-GO approach based on the group distributional robust optimization. 

**Quality:** In addition to empirical validation, the convergence of the proposed approach was theoretically analyzed. It showed that Rho-1 (Lin et al., 2024a) could be considered as a special case of the proposed framework. Experimental results demonstrated that the proposed approach outperformed standard supervised fine-tuning in large language models. The learning behavior of different grouping methods was analyzed.

**Clarity:** The paper is well-written. Most of the proposed techniques in SFT-GO are easy to follow. The experimental settings are presented in the submission (as well as the appendix).

**Significance:** The proposed approach provides a generic alternative for supervised fine-tuning in LLMs. It is also flexible to extend the proposed approach by considering more advanced token importance estimation strategies in various downstream tasks.

### Weaknesses
**W1:** The motivation of this paper is not well justified and explained. 
- Figure 1(b) shows the average CE losses of important and unimportance tokens. It states that "at the beginning of training, the losses for both groups are similar". But it seems that important and unimportance tokens have significantly different loss curves at the beginning of training in Figure 1(b).
- Following the previous concern, section 3 shows the key findings of Figure 1(b) that the losses for the two groups start similarly but diverge over time. However, the decay function in Eq. (5) allows the model to "focus more on reducing the worst-group loss during the initial training phases and gradually shift emphasis back to standard loss as training processes". This seems to be contradictory. Since the losses diverge later, it might make more sense to focus on the worst-group loss more as training goes on, not less.
- Line 162-163 states that models focus on reducing the loss of less informative tokens, possibly because they are easier to optimize. It is confusing why the loss of less informative tokens is easier to optimize.

**W2:** The theoretical analysis of SFT-GO is unconvincing.
- Proposition 1 compares the performance of the worst-performing group using SFT-GO and standard supervised fine-tuning. However, the proof in Appendix B shows that "for simplicity, prove without the constant $\lambda$ in Eq. 3." It is unclear whether Proposition 1 holds for any value of $\lambda$.
- It seems that Proposition 1 focuses only on the training loss over the worst-performing group. The generalizability of fine-tuned models using SFT-GO and standard supervised fine-tuning is still unclear.
- Lines 73-74 mention the theoretical properties that "minimizing our objective improves the accuracy of the model in all token groups compared to standard training". This theoretical property is not justified.

**W3:** The baseline comparison and hyperparameter sensitivity can be further discussed.
- Lines 296-297 show that when $\lambda=1$ in Eq. 3, Rho-1 aligns with the proposed training objective (the appendix also mentions that Rho-1 does not have a tunable $\lambda$). But it is unclear whether the experiments consider Rho-1 as one baseline approach in existing work. If so, it seems that the proposed approaches based on TF-IDF and LLMLingua-2 have marginal performance improvement compared to the existing Rho-1 baseline.
- The selection of the hyperparameter $\lambda$ is not discussed. It is unclear why $\lambda=0.9$ is selected for all settings.

### Questions
(1) Proposition 2 discusses the convergence rate of the proposed approach. However, compared to traditional DRO, this work focuses on the supervised fine-tuning process, by initializing the parameters using a pre-trained LLM. In contrast, traditional DRO will train a neural network from scratch. Will the different initialization approaches affect the convergence rate? More specifically, will pre-trained LLM enable improving the convergence rate compared to models with random initialization?

(2) Lines 274-275 show that "TF-IDF has proven to be a robust statistical measure for token importance". This proof can be further validated, especially when it also mentions that TF-IDF "does not account for contextual information".

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces SFT-GO, a supervised fine-tuning objective that augments standard CE with a worst-group token loss, where “important” vs “unimportant” tokens are pre-selected by a separate grouping/importance heuristic. The method is simple to implement and is evaluated by fine-tuning LLaMA-3.2-3B and LLaMA-3.1-8B on LIMA/Alpaca, reporting modest average gains across several standard benchmarks. The paper also includes a convergence discussion under abstract smoothness/convexity assumptions.

### Strengths
1. Clear and simple idea: the objective is easy to reproduce and plug into an existing SFT pipeline; training-time hyperparameters are straightforward (importance threshold, mixing weight, optional schedule).
2. Ablations exist: the paper varies the importance threshold and the mixing schedule and shows that the method does not collapse under reasonable ranges.
3. Reproducibility: datasets, backbones, and overall training setup are sufficiently specified; the method does not require invasive code changes.

### Weaknesses
1. Empirical impact is marginal: reported gains are small (often ~1–2 points or within noise), with several tasks showing negligible or no improvement. The paper does not present statistical significance or per-task confidence intervals, so it is hard to assess robustness.
2. Limited scope of models and data: only two relatively small LLaMA variants are tested, both within the same family and on narrow English instruction data. There are no results on larger backbones, multilingual settings, or human judgments—limiting external validity.
3. Unconvincing motivation figures: It is not convincing to treat Fig. 1(b) as the motivation, although "the loss for more informative, higher-value tokens remains relatively stable", the loss for important token groups declines consistently, and the figure does not show the convergence of the loss curve.
4. Theory is largely generic: the convergence argument hinges on broad assumptions under which many losses would also converge; the paper does not actually prove that the proposed ($L_{GO}$) satisfies these assumptions in the non-convex LM setting, nor does it connect the assumptions to practical training dynamics.
5. Missing justification vs CE: beyond intuition, there is no theoretical or diagnostic evidence (e.g., calibration, gradient-conflict, or error-type analyses) explaining *why* emphasizing worst-group tokens should outperform CE on downstream metrics. The preliminary plots do not make this case.
6. Potential implementation ambiguities: the objective contains a (\max) over token groups; the paper lacks detail on gradient flow/stability near ties, and on how frequently the “worst” group switches during training.

### Questions
1. Scaling & generality: Do the gains persist for larger backbones and different model families?
2. Objective mechanics: In practice, does the gradient back-propagate only through the currently worse token group, or through both groups? How are ties handled, and how stable are updates when the arg-max group flips frequently?
3. Assumptions vs reality: Can you prove that ($L_{GO}$) satisfies the assumptions used in your convergence discussion? If not, can you provide empirical diagnostics linking those assumptions to observed training behavior?
4. Why better than CE? Beyond intuition, can you show analyses (e.g., error types, per-token calibration, gradient interference, or learning-curve segments) that explain when/why ($L_{GO}$) helps—and when it does not?

### Soundness
2

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
The authors introduce SFT-GO, a supervised fine-tuning with group optimization, where they put the emphasis on token level selection rather than data instance selection like on other sampling methods introduced in the literature for SFT. They correctly highlight the gap in recent works in the literature that not enough in depth research has been conducted on token level sampling optimizations for 
SFT.

They make two main claims: (1) minimizing our objective improves the accuracy of the model in all token groups compared to standard training, and (2) a simple mini-batch procedure preserves the O(1/√T) convergence rate and converges to the global optimum. Their methodology to show this focuses on a) a statistic based method using TF-IDF scores, b) a semantics based method using token selection probabilities, and c) a task-specific approach using an external model. Their evaluation focuses on two main datasets, LIMA, and Alpaca.

The main contribution of the paper is introducing the analytical foundation of their group optimization setup for token selection. They define a group function and use that to define a new loss function which is a combination classical avg loss over all tokens + a worst case scenario of loss over tokens picker or not picked by the grouping function. See equations 1-4. Then they introduce two propositions to prove achieving better loss, an better convergence time.

Overall, the paper is well written, with their main contribution being on the analytical aspects of the group optimization. They

### Strengths
- The paper provides compelling empirical evidence that standard supervised fine-tuning under-optimizes semantically important tokens relative to common functional tokens, motivating the need for differential treatment across tokens.

- The paper is strong in its mathematical rigor and analytical proof. It successfully proves the two propositions, with fairly high complication in the proof process utilizing algebraic massages of probabilistic values and inequalities. Proposition 2 utilizes Jensen inequality successfully derive an error bound.

- The authors demonstrate that the method generalizes across several grouping strategies (TF-IDF, LLMLingua-2, and Rho-1), highlighting that the framework is not tied to a specific token importance estimator.

### Weaknesses
- The choice of token importance threshold 𝜂 meaningfully affects performance, and the paper shows non-monotonic behavior as this parameter varies. This suggests additional tuning is required.

- The method’s effectiveness varies significantly across grouping strategies. For example, LLMLingua-2 performs better than TF-IDF, reflecting dependence on access to an external semantic model. Thus, SFT-GO’s benefits are not inherent—they rely on choosing a strong importance estimator.

- Although the paper motivates why token level optimization is needed, it still lacks enough comparison to other forms of data sampling. Other techniques include sampling methods using optimal design, information gain importance sampling, etc.

### Questions
The authors provide a justification for their theoretical assumptions in Appendix C. It would be great if I can learn more on what other challenges were preventing them to lift these restrictions and prove more general results with less restrictive assumptions.

### Soundness
3

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
2

### Summary
SFT-GO is an updated supervised fine-tuning algorithm for LLMs where the authors propose an importance-based token-grouping instead of treating all tokens equally. This grouping based on a weighted combination of the standard cross-entrpopy loss and a worst-group loss. This encourages the model to perform well on rare but semantically rich tokens as well. Empirical results on the Alpaca and LIMA benchmarks using the llama 3.1 8B and 3.2 3B models shows a consistent improvement over 7 reasoning and QA benchmarks.

### Strengths
1. Connecting group DRO to LLM finetuning is simple and intuitive to understand. An added benefit that can be expected is indirectly reducing token-level spurious correlations (aka reliance on filler words, etc.). 
2. The empirical results across different LLM benchmarks show an improvement in average performance for minimal change in training setup. 
3. The definition of the grouping function is interesting - particularly the use of llmlingua 2 for semantics-based grouping and its effectiveness. 
4. A study of token importance during training could be a whole paper in itself - a research field that could deserve some attention. Particularly interesting when it comes to multi-language or multi-modal training.

### Weaknesses
1.  The dependence on tools like llmlingua for grouping can create a suboptimal dependency on certain domains (for instance the drop in performance in Math QA in Table 2). As mentioned by the authors, any input biases in these models or the training data (in case of TF IDF) will be amplified in training.
2.  Qualitative examples of the groups and semantically-rich tokens determined by their algorithm would make it easier to support these otherwise intuitive claims. 
3. While few works consider token-level importance, there are lots of sample-level data valuation frameworks [1] [2] . This paper could have used a baseline to compare how SFT-GO compares to these methods. The correlation (or lack of) between high-valued samples and high valued tokens would be interesting to note. 


[1] What is Your Data Worth to GPT? LLM-Scale Data Valuation with Influence Function
[2] GREATS: Online Selection of High-Quality Data for LLM Training in Every Iteration

### Questions
1. Can the authors offer an experiment to prove the convergence rates are the same as standard SGD under convex assumptions, as claimed in the theoretical analysis. 
2. Impact of SFT-GO on downstream preference-optimization/ alignment tasks is unclear - can the authors comment on the same?
3. An interesting future experiment could be the interaction between SFT-GOs important token groups and attention saliency maps.

### Soundness
3

### Presentation
3

### Contribution
2
