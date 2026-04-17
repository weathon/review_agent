# Beyond Pairwise: Empowering LLM Alignment With (Ranked) Choice Modeling

- Decision: Accept (Poster)
- Scores: 8, 8, 2

## Abstract
Alignment of large language models (LLMs) has predominantly relied on pairwise preference optimization, where annotators select the better of two responses to a prompt. While simple, this approach overlooks the opportunity to learn from richer forms of human feedback, such as multiway comparisons and top-$k$ rankings. We introduce \textit{Ranked Choice Preference Optimization} (RCPO), a unified framework that bridges preference optimization with (ranked) choice modeling via maximum likelihood estimation. RCPO supports both utility-based and rank-based models, subsumes several pairwise methods (such as DPO and SimPO) as special cases, and provides principled training objectives for richer feedback formats. We instantiate this framework with two representative models (Multinomial Logit and Mallows-RMJ). Experiments on Llama-3-8B-Instruct, Gemma-2-9B-it, and Mistral-7B-Instruct across in-distribution and out-of-distribution settings show that RCPO consistently outperforms competitive baselines. RCPO shows that directly leveraging ranked preference data, combined with the right choice models, yields more effective alignment. It offers an extensible foundation for incorporating (ranked) choice modeling into LLM training.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces RCPO, a framework that generalizes LLM alignment beyond pairwise preferences. It connects preference optimization to choice modeling via MLE, unifies DPO and S-DPO (as MNL-PO-Discrete) under this framework, and derived loss functions for MNL and Mallows-RMJ models choice models towards single-best and top-k rankings.

The authors demonstrate its effectiveness by fine-tuning Llama-3-8B-Instruct and Gemma-2-9B-it on preference data derived from UltraFeedback and Skywork-Reward-V2-Llama-3.1-8B reward mode. The results consistently outperformed baselines like DPO/KPO and S-DPO.

The paper also covers practical tricks to implement the Mallows-RMJ objectives, like sigmoid-smoothing approximation for non-differentiable step functions, and use entropy to estimate the dispersion parameter.

### Strengths
The paper presents a principled and unified view connecting preference optimization, choice modeling, and MLE, bridging the literature on choice models with LLM alignment. The derivation of the new methods is very principled, and the presentation is clear.

The paper proposes two new methods (MNL and Mallows-RMJ based) for leveraging single-best and top-k rankings in preference optimization. The proposed methods are practical, providing practitioners with useful tools for exploring different sampling and ranking strategies in LLM post-training.

The experimental design and outcomes are overall convincing.

### Weaknesses
While the authors provide reasonable baselines for the pairwise and single-best scenarios, the evaluation of the top-k methods could be more robust by including something like a vanilla DPO method trained on all pairs implied by the top-2 ranking, especially given the impact of negative sampling on DPO; it is unclear how much of improvement from the “-Top2” methods come from a sheer negative sampling change.

Additionally, while the paper positions RCPO as a generalized framework and provides good insights on how choice modeling and MLE connect, the core methods derived feels a bit fragmented as their derivation requires several very specific assumptions. The paper feels more like a collection of methods derived under a similar principle. E.g., Citing  “For the baseline method, I used RCPO with MNL choice model for top-k choice” sounds a bit awkward.

### Questions
Could the authors provide further reasoning/evidence on the suitability of Skywork-Reward-V2 for labeling the complex reasoning tasks in AlpacaEval 2 and Arena-Hard? Asking because the qualitative example in Table 15 shows the aligned models replicating a mathematical error (x**2 − x − 8 = (x − 2)(x + 4)), and curious if that could be related to RM quality (i.e. did the preference labeler catch the error?)

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
2

### Summary
This paper introduces Ranked Choice Preference Optimization (RCPO), a unified framework designed to improve Large Language Model (LLM) alignment by leveraging richer human feedback, such as top-k rankings, instead of relying solely on pairwise comparisons.
Empirical evaluations on Llama-3-8B and Gemma-2-9B show that RCPO variants consistently outperform competitive baselines on major benchmarks, with the rank-based Mallows-RMJ model achieving particularly strong results using top-2 feedback.

### Strengths
The paper presents the generalized framework: Ranked Choice Preference Optimization (RCPO) which subsumes several existing pairwise methods, including Direct Preference Optimization (DPO), SimPO, and R-DPO, viewing them as special cases.

It is not limited to pairwise comparisons, but can directly leverage richer human feedback formats, such as multi wise comparisons, single-best selections from a set, and top-k rankings.

Empirical studies on Llama-3-8B-Instruct and Gemma-2-9B-it show that RCPO variants consistently outperform competitive baselines (including DPO, SimPO, and IPO) on widely adopted benchmarks like AlpacaEval 2 and Arena-Hard. The best-performing variant, Mallows-RMJ-PO-Top2, surpassed the strongest non-RCPO baseline (IPO) by significant margins.

The improvements are shown on Llama-3 and Gemma-2 base models.

### Weaknesses
The paper only shows experimental results with Top-2 feedback. The authors say that they stop at Top-2 since the performance does not always increase with more feedback. It would be very interesting to see when this happens. Only seeing Top-2 results, when the obvious next step would be to try a higher number Top-k, gives me that the authors are trying to hide something.

The experiments are conducted only on relatively smaller models, specifically Llama-3-8B-Instruct and Gemma-2-9B-it. The paper does not demonstrate whether the gains observed with RCPO scale to larger models.

### Questions
Some of the equations in the appendix move out of bounds. You should fix that.

Did you run any Top-k experiments with k > 2? If not, why not? Just that it might get worse doesn’t sound like a good reason. It might get better as well.

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
Pairwise comparison forms the foundation of model alignment, and despite the emergence of alternative approaches such as KTO and joint preference optimization, pairwise methods remain dominant in the community. In this work, the authors propose a framework that extends beyond traditional pairwise comparisons by incorporating rank-order information. They introduce two complementary frameworks under this paradigm (called RCPO) and evaluate them both against each other and against established baselines such as DPO, demonstrating consistent improvements from the proposed approach.

### Strengths
I appreciate an alternate modeling for the preference alignment, the associated literature, and the corresponding MLE estimator formulations in the LLM alignment setup, with a given optimal reward structure. I also appreciate the gradient analysis, and to my understanding, the MNL style approach seems close to one-vs-many, where the MALLOWS-RMJ MODEL seems close to one-vs-one, but weighted by the size of the universe.

### Weaknesses
### Overall Comments
This work would benefit from more experiments and broader comparisons with related methods. Given the current limitations, I don’t think it’s yet ready for this venue.

---

### Related Work on Going Beyond Pairwise Ranking
The paper should include quantitative comparisons with several recent methods that also move beyond pairwise preference modeling:

- **LiPO-λ (Listwise Preference Optimization through Learning-to-Rank):** Uses LambdaLoss with permutation-aware weighting and improves over DPO on Reddit TL;DR and Anthropic-HH.  
- **KPO (K-order Ranking Preference Optimization):** Extends Plackett–Luce models with adaptive \(K\) selection and shows better sample efficiency on recommendation benchmarks.  
- **PPA (Permutative Preference Alignment):** Optimizes differentiable NDCG-based objectives and outperforms pairwise methods on AlpacaEval.  
- **AMP/MDPO (Automated Multi-level Preference Optimization):** Handles multi-level preferences and achieves strong results on hallucination reduction benchmarks.  

Adding comparisons with these works would make the evaluation more complete.

---

### More Models
The results are currently based on only two models. It would help to include additional models such as **Qwen2.5** or **Mistral** to test how well the method generalizes.

---

### More Datasets
It would also strengthen the paper to include more datasets, such as **TL;DR**, **Anthropic-HH**, **UltraFeedback**, and **WildBench** (which is more challenging than AlpacaEval 2.0).

---

### Limited and Inconsistent Improvements
The improvements are not consistent across models or datasets.  
For example:
- On **LLaMA**, LC–SimPO performs much better.  
- On **Gemma**, **DPO** performs better.  
- On **Arena-Hard**, the improvements are small and may not be significant.

---

### Ablation Studies
1. The paper mentions that \(k=2\) performs best, but it would be useful to show how performance changes as \(|S|\) and \(k\) vary together.  
2. It would help to test what happens if the \(|S|-i\) weighting in the Mallows RMJ model is removed, since that still represents a valid one-vs-one setup.  
3. The paper should also study **sample efficiency**—how performance changes as the amount of training data increases (e.g., WR and LC at different data budgets).
4. Sensitivity analysis on $\beta$

### Questions
- It is important to consider the practical challenges of collecting ranked-choice data, both from an implementation standpoint and a psychological one. Given the current data collection pipelines used by major industry LLMs, adopting a ranking-based approach would represent a significant paradigm shift. Moreover, the usefulness of rank information—specifically, the choice of k, |S|, is highly task-dependent. This raises questions about whether the additional computational and monetary costs of obtaining such data are justified, especially when the observed performance gains appear relatively modest.

### Soundness
2

### Presentation
2

### Contribution
2
