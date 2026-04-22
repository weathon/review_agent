# Curriculum-RLAIF: Curriculum Alignment with Reinforcement Learning from AI Feedback

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 6, 4, 2

## Abstract
Reward models trained with conventional Reinforcement Learning from AI Feedback (RLAIF) methods suffer from limited generalizability, which hinders the alignment performance of the policy model during reinforcement learning (RL). This challenge stems from various issues, including distribution shift, preference label noise, and mismatches between overly challenging samples and model capacity. In this paper, we attempt to enhance the generalizability of reward models through a data-centric approach, driven by the insight that these issues are inherently intertwined from the perspective of data difficulty. To address this, we propose a novel framework, $\textit{Curriculum-RLAIF}$, which constructs preference pairs with varying difficulty levels and produces a curriculum that progressively incorporates preference pairs of increasing difficulty for reward model training. Our experimental results suggest that reward models trained with Curriculum-RLAIF achieve improved generalizability, significantly increasing the alignment performance of the policy model by a large margin without incurring additional inference costs compared to various non-curriculum baselines. Detailed analysis and comparisons with alternative approaches, including data selection via pretrained reward models ond self-selection mechanisms demonstrate the superiority of our approach in terms of simplicity, efficiency, and effectiveness.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the poor generalization of reward models in Reinforcement Learning from AI Feedback (RLAIF), which limits policy alignment due to distribution shift, label noise, and sample difficulty mismatches. 
This paper proposes Curriculum-RLAIF, a framework that builds preference pairs of varying difficulty and trains the reward model with the curriculum from easy to difficult pairs.

### Strengths
1. This paper provides a novel aspect to address the reward model generalization issue (distribution shift, noise in labeling, capacity limit to learn hard data) in RLAIF by a data-centric approach.
2. This paper provides substantial experiments, including ablation studies, to demonstrate the effectiveness of its framework.

### Weaknesses
1. This method solves the reward model generalization issue at the cost of training different reward models for different policy models, respectively, which means the reward model and its curriculum data cannot be reused for different policy models. Authors should include this limitation in the paper.
2. Evaluation metrics are win scores that are judged by LLMs, which can introduce bias. Automatic benchmarks without LLMs are encouraged to be included, such as IFEval and LiveBench.
3. Presentation of this paper, especially the Experiments and Results sections, is confusing. Terminologies, concepts, and notations are not consistent. See Questions 1 to 5.
4. The motivation of this method is confusing. See Questions 6 and 7.

### Questions
1. What is the definition of $\Delta r$? Is $\Delta r$ defined differently in the Experiments than its first definition in Line 153? I'm assuming the reward in Line 153 is ground-truth reward and you use TextEval-Llama3.1-70B to estimate it. But in the Experiments section, you seem to use another way to estimate it?
2. Internal Eval, External Eval, and Implicit Eval are confusing for the first time. They sound like evaluation metrics. I suggest changing the name.
3. What is the formula for Internal Eval, External Eval, and Implicit Eval? It would be better to write them down clearly or attach them in the appendix.
4. For these three baselines, did you also divide them into 3 difficulty levels, easy, bridge, hard?
5. What is the initialization of the reward model and policy model in Table 1? Does the base model in this paper mean the instruct model instead of a pretrained model?
6. Can you elaborate on "However, relying on such estimates at scale is computationally expensive, as it requires reward evaluation across all query-response pairs."? Do you mean the number of pairs by ${n}\choose{2}$? But it seems you only need to run inference for $n$ times to obtain $r(y_i), i=1, ..., n$ and the computation of the difference is LLM-free, which is cheap. 
7. A follow-up question for 6. "such estimates" is expensive, but auto-regressive sampling from the policy model is also expensive. I don't think the computational cost (FLOPs) of sampling from the policy model is cheaper than TextEval-Llama3.1-70B. If so, the motivation of the method is not valid.
8. Do the results suggest that Curriculum-RLAIF implicitly estimates $\Delta r$ better than Internal Eval?

I would like to raise the score to 6 if Question 6 and 7 can be addressed, and can further raise it to 8 if Question 1 to 5 can be resolved.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Curriculum-RLAIF, a framework for improving reward model generalizability in RLAIF through curriculum learning. Conventional reward models often suffer under distribution shifts, label noise, and varying sample difficulties. Their approach combines quality-aware sampling (guided prompting for easy samples, random sampling for hard samples) with controlled pairing to create preference data at varying difficulty levels. Reward models are trained progressively from easy contrastive pairs through intermediate "bridging pairs" to difficult random pairs. Experiments on harmlessness, helpfulness, and summarization tasks show consistent improvements over non-curriculum baselines across three tasks and two small-scale base models.

### Strengths
- Interesting preliminary study on the relationship between confidence of the reward model and the resulting reward distance. 

- The proposed curriculum method is simple and data-centric, reducing reliance on both noisy preference labels and costly online difficulty scoring during training.

- In Tables 1-2, they show consistent improvements across three alignment tasks (harmlessness, helpfulness, summarization) and two base models (Gemma-1-2B, LLaMA-3-8B), suggesting the method generalizes across different settings.

- The paper systematically isolates contributions from data sources (Table 2: Drnd vs. curated mixture) and curriculum orderings (Table 3: Cbrg vs. Crev, Cdis, Cmix, Cach). This supports their design choices.

- In Figure 3 and Figure 5, the reward-distance distributions show that curriculum stages progress from easy to hard with smooth overlaps between adjacent stages. This seems to show that the curriculum generally works. 

- The method requires expensive preference annotation (via off-the-shelf LLM) for only 25% of data (Drnd stage), while contrastive and bridging pairs are annotation-free. Table 4 shows a 75% reduction in labeling cost vs. conventional RLAIF.

### Weaknesses
**(1)** In Tables 1-2, they show that Curriculum-RLAIF outperforms strong baselines (Conventional RLAIF, Internal Eval) by 2-6 points. For example, on LLaMA-3-8B summarization, Conventional RLAIF achieves 0.84, Curriculum-RLAIF achieves 0.92. However, Internal Eval with curated data achieves 0.95. Without error bars or statistical tests, it is unclear if improvements are reliable. All results in Tables 1-5 lack error bars, confidence intervals, standard deviations, or multiple seeds. Given relatively small margins, it is not possible to assess significance. Bootstrap CIs over a larger number of test prompts or mean±std over multiple random seeds would help here.

**(2)** Limited task and model coverage:
- The authors only test dialog alignment (harmlessness/helpfulness) and summarization. No evaluation on reasoning tasks (math, code), instruction-following, or other domains where RLAIF is commonly applied.
- Only two small-scale backbones have been tested (2B, 8B parameters). No evaluation on larger models (30B+, 70B) or more recent reasoning-focused models (e.g., qwen-3, DeepSeek-R1-distill), where curriculum learning effects might differ. Broader model coverage would strengthen generalizability claims.

**(3)** The Method assumes that y+ ∼ p(y|x, g+) reliably beats y ∼ p(y|x) in D+brg and y ∼ p(y|x) reliably beats y− ∼ p(y|x, g−) in D−brg. However, there is no empirical validation of:
- How often does guided-good fail to beat random?
- How often does random fail to beat guided-bad?
- How is violated preference noise handled?
- Table 3 shows anchored curriculum Cach (which enforces preference via conditioning) achieves comparable performance, suggesting these assumptions may not be critical.

**(4)** All policy evaluations use GPT-4o as a judge (Sec. 4), and the robustness to alternative judges has not been unexplored. I don’t think this is a major problem. But it would be interesting to see an exploration of different judges here, who may differ in their preferences.  

**(5)** In Table 4, Appendix D, the authors only count the preference annotation cost, but ignore the response generation overhead. The guided sampling requires additional prompting (contrastive instructions). Moreover, no wall-clock times or FLOPs counts are reported. The Internal Eval may be more expensive than claimed due to repeated evaluations (9/4 N factor), but an actual runtime comparison is missing.

### Questions
1. Can you provide error bars or confidence intervals for all results? Specifically, bootstrap 95% CIs over the 1000 test prompts, or mean±std over random seeds for Tables 1-3 and Table 5? Which improvements remain statistically significant after accounting for variance?

2. For Drnd pairs, what percentage uses LLM labeling vs. human annotation? What are the inter-annotator agreement stats? What quality filters or checks were applied to CoT reasoning chains from LLaMA-3.3-70B? Any calibration analysis comparing LLM judgments to human judgments?

3. How sensitive are results to alternative judges? Can you provide cross-judge agreement analysis on a subset?

4. How does the approach perform on larger models (30B+, 70B parameters) or recent reasoning models? Moreover, does your method have value for other reasoning tasks (math, code) beyond dialog alignment and summarization?

6. What are the actual wall-clock times and total FLOPs for all methods, including response generation? How does guided sampling overhead compare to preference labeling cost in practice?

### Soundness
2

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
5

### Summary
The paper addresses a key limitation of Reinforcement Learning from AI Feedback (RLAIF) - the poor generalization of reward models, which negatively impacts downstream policy alignment.

To mitigate this issue, the authors propose Curriculum-RLAIF, a static curriculum strategy comprising three stages:
- Sampling completions for prompts using a mixture of guided (Yang et al., 2024) and random sampling,
- Pairing completions to form a preference dataset, and
- Applying an easy-to-hard curriculum over the preference dataset using coarse-grained stages.

Empirical results demonstrate that Curriculum-RLAIF improves policy alignment performance over both non-curriculum and curriculum-based baselines.

### Strengths
The paper is clearly written, with a well-motivated discussion of existing RLAIF limitations and a comprehensive review of related work.

The proposed approach is conceptually simple yet effective, showing consistent improvements across experimental settings.

The empirical analysis is systematic, supported by detailed ablations and comparisons.

### Weaknesses
The proposed Curriculum-RLAIF is a static curriculum strategy, while the internal evaluation baseline used in experiments is adaptive to the reward model's learning progress.

Table 2 suggests that the internal evaluation baseline could perform even better if applied to the full dataset (not just $\mathcal{D}_\text{rnd}$) and with more fine-grained curriculum stages. While adaptive strategies incur higher computational cost, they are likely to yield better final performance than static curricula.

It would be valuable to compare the proposed method and the adaptive baseline (allowing more fine-grained curriculum stages for the baseline) in terms of final alignment performance and total wall-clock time.

Additional clarifications and considerations (no new experiments needed):
- Constructing a full preference dataset $\mathcal{D}_\text{full}$, including all pairs of completions (contrastive, bridge, random) per prompt, may better capture prompt-level difficulty.
- The proposed approach uses an easy-to-hard ordering; however, examples of moderate difficulty can sometimes yield better learning efficiency.
- An adaptive variant could periodically sample prompts and select completion pairs with moderate reward gaps based on the current reward model (internal evaluation).

### Questions
Please see the discussions in the weaknesses section above.

In Table 1, it appears that curriculum baselines use only $\mathcal{D}_\text{rnd}$, while the proposed Curriculum-RLAIF uses the full dataset. In Table 2, baselines seem more competitive with the full dataset. Please clarify this explicitly in the table caption.

For both Curriculum-RLAIF and the curriculum baselines, how is the stage transition determined? Is it based on training for a fixed number of epochs per stage before moving to the next?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper tackles  the problem of training reward models with curriculum to improve generalization. First, the authors expose the fundamental challenges of learning from diverse quality data i.e. evaluating the difficulty of a pair of responses and show that reward differences from a pre-trained LLM across the responses are surrogates for label noise / difficulty. Further, using controlled data generation and prompt augmentation they generate a dataset of increasingly difficult preference pairs. Training with the introduced curriculum leads to better reward models and better policy performance after PPO finetuning. over baselines without a curriculum. Further, they ablate their curriculum design to show the contribution of the different components.

### Strengths
- The paper presents a thorough literature review and the problem statement is clearly laid out.
- It presents an interesting analysis on an existing dataset where the response pairs with low confidence scores (i.e. humans find them difficult to label) also exhibited lower accuracy when using an off-the shelf LLM for ratings. 
- The authors further show that existing models have a higher difference in predicted rewards which enables them to use this as a surrogate to evaluate response pair noise or difficulty.
- The method implicitly uses the way responses are generated to pair them up and create a curriculum of easy to difficult response pairs, which avoids the expensive LLM based evaluations.
- The authors carefully ablate different parts of the method, including the dataset sources and curriculum design. However, the ablation presents mixed results towards the utility of the proposed method (while it does outperform all the non-curriculum based methods).

### Weaknesses
- Intuitively it seems like since y1 and y2 are labelled using human annotations or a better model with context, they should have lower noise than pairing a bad sample with a random sample as the latter case uses less context to generate the preference ranking? So, this seems to be a wrong preference order? It would be interesting to evaluate the validity of the curriculum design against a stronger model or human evals.
- Table 2 shows that the proposed curriculum is not the dominant method and the base evaluation methods outperform Curriculum-RLAIF. Also the performance for the baseline curriculum methods in Table 1 is lower than the ones in Table 2? What is the data source for the results in the Table 1? And ideally it should be the best performing setting across each baseline that is presented in a single table.
- Overall, the paper is well written but the experiments do not provide strong evidence that the proposed curriculum method significantly outperforms the baselines specially the internal evaluation baseline if the preference data is a mixing of the paired and random pairs. If the authors can provide additional evidence, and clarify the above questions, I will be happy to raise my scores.
- Typo in table 1
- Looking at Figure 3, it seems like the Internal evaluation method creates distinct non-overlapping chunks of the dataset and should be prone to catastrophic forgetting but that does not seem to be happening. Is the curriculum training merging the new dataset?

### Questions
See weaknesses above.

### Soundness
1

### Presentation
2

### Contribution
2
