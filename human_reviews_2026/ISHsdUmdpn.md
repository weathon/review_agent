# Compute as Teacher: Turning Inference Compute Into Reference-Free Supervision

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Where do learning signals come from when there is no ground truth in post-training? We propose turning exploration into supervision through Compute as Teacher (CaT), which converts the model's own exploration at inference-time into reference-free supervision by synthesizing a single reference from a group of parallel rollouts and then optimizing toward it. Concretely, the current policy produces a group of rollouts; a frozen anchor (the initial policy) reconciles omissions and contradictions to estimate a reference, turning extra inference-time compute into a teacher signal. We also offer a way to turn such an estimated reference, generated with any inference method, into rewards in two regimes: (i) verifiable tasks use programmatic equivalence on final answers; (ii) non-verifiable tasks use self-proposed rubrics—binary, auditable criteria scored by an independent LLM judge, with reward given by the fraction satisfied. Unlike selection methods (best-of-$N$, majority, perplexity, or judge scores),  synthesis may disagree with the majority and be correct even when all rollouts are wrong; performance scales with the number of rollouts. As a test-time procedure, CaT provides large relative improvements on three instruction-tuned models: Gemma 3 4B, Qwen 3 4B, and Llama 3.1 8B (up to +27% on MATH-500; +12% on HealthBench). With reinforcement learning (CaT-RL), we obtain further gains (up to +33% and +30%) while using $9\times$ less test-time compute, with the trained policy surpassing the initial teacher.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Compute as Teacher (CaT). CaT serves two functions: (1) it can be used as an inference-time algorithm to improve performance at test time and (2) it can be used to scale inference-time compute for RL while also enabling label-free training. This is enabled by using an anchor model `\pi_0` to aggregate policy "exploration" rollouts into a single response `s`. Crucially, the anchor model is prompted in such a way as to allow for disagreements with the majority consensus. This single response `s` may be returned directly during inference, or may be used as a reference for label-free training during RL. Importantly, CaT works in both the verifiable and the non-verifiable domains, with the latter enabled by self-written rubrics generated via conditioning on `s`.

To support their claims, the authors demonstrate that CaT yields improvements when used for RL training and as an inference-time algorithm. This is done by assessing three different models on two different datasets: MATH-500 (verifiable) and HealthBench (non-verifiable). The authors also demonstrate that (1) `s` may disagree with the consensus decision, and may even yield final answers that are absent in the original rollouts, (2) CaT + RL is superior to CaT + SFT and (3) CaT scales with the number of rollouts `G`, thereby enabling effective scaling of inference compute.

### Strengths
- This work aims to address three important issues faced by the community at the moment: (1) how do we improve model performance when no labels are provided? (2) how do we improve model performance using RL on non-verifiable tasks? and (3) how can we effectively scale compute during RL training? I appreciate the authors' attempt to tackle all of these problems simultaneously.
- While using aggregation as a technique for inference-time scaling is not new, few works explore using aggregation for reference-free training. Furthermore, the works that do (that I know of) rely on majority-voting mechanisms for aggregation, but this limits learning to "self-distillation" of majority consensus, which (at first glance) should not enable the trained model to solve new problems that the base model cannot already solve. CaT overcomes this by allowing the model to synthesise new aggregations that may differ from any of the original rollouts, which I believe is a novel contribution.
- The writing and overall presentation is very clear.

### Weaknesses
While the proposed method is promising, my main concern is that the experimental evidence provided is somewhat limited. The authors only evaluate on two datasets: MATH-500 (verifiable rewards) and HealthBench (non-verifiable rewards): 
- MATH-500 is a relatively older and easier dataset that the base models (or at least, the instruct versions of those models) are able to attain very high scores on already: this is evidenced by the very small gap between "CaT"/"majority vote" and "single sample" for the strongest model (Qwen3-4B) in Figure 6, which suggests that consensus-based algorithms are hardly useful at all on this dataset to begin with. I would like to see further evidence of CaT's effectiveness on a larger number of and perhaps also more challenging collection of test sets (e.g. AIME, HMMT, LiveCodeBench etc.), where I believe your method should be more effective anyways. I would also like to see more RL baselines (e.g. RL with majority vote, RL with perplexity-based methods). I see related results in Table 6 but I assume these are for CaT at inference time rather than during RL - please let me know if I am mistaken.
- The authors train and evaluate CaT on the same dataset (page 6: "For MATH-500, we train and test on the same 500 questions..."). While I appreciate the fact that training is reference-free and so there is no risk of label memorisation, this still makes for an unfair compute-controlled comparison with the baselines: for a given test point `x_i`, you have already spent train-time compute "figuring out" (via exploratory rollouts) the correct answer to this point, which you then learn from via RL. As such, I believe the authors should report experimental results on a held-out test set in addition to the train set, or account for this in the baselines (by using relatively more test-time compute for those experiments).
- HealthBench is a domain specific dataset. While this is not an issue, I believe that the experimental evidence could be strengthened by including a more diverse set of instruction-following benchmarks (e.g. MT-Bench).
- The experimental results for HealthBench compare CaT with (1) RL + model-as-a-judge, where the model acts as a reference-based judge comparing `s` with each rollout and (2) RL + physician-written rubrics. The missing baseline here is RL with self-written rubrics that are independent of `s` (i.e. generate each rubric conditioned only on the question, as with existing rubric-related work), as this will tell us whether the effectiveness of CaT RL in the non-verifiable domain is due to the core CaT algorithm (generating `s` and using this as a pseudo-reference) or merely due to rubric-based training.

A second weakness is a lack of analysis demonstrating how CaT drives improvements: 
- The authors claim that CaT improves upon existing baselines by allowing the model to learn from references that may be different to anything in the rollout set/in the minority (Page 2: "Parallel rollouts diversify partial competencies and different generations surface different sub-facts or solution steps...", "Conditioning the anchor on the set of rollouts
enables ensemble-like error correction within the model’s generative space..."). They provide a percentage (14%) of times this occurs (at inference-time?), but I would like to understand (1) during RL, when this does occur, how often is `s` actually correct? (2) how does this fraction change over the course of RL? (3) does the model ever solve previously "unsolvable" questions in `s` (i.e. does the aggregation mechanism enable the model to attain correct answers that could not be generated using normal rollouts under the policy?). I ask this because I would like to understand whether the primary mechanism driving learning is simply “sharpening” through training on the consensus answer (albeit with some noise ~14% of the time), or whether gains are instead driven by these diverse, minority answers.
- On the plateauing of performance: you argue that performance plateaus due to diversity collapse in the rollouts. This suggests that main driver of performance is indeed sharpening (whether the modal answers are reinforced). If this is the case, how does CaT meaningfully differ from consensus based RL-training and inference-time algorithms?
- How do CaT's outputs differ qualitatively from its policy rollouts? e.g. length, diversity of answers, reasoning strategies used etc. This may shed some light on how CaT qualitatively drives improvements.

### Questions
For me to increase my score to an acceptance, I would like to see my point about limited experimental evidence addressed through additional experiments. I may further increase my score given additional analysis on how exactly CaT drives improvement.

Some additional questions:
- How many replicates are used for the error bars in Figure 6?
- One of the downsides to CaT is that all `G` outputs must fit into context when generating `s`. This may cause problems for problems where the rollouts are longer. How might you circumvent this issue?

### Soundness
1

### Presentation
4

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
This work present Compute as Teacher (CaT) to utilize a model's exploration/stochasticity through parallel rollouts and a synthesis step to bootstrap a teacher signal from a frozen anchor. They compare their approach to different selection methods on both MATH-500 and HealthBench with three different models and show strong test-time and RL results.

### Strengths
The motivation for the work is strong, and a synthesis prompt offers a simple approach for bootstrapping from the frozen anchor model. Additionally, the distinct advantage of synthesis (may disagree with majority or find the correct solution even when all rollouts are incorrect) is compelling.

### Weaknesses
I mainly found issue with the presentation of the results. The graphical representation, while visually appealing, is difficult to parse and compare the results of different methods. In addition, the claims made do not seem to be fully supported by the corresponding results. Below are my main issues.

- Somewhere in the main text or appendix the actual values should be reported in a Table. Throughout the paper it is difficult to tell how CaT compares to the baseline approaches. For example, the format from Figure 8 that shows the raw accuracy scores instead of relative improvements would be more helpful for parsing the results.
- Result 1 does not seem to be true on MATH-500, as the error bars seem to overlap for all models. Even on HealthBench, the improvement seems to be marginal. Can the authors provide additional information on the significance of this gain?
- Similarly, the results in Figure 4 and 6 seem to be the same results - why are these not consolidated into one Figure/Table? CaT also underperforms or performs similarly to Self-BoN on 4/6 of the dataset-model pairs, but the authors claim that "CaT is superior to all baselines." Could the authors also provide additional information on the significance of this gain? I do not see the benefit based on the collected results of the synthesis prompt over just performing BoN.
- The SFT result is also unclear - it seems to not help at all compared to the initial model. Do the authors have motivation for why this is the case?
- Why is Figure 7 analyzed over only two models, and why are they two different models? Do results 5/6 hold across all models or just Gemma 3 4B and Qwen 3 4B, respectively? For example, it seems Majority would perform the same as CaT in Figure 7 (right) with Llama 3.1 8B.
- "Error bars are standard error." Forgive me if I am misunderstanding, but is this calculated across multiple rollouts (i.e. the standard error for the Initial model in Figure 4 is computed across N rollouts, and CaT is computed across 8N rollouts)? I could not find mention of what N is.

### Questions
Most of my questions are listed in the Weaknesses section.
- Why do the authors utilize the word 'combine' in the freeform prompt and 'synthesize' in the reasoning prompt? Also I am curious how robust the final performance is to the synthesis prompt.
- The authors claim "disagreement with majority on 14% of questions, and disagreement with all rollouts on almost 1%." Could the authors expand on this result? For example, how frequently does CaT disagree with BoN? Can these disagreements be broken down into any categories to better understand the benefits of CaT?

### Soundness
2

### Presentation
2

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
This paper introduces Compute as Teacher (CaT), a method that converts inference-time compute into reference-free supervision. For each prompt, a current policy model generates multiple parallel rollouts, which are then synthesized by a frozen anchor model (the initial policy) into a single estimated reference answer. This synthesized reference serves as the supervision signal. The authors designed two regimes:
(a) **Verifiable tasks** (e.g., mathematics) — rewards are computed via programmatic equivalence checks; (b) **Non-verifiable tasks** (e.g., open-domain dialogue, health advice) — the model generates self-proposed rubrics, binary evaluation criteria scored by an independent LLM judge.

The paper tests both inference-time synthesis (CaT) and reinforcement learning with synthesized supervision (CaT-RL). Experiments across Gemma 3 4B, Qwen 3 4B, and Llama 3.1 8B on MATH-500 and HealthBench show substantial improvements. CaT outperforms selection and self-consistency baselines, demonstrating that reinforcement learning with synthetic supervision can improve beyond the initial synthesized teacher.

### Strengths
Reviewer found the following strengths of the paper:

**- Novelty and Conceptual Contribution**:
CaT proposes a new way to transform inference computation (different rollouts generated by the model) into self-supervision without human references, bridging selection-based self-consistency and self-play reinforcement learning. The idea of synthesizing multiple rollouts via a frozen anchor, rather than selecting one, is conceptually new. 

**- Experimental Breadth and Depth**:

The authors test both inference-time and training-time variants across multiple models and tasks, providing thorough empirical validation.
Ablation and mechanism analyses (e.g., scaling with rollout number, disagreement analysis) are comprehensive and align with claims.

**- Baseline Coverage:**
The comparisons span single-sample, best-of-N, perplexity, mutual predictability, majority voting, and SFT methods, etc, showing consistent and notable gains for CaT.

### Weaknesses
Though the paper is suffering by the following issues:

**a. Heavy Reliance on Prior Terminology and Concepts**: The paper is **difficult to follow because it assumes extensive prior knowledge** of recent LLM post-training methods, such as rubrics, self-editing, and test-time-compute. These concepts are used without sufficient background or intuitive explanation, especially in the **Introduction**. As a result, readers unfamiliar with recent self-training or LLM-as-a-judge literature may struggle to understand how CaT differs from prior work or why it is a conceptual advance. Such presentation styles thus weaken the clarity of the paper’s core contribution.

**b. Lack of Clear Intuition Behind the “Synthesis"**: While the synthesis of multiple rollouts plays as a core idea of the paper, **why** we should do it to yield a better reference is not fully justified. The authors instead mostly describe how synthesis is done but provide little intuitive reasoning or fundamental reasons to expect better results than selection or ensembling. Without a clear and convincing insight, **the synthesis idea feels somewhat heuristic or ad-hoc**.

**c. Reward Reliability in Non-Verifiable Tasks and Dependence on Anchor Model Quality**: Rubric-based rewards are promising but rely **heavily on the accuracy and stability of the LLM judge**. The paper indeed does not analyze the sensitivity to judge model choice or rubric quality over training iterations. Furthermore, the effectiveness of CaT critically depends on the frozen anchor model’s capacity to perform meaningful synthesis over multiple rollouts. If the anchor lacks sufficient reasoning or summarization ability, the synthesized output may simply mirror the same errors or contradictions rather than correcting them. In such cases, the supervision signal could become noisy or even misleading. The paper has not studied how anchor strength affects performance, leaving it unclear to the 
Review whether gains come from the CaT mechanism itself or from the anchor’s inherent capability.

**d. ** Compute and Efficiency Concerns**: Although the authors claim low overhead, the synthesis and rubric evaluation steps may introduce significant inference costs. Thus, quantitative analysis of the compute-to-performance trade-off is important to strengthen the argument.

### Questions
Given the above waekness (except the terminology presentation), here are the questions that the Reviewer be confused:

- What is the underlying intuition for why synthesizing multiple rollouts should produce a better answer than selecting the best one (e.g., via majority vote or confidence scoring)?

- How sensitive is CaT’s performance to the quality or scale of the frozen anchor model? If a weaker model is used as the anchor, does the synthesized reference still improve over rollouts, or does performance degrade?

- For non-verifiable domains, how stable are rubric-based rewards across different judge models (e.g., GPT-4o vs. smaller open models)?
Do rubric criteria drift or degrade as the policy improves over training?

- How does the additional inference and judging cost of CaT compare quantitatively to standard best-of-N or self-consistency methods?

### Soundness
3

### Presentation
2

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
The paper asks whether compute can serve as a teacher in environments where it is either hard to get post-training data or there is no ground truth. The authors demonstrate that this can be done using CaT and CaT-RL, which generate many rollouts and put these rollouts in context without the original question and ask the original model to synthesize the answers, upon which it is graded using either a self-generated rubric or verifiable rewards. In both cases, CaT does impressively in datasets and beats various baselines in a comprehensive set of analyses.

### Strengths
Paper is concise and extremely well written. Broader research question is well motivated within the literature and the past work sections help clear up how the paper uniquely contributes compared to other works. Improvements over baselines are high and results and analyses tell a cohesive and convincing story about how compute can serve as a reward signal.

### Weaknesses
My biggest complaint is that CaT, as constructed, seems a bit arbitrary in its various design decisions. I get that this isn't a problem if the overarching paper story is more towards whether compute CAN serve as rewards---as it is simply a demonstration that it can, but if the framing is more towards using CaT as a general framework then I have some issues with it. 

1. There isn't direct evidence provided by the authors that not including the question in the synthesis prompt actually does anything (or maybe I missed it?). The authors present an intuitive argument but never provide an ablation comparison on CaT but with including the question text in context along with the rollouts. Thus, it feels like an arbitrary decision that isn't justified if you want to propose CaT as a general framework. 

2. In the self-generated rubric, there are also some seemingly-arbitrary design decisions. For instance, limiting to yes/no questions might be less expressive than likert scale questions or similar finer-grained ratings. Why not try those instead if model capabilities permit this?

3. Another detail is that one could imagine doing the rollouts with different prompt rephrasings or conditioning on other differing information in order to encourage exploration, which the authors point out as an issue.

4. Lastly, simply aggregating by putting everything into context may not be the best at larger rollout numbers --- maybe asking the model to filter for diverse responses will yield better combinations?

Overall, these seem like design decisions that don't fully justify the authors' central focus of CaT as a general framework, but do answer the research question of whether you can use compute as training in general. Therefore I would personally favor that framing.

### Questions
1. "Since CaT resolves disagreements and omissions to produce better estimated references, it can no longer improve over the individual rollouts if they tend to agree too much."
Do the authors have any ideas on how to potentially alleviate this? Otherwise CaT seems like a one-time boost to any model family rather than a stable method to continuously improve models (which of course is still valuable, but the latter is much more extremely so). 

2. I don't really buy that CaT can meaningfully scale with rollouts beyond a certain extent; do the authors believe there will be settings where there are meaningful increases after a reasonably large magnitude (e.g., 20)? Maybe by then, a different aggregation rule would be more preferable?

### Soundness
4

### Presentation
4

### Contribution
3
