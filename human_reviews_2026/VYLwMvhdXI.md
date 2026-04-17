# Scaling Laws for Generative Reward Models

- Decision: Reject
- Scores: 8, 2, 4, 4

## Abstract
We study the scaling behavior of generative reward models (GenRMs) for reinforcement learning from AI feedback (RLAIF) when used as drop-in replacements for Bradley-Terry models to optimize policies. Building on established scaling laws for reward model overoptimization, we investigate whether GenRMs, particularly those employing chain-of-thought reasoning, exhibit different robustness properties as policies drift from their training distribution during gradient updates. Using the Qwen3 model family (0.6B--14B), our study includes systematic evaluation of thinking GenRMs (trained via GRPO) against answer-only variants (trained via SFT) across policy size, reward model size, reward model type, training budget, and the $\beta$ parameter in online DPO. Our results reveal a consistent evaluator-rewarder gap: thinking GenRMs outperform answer-only variants by 1--2\% on validation tasks, yet these gains diminish---and often reverse---during policy optimization, where answer-only GenRMs achieve higher Gold Elo and more stable proxy-Gold alignment. We find that reward model scale is the most decisive factor for policy quality, with gains continuing even when the GenRM far exceeds the policy in parameters. Moreover, intermediate GRPO checkpoints of thinking judges can outperform fully-trained checkpoints as rewarders, despite worse static accuracy. We track these dynamics with Elo arenas under both proxy and Gold evaluation, providing a fine-grained proxy--Gold alignment diagnostic beyond saturated validation metrics.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper studies the scaling laws of generative reward models, which outputs preference as tokens rather than scalar rewards. The authors studies various aspects, such as answer vs thinking, reward model size, policy model size, and the relationship between reward and policy model size. The main conclusion is that thinking GenRMs are better as verifiers, while answer GenRMs are better as rewards.

### Strengths
* The writing is pretty clear. The problem setup in section 2 is very easy to read, making it very clear even for reads not familiar with GenRM.
* The experiments are well conducted. The results are very clear and the authors summarized it very succinctly.

### Weaknesses
* A few things can be clarified which I list in the questions section.

### Questions
* Baselines refer to RMs that were not fine tuned, right? It's worth mentioning it in section 3 since the word "baseline" is almost never used in the paper until the results.
* What are the scatter points in the plots? 
* How to actually read these plots? Typically in reward over optimization studies such as [1, 2], the x axis is KL, training steps, etc and the y axis is gold reward. However in your plots the x axis is proxy rating, which makes it a bit hard to interpret the plots.

[1] [Gao, L., Schulman, J., & Hilton, J. (2023, July). Scaling laws for reward model overoptimization. In International Conference on Machine Learning (pp. 10835-10866). PMLR.](https://arxiv.org/abs/2210.10760)

[2] [Rafailov, R., Chittepu, Y., Park, R., Sikchi, H. S., Hejna, J., Knox, B., ... & Niekum, S. (2024). Scaling laws for reward model overoptimization in direct alignment algorithms. Advances in Neural Information Processing Systems, 37, 126207-126242.](https://arxiv.org/abs/2406.02900)

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates how GenerativeRMs scale in reinforcement learning from AI feedback. In creative-writing preference tasks, ‘thinking’ GenerativeRMs trained with GRPO outperform ‘answer-only’ ones as evaluators, but during DPO training, ‘answer-only’ RMs train policies with higher Gold Elo than ‘thinking’ RMs across model sizes,.

### Strengths
**\[S1\] Relevance to Core Challenges in LLM Alignment**

The paper addresses a timely and important problem in LLM alignment, exploring how generative reward models scale and behave during policy optimization

**\[S2\] Comprehensive Empirical Examination.** 

The work provides a broad empirical investigation across model scales and configurations, offering useful insights into the relationship between evaluator performance and policy optimization behavior.

### Weaknesses
**\[W1\] Misrepresentation of Background Knowledge**  
The paper mischaracterizes the relationship between GenerativeRM and conventional reward models (scalar-headed RM). These approaches differ primarily in how they generate reward scores and chosen-rejected pairs, not in their fundamental learning processes. Both ultimately utilize the Bradley-Terry function as their optimization objective. Therefore, the authors' explanation of GenerativeRM in the introduction lacks technical accuracy and is not convincing.

**\[W2\] Insufficient Experimental Design**

While the authors aim to analyze the scaling laws of GenerativeRM, their experimental setup contains several significant limitations:

1. **Robustness of the gold evaluator**: As noted in their own limitations section, the gold evaluator is a fine-tuned Qwen3-32b model with only 79% accuracy. This creates a substantial risk that the generative RMs are being overoptimized to match this particular model rather than demonstrating true performance improvements.  
2. **Lack of domain generalization**: The experiments are conducted to a single domain without OOD evaluation. This raises questions about whether the observed results would hold across different domains, limiting the paper's broader applicability.  
3. **Limited optimization methods comparison**: The exclusive use of DPO represents a methodological weakness. Including alternative policy optimization functions such as PPO and BoN would establish greater fairness and strengthen the validity of their findings.

**\[W3\] Conceptual Issues with Comparative Analysis**  
The comparison with answer-only GenerativeRM raises important conceptual questions. The core objective of GenerativeRM lies in its ability to use CoT reasoning / "thinking" capabilities to better evaluate responses. Training in an answer-only format effectively negates these advantages, making it conceptually indistinguishable from scalar-headed RM. A more meaningful analysis would include direct comparisons with conventional RM approaches to properly assess the claimed benefits of GenerativeRM.

### Questions
**\[Q1\]** DPO is fundamentally an offline methodology, so what is the justification for referring to it as "online DPO" in this work? This terminology appears inconsistent with the standard understanding of DPO.

**\[Q2\]** Line 152: The distinction between "human" and "gold" datasets remains ambiguous. The authors should provide precise definitions of these datasets, their composition, and the specific rationale for maintaining this distinction in the experimental protocol.

**\[Q3\]** Line 158-159: What’s the rationale behind re-annotating the LitBench using the gold model?

**\[Q4\]** Line 172-173: The response length appears quite short, which suggests responses might be getting cut off. Were there cases where responses were truncated prematurely? If so, this could significantly affect the evaluation quality, yet no analysis of this limitation was provided.

**\[Q5\]**  Which model was used as the policy in the experiments?

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper studies scaling laws for generative reward models (GenRMs) used as judges in preference-based alignment. Using Qwen3 models (0.6B–14B) and online DPO for policy training, the authors compare Answer-Only GenRMs (SFT, verdict token) to Thinking GenRMs (GRPO, rationale + verdict). On static, in-distribution evaluation, Thinking GenRMs modestly outperform Answer-Only (~1–2% accuracy). But during policy optimization, Answer-Only judges yield higher Gold Elo and appear more robust across sizes and $\beta$ settings; scaling the judge size consistently helps policies, often more than scaling the policy itself. The work introduces a unified Elo arena anchored to a Qwen3-32B “Gold” evaluator and reports rise-then-fall curves with $\sqrt(KL)$ consistent with over-optimization scaling laws.

### Strengths
- Clear evaluator vs. rewarder divergence: Thinking helps static judging but underperforms as a reward signal under matched budgets.
- Broad sweeps over judge/policy size and $\beta$; consistent PvG analysis and Elo reporting.
- Useful scaling-law framing (rise-then-fall vs $\sqrt(KL)$) and checkpoint analyses.

### Weaknesses
- Anchor bias / format asymmetry: The Gold evaluator is Answer-Only, which could favour Answer-Only proxies despite cross-judging. A thinking-style Gold or human adjudication would test robustness.
- Domain scope: Limited to creative writing; unclear if results extend to safety/helpfulness/dialogue or verifiable domains (math/code)
- Single backbone / algorithm: Only Qwen3 and online DPO are used; family/algorithm dependence remains open.
- Why thinking hurts remains mostly qualitative; more direct diagnostics would help.
- LLM usage clarity: It’s not fully clear whether any LLM assistance (e.g., for data filtering, rubric drafting, label consolidation, prompt generation) was used outside the main GenRM roles; if used, that should be documented for reproducibility. 
- Several citations are missing in the related work part: (?). Please fix them.

### Questions
- Could you evaluate with a thinking-style Gold (or a human-anchored subset) to check if the Answer-Only advantage persists under a different anchor?
- How sensitive are results to inference-time compute for the thinking judge during training (e.g., k rationales, majority vote)? If you increase k during training under matched compute, does the gap close?
- Can you replicate a subset with a second family (e.g., Llama/Mixtral)?
- Please clarify any ancillary LLM usage (data prep, rubric creation, filtering, related work). If none, state explicitly; if yes, list models, prompts, and safeguards for leakage.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates how Generative Reward Models (GenRMs) behave when used for reinforcement learning from AI feedback (RLAIF), particularly as replacements for traditional Bradley–Terry scalar reward heads. The authors study GenRMs trained to “think” (i.e., generate rationales before verdicts) using GRPO versus simpler answer-only versions trained with supervised fine-tuning (SFT), across varying scales of policy and reward models (0.6B–14B parameters) in the Qwen3 family. Results indicate that answer-only models can be more effective as evaluators.

### Strengths
Authors are tackling a relevant and interesting topic, and the experiment suite is abundant in ablations. It is particularly encouraging to see work on generating reward models on hard to verify environments and while I have concerns with the current state of the work I want to encourage the authors to keep pressing on this.

### Weaknesses
My main concern is, while the work is promising, I don't see that the experiments support the main conclusions that authors draw. Particularly:

* Prior to training, thinking models perform better than the answer-only models. Then authors perform two distinct training procedures for answer only models (SFT) and thinking ones (GRPO), an  then they test them with in-distribution data. Thus, this is not an apples-to apples comparison.  To get the conclusions valid authors should have tested both methods with the same training technique (or show evidence that each kind of model works better with a different one) and then test them with a new data distribution. Note that for the baseline models this was an out-of-distribution set. I think there is worth on the results that authors currently have, but the claims don't align.

* it was very contradictory where authors say they want to analyse the presumption that larger models reward better but the whole experiment is based on having a bigger model as gold evaluator

* It was not very clear what the ablations where trying to accomplish sometimes, would have been beneficial.

* Some broken references at the related lit section

### Questions
* What was the test-train split done on the original dataset?

### Soundness
1

### Presentation
2

### Contribution
2
