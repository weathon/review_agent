# Training Long-Context, Multi-Turn Software Engineering Agents with Reinforcement Learning

- Avg Score: 3.60
- Decision: Reject
- Scores: 4, 2, 4, 6, 2

## Abstract
Research on applications of reinforcement learning (RL) to large language models has mostly been focused on single-turn problems, such as mathematical reasoning or single-shot code generation. While these problems can be viewed as token-level multi-turn Markov decision processes (MDPs), this view corresponds to a degenerate case of multi-turn interaction where the environment provides no feedback. This contrasts with many real-world domains, such as software engineering (SWE), which require rich multi-turn interactions with a stateful environment that responds to each action with a non-trivial observation. To bridge this gap, we demonstrate the successful application of RL to this general regime. Our methodology begins with rejection fine-tuning (RFT) using execution feedback to train a policy to follow instructions and formatting effectively, followed by a synchronous RL pipeline using DAPO for iterative improvement. Applying this pipeline to Qwen2.5-72B-Instruct, we increase its Pass@1 on the SWE-bench Verified benchmark from 11% to 39%, substantially improving upon the 20% RFT baseline. On the May and June splits of SWE-rebench, the resulting agent achieves Pass@1 of 35% and 31% respectively, competitive with even larger models such as DeepSeek-V3-0324 or Qwen3-235B-A22B, demonstrating that our methodology offers a practical approach for training capable agents for multi-turn interactive tasks using open-weight models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents a comprehensive RL pipeline for training long-context, multi-turn software engineering agents. The authors employ a two-stage training process: (1) rejection fine-tuning (RFT) for instruction and formatting alignment, and (2) DAPO-based reinforcement learning for iterative improvement. The proposed agent significantly outperforms the baseline and competitive with larger models.

### Strengths
1. The paper demonstrates that RL can substantially improve the capability of SWE agents in multi-turn, interactive settings.
2. The integration of RFT and DAPO is practical and carefully tuned for stability under long context lengths.
3. Results on multiple benchmarks and comparisons prove good results.

### Weaknesses
1. The method mostly applies existing components (RFT + DAPO) to a new domain. It brings the potential risk of lacking the conceptual or algorithmic innovation beyond careful system design.
2. The empirical study could be strengthened by isolating the effects of individual components proposed in this work.
3. The authors treat SWE-ReBench as an interactive environment, the model is repeatedly trained within the same repository. This setup may lead to environment overfitting, where the agent learns to exploit the reward dynamics of SWE-ReBench rather than genuinely generalizing to unseen repositories. The authors are recommended to evaluate on unseen environments or cross-repository splits to validate generalization.

### Questions
Please refer to the weakness section. My main concern is the incomplete ablations and generalization studies.

### Soundness
3

### Presentation
2

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
Authors introduce a post-training pipeline for training a software engineering (SWE) agent. They construct the training dataset by carefully filtering SWE-Rebench dataset. Then, authors employ two-stage training. The first stage is Rejection Fine-tuning: authors generate 10 trajectories per prompt, and SFT on successful trajectories. In the second stage, DAPO is employed with the outcome feedback; length penalty is added to the reward. These methods make a significant improvement on the well-established SWE-Bench Verified benchmark.

### Strengths
Quality: The proposed method can serve as a useful, clean baseline for future research on SWE tasks. Unlike many of the previous papers, authors make minimal dependency on proprietary LLMs for data generation, therefore there is neither dependency the capability of these LLMs nor  terms-of-service issues with proprietary LLMs. Also, authors employ well-established techniques: two-stage post training pipeline with SFT and RL, DAPO algorithm for RL, length penalty on the reward. Adoption of standard techniques provide bigger confidence on the reliability of the findings. Also, this makes it easier for the proposed method to adopt subsequent methodological improvements. Therefore, the quality of the execution is notable.

Significance: As mentioned above, the paper could establish a clean baseline for the task, which will be significant. The performance improvement, however, is not significant as comparable results can be achieved with better scaffoldings.

Originality: RL on SWE tasks is well-established. Multi-turn RL might've been absent at the time of submission, but there are concurrent work such as CWM https://ai.meta.com/research/publications/cwm-an-open-weights-llm-for-research-on-code-generation-with-world-models/ . In any case, I don't think ICLR readers would be surprised that RL on multi-turn SWE task works, as RL on multi-turn agentic tasks are already common.

Clarity: The paper is straightforward to read and understand.

### Weaknesses
The work is not presented in a way that highlights authors' main contributions. First five pages are devoted to literature review and problem formulation. While I appreciate authors' attempt to make the paper self-contained and rigorous, this method of presentation is at the cost of not leaving much room for authors to discuss their key contributions. For example, I find PPO->GRPO->DAPO discussion to be quite tangential to authors' key contributions; authors could simply say they use DAPO and move on. 

From my perspective, key technical innovations of this paper is majorly 1) how SWE-Rebench data were carefully processed, and 2) how various hyperparameters of RL were set for strong results, given authors use standard methodology for the rest (DAPO, 2-stage training, etc). Unfortunately, these contributions are briefly described without detailed analysis, discussion, and ablation.

### Questions
Line 269: which LLM was used for the quality estimation, and what was the prompt?

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper deals with a common limitation in current reinforcement-learning (RL) application for large language models (LLMs); that is, most prior works focus on single-turn tasks (e.g., math or single-shot code generation), ignoring the multi-turn nature of tasks such as software engineering.
In this paper, a two-phase training pipeline combining rejection fine-tuning (RFT) with following RL via dynamic sampling policy optimization (DAPO) is proposed, and the effectiveness of this pipeline is demonstrated by empirical results on Qwen2.5-72B-Instruct, which supports long-context use case.

### Strengths
1. The research topic is clear, practical and timeliness. 

The research gap that most LLM-with-RL studies focus on single-turn tasks, while many real-world tasks like SWE require long-horizon and interactive reasoning. To address this gap, this paper provides a reasonable design for the practical SWE scenario.\\

2. Strong empirical results. 

The result of experiments shows great improvement on the selected base model (14.5% -> 36.5%), even comparable with larger model (e.g., DeepSeek-V3-0324) and commercial model (Claude Sonnet 3.5, gpt-4.1-2025-04-14) as reported in this paper and external reference (https://swe-rebench.com/).

### Weaknesses
1. Limited novelty in algorithmic design.

Both RFT and DAPO are not new. Previous work, such as Yuan et al., (https://arxiv.org/abs/2308.01825) has leveraged this technique to improve reasoning ability, and DAPO is from Yu et al., as cited. Some modifications, such as turn penalty, are more engineering rather than a theoretical innovation

2. Limited ablation to each component.

Although in Table.1, we can see improving performance as adding more stages, the effect of each component in the training pipeline is not fully discovered. For example, what if the RFT phrase or stage 1 in RL phrase is removed? By adding these comparisons, the necessity of each component will be more convincing.

3. Limited comparisons in perspective of RL algorithm and base model.

On one hand, In 3.2, we see a clear introduction of DAPO, but what makes DAPO better than GRPO or PPO  in SWE tasks remains unclear. Plus, how effective the modification to length penalty in DAPO is expected to be seen. Therefore I suggest an algorithmic analysis or experiment added in this paper.

On the other hand, though we see the improvements on Qwen2.5-72B-Instruct, two questions are naturally raised, why choose this model (any reasons in architecture or size?), and, what if we apply the proposed pipeline to another model? 

4. Ambiguous connections among challenges, the proposed method, and findings.

Although in line 94, the proposed pipeline is claimed to address aforementioned challenges, I cannot directly link to how this pipeline becomes a solution, especially in the problem of sparse rewards. Further, in the second paragraph of 5.2, the authors attribute the degrading performance to distribution mismatch and claims performance will recover if sampling is unbiased. Please justify this issue deeper  and point out how the proposed training pipeline recovers the biased gradient updates.

### Questions
## Questions
1. The spec. of Qwen2.5-72B-Instruct shows the maximum of context length is 128k, but many results are tested on data with length up to 131k. Did you do any modification like truncation? Plus, you mentioned context parallelism in the appendix. Does this technique help deal with data with length exceed 128k?

2. In 4.2, the RFT yields 6548 trajectories. Do these come from 7249(tasks) * 10(runs)? and if so, how many tasks remain in this stage?

3. In line 314, you mentioned thousands of problems, are these data same as RFT's, filtered data (7249 tasks), or others?

4. In table 2, line 418, SWESynInfer-72B used the same base model as yours, but their "before" score is much higher than 11.4%. I founded they did maybe extensive prompt engineering, so I am wondering what if the propsed pipeline starts from this setting?

## Suggestion
Beside the suggestions appended on weaknesses,
1. More models with similar size can be compared as well. In Table.1, we see many rival model with various size other than 72B, and in Table 2, most models have 32B parameters. Therefore, a 32B model is suggested toward a fairer comparison.

------
If most of the weakness and questions are solved, I would be willing to increase my ratings.

### Soundness
2

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
3

### Summary
This paper addresses the challenge of applying reinforcement learning (RL) to large language models (LLMs) for multi-turn interactive tasks with stateful environments, a setting more aligned with real-world domains such as software engineering (SWE). The authors propose a methodology combining rejection fine-tuning (RFT) with execution feedback and a synchronous RL pipeline utilizing DAPO for iterative improvement. Using Qwen2.5-72B-Instruct as the base model, the pipeline achieves significant performance improvements on SWE-bench Verified and SWE-rebench benchmarks, showcasing the potential of the approach for training capable agents for multi-turn tasks.

### Strengths
1. The use of RL to train agents for multi-turn, stateful interactions is highly relevant for advancing LLM-based applications in real-world domains such as SWE. The focus on a structured RL pipeline for this problem is valuable and timely.
2. The combination of RFT and DAPO appears to be effective, as evidenced by the substantial improvements in Pass@1 scores on SWE-bench Verified and SWE-rebench. These results demonstrate the practicality of the approach, particularly when using open-weight models like Qwen2.5-72B-Instruct.
3. The experiments are built on a strong base model (Qwen2.5-72B-Instruct) and include comparisons with competitive benchmarks (e.g., DeepSeek-V3-0324, Qwen3-235B-A22B). The results are well-documented and provide convincing evidence of the proposed method's effectiveness.
4. The use of DAPO, particularly in the context of RL for SWE tasks, appears well-motivated and thoughtfully integrated into the pipeline.

### Weaknesses
1. While the results are promising, the novelty of DAPO as a contribution is not entirely clear. The paper would benefit from a clearer comparison with concurrent or prior approaches to clarify how DAPO differs from related methods. This is particularly important because RL for LLMs is a rapidly evolving field, and comparisons with recent work may be necessary to establish the significance of the contribution.
2. The paper's presentation could be improved. For example: The novelty of the work is not emphasized strongly enough in the introduction or methodology sections. It would help to explicitly highlight what aspects of RFT + DAPO are novel and how they advance the state-of-the-art. Some information feels basic or redundant. For instance, Figure 1 occupies a large amount of space but does not provide substantial insights beyond what is described in the text. Consolidating or reformatting such figures could make room for more detailed explanations of the methodology or comparisons with related work.

### Questions
1. How does the method scale with even more complex environments?
2. What are the computational costs of the RFT + DAPO pipeline compared to baseline fine-tuning or RL methods?
3. Are there specific failure modes observed during training or evaluation?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper trains a long-context, multi-turn SWE agent with a two-phase recipe: (1) Rejection fine-tuning (RFT) on self-generated, test-passing trajectories to fix tool-use/formatting issues; (2) synchronous on-policy RL with DAPO wih turn-level length penalty. Starting from Qwen2.5-72B-Instruct, they report Pass@1 on swe-bench verified improvement (from 11% to 39%) and swe-bench May/June respectively.

### Strengths
- This paper presents a clear, end-to-end agent training for multi-turn SWE and clear problem formulation with POMDP. Also, the presented two-phase recipe is standard yet effective, RFT follows by RL.
- I like the transparency of the good engineering and the negative results about the decoding mismatch part. The paper also cautions against decarding over-long trajectories (could hide looping and make the model not-generalizable). This provides good valuable guidance to the commnuity and rarely documented.

### Weaknesses
- The core ingredient: Rejection Sampling Finetuning (RFT) followed by on-policy RL is a well-executed application to various tasks with verifiable reward such as math reasoning task and single-turn/multi-turn code generation task. The algorithm design choice DAPO/GRPO is known and the new bit is tailoring the reward shaping to turn count and getting a big model to 131k context length stably. While the application and scaling are novel and successful, I’m missing a sharper “what’s truly new vs. engineered best-practice” positioning. 
- There's very limited ablations study to justify the algorithm choice or the pipeline, for example ablating the proposed length penalty and varying different group size G. The point I mentioned in the Strength, though interesting as a read, lacks concrete number and experiments for backing:
  - How the training looks like for the decoding mismatch?
  - What's the perf when you decard over-long trajectories compared to the main exp. in the paper? Empirical evidence of model not being able to generalize to such looping trajectories in test time and maybe provide some stats and analysis?
  - The main selling point is the long-context training, could the author also provides test time performance when under different seq length to justify the effectiveness?

Overall, I read this paper as closer to an interesting tech report and a lot of claims are floating around without backing numbers and experiments. That being said, more empirical evidence are needed to justify the claims in the paper. This my main motivation of not recommending an acceptance.

### Questions
Please see above.

### Soundness
2

### Presentation
3

### Contribution
2
