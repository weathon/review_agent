# WebSeer: Training Deeper Search Agents through Reinforcement Learning with Self-Reflection

- Avg Score: 5.67
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6, 8, 6

## Abstract
Search agents have achieved significant advancements in enabling intelligent information retrieval and decision-making within interactive environments.
Although reinforcement learning has been employed to train agentic models capable of more dynamic interactive retrieval, existing methods are limited by shallow tool-use depth and the accumulation of errors over multiple iterative interactions.
In this paper, we present WebSeer, a more intelligent search agent trained via reinforcement learning enhanced with a self-reflection mechanism. Specifically, we construct a large dataset annotated with reflection patterns and design a two-stage training framework that unifies cold start and reinforcement learning within the self-reflection paradigm for real-world web-based environments, which enables the model to generate longer and more reflective tool-use trajectories. 
Our approach substantially extends tool-use chains and improves answer accuracy. Using a single 14B model, we achieve state-of-the-art results on HotpotQA and SimpleQA, with accuracies of 72.3\% and 90.0\%, respectively, and demonstrate strong generalization to out-of-distribution datasets.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes WebSeer, a search agent capable of performing multi-turn, real-world searches for QA tasks. The authors introduce a two-stage training pipeline that includes SFT and RL. After training, WebSeer demonstrates improvements in both the number of effective tool calls and accuracy.

### Strengths
1. Training an autonomous search agent capable of conducting deep, strategic, and reflective searches addresses an important and challenging research problem, and also has significant practical application value.

2. WebSeer extends to real-world search scenarios and incorporates more reasonable tools, such as Webpage Reader, enhancing its applicability.

3. The method achieves SOTA performance on HotPotQA and SimpleQA benchmarks.

### Weaknesses
1. In the experiments (Table 1), WebSeer (a 14B model) is compared against baselines using smaller 7B/8B models. This raises uncertainty about whether the observed improvements result from the proposed method itself or simply from scaling up model size.

2. The SRRL method is a key contribution of the paper. However, Figure 5 (top) shows that SRRL only improves accuracy by 3.6% on HotPotQA and 2.9% on SimpleQA. Were the experiments conducted multiple times to assess variance? The relatively small gains make it unclear whether the improvement is statistically significant. 

3. A main feature of SRRL is that it allows the model to submit multiple answers within a single dialogue turn. However, there is no ablation study isolating the effect of this design choice. How would performance change under a conventional setting that allows only one submission during training?

4. At Line 241, the paper claims that its RL framework unifies SFT and RL under a self-reflection mechanism. However, Sec 2.4 does not describe how this unification is achieved. Based on the content, SRRL appears to directly apply GRPO for policy optimization.

5. (minor) Lines 182-183 contains repeated sentences.

### Questions
1. The paper emphasizes that WebSeer benefits from a self-reflection mechanism. But in Case study 1.2, the agent's outputs show no reasoning or reflection step. It directly generate tool calls in each round. Could the author clarify how self-reflection is implemented?

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
3

### Summary
This paper introduces a search agent designed for multi-step question-answering on the web. The agent's core innovation is a two-stage training framework: 1) using multi-turn rejection sampling, where a reasoner model's proposed answers are validated by a separate verifier, to generate a high-quality dataset of self-correcting reasoning paths. 2) using Self-Reflective Reinforcement Learning,  where the agent can submit multiple answers and receive textual feedback on its performance, allowing it to iteratively refine its strategy.

### Strengths
The research problem—improving the depth and self-correction capabilities of search agents—is interesting and important, as current agentic systems may struggle with error accumulation and shallow reasoning.


The proposed two-stage training framework, which combines supervised fine-tuning on curated reflective trajectories with reinforcement learning, is a logical approach to the problem.


The reward mechanism in the RL stage, which provides explicit feedback on answer quality and allows for multiple submission attempts, is a reasonable design choice.

### Weaknesses
Overall, the presentation of the methodology could be further improved.

For example, the paper's methodology relies on an independent "verifier" model (V) to generate the SFT dataset. The description of this verifier is insufficient (only mentioned between line 188 and line 197). It is unclear how the verifier is implemented. If the verifier itself is unreliable or follows an incorrect tool-use path, its judgments would be flawed, which could introduce noise or incorrect patterns into the SFT dataset and undermine the entire training pipeline. The paper needs to detail the verifier's implementation.


In addition, the term "F1-based feedback" is not standard in reinforcement learning literature. While the paper explains it as a feedback signal, it should be more formally defined as part of the observation space or reward function to avoid ambiguity. (It is possible that the authors refer to F1 score in classification. But according to line 253, it appears that F1 is calculated based on one predicted answer and one ground-truth answer, while the F1 score in classification should be calculated based on a set of predictions and GT answers.)


(Minor issue) The paper requires careful proofreading. For instance, lines 182 and 183 contain nearly identical, repetitive sentences: "To address this limitation, we propose a multi-turn rejection sampling method to collect reasoning paths that incorporate reflective patterns. To overcome this limitation, we propose a multi-turn rejection sampling method that collects reasoning paths enriched with reflective patterns."

### Questions
Please refer to weakness #1 and #2.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper tackles the challenge of training open-domain, web-based question-answering agents capable of multi-hop reasoning and robust self-correction. Existing web agents often stop prematurely, fail to verify their reasoning, or overfit to closed-corpus retrieval. The proposed system, WebSeer, aims to build deeper and more reflective reasoning trajectories by combining:

- Self-Reflective Supervised Fine-Tuning (SFT): multi-turn rejection-sampling trajectories verified by an independent verifier model, teaching the agent to backtrack and self-correct.

- Self-Reflective Reinforcement Learning (SRRL): a reinforcement phase where the model can submit intermediate answers multiple times, receive F1-based feedback, and continue improving under a shaped reward.

The final agent interacts with real web tools (search, webpage reader, code executor) and is evaluated across seven multi-hop QA benchmarks, achieving state-of-the-art performance and strong generalization to out-of-distribution datasets.

Contributions:

Introduces a two-stage self-reflection training pipeline that operationalizes “reason-verify-refine” behaviors in web agents.

Proposes a self-reflective RL framework (SRRL) that allows iterative answer submission and feedback-guided reasoning.

Designs a trajectory-level reward that balances answer correctness, brevity, and depth of reasoning.

Demonstrates substantial improvements over strong baselines (Search-r1, Qwen2.5-14B, etc.) on both in-domain and open-web QA datasets.

### Strengths
Clear motivation and novelty: The idea of explicit self-reflection integrated into both SFT and RL phases is well justified and fills a gap between shallow tool use and deep verification-oriented reasoning.

Strong empirical performance: WebSeer consistently outperforms baselines on seven benchmarks, with notable OOD generalization (e.g., +15.9 on NQ, +27.2 on 2Wiki, +4.0 avg over Qwen2.5-14B).

Well-structured training pipeline: The paper carefully separates data construction, SFT masking (to avoid overfitting to observations), and SRRL fine-tuning, providing a replicable framework.

Insightful analysis: The ablation studies show how self-reflective trajectories and multi-submission feedback both contribute to deeper tool usage and higher final accuracy.

### Weaknesses
Limited transparency of verifier reliability: The paper assumes the verifier’s correctness predicate $\psi$
 yields high-quality reflective trajectories, but quantitative verifier accuracy or error propagation analysis is missing.

Reward design justification: The choice of exponential discount on multiple submissions is somewhat ad-hoc; an analysis of sensitivity to discount parameters or exploration–exploitation balance would strengthen claims.

Computational cost: The reflective data collection (multi-turn verifier rejection sampling) seems expensive; runtime statistics or data efficiency analysis are lacking.

Ablation on real-web vs synthetic setting: While generalization results are good, it remains unclear whether the model trained with restricted search data directly scales to dynamic, noisy real-web environments.

Clarity and formalism: Some algorithmic details (e.g., GRPO vs DAPO implementation differences) could be more rigorously formalized.

### Questions
How accurate is the independent verifier during trajectory construction? Did you measure verifier precision/recall or its effect on SFT data quality?

How sensitive is WebSeer’s performance to the number of verifier retries 
$K$ and the validity predicate threshold?

In SRRL, how did you balance the trade-off between early termination (short trajectories) and the benefit of deeper reflection?

Could the approach generalize beyond QA (e.g., web-based decision-making or scientific reasoning tasks)?

What is the relative wall-clock cost of SFT vs SRRL stages, and is RL training feasible for larger backbones (e.g., 70B models)?

Would a joint training of verifier and reasoner be possible or beneficial compared to separate optimization?

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
3

### Summary
The paper introduces WebSeer, a web search agent that learns to build deeper and more reliable tool chains for multi hop questions. It first collects long reflective trajectories by rejection sampling for supervised fine tuning, then runs a self reflective RL stage where the model can resubmit an answer, read feedback, and refine its plan. The agent uses a small toolset for search, page reading, and code execution, and it treats submitting the final answer as a tool. Training is site restricted on Wikipedia, while inference runs on the open web. A 14B model beats strong baselines on seven benchmarks and also does well on FanOutQA, FRAMES, SimpleQA, and Bamboogle. Taken together, the two stage recipe turns shallow tool use into steadier and longer chains and lets one model plan, act, verify, and decide when to stop. It also shows that learning in a clean, restricted setup can carry over to the open web with strong gains.

### Strengths
- Clear and practical idea: learn long, reflective tool use in a clean Wiki-only setup, then run on the open web. The two-stage SFT → SRRL recipe is simple, and “submit answer” is treated as a tool, which makes stopping explicit.
- Strong results across many benchmarks and tough OOD sets, with consistent gains over strong web-agent baselines.
- Good behavior analysis and ablations that explain why it works: tool-call depth becomes more balanced, removing SRRL hurts, and the SFT data mix matters.
- Interfaces and evaluation are well specified, with concrete tool APIs and LLM-as-a-judge prompts, plus transparent training and compute details.
- Meaningful impact: the method makes web agents plan deeper and verify better, and the recipe is easy to adopt in other agent settings.

### Weaknesses
- The open-web transfer story is promising but the ablations are only partial. The paper already shows that limiting RL to a single submission hurts, analyzes tool-use distributions, and studies how the SFT data mix changes behavior. What’s missing are comparisons across training regimes (restricted vs mixed vs open), a sensitivity check for the page-reading/normalization stack, and systematic curves that relate accuracy to chain length under different tool budgets, search-depth caps, and submission-discount settings. A short discussion or small exploratory study on these fronts would make the restricted-to-open claim feel more complete.

- The SFT verifier is under-specified and hard to reason about. We don’t know the model family/size, the prompt template and decoding settings, or the retry budget. It’s unclear how it accepts or rejects a candidate (string match vs semantic check), how it handles aliases and partial answers, or whether it uses the same tools and constraints as the agent. We also don’t see rates: how often it disagrees and forces a re-check, what fraction of trajectories get kept, how long those trajectories are, and how much of the SFT set ends up single-step vs multi-step. Without these basics, reproducibility is limited and it’s difficult to tell how much the verifier shapes the data distribution and later RL stability.

### Questions
- Restricted to open transfer
Did you consider building a training set that mixes a small portion of open-web episodes with wiki episodes, then run SFT and also SRRL on the mixed set? If yes, how did accuracy, chain length, and number of submissions change as the share of open-web episodes grew? If not, what behavior would you expect and why?

- SFT verifier details
What model do you use for the verifier? What is the prompt, decoding setup, retry budget, and timeout? What rule accepts or rejects a candidate answer, and does the verifier use the same tools and constraints as the agent?

- Verifier impact
How much does verifier stringency change the SFT distribution and later RL stability? If you lower or raise the verifier bar while keeping SFT size fixed, how do single-step share, tool-call patterns, and final accuracy move?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes WebSeer, a tool-use agent trained with reinforcement learning (RL) that incorporates a self-reflection mechanism. Specifically, the agent is equipped with searching, webpage reading, and code execution tools. The training process contains a supervised fine-tuning (SFT) stage followed by an RL stage. This RL stage mostly follows GRPO, but is designed to encourage self-reflection. The resulting 14B model achieves state-of-the-art results on several multi-hop question answering datasets and demonstrates strong generalization to out-of-domain datasets.

### Strengths
1. The paper tackles the relevant and interesting problem of training tool-use agents. It presents a solid approach based on RL, with a notable mechanism for encouraging self-reflection (though some technical details remain unclear, as noted in the questions below).
2. The empirical performance is strong. The model outperforms baselines on most target datasets and also demonstrates good OOD generalization.
3. The analysis in Sec. 3.3 provides helpful insights on behaviors of tool-use agents trained with RL and empirical lessons on how to train them.

### Weaknesses
1. Limited discussion of related work: There is a significant body of work on using RL for general tool use, not limited to search [1,2,3]. The paper would be stronger if it more comprehensively discussed and positioned itself relative to this broader line of related work.
2. Narrow experimental scope: The evaluation is focused on multi-hop question answering. While I think this is acceptable and does not undermine the paper's core contributions, showing the framework's effectiveness on other types of tasks would make the work even more impactful. If time and resources do not permit, it would be great if the authors can include some discussions.

[1] ReTool: Reinforcement Learning for Strategic Tool Use in LLMs
[2] Tool-Star: Empowering LLM-Brained Multi-Tool Reasoner via Reinforcement Learning
[3] Synthetic Data Generation & Multi-Step RL for Reasoning & Tool Use

### Questions
Questions regarding the RL algorithm:
1. In SRRL, if a rollout achieves a low reward in an earlier step but a better reward in a later step (via self-reflection), how is the advantage calculated? Is it based only on the final reward, or does it consider every reward achieved throughout the trajectory?
2. Alternatively, is every answer submission considered a separate rollout? (e.g., separate trajectories sharing initial steps but with different ending steps/submission times).
3. Following 2., if not, does it mean the model will generate a trajectory containing multiple answer submissions? In that case, how do you determine when to end an trajecotry during inference?
4. Could you elaborate on how the model's behavior differs between SRRL and regular GRPO? More analysis or examples here would be helpful.

Questions regarding experiments:
1. What is the specific base model used for training (e.g., Qwen2.5-14B or Qwen3-14B)?
2. It appears the "Qwen3-14B w/ Web Search tools" baseline is missing from Table 1. Can you add the results?
3. Figure 5 is difficult to interpret. It might be clearer to present a direct numerical comparison between your model vs. the no-SFT baseline, perhaps in a format similar to Table 5.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 6

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents WebSeer, a novel approach for training LLM search agents, i.e. agents able to perform iterative calls to Web search engines to answer a given query. The approach consists of two main stages: 1) supervised fine-tuning (SFT) on high quality multiturn search interactions,  and 2) a novel self-reflective reinforcement learning (SRRL) pipeline. The approach is used to finetuning Qwen 14B and achieves SOTA benchmark scores for it's size when given access to Web search.

### Strengths
- The paper studies a relevant problem: Tool calling (such as Web searches) is a very timely and challenging area of research, unveiling a lot of potential for LLMs. 
- The presented WebSteer approach is sound and, to the best of my knowledge, novel.
- The experiments ablate two key choices made: 1) self-reflection and 2) the cold-start initialization via SFT, both proved to be beneficial.
- Trained LLMs reach SOTA benchmark scores for their sizes.

### Weaknesses
- The analysis claims that the method's effectiveness is heavily dependent on model capacity. Only the 14B model showed consistent performance and stable behavior improvements, while smaller 3B and 7B models showed degraded performance after SFT and instability during RL. I find this quite puzzling and would like to know if the authors have further explanations on why this is the case. In particular, some qualitative example after SFT or during RL would further illustrate the cause. 

- The ablations focus primarily on the training methodology (SRRL, cold-start). It would be beneficial to see further ablations, e.g., on the impact of the reward design components ($R_{format}$ and $R_{correct}$).

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
