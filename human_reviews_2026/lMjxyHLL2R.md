# ABBEL: LLM Agents Acting Through Belief Bottlenecks Expressed in Language

- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
As the length of sequential decision-making tasks increases, it becomes computationally impractical to keep full interaction histories in context. We introduce a general framework for LLM agents to maintain concise contexts through multi-step interaction: Acting through Belief Bottlenecks Expressed in Language (ABBEL), and methods to further improve ABBEL agents with RL post-training. ABBEL replaces long multi-step interaction history by a belief state, i.e., a natural language summary of what has been discovered about task-relevant unknowns. Under ABBEL, at each step the agent first updates a prior belief with the most recent observation from the environment to form a posterior belief, then uses only the posterior to select an action. We systematically evaluate frontier models under
ABBEL across six diverse multi-step environments, finding that ABBEL supports generating interpretable beliefs while maintaining near-constant memory use over interaction steps. However, bottleneck approaches are generally prone to error propagation, which we observe causing inferior performance when compared to the full context setting due to errors in belief updating. Therefore, we train LLMs
to generate and act on beliefs within the ABBEL framework via reinforcement learning (RL). We experiment with belief grading, to reward higher quality beliefs, as well as belief length penalties to reward more compressed beliefs. Our experiments demonstrate the ability of RL to improve ABBEL’s performance beyond the full context setting, while using less memory than contemporaneous approaches.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes regularly asking a language model to express its current belief state in natural language as a way to alleviate degradation caused by context length in long interactions. First, it evaluates frontier models in their ability to summarize context sufficiently into a belief state, identifying some failure modes like propagation of incorrect beliefs, and repeating uninformative actions because they don't reflect as changes in belief states. Then, it uses GRPO (group size 2) train LLMs to better generate and use belief states. Direct training improves LLM use of belief states, but does not match full context training. To further improve belief states, the paper proposes belief grading rewards, which can then outperform full history training on simple multi-turn tasks.

### Strengths
1. Training LLMs to produce and use summaries of context to alleviate long-context issues and costs is an interesting and important direction.

2. The analysis on how belief state compaction, while somewhat effective, still has failures modes without training is helpful, and creates a clear logical flow.

3. It is good that the paper considers both outcome reward training, but also a stronger (albeit seemingly very task specific) way to grade the beliefs themselves.

### Weaknesses
1. The core idea of summarizing LLM context has a long line of prior work [e.g. 1,2,3] which is not discussed and should be implemented and compared as baselines in this work. The only baseline compared is Mem1, however the methodology and results here are questionable. For example, the baseline is not reproduced in the paper's evaluation setting, and instead numbers are taken directly from the paper. Figure 7(b) is referenced as showing Mem1 states being "much longer", but actually has no Mem1 results! In fact, in 7(c), Mem1 seems to have lower token usage. 

2. The "best" method, "belief grading" is not described clearly. It seems like it involves using a much more capable model to perform analysis very specific to the Combination Lock task (simplified version of wordle with just 3 digits). How generalizable is this method? Why was it not used in the multi objective QA environment? This is important because without belief grading, the RL method performs worse than full context RL on the task. Perhaps switching to a more suitable and realistic task which actually tests long context capabilities would be better to demonstrate the benefits of outcome reward RL for generating and using belief states?

[1] LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression
Huiqiang Jiang, Qianhui Wu, Xufang Luo, Dongsheng Li, Chin-Yew Lin, Yuqing Yang, Lili Qiu.

[2] MemGPT: Towards LLMs as Operating Systems
Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G. Patil, Ion Stoica, Joseph E. Gonzalez

[3] Recursively Summarizing Enables Long-Term Dialogue Memory in Large Language Models
Qingyue Wang, Yanhe Fu, Yanan Cao, Shuai Wang, Zhiliang Tian, Liang Ding

### Questions
Main question: How exactly is belief grading implemented, and how generalizable is this methodology for other, realistic tasks? Which component of it helps the most, the supervision signal from a much stronger model, or something else? How does it compare to distillation from the stronger model?

Suggestion: Figure 7 is hard to read due to the extremely thin bars. Consider making the X axis (number of objectives) log-scale.

Suggestion: The paper keeps switching environments across sections by just citing the works that proposed them. It would be helpful for readers if the paper is self-contained, and each section begins with a clearer description of the environment being used.

Nitpick: "multi-objective QA" is called "realistic" in the discussion, when its a highly synthetic task obtained from randomly combining QA from existing retrieval datasets. I would refrain from calling this environment realistic throughout the paper.

Question: Why were the same environments not used throughout? For example why not train/test for the environments used in Figure 4, which are much more interesting than Combination Lock?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
- This work proposes using prompt-based belief states to summarize long-context information from LLM-user interactions, and selecting the next actions based only on these belief states rather than conditioning on the full context.
- Shows that without training, the proposed belief-bottlenecked approach ("ABEEL") under-performs compared to full-context or belief-prompting baselines (where both the context and the belief states are available).
- Shows empirically with Qwen 7B-Instruct that RL post-training has some potential to improve ABEEL compared to ABEEL without training on combination lock and the multi-objective QA task.

### Strengths
- Improving the performance of LLMs in multi-turn interactions is an interesting problem, but this reviewer is not fully convinced of the novelty or significance of this work due to limited empirical demonstrations (see Weaknesses).
- Clarity: The writing is clear, and Figure 1 clearly illustrates the difference between ABEEL and the existing approaches (vanilla and belief prompting).

### Weaknesses
1: Questions about the effectiveness of belief-bottlenecked policies: 
- Figure 2 shows that belief-bottlenecked models perform significantly worse than full interaction-based models or models that incorporate both belief and past history. Given the efforts and advances in increasing context lengths for newer models, it is unclear what advantages belief-bottlenecked models offer that long-context models cannot handle. The performance improvement achieved through post-training (when comparing among belief-bottlenecked policies) is insufficient to establish their effectiveness in multi-turn settings, especially if they are still outperformed by existing long-context models. 
- The results would be more compelling if there were some domains that long-context models cannot handle but belief-bottlenecked models can.

2: Questions about generalizability across model types and sizes:
- For the combination lock and multi-objective QA tasks, only one model (Qwen-7B-Instruct) is trained to demonstrate ABEEL and ABEEL-Length Penalty. 
- The results would be more convincing if the same experiments were repeated with different models and model sizes to show that the effectiveness of post-training with the ABEEL framework generalizes beyond a single model class or type.

3: Insufficient empirical evidence:
- It is unclear whether the performance differences of ABEEL-Length Penalty compared to the other models are statistically significant. In Figure 7, it is also unclear whether the models were trained with more than one seed.
- For Table 1: since Qwen-7B-instruct model is trained (ABEEL, and ABEEL-Length Penalty), it would be useful to also compare with zero-shot Qwen-7B-instruct  in addition to zero-shot Qwen-14B-instruct. 
- The advantages of using ABEEL compared to vanilla and belief prompting are not clearly demonstrated in the experiments (Figure 5a and 5b). In particular, ABEEL (seed 2) performs worse than belief prompting and vanilla prompting, raising questions about the strength of the proposed approach compared to the baselines (see Point 1 raised above). 

While this paper addresses an interesting problem of effectively handling long contexts in multi-turn interaction settings with LLMs, it could be significantly improved by addressing the concerns regarding the novelty and significance of the empirical results.

### Questions
Questions about multiple training seeds and using models other than Qwen 7B-Instruct are raised in Weaknesses to strengthen claims about empirical performance of ABEEL. 

Additionally, there are other multi-turn domains that the authors could consider, where vanilla models have been reported to perform poorly. 
- For example, Laban et al., 2025 evaluates models on a suite of tasks such as coding, math, and summarization, where task-relevant information is gradually revealed through multi-turn interactions, so it is crucial to remember and maintain beliefs about key problem details. 
- In Zhao et al., 2025, "Long-context retrieval" section may be of particular interest to the authors. This task focuses on accurately retrieving and summarizing relevant information about user preferences in long-context conversations. 

For both of these benchmark tasks, one could imagine belief states being beneficial and leading to improved performance compared to naïve long-context models.

- Laban et al., 2025. "LLMs get lost in multi-turn conversation". https://arxiv.org/pdf/2505.06120.
- Zhao et al., 2025. "Do LLMs recognize your preferences? Evaluating personalized preference following in LLMs." https://arxiv.org/pdf/2502.09597

### Soundness
2

### Presentation
2

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
ABBEL replaces growing histories with a compact natural-language belief and acts only on that, giving near-constant memory and shorter reasoning. Zero-shot lags full history, but RL, especially belief grading, recovers or beats it and enables a memory–accuracy trade-off. In multi-objective QA it outperforms MEM1 with a shorter internal state; remaining issues are belief-update errors, hallucinated past steps, and repeated actions when beliefs stall.

### Strengths
* Clear, interpretable bottleneck: separating stored “belief” from transient reasoning is simple,
  model-agnostic, and yields near-constant memory across steps while often reducing tokens and
  action-side reasoning.
* Solid empirical sweep and diagnostics: six environments, ablations (vanilla / belief-prompting /
  ABBEL), and candid analysis of failure modes (propagated belief errors, hallucinated past steps).
* RL contributions are practical: outcome-based RL recovers most performance; belief-grading
  reduces regret and allows explicit length–performance trade-offs; competitive/better than MEM1
  in long-horizon QA with a more compact internal state.

### Weaknesses
* Novelty/positioning: very close to prior “learned memory” agents (MEM1/VeRL/rLLM); the
  belief–reasoning split reads as incremental rather than fundamentally new. Missing/under-cited
  contemporaries (e.g., MemAgent) weaken SOTA claims.
* Baselines/fairness: QA compares ABBEL-RL (7B) to an untrained 14B full-history model; no
  apples-to-apples 7B full-history RL baseline reported. Combination-Lock gains hinge on a toy
  setting and ground-truth belief grading; generalization to realistic tasks is unclear.
* Practicality: two calls per step (belief + action) and RL/aux-task training cost; belief-quality
  heuristics for real domains are unspecified; observed loops where the agent repeats uninformative
  actions when beliefs don’t update.

### Questions
* Did you train and evaluate a 7B full-history (vanilla) RL baseline in QA and Combo-Lock? If so,
  please report; if not, this is needed to isolate bottleneck benefits from RL effects.
* How will you grade beliefs without ground-truth posteriors in realistic settings (retrieval QA, SWE)?
  Would a constrained/structured belief schema (e.g., JSON slots) reduce hallucinations and make
  grading feasible?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces ABBEL, a framework that replaces growing multi-turn interaction histories with a compact, natural-language belief state. The authors identify frontier models have inferior performance because of error in belief updating. Thus, finetuning LLMs with belief grading improves LLMs in a multi-objective QA environment and shows promising results.

### Strengths
1. Usage of belief state to compress the history trajectory to assist LLM for effective actions sampling, making the internal state compact and inspectable. 
2. RL with belief grading improves small language model in multi-objective QA tasks. Use a model to parse belief into characters and then comparing with ground truth posterior is a simple yet effective solution.

### Weaknesses
1. RL with belief grading heavily relies on ground truth posterior. It’s unclear how robust grading signals will be in complex, non-synthetic settings where ground-truth posteriors aren’t computable. (as mentioned in the "Limitations" sections)
2. Only one benchmark in the main text. I would expect more benchmarks on results of RL + belief grading, like WebShop [1] as the Mem1 [3] authors did, and maybe also ALFWorld [2], a text based environment for agents to reason and interact with.
3. It seems to me that a lot of context is spent on evaluating frontier models with belief bottlenecks. I do believe that this is an important finding for motivation, but I believe more explanations and results can be put in the appendix, where the main text can focus on RL with belief grading.

[1] Yao, Shunyu, et al. "Webshop: Towards scalable real-world web interaction with grounded language agents." Advances in Neural Information Processing Systems 35 (2022): 20744-20757.

[2] Shridhar, Mohit, et al. "Alfworld: Aligning text and embodied environments for interactive learning." arXiv preprint arXiv:2010.03768 (2020).

[3] Zhou, Zijian, et al. "MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents." arXiv preprint arXiv:2506.15841 (2025).

### Questions
1. Would you providing some examples/ideas of applying heuristic function to model ground truth posterior when computing ground truth posterior is not practical?
2. Is the core different between ABBEL and MEM1[1] only about separating belief state update into multiple steps accordingly?

[1]Zhou, Zijian, et al. "MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents." arXiv preprint arXiv:2506.15841 (2025).

### Soundness
3

### Presentation
2

### Contribution
2
