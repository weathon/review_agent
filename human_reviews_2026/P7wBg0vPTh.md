# RLVER: Reinforcement Learning with Verifiable Emotion Rewards for Empathetic Agents

- Decision: Accept (Poster)
- Scores: 6, 4, 6

## Abstract
Large language models (LLMs) excel at logical and algorithmic reasoning, yet their emotional intelligence (EQ) still lags far behind their cognitive prowess.  While reinforcement learning from verifiable rewards (RLVR) has advanced in other domains, its application to dialogue—especially for emotional intelligence—remains underexplored. In this work, we introduce RLVER, the first end-to-end reinforcement learning framework that leverages verifiable emotion rewards from simulated users to cultivate higher-order empathetic abilities in LLMs. Within this framework, self-consistent affective simulated users engage in dialogue rollouts and produce deterministic emotion scores during conversations, serving as reward signals to guide the LLM's learning. Fine-tuning publicly available Qwen2.5-7B-Instruct model with PPO boosts its Sentient-Benchmark score from 13.3 to 79.2 while largely preserving mathematical and coding competence.  Extensive experiments reveal that: (i) RLVER consistently improves multiple dialogue capabilities; (ii) Thinking and non-thinking models show distinct trends—thinking models excel in empathy and insight, while non-thinking models favor action; (iii) GRPO often yields stable gains, while PPO can push certain capabilities to a higher ceiling; (iv) More challenging environments are not always better—moderate ones can yield stronger outcomes. Our results show that RLVER is a practical route toward emotionally intelligent and broadly capable language agents.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents RLVER, an application of on-policy RL (PPO/GRPO) to improve empathetic, supportive dialogue in LLMs. A “sentient” user simulator (SAGE) provides a deterministic terminal emotion score as the reward. The method further enforces a think-then-say output format. On the Sentient Benchmark, RLVER shows large gains over the Qwen2.5-7B-Instruct base while roughly maintaining general abilities.

### Strengths
1. Clear systemization: A reproducible pipeline that connects a verifiable scalar reward to standard RL for a human-centric objective.

2. Consistent empirical gains: Strong improvements with informative ablations (PPO vs. GRPO; with/without explicit thinking).

3. Useful insights: Explicit thinking tends to help; overly strict simulators hurt learnability; GRPO is stabler, while PPO can reach higher peak scores.

4. Transparent reporting: Prompts, rollout settings, and evaluation protocol are documented clearly.

### Weaknesses
1. Limited methodological novelty: No new RL objective/optimizer; primarily an engineering/system contribution.

2. Heavy reliance on SAGE: Data generation, reward, and primary evaluation all hinge on the same framework, raising self-referential/reward-hacking concerns and limiting external validity.

3. Sparse human evaluation: Empathy is inherently human-centric; limited human blind ratings undermine real-user impact claims.

4. Single base model: Results are shown only on Qwen2.5-7B; cross-architecture/size/language generality is unclear.

5. Safety & multi-objective trade-offs under-specified: Optimizing “feeling better” risks sycophancy or avoidance when boundary-setting or factual correction is needed.

6. Cost/scalability underreported: Multi-turn on-policy RL is expensive; a fuller accounting of token/sample costs and deployment implications would help.

### Questions
1. Broader benchmarks: Have you evaluated or planned to evaluate on established empathy/affect datasets such as EmpatheticDialogues [1], EmoryNLP [2], ESConv [3], EmoBench [4], and ToMBench [5] to verify external validity beyond SAGE?

2. Model generalization: Given that experiments train only on Qwen2.5-7B-Instruct, do you plan to extend to additional base models to test portability and sensitivity to architecture/scale?

3. Human evaluation: Will you include human blind ratings and report inter-rater agreement and correlation with SAGE scores?

[1] Rashkin, Hannah, Eric Michael Smith, Margaret Li, and Y-Lan Boureau. 2019. “Towards Empathetic Open-domain Conversation Models: A New Benchmark and Dataset.” In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, 5370–5381. Florence: Association for Computational Linguistics. h

[2] Zahiri, Sayyed M., and Jinho D. Choi. 2018. “Emotion Detection on TV Show Transcripts with Sequence-based Convolutional Neural Networks.” In Proceedings of the AAAI Workshop on Affective Content Analysis (AFFCON’18), 44–51. 

[3] Liu, Siyang, Chujie Zheng, Orianna Demasi, Sahand Sabour, Yu Li, Zhou Yu, Yong Jiang, and Minlie Huang. 2021. “Towards Emotional Support Dialog Systems.” In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), 3469–3483. 

[4] Sabour, Sahand, Siyang Liu, Zheyuan Zhang, June Liu, Jinfeng Zhou, Alvionna Sunaryo, Tatia Lee, Rada Mihalcea, and Minlie Huang. 2024. “EmoBench: Evaluating the Emotional Intelligence of Large Language Models.” In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 5986–6004. 

[5] Chen, Zhuang, Jincenzi Wu, Jinfeng Zhou, Bosi Wen, Guanqun Bi, Gongyao Jiang, Yaru Cao, et al. 2024. “ToMBench: Benchmarking Theory of Mind in Large Language Models.” In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 15959–15983.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes RLVER, a reinforcement learning framework designed to enhance the emotional intelligence (EQ) of large language models (LLMs) by optimizing them with verifiable emotion rewards. The framework leverages the SAGE (Sentient Agent as a Judge) simulator — an LLM-driven environment that models user affect and generates deterministic emotion scores — to provide reward signals for PPO and GRPO training. The authors report substantial improvement on the Sentient Benchmark (13.3 → 79.2) while maintaining performance on math, coding, and instruction-following tasks.
The idea of integrating psychologically-grounded verifiable feedback into RL for emotion modeling is promising. However, the current paper lacks methodological clarity, validation rigor, and experimental diversity to support its central claims.

### Strengths
- The paper explores a novel and meaningful problem — improving emotional intelligence in LLMs via reinforcement learning.

- The notion of verifiable emotion rewards represents a new attempt to make affective reinforcement signals interpretable and consistent.

- The reported improvements on the Sentient Benchmark are substantial and suggest potential in empathy-oriented RL training.

### Weaknesses
- The method section, particularly Figure 1, is hard to interpret. The mapping between variables and their roles in the RL process is unclear. No pseudocode or formal algorithmic steps are provided, making it difficult to reproduce or verify the approach.
- No validation is presented comparing SAGE-generated emotion scores with human judgments or psychological gold standards. This undermines the “verifiable” aspect of the proposed reward, as it lacks empirical reliability testing. 
- Experiments are conducted only on Qwen2.5-7B-Instruct, a single Chinese-centric open model. The paper lacks evidence that RLVER generalizes to different architectures or scales (e.g., LLaMA, Mistral, Gemma). This severely limits the generalizability and reproducibility of the method.
- No comparison is made with other empathy or affective dialogue datasets, such as EmpatheticDialogues, ESConv, or MELD. As a result, it is unclear whether RLVER’s improvements translate beyond its own simulation environment.


- All results are derived from simulated environments, with no human-in-the-loop testing. Without qualitative or quantitative human evaluation, the claim that RLVER produces more empathetic and emotionally aligned behavior remains unverified.

### Questions
- Provide detailed definitions for all mathematical symbols.
-  Add an algorithm box or pseudocode that explicitly connects the emotion score computation with PPO/GRPO updates.
- Include human evaluation comparing SAGE emotion scores with real human annotations. Report quantitative metrics (e.g., correlation, agreement rates) to establish reward fidelity.
- Explore different reward designs (per-turn, cumulative, delta-based)

### Soundness
2

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
4

### Summary
The paper introduces RLVER (Reinforcement Learning with Verifiable Emotional Rewards) - a reinforcement learning framework designed to improve the emotional intelligence of large language models (LLMs) through verifiable reward signals generated by a simulated user named SAGE.
Using PPO and GRPO on top of Qwen2.5-7B-Instruct, the authors train the model to optimize for “emotional scores” in dialogue, measured via a new benchmark called Sentient Benchmark. The best configuration (PPO + “thinking” mode) achieves a jump from 13.3 to 79.2, approaching the level of frontier proprietary models while maintaining stable mathematical and coding capabilities.
The study further examines the effects of simulated user complexity, the impact of chain-of-thought (“thinking”) prompts, and includes a multi-competency evaluation using “LLM-as-a-Judge”.

### Strengths
- Originality: Clear conceptual innovation — reinforcement learning guided by verifiable emotional scores from a controllable simulator (SAGE).
- Quality: Demonstrates substantial improvement (Sentient 79.2 vs 13.3 baseline) while preserving general abilities in math and code.
- Clarity: Writing is structured; diagrams effectively communicate architecture and reward flow.
- Significance: Addresses an underexplored but crucial domain — emotional intelligence in LLMs - with a scalable, privacy-safe RL training loop.
- Analytical depth: Includes ablations across RL methods (PPO vs GRPO), analysis of “thinking” format, and breakdown by emotional competencies.

### Weaknesses
- Inconsistent reward definition. Emotional scores are defined in [0, 100], yet Table 1 states success as “score > 100” - logically impossible. Likely an error needing correction.
- Unclear determinism of SAGE simulator. The simulator is described as deterministic but implemented via a generative model (DeepSeek-V3). The temperature and decoding strategy for reproducible rewards are unspecified.
Overreliance on one simulator. All benchmarks (Sentient, chit-chat) derive from SAGE, raising concerns of overfitting and limited external validity without human evaluation.
- LLM-as-a-Judge limitations. Despite reported inter-rater consistency, subjective competence scoring by other LLMs risks circularity and bias. Human validation is missing.
- Ambiguity in setup details. Dialogue length differs between text (10 turns) and appendix (8 turns). Hyperparameters and compute cost are under-reported.
- Lack of robustness tests. No ablations with alternate simulators or format-reward removal, which would confirm generality.

### Questions
1. Please clarify the reward success threshold (Table 1) and ensure consistency with the [0–100] range.
2. Specify the exact number of dialogue turns used for training and evaluation (8 or 10?).
3. How is determinism achieved in SAGE reward generation (temperature? seed?)? Provide variance statistics if re-evaluated multiple times.
4. Did you perform any human evaluation (blind pairwise or Likert rating) to validate emotional quality outside SAGE?
5. Can you report robustness to simulator replacement (e.g., using a different model for rewards)?

### Soundness
2

### Presentation
3

### Contribution
2
