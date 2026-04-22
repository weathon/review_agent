# Elo-Evolve: A Co-evolutionary Framework for Language Model Alignment

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 4, 6

## Abstract
Current alignment methods for Large Language Models (LLMs) rely on compressing vast amounts of human preference data into static, absolute reward functions, leading to data scarcity, noise sensitivity, and training instability. We introduce Elo-Evolve a co-evolutionary framework that redefines alignment as dynamic multi-agent competition within an adaptive opponent pool. Our approach makes two key innovations: (1) eliminating Bradley-Terry model dependencies by learning directly from binary win/loss outcomes in pairwise competitions, and (2) implementing Elo-orchestrated opponent selection that provides automatic curriculum learning through temperature-controlled sampling. We ground our approach in PAC learning theory, demonstrating that pairwise comparison achieves superior sample complexity (O(1/ε) vs O(1/ε2)) and empirically validate a 4.5$\times$ noise reduction compared to absolute scoring approaches. Experimentally, we train a Qwen2.5-7B model using our framework with opponents including Qwen2.5-14B, Qwen2.5-32B, and Qwen3-8B models. Results demonstrate a clear performance hierarchy: point-based methods $<$ static pairwise training $<$ Elo-Evolve across Alpaca Eval 2.0 and MT-Bench, validating the progressive benefits of pairwise comparison and dynamic opponent selection for LLM alignment.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This work introduces a novel method of preference optimization by using a dynamic elo as the reward for RLHF. The idea is clear and the method is clearly presented. Experiments have shown significant improvement.

### Strengths
The method is simple while surprisingly effective in practice with Qwen base models. The ablation study is sufficient.

### Weaknesses
- The first one of the and main issues is that the contribution is claimed to be "eliminating Bradley-Terry model dependencies". But this isn't new at all. Bunch of previous works have addressed this or propose this as contribution such as [1,2,3,4]. Especially INPO [4] is exactly directly learning from binary win/loss outcomes. These are missing literatures as well, the author should include these related works, compare and clarify the contribution of this work. 

- Following the above issue, although it is interesting to see a significant improvement in AlpacaEval, it's unclear how to compare with INPO [4].

- Another main issue is that experiments are only conducted with single type of base model Qwen. It's necessary to test the methods on other base models to demonstrate the effectiveness.

Writings:
- I personally think using symbols like "<" in writing especially abstract is informal.
- Some notational issues around Eq. (3), such as unclear what is R in the definition of E. Should that be R_t?

In general, there are notable limitations of current manuscript.



[1] Munos, Rémi, et al. "Nash learning from human feedback." Forty-first International Conference on Machine Learning. 2024.
[2] Wang, Mingzhi, et al. "Magnetic preference optimization: Achieving last-iterate convergence for language model alignment." arXiv preprint arXiv:2410.16714 (2024).
[3] Tang, Xiaohang, et al. "Game-Theoretic Regularized Self-Play Alignment of Large Language Models." arXiv preprint arXiv:2503.00030 (2025).
[4] Zhang, Yuheng, et al. "Iterative nash policy optimization: Aligning llms with general preferences via no-regret learning." arXiv preprint arXiv:2407.00617 (2024).

### Questions
- This method compared to self-play-based methods is significantly more expensive since typically a larger model is required to act as opponent for Elo-Evolve. Given that, is it more worthy investing computing in scaling base model (i.e. using a larger base model) and conduct self-play or even normal RLHF (e.g. Point GRPO)?

### Soundness
2

### Presentation
2

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
Existing alignments methods typically distill human preference data into reward models. The paper proposes an alternative using a dynamic multi-agent framework to bypass traditional bradley-terry approaches. They show strong performance as a Qwen-2.5-7B model trained with this approach is able to improve on AlpacaEval 2.0 and MT Bench.

### Strengths
1.	The adaptive curriculum learning approach used in Elo-Evolve is interesting where a different reference opponent model is used at each stage of the training allowing the model to progressive improve against stronger opponents.

### Weaknesses
1.	The claim in Lines 40-45 seems un/under-substantiated. Claim 1 is not supported by any literature and there has been evidence such as HelpSteer2-preference [1] that shows only 10 thousand samples is enough for training high quality reward models. For claim 2, it’s not clear what sub-optimal sample complexity is and claim 3 is supported by 1 paper from 2020, even though the post training field has evolved substantially since then.
2.	The results in Table 1 don’t seem to be very strong. For instance, the Elo-Evolve performs at 38.03 on AlpacaEval 2 LC while the Point GRPO is at 37.41, which is presumably within one SD. On MT Bench, the performance of Elo-Evolve is at 8.04 while DNO is at 7.97, which is also unlikely to be substantially different. The “vs. Qwen xxx” baselines should be interpreted as ablations but in this case, some ablations outperform the Elo-Evolve algorithm which suggest that maybe a static opponent might be good enough.
3.	Metrics are slightly outdated – with AlpacaEval 2 and MT Bench both from late 2023, with known flaws such as length bias. AlpacaEval 2 Length Control is the most recent (from early 2024) and should be used instead of AlpacaEval 2 WR (rather than in addition to it). Furthermore, I think more up-to-date metrics such as Arena Hard [5] or WildBench [6] should be used (both from late 2024) since they reflect capabilities of recent models better (e.g. Qwen 2.5 was released in late 2024).
4.	I think a missing baseline is to use the same LLM-Judge (Qwen3-14B-Instruct) and just use the generated responses against one another in GRPO. This can help us to understand whether the improved performance is due to using a better judge (compared to the WorldPM point-RM) or because of the reference responses from the (adaptive) opponents. Using the LLM-Judge by itself has been done many times e.g. [3] and is likely to be useful without needing for external model responses.

[1] HelpSteer2-Preference: Complementing Ratings with Preferences https://arxiv.org/abs/2410.01257

[2] AlpacaFarm: A Simulation Framework for Methods that Learn from Human Feedback https://arxiv.org/abs/2305.14387

[3] Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena https://arxiv.org/abs/2306.05685 

[4] Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators https://arxiv.org/abs/2404.04475

[5] From Crowdsourced Data to High-Quality Benchmarks: Arena-Hard and BenchBuilder Pipeline https://arxiv.org/abs/2406.11939

[6] WildBench: Benchmarking LLMs with Challenging Tasks from Real Users in the Wild https://arxiv.org/abs/2406.04770

### Questions
1.	Why is there a need to report performance at steps 100/300/500? My understanding is that only the optimal step across a run should be reported since different methods might have different optimal training steps. Table 1 and 2 currently looks confusing with too many values and bolded values.

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
4

### Summary
This paper proposes a game-theoretic alignment algorithm for large language models that leverages a pool of opponents. In Elo-Evolve, the proposed method, an LLM is trained by playing against opponents matched based on their ELO scores. ELO-based matching provides a natural learning curriculum, enabling the resulting LLM to achieve competitive performance across various benchmarks.

### Strengths
- The presented idea of using ELO rating for opponent matching is reasonable and presented clearly.
- Performance gain seems consistent.
- Clever length bias mitigation is used.

### Weaknesses
1. The paper lacks comparison and discussion regarding self-play alignment methods. In recent years, significant attention has been devoted to alignment algorithms based on self-play.
    - Such methods do not rely on the Bradley–Terry model and often leverage game-theoretic ideas. Ideally, the current manuscript could be much stronger by providing a discussion of self-play methods (e.g., how Elo-Evolve could outperform self-play methods) and including empirical comparisons.
2. Compared to self-play methods, Elo-Evolve requires additional pre-trained opponent models. While a self-play method typically requires a pre-trained generalized preference model, Elo-Evolve additionally depends on pre-trained opponents.
3. The paper allocates a non-trivial amount of space to the discussion of the benefits of relative reward signals (e.g., Sections 3 and 5.2). Although these are interesting results, the claims do not specifically support Elo-Evolve, but rather broadly support all methods that use a generalized preference model. The limitations of the Bradley–Terry model have already been discussed several times in the self-play literature (although I do not think the exact argument has been presented before).

For self-play alignment methods, see, for example,

[1] Wu, Yue, et al. "Self-play preference optimization for language model alignment." arXiv preprint arXiv:2405.00675 (2024).

[2] Tang, Xiaohang, et al. "Game-Theoretic Regularized Self-Play Alignment of Large Language Models." arXiv preprint arXiv:2503.00030 (2025).

[3] Munos, Rémi, et al. "Nash learning from human feedback." Forty-first International Conference on Machine Learning. 2024.

### Questions
1. Can Elo-Evolve be used to train a model that is significantly better than the provided opponents? How can we push the state-of-the-art of language models using Elo-Evolve beyond the strongest provided opponent?
2. What would be the advantages of Elo-Evolve over self-play-based alignment methods, such as SPPO?

### Soundness
3

### Presentation
3

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
This paper proposes Elo-Evolve, a co-evolutionary framework that aligns LLMs through dynamic multi-agent competition. Instead of static reward models, the policy learns from binary win/loss signals in pairwise matches. An Elo-based opponent selection introduces automatic curriculum learning: the model faces similar-strength opponents early and stronger ones later. Experiments on UltraFeedback with Qwen models show consistent gains on AlpacaEval 2.0 and MT-Bench over point-based and static pairwise baselines.

### Strengths
1. The dynamic opponent selection is interesting. The temperature parameter offers a clean way to balance focus and diversity—small T for close-strength opponents, large T for variety. This forms an automatic curriculum where training difficulty grows with model ability.

2. Each prompt selects its own opponent, leading to smoother and more stable training.

3. Replacing scalar rewards with binary win/loss is well-motivated; both the PAC-theoretic analysis and experiments support its efficiency and robustness.

### Weaknesses
1. The framework introduces several components, which makes the system design a little complex. It would be helpful to include a simple baseline, where the model is trained sequentially against Qwen2.5-14B, Qwen2.5-32B, and Qwen3-8B as progressively stronger opponents. The prompts can be divided into three groups either randomly or based on their difficulty, for example using a reward model to estimate complexity. Such a baseline would help clarify how much the dynamic Elo scheduling improves over a manually designed curriculum.
2. Because Elo ratings are continuously updated, opponent strength may fluctuate during training. When a main opponent weakens, as observed at Step 500 on MT-Bench, the policy appears to over-adapt to easier adversaries. This may affect measured progress and limit further improvement.

### Questions
1. The use of Qwen3-14B-Instruct as the judging model instead of a specialized reward model is not fully discussed. A short explanation of this choice would improve the clarity.

### Soundness
3

### Presentation
3

### Contribution
3
