# LLM CHESS: Benchmarking Reasoning and Instruction-Following in LLMs through Chess

- Avg Score: 4.67
- Decision: Reject
- Scores: 4, 6, 4

## Abstract
We introduce LLM CHESS, an evaluation framework designed to probe the generalization of reasoning and instruction-following abilities in large language models (LLMs) through extended agentic interaction in the domain of chess. We rank over 50 open and closed source models by playing against a random opponent using a range of behavioral metrics, including win and loss rates, move quality, move legality, hallucinated actions, and game duration. For a subset of top reasoning models, we derive an Elo estimate by playing against a chess engine with variably configured skill, which allows for comparisons between models in an easily understandable way. Despite the simplicity of the instruction-following task and the weakness of the opponent, many state-of-the-art models struggle to complete games or achieve consistent wins. Similar to other benchmarks on complex reasoning tasks, our experiments reveal a clear separation between reasoning and non-reasoning models. However, unlike existing static benchmarks, the stochastic and dynamic nature of LLM CHESS uniquely reduces overfitting and memorization while preventing benchmark saturation, proving difficult even for top reasoning models. To support future work on evaluating reasoning and instruction-following in LLMs, we release our experimental framework, a public leaderboard, and a dataset of associated games.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
LLM CHESS introduces an agentic, chess-based benchmark to jointly evaluate reasoning and instruction-following in 50+ LLMs using full-game play, per-ply quality metrics, and engine-grounded Elo, with code and a public leaderboard released for reproducibility. Empirically, most models struggle even versus a random opponent while reasoning-enhanced models fare better.

### Strengths
1. The choice of chess as the testbed is conceptually solid. It naturally embodies combinatorial search, long-horizon planning, and rule-based reasoning, making it a meaningful domain.
2. The analyses and experiments are extensive. The consistent advantage of reasoning-enhanced “thinking” models over standard LLMs provides credible support for the benchmark’s claims.
3. The framework is reproducible and extensible, with open code, public leaderboards, and adjustable opponent strengths, allowing the benchmark to evolve as models improve.

### Weaknesses
1. Most LLMs obtain nearly zero Win/Loss in Table 4, suggesting that the current difficulty curve may be poorly calibrated. It remains unclear whether the benchmark measures reasoning limitations or simply overwhelms models with excessive interaction complexity.
2. Figure 1 conveys little information, with too much large white space and minimal data illustration. Core analyses such as the ablation in experiments should be added into the main passage instead of in the appendix.
3. The agentic interface itself adds heavy cognitive and formatting burdens. Since removing it in ablations leads to more than 20 % performance gains, failures may be caused by API-understanding errors rather than reasoning problems.
4. The paper implicitly equates chess reasoning with general reasoning ability, yet with no cross-task validation (e.g., MATH or BBH scores). The validity of reasoning in this benchmark remains unverified.
5. Writing quality and narrative flow are weak.

### Questions
1. The evaluation focuses exclusively on LLMs; including non-LLM or rule-based baselines would anchor what the result actually represents and clarify how the benchmark scales across architectures.
2. Can this framework extended to other board games?
3. Runtime and API costs are not reported, which is necessary to assess practicality and reproducibility.
4. Because the benchmark uses open interaction protocols, will targeted fine-tuning on its trajectories quickly inflate leaderboard scores, without improving underlying reasoning?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper benchmarks language models as chess-playing agents, finding that reasoning models perform at near the median player on chess.com (looking briefly online at the distribution of elo scores.) Most of the models without reasoning (long think chains) fail to beat an agent playing random moves.

I think there are fair concerns about why/whether we would use an LLM for a task like this, but it seems like this paper offers a resource that should exist and I think that people will enjoy seeing.

### Strengths
* The paper is clear and easy to follow
* Overall feels like a solid work
* I like how you setup Figure 4a (and b)

### Weaknesses
* The primary contribution is the resource here. However, it is unclear what information/inferences use of the resource will offer. What will future users of the benchmark learn from the results?
(See questions)

### Questions
* Q: What lessons would you draw from this work? What lessons would you draw from future results on this benchmark?
* Chess was/is a good test bed for reasoning, but now we have models that are pretty good at reasoning, and we have chess models that are good at chess. It is not clear why or where this benchmark fits in. There is an analogy to arithmetic but I'm not sure it fully answers the question.
* Q: Have you seen this `https://www.kaggle.com/game-arena`? Any comments or comparisons?

* I would recommend using something other than "Random Player". It sounds like you're randomly picking human players off of chess.com or something. Perhaps "Random Agent"

### Soundness
3

### Presentation
3

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
This study proposes LLM CHESS, a novel and comprehensive benchmark framework designed to evaluate the reasoning and instruction-following capabilities of Large Language Models (LLMs) within the complex, strategic domain of chess. Its core methodology utilizes an agentic interaction setup where LLMs play full games via tool calls: `get_current_board`, `get_legal_moves`, and `make_move`. The study presents a large-scale evaluation of over 50 models. It first assesses their baseline capabilities and instruction adherence by playing against a random agent. Subsequently, high-performing models are pitted against a chess engine (Komodo's Dragon 1) with configurable skill levels to estimate their Elo ratings. Key findings reveal a significant performance gap between so-called "reasoning-enhanced" LLMs and standard ones. Most models struggle to reliably defeat even a random player due to instruction-following failures. Even the top-performing model achieves an Elo of only ~758, highlighting a stark discrepancy between LLM capabilities in dynamic, strategic environments and their performance on other reasoning tasks like math and programming. To foster future research, the authors have open-sourced the experimental framework, a public leaderboard, and the dataset of games.

### Strengths
1. The agentic framework is a key innovation. It not only evaluates the quality of moves but also tests the model's integrated ability to follow instructions and use tools. The design is highly scalable—difficulty can be increased by simply raising the opponent's skill level—ensuring the benchmark's long-term relevance.
2. The breadth of the study, covering over 50 models, is commendable. This large scale provides a robust foundation for drawing conclusions about the current state of LLMs on this task, making the findings more convincing.
3. The well-designed ablation studies offer deep insights into why models fail, demonstrating that model performance is highly sensitive to prompting and interaction formats. This points to a lack of robust generalization capabilities.

### Weaknesses
1. A core part of the analysis distinguishes between "reasoning-enhanced" and "standard" models. The argument would be strengthened if the paper provided a more explicit and operational definition for this classification (e.g., based on specific test-time algorithms, architectural features, or training methods).
2. The analysis could be enriched by more granular case studies of errors. For instance, analyzing the types of mistakes (distinguishing between simple tactical blunders and deeper strategic misunderstandings) or identifying common patterns in instruction-following failures could offer a more nuanced understanding of the models' cognitive limitations.
3. The default agentic setup concurrently tests strategic thinking, tool use, and format adherence. While the ablation studies help to deconstruct this, it raises the question of whether the benchmark could be designed to isolate these skills more directly. For instance, a model might be a strong strategist but a poor tool-user, and the current primary metrics may not clearly differentiate between these two cases.

### Questions
1. Could you provide a more formal or operational definition for "reasoning-enhanced" models as used in this study? Is this distinction based on specific test-time algorithms (e.g., search, self-consistency), architectural features, or particular training methodologies?
2. The high rate of instruction-following failures is a key finding. It would be very informative to see a more fine-grained classification of these errors. For example, do models get stuck in loops (e.g., repeatedly calling get_current_board), fail to parse the available actions, or generate syntactically invalid moves in the make_move tool call? This could help differentiate between failures in attention, comprehension, or action generation.
3. The Mixture-of-Agents (MoA) experiments yield interesting but limited gains, and performance appears highly sensitive to the choice of proposer and aggregator models. Could you please elaborate on the model's sensitivity to this configuration? For instance, what might the results be if a stronger model were used as the aggregator? Furthermore, what specific prompt or mechanism was used by the aggregator to synthesize the proposals?
4. Regarding line 404, does the phrase "by removing actions and instead supplying the removed information automatically" mean that the current board state and legal moves are provided to the LLM automatically, bypassing the need for it to call the corresponding tools?
5. On pages 20 and 21, some of the prompt text extends beyond the page margins. We suggest adjusting the formatting to improve readability.

### Soundness
3

### Presentation
3

### Contribution
3
