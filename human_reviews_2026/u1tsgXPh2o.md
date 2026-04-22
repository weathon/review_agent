# Play to Generalize: Learning to Reason Through Game Play

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 8

## Abstract
Developing reasoning capabilities in multimodal large language models (MLLMs) remains challenging. Motivated by literature suggesting that gameplay promotes transferable reasoning skills, we propose a novel post-training method, Visual Game Learning (ViGaL), where MLLMs develop generalizable reasoning skills through playing arcade-like games. Specifically, we show that training a 7B-parameter MLLM via reinforcement learning (RL) on simple games like Snake significantly enhances the downstream performance on multimodal math benchmarks like MathVista, and on multi-discipline questions like MMMU, without seeing any worked solutions, equations, or diagrams during RL.
Remarkably, our model outperforms specialist models post-trained on benchmark-oriented multimodal reasoning data, while preserving the model’s performance on general visual benchmarks, a challenge where specialist models often fall short.
Our findings suggest that multimodal reasoning can emerge from gameplay, pointing to a promising strategy of designing surrogate tasks for RL post-training. The code is available at https://yunfeixie233.github.io/ViGaL.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes Visual Game Learning (ViGaL), a post-training paradigm where an MLLM (7B) is fine-tuned via reinforcement learning on simple arcade-style games (e.g., Snake and a Rotation task) to acquire transferable reasoning skills. Without using any math/benchmark-specific data during RL, ViGaL improves downstream multimodal reasoning (MathVista and MMMU) while preserving general visual abilities, outperforming specialist models trained on curated reasoning data.

### Strengths
1. Overall, this is a clearly written and well-structured paper with polished figures that make the pipeline easy to follow from setup through results. 

2. The core idea—leveraging controllable visual games with an RL loop as a scalable surrogate to induce transferable reasoning—is novel and timely, pushing beyond curated math datasets toward a more generalizable training signal. 

3. Empirically, the paper shows that gameplay-driven RL improves downstream multimodal reasoning without eroding general vision capabilities, and it consistently outperforms SFT when trained on the same data. The evidence is strengthened by broad evaluations and careful ablations (modalities, difficulty control, data scaling), suggesting robustness rather than benchmark luck. 

4. The approach is also practical: simple game environments, rule-based rewards, and a straightforward RL recipe lower the barrier to adoption and invite community exploration.

### Weaknesses
1. The paper proposes a promising direction, but the evidence base feels narrow: only two games (Snake and Rotation) are used, which makes it hard to argue for broad “play-to-generalize.” The paper explains that these two games let the authors probe “reasoning and perception” (Snake for spatial/planning, Rotation for angle perception) , yet this rationale is thin and the selection criteria appear ad hoc.

2. The idea of the paper would read as substantively stronger with diversity and robustness checks across more games.

3. The paper proposes that different games improve different math sub-skills, but the current analysis, while suggestive, is correlational and limited to two sources.

4. The paper proposes training via two games (Snake, Rotation) and shows encouraging transfer, but it does not yet establish games as a better alternative to standard post-training (e.g., curated math/logic data or instruction tuning) under controlled, budget-matched conditions.

### Questions
1. Can you articulate a principled game-selection taxonomy (e.g., planning, geometric transforms, partial observability, stochasticity) and explain why Snake and Rotation instantiate distinct buckets?

2. What ex-ante criteria (beyond convenience) determined these two games, and which candidate games were considered but rejected? If any rejected candidates, were they rejected before or after experiments, and why? 

3. Do you observe negative transfer for any added game types?

### Soundness
3

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
4

### Summary
This paper proposes ViGaL (Visual Game Learning), a post-training method that applies reinforcement learning (RL) on simple visual games—primarily Snake—to improve the multimodal reasoning performance of a 7B MLLM on downstream benchmarks such as MathVista and MMMU. The authors show that models trained on these games achieve modest gains on reasoning tasks, even though no domain-specific reasoning data is used during training. The core claim is that gameplay can serve as a surrogate task to develop transferable reasoning skills.

While the empirical results demonstrate some performance improvement, the work largely repackages well-established ideas—such as multi-task learning, curriculum learning, and pretext tasks—under a new but superficially motivated paradigm. Crucially, it fails to provide any principled insight into why or when such transfer occurs, nor does it offer a methodology for designing effective auxiliary tasks for a given downstream goal. As such, the contribution appears incremental rather than foundational.

### Strengths
1) Clean Experimental Setup and Broad Benchmarking: The paper evaluates the proposed method on a wide range of established multimodal reasoning benchmarks (e.g., MathVista, MMMU, CLEVR+), which lends credibility to the reported performance gains. The experimental design is generally sound, and the use of rule-based RL avoids the complexity of reward modeling.

2) Demonstration of Cross-Domain Performance Gain: It is empirically shown that training on a simple game can lead to measurable improvements on unrelated reasoning tasks. This serves as a proof-of-concept that some form of transfer is possible, even if the mechanism remains opaque.

### Weaknesses
1) Lack of Novelty: Repackaging Known Ideas Without Substantial Advance

The idea that training on one task can improve performance on another—i.e., multi-task learning (MTL) or transfer learning—is decades old. The use of pretext tasks in self-supervised learning has been standard in vision and NLP. Even within RL, curriculum learning and autocurricula have long demonstrated that simple environments can give rise to complex behaviors. The paper does not convincingly argue why using Snake as a surrogate task is fundamentally different or more effective than prior approaches. It presents an anecdotal case rather than a general principle.

2) No Theoretical or Mechanistic Insight

What specific skills are learned in Snake that transfer to math reasoning?

Are there measurable properties of an auxiliary task (e.g., action space complexity, reward sparsity) that predict transfer success?

The most pressing practical question—how should one design or discover an effective auxiliary task for a given downstream goal?—is completely unaddressed. The choice of Snake appears arbitrary. Why not Tetris? Why not a maze? The paper provides no criteria, heuristics, or framework for answering this.

### Questions
1) Has the phenomenon of using simple, structured tasks to improve performance on complex ones been studied before the LLM era? If so, shouldn't this paper engage more deeply with that literature to justify its claimed novelty?

2) Please see the weakness.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes using gameplay as a surrogate reasoning task to improve multimodal reasoning abilities of MLLM.
The model is post-trained with reinforcement learning on two simple games — Snake (2D grid planning) and Rotation (3D object rotation prediction). It uses a lightweight reward (accuracy + format) and no KL regularization. After training, the model shows improved performance on downstream reasoning benchmarks (math, geometry, and multimodal QA), while maintaining its visual perception capabilities.

### Strengths
- The idea of using structured games to indirectly train reasoning is novel and very interesting to me.
- The performance of using games improves  5–8% on math and spatial tasks, showing gameplay can transfer reasoning skills.
- The authors provided careful analysis on reward, prompts, and difficulty to show the effectiveness of each component.

### Weaknesses
- It is not very clear how to design a game for different types of reasoning abilities. That being said, although games are useful for post training, it requires design for each task.
- The authors are currently only designing two kinds of games with spatial/math-like games used, it is unclear whether other reasoning abilities can also be solved by games.
- The paper provides limited analysis of why the game can improve the math reasoning ability. Specifically, what kind of questions in the benchmark are these post-trained models most benefited from.

### Questions
Overall, this is a great paper. I would like to see the authors answers to my weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes Visual Game Learning (ViGaL)  that uses RL post-training on simple visual games (e.g., Snake and a Rotation task) to elicit reasoning capability in multimodal LLMs in other domains such as math. Instead of training on math or benchmark-style reasoning data, a MLLM is fine-tuned to play these games with rule-based rewards; the resulting model then transfers the acquired skills to out-of-domain tasks while retaining general visual abilities. The paper argues that gameplay can serve as a scalable surrogate task for RL post-training to unlock broadly useful multimodal reasoning.

### Strengths
- The paper shows that RL post-training purely on simple visual games (Snake, Rotation) yields measurable gains on other seemingly unrelated domains such as math despite no direct supervision from those tasks. This is a surprising finding and is worth spreading.
- The gameplay setup enables verifiable rule-based rewards that are friendly to reasoning training, avoiding the need for expensive reward models or human labels. The fact that it can generalize to other domains shows that it has high potential.
- Another benefit of using game play for reasoning training is that there are a large amount of games out there with highly diverse contents. They cover all kinds of aspects of human skills, and can be a rich supplement to reasoning training data.
- The fine-grained analysis in section 3.1 is interesting. For example it connects specific games to specific math subfields, e.g. the game Rotation aligns with angle/length questions.

### Weaknesses
* The games are relatively simple, which is understandable though because it is the first effort to explore this direction.

### Questions
* The prompts include thinking instructions synthesized by GPT-4o. Are they necessary? How much do they affect the results?

### Soundness
3

### Presentation
3

### Contribution
3
