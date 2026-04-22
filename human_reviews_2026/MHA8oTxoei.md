# Investigating the Link Between Representational Similarity and Model Interactions

- Avg Score: 5.50
- Decision: Reject
- Scores: 6, 4, 6, 6

## Abstract
Researchers have shown that neural similarity among humans predicts social closeness and cooperative success, whereas innovation often emerges from interactions among dissimilar individuals. We investigate whether these principles extend to artificial intelligence by examining interactions between large language models. In our experiments, 276 model pairs interact across eight games spanning both cooperation and novelty. We find that pairs with more similar representation spaces achieve significantly higher cooperation but exhibit reduced novelty and creativity. The effects of representational similarity on cooperation and novelty remain robust even after isolating other factors such as performance disparity and model size. We also find that similarity in the early layers consistently exhibits the strongest effect across games, compared to the middle and later layers. This suggests that a central factor underlying the observed trend is the extent to which the two models share lexical and semantic grounding. These findings suggest that representational similarity can be an important consideration in multi-agent system design.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates the relationship between the internal representational similarity of LLMs and their behaviors when interacting in multi-agent scenarios. The authors conduct a large-scale empirical study involving 276 pairs of 23 distinct open-weight LLMs. These pairs engage in eight tasks, systematically divided into two categories: four games and four generative tasks designed to measure novelty.

### Strengths
- The paper's core idea, bridging human neuroscience and multi-agent LLM systems, provides a lens (internal representations) through which to understand and predict complex emergent behaviors, moving beyond purely output-level analysis.
- The scale of the study (276 pairs, 23 models) is substantial. Also, the choice of CKA is well designed.

### Weaknesses
My main concerns are about the game designs:

- As per my initial thoughts, for the "word guessing game," its significant result (+66.2%) seems intuitively predictable. This game is essentially testing the "predictive alignment" of the two models. If two models have similar representations (especially if trained on similar corpora), their probability distributions for a "target word" given a start alphabet during decoding will naturally be more similar. Therefore, their ability to "guess" each other's word seems self-evident (This feels almost tautological). This is more a direct reflection of "homogeneity" than "cooperation."

- Another question is whether the agents' (LLMs') moves within a single round are generated simultaneously or sequentially. This ambiguity leads to a dilemma that challenges the validity of the "collaboration" findings:
 1. If moves are sequential, i.e., one agent sees the other's prediction/move before generating its own: This would invalidate the tasks as true game-theoretic tests. It would introduce a profound first-mover advantage. For instance, in the "Divide-a-Dollar" game, the first agent could simply claim $0.99, leaving almost nothing for the second agent, thus preventing any meaningful test of collaborative strategy.
 2. If moves are simultaneous, i.e., each LLM's generation is independent within the round (my guess this is what the experiment is doing.): This raises the fundamental question of how "collaboration" is being measured. If the agents' decisions are made in isolation without knowledge of their partner's current move, the outcome is simply the result of two independent generations. It is then unclear how this setup evaluates an interactive collaborative process versus just the pre-existing alignment of two models making independent decisions.

The authors may need to provide more explanation of this intra-round mechanism, as the current ambiguity makes it a bit difficult to interpret what is truly being evaluated as "collaboration" in these games.


Another weakness is about insufficiently contextualized Related Work

In the "Related Work" section, a significant and overlooked area is the existing body of work on the designing of multi-agent collaboration. The paper does not review prior methods or established practices for designing and optimizing multi-agent systems for cooperative success. This omission makes it difficult to assess the practical utility of the authors' proposal (using representational similarity) against current or alternative approaches in the field of multi-agent AI.

Instead, the review dedicates substantial space to the analogy between human neural alignment and LLM representational alignment. While this serves as the primary motivation for the hypothesis, this link is arguably a speculative claim, given the fundamental differences between biological brains and artificial networks. The paper would be stronger if it grounded its claims more firmly within the established literature of multi-agent systems optimization, rather than relying so heavily on this novel neuro-AI analogy.

### Questions
- Please explain regarding the game design, as stated in weakness section.
- In section 5.2, your finding that increased representation similarity reduces uniqueness but has "no systematic trend" on quality is fascinating. This implies that using dissimilar models for brainstorming could be a "free lunch" (gaining novelty at no quality cost). Do you believe this is the case?

### Soundness
2

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
The paper investigates how representational similarity between LLMs affects their interactions in multi-agent settings. Across 276 model pairs from 23 open-weight families, the authors measure representational similarity using CKA and test its relationship with collaborative behaviors in various tasks. Results show that higher representational similarity leads to stronger cooperation but reduced novelty and creativity.

### Strengths
- The paper studies an insightful problem: how representational similarity between LLMs predicts their collaborative behavior. This introduces an interesting new dimension for understanding and designing cooperative multi-agent systems.

- The experimental scope is comprehensive, involving 276 model pairs from 23 open-weight LLM families and covering diverse tasks across cooperative and creative settings. The breadth of analysis allows for good statistical validity and generalizability.

- The work performs multi-perspective analyses, examining both performance-based cooperation and creativity-based novelty metrics, which provides a balanced view of how model similarity influences collaboration.

### Weaknesses
- Some experimental details are insufficient. The paper does not describe prompts for each task, provide qualitative examples of model outputs, or clarify how multi-turn collaboration unfolds. For instance, it is unclear whether both models generate responses at every round, how messages are shared, when interactions terminate, and how the final cooperative solution is determined.

- I think the individual model’s capability could be a confounding factor. A model’s solo performance might dominate the pair’s overall outcome, and weaker models could create a bottleneck effect regardless of representational similarity. Have you observed such a phenomenon? If not, what might explain it, given that the dominance of one model seems intuitive? This raises the question of whether the observed correlation truly reflects representational alignment or simply performance disparities.

- Based on the above point, I think a problem is that the paper reports correlations between representational similarity and cooperative performance, but it never establishes causality. It remains unclear whether similar representations cause better cooperation, or whether both emerge as a byproduct of other factors (e.g., model solo performance, model size, or architectural homogeneity). Without controlled ablations, the causal interpretation is weak.

- Although the experiments span diverse tasks, all are relatively synthetic (simple cooperative games or creative generation). These setups may not generalize to realistic multi-agent scenarios involving planning, negotiation, or dynamic tool use.

- Figure 9 is ambiguous and requires better explanation. It is unclear whether it represents a model’s averaged cooperative performance across all partners, or some normalized measure of interaction gain. A clearer caption, explanation, or example would improve interpretability.

### Questions
See above

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates whether representational similarity between large language models (LLMs) predicts their cooperative or creative behavior in multi-agent interactions. Using 276 model pairs spanning 23 open-weight LLMs from eight model families, the authors evaluate how Centered Kernel Alignment (CKA) scores between model representations relate to performance across four cooperative games (word guessing, public goods, divide-a-dollar, and Keynesian Beauty Contest) and four creative tasks (story writing, biography, haiku, and vacation brainstorming).
Empirical results show a consistent positive correlation between representational similarity and cooperation, and a negative correlation with novelty and diversity. Mixed-effects regressions are used to control for model-specific effects. The paper concludes that representational similarity is a key factor shaping inter-model dynamics in multi-agent LLM systems, suggesting a design tradeoff between cooperation and creativity.

### Strengths
1.Originality: The paper offers a novel and well-motivated perspective by quantitatively linking representational similarity with interactive behaviors among large language models.

2.Methodological rigor: The experimental design is comprehensive, involving diverse cooperative and creative tasks, appropriate mixed-effects regression modeling, and ablations that control for model family, tokenizer, and size.

3.Conceptual depth: The study effectively bridges AI multi-agent interaction research with principles from human social neuroscience, grounding its hypotheses in established cognitive findings.

4.Relevance: The findings provide actionable insights for designing AI collectives, highlighting a meaningful tradeoff between representational similarity (stability and cooperation) and diversity (creativity and novelty).

### Weaknesses
1.Limited mechanistic insight: While correlations are strong, the causal mechanism behind representational similarity’s behavioral influence remains unclear.

2.Task diversity: Cooperative and creative tasks are all text-based; inclusion of multimodal or grounded tasks could test generality.

3.Single similarity metric: The study primarily uses CKA; incorporating neuron-level or mutual information–based representational measures could yield deeper understanding.

4.Interpretability gap: The work does not yet pinpoint which aspects of representation alignment (semantic, syntactic, or higher-order reasoning) drive cooperation versus novelty.

5.Potential dependence on fine-tuned model families: Since most models share similar instruction-tuning paradigms, the observed trends may differ for foundation models without instruction tuning.

### Questions
1.Could the observed tradeoff between cooperation and novelty be an artifact of temperature or sampling strategy (e.g., 0.7 fixed temperature)?

2.How stable are these trends across different conversation lengths or asymmetric tasks (e.g., mentor–student dialogues)?

3.Does representational similarity within specific layers (e.g., middle vs. final layers) predict distinct aspects of cooperation or creativity?

4.Would using a nonlinear similarity measure (e.g., RBF CKA or SVCCA) change the direction or magnitude of the observed effects?

5.Could representational diversity be deliberately engineered (e.g., through orthogonalization penalties) to optimize the cooperation–creativity balance?

6.Regarding Section 3.1, the computation of representational similarity is based on a two-step process (representation extraction and similarity calculation using CKA). Were alternative or complementary similarity estimation approaches (e.g., model stitching, activation distance, or feature subspace comparison) explored? Additionally, given that inputs share the same tokenizer, could this influence the measured similarity and thus confound interpretation? Please clarify the methodological justification and the source or precedent of this computation pipeline, and whether it follows a standard or widely accepted protocol.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates whether representational similarity (measured via CKA) predicts interaction outcomes in multi-agent LLM systems. Through experiments with 276 model pairs across 8 tasks (4 cooperation games, 4 creativity tasks), the authors find that higher representational similarity correlates with better cooperation but reduced novelty.

### Strengths
**Novel research direction**: First systematic study linking internal representations to multi-agent behavior, bridging neuroscience insights with AI systems.

**Comprehensive experimental design**: 276 model pairs from 23 models across 8 diverse tasks, with rigorous mixed-effects regression controlling for model-specific effects.

**Strong theoretical grounding**: Well-motivated by established neuroscience literature on neural synchrony and cooperation (Parkinson et al., 2018; Reinero et al., 2021).

**Robust analysis methodology**: Multiple CKA variants, probe datasets, and reference models tested to ensure findings aren't artifacts of specific choices.

**Clear practical implications**: Provides actionable insights for multi-agent system design regarding model selection trade-offs.

### Weaknesses
**Limited evaluation on coordination benchmarks**: No results on established multi-agent coordination benchmarks like MultiAgentBench(Zhu et al. ACL 2025), at least discuss them in related work. The custom games (Section 4.1) may not capture real coordination complexity.

**Disconnect from practical deployment**: Current multi-agent systems typically use single model types for consistency and cost. The paper doesn't address why practitioners would adopt heterogeneous model deployments.

**Missing comparison with homogeneous systems**: No empirical comparison showing when heterogeneous pairs outperform single-model multi-agent systems. This limits practical applicability.

**Narrow task scope**: Focus on self-designed games and creative writing tasks. Missing evaluation on practical domains like collaborative coding, math problem-solving, or scientific reasoning where multi-agent systems are actually deployed.

**Incomplete analysis of model selection**: While showing similarity effects, the paper doesn't provide methods for selecting optimal model combinations for specific tasks. How should practitioners choose model pairs given a task?

**Expected findings**: While the results (similar models good at collaborations, different are more creative), the results seems just as expected, would expect some more deep analysis and deeper finding

### Questions
**Benchmark evaluation**: Why not evaluate on MultiAgentBench or similar established benchmarks that specifically test coordination and competition abilities?

**Practical deployment**: Given that most production systems use single model types, what scenarios would justify the added complexity and cost of heterogeneous model deployment?

**Performance comparison**: Do any heterogeneous model pairs outperform the best single-model multi-agent system? What's the performance-cost trade-off?

**Task generalization**: How do findings transfer to complex reasoning tasks (coding, mathematics, scientific discovery) where multi-agent systems show real benefits?

**Model selection algorithm**: Can you provide a principled method for selecting model pairs given task requirements and budget constraints?

**Scaling implications**: How does representational similarity affect systems with >2 agents? Does the trend hold for larger agent networks?

**Temporal dynamics**: Do interaction patterns change over multiple rounds? Does prolonged interaction reduce the effect of initial similarity differences?

**Failure analysis**: What specific coordination failures occur with dissimilar models? Can you provide detailed case studies beyond aggregate metrics?

**Cost-benefit analysis**: Given API costs (Table 1 mentions costs but doesn't analyze), when is diversity worth the expense compared to using multiple instances of the best model?

### Soundness
4

### Presentation
4

### Contribution
4
