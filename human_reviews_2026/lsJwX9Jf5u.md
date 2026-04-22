# Emergence of Superposition: Unveiling the Training Dynamics of Chain of Continuous Thought

- Avg Score: 6.40
- Decision: Accept (Poster)
- Scores: 8, 6, 6, 6, 6

## Abstract
Previous work shows that the chain of continuous thought (continuous CoT) improves the reasoning capability of large language models (LLMs) by enabling implicit parallel thinking, and a subsequent work provided theoretical insight by showing that a two-layer transformer equipped with continuous CoT can efficiently solve directed graph reachability by maintaining a superposition of multiple reasoning traces in the continuous thought. However, it remains unclear how the superposition mechanism is naturally learned from gradient-based training methods. To fill this gap, we theoretically analyze the training dynamics of a simplified two-layer transformer on the directed graph reachability problem to unveil how the superposition mechanism emerges during training in two training stages -- (i) a *thought-generation* stage that autoregressively expands the continuous thought, and (ii) a *prediction* stage that converts the thought into the final answer. Our analysis reveals that during training using continuous thought, the index-matching logit, an important quantity which reflects the strength of the model's local search ability, will first increase and then remain bounded under mild assumptions. The bounded index-matching logit effectively balances exploration and exploitation during the reasoning process: the model will exploit local problem structures to identify plausible search traces, and assign comparable weights to multiple such traces to explore when it is uncertain about which solution is correct, which results in superposition. Our experimental results tracking the growth of logits further validate our theory.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper focuses on the gap of whether superpositional reasoning (superposition) in continuous Chain-of-Thought (continuous CoT) naturally emerges under standard gradient training, chooses directed graph reachability, and provides a theoretical analysis of the training dynamics of a simplified two-layer Transformer, explicitly distinguishing the two phases of “thought generation” and “answer prediction” to explain the source of the mechanism; the core conclusion is that under continuous-thought training, the “index-matching logit” (which characterizes local search strength) first increases and then, under mild assumptions, remains bounded, thereby achieving an exploration–exploitation balance under uncertainty: it both leverages local structure to advance and assigns comparable weight to multiple candidate paths, so superposition arises naturally; the authors also validate this by experiments that track the growth of logits.

### Strengths
1. Focused problem that fills a gap: the paper centers on the core mechanism of why superposition would naturally arise during training in continuous Chain-of-Thought (continuous CoT) and provides an explanation from the perspective of training dynamics, going beyond prior work that only proves the existence of feasible solutions/expressivity without discussing whether they can be learned by gradient methods. The abstract and introduction clearly define the goal and differentiated positioning.

2. Solid key theoretical results: it formally compares two objectives (COCONUT vs. COCONUT-BFS), proving that under the former the index-matching logit is bounded whereas under the latter it diverges; it thus concludes that a bounded logit leads to a smoother next-step distribution, thereby retaining multiple candidate paths and promoting superposition. This conclusion offers a convincing explanation for why training does not over-commit to a single path.

3. Mutual support between empirics and theory: a two-layer GPT-2 decoder trained from scratch achieves 96.2% accuracy on a subset of ProsQA; during training the “frontier vs. non-frontier” logit gap first rises and then stabilizes, consistent with the “bounded” theory; length extrapolation appears at longer steps. These observations support the chain “bounded logit → superposition/exploration–exploitation balance.”

### Weaknesses
1. Ablation and robustness analyses are somewhat limited: there is a lack of systematic ablations over factors such as number of layers/heads/width and learning rate; nor are results under different random seed settings presented (there is only a verbal note in a footnote that different random seeds were used). The dataset is limited to a subset of ProsQA. It is recommended that the authors conduct more detailed experimental validation on more related datasets and under more settings.

2. (Small suggestion) To facilitate readers’ quick understanding of the paper’s contributions, it is recommended to describe an informal version of the conclusions in the first half of the paper.

### Questions
Based on your theoretical understanding, can you propose any targeted insights or approaches for enhancing a model’s reasoning ability during the training process?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper offers a theoretical and empirical account of how the superposition principle arises in transformers under gradient-based training. Focusing on a two-layer transformer with continuous-thought decoding, the authors delineate two curriculum stages: (i) autoregressive generation of continuous thoughts and (ii) final prediction from the aggregated latent sequence. They show that, under an appropriate loss, attention logits remain bounded in Stage I, which expands the next thought. Leveraging max-margin analysis, they establish direction convergence and obtain accuracy guarantees on unseen examples.

### Strengths
- Theoretical rigor: Under some assumptions, the paper provides rigorous theory about how bounded attention logits and superposition naturally arise from gradient-based training. From a methodological point of view, this paper simplifies the difficulties of theoretical analysis by closing the dynamics through some assumptions.
- Insight into reasoning processes: Through the focus on the graph reachability problem, the analysis reveals concrete relationships between loss design and the evolution of attention weights.
- Experimental verification: The theoretical results were partially verified through some experiments. Section 5 provide the evidence that the theoretically predicted logit behaviors manifest during actual learning.

### Weaknesses
- The clarity of the article's presentation needs significant improvement. For example:

  - In the third subsection of section 3, the article introduces prior knowledge about the first decoder layer (Eq. (1)) based on some related work. This should be stated as an assumption and verified through experiments.
  - Similarly, the reparameterization introduced in Eq. (3) and Eq. (7) should also be considered as an assumption and partially verified through experiments.
  -  In section 5.1, the explanation of $\mu$ is confusing. More intuition or discussion is needed to explain why the chosen logit can be a proxy for $\mu$. In contrast, the explanation in 5.2 does not have this problem.
  - On page 5, the fourth line from the bottom considers two losses under permutation. Intuitively, the permutation invariance of the data will lead to similar properties of the parameters (as shown in Lem. 1), but the relevant insight needs to be pointed out more explicitly.
  - The statement of Theorem 2 needs to be improved. According to the expression of $\beta$, all parameters should be regarded as vectors, and the theorem should show the boundedness of the operator between vectors.

- The plausibility of the assumptions or motivations is unclear.
  - In Eq. (3), the initialization of $\mu$ is setted as 0. However, it is in an inductive process. Specifically, this is step c of the course learning. Assuming $\mu$ is 0 does not seem reasonable considering that the attention parameters have been changed during the previous training.  Additional discussion or proof is needed to complete the inductive process.
  - As mentioned before, priors about the behavior of reparameterization and the first layer of attention need to be empirically supported.

- The significance of some experiments is unclear.

In Section 5.2, the article analyzes the attention of the answer token and the reasoning token. Although it shows similar results to the first half of Theorem 3, it does not show how the behavior of accuracy changes as the direction converges.

- Scalability of experiments.
The paper experiments on a two-layer transformer. The relevant experiments, definitions, and detailed analysis are not yet clear in their guidance for training with more layers or for attention behavior. A supplementary example with more layers would be helpful.

### Questions
Please refer to the Weaknesses.

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
This work theoretically studies how the superposition mechanism in continuous CoT is learned from gradient-based methods. It derives theoretical results by formulating the training dynamics of a two-layer transformer on the graph reachability problem in two training stages of thought generation and prediction. It also reveals the importance of the bounded index-matching logit and its connection to the tradeoff between exploration and exploitation.

### Strengths
1. This work addresses an important problem: as the previous work (Zhu et al., 2025) theoretically shows that the core superiority of continuous CoT lies in reason by superposition, this work solves a next question: how the superposition can be learned by gradient-based methods. 

2. The contributions of this work are sufficient as it derives multiple novel theoretic results on the gradient-based learned superposition mechanism.

### Weaknesses
1. The presentation should be improved. 
(1) The draft contains many typos and inconsistency, which should be thoroughly revised. 
(2) As the content is mostly hard to follow, more intuitive visuals may need to be added for better understanding. 

2. This work discusses only two-layer transformer. It would be better to discuss about the generalization to models with more layers. 

3. Practical implication may be supplemented. Based on the theorems proposed in this work, does one design a better training recipe for continuous CoT?

### Questions
Please refer to the Weaknesses.

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
2

### Summary
Building on recent work in continuous Chain-of-Thought, the authors seek to explain why the training procedure yields strong reasoning behavior. They decompose the model’s prediction into two stages, develop a mathematical analysis focusing on the index-matching logit, an important quantity in the model, and present experiments that support their claims.

### Strengths
1. The paper formulates a clear, important problem in continuous Chain-of-Thought, explaining why such capability is enabled during training.
2. The claims are supported by precise mathematical derivations and well-designed experiments that substantiate the analysis.

### Weaknesses
1. The writing is careful but not intuitive. As a non-expert reader in this domain, it is important to provide a more intuitive background introduction. In this paper, it is especially important to introduce index-matching logic and related concepts gradually and intuitively. 
2. The paper is focused on one single task of graph reachability. Could it be tested on more tasks and how would it be generalized to other tasks?

### Questions
See weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates the superposition mechanism underlying continuous chain-of-thought (continuous CoT, or COCONUT) in large language models (LLMs), a paradigm where reasoning traces are maintained in a continuous latent space rather than discrete tokens. Building on prior work that theoretically showed a two-layer transformer with continuous CoT can solve directed graph reachability via superposition of reasoning traces, the authors address a critical open question: Do gradient-based training methods naturally induce this superposition?

To answer this, the authors analyze the training dynamics of a simplified two-layer transformer on the directed graph reachability task, splitting training into two stages: (1) a thought-generation stage, where the model autoregressively expands continuous thoughts; and (2) a prediction stage, where it converts these thoughts into a final answer. Their core theoretical insight is that the index-matching logit—a metric of the model’s local search ability—remains bounded under mild assumptions during COCONUT training. This bounded logit balances exploration (assigning comparable weights to multiple plausible reasoning traces) and exploitation (leveraging local graph structure), directly enabling superposition. The authors validate these claims with experiments on a GPT-2-style model, showing logits stabilize at bounded values (consistent with theory) and the model achieves 96.2% test accuracy on graph reachability.

### Strengths
1. Mechanistic insight: The paper demystifies a core advantage of continuous CoT—superposition—by tying it to a measurable quantity (index-matching logit) and its training dynamics. This is a rare example of "reverse-engineering" why a latent reasoning method works, rather than just showing it works.

2. Theory-experiment alignment: Experiments are not just "proof of concept" but directly validate theoretical predictions (e.g., logit saturation, length generalization to c=3,4 despite training only on c=1,2). This strengthens confidence in the mechanistic claims.

3. Practical relevance: The focus on single demonstrations per sample (COCONUT) reflects real-world training constraints (where large CoT datasets are costly to annotate). This makes the results more applicable than methods relying on exhaustive BFS-style supervision.

### Weaknesses
1. Task generality: The analysis is limited to directed graph reachability, a synthetic task with well-defined "reasoning traces" (paths). It is unclear if the superposition mechanism extends to tasks with more ambiguous traces (e.g., open-domain QA, where reasoning steps are less structured). The paper should acknowledge this limitation and suggest future work on diverse tasks.

2. Model scalability: The experiments use a small two-layer model. While this is standard for mechanistic work, the paper could include a brief discussion of how deeper models might interact with superposition (e.g., whether layers stack superposition or refine it) to address scalability concerns.

3. Alternative mechanisms: The paper does not rule out competing explanations for superposition. For example, could the model’s weight tying (assumed for convenience) or curriculum learning (multi-stage training) also contribute to superposition? A short ablation on weight tying or curriculum learning would strengthen the causal link between COCONUT training and superposition.

### Questions
1. The paper notes that logits plateau in practice but grow unbounded in theory, attributed to interactions between thought-generation and prediction stages. Could you elaborate on why these interactions cause plateauing? For example, does the prediction stage’s loss implicitly regularize the thought-generation logits, or is it a numerical effect (e.g., gradient clipping in practice)?

2. The model exhibits "length generalization" (e.g., generating c=4 continuous thoughts despite training on c≤2). Do you have evidence that this generalization relies on superposition? For example, if you perturb the superposition (e.g., zero out weights for non-optimal traces), does length generalization break?

3. The theoretical analysis assumes linear attention. How might non-linear attention (e.g., softmax attention, the standard in LLMs) affect the index-matching logit’s dynamics? Would softmax attention’s normalization further bound logits, or introduce new complexities (e.g., vanishing gradients)?

### Soundness
3

### Presentation
3

### Contribution
3
