# Combinatorial Creativity: A New Frontier in Generalization Abilities

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Artificial intelligence (AI) systems, and Large Language Models (LLMs) in particular, are increasingly employed for creative tasks like scientific idea generation, constituting a form of generalization from training data unaddressed by existing conceptual frameworks. Despite its similarities to compositional generalization (CG), combinatorial creativity (CC) is an \emph{open-ended} ability. Instead of evaluating for accuracy or correctness against fixed targets, which would contradict the open-ended nature of CC, we propose a theoretical framework and algorithmic task for evaluating outputs by their degrees of \textit{novelty} and \textit{utility}. From here, we make several important empirical contributions: (1) We obtain the first insights into the scaling behavior of creativity for LLMs. (2) We discover that, for fixed compute budgets, there exist optimal model depths and widths for creative ability. (3) We find that the \emph{ideation-execution gap}, whereby LLMs excel at generating novel scientific ideas but struggle to ensure their practical feasibility, may be explained by a more fundamental \emph{novelty-utility tradeoff} characteristic of creativity algorithms in general. Though our findings persist up to the 100M scale, frontier models today are well into the billions of parameters. Therefore, our conceptual framework and empirical findings can best serve as a starting point for understanding and improving the creativity of frontier-size models today, as we begin to bridge the gap between human and machine intelligence.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper seeks to formalize combinatorial creativity as a distinct generalization capability and introduces a framework to evaluate creativity as a continuous trade-off between novelty and utility in structured conceptual spaces. The authors operationalize creativity via graph-based constrained path-generation tasks, paired with explicit novelty and feasibility scoring, enabling systematic measurement beyond binary correctness. Using this setup, they conduct controlled scaling experiments across Transformer architectures with varying parameter sizes and uncover two key findings: (i) creativity does not scale monotonically with size but has optimal depth-width ratios, and (ii) a novelty–utility trade-off exists even for bigger models. The work provides the empirical evidence that creative capability saturates beyond simple scaling.

### Strengths
Timely contribution to the recent area of improving creativity in LLMs, with new creativity evaluation metrics.

Findings here can help practitioners choose model hyperparameters for optimal creativity.

### Weaknesses
Focuses on a rather limited aspect of creativity (Combinatorial Creativity) which is already included in other work (e.g. https://arxiv.org/pdf/2504.15266). There is also a lack of comparison or description of the difference between these similar works with proposed approach.

Lack of in-depth analysis of observations and findings here, for example, why is there a depth-width trade-off for creativity.

### Questions
What is the key difference between the proposed methods and other recent and similar work (e.g. https://arxiv.org/pdf/2504.15266)?

Comment:
Figure 1 & 2: I would suggest the authors to use these figures to more clearly illustrate the overall concepts of the contribution. It is unclear what message the current illustration is trying to convey.

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
This paper introduces a formal framework to evaluate "combinatorial creativity" (CC) in LLMs, distinguishing it from compositional generalization (CG). The authors propose an open-ended, algorithmic task where creativity is modeled as constrained pathfinding in a large, synthetic "conceptual space" (a graph). The framework provides continuous metrics for "novelty" (based on path length and label rarity) and "utility" (based on satisfying inclusion/exclusion constraints). The authors then empirically test decoder-only Transformers (based on a GPT-2 architecture) at 1M, 10M, and 100M parameter scales. Key findings include: (1) an optimal, non-monotonic width-to-depth ratio for creativity at a fixed parameter budget; and (2) a persistent novelty-utility tradeoff that does not diminish with scale, which the authors propose as a potential explanation for the "ideation-execution gap" in scientific idea generation.

### Strengths
1. The paper provides a mathematically precise framework (Definitions 1-6) for defining and measuring one specific view of combinatorial creativity.
2. The initial sections (Introduction, Background) provide a good survey of creativity and clearly situate the work against compositional generalization (Table 1).
3. The architectural sweep (Figure 3) is systematic for the defined task, and the error analysis (Figure 5) provides some insight into failure modes on this task.

### Weaknesses
1. The paper's primary flaw is its severe scale limitation. Experiments top out at 100M parameters using a GPT-2-like architecture. There is no evidence that these findings can scale up to the performance of modern, multi-billion-parameter large language models, especially those that exhibit 'emergent abilities' not present at the 100M scale. This failure to test at a relevant scale means the framework's applicability to the models the community actually uses is unproven, which severely undermines the paper's contribution.
2. The paper provides zero evidence that performance on this synthetic, semantically-vacant graph task correlates with any real-world creative ability. This "toy" task setup severely limits the generalizability and impact of all findings.
3. The key takeaways, such as the existence of an optimal E/L ratio for a 100M toy model and the persistence of a tradeoff at this small scale, are simply not interesting or impactful conclusions for a top-tier conference. The payoff is not worth the setup.
4. The paper is rife with over-claiming.
  - Claiming this 100M-parameter-limit study "casting doubt on the long-term creative potential of LLMs" is an enormous, unsupported extrapolation.
  - Claiming to explain the "ideation-execution gap" via the simplistic mapping in Table 2 is hollow.
5. The novelty metric (Eq. 2) hard-codes "longer is more novel," which is debatable. The use of an undirected graph to "prevent the reversal curse" feels like an artificial simplification.
6. The paper evaluates creativity using greedy decoding, focusing on a single "best" output. This ignores diversity, a fundamental component of creativity.

### Questions
*Also see weakness above

1. The paper's framework is validated only on 100M GPT-2 models. Can you provide any evidence or strong theoretical argument for why this framework and its findings (e.g., optimal E/L ratios, tradeoffs) would be relevant to evaluating modern, multi-billion-parameter foundation models (e.g., Llama 3.1-70B)?
2. Could you comment on the apparent disconnect between the ambitious, broad theoretical framework for 'creativity' and the very narrow, specific empirical study on an outdated architecture? The payoff (e.g., "an optimal E/L ratio exists") seems uninteresting and incommensurate with the grand setup.
3. The novelty metric (Eq. 2) assumes longer paths ($h$) are more novel. How sensitive are your findings to this choice? What happens if you define novelty as an inverse function of path length, rewarding "elegant shortcuts"?
4. Why did you not attempt to validate this framework on even one real-world, non-synthetic task? Even a simple task on a real knowledge graph (e.g., ConceptNet) would make the claims much more convincing.

I want to emphasize that I agree with the overall research direction, which I find valuable. I strongly encourage the authors to address these questions in their rebuttal. Thank you for your work!

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
This paper explores combinatorial creativity (CC) as a new frontier in the model of generalization behavior. The authors discuss various forms of creativity and compare CC with other types of generalization and other CC frameworks from various aspects, e.g., open-endedness, degree of novelty, and degree of utility, where the proposed CC framework measures the creativity based on sampling from the joint distribution of novelty and utility. The authors then experiment with the framework with the GPT-2 decoder and compare the architectural creativity across models from 1M to 100M scales. Results show the relevance between creativity scores and model architecture design, the trade-off between novelty and utility, and the potential ideation-execution gap. With the experiments and analysis, the authors argue that the proposed framework can help understand the creativity of modern AI models.

### Strengths
1.	This paper is well-written and easy to follow, with clear motivation on the framework and discussions.
2.	The discussion of creativity (Section 2.1 and Section 2.2) is interesting, covering classic and modern frameworks.
3.	This work provides a systematic framework for quantifying the degree of novelty and utility, which can be used for automatic system-level evaluation.

### Weaknesses
1.	The discussion is missing a line of recent literature on applying combination creativity for ideation. Please check: Scideator (https://arxiv.org/abs/2409.14634), CHIMERA (https://arxiv.org/abs/2505.20779), and Ramon Llull’s thinking machine (https://arxiv.org/abs/2508.19200) 

2.	The authors claim that the findings provide insights for the ideation-execution gap proposed in Si et al. (2025). However, the original gap was describing the quality between the ideas proposed by the frontier language models and the execution done by real human researchers, with the potential help of code agents. It is hard to draw transferable insights of the relevance between the failure mode at the ideation execution experiments and the current framework with 100M scale training (and token-level definition of  novelty)

3.	Following point 2, the current experiment settings and scales limit the generalization and applicability of findings to the real tasks requiring novelty (e.g., ideation). (1) The random graph task (introduced in Section B.1) is too ad-hoc and hard to generalize to natural language tasks (since the tokens and n-grams are innately relevant). To make my comment concrete, please check A Taxonomy of Transcendence (https://arxiv.org/pdf/2508.17669), where the authors construct a knowledge graph-based simulation to assess the potential generalization behavior, where the fact-based node construction can potentially provide extended insight to general creativity discovery; (2) the scales of the models used also limit the strengths of the claim on the scaling behavior. As a common practice to show the scaling behavior, the authors should train a series of models following the chinchilla scaling laws (https://arxiv.org/pdf/2203.15556) to hedge the confounders, such as over-/under-training, i.e., train and test a series of models following their estimated optimal training flops.

### Questions
1. Most of the experimental settings are introduced in Appendix B. Is it possible to adjust the flow to cover at least some details in the main paper?

2. Figure 4 seems to provide limited insight. It seems intuitive that the novelty drops with more utility constraints. Is it possible to consider more conditions for the visualization, e.g., the complexity of the constraints?

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
2

### Summary
This paper introduces a formal framework for evaluating combinatorial creativity (CC) in AI systems, modeling it as path-finding in conceptual graphs with logical constraints. Through systematic experiments on decoder-only Transformers (1M-100M parameters), the authors establish: (1) scaling laws for creativity, (2) optimal architectural configurations (depth ~8 layers, E/L ratio 200-300 for fixed budgets), (3) a fundamental novelty-utility tradeoff that persists at scale, and (4) error analyses connecting to the ideation-execution gap in LLM-generated scientific ideas.

### Strengths
- First formal framework for combinatorial creativity in AI with explicit, measurable novelty (Definition 4) and utility (Definition 5) functions. Established with interesting connection with problems in graph-theory.

- Systematic design of experiments: data construction reflects the task design, and exhaustive hyper parameter search to find the optimal model structure.

### Weaknesses
1. **Lack of Semantic Grounding:** The graph in the training data is synthetically generated with random connectivity. Edge labels are also randomly assigned across many node pairs without semantic constraints. Real creativity involves understanding semantic relationships between two nodes (knowledges, facts, etc.), and real edges connecting two nodes may not have exactly the same label (e.g., can be paraphrased). So it is unclear whether findings transfer to real conceptual spaces where semantic constraints matter.

2. **Edge Label Space Design:** Edge label vocabulary (~26 labels) is much smaller than node space. From a graph-theory perspective, the query complexity can be much lower than expected. It is also unclear whether the optimal width/depth ratios are artifacts of this design choice. The authors should explore when the edge labels are expanded, how will the optimal trade-off change.

3. **Presentation Issues:** For example, inclusion/exclusion constraints $(I, X)$ are introduced in Definition 3 before their purpose is explained (Definition 5), and such explanation is implicit. It is better to re-mention $x=(u, v, I, X)$ in Definition 5.

### Questions
Aside from the concerns raised in the Weaknesses part, I have the following questions:

1. In Definition 5, utility scales by number of constraint labels ($|I|$, $|X|$) rather than number of affected edges. Should they be replaced with the actual number of edges? If label 'a' appears in $1000$ edges but label 'b' in $10$ edges, excluding 'a' should be harder than excluding 'b'.

2. In Definition 4, what is $w_{l_i}$? Is this the portion of label $l_i$ in the connecting edges of node $v_{i - 1}$? Also, since for a node $v_{i - 1}$, there can be multiple edges with the same label $l_i$, choosing the next node $v_i$ is also not deterministic. Shall it be considered into $S(P)$ as well?

3. Section 3.3: How are "utility levels" formally defined? Is level $l = |I| + |X| + 1$?

### Soundness
2

### Presentation
3

### Contribution
2
