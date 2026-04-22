# On the Thinking-Language Modeling Gap in Large Language Models

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 6, 4, 6

## Abstract
Large Language Models (LLMs) demonstrate remarkable capabilities in solving complicated reasoning tasks by imitating the human thinking process from human languages. However, even the most capable LLMs can still fail in tasks that are simple for humans. To understand the gap, we construct structural causal models of next-token predictors in human languages. As language is primarily a tool for humans to share knowledge instead of thinking, modeling human thinking from languages can integrate language expression biases into LLMs. More specifically, we show that LLMs can fail to understand implicit expressions -- expression patterns occur less frequently during training. Consequently, LLMs can easily overlook critical information when biased by implicit expressions. We verify our theoretical claims with carefully constructed realistic datasets containing implicit expressions. Furthermore, we also propose a prompt-level intervention to instruct LLMs to carefully expand and focus on all the expressions available. The empirical success of the prompt-level intervention across 11 tasks and 4 representative LLMs, along with the improvements over general reasoning tasks, reaffirms our findings. Our code is publicly available at the project website: https://causalcoat.github.io/lot

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes that large language models (LLMs) struggle with reasoning not because of limited capacity, but because of a fundamental mismatch between the structure of human thought and the structure of language they are trained on. It introduces a Structural Causal Model (SCM) that distinguishes between latent "thought" variables—representing causal reasoning steps—and observable "language" variables that express those thoughts in flexible, sometimes misleading ways. Since natural language can present thoughts in anti-causal or implicit orders, LLMs trained via next-token prediction learn surface correlations rather than true causal relations.

To validate this theory, the authors construct datasets where the underlying reasoning is fixed but linguistic expression varies in two dimensions: L-implicitness (how explicitly key entities are stated) and q-implicitness (how much irrelevant context is added). Across tasks such as WinoControl, BBQ, and Alice, LLM accuracy drops consistently as sentences become more implicit, even though the logical content remains unchanged. This supports the claim that language form—not reasoning complexity—drives many reasoning errors.

Finally, the paper proposes a “Language-of-Thought” (LoT) prompting strategy that asks models to “observe, expand, and echo” all relevant information before reasoning. LoT improves accuracy across multiple benchmarks, suggesting that aligning linguistic expression with underlying causal thought helps LLMs reason more reliably. Overall, the paper argues that reasoning failures in LLMs arise from the limits of language as a representation of thought, and that interventions at the language level can partially bridge this gap.

### Strengths
- This paper adopts a cognitive framing—the *Language of Thought*—to discuss the potential gap between reasoning and language. Through theoretical proofs, it clearly shows that reasoning structures may be unique to specific problems, whereas linguistic expressions can vary widely and do not always align with the underlying reasoning. This insight explains why LLMs may fail to reason accurately, even when solving essentially the same questions.

- The paper introduces multiple benchmarks and defines two types of implicitness. It systematically manipulates linguistic materials across different levels of implicitness, providing empirical evidence that higher implicitness (both L-implicitness and q-implicitness) consistently harms LLM reasoning performance.

- The paper further presents a set of prompt-based interventions, demonstrating performance improvements that conceptually match the predictions derived from the theoretical framework.

- The writing is clear, the visualizations are well-organized, and the overall presentation is coherent and easy to follow.

### Weaknesses
- No reasoning-specific models are tested. While the paper convincingly demonstrates that the language–thought gap exists in LLMs, it would be more informative to see whether reasoning models with post-training (e.g., RL or supervised reasoning finetuning) can overcome this limitation. This would clarify whether post-training paradigms still imitate language-based reasoning or genuinely stimulate model thinking. Such evidence could add significant insight to the paper’s central argument.

- Only linguistic-based reasoning benchmarks are used. Although this is understandable given the paper’s focus, reasoning in general spans broader domains—such as mathematics, coding, causal inference, and abstraction—where linguistic structures interact with deeper symbolic or logical reasoning. Extending the evaluation to such tasks could strengthen the generalizability of the findings.

- No human benchmarks are included. This omission is minor within the paper’s current scope, but comparing humans and LLMs could help answer whether models truly learn to think or merely imitate linguistic reasoning. Such evidence would clarify whether LLMs’ reasoning failures are

### Questions
I currently do not have questions.

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
The paper develops a structural causal model (SCM) of how next-token training on human language can induce language-driven biases in reasoning, especially when expressions are implicit or appear in anti-topological order. The authors then propose a prompt-level intervention—LoT: “please observe, expand, and echo all relevant information”—to surface implicit content. They construct WinoControl and evaluate LoT (and its Echo/Expand ablations) on WinoBias, BBQ, and Alice dataset, plus eight reasoning benchmarks across up to six LLMs. Empirically, LoT often improves over CoT.

### Strengths
1. Precise problem setup. The paper defines the phenomenon clearly and ties it to a concrete causal model.
2. Clean factorization. It separates L-implicitness (how things are said) from q-implicitness (what context is needed) and analyzes them independently.
3. Broad evaluation. Results are reported across many datasets and models.

### Weaknesses
1. Narrow training objective. The analysis focuses on autoregressive next-token prediction and doesn’t discuss masked/bidirectional or fill-in-the-middle training.
2. Missing structured baselines. Methods like self-consistency, Tree-of-Thoughts, or Graph-of-Thoughts aren’t compared under matched budgets.
3. Limited failure analysis. There’s no human study comparing when CoT fails vs. your method succeeds (and the reverse), beyond LLM-as-judge signals.

### Questions
1. Beyond AutoRegressive training: Do your theoretical claims and empirical gains hold for masked/bidirectional or fill-in-the-middle pretraining?
2. Baselines & budgets: How does your method compare to self-consistency, Tree-of-Thoughts, and Graph-of-Thoughts with the same token/latency budgets and decoding settings?
3. Human evaluation: Did you run human annotations to categorize failure cases (CoT→fail / yours→pass and vice versa)? What patterns or error taxonomy emerge?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper aims to understand the gap between language representation and thinking representation. The authors suggest that relying on language for reasoning representation leads to biases in a model reasoning process. They propose a structural causal model to model language structure and its relation with the reasoning process. In addition, they show that Next token predictors are likely biased to use incomplete information for reasoning because of language structure. The main contribution of this work are prompting techniques based on removing such language biases, forcing the CoT LLMs based reasoning models to use all the available information when generating reasoning solutions. An exhaustive empirical analysis is made for several LLMs and complex reasoning benchmarks.

I think the language-thought gap is properly justified and might be a real issue since reasoning biases are generated by language structure. I would highlight structural causal models as a useful theoretical tool to model language and measure next-token predictors reasoning biases. However, regarding its relevance, this work lacks generality in its theoretical results when just covering the 2-premises QA case. It is unclear that the proposed prompting techniques are justified for more complex reasoning tasks and more work should be done in this sense.

On the other hand, exhaustive experiments are appreciated and general improvements are clear over a vast amount of benchmarks when using the LOT prompting strategy. Yet, the ablation analysis is unconclusive for the case of expansion prompting, which rises some doubts on the real contribution of LOT as a whole. In conclusion, I believe this work is properly motivated but improvements could be done in generalizing the theoretical results and also incorporating fine tunning strategies for lowering the language-thought GAP in addition to prompting techniques

### Strengths
1) It is a clearly relevant motivation to understand LLMs learnt biases from our language structure. Moreover, this work tries to formally model how language structure and reasoning interact, given place to a better understanding of reasoning behaviour in LLMs.

2) They give us a way to quantify learnt reasoning biases and modify the way of prompting LLMs in order to reduce this effect.

3) A lot of experiments where done over several modern LLMs over reasonable complex datasets. Showing that in general the proposed approach (slightly) improves reasoning performance.

### Weaknesses
1) Notation is confusing and non standard. At the beggining is hard to parse the use of expression sets, the use of \pi for order (usually \pi is left for permutations). Definitions are not complete and misleading, for example, definition 2.1 assigns a conditional probability on l_k but l_k is part of the given sequence.

2) Even though the gap between language based reasoning and human reasoning is a generlized issue with modern LLMs, the main theorem (2.4) just cover the particular case of two-premise QA. Which is ommited in later discussions, assuming the result applies for an arbitrary amount of premises (i.e they say without loss of generality but never prove the general case).

3) The expansion prompting technique is shown to even exacerbate a model’s biases but no fixes or arguemnts are given for this result.

4) It is unclear why prompting techniques are the best way to solve the so called language-thought gap. No discussion is made on improving training data or model’s architecture.

Minor comments
Typo in line 137: “LLMS LLMS”
Typo in line 155: “Even if the…”
Typo in line 234: “motivation of LOTis”

### Questions
1. Could the authors clarify whether Theorem 2.4 formally extends to problems with more than two premises (k > 2), and if so, provide a general statement or assumptions required for this extension?

2. Can the authors justify (or at least discuss) why prompting is considered the most appropriate intervention rather than modifying the training objective, data, or architecture to reduce the language-thought gap?

### Soundness
2

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
2

### Summary
This paper investigates the thinking-language modeling gap in LLMs, which is motivated by the argument that language primarily serves communication rather than reflecting thoughts, and training LLMs on human-written languages can introduce bias into their reasoning. The authors build structural causal models (SCMs) to formalize how NTP training can lead to incomplete reasoning when expressions are not aligned with the underlying causal structure. They introduce implicit expressions that cause LLMs to overlook critical information and verify the effectiveness by constructing a controlled dataset and varying levels of implicitness. In addition, they propose a prompt-based intervention method called Language-of-Thought (LoT), which instructs LLMs to observe, expand, and echo relevant information. Evaluations on benchmarks (WinoBias, BBQ, Alice) demonstrate LoT reduces biases and improves reasoning across open and closed LLMs. Additional evaluations on general reasoning tasks, where CoT is limited, show most gains across 6 LLMs.

### Strengths
- The use of SCMs provides a structured, causal lens to analyze LLM reasoning failures. The formulation offers a novel theoretical contribution that explains phenomena including order sensitivity and context overlooking, and could potentially influence and inspire future work.
- LoT prompt is simple, easy to apply, and model-agnostic, showing consistent improvements across multiple LLMs and tasks. Token cost studies demonstrate that the improvements do not come from increased output length. 
- The designed WinoControl dataset allows controlled experiments on implicitness levels and enables verification of the hypothesis.

### Weaknesses
- Theoretical assumptions of perfect knowledge and Markov conditions are simplified and could limit broader applicability. 
- Model behaviors are validated by the LLM-as-judge approach, which could raise concerns. Manual verification would strengthen claims.
- The behaviors and performances of LoT in more advanced and popular settings are unclear, including self-consistency[1], tree-of-thoughts[2], and ReAct [3]. 

[1] Wang, Xuezhi, et al. "Self-consistency improves chain of thought reasoning in language models."
[2] Yao, Shunyu, et al. "Tree of thoughts: Deliberate problem solving with large language models."
[3] Yao, Shunyu, et al. "React: Synergizing reasoning and acting in language models."

### Questions
- How sensitive are LoT improvements to the exact phrasing of the prompt? 
- In Theorem 2.4, how do violations of perfect knowledge or Markov conditions affect the bound's usefulness?
- What training-time mitigation methods do you envision?

### Soundness
3

### Presentation
3

### Contribution
3
