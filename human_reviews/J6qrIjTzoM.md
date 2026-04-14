# Interpretability of Language Models for Learning Hierarchical Structures

- Decision: Reject
- Scores: 6, 8, 3, 8

## Abstract
Transformer-based language models are effective but complex, and understanding their inner workings is a significant challenge. Previous research has primarily explored how these models handle simple tasks like name copying or selection, and we extend this by investigating how these models grasp complex, recursive language structures defined by context-free grammars (CFGs). We introduce a family of synthetic CFGs that produce hierarchical rules, capable of generating lengthy sentences (e.g., hundreds of tokens) that are locally ambiguous and require dynamic programming to parse. Despite this complexity, we demonstrate that generative models like GPT can accurately learn this CFG language and generate sentences based on it. We explore the model's internals, revealing that its hidden states precisely capture the structure of CFGs, and its attention patterns resemble the information passing in a dynamic programming algorithm.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work investigates how LM grasp complex, recursive language structures defined by context-free grammars (CFGs). It develops a multi-head linear probing method to verify that the model’s hidden states linearly encode NT (non-terminal) information, which is interesting as pre-training on CFG-generated strings does not expose the CFG structure. The results also suggests that GPT models learn CFGs by implementing a dynamic programming-like algorithm.

### Strengths
(1) I think the idea of studying LM's understanding of synthetic CFG is super interesting and novel.

(2) Some of the findings (that LM can really learn those complicated CFG, and implementing DP-like operation inside) are quite surprising.

### Weaknesses
(1) The paper very condensed, and I would enjoy reading it more if it is simplified (including figures, tables, notations, etc.).

(2) The scope is centered around CFG, which is novel, but still limited.

### Questions
N/A.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes an analytical method for explaining the inner workings of Transformers. By defining CFG rules with specific properties to generate synthetic data for language models to learn, the consistency between the model's outputs and the CFG rules is observed using various methods, including generating subsequent content from a given prefix to observe whether it strictly follows the CFG rules, as well as probing the hidden states with methods like probing heads. Numerous pieces of evidence suggest that the model's hidden states can accurately capture the structural information of the CFG, while also verifying that self-attention works like dynamic programming.

### Strengths
1. Although the self-attention in Transformers has been speculated to simulate structures similar to human languages, and many related works have empirically validated this point (e.g. https://arxiv.org/abs/2004.14786), there have been few systematic demonstrations. This paper systematically demonstrates that Transformers can learn CFG grammars with long-range dependencies and local ambiguities, two properties that are highly relevant to human language. The entire deduction approach is scientific and rigorous.

2. The method of defining generative rules with certain properties to generate synthetic data and then simulating and analyzing it with Transformers may provide valuable insights for future related work.

3. The paper is well-written, with a clear and organized structure.

### Weaknesses
1. Considering programming languages, which are also CFGs, Transformers may not only learn the syntax of a program but also simulate the execution logic, such as sorting and other complex behaviors. These capabilities can also be learned by large models. Therefore, the inner workings of Transformers seem not limited to latent structure learning but also involve fitting causal logic. Learning CFGs may be a subset of their capabilities, though I acknowledge that this is not something that can be fully discussed in a single work.

### Questions
1. Some studies, such as (https://arxiv.org/abs/2203.00281), explicitly compose tokens via dynamic programming to learn latent structures and sentence representations. These methods even perform slightly better than Transformers on text representation tasks, yet their reasoning capabilities are significantly weaker. They can also learn CFG structures and even explicitly model the dynamic programming process, similar to the abilities of Transformers described in the paper. So what accounts for their disparity on reasoning ability ?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper studies how well transformer language models can learn to process structured context-free languages, specifically those that require handling long-range dependencies and resolving local ambiguities. To do this, the authors use artificially generated context-free languages and find that transformers can learn these languages, but the type of positional encoding used in the model makes a difference. They also introduce a probing method to test how well the models learn the structure of a context-free language. The authors suggest that transformers might be processing input strings similarly to dynamic programming techniques used in context-free parsing.

### Strengths
The paper makes useful contributions in studying general language models with targeted formal languages. I find the following to be particularly useful:
- Examining language models with formal languages is a valuable approach for understanding their capabilities and limitations.
- The use of synthetic grammars is well thought out and effective, especially since these grammars are designed to test specific properties, like handling long dependencies.
- The results cover a good range of transformer design choices, providing a thorough look at how these models handle different CFG structures.

### Weaknesses
There are areas where clarity and focus of the paper could improve:
- **Complexity in Presentation**: The paper is sometimes hard to follow. The Introduction mixes motivation, literature review, and specific results, which makes it less effective as an overview. The amount of notation used and the naming of the CFGs, like `cfg3a`, is also hard to connect with anything concrete, and references to niche topics like the 2006 IOI problem add to the complexity.
- **Dense Results Section**: With so many results presented, organization is key to helping readers keep track. Referring to results by descriptive names rather than numbers might help readers better understand each finding at a glance.
- **Setup Comments**:
    - It’s worth noting that long-range dependencies don’t require context-free languages; even simple regular languages (e.g., “first symbol = last symbol”) can capture long dependencies.
    - The CFGs defined in the paper technically describe finite languages (and are therefore regular) because recursion depth is limited by L. This limitation might make it hard to tell whether the models are learning true recursion or just memorizing up to the required depth.
- **Comparison to Related Work**: Adding a discussion of related studies, especially those testing LMs on the Chomsky hierarchy, would help position the results more clearly within the existing research landscape.

### Questions
- In the Conclusion: While you might not want to discuss ongoing or future work in detail, it would be helpful to hear a bit more about potential directions or applications of this research.
- RNNs are often compared to finite-state automata or pushdown automata. Did you consider comparing transformers to simple RNNs or LSTMs in this study?

### Soundness
3

### Presentation
1

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
This paper systematically investigates how Transformer language models learn and process complex recursive language structures by designing a set of synthetic context-free grammars (CFGs) with long-range dependencies and local ambiguities. The main contributions are: (1) introducing a multi-head linear probing method to analyze internal representations; (2) developing new visualization and quantification tools to understand attention patterns; (3) demonstrating that GPT models can learn complex CFGs in a dynamic programming-like manner and revealing the critical role of boundary-based attention in handling long-range dependencies; (4) validating the model's effectiveness and robustness in handling such complex language structures. These findings offer new insights into the inner workings of large language models.

### Strengths
The highlight of this paper is its novel approach to studying large language models by using carefully designed synthetic CFGs to explore model mechanisms. This method is innovative because it offers a controlled, quantifiable experimental environment, making complex model behaviors interpretable. By demonstrating how CFGs can be used to investigate the learning processes of models, this work provides a powerful analytical framework for future researchers. Combining theoretical analysis with experimental design, this approach not only aids in understanding existing models but also inspires new directions for probing the inner workings of even larger language models.

### Weaknesses
The paper mainly focuses on studying canonical CFGs, and the exploration of non-canonical CFGs is not sufficiently in-depth.

The research results are primarily based on specific synthetic CFGs, and their generalizability needs to be further verified.

### Questions
The paper suggests that Transformer models learn CFGs by emulating dynamic programming (DP) algorithms. Would it be possible to test this hypothesis more directly, perhaps by comparing the model’s internal representations to the states of known DP algorithms? The current evidence feels somewhat indirect.

The paper also notes BERT’s weaker performance in predicting deep non-terminal (NT) structures but doesn’t delve into the reasons behind this. While attributing it to the locality of the MLM task seems plausible, more direct evidence would strengthen this claim.

Could the authors discuss potential limitations? Acknowledging the method’s constraints would be helpful for future research directions.

### Soundness
4

### Presentation
3

### Contribution
4
