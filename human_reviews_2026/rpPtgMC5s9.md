# Relational Transformer: Toward Zero-Shot Foundation Models for Relational Data

- Decision: Accept (Poster)
- Scores: 6, 6, 6

## Abstract
Pretrained transformers readily adapt to new sequence modeling tasks via zero-shot prompting, but relational domains still lack architectures that transfer across datasets and tasks.
The core challenge is the diversity of relational data, with varying heterogeneous schemas, graph structures and functional dependencies.
In this paper, we present the Relational Transformer (RT) architecture,
which can be pretrained on diverse relational databases and directly applied to unseen datasets and tasks without task- or dataset-specific fine-tuning, or retrieval of in-context examples. RT (i) incorporates task specification via task table prompting, (ii) tokenizes cells with table/column metadata, (iii) is pretrained via masked token prediction, and (iv) utilizes a novel Relational Attention mechanism over columns, rows, and primary-foreign key links.
Pretrained on RelBench datasets spanning tasks such as churn and sales forecasting, RT attains strong zero-shot performance,
averaging 93% of fully supervised AUROC
on binary classification tasks
with a single forward pass of a 22M parameter model,
as opposed to 84% for a 27B LLM.
Fine-tuning yields state-of-the-art results with high sample efficiency. Our experimental analyses show that RT's zero-shot transfer leverages task context,
relational attention patterns and schema semantics. Overall, RT provides a practical path toward foundation models for relational data.
Code, models, data: https://github.com/snap-stanford/relational-transformer.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper tackles the open problem of building a schema-agnostic foundation model for relational databases that can transfer across datasets and tasks without task-specific fine-tuning or in-context retrieval. The proposed Relational Transformer operates at the cell level, tokenizing each value together with its column and table names, pretraining with masked-token prediction, and introducing Relational Attention masks that selectively attend across columns, features, foreign-key neighbors, and a global channel. A task table is concatenated to provide task context for zero-shot prediction. On RelBench datasets spanning classification and regression tasks, RT shows strong zero-shot transfer—e.g., the abstract reports ~94% of fully supervised AUROC for binary classification using a 22M-parameter model, and substantially better sample efficiency during fine-tuning—while a 27B LLM under comparable context performs markedly worse at much higher inference cost. The paper presents ablations isolating the effect of self-labels, schema semantics, and attention masks, and discusses current scope limits (e.g., link prediction/recommendation). Overall, the core contributions are: (i) a cell-level tokenization that unifies relational prediction as masked token prediction; (ii) a set of relational attention masks that encode schema structure; and (iii) a zero-shot task-table prompting interface enabling transfer across heterogeneous schemas.

### Strengths
1. The cell-level tokenization plus Relational Attention masks is a clean, general design that bridges tabular “foundation models” and relational deep learning. The work articulates a concrete, reproducible path to schema-agnostic pretraining over diverse databases and makes a credible zero-shot case relative to text-serialized LLMs and graph-centric foundations.
2. The empirical suite covers multiple tasks from RelBench, reports zero-shot vs continued pretraining vs fine-tuning, and includes context and attention ablations. The paper contrasts RT with LLM baselines and graph/tabular lines, and provides architectural ablations showing column attention’s disproportionate effect on zero-shot.

### Weaknesses
1. Missing compute, throughput, and memory comparisons vs. graph-centric models. The model uses sparse masks compiled to FlexAttention; some training details are given, but no throughput comparisons vs. Griffin/RelGT or cost-for-quality trade-offs are reported.
2. Ambiguity in the pretraining exposure conditions (Maybe/No/Yes) undermines interpretability of zero-shot tables. Tables 1–2 have a column “Target dataset ∈ pretraining? → Maybe / No / Yes,” but the meaning of “Maybe” is not clearly defined in the main text.

### Questions
1. Do you observe consistent gains as the number/diversity of pretraining databases grows? A simple scaling-law study over #datasets and steps would help calibrate the “foundation model” claim.
2. Beyond entity-level classification/regression, how would RT fare on link prediction or forecasting formulated without self-labels? Any preliminary results or blockers?

### Soundness
2

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
3

### Summary
This work proposes the Relational Transformer (RT), a RDB foundation model with zero-shot prediction capacity.  Facing the core challenge of heterogeneous schemas, graph structures, and functional dependencies, it leverages table metadata and task text definition to adapt to different input data and task. Moreover, it models RDB in cell level and passing messages between cells with relation attention. RT attains strong zero-shot and fine-tuning performance on RelBench.

### Strengths
1. Clear method illustration. Figure 1 shows the problem formulation and method, especially relation attention design, clearly. 
2. Strong empirical performance. It outperforms Griffin, a strong baseline. Moreover, to our best knowledge, it is the first RDB model with zero-shot capacity.

### Weaknesses
1. Code is not available.
2. Tables and Figure in Page 8 looks messy.
3. Ablation study in Table 3 should further include ablation of task description and schema data.

### Questions
1. RT relies on table metadata and task description, which are not always available in real-world cases. Can RT works in case that the meta data and task text description not available? 
2. RT is a cell-level model, where each cell is a token. Will it take more computation resource than row-level model Griffin?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes Relational Transformer, a pre-trained model for relational databases. The model treats each cell in a given relational database as a token for the transformer; the proposed transformer holds specialized attention mechanisms that adapts to specific traits of relational databases; the pre-training is carried out with self-supervised objective of masked token prediction. To show the effectiveness of the Relational Transformer, it showcases with RelBench, in which leave one-database out approach is implemented for pre-training and evaluation.

### Strengths
In general, the paper is well-written and easy to follow. The Relational Transformer attempts to build a pre-trained model for relational databases, which can be of non-trivial impact towards the foundation models for relational databases.

### Weaknesses
-	It would help to clarify the strength of the proposed Relational Transformer with some descriptions of the RelBench databases. This would include not only the basic statistics, but also the overlap of column names across the databases (possibly measuring the similarities of the llm embeddings), how the numerical values are distributed, etc., While diverse the RelBench maybe, I am uncertain as to how much the databases are curated so that they meet the standards to be included in the benchmark, and this may be in favor of having the zero-shot abilities for the Relational Transformer. Moreover, the characteristics of RelBench may give insights on ‘enabling large-scale pretraining’ as the paper claims for Relational Transformer.
-	It would be helpful to include examples that could highlight the importance of zero-shot learning on relational databases.
-	One of the possible extensions could be incorporating meta-data(base) information (possibly through analyzing the encoding steps).
-	While there could be some space constraints, it would be helpful to see a figure (possibly with an example from Figure 1) on how zero-shot prompting is conducted for understanding.

### Questions
-	What are some concrete examples on the usefulness of zero-shot abilities for relational databases?
-	In Algorithm 2, is there a reason for the specific order of different data types?
-	How curated are databases in RelBench?
-	How does the Relational Transformer perform with respect to the computation time?
-	Can Relational Transformer be used as a feature extractor (e.g., sentence transformer as in LLMs)?
-	What does it mean by the sentence ‘While task rows provide “in-context labels”, our setting is not few-shot as explicit subgraph-label pairs are not required.’? If the input of the prediction contains past labels, does this mean that the Relational Transformer calculates the attention between what to predict and the past labels (possibly through column attention?
- What is the reason behind the choice MiniLMv2 as the language model?

### Soundness
3

### Presentation
3

### Contribution
3
