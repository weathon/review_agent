# Task Descriptors Help Transformers Learn Linear Models In-Context

- Decision: Accept (Poster)
- Scores: 6, 5, 6, 8

## Abstract
Large language models (LLMs) exhibit strong in-context learning (ICL) ability, which allows the model to make predictions on new examples based on the given prompt. Recently, a line of research (Von Oswald et al., 2023; Aky¨urek et al., 2023; Ahn et al., 2023; Mahankali et al., 2023; Zhang et al., 2024) considered ICL for a simple linear regression setting and showed that the forward pass of Transformers is simulating some variants of gradient descent (GD) algorithms on the in-context examples. In practice, the input prompt usually contains a task descriptor in addition to in-context examples. We investigate how the task description helps ICL in the linear regression setting. Consider a simple setting where the task descriptor describes the mean of input in linear regression. Our results show that gradient flow converges to a global minimum for a linear Transformer. At the global minimum, the Transformer learns to use the task descriptor effectively to improve its performance. Empirically, we verify our results by showing that the weights converge to the predicted global minimum and Transformers indeed perform better with task descriptors.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper studies in-context learning with task descriptions. It uses a synthetic setup of linear regression, where the means from which the inputs are samples serve as the task description. Theory shows that a single-layer linear attention model can reach the optimal solution and the optimal solutions are characterized. Experiments show that a single-layer can indeed reach the optimal solution, but deeper models still perform better.

### Strengths
- Important research question on the effect of task descriptions in ICL
- Simple synthetic setup to study 
- Theory showing optimality of a specific model class

### Weaknesses
- The biggest issue I have is whether the synthetic setup is a good proxy to study ICL, and specifically task descriptors. I understand similar synthetic setups were used to study ICL. But why is giving the mean a good task description? How does it simulate the natural language case, for example the one cited from Brown et al.? 
- It would be helpful to explicitly highlight the insights drawn from the various lemmas and theorems throughout section 4. Unfortunately I had a hard time following this part and so I indicate this is a low-confidence review. 
- The experiments are a nice edition to the theory, but I'm confused about the single-layer being inferior to the deeper models. 
- See other comments and questions below.

### Questions
1. It's quite puzzling to understanding the embedding matrix formulation of the problem. I'm used to seeing in-context examples as running examples in text, which are made of words that are embedded. But here, the embedding matrix itself is the object to learn. More specifically, given this embedding matrix, the objective is to learn the bottom-right entry, which contains the prediction y_query. What are the trainable parameters? Is E itself updated during training? How does this formulation relate to the usual ICL setup of running text? 

2. What are multiple tasks mentioned in equation 6 and does it make sense to take the expectation over all of them?  What is a task specifically?

3. What are the different sequences near equation 7? Are there multiple sequences per task? One sequence per task? 

4. Can you motivate the initialization in page 4? any clearer motivation besides having two matrices have the same norm?

5. I am possibly missing some background to understand this, but what do you mean by "We run gradient flow on the population loss"? How is the optimization done exactly? 

6. What is the significance of characterizing the optimal solutions in section 4? How should they be interpreted and what does it tell us about ICL with descriptions more broadly? 

7. If a single linear layer transformer can reach the optimal solution, then why do the experimental results show deeper models to perform better? Is it because of training difficulties with the single layer case? Is it just a sample complexity issue, with the single-layer model not having enough samples?

8. Since there's no pre-training and fine-tuning going on here, it seems like all instances of "pretraining" could just be changed to "training".

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper presents a theory analysis of the task descriptor usage in in-context learning (ICL)  for Transformer models.
The analysis is conducted based on a simple linear regression problem (i.e., given $x$ drawn from Gaussian $(\mu, \mathcal{I})$ predict $y$  = $w^Tx$ where $w$ is the underlying generative weights), and the task descriptors here are the $\mu$ for the input distribution.

Theoretically, the analysis results show that the provided task descriptors could help the Transformer directly use the $\mu$ provided in specialized input embedding given infinite samples $n$. Furthermore, the authors show that the model guaranteed to converge to the global optima from a a reasonable initialization. Empirical results show that the task descriptor could help a 1-layer Transformer achieve significantly lower loss, and the findings generalize to multi-layer Transformer model as well.

### Strengths
- Overall solid analysis (theory and empirical ) for the task descriptor usage in ICL of Transformer models.

### Weaknesses
- The analysis and results are overly narrow, focusing solely on linear regression. This limited scope makes it difficult to generalize the findings to fundamental tasks like sentiment classification with in-context learning (ICL).

- The paper's restricted scope can be partially attributed to defining the task descriptor as the mean $\mu$. This is a non-standard approach, as task descriptions in practical ICL applications typically convey semantic instructions (e.g., `Classify the review as positive or negative.`) rather than characterizing the input distribution $x$ (e.g., the lexical and syntactic patterns of review text).

- (Minor) Figure 2 reveals that the performance gap between models with and without task descriptors narrows as model complexity increases (from 1-layer to 2-layer architectures). This raises questions about the utility of task descriptors in modern large-scale models. The diminishing effect of task descriptors with increasing model size may limit the long-term relevance of this analysis, particularly given the trend toward ever-larger models.

### Questions
- Would the loss curve comparison change with larger models and complex data samples (higher dimension than 5)? 12-Layer Transformer such as GPT-2?

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
3

### Summary
This paper proposes a new mathematical setting for theoretical in-context learning analysis. It propose a setting which models the task description in in-context learning. The author discussed the optimal soluations for a 1-layer linear self-attention network in two situations, where samples are sufficient or limited, and prove that such optimal soluations can be achieved under gradient flow training with a reasonable initialization. Also, empirical verification for 1-layer transformers are also provided.

### Strengths
1.	The author models the task description part in in-context learning and provide comprehensive analysis.
2.	Both limited-sample and infinite-sample situations are considered. 
3.	Experiments further strengthened their analysis.

### Weaknesses
1.	The conclusions based on a 1-layer linear self-attention model and gradient-flow might be a little weak.
2.	Though introducing task description is very interesting, the techniques used and the theorems got might be a little incremental.

### Questions
1.	Might the author provide some insights/comparisons about what real difference the task description brings for ICL? It would be very helpful.
2.	Proofs of more complex situation than gradient flow / 1-layer LSA would be very helpful, especially for cases where global optimal can not achieved (which I guess is more often in reality?)

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper explores the role of task descriptors in enhancing the in-context learning (ICL) capabilities of Transformers. The authors show that incorporating task descriptors into the input prompt leads to improved ICL performance for linear models. They present theoretical findings indicating that gradient flow converges to a global minimum for simple linear models when task descriptors are used, and they empirically verify these results by demonstrating weight convergence to the predicted global minimum. The paper includes a theoretical analysis of how task descriptors facilitate in-context learning in linear regression settings, illustrates how Transformers can leverage these descriptors to enhance performance, and provides empirical validation of the theoretical findings through experiments with various Transformer configurations.

### Strengths
- The paper offers a novel perspective on how task descriptors can improve the ICL performance of Transformers.
- The paper presents a theoretical analysis of how task descriptors enhance in-context learning in linear regression settings, showing how Transformers can utilize these descriptors to boost performance. 
- It also offers empirical validation of the theoretical findings through experiments conducted with different Transformer configurations.

### Weaknesses
- While the paper provides a theoretical analysis, it may lack a deeper exploration into the underlying mechanisms of how Transformers integrate task descriptors during the learning process.
- The experiments, although comprehensive, are limited to a specific setup. More diverse datasets and problem settings could provide a more robust evaluation.

### Questions
- What are the specific mechanisms through which task descriptors improve ICL in Transformers? Further analysis or experiments that shed light on this process would be valuable.

### Soundness
3

### Presentation
2

### Contribution
3
