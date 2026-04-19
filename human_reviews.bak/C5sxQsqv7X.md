# SELECTFORMER: PRIVATE AND PRACTICAL DATA SELECTION FOR TRANSFORMERS

- Decision: Reject
- Scores: 8, 5, 5, 6, 3

## Abstract
Critical to a free data market is $\textit{private data selection}$, i.e. the model owner selects and then appraises training data from the data owner before both parties commit to a transaction. To keep the data and model private, this process shall evaluate the target model to be trained over Multi-Party Computation (MPC). While prior work suggests that evaluating Transformer-based models over MPC is prohibitively expensive, this paper makes it practical for the purpose of data selection. Our contributions are three: (1) a new pipeline for private data selection over MPC; (2) emulating high-dimensional nonlinear operators with low-dimension MLPs, which are trained on a small sample of the data of interest; (3) scheduling MPC in a parallel, multiphase fashion. We evaluate our method on diverse Transformer models and NLP/CV benchmarks. Compared to directly evaluating the target model over MPC, our method reduces the delay from thousands of hours to tens of hours, while only seeing around 0.20% accuracy degradation from training with the selected data.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
In the free data market, model owner would like to trade data from data owner to be able to maximize the model accuracy, which requires to select and appraise portions of data points. However, the data selection should be private to both model and data owner in order to keep model parameters and data points private. A technique named MPC can be utilized to jointly evaluate the model privately with forward passes while privacy of both parties could be preserved.  Existing MPC approaches could approximate expensive nonlinearity with cheaper operations for transformer inference, which transformer models are applied in the most deep learning tasks. Meanwhile, they suffers from considerable runtime overhead and poor selection utility. In this work, authors propose an approach to accelerate MPC-based private data selection while preserve the utility of the selection.

### Strengths
1. The assumption for the unlabeled and probably imbalanced dataset D from data owner is a practical consideration, which makes the work more convincing and challenging than prior related works.
2. Multiphase selection is a smart strategy. Instead of selecting appropriate data at once. Filtering out most irrelevant data points in the early stage is a way more efficient approach with smaller models and larger models in the later phases provide accurate selection based on filtered data.
3. Approximation of nonlinear operations with multiple shallow MLPs can simulate the utility of nonlinearity well and reduce high dimensions into much lower dimensions to provide huge efficiency improvement.

### Weaknesses
1. While a semi-honest setting is convenient and efficient for the research purpose, a setting against malicious adversary is more practical. The assumption based on faithful protocol execution by both parties is not quite strong in the real-world application.
2. The reveal of target model architecture could be dangerous. For example, an adversary can potentially retrieve parameters of certain layer. Instead, I think the target model could be trained with MPC protocols.

### Questions
1. Since the setting of model and data owner is based on two parties. Instead of MPC, have you considered to solve this problem in the zero-knowledge proof setting? ZK approach allows model owner to select and appraise the data from data owner by privately committing parameters and data, and providing verifiable results without data leakage. It may be more efficient than MPC (not a conclusion and probably interesting for the future investigation).
2. In the threat model of Section 2.1, can you explain more about what can be revealed (like the purchase budget B mentioned in the Section 4.1) and what should be private (like model parameters and data points)? Or also merge the privacy guarantee in the Section 4.1 into the threat model.
3. What is the threshold of phase i? How should we determine when the early or later phases are?

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
1. The paper presents a technique called SELECTFORMER for private and practical data selection for Transformers. 
2. The contributions of the paper are threefold: (1) a new pipeline for private data selection over MPC, (2) the use of low-dimensional MLPs to emulate high-dimensional nonlinear operators, and (3) a parallel, multiphase scheduling approach for MPC. 
2. The goal is to enable the model owner to select and appraise training data from the data owner privately before committing to a transaction. 
3. The technique utilizes Multi-Party Computation (MPC) to evaluate the target model over the selected data.

### Strengths
1. The research problem is unique and import for practical use.

2. The technique is evaluated on various Transformer models and NLP/CV benchmarks, showing a significant reduction in delay from thousands of hours to tens of hours, with only a minimal accuracy degradation of around 0.20% compared to training with the selected data.

3. The selection process is designed as a multipass sieve, where earlier phases use smaller selector models for quick filtering of redundancy, and later phases use larger selector models for more precise selection.

### Weaknesses
1. The first contribution-emulates high-dimensional nonlinearity with low-dimensional MLPs. This is a not novel technique. Besides, training MLPs to approximate every non-linear operator is very cost. Does author evaluate the time cost here?
2. The MPC protocol hide the computation behind the communication data exchange is not novel. Any MPC protocol would try to minimize the total latency like this way. It is not unique here. Thus, I think this contribution is a weak statement.


Reference:

[1] Lu, Lu, et al. "Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators." Nature machine intelligence 3.3 (2021): 218-229.

### Questions
See Weakness part.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This work proposed an MPC-based private data selection framework for large Transformer models. The main technical contribution of MPC is replacing high-dimensional nonlinearity with low-dimensional MLPs. Besides, a batch evaluation is used in MPC.

### Strengths
An MPC-based private data selection framework for large Transformer models.
+ Nonlinearity evaluation with low-dimensional MLPs.
+ Multi-phase selection
+ Parallel MPC executions

### Weaknesses
- This work seems to simply combine the techniques of data selection and secure inference on LLMs.
- Replacing high-dimensional nonlinearity with low-dimensional MLPs seems less general.
- The batch evaluation is a widely used method in PPML and lacks novelty.

### Questions
1. Is this non-linear evaluation method only applicable in the proposed data selection setting? Can it be extended to general MPC-based LLM?
2. What are the differences between MPC-based data selection on LLMs and secure inference on LLMs?
3. Why do the authors focus on LLMs? Can this method be applied to CNNs?
4. Does this work use a third party for generating correlated randomness for MPC similar to MPCFormer?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
In this paper, the authors proposed a method to approximate transformer models for training using MLP instead of non-linear functions such as Softmax and layernorm.

### Strengths
It seems that the performance of the Softmax function is important in transformers when MPC or FHE is considered. The precision of Softmax approximation has a very large impact on the overall inference performance. In this paper, they suggested to use MLP instead of them.

### Weaknesses
The core ideas proposed in the paper are described on pages 4 and 5, but the description in this part is somewhat unclear. In particular, it is unclear whether data transfer between proxy models occurs using MPC in the multi-phase selection section, or not. This part needs to be restated more clearly.

### Questions
In the multipass selection phase, it said, "The forward pass computes the prediction entropy values, which are encrypted." The question is, what kind of encryption scheme is used? Is it a homomorphic encryption? How can you generate secret shares from the encrypted entropy values for the input of the following proxy model?

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 5

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper explores the efficient acquisition of data necessary for training artificial intelligence models using multi-party computation techniques. To effectively train models at a fixed cost when acquiring data from data owners, it is necessary to assess the quality of the data. This paper presents a method for data appraisal, allowing model owners to select data that is advantageous to them without accessing the data themselves.

### Strengths
The approach of efficiently purchasing data for training artificial intelligence systems within a fixed budget is a novel research direction. This seems to be a necessary research topic not only for AI security but also for various AI training scenarios. The paper successfully persuades the need for the research direction and research topic.

### Weaknesses
It seems to lack a clear and precise explanation of the technical aspects of the paper. All the technical details regarding the research method are quite ambiguous, making it difficult to understand the core ideas of the paper. While the paper mentions proposing data selection and appraisal methods when training artificial intelligence models using MPC, it does not precisely explain how these techniques are related to MPC protocols and security. Additionally, it doesn't provide a clear explanation of why each specific technique is necessary and what problems they aim to solve, making it hard to grasp the reasons behind the use of these techniques.

Furthermore, the paper does not offer a clear protocol or source code to understand how each operation is exactly performed. The technical aspects of the paper remain ambiguous, as there is no clear protocol or algorithmic explanation. (I believe that if the protocols and algorithms are given clearly, code submission itself can be considered optional.) The paper lacks explanations for the figures, making it challenging to understand their meanings. 

Therefore, I think this paper is not yet ready for presentation at a conference, and a complete rewrite is necessary to clarify each algorithm and protocol and make it more reader-friendly.

### Questions
Some specific areas of ambiguity include:

1. How each operation is computed using MPC.
2. How entropy is calculated if MPC is used.
3. How the index with the highest entropy is jointly found.
4. The definition and details of the QuickSelect algorithm.
5. The interpretation of Figures 1 and 2.
6. The role and precise definition of the proxy model.
7. The dimensions and hyperparameters used for the MLP that replace the nonlinear function.
8. The data used for proxy model training.
...

There are many other unclear aspects that need to be clarified to give the paper more value.

### Soundness
2 fair

### Presentation
1 poor

### Contribution
2 fair
