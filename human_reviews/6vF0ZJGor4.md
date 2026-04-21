# ImplicitSLIM and How it Improves Embedding-based Collaborative Filtering

- Avg Score: 5.00
- Decision: Accept (poster)
- Scores: 6, 8, 3, 3

## Abstract
We present ImplicitSLIM, a novel unsupervised learning approach for sparse high-dimensional data, with applications to collaborative filtering. Sparse linear methods (SLIM) and their variations show outstanding performance, but they are memory-intensive and hard to scale. ImplicitSLIM improves embedding-based models by extracting embeddings from SLIM-like models in a computationally cheap and memory-efficient way, without explicit learning of heavy SLIM-like models. We show that ImplicitSLIM improves performance and speeds up convergence for both state of the art and classical collaborative filtering methods. The source code for ImplicitSLIM, related models, and applications is available at https://github.com/ilya-shenbin/ImplicitSLIM.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper focuses on an influencial autoencoder-based collaborative filtering model SLIM and draws inspiration from its optimization objective and correspondingly proposes ImplicitSLIM. ImplicitSLIM combines the advantages of SLIM's objective function and Locally Linear Embeddings. ImplicitSLIM also introduces a novel regularization item and an initialization method for embedding-based collaborative filtering models. Experimental results are presented to demonstrate the effectiveness of the proposed model.

### Strengths
+ The authors propose a novel and general method that enhances performance of embedding-based CF models, which has practical significance.
+ The theoretical analysis of existing works on autoencoder-based models is persuasive and easy to follow. The paper provides a clear and insightful review of existing methods from an optimization perspective.
+ ImplicitSLIM is well-motivated and presents a novel solution for embedding-based CF.
+ The paper is generally well-written, ensuring readability and clarity.

### Weaknesses
- The paper introduces various settings for ImplicitSLIM, but it would be beneficial to analyze and summarize the computational complexity of different variants respectively to provide a clearer understanding of their efficiency.
- The paper could benefit from more detailed experimental studies on the influence of hyperparameters. Given the presence of multiple regularization and optimization items in the method, it would be more illustrative to have the model performance w.r.t. different parameter settings.
- Although ImplicitSLIM has shown competitive performance against traditional linear encoders, it shows limitted improvement on deeper models like UltraGCN and RecVAE.

### Questions
Please refer to the weaknesses.

### Soundness
4 excellent

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper introduces ImplicitSLIM, an approach leveraging insights from established linear models such as EASE and LLE. The authors advocate its use for initializing and regularizing item and user embeddings across various collaborative filtering architectures. ImplicitSLIM streamlines the process of extracting embeddings, exhibits robust generalization capabilities, and accelerates convergence of the downstream models. As a comparatively lightweight and effective solution, it has the potential to become a valuable tool in representation learning for collaborative filtering, contributing to both theoretical understanding and practical implementation. 

The text is well-written and easy to follow. The presentation of the approach is transparent and mathematically sound. The obtained results are convincing and demonstrate the advantages of the proposed solution. I'd vote to accept the paper.

### Strengths
- mathematically sound approach with a closed-form solution
- showcases practical efficiency in the standard collaborative filtering task
- good generalization capabilities

### Weaknesses
- not a standalone approach which makes training less straightforward
- applicable to embedding-based models only
- the source code is not provided

### Questions
There's a promise to provide empirical comparison with more natural-looking regularizer in Appendix E.2. However, not much information is provided there. The promise creates an expectation that there will be a more substantial comparative data with numbers and graphs. Is it planned to be provided?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
1: You are unable to assess this paper and have alerted the ACs to seek an opinion from different reviewers.

### Summary
This paper proposes a novel unsupervised learning approach for sparse high-dimensional data. The method learns local structure of data (embeddings) in the embedding space where the embeddings of similar objects to be similar.

### Strengths
This paper learns embeddings with closed form solutions.
Good Experimental study.

### Weaknesses
- This paper is very theoretical and hard to follow the formulas.

### Questions
This paper is too theoretical than any other submissions on NeurIPS and ICLR. I hardly follow the content.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposed a method named implictSLIM that can be integrated with other embedding-based methods.

### Strengths
Strengths:

-	The approaches addresses the memory-intensive and scalability issues of SLIM-like models in collaborative filtering.
-	The authors provided various experiments on publicly available benchmark datasets 
-	Source code is also provided 
-	Many appendices were given for more explanation

### Weaknesses
Weaknesses:

-	In Section 3 Proposed Approach, the authors should explain more on the formular choices when developing ImplicitSLIM. For example, why do we use LLE, but use the neighbourhood of NN(i) = {1,2,…,I} \ {i} to make it ‘global’ (Section 3.1)? Why do we drop the sum-to-one constraint (Section 3.2) (the authors did mention they have no good reasons, but why)? In my opinion, Section 3 is very important and the authors should provide deeper explanations and discussions about this. Otherwise it’s very hard to convince the readers.
-	In the first paragraph after Figure 2, the authors mentioned that “In addition, ALS applied to MF regularized by …. about 5x faster”, I may missed it but where do we find the ‘5x faster’ comparison? 
-	In Figure 1, why ImplicitSLIM init + SLIM reg and SLIM-LLE init cannot have results with > 500 embedding dimension? Could we please add an appendix on the ‘high computational costs’?
-	In Section 4.1, in the last sentence, the authors mentioned that “this procedure may be less stable … fewer calls to ImplicitSLIM”. Why is that?
-	In Table 1, the performance results are not really impressive. For example, in the Appendix E.1, the authors mentioned that “Moreover, ImplicitSLIM is also faster than EASE… could replace EASE in some cases due to lower computational time and comparable performance.”. Please explain which cases? 
-	In Appendix E.3, Table 5, why RecVAE + ImplicitSLIM cannot perform better than RecVAE? 
-	References should be sorted as the current version is hard to follow (minor)

Overall, more works need to be done.

### Questions
Please refer to the above comments.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
