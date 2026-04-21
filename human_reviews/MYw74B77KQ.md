# NUDGE: Lightweight Non-Parametric Fine-Tuning of Embeddings for Retrieval

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 6, 5, 5

## Abstract
$k$-Nearest Neighbor search on dense vector embeddings ($k$-NN retrieval) from pre-trained embedding models is the predominant retrieval method for text and images, as well as Retrieval-Augmented Generation (RAG) pipelines. In practice, application developers often fine-tune the embeddings to improve their accuracy on the dataset and query workload in hand. Existing approaches either fine-tune the pre-trained model itself or, more efficiently, but at the cost of accuracy, train adaptor models to transform the output of the pre-trained model. We present NUDGE, a family of novel *non-parametric* embedding fine-tuning approaches that are significantly more accurate and efficient than both sets of existing approaches. NUDGE directly modifies the embeddings of data records to maximize the accuracy of $k$-NN retrieval. We present a thorough theoretical and experimental study of NUDGE's non-parametric approach. We show that even though the underlying problem is NP-Hard, constrained variations can be solved efficiently. These constraints additionally ensure that the changes to the embeddings are modest, avoiding large distortions to the semantics learned during pre-training. In experiments across five pre-trained models and nine standard text and image retrieval datasets, *NUDGE runs in minutes and often improves NDCG@10 by more than 10\% over existing fine-tuning methods. On average, NUDGE provides 3.3$\times$ and 4.3$\times$ higher increase in accuracy and runs 200$\times$ and 3$\times$ faster, respectively, over fine-tuning the pre-trained model and training adaptors.*

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper introduces NUDGE, a novel method for retrieval that directly fine-tunes index embeddings by treating them as "model parameters". The theoretical analysis yields two variants, NUDGE-M and -N, and elucidates their time complexities. Experimental results demonstrate that NUDGE significantly outperforms existing methods for in-distribution data and achieves comparable performance for out-of-distribution scenarios.

### Strengths
[S1] The proposed method is novel, simple, and easy to implement.

[S2] The experiments effectively demonstrate the efficacy of NUDGE for in-distribution data.

[S3] The theoretical analysis clearly delineates the characteristics of NUDGE-N and NUDGE-M.

[S4] The paper is well-structured and easy to follow.

[S5] NUDGE represents a potential new paradigm for fine-tuning.

[S6] The authors have provided the code with their submission.

### Weaknesses
[W1] Fine-tuning the embedding directly makes the database difficult to update. Specifically, if new photos are added to the database, their embeddings must also be trained. This requires relabeling (or pseudo labeling) at least a few queries whose GT answers include the new DB images, which incurs additional cost (or the new DB embedding will not be updated). This weakness, however, does not appear in previous two lines of works, ie. fine-tuning the foundation model, or training a new lightweight model to refine the embeddings. Consequently, this limitation somewhat impacts the practicality of NUDGE.



[W2] NUDGE is unable to refine query embeddings. It has been well established that query embedding refinement can significantly enhance retrieval performance (as evidenced by query expansion and its variants). Both fine-tuning the foundation model and training an embedding update model are capable of updating query embeddings. However, NUDGE cannot accomplish this because the GT label for the input query is unknown, making it impossible to train. This limitation leaves considerable room for improvement.



[W3] The generalizability of the proposed method is questionable. Intuitively, embedding update methods are likely to suffer from significant overfitting issues. The authors have invested considerable effort to address this, including constraining the $\Delta$ and normalizing the updated embeddings (either pre- or post-update). While NUDGE-N ultimately achieves acceptable results for OOD data, overall confidence in its generalizability remains limited. For OOD scenarios, NUDGE-N shows only minor improvements on the BGE-S benchmark, with many other benchmarks left untested. When the embedding is not normalized (i.e., NUDGE-M), the generalizability appears notably poor. While it's plausible that the transition from NUDGE-M to NUDGE-N could yield such significant improvements, additional experiments elaborating on generalizability would be beneficial, as this might constitute a major drawback of NUDGE. Nevertheless, if NUDGE can perform adequately across most OOD benchmarks, or even in its current form, it may still be considered acceptable given the other strengths of this work.



[W4] Nit: The authors should follow the ICLR 2025 template guidelines by placing table captions above the tables.

### Questions
[Q1] It seems appropriate to position NUDGE as the third paradigm for retrieval task fine-tuning. The three paradigms can be summarized as follows: 1) Fine-tuning the foundation model; 2) Training a tiny model to refine the embedding; 3) directly fine-tuning the embedding. These approaches present interesting trade-offs. In terms of training cost, the order is 1 > 2 > 3, while for cost of adding more photos to the DB, the order is reversed: 3 > 2 > 1. Fine-tuning a foundation model is computationally intensive, but it easily accommodates new images added to the database. NUDGE, on the other hand, is less demanding to train, but requires additional computation to update embeddings for newly added DB images. In this context, the second paradigm appears to offer the best balance among the three. Do you have any insights on this?



[Q2] How is the generalizability of other NUDGE variants?

 

Justification of my score:

Although concerns about generalizability and practicality have been raised, the overall contributions of NUDGE—such as its efficient training, strong performance, and introduction of a new fine-tuning paradigm—significantly outweigh its weaknesses. Based on these considerations, this paper merits acceptance. However, given the identified limitations, I look forward to discussing with the authors and other reviewers to further evaluate its merits. At present, I believe NUDGE represents a valuable and striking contribution to the retrieval community. I hope the authors can address my concerns or, at the very least, discuss them as limitations of this work in future revisions.

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
The context is embedding-based retrieval, for Q&A on documents, RAG, text-to-image retrieval, etc. The authors propose to do domain adaptation by "fine-tuning" the document embeddings with a supervision. The results show that this is more accurate (and faster) than to fine-tune the feature extraction model or to train a transformation model on top of frozen features.

### Strengths
S1. The study is an interesting exploration of dataset side adaptation, in the case of a "closed world" dataset that is frozen. 

S2. The approach and baselines are relevant. The adaptation relies on simple linear algebra, so it is fast and much can be done in closed form.

S3. Convincing results: since the training acts directly on the embeddings (as opposed to network parameters) it has more degrees of freedom than when training on the model (indirect fine-tuning)

### Weaknesses
W1. The "elephant in the room" for this method is that it does not work on out-of-sample documents. Since the embeddings of the database are adjusted at fine-tuning time, new documents cannot benefit from it, unless new training data is added for them as well, which becomes a really narrow application setting. This is all the more apparent since out-of-distribution on the queries side is considered in this work.

W2. The method works only on densely annotated training data, since a document that is never returned to a query will not be adapted. This is apparent in the KILT dataset where documents that do not appear in any result are simply removed. 

W3. Because of this particular setting, the queries in experiments must be split in train+val+test, which is unusual. Therefore there are no comparisons with state-of-the-art methods.

### Questions
I would be willing to raise my rating if the "closed-word" setting of the paper was clearly discussed and mentioned as a limitation compared to the baselines it compares with. It is not even sure it's legitimate to call the method "fine-tuning". However, it *is* an interesting study of what can happen if embeddings are optimized directly (it's just not practical beyond a benchmark setting). 

For out-of-sample results, it would be interesting to see what would happen if an actual feature transformation model was trained on top of the embeddings found by this method.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper introduces NUDGE, a non-parametric embedding fine-tuning approach designed to enhance k-nearest neighbor (k-NN) retrieval accuracy in text and image retrieval tasks. Two main variants, NUDGE-M and NUDGE-N, apply controlled modifications to embeddings, with NUDGE-M using bounded changes and NUDGE-N adding normalization to improve out-of-distribution generalization. Experimental results across multiple datasets show that NUDGE consistently outperforms traditional PTFT and Adapter methods.

### Strengths
1. **Intuition of the Proposed Method**: The paper clearly presents NUDGE's core idea of directly modifying embeddings as parameters, offering a simple, intuitive approach to fine-tuning that improves retrieval accuracy without altering the model.

2. **Extensive Experimental Evaluation**: Experiments are comprehensive, covering diverse retrieval tasks (text, cross-modal) and configurations, validating NUDGE's effectiveness, efficiency, and adaptability across different datasets and models.

### Weaknesses
1. **Theoretical Issues**:
   - Equation (1), defining MaxA-EFT, is disconnected from the actual task as it primarily maximizes inner product similarity without a clear link to retrieval-specific objectives.
   - The NP-hard nature of MaxA-EFT, while noted, lacks clear relevance to the search task, and the authors fail to define an explicit optimization problem for solving it.
   - Equation (2) being unbounded is more of a general mathematical property and seems unrelated to the paper’s focus on non-parametric methods.
   - Line 212’s constraint on \(\Delta\) uses matrix norm notation, which doesn’t adequately capture the constraints on embedding norms, indicating a need for clearer notation.

2. **Experimental Issues**:
   - Table 3’s results are overly generalized under “across text datasets,” obscuring which datasets contributed to the reported metrics.
   - In BEIR benchmark results (Table 5 and Table 9), the first column shows identical performance across all methods, which is counterintuitive and suggests either data or methodology issues.
   - The GPU speedup in Table 6 is marginal compared to the CPU, which raises efficiency concerns, as an A100 GPU should provide significantly higher speedup for such operations relative to CPU processing.

### Questions
As the modifications of the embeddings still require learning and would be sufficiently large given a large dataset, how much benefit does this method still have when the dataset is large?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This work presented a family of novel non-parametric embedding fine-tuning approaches that are significantly more accurate and efficient than both sets of existing approaches.

### Strengths
1.The s NUDGE, a non-parametric embedding fine-tuning method, which innovatively addresses the limitations of parametric approaches in k-Nearest Neighbor (k-NN) retrieval. This originality stems from shifting focus to the embeddings themselves rather than model parameters, offering a novel optimization perspective for retrieval tasks. 

2.Comprehensive experimental and theoretical analyses validate NUDGE's efficacy across five pre-trained models and nine retrieval datasets.

3.The paper is well-organized and presents its methodology and experimental findings clearly.

4.By providing a lightweight, model-agnostic solution with strong generalization to out-of-distribution queries, NUDGE offers substantial improvements in both computational efficiency and accuracy.

### Weaknesses
1. It does not extensively explore or compare alternative constraint types. For example, additional regularization techniques, such as sparsity constraints or adaptive thresholds.

2. Lack a detailed ablation study that examines how skewed distributions impact both in-distribution and out-of-distribution performance could reveal if other normalization techniques or dataset-specific adjustments would better manage this overfitting.

3. While the optimization and constraint methods are described, some of the choices behind the approach could use further explanation.

### Questions
1. What is the sensitivity of NUDGE's performance to hyperparameter selection, particularly the choice of γ. If 'NUDGEM sets γ by solving MaxA-EFT on the validation set', how should γ be set on the training set?

2. In Eq.1, the range of j is confusing, what does '\Y_j' represent?

3. Why do NUDGE-M and NUDGE-N methods have similar average results on image retrieval pre-trained models. （CLIP-B and CLIP-L）

4. The criteria for selecting the constraint boundaries in NUDGE-M and the specific reasoning behind the grid search implementation for NUDGE-N's gamma optimization are not fully detailed.

5. How to mitigate the impact of unreliable training query embeddings when fine-tuning based on these embeddings?

### Soundness
3

### Presentation
3

### Contribution
3
