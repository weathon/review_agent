# Lifelong-Learning Embeddings: Incremental and Continual Representation Learning for Dynamic E-Commerce Trends

- Decision: Reject
- Scores: 2, 2, 4, 4

## Abstract
E-commerce is a highly dynamic domain where products and consumer behaviors evolve rapidly. Embedding-based representations are central to deep learning–based personalization systems, yet conventional embeddings are static and therefore, they cannot easily incorporate new tokens (e.g., new products) without retraining, which is costly and often infeasible due to privacy or data retention constraints. To address this, we propose Lifelong-Learning Embeddings, a framework that (1) incrementally extends embeddings to integrate new tokens, (2) adapts embedding dimensionality to balance expressiveness and efficiency, and 
(3) employs continual learning to mitigate catastrophic forgetting. Experiments on a real-world dataset and two benchmark datasets show that our approach consistently outperforms static embeddings in accuracy while incurring only modest training-time overhead, demonstrating its effectiveness and adaptability in dynamic e-commerce environments.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper focuses on the challenge of learning embeddings in dynamic e-commerce environments, where products and consumer behaviors change rapidly. Traditional static embeddings are inadequate if they fully retrain the model to incorporate new tokens, which is computationally expensive and often infeasible due to privacy or data retention constraints. To tackle this, the authors propose LLE, a framework that incrementally updates embeddings to include new tokens, adapts embedding dimensionality for efficiency, and applies CL to mitigate catastrophic forgetting.

### Strengths
- The paper targets a practical and important challenge in e-commerce recommendation systems
- Overall, the paper is well written and easy to follow.

### Weaknesses
- The novelty of the paper is limited. The proposed embedding addition step and CL step in LLE are standard methods widely used in existing systems. The proposed idea of alignment is closely related to the dimensionality change in the second step, which is relatively uncommon in practical systems, but the proposed alignment loss is not new and the motivation of adaptive embedding sizes are not well justified.
- While the authors claim that their method is a drop-in replacement for existing embedding layers, but this is not true if they have dynamic embedding sizes because most neural models don’t support this.
- Experiments are conducted with simple baselines without considering many related works on cold-start recommendation and auto-regressive recommendation. In fact, there have been many studies in cold-start recommendation that can generate embeddings for new items (e.g., via LLMs) and train them together with the old ones.

### Questions
How does LLE perform if incorporated into a fully end-to-end training of a SOTA model in recommendation tasks?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This work studies a new embedding learning method for recommender systems called Lifelong-Learning Embeddings (LLE).
The key idea is that as new tokens (categorial featues) are introduced to the vocabulary at time step $t + 1$,
LLE gives a way to:
- retain previously learned embeddings for all tokens up to time $t$, and
- automatically resize embedding dimensions based on validation loss while minimizing information loss via continual learning (CL).

The authors apply LLE to 1 proprietary industirial dataset and 2 public benchmarks (YooChoose and RetailRocket).
The time scale of vocabulary shifts considered is 1-week, so Algorithm 1 is run
on each week of data and presented in Section 6. Overall, the approach and experiments are interesting and compelling.

### Strengths
- This work attackes a very important problem (i.e., good embedding learning methods for online recommender systems)
- Their method adapts the dimension of the embeddings over time.
- Self-contained layer designed to replace existing embedding layer without modifying downstream part of model too much
- Good choice of datasets and interesting experiments

### Weaknesses
- How do we deal with large-vocabulary features (e.g., cardinality = 10^9 id features)?
  * This work doesn't mention the hashing trick ("Feature hashing for large scale multitask learning" [Weinberger et al., ICML 2009])
  * How can LLE afford to keep collionless embeddings for tokens in a growing vocabulary? As written, this can blow out RAM.

### Questions
**Questions**
- [063] Are the "knowledge retention" and "compactness properties" contradictory?
  If we keep computational efficiency constant (e.g., the total bottom
  embedding dimension) and increase one feature's embedding dimension,
  then we necessarily reduce another feature's embedding dimension. How does
  LLE reconcile this? Line 170 mentions that CL is used. How does this work? It seems that we're forced to degrade model quality.
- If a new token arrives at each online example (e.g., a new search query),
  how do you solve the CL step (line 201) fast enough? It's not clear to me
  how this can scale up to an industry setting. It seems helpful to discuss how large a data iteration $D^(t)$ should be
  (e.g., a batch of 2048 events in real time or O(hour) data).
- Re Figure 2: What are the "Unknown tokens" in subplots (d--f)? Do
  they correspond to the "New Tokens" in plots (a--c)? If so, why aren't the
  first and second columns in plots (d--f) mostly orange? Is this due to the
  removal of the 99.5th percentile of interactions?
- In line 283, which transformer classifier / reference are you using?
- In Algorithm 1 Line 2, how do you decide to update
- How do you choose $\Delta d$ when validation loss overfits/underfits (line 311)?
- In Figure 6, the average embedding size of `Industrial` increases over time.
  This increases model resource cost. If we have a fixed budget on resource cost,
  what would happen in this case?
- Given that the alignment loss function in line 200 is generic, it would be nice to
  see experiments where everything is fixed but we sweep over different choices of $\ell_{\text{align}}$.

**Misc**
- [039] Suggestion: Consider adding the reference "Unified Embedding: Battle-tested feature representations for web-scale ML systems" [Coleman et al., NeurIPS 2023]
- [055] Typo: "?" --> missing reference
- [421] Suggestion: move the legend from figure 5(a) to a shared legend for all three figures.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the challenge that traditional and static embeddings in e-commerce cannot handle the rapid introduction of new products without costly retraining. The authors propose a framework called Lifelong-Learning Embeddings, which is designed to (1) incrementally add new tokens to the embedding table, (2) adapt the embedding dimensionality to balance performance and efficiency, and (3) use continual learning strategies to prevent ``catastrophic forgetting'' of previously learned information. Experiments on a real-world industrial dataset and two public benchmarks show that LLE outperforms static embeddings in accuracy, while only incurring a modest training-time overhead.

### Strengths
1. This paper studies the incremental learning in recommender systems, which is an interesting investigation with practical deployment consideration.
2. The experimental results demonstrate the effectiveness of the proposed methods over several baselines.

### Weaknesses
1. The primary concern with this paper is the insufficient motivation. Given the proposed setting of weekly data updates, this already constitutes a relatively low-frequency batch update strategy that is entirely feasible in industrial practice. More critically, the experimental results reveal a non-trivial gap between the proposed method and the upper bound. For AUC as the evaluation metric (the paper does not clearly specify whether this is a CTR prediction task), AUC is typically a ranking metric that tends to yield relatively high scores. Therefore, while the numerical differences may appear modest, they actually represent significant performance degradation in real-world e-commerce scenarios. Additionally, the proportion of new products introduced each week is inherently low, making full model retraining actually manageable for e-commerce platforms, which are extremely eager to maximize commercial profits and are willing to sacrifice computational costs (linear computational overhead is typically acceptable in such scenarios). Consequently, this incremental learning approach seems unnecessary for weekly update frequencies. In practice, asynchronous embedding table updates would be entirely viable at this cadence. In addition, the proposed method might be more compelling and well-motivated for daily update scenarios.
2. The paper presentation requires further polishing. Several issues need attention, including but not limited to: Lines 55, 181, and 212.

### Questions
1. Have the authors considered the model's performance under daily update frequencies?

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
4

### Summary
The authors propose a method for incrementally and continually updating embeddings in a dynamically changing e-commerce environment. The proposed approach consists of three modules. The first module maps tokens either to existing embeddings or to a new latent space. The second module handles changes in embedding dimensions by assigning values that include copies of the existing embeddings. The third module computes the final embedding values by aligning the semantics of overlapping tokens through contrastive learning. The authors evaluated their method on three datasets, including one private dataset, and achieved superior AUC performance compared to the baselines. Additionally, an ablation study was conducted to analyze the performance contributions of each component in the proposed method.

### Strengths
The authors proposed a very simple and intuitive approach. The proposed method can operate independently of both the training scheme and the embedding size, and offline experiments demonstrated superior performance compared to the baselines. The ablation study is also well-formulated and clearly structured.

### Weaknesses
The comparison criteria are too simplistic. For instance, in practical scenarios, the reason for using LLE might be that Baseline 2 in Section 5 is infeasible — likely due to the large amount of training data and high computational cost. If that’s the case, how would the results change if the training data were sampled to match LLE’s data size? It would also be meaningful to compare the proposed method with various cold-start techniques.

The embeddings for new tokens were initialized with either average or random values. Since product metadata or textual information were not utilized, this approach has limitations in addressing real-world cold-start problems.

Embedding quality was evaluated solely based on purchase prediction (AUC). The generalization ability of the model has not been validated through other downstream tasks such as recommendation, CTR prediction, or user similarity estimation.

### Questions
When lifelong learning continues, contrastive learning (CL) is applied to ensure that overlapping tokens retain consistent semantics in their embeddings.
However, one might question whether this approach limits potential improvement in embedding quality. Since CL primarily enforces consistency rather than optimization, embeddings for overlapping tokens may become resistant to beneficial updates that could better capture new contexts or evolving semantics.

### Soundness
3

### Presentation
3

### Contribution
2
