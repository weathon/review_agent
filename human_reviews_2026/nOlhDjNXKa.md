# GNN-as-Judge: Unleashing the Power of LLMs for Graph Learning with GNN Feedback

- Decision: Accept (Poster)
- Scores: 8, 6, 4, 2

## Abstract
Large Language Models (LLMs) have shown strong performance on text-attributed graphs (TAGs) due to their superior semantic understanding ability on textual node features. However, their effectiveness as predictors in the low-resource setting, where labeled nodes are severely limited and scarce, remains constrained since fine-tuning LLMs usually requires sufficient labeled data, especially when the TAG shows complex structural patterns. In essence, this paper targets two key challenges: (i) the difficulty of generating and selecting reliable pseudo labels on TAGs for LLMs, and (ii) the need to mitigate potential label noise when fine-tuning LLMs with pseudo labels. To counter the challenges, we propose a new framework, GNN-as-Judge, which can unleash the power of LLMs for few-shot semi-supervised learning on TAGs by incorporating the structural inductive bias of Graph Neural Networks (GNNs). Specifically, GNN-as-Judge introduces a collaborative pseudo-labeling strategy that first identifies the most influenced unlabeled nodes from labeled nodes, then exploits both the agreement and disagreement patterns between LLMs and GNNs to generate reliable labels. Furthermore, we develop a weakly-supervised LLM fine-tuning algorithm that can distill the knowledge from informative pseudo labels while mitigating the potential label noise. Experiments on multiple TAG datasets demonstrate that GNN-as-Judge significantly outperforms existing methods, particularly in low-resource regimes where labeled data are scarce.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces GNN-as-Judge, a novel framework designed to enhance few-shot semi-supervised learning on TAGs by leveraging the complementary strengths of LLMs and GNNs.The key idea is to use the GNN as a structural judge to guide pseudo-label generation and selection for the LLM. The framework first identifies structurally influential unlabeled nodes based on graph topology, then divides them into agreement and disagreement sets according to the consistency between LLM and GNN predictions.Reliable pseudo-labels from the agreement set are used for instruction tuning, while informative but uncertain examples from the disagreement set are used for preference tuning, with GNN confidence guiding label selection.The authors further propose a weakly-supervised fine-tuning objective that integrates both instruction and preference tuning.Extensive experiments on multiple TAG datasets show that GNN-as-Judge consistently outperforms state-of-the-art baselines, particularly in low-resource settings.

### Strengths
Originality: The paper proposes a novel GNN-as-Judge framework, enabling collaborative pseudo-label generation on text-attributed graphs. This cross-model feedback design is original and bridges two previously distinct paradigms—graph representation learning and language modeling.
Quality: The methodology is technically sound and well-structured. The framework combines influence-guided node selection, agreement/disagreement-based pseudo-labeling, and weakly-supervised fine-tuning with preference optimization. Each module is motivated, formalized, and validated experimentally, showing consistent improvements over strong baselines under few-shot settings.  
Clarity: The paper is clearly written, with a logical presentation of concepts and a well-organized methodology. Key formulations and algorithmic steps are explained with sufficient detail and supported by intuitive illustrations, making the overall approach accessible to both graph learning and NLP researchers.  
Significance: The proposed approach substantially enhances the usability of LLMs for graph-based semi-supervised learning, particularly in low-resource scenarios. By effectively leveraging GNN feedback to improve pseudo-label reliability, the method has strong potential to influence future research on multimodal graph-text learning and weak supervision paradigms.

### Weaknesses
1.No analysis of preference score threshold τ: The sensitivity analysis only shows results for different τ values but doesn't explain why τ=0.7 was chosen or analyze the disagreement set size vs. quality trade-off  
2.Training time analysis is incomplete: Figure 5 shows total training time but doesn't break down where the time is spent (influence computation, GNN training, LLM inference, LLM fine-tuning)  
3.No failure case analysis: The paper doesn't discuss scenarios where GNN-as-Judge might fail or perform poorly

### Questions
1.Hyperparameter selection without validation set: In true few-shot scenarios, even validation labels might be scarce. How would you recommend setting K, τ, and λ when validation performance is unreliable?  
2.Disagreement set contribution: From Table 5, the disagreement accuracy varies dramatically across datasets (31.55% for Pubmed vs. 66.56% for Citeseer). How does this variation affect the final performance? Is there a correlation between disagreement accuracy and overall improvement?  
3.What happens with different GNN architectures? You use GCN as the default GNN and briefly mention H2GCN for heterophilic graphs. How sensitive is the method to the choice of GNN architecture (e.g., GAT, GraphSAGE, GIN)?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper addresses the challenge of generating reliable pseudo labels in a semi-supervised node classification setting where labeled data are scarce. Pseudo-labeling in such cases faces two key issues: (1) selecting which unlabeled nodes should receive labels, and (2) mitigating noise in the generated pseudo labels. To address these, the authors propose an influence-guided node selection strategy, where each unlabeled node’s representation influence from labeled nodes is quantified, and only those with high influence scores are chosen for pseudo labeling. Among these selected nodes, the agreement set nodes where both the GNN and the LLM agree on the label is used as high-confidence pseudo nodes. For the disagreement set, only nodes with a preference score exceeding a predefined threshold are retained. The authors then fine-tune the LLM using a unified objective that combines instruction tuning on the agreement set and preference tuning on the filtered disagreement set. Experimental results on multiple datasets show the proposed method can outperform the recent baselines.

### Strengths
- The paper clearly articulates the challenges of applying LLMs to few-shot semi-supervised learning on text-attributed graphs, particularly the difficulties in generating reliable pseudo-labels and mitigating label noise.
- The idea of using a GNN as a judge to guide pseudo-labeling for an LLM is conceptually interesting
- The idea of selecting nodes from the disagreement set is interesting, as these represent challenging (hard) examples where the two models diverge. This is more sophisticated than simply selecting high-confidence (easy) pseudo-labels
- The influence-guided node selection based on graph structure is well-motivated. Theorem 1 provides a computationally tractable upper bound for node influence which identifies unlabeled nodes most strongly affected by labeled data.
- Theorem 2 formally justifies why agreement sets yield higher-quality pseudo labels 

- The paper includes extensive experimental detail with existing models and datasets, cross dataset experiments, and strong ablation study

### Weaknesses
- The influence metric (Eq. 1) formalizes how labeled nodes affect unlabeled ones through message passing, and high-influence nodes are prioritized for pseudo labeling.  Selecting only high-influence nodes (those strongly affected by labeled data) introduces bias: it oversamples regions near labeled nodes and ignores distant or boundary regions that may be crucial for generalization. The paper provides no analysis of how it affects underrepresented classes. 

- The framework's central premise is that GNNs provide more reliable predictions than LLMs in disagreement cases. However, Table 5 shows that GNN predictions in the disagreement set are correct only 31.55% of the time for Pubmed, 37.38% for ogbn-arxiv, and 37.24% for ogbn-products. This creates a logical gap: if the GNN is frequently wrong in disagreement cases (which is why there's disagreement in the first place), then using its predictions to train the LLM could inject substantial noise and potentially degrade performance. Moreover, the paper lacks an analysis to quantify how often: (1) GNN is correct and LLM wrong, (2) LLM is correct and GNN wrong (harmful preference), or (3) both are wrong.

- Additionally, in the preference tuning framework, when GNN and LLM disagree, the LLM's prediction is automatically assigned as the "dispreferred" response. However, there is no validation that the LLM is actually wrong in these cases. The dispreferred label could be the correct answer. This creates a scenario where the framework actively penalizes the LLM for making correct predictions.

### Questions
1. Theorem 2 assumes independence between GNN and LLM errors. Have you measured the actual correlation or overlap between their error patterns to validate this assumption?

2. Table 5 shows that GNN accuracy in disagreement cases is low. How do you justify treating the GNN as the “teacher” in those instances?

3. Do certain classes benefit more from influence-guided selection or from preference tuning? How does your method perform when the labeled set is imbalanced?

### Soundness
3

### Presentation
4

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
The core of this paper is to use GNN's ability to understand structures to fine-tune LLM by selecting more reliable pseudo labels. Meanwhile, nodes that do not have particularly reliable labels but show certain preferences are used as soft labels for weakly supervised learning.

### Strengths
S1- The author uses a lot of theoretical analysis to prove the rationality of the method.

S2- The experimental results are excellent on multiple datasets.

S3- The technical route is reasonable and there is no technical loophole in the method itself.

S4- This paper is presented at a very high standard of presentation.

### Weaknesses
W1- Researchers have practiced using large language models to label GNN or LLM. The author of this article has elevated the method to a very high concept, namely GNN as a referee, because it can understand structural information, but its essence is still a common process of: 1. select some candidate nodes to query the large model; 2. After the inquiry, further screen for pseudo labels by comparing the predicted results of GNN and LLM; 3. The pseudo labels with high confidence are used as strong supervision signals, while those with relatively low confidence are used as weak supervision signals.  The method in this paper does not involve significant innovation in paradigm, but rather represents a better technological implementation of the three processes mentioned above.

W2- Many of the technical details are referenced but not explained in detail throughout the article. Related work also has the same problem, such as citing the literature but not specifically explaining the core contributions of these methods, resulting in the inability to quickly grasp the core differences between the proposed methods and previous methods after reading.

W3- The paper uses too many symbols to show the content, in fact, I read the process, not careful analysis of definitions, theorems, etc. , can also know what the author's technical route, so some theoretical analysis may be redundant.

### Questions
Q1- Could the author explain in simple terms how the set of disagreement is defined? That is, how is the disagreement set obtained?

Q2- Can the author simplify the algorithm flowchart? I think the entire technical route may not require so many symbols to illustrate.

Q3- What is the citepseer dataset? It seems to be a typo.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper studies few-shot semi-supervised node classification on text-attributed graphs (TAGs). It proposes GNN-as-Judge, a framework that combines a structure-aware GNN and a text-centric LLM to curate high-quality pseudo-labels and mitigate label noise. The method first selects a subset of unlabeled nodes using an influence-guided criterion, then exploits LLM–GNN agreement and disagreement with GNN preference scores to construct pseudo-labeled data. The LLM is fine-tuned with a weakly supervised objective that integrates instruction tuning on the agreement set and preference tuning on the disagreement set. Experiments on multiple benchmarks show consistent gains, especially under low-resource regimes.

### Strengths
Strengths:
1) GNN-as-Judge paradigm: Positioning the GNN as a judge complements the LLM’s lack of structural inductive bias, yielding more reliable pseudo-labels on TAGs.
2) Collaborative pseudo-labeling: Using both agreement and disagreement—filtered by GNN preference margins—captures easy and hard examples, enhancing learning signals beyond confidence-only self-training.
3) Weakly supervised fine-tuning: The combined instruction-tuning plus preference-tuning objective leverages informative but potentially noisy hard examples while reducing overfitting risk.
4) Empirical validation: The approach outperforms strong GNN and LLM-based baselines across 3/5/10-shot settings and shows promising cross-dataset zero-shot transfer, indicating practical value in low-label scenarios.

### Weaknesses
Weaknesses:
1) Gap Between Theory and Practice: The paper's theoretical proofs rely on simplified assumptions (like linear models and independent errors) that may not hold true in the actual, more complex experimental setup.
2) Unreliable GNN Judge: The method assumes the GNN is a trustworthy "judge" when it disagrees with the LLM. This assumption can fail on graphs with noisy structures or unusual patterns, leading the LLM astray with flawed guidance.
3) High Computational Cost: The multi-stage pipeline is computationally expensive, demanding significant time and memory. This raises practical concerns about its scalability to massive, real-world graphs.
4) Complex Hyperparameter Tuning: The framework introduces several new hyperparameters that need careful tuning for each dataset. This adds complexity and makes the method harder to apply effectively without extensive experimentation.

### Questions
n/a

### Soundness
3

### Presentation
3

### Contribution
3
