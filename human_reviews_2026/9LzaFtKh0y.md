# Learning to Defer on Anonymously Annotated Data

- Avg Score: 5.33
- Decision: Reject
- Scores: 6, 4, 6

## Abstract
Recent advancements in machine learning have prompted the development in human-machine cooperation to leverage the efficiency of machines and the reliability of human expertise. One such approach is *learning to defer* (L2D), where a model learns to selectively defer decision-making to humans based on their historical performance on labelled data. Traditional L2D methods require the same set of human experts in both training and deployment phase, so that the system can leverage their historical performance to allocate queries accordingly. This human-specific nature, however, renders inflexibility in dynamic real-world environments where expert availability can fluctuate due to leave, retirement, or the integration of new team members. To address this challenge, we propose leveraging anonymously-annotated datasets, which are commonly available in practice, to infer annotation patterns and cluster human annotators based on behavioural similarities. Building upon the clustering of human experts, we develop a variant L2D, known as L2D-Clusters, that defers queries to a cluster rather than a specific expert, with one expert from the cluster randomly selected to make the final decision. Empirical results show that our clustering aligns with known annotator behaviour and that L2D-Clusters performs comparably to expert-specific L2D, especially in onboarding scenarios with limited annotator-identified data.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes L2D-Clusters as a framework extending L2D to anonymously annotated data.  The algorithm does not need fixed expert identities; instead, it uses an LDA-based model to cluster annotators by behavioral similarity and defers decisions to expert groups.

### Strengths
1. The paper seems to solve the limitations of existing L2D methods by introducing clustering on an anonymously labeled dataset, thereby not needing annotator identities.

2. The LDA modeling seems novel for me (though I am not familiar with L2D works).

3. The empirical results look good.

### Weaknesses
I am not an expert in L2D, but I feel that the model seems to be a little impractical. The model assumes that all experts in one cluster share the same labeling pattern $h(x, \theta_z)$, which is a probability vector allocating fixed probabilities to different labels to $x$. Is this too simplified? This assumption seems strong, as real annotators can demonstrate individual variability or context-dependent noise.

### Questions
How sensitive is your method to heterogeneity within one cluster?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper tackles the "out-of-distribution problem" for human experts, where the available annotators change between training and deployment. The proposed method addresses this by learning to defer tasks to clusters of behaviorally similar experts, identified from anonymously-annotated data, rather than to specific individuals.

### Strengths
1. The problem of "out-of-distribution" for human experts is interesting
2. The proposed method is simple, intuitive, and effective

### Weaknesses
1. The paper does not provide a theoretical analysis to substantiate the method's effectiveness, such as performance guarantees or robustness analysis.
2. The performance is critically dependent on the quality of the expert clustering, a dependency amplified by the stochastic nature of randomly selecting an expert for the final decision. The validation of this clustering is confined to a limited set of datasets, and its efficacy in diverse, real-world scenarios remains unverified.
3. The empirical evaluation relies heavily on synthetic experts. The conclusions would be significantly more compelling with the inclusion of a real human study.

### Questions
See weaknesses

### Soundness
2

### Presentation
2

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
The paper tries to fix a realistic weakness in learning-to-defer systems, where a model decides whether to answer or pass a case to a human. Usual methods assume the same named experts appear during training and testing, which is unrealistic. Here, the authors learn groups of similar annotators from anonymous label counts using a simple topic-model approach. When a new person arrives, the system observes a few of their labels and assigns them to one or more groups. At prediction time, the model either answers by itself or defers to one of these groups and picks any available person within it. A simple control limits how many cases go to humans.

The novelty is that the method works without knowing annotator identities or needing ground-truth labels. It also introduces a realistic onboarding scenario with many experts, each labeling only a few examples, and with noise that depends on the input itself. Experiments on CIFAR-100, dopanim, and Chaoyang datasets show that this group-based approach beats both person-specific and population-based baselines when little data is available for each expert. Once every expert has plenty of labeled data, the standard person-specific systems become stronger. Hard clustering works best overall.

### Strengths
The paper tackles a relevant gap in L2D. The anonymous-data LDA modeling is appropriate for multinomial label counts and cleanly separates training of clusters from later per-expert assignment using a small identified set, which reduces the dependence on an available ground truth. The L2D-Clusters architecture is a natural hierarchical mixture of experts with a gating function over clusters and random expert selection within a cluster, to wihch a workload constraint is added.
The onboarding benchmark captures conditions with many experts and few labels per expert with instance-dependent noise; it shows compelling improvements over expert-specific and population-based baselines in the small-data regime.
Writing is clear enough to follow the modeling and training details, and appendices provide derivations and the algorithm used in practice.

### Weaknesses
The L2D-Clusters section assumes all experts annotate the same training samples to avoid latent-variable complications but, as far as I can see, it weakens the generality of the claim about handling missing annotations and, in my view, should be mentioned more clearly in the paper.
Randomly picking an expert inside a cluster may introduce variance across the humans are subjected to---an alternative within-cluster selection rule could be investigated.
Experientally, some ablation studies are missing to assess the isolated contribution of workload constraint, online-EM momentum, or hard vs soft assignments.

### Questions
How sensitive are your results to the Dirichlet concentration prior for cluster mixtures? How sensitive are they to the choice of the parameter K? Can you report the performance variability (and, possibly, clustering stability/variaiblity) on an alpha-K grid?
  
Can the simplifying assumption that all experts annotate the same samples be relaxed?

Could you comment on any class-imbalance issues that may arise with your method?

### Soundness
3

### Presentation
3

### Contribution
3
