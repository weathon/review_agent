# Nexus: Specialization meets Adaptability for Efficiently Training Mixture of Experts

- Decision: Reject
- Scores: 3, 5, 6

## Abstract
Efficiency, specialization, and adaptability to new data distributions are qualities that are hard to combine in current Large Language Models. The Mixture of Experts (MoE) architecture has been the focus of significant research because its inherent conditional computation enables such desirable properties. In this work, we focus on "upcycling" dense expert models into an MoE, aiming to improve specialization while also adding the ability to adapt to new tasks easily. We introduce Nexus, an enhanced MoE architecture with adaptive routing where the model learns to project expert embeddings from domain representations. This approach allows Nexus to flexibly add new experts after the initial upcycling through separately trained dense models, without requiring large-scale MoE training for unseen data domains. Our experiments show that Nexus achieves a relative gain of up to 2.1% over the baseline for initial upcycling, and a 18.8% relative gain for extending the MoE with a new expert by using limited finetuning data. This flexibility of Nexus is crucial to enable an open-source ecosystem where every user continuously assembles their own MoE-mix according to their needs.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper provides a MoE framework called Nexus that improves efficiency, specialization, and adaptability.

### Strengths
1) Efficiency, specialization, and adaptability are important when training a MoE network.
2) The paper provides a novel router based on expert embeddings that outperform previous approaches for combining dense experts.
3) The paper provides detailed experiments on the designed methods.

### Weaknesses
1) Lack of technical contributions:
- The methodology part of the paper is too short.
- Some methods mentioned in the paper have been explored in previous research. For example, “shared expert” has been discussed in Deepseekmoe [1], and the adaptation to new domains (expert merging) has been discussed in Mergekit [2]. 

2) Lack of comparison baselines:
This paper lacks baseline models. There are already many open-sourced MoE networks for comparison, e.g., Mixtral [3], Deepseekmoe [1]. The author should compare these models, in terms of training time, data, performance, etc.

[1] Dai D, Deng C, Zhao C, et al. Deepseekmoe: Towards ultimate expert specialization in mixture-of-experts language models[J]. arXiv preprint arXiv:2401.06066, 2024.

[2] https://github.com/arcee-ai/mergekit

[3] Jiang A Q, Sablayrolles A, Roux A, et al. Mixtral of experts[J]. arXiv preprint arXiv:2401.04088, 2024.

### Questions
As in Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This work propose Nexus, a strategy for upcycling dense LLM checkpoints into SMoE while enhancing the experts specialization and adaptation to new domains. Nexus works by first independently train a seed LLM on each domain and then upcycle them via a domain-aware router. The authors compare Nexus against other upcycling strategies in the language modeling task. The results show that Nexus can offer some performance gains while improving the specialization and adaptability.

### Strengths
- SMoE is an important and interesting research direction. This work shows that improving experts specialization can be helpful and can generalize the new domains.
- The experiments involve training LLMs from scratch at a reasonable scale.
- The method is simple and intuitive sound. The results are encouraging and can corroborate the motivation.

### Weaknesses
- One major drawback of this work is that Nexus introduces a dependency on the domain associated to each sample, which does not exist in the baselines. Conceptually, I think this is not a Wdesirable property for real-world scenarios as it requires additional data annotation during deployment. Moreover, training and inference could be more susceptible to data mislabeling or low quality-data. Furthermore, this also introduces an advantage of Nexus over the baselines since it receives an additional input signal. I believe this dependency is an important aspect that is not fully explored in this work. 

- The paper does not include a complexity analysis.

### Questions
**Questions for the authors**

- How would the dependency on the domain impact the performance and practicality of this work?
- What are the training, inference, and memory costs of Nexus compared to the baselines?
- Comparison with the seed model is not entirely fair as it is only trained on the SlimPajama dataset and not on the domain specific data. Two baselines for further comparison could be: (i) continue training the seed model on a mix of all these domain data; and (ii) maintaining a collection of all specialized experts and select the corresponding one for inference based on the domain index, this can be considered as the fully specialized baseline. 
- Will the authors make the data and source code publicly available for reproducibility and further research?

### Soundness
3

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
4

### Summary
This paper introduces Nexus, a method for expanding dense models into Mixture-of-Experts (MoE) models, along with a novel router that introduces strong domain inductive bias and encourages expert differentiation. The experiments validate the advantages of this router in merging separately trained dense experts into an MoE model and in extending to new domains, compared to directly merging models and using a standard linear router. The analysis experiments demonstrate that Nexus achieves domain-specialized routing and shows better performance than dense models under different load balancing and training data ratio settings.

### Strengths
- The paper is clearly written and easy to understand.
- The analysis experiments show that the proposed method indeed encourages experts to specialize.
- The method of incorporating domain embedding information is straightforward and has been shown to outperform linear routers and model merging through comparative experiments.

### Weaknesses
- The comparison to related work is inaccurate. In lines 516-519, the paper discusses ensemble methods, stating that these methods require computing all experts rather than sparse activation. However, both BTM and C-BTM compare results using different top-k values after ensembling model groups, and C-BTM even outperforms MoE with the same activation parameter count. If Nexus emphasizes the importance of specialization, these methods that conditionally ensemble specialized LLM experts should also be compared.
- The comparison to MoE baselines is not entirely reasonable. The Nexus router has more parameters than the linear router. Using the MLP-router from ModuleFormer as the baseline might be more appropriate to exclude the influence of parameter differences. Additionally, in Experiment 5.2, it might be more suitable to expand the parameters of the linear router's weights after expanding the experts rather than resetting them. The poor performance of the linear router at 200M tokens in Figure 3 (left) might be due to the randomly initialized router being undertrained.
- The paper does not explain the necessity of composing domain-specific experts into experts. It only compares merging dense models and initializing MoE with dense experts, but does not compare:
   1. Training a dense model with the mixed data described in the paper. Since the number of tokens used to train the seed model (25B/40B) is similar to the number of tokens used for subsequent domain-specific training, the seed model may not have converged, and domain-specific training can cause significant changes to the model. Merging specialized models at this point might degrade model performance. The authors could consider adding results from training a dense model on all data (including initial and subsequent data) to better illustrate the significance of converting dense models to MoE.
   2. Experiments based on a stronger pretrained foundation model, similar to Branch-Train-Mix. The ratio of training data for the seed model and expert models in the paper differs significantly from real-world domain extension scenarios. Due to the large gap between expert models and the seed model mentioned in point 1, the conclusions drawn from training after merging might differ significantly from real-world scenarios.
   3. Showing the performance of each expert model on the corresponding domain task in Table 1.
- Figures 4 and 5 lack baseline comparisons when explaining specialization. Since experts are already specialized and have clear preferences, it is reasonable to see distinct biases. Comparisons with the linear router would further support these findings.

### Questions
- The performance of the code domain decreases after the code expert is merged into the MoE model. Do experts from other domains have similar issues?

### Soundness
2

### Presentation
3

### Contribution
2
