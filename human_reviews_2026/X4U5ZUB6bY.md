# GatePro: Parameter-Free Expert Selection Optimization for Mixture-of-Experts Models

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 6, 4

## Abstract
Modern large language models leverage Mixture-of-Experts (MoE) architectures for efficient scaling, but face a critical challenge: functionally similar experts are often selected simultaneously, creating redundant computation and limiting effective model capacity. Existing auxiliary balance loss methods improve token distribution but fail to address the underlying expert diversity problem. We introduce GatePro, a novel parameter-free method that directly promotes expert selection diversity. GatePro identifies the most similar expert pairs and introduces localized competition mechanisms, preventing redundant expert co-activation while maintaining natural expert specialization. Our comprehensive evaluation demonstrates GatePro's effectiveness across model scales and benchmarks. Analysis demonstrates GatePro's ability to achieve enhanced expert diversity, where experts develop more distinct and complementary capabilities, avoiding functional redundancy. This approach can be deployed hot-swappable during any training phase without additional learnable parameters, offering a practical solution for improving MoE effectiveness.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces GatePro, a parameter-free method that addresses expert selection diversity in Mixture-of-Experts models by introducing localized competition between functionally similar expert pairs. The approach identifies similar experts via cosine similarity and applies penalties to prevent their co-activation.

### Strengths
This paper identifies the expert selection diversity problem in MoE architectures, and proposes a simple yet effective solution that requires no additional learnable parameters. The experimental evaluation is comprehensive, covering multiple model scales, training stages, and benchmarks. The hot-swappable design offers practical value for real-world deployment.

### Weaknesses
(1) While the paper presents an intuitive motivation for introducing competition between similar experts, it lacks analysis to explain why cosine similarity of gating weights is the optimal metric for identifying functionally redundant experts. Two experts might have similar gating weight directions but could still process tokens differently due to their internal FFN parameters. For example, two experts might both be activated by technical tokens (leading to similar gating weights), but one could specialize in mathematical reasoning while the other handles programming syntax. 

(2) The paper introduces a fixed penalty coefficient λ = 10⁻⁴ for the losing expert in the competition mechanism, but provides limited investigation of its sensitivity. Similarly, the paper uses k=6 active experts across all experiments, but doesn't explore how GatePro's effectiveness might vary with different k values.

(3) The paper demonstrates that GatePro reduces expert similarity and increases diversity metrics, but it provides limited insight into what kinds of specializations the experts actually learn. Do experts under GatePro develop more interpretable specializations? 

(4) The paper identifies the most similar expert j*(i) for each expert i based on gating weight similarity, but does not clearly specify how often this mapping is updated during training. During training, experts continuously learn and specialize, causing their similarity relationships to change. If the mapping is infrequently updated, it may not adapt to the evolving specialization of experts during training. If recomputed at every step, this adds computational overhead that is not accounted for.

### Questions
The experimental evaluation primarily compares GatePro against baseline MoE with standard balance loss, but does not compare against other potential diversity-promoting approaches mentioned in related work. How does GatePro compare to other methods such as entropy regularization, orthogonality constraints, or decorrelation losses?

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
3

### Summary
This paper proposes GatePro, a parameter-free method to improve expert selection in MoE models by introducing localized competition between the most similar experts. The approach reduces redundant expert co-activation, improves expert diversity, and leads to consistent performance gains across model scales and benchmarks.

### Strengths
The method is parameter-free and can be added or removed during training without affecting the architecture, which makes it easy to deploy in practice.

The approach directly targets expert redundancy and shows consistent improvements across multiple scales and tasks, including both reasoning and knowledge benchmarks.

The analysis is thorough, including expert utilization, similarity metrics, and training-stage comparisons, which provides clear evidence that the method improves expert diversity.

### Weaknesses
The experiment lacks computational cost and stability analysis. Calculating the similarity matrix becomes burdensome as the number of experts increases, especially when expert parallelism is enabled, requiring frequent communication. Therefore, a quantitative analysis of training speed, memory usage, and communication costs is needed.

The experiment fixed the hyperparameter coefficients for the competition penalty; changing these coefficients may have different effects on model performance. More sensitivity analysis or theoretical justification is required.

### Questions
Refer to Weaknesses

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
3

### Summary
The paper raises a method to cut down expert redundancy and enhance diversity by manipulating logits. It identifies the most similar expert pairs and introduces localized competition mechanisms, which suppress the logits of the loser. With experiments of decent scale, it reported consistent performance improvement and provided much analysis.

### Strengths
1. It reports better performance.
2. Large-scale experiments have been conducted.
3. Much analysis is provided.

### Weaknesses
1. Will such a small \(\lambda\) (1e-4) on the logits actually affect routing and training?  

2. I don’t understand how you calculate the zero-token count. Is it calculated over the whole batch? It seems strange that so many experts could be unused after 1k steps. Did I miss something?  

3. Could you also provide the loss curve and the training loss gain?

### Questions
1. I would like to raise the same questions as I raised in the 'Weakness' section.
2. How do you choose the size of lambda? Is there any ablation?

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
This paper proposes a novel method for Mixture-of-Experts (MoE) to balance the selection of experts. Overall, the paper is mostly clear in its writing and explanations and demonstrates an improvement over a naïve baseline. However, I was left with 2 main questions:

•	Is expert selection that is balanced actually a problem? I’m inclined to agree with the authors here that it makes sense to use all of the parameters possible in a model as much as possible, but lots of problems we encounter have a long-tail distribution. Due to this fact, does it make sense that activations of routings should also have a long-tail? Probably not, but is there anything that you can show (even somewhat anecdotally) with your proposed model that demonstrates why competitive propagation is more useful? I see that your accuracy improves in your tables and figures, but is there a clearer way to show why? For instance, looking at imbalanced classes in a test set?

•	Your experiments clearly show you beat the common MoE baseline (Fig 2, Table 1). Figures 3 & 4 are more ablations with the balancing. However, there are a lot of papers that have proposed improvements to Mixture-of-Experts and it would make sense to compare to some of those that have also improved over this same baseline. Could you show an experiment compared to another balancing router MoE method that has been published?

My second question about baselines comes from a desire to see the method compared to recent methods. Your related works section (2) has a lot of potential baselines that you could chose to consider. I’d like to see one of them.


Caption for Figure 1 should be more descriptive. The method is explained in the text, and in equations, and pseudo-code, but the figure is very opaque and it should describe to the reviewer what it they are looking at.

### Strengths
Interesting new method that beats a naive baseline.

### Weaknesses
Stronger baselines needed.

Caption for Figure 1 should be more descriptive. The method is explained in the text, and in equations, and pseudo-code, but the figure is very opaque and it should describe to the reviewer what it they are looking at.

### Questions
Why does the method matter? Re: my question about long-tail distributions above.

Please show another more recent method as a stronger baseline.

### Soundness
2

### Presentation
3

### Contribution
3
