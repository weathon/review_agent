# Explain Yourself, Briefly! Self-Explaining Neural Networks with Concise Sufficient Reasons

- Decision: Accept (Poster)
- Scores: 6, 6, 6

## Abstract
*Minimal sufficient reasons* represent a prevalent form of explanation - the smallest subset of input features which, when held constant at their corresponding values, ensure that the prediction remains unchanged. Previous *post-hoc* methods attempt to obtain such explanations but face two main limitations: (1) Obtaining these subsets poses a computational challenge, leading most scalable methods to converge towards suboptimal, less meaningful subsets; (2) These methods heavily rely on sampling out-of-distribution input assignments, potentially resulting in counterintuitive behaviors. To tackle these limitations, we propose in this work a self-supervised training approach, which we term *sufficient subset training* (SST). Using SST, we train models to generate concise sufficient reasons for their predictions as an integral part of their output. Our results indicate that our framework produces succinct and faithful subsets substantially more efficiently than competing post-hoc methods while maintaining comparable predictive performance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper aims at the generation of minimal sufficient reasons for model decision explanation. The authors provide theoretical results on the intractability of getting cardinally minimal explanations under specific settings, showing the difficulty of this problem. The authors also propose a new self-supervised approach called sufficient subset training to reduce cost for generating faithful sufficient reasons and reduce sensitivity to OOD examples.

### Strengths
* The paper provides rigourous theoretical results on the computational complexity of generating minimal sufficient reasons in different settings. 
* The paper is well-written and easy to follow. 
* The proposed SST method provides an elegant and computationally efficient way to generate a sufficient reason. The experiment results show strong scalibility to larger datasets.

### Weaknesses
While the paper presents rigorous theoratical results and a novel empirical method, it's a bit unclear to me how these two contributions are connected. Discussion on how the theoratical analysis motivates the SST method could make the paper more integrated.

### Questions
1. In Table 2, how is the faithfulness of SST training using different masking strategy in training? It will be interesting to see if model trained with specific masking stretagy also generalizes to other faithfulness metrics. 
2. In Figure 4, is Cardinality Mask Size (%) axis refering to the percentage of mask size? According to the figure, cardinality goes to 0.5% instead of 50% as discussed in Line 428-429.

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
The authors propose a novel training framework called Sufficient Subset Training, aimed at generating minimal sufficient reasons as integral outputs of neural networks. 

Unlike post-hoc methods that face computational challenges and out-of-distribution (OOD) concerns, SST directly incorporates the explanation generation process during training. 

This is achieved by adding dual propagation and integrating two additional losses: (i) a faithfulness loss for ensuring sufficiency and (ii) cardinality loss for promoting minimal subsets. The method is validated through experiments on various image and language tasks, demonstrating that it provides concise, faithful, and efficient explanations, outperforming several post-hoc methods at finding minimal subset.

### Strengths
- The introduction of SST as a self-explaining mechanism is novel and could be impactful for fields focusing on model interpretability.
- The paper thoroughly defines different sufficiency types and explains how SST addresses them through tailored masking strategies.
- The experiments cover a range of datasets and architectures, showing that SST can be applied across different domains.
- The theoretical analysis on the intractability of obtaining minimal sufficient reasons adds depth to the contribution.

### Weaknesses
However, I have identified some issues, both major (**M**) and minor (**m**), that require attention.

**M1** Real Practical Insights Are Limited:  while the theoretical framework and empirical demonstrations are solid, the practical insights into what we learn about the internal workings of neural networks are minimal. The method emphasizes explanation generation without significantly advancing our understanding of how or why models arrive at decisions. Addressing this could bridge the gap between theoretical contributions and **practical interpretability**.

**M2** Validity of Faithfulness Metrics: the paper does not thoroughly discuss the potential limitations of the faithfulness metric used in the evaluation. Faithfulness as defined could lead to explanations that superficially appear sufficient but do not align with internal mechanism or key feature used by the model. Say it otherwise, the model may be using completely different strategy to classify x and xS; zS¯.

**M3** Limited Hyperparameter Discussion: the paper mentions tuning the cardinality loss coefficient but provides no analysis on the impact / visual results of tuning the hyperparameters. What are the results of adjusting \( \tau \), step size \( \alpha \) in robust masking. A sensitivity analysis would add valuable context.

Now for the minors (**m**) issues:

**m1** Related Work Section Is Limited:  
The related work section lacks depth, with insufficient coverage of recent advancements in XAI and self-explaining methods, especially those published in top venues over the last few years.

**m2** Justification for Masking Strategies:  
The choice of certain masking strategies (e.g., baseline and probabilistic) could be better justified. Also, could some baseline lead to better results / scores ? Overall, the rationale for selecting these over alternative approaches should be discussed in more detail.

**m3** Typos and Clarity:  
I havent found a lot of typo, just "probablistic" (should be "probabilistic").

### Questions
- Could we incorporate some prior like finding mask in a lower dimensional subspace (7x7 grid of the original image) to allow a smoother explanation and more tractable problem for the model ? I think this would be valuable for both interpretability purpose and theoretical verficiation (it become easier to generate gt).
- How does the method's approach compare when applied to more interpretable models (e.g., decision trees)?
- Can the SST approach provide insights into potential biases in training data by analyzing consistent feature patterns?


**Overall, while the theoretical considerations and framework of the paper are strong, the practical interpretability of models remains underexplored. Despite this, the proposed method is an interesting step toward integrating explanation generation within model training, justifying an acceptance with room for further practical development.**

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
This paper proposes a new training method called Sufficient Subset training, which proposes learning neural networks with a dual objective that first predicts the label but also predicts a mask of sufficient inputs that is enough to predict the object. They show that training a network this way gives faithful sufficient subset explanations, that are often smaller than those generated with post-hoc methods, and much faster to produce.

### Strengths
- Very well written, the description of previous works in terms of 3 definitions is particularly clear and helpful. 
- Interesting theoretical results justifying the use of training time intervention
- Clever and relatively simple method
- Some good results, especially on MNIST

### Weaknesses
- Decent loss in performance/overall less impressive results on ImageNet and BERT
- Table 2 and 3 missing sufficiency results of your models on the metric it wasn't trained on. Seems unfair to report baselines on all metrics but your methods only under the best metric.
    - I'm worried the self-explanation might overfit to the specific training scenario and not work well in other settings.
- Having very disjoint input sets looks pretty weird for image explanations, what do you think is the use case for this type of explanation? Overall could use some more discussion on why sufficiency explanations are useful/what is the proposed use case. Have you experimented with favoring more continuous regions?

### Questions
- What's stopping the model from learning a "cheating" solution such as it almost always only looks at a certain subset of the inputs and this subset is always the explanation? This would probably not be what we want?
- What is the robustness of your different masking strategies on Table 2 and Table 3 when evaluated in a setting not trained on?
- Is the epsilon ball used too small? Theoretically the robust sufficient reasons case should be harder than baseline etc, but seems like you can learn much smaller explanations with this method. As an extreme case, an adversarially robust model could have a sufficient explanation of size 0 and still be faithful in this metric, but this seems pretty detached from the idea of sufficient explanation.

### Soundness
3

### Presentation
3

### Contribution
2
