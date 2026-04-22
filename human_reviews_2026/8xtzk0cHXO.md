# A Cooperation Index for Model Pruning

- Avg Score: 2.50
- Decision: Reject
- Scores: 2, 2, 2, 4

## Abstract
In complex models, tools for measuring parameter importance identify its core
 functional element and improve both generalizability and interpretability by pruning
 redundant ones. Effective pruning relies on these tools, which serve as decision
making criteria. The SHAP Value (SV) has recently been considered such a
 criterion, interpreted as measuring the average marginal contribution across all
 possible paths of parameter accumulation. However, we find that this averaging
 process of SV systematically overweights redundant parameters. Instead, we
 propose that measuring the speed of decay of the marginal contribution can serve
 as a more effective decision-making criterion. Specifically, we quantify the number
 of cooperative contribution for each parameter and show that this criterion is more
 effective for parameter pruning in backward elimination, leading to a more optimal
 set of remaining parameters.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes a Cooperation Index for pruning that measures how often a parameter’s marginal contribution exceeds its own Shapley average along sampled permutations. This captures the decay speed of contributions rather than relying only on the mean. A two level approximation is presented. First, vertex sampling fits a regression surrogate on the subset hypercube. Next, permutation sampling estimates marginal contributions. Pseudocode and empirical validation are provided on VGG 16 and ResNet 18 across MNIST, CIFAR 10, CIFAR 100, and Tiny ImageNet.

### Strengths
[1] The motivation is clearly articulated: averaging marginal contributions can overweight redundant parameters; counting cooperative paths aligns better with pruning decisions where replaceability matters

[2] The proposed index is intuitive and interpretable: parameters that consistently help across many contexts are preserved, while sporadically helpful ones are down-weighted.

### Weaknesses
[1] Evaluation focuses on very low pruning ratios, approximately 1 to 3 percent. The absence of full accuracy versus sparsity curves limits claims about robustness at moderate or high sparsity.

[2] Scalability evidence is limited to mid scale convolutional networks. Results on transformer architectures or larger models are missing.

[3] Theoretical analysis could be deepened. Mathematical properties of the index such as monotonicity, relations to Shapley axioms, and performance guarantees under milder assumptions are not fully explored.

[4] Ablations and statistics are sparse. Sensitivity to permutation and vertex sample counts, surrogate architecture, and sampling distributions is not systematically quantified. Standard deviations and confidence intervals are not consistently reported.

[5] Presentation can be improved. Some figures are dense and lack error bars. The text repeats parts of related work and could be tightened.

### Questions
[1] How does the method behave at higher pruning ratios such as 10 to 60 percent. Are there inflection points where performance degrades more steeply than methods based on Shapley value or magnitude-based baselines?

[2] How sensitive is the ranking to surrogate model misspecification or to underfitting and overfitting on the subset hypercube. Can uncertainty in the surrogate be propagated into confidence intervals for the Cooperation Index?

[3] How would the definition adapt to structured pruning across channels, heads, or layers where units are grouped rather than independent?

[4] In low redundancy regimes, as hinted by the Tiny ImageNet results, could an adaptive blend between the Cooperation Index and the Shapley Value reduce worst-case degradation?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
To address the issue of overweighting redundant parameters in Shapley values, the authors proposed a simple metric Cooperation Index (CI) that utilizes the speed of decay of marginal contributions, by incorporating permutations beyond standard Shapley values calculations. They conducted several pruning experiments on image datasets and demonstrated superior performance in terms of accuracy over previous methods like Shapley values and LOCO.

### Strengths
1. The authors proposed a simple extension of Shapley value, Cooperation Index (CI) that achieves better pruning performance on several empirical datasets than previous methods.
2. The paper is easy to follow with useful visual explanations, such as Figure 1 to help readers understand the intuition.

### Weaknesses
1. Unlike Shapley values that satisfy the four important theoretical properties, CI lacks formal theoretical justifications. It seems CI is more designed for pruning-oriented rather than a fairness-oriented approach.
2. The criteria to choose pruning ratios in Table 2 are not clearly stated. The pruning ratios varies from 3% to 1% without further discussion on rationales.
3. A more detailed analysis of sampling number for stability of CI from line 426 to 431 would be appreciated. It's likely a number related to sample size and dimension. It would be helpful to see the trend of convergence for CI wrt sample size and dimension to guide users to choose parameters in practice, rather than demonstrating results using only ResNet-18. 
4. The experiments focus exclusively on no-tabular data sets. How does CI work on tabular data sets, or low-dimension datasets? 

Some typos:  

Line 84. Missing space after "uncompromised".   
Line 95. Missing space after "effectively".

### Questions
1. Shapley values satisfy some important theoretical properties (efficiency, symmetry, dummy, and additivity), which of these, if any, do CI satisfy?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper argues that SHAP is not a good metric for model pruning as it is overweights redundant parameters. The authors propose a novel metric called Cooperation Index, which instead quantifies how consistently a parameter's marginal contribution exceeds its own average to better identify essential parameters.

### Strengths
* The paper's critic about Shapley values being a bad metric for model pruning is insightful. The authors compellingly explain the motivation for their work. 
* The proposed Cooperation Index metric is an elegant and intuitive metric which aims to address this identified limitation.

### Weaknesses
The paper's theoretical motivation is unfortunately not matched by a sound and sufficient experimental evaluation:

* **Experiments**: 
1. Pruning is useful for making large and computationally expensive models more efficient. However, the authors exclusively test their method on small models like VGG and ResNet-18 that are already fast and do not have any need for pruning. It is unclear why one would develop a pruning method that scales poorly with the number of parameters and cannot be easily applied to large models that need the pruning the most.
2. Evaluation is limited to very old and small-scale ConvNets (VGG-16 and ResNet-18) and entirely ignores transformers, which currently represent the dominant paradigm.
3.  Authors do not include a ViT-tiny model. My suggestion that if this model is too expensive to run for this algorithm, then they need to scale it down even further, but the experiments for the transformer must be presented.
4. The paper fails to compare against modern pruning methods. For instance Wanda [1], which is a simple, fast, and highly effective method for large models. Its absence makes it difficult to judge the practicality of the proposed method against the current state of the art. 
5. Missing baselines from NLP. For instance, authors could have included tiny version of BERT and GPT.
* **Runtime**: despite proposing a two-step approximation scheme, the paper provides no empirical runtime to quantify its computational cost. This is especially concerning as the method inherits the factorial complexity of Shapley values, which is fundamentally at odds with the scaling laws (the more parameters the better). I believe the authors need to provide an actual runtime of the algorithm and compare it with a runtime of the competing baselines.


[1] A Simple and Effective Pruning Approach for Large Language Models, Sun et al., 2023

### Questions
The dominant trend in deep learning shows that model performance scales predictably with size, favoring methods with low polynomial complexity in terms of the number of parameters. Given this, how do the authors envision their computationally intensive approach fitting into the current landscape? Is there a specific application where such computational costs are justified?

### Soundness
1

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
3

### Summary
The authors argue that the popular SHAP Value (SV) method, by averaging marginal contributions, systematically overweights redundant parameters. The paper introduces a new metric called the Cooperation Index (CI). CI quantifies the consistency of a parameter's contribution by measuring the frequency of "cooperative paths" - permutations where the parameter's marginal contribution exceeds its average SV. Experiments on VGG-16 and ResNet-18 show that CI-based pruning more effectively preserves the model's core functional elements.

### Strengths
- The paper provides a conceptual critique of the SHAP Value (SV) as a pruning criterion. It formally identifies that SV's averaging mechanism fails to distinguish between redundant parameters (replaceable, high variance in marginal contributions) and cooperative parameters (consistent contributions).
- The proposed Cooperation Index (CI) is a novel metric that directly addresses this identified limitation. 
- The work addresses the exponential computational complexity of the metric with a practical two-level approximation scheme. 
- The experimental results demonstrate the effectiveness of the CI criterion. Across multiple datasets and architectures, CI-based pruning achieves superior or competitive accuracy compared to SV and other baseline methods.
- The paper is well-written and clear.

### Weaknesses
- Specific Evaluation Conditions: The empirical validation is conducted under non-standard conditions, as models are intentionally overfitted on heavily reduced datasets (e.g., 1/100 MNIST) . The method's effectiveness is not guaranteed in standard training regimes. Has its effectiveness also been evaluated when overfitting is caused by other factors, such as prolonged training duration?
- Focus on Low Pruning Ratios: The main experiments (Table 2) almost exclusively focus on very low, "delicate" pruning ratios (e.g., 1-3%). This narrow scope fails to demonstrate the method's scalability and performance at higher, more practical levels of sparsity.
- Flawed Statistical Reporting: The paper lacks statistical rigor by explicitly reporting the "best performing result in terms of accuracy" from five runs, rather than the mean and standard deviation. This practice of "cherry-picking" results may significantly overstate the method's true performance.
- Dependence on Approximation Accuracy: The CI calculation is critically dependent on the accuracy of the regression function used in the two-level approximation scheme. The study does not sufficiently analyze the method's sensitivity to potential errors introduced by this approximation eg. the convergence of CI is shown only for the first few filters.
- Ambiguity in Tie-Breaking: The illustrative toy example shows identical CI scores (0.25) for four of the five parameters in the initial stage. The paper fails to explain the tie-breaking mechanism used to select $w_4$ for pruning, making the selection criteria ambiguous.
- Rigid Contribution Threshold: The Cooperation Index relies on a strict binary threshold (above or below the mean SV) to classify contributions. This rigid classification may inaccurately assess parameters whose marginal contributions are consistently very close to the average.
- Limited Dataset Variety: The evaluation is restricted to a few datasets . The conclusions would be strengthened by validation on a more diverse set of datasets, such as Fashion-MNIST, SVHN, KMNIST, STL-10, Caltech-101.
- Incomplete Baseline Comparison: The MCI baseline method was prematurely dismissed from the main experiments based only on poor performance in the synthetic example. For a complete and fair comparison, its results on the main experiments should be included regardless, for example, in the appendix.

Figure 8 is unreadable - the font size is too small.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
