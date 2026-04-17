# Group-Label-Free Validation for Spuriously Correlated Data

- Decision: Reject
- Scores: 2, 6, 4

## Abstract
Deep learning models are known to be sensitive to spurious correlations between irrelevant features and the labels.
These spurious features can negatively affect the model’s generalization and robustness, particularly for groups consisting of examples without spurious correlations. 
Early approaches address this issue by requiring group labels in the training set, and more recent methods aim to reduce reliance on group labels in training; however, many state-of-the-art approaches still require a validation set with group annotations for hyperparameter tuning or model selection, which are often unavailable or costly to obtain. 
In this work, we propose SIEVE, a plug-and-play module that constructs a group-aware validation set for robust model evaluation under spurious correlations, without using any group annotations. SIEVE identifies confusing training examples based on feature-space similarity, and iteratively separates them into spurious and non-spurious subsets based on differing loss dynamics patterns, which we discovered in our data analysis.
The selected samples are assigned pseudo group labels and used as a surrogate validation set for model selection. Our method is annotation-efficient, easy to implement, and compatible with existing methods that rely on group-labeled validation sets for hyperparameter tuning and model selection. Experiments on benchmark datasets demonstrate that SIEVE enables robust model selection without access to group labels, achieving performance competitive with methods that use true group annotations.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes SIEVE (Spurious-aware IterativE Validation Example selection), a framework for building group-label-free validation sets to evaluate and tune models under spurious correlations. SIEVE identifies “confusing” examples close to the decision boundary in feature space, and then distinguishes between spurious and non-spurious examples based on their loss dynamics during early training: spurious examples tend to have rapidly decreasing losses, while non-spurious ones decrease more slowly or increase. By iteratively labeling these examples and aggregating them into a pseudo group-labeled validation set, SIEVE enables model selection without true group annotations. The method integrates easily into existing training pipelines such as ERM, JTT, AFR, and DaC. Experiments on Waterbirds, Dominoes, Metashift, and CelebA show that models validated with SIEVE perform comparably to those using real group-labeled validation sets.

### Strengths
1. The paper tackles a well-recognized and practical gap: robust model selection under spurious correlations when group labels are unavailable.

2. Compared to some heavy reweighting or adversarial methods, SIEVE’s additional cost (one short training epoch and distance computations) is modest.

### Weaknesses
1. There are already several related papers addressing the same goal of mitigating spurious correlations without group supervision [1, 2]. Moreover, as shown in Table 1, ERM-SIEVE underperforms compared to several baseline methods without SIEVE, raising questions about its practical effectiveness.

2. SIEVE selects confusing samples based on nearest neighbor distances in a pretrained feature space. However, the interpretation of “confusing” versus “non confusing” examples is heuristic and under justified.

3. The reliance on a single epoch to measure loss change seems arbitrary. Since the authors only train one epoch when computing the per sample loss change ΔL, the results may depend on the order of samples or mini batches rather than any intrinsic signal of spuriousness. The paper does not analyze or control for this randomness, so the method’s stability under different data orderings is uncertain.

4. The empirical gains reported in Table 1 are small and inconsistent. In many dataset and method combinations, SIEVE variants nothing or even underperform their baselines, often within one standard deviation. If I understand correctly, the paper does not include any baseline results where neither training nor validation uses group labels in Table 1. The non-SIEVE baselines still depend on group-labeled validation sets for hyperparameter tuning.

[1] Ghaznavi et al., “Trained Models Tell Us How to Make Them Robust to Spurious Correlation without Group Annotation” https://arxiv.org/abs/2410.05345

[2] Kirichenko et al., “Last Layer Re-Training is Sufficient for Robustness to Spurious Correlations” https://arxiv.org/abs/2204.02937

### Questions
1. Regarding Figure 2: The boundary between confusing and non-confusing examples appears unrealistically sharp. Could you explain in detail how Figure 2 was generated, including what metric or threshold was used to separate these two groups and whether any preprocessing or normalization influenced this clarity?

2. Regarding Figure 3: Shouldn’t there also be non-confusing but spurious examples in the figure? It seems that these cases are missing or mislabeled. Could you clarify on this?

3. Regarding Lines 185-187: The statement “For the non-confusing examples, they are relatively easy to classify, thus whether they are spurious or non-spurious, both the true decision boundary and the spurious decision boundary may classify them correctly” seems incorrect. If the classification relies on the spurious decision boundary, then non-confusing non-spurious examples would still be misclassified. Could you clarify?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces SIEVE, a module that constructs a group-labeled validation set to train models robust to spurious correlations, eliminating the need for manual group annotations. The method first identifies 'confusing' training examples based on feature-space similarity and then leverages their distinct training loss dynamics: it assigns 'spurious' pseudo-labels to examples whose loss decreases rapidly and 'non-spurious' pseudo-labels to those whose loss decreases slowly or even increases. This resulting surrogate validation set can be integrated into existing robust training pipelines for model selection.

### Strengths
The core idea of using training dynamics to identify minority group examples is not entirely new; related concepts like learning speed have been explored in works like SSL (Zhu et al., 2023). However, the paper's primary original contribution is the two-stage filtering process. It first isolates a small, informative subset of "confusing" examples using feature-space distance, and then applies the loss dynamics analysis. The insight that the signal for separating spurious from non-spurious examples is strongest near the decision boundary, is a novel refinement that makes the approach more reliable than applying it to the entire dataset.

The empirical evaluation is of high quality and is a major strength. The authors validate their method across four standard and diverse benchmark datasets. Crucially, they don't just show that SIEVE works with a basic ERM model; they demonstrate its "plug-and-play" capability by successfully integrating it into three different state-of-the-art methods (JTT, AFR, DaC). The results consistently show that the SIEVE-generated validation set leads to performance that is competitive with, and sometimes superior to, using a ground-truth validation set.

### Weaknesses
The method's success hinges on the assumption that within "confusing" examples, spurious samples have rapidly decreasing loss while non-spurious ones have slowly decreasing or increasing loss. While this is convincingly demonstrated on four benchmarks, the paper could do more to explore the boundaries of this assumption. The work could be strengthened by designing a synthetic or semi-synthetic experiment where this assumption is deliberately weakened. For example, create a setting where the causal feature is highly complex and the spurious feature is only moderately easier to learn. In such a scenario, the loss dynamics might not show such a clear separation. Characterizing this failure mode would provide a deeper understanding of the method's applicability and limitations, making the paper more complete.

The paper defines confusing examples based on the minimum distance to an example from the opposite class in a pretrained feature space. This is a reasonable proxy for being near a decision boundary, but it is presented as the sole option without much justification against alternatives. The authors could improve the paper by briefly discussing alternative definitions for the "confusing" set. For instance, methods based on model uncertainty (e.g., high prediction entropy) or energy of the logits (-logsumexp) or ensemble disagreement could also identify hard-to-classify examples.

### Questions
1. The core of your method relies on the observation that spurious examples have rapidly decreasing loss while non-spurious ones have slowly decreasing or increasing loss. Could you elaborate on the conditions under which this assumption might fail? For instance, in a setting where the causal features are highly complex and the spurious features are only marginally simpler to learn, would the loss dynamics still show a clear enough separation for SIEVE to effectively distinguish between the groups?

2.  The selection of pseudo-spurious and non-spurious examples is governed by a quantile threshold `τ`. While the sensitivity analysis shows robustness within a narrow range, a practitioner would still need to manually set this hyperparameter. Have you explored any data-driven heuristics to automate this selection? For example, could the tails of the loss change distribution be identified automatically using outlier detection or by fitting a mixture model, thereby making the method more self-contained?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes SIEVE (Spurious-aware Iterative Validation Example selection) as a method for constructing group-aware validation sets without requiring explicit group annotations. The insight follows how confusing training examples, which are close to the decision boundary, exhibit different loss dynamics when they are spurious (rapid loss decrease) versus non-spurious (slower or even loss decrease).  SIEVE leverages this observation to iteratively select and pseudo-label validation examples, enabling robust model selection for methods that try to address spurious correlations. Experiments on four benchmark datasets show that SIEVE achieves comparable performance with methods that use true group labels when integrated.

### Strengths
The paper introduces a practical and modular solution for the bottleneck of needing direct group-labeled validation for robust model selection under spurious correlations.

The authors show empirically across 4 benchmark datasets that SIEVE-validated models can match or beat methods using true group labels.

The analysis on sensitivity and robustness checks for architectures and hyperparameters is indicative of the method’s stability and overall robustness.

### Weaknesses
I am confused about how Algorithm 2 allows training either on the full training set or with the SIEVE-selected validation samples removed. The authors claim either setting can be selected based on the desired setup. However, if the full training set option is used, evaluation on SIEVE-selected samples coud cause contamination between train and validation. Either the paper should standardize or remove the validation examples from train, or justify the effect of including it otherwise.

While the empirical observation about loss dynamics is compelling, the paper lacks concrete analysis of why this phenomenon occurs or under what conditions it holds. The conceptual explanation in Section 2.1 is intuitive but seems informal. What happens when the assumptions are violated, for eg., when loss dynamics don’t separate as cleanly, or when multiple dominant spurious features occur? 

The precision of the method drops when the data contains small/hard minority groups. This can hurt methods that depend on correctly surfacing minority samples, which is quite notable for JTT-SIEVE on Waterbirds (88% to 79%). Further discussion on potential mitigation strategies are warranted.

It would be more convincing for the claims of generalisability if experiments with SIEVE go beyond vision only tasks to also include tabular or text classification tasks where there are known spurious correlations. This would also help strengthen the justification on loss dynamics.

### Questions
Could you clarify which setting of validation samples (with or without exclusion from training) was used in the experiments? Additionally, justify if both settings were reported?

Can you provide more formal analysis of when and why the loss dynamics pattern holds? Are there specific properties of the spurious features or data distribution that are required?

Can you investigate more deeply why JTT-SIEVE underperforms on Waterbirds? Are there modifications to SIEVE that could better serve methods requiring accurate minority group identification?

How does SIEVE extend to cases with multiple spurious attributes or continuous spurious features? The current evaluation only considers binary spurious/non-spurious divisions.

Would it be possible to include non-vision datasets (e.g. MultiNLI) to demonstrate generalization beyond vision/ CNN based setups?

### Soundness
2

### Presentation
3

### Contribution
3
