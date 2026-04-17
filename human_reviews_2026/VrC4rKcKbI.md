# Revisiting Feature Interaction Selection in Neural Additive Models

- Decision: Reject
- Scores: 2, 2, 2

## Abstract
In this work, we revisit the paradigm of feature interaction selection for additive models.
This paradigm generalizes the selection of a model's input features to the selection of a model's feature interactions by equipping any model with the additive structure of a generalized additive model.
When applied to neural networks,
this restricts the network's learned representations to interactions between the specified sets of features.
In the study of the training dynamics of these neural additive models,
we discover a new phenomenon which we call `medium dimensionality',
corresponding to a balance between data complexity and model complexity.
We find that this phenomenon helps to explain the good performance of additive models on tabular datasets.
We moreover find that the tool of additive models is able to 
unify insights for many of the recently explored phenomenon of deep learning theory:
double descent, grokking, leap dynamics, and the staircase property.
Finally, we present developments on selections algorithms and neural additive models, benchmarking performance across a suite of tabular datasets.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper revisits the use of Neural Additive Models and the feature interaction selection paradigm. The authors introduce a new concept called "medium dimensionality," describing a balance between data complexity and model complexity. They argue this phenomenon explains why additive models perform exceptionally well on tabular datasets. Furthermore, the authors further argue that the additive model framework serves as a valuable tool to unify and provide a data-centric perspective on several major deep learning theories, including double descent, grokking, and the staircase property.

The authors then proceed to also present a key algorithmic and architectural improvements for training these models. The authors do so by proposing a "batchwise " selection algorithm for identifying feature interactions, which they find easier to tune than previous methods. They also incorporate "masked training" to compute interaction scores, significantly accelerating the process. Finally, they introduce a "mixed block sparsity" architecture that balances GPU parallelism and memory requirements, proving to be both faster and more memory-efficient than prior implementations. The paper benchmarks these developments, demonstrating the value of NODE-GAMs for both applied performance and theoretical insight.

### Strengths
The paper's main strength lies in its ambitious conceptual setting, which aims to provide a novel and valuable contribution by attempting to create a unifying framework to explain four complex deep learning phenomena, such as double descent and grokking. The authors do so through the relatively simplistic lens of additive models. This is a positive highlight, as it offers a simpler, data-centric explanation for these intricate, and well-studied behaviors in the field.

Beyond this theoretical insight, the paper delivers algorithmic and architectural improvements. It introduces a "batchwise" feature interaction selection algorithm that appears to be an enhancement over the previous layerwise method, offering better performance, faster computation, and simpler tuning with fewer hyperparameters. Furthermore, the proposed "mixed block sparsity" architecture appearingly provides a practical solution as it strikes an effective balance between sequential and parallel approaches to be both faster/more space-efficient.

### Weaknesses
I have several significant issues with the paper in its current form.

1. Poor Clarity and Precision in Core Concepts

Firstly, the introduction of key deep learning concepts in Section 2 is quite poorly written and lacks precision.#

On Double Descent: The description of double descent merely mentions "two local minima" but completely omits the most critical components of the phenomenon: the spike in test error at the interpolation threshold and the subsequent increase and decrease in test error as model complexity grows. While many readers may be familiar with the concept, it is crucial for an academic paper to define its terms with clarity and precision. 

On Grokking: The explanation for grokking is similarly imprecise. The authors characterize one of the learning phases as "weight decay." This is very vague. It would be more accurate to describe this as a compression or regularization-driven search phase, or at the very least, the authors should explicitly define what they mean by "weight decay" in this specific context.

On Notation: In Section 2.2, the authors fail to introduce their notation properly. A clear example is in Equation (4), where the notation is used without any prior definition, leaving the reader to guess its meaning.

2. Unsubstantiated Claims and Vague Concepts

The paper makes several bold claims that are either unsupported or poorly integrated into the narrative.The authors make a broad and, frankly, rude and unnecessary assertion on line 187 that high-dimensional statistics is "mainly applied to specific domains like biostatistics." This is an inaccurate generalization that dismisses a vast field of research. 

The newly introduced concept of "medium dimensionality" feels flat. The paper does not clearly demonstrate how or where this concept is truly used. It would be essential for the authors to clarify which of their settings, experiments, or results hold specifically for medium dimensions and not for high dimensions. As it stands, the term is introduced without a clear purpose or payoff.

3. Weak Experimental Design and Justification

Several of the experiments and claims in Section 3 are unconvincing or poorly justified. In Section 3.1, the authors describe a synthetic dataset but would be better served by explicitly and formally defining what this dataset is rather than just describing it in prose. In Section 3.2, the authors state, "even if we know the true model, it may not be the best to fit with finite data." This is a very bold claim to include without solid proof, clarification, and strong experimental backing. The provided experiments are in my opinion too weak to support such a general statement.

The experiment in Section 3.3 is also problematic. The authors introduce an uncommon multiplicative setting to demonstrate grokking. The choice of this dataset and the $y \leftrightarrow x$ relationship seems arbitrary. The results are not convincing, as they only appear to show the grokking phenomenon for a single step

4. Structural and Narrative Flaws 

The paper suffers from a lack of cohesion, in my opinion it currently reads a bit like a collection of disparate parts. Section 4 feels completely disconnected from the rest of the paper. After a theoretical setup in the preceding sections, this section on algorithmic improvements arrives "out of nowhere", ruining the flow of the paper. 

The results in Section 5 (Tables 1 and 2) are concerning. In several instances, the optimized models with tuned hyperparameters perform worse than their untuned counterparts. For example, in Table 2, the tuned GA2M EBM has a lower normalized MSE than its counterpart, and the same occurs for the SVM on the Appliances Energy dataset. This pattern repeats across both tables and multiple datasets/methods, undermining the claims about the proposed optimizations. If i misundsterstood this, I would greatly appreciate if the authors could clarify it.

Finally, the conclusion's mention of "tightening statistical convergence rates" also seems to appear from nowhere and is not linked to any of the paper's actual content or analysis. In general, while the theme of the paper is additive models, it feels like a bundle of many small, underdeveloped analyses. Unfortunately, I do not believe any of these analyses are independently sufficient for acceptance, and their links are too weak. I strongly suggest the authors focus on building a better narrative and either thoroughly corroborate the claims I pointed out or remove them entirely.

### Questions
Please see weaknesses.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
this paper analyzes a few deep learning phenomena with NAM (neural network generalized additive models). The authors focused on the set of medium dimensionality where 2^d >> n (d is the feature dimension, n is the number of training data) and reproduced a few phenomena (double descent, staircase dynamics. From these observations, the authors proposed batchwise feature interaction selection/masked training/mixed block sparsity for improving NAM.

### Strengths
* this paper shows some lead that one can improve the NAM by leveraging known properties of deep learning training dynamics
* focusing on medium dimensionity setup is useful because many real world problems fall into this category

### Weaknesses
the paper presentation is pretty bad:  
* figure 2 is not readable: all subtitle are the same,
* figure 3 is hard to understand
* figure 6 is much hard to reader than that if you just present a table


on the other side, the connection from section 3 to section 4 can be significantly improved. it's probably the most important part (where you observe somethings and then adjust the methodology accordingly) so I would suggest spend more sentences/paragraphs to highlight how and why you made the proposals on section 4 based on what you found on section 3.

### Questions
questions about medium dimensionality
1. it's a quite wide spectrum between [d, 2^d]. does things happen in the whole spectrum or there are some more precision criterion?
2. is the medium dimensionality feature for data, or for model, or for both together?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper tackles the problem of feature interaction selection for neural additive models. The authors first review existing deep learning theory and empirical observations of training dynamics. Then, they experimentally show that the relationship between data complexity and model complexity helps to explain the good performance of additive models. Additionally, they demonstrate that the phenomenon of deep learning theory can be observed in additive models. Finally, the authors propose a feature interaction selection method by modifying the existing SIAN method, and apply it to several tabular datasets.

### Strengths
- This paper focuses on the deep learning phenomena, such as double descent and grokking, on neural additive models. The authors experimentally show that these phenomena can be observed in neural additive models.
- The authors propose a feature interaction selection method by modifying the existing method, SIAN. The proposed method introduces a batch-wise selection and masked training to SIAN. In addition, the authors employ a mixed block-wise sparsity approach to accelerate the computational time of neural additive models.

### Weaknesses
- The presentation of this paper is not well. The relationship between empirical observations of additive models presented in Section 3 and the algorithm development in Section 4 is not clear.
- The experimental results do not support a clear advantage of the proposed method. The reviewer assumes that the SIAN-X in Tables 1 and 2 refers to the proposed method. There is no comparison between the original SIAN and the proposed method.
- The ablation study is missing to show the effectiveness of each component of the proposed method, batchwise selection, and masked training.
- The proposed method seems an incremental extension of the existing SIAN. The novelty and significance of the proposed method are limited.
- The detailed experimental settings, such as the hyperparameter settings for SIAN, are missing.

### Questions
- The authors should clarify how the observations in Section 3 lead to the proposed method in Section 4.
- How is the contribution of each component in the proposed method for the performance?? An ablation study is necessary to clarify this point.
- How do the authors determine the hyperparameters for SIAN and the proposed method? ($K$, $B$, $T$, etc.)
- The following literature tackles a feature interaction selection for neural additive models. In addition, the efficient implementation of neural additive models is mentioned, which is related to Section 4.3 of this paper.

Kishimoto, Y., et al., Neural Additive and Basis Models with Feature Selection and Interactions, PAKDD 2024, https://doi.org/10.1007/978-981-97-2259-4_1

### Soundness
2

### Presentation
1

### Contribution
2
