# Neat: Nonlinear Parameter-efficient Adaptation of Pre-trained Models

- Decision: Reject
- Scores: 3, 6, 5, 6, 5

## Abstract
Fine-tuning pre-trained models is crucial for adapting large models to downstream tasks, often delivering state-of-the-art performance. However, fine-tuning all model parameters is resource-intensive and laborious, leading to the emergence of parameter-efficient fine-tuning (PEFT) methods. One widely adopted PEFT technique, Low-Rank Adaptation (LoRA), freezes the pre-trained model weights and introduces two low-rank matrices whose ranks are significantly smaller than the dimensions of the original weight matrices. This enables efficient fine-tuning by adjusting only a small number of parameters. Despite its efficiency, LoRA approximates weight updates using low-rank decomposition, which struggles to capture complex, non-linear components and efficient optimization trajectories. As a result, LoRA-based methods often exhibit a significant performance gap compared to full fine-tuning. Closing this gap requires higher ranks, which increases the number of parameters. To address these limitations, we propose a nonlinear parameter-efficient adaptation method (NEAT). NEAT introduces a lightweight neural network that takes pre-trained weights as input and learns a nonlinear transformation to approximate cumulative weight updates. These updates can be interpreted as functions of the corresponding pre-trained weights. The nonlinear approximation directly models the cumulative updates, effectively capturing complex and non-linear structures in the weight updates. Our theoretical analysis demonstrates taht NEAT can be more efficient than LoRA while having equal or greater expressivity. Extensive evaluations across four benchmarks and over twenty datasets demonstrate that NEAT significantly outperforms baselines in both vision and text tasks.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
The paper introduces NEAT, a new method for fine-tuning pre-trained models efficiently by using nonlinear transformations. Traditional methods like Low-Rank Adaptation (LoRA) struggle to capture complex, nonlinear patterns in weight updates due to their linear approximation approach. NEAT overcomes this by incorporating a lightweight neural network that transforms pre-trained weights, allowing it to model more intricate updates without increasing the parameter count significantly. Theoretical analysis shows that NEAT is more expressive than LoRA with fewer parameters, and experiments demonstrate its superior performance across a wide range of vision and language tasks compared to existing parameter-efficient fine-tuning methods.

### Strengths
1. The paper is well-written and easy to follow.

2. The authors provide theoretical analysis to establish a lower bound for their method.

### Weaknesses
1. I have concerns about the paper's motivation, particularly the claim that LoRA struggles to capture complex, non-linear components and efficient optimization trajectories. In my opinion, LoRA models the differences in weight matrices, and the weight matrices are linear transformations. In neural networks, non-linearities are applied to activations, not the weight matrices themselves. Therefore, there is basically no motivation to add non-linearity to model the weight changes. Introducing non-linearity into the low-rank matrices A and B may contradict the core idea of "efficient fine-tuning" by adding unnecessary complexity.

2. The paper does not include comparisons of time or memory efficiency, which are crucial factors in the context of efficient fine-tuning methods. Including such evaluations would provide a more comprehensive understanding of the method's practicality.

3. The paper relies heavily on results from other studies, and I have concerns about their reproducibility and thus the fairness of these comparisons. It would be beneficial to introduce consistent experimental settings and perform fair comparisons to ensure the validity and reliability of the conclusions drawn.

### Questions
Please refer to the weakness part.

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Existing finetuning methods use linear adapters, either as an additive or multiplicative adaptation. However, this may be too restrictive of a function class to have as an adapter. Neat instead uses a small MLP to act as its adapter, with the architecture focused on in this paper being a 1-layer MLP. This has around the same number of parameters as LoRA for square parameter matrices, yet also contains an expressive nonlinearity. Empirically, this nonlinearity improves upon LoRA on multiple baselines. Theoretically, Neat is shown to be as expressive as LoRA depending on how the input and output dimensions compare, and when they are equal, Neat is at least as expressive as LoRA.

### Strengths
- The motivation is quite clear. Extending these fine-tuning methods to a non-linear adapter is quite natural.
- There is a theoretic understanding of the expressiveness of this method, both as an upper bound and a lower bound as compared with LoRA with varying parameter counts.
- The writing was clear, with both the method and any desired details easy to find through a cursory glance. 
- Experiments into changing depth provide some insight into the behavior of changing the parameter count of Neat.

### Weaknesses
- For Proposition 5.1, there is an assumption that the loss function is invariant under projection to a left singular space. Listing some losses that are invariant under this projection would strengthen that result.
- For Figure 2 (and possibly other experiments), a single run for each depth doesn't provide strong confidence that this trend wasn't from random chance. A few more randomly initialized experiments to have some idea of the fluctuations over multiple runs would be useful.
- The experiments run are only for Neat. The other methods' accuracies come from other work rather than being reproduced, which may result in some differences simply from the training implementation.

### Questions
- Would it be possible to run the experiments with LoRA in your codebase? Having a comparison between the two methods under the training loop would make the benefit of Neat more convincing.
- In Table 3, the adaptation either starts at layer 4 with a single hidden dimension or is applied to all layers with 8 hidden dimensions. Is there any reason why these specific options were chosen? 
- Can the parameter counts be reported in the experiments? At least for both Neat and LoRA to have that comparison.

Small things:
- In Tables 1 and 3, underlining the second-best accuracy would be nice. 
- In Figure 2, can the depth 0 (i.e. base model) accuracies be added to the figure?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes NEAT to learn the adaptation matrices with the nonlinear transformation of the original weight matrices. NEAT applies an MLP to the pre-trained weight matrices and outputs their additive updates. Both theoretical and empirical analyses are provided to demonstrate the effectiveness of NEAT.

### Strengths
- The proposed method is simple.
- The authors have conducted extensive analysis of their proposed method.
- NEAT achieves promising results on both language and vision tasks.

### Weaknesses
- The authors argue that the main drawback of LoRA lies in the `low-rank` adaptation matrix. However, the employed multi-layer network in NEAT, which is $\sigma(\sigma(W^0 \theta_1)\cdots\theta_{l-1})\theta_l$, also leads to low-rank output. Let $\hat{A}=\sigma(\sigma(W^0 \theta_1)\cdots\theta_{l-1})$ and $B=\theta_l$. $\hat{A}$ can be regarded as the approximation of the matrix $A$ in LoRA. Given any $l\in\mathbb{N}$, and the same size $r$ where $A,\hat{A}\in\mathbb{R}^{d_1\times r}$ and $B\in\mathbb{R}^{r\times d_2}$, the rank 
 of the learned adaptation matrix $AB$ or $\hat{A}B$ always equals $r$. Therefore, `NEAT fails to address the low-rank drawback of LoRA`. 
- Given the high similarity between LoRA and NEAT, the authors did not provide sufficient effectiveness analysis. For example, why does the nonlinear approximation of LoRA perform better than the original LoRA? The authors should perform an ablation study on LoRA and NEAT given the same value of $r$ with different numbers of layers $l$. NEAT requires additional multi-layer neural networks to approximate the low-rank matrix. Therefore, what is the computational cost NEAT should take to outperform LoRA and the other PEFT methods? 
- The proof of proposition 5.1 is not closed. It only covers the equality case; however, the strict inequality is not yet established.
- The author should provide the parameter volumes for the methods in Tab 1-2 for a fair comparison.
- More datasets should be included in Figure 2. Results on a single dataset are less convincing and can not give rise to persuasive conclusions.

### Questions
Please refer to weaknesses.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes NEAT, a method that introduces non-linear structures to PEFT methods. Theoretically, NEAT is shown to be more efficient than LoRA while maintaining equal or superior expressivity. The authors present comprehensive experiments to highlight NEAT's advantages and follow this with a thoughtful discussion on the impact of parameters, number of layers, and choice of activation functions.

### Strengths
1. The paper is well-written and easy to follow. The ideas develop naturally, and the authors provide solid theoretical guarantees to support their claims.
2. The experimental validation is robust, covering a diverse range of tasks in both vision and language models. The comparison with various baselines adds to the credibility of the results and makes the case for NEAT compelling.

### Weaknesses
1. One area that could be strengthened is the comparison between NEAT and its linear counterpart. NEAT introduces two key modifications compared to LoRA ($y=(W_0+BA)x$): (a) the introduction of non-linearity and (b) the addition of a multiplicative $W$ in the tunable component. Without an ablation study isolating these changes—such as comparing (a) LoRA ($y=(W_0+BA)x$), (b) Non-linear LoRA ($y=(W_0+\sigma(B)A)x$), (c) Multiplicative LoRA ($y=(W_0+W_0BA)x$)—it is difficult to pinpoint the main contributors to the observed performance improvements. Including such comparisons would enhance the comprehensiveness of the analysis.

2. For the theoretical analysis, it may be worth considering that in many models, particularly transformers, $\sum (d^l_1+d^l_2)$ can often be approximated as $2\sum (d^l_2)$, as sequential layers often satisfy $d_1^{l-1} = d_2^l$. From this perspective, NEAT might not offer significant theoretical advantages in expressivity as initially suggested.

### Questions
1. Adding non-linear activations and additional layers is likely to increase runtime and the number of FLOPs during training. Could the authors provide an analysis of the typical runtime and computational overhead associated with NEAT?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper presents NEAT (Non-linear Efficient Adaptation Technique), an innovative approach for efficient fine-tuning that leverages non-linearity to enhance model performance. The core argument is that traditional alternatives, such as LoRA weight updates, face limitations in effectively capturing complex non-linear relationships within data, thus impacting optimization. Experimental results demonstrate that NEAT achieves superior performance compared to these baseline methods.

### Strengths
- The paper addresses an important and timely application, which holds substantial interest for the research community.
- The writing is clear, the organization is logical, and the flow of ideas supports readability.
- Experiments across multiple datasets underscore NEAT’s superior performance over baselines.
- The proposed method appears straightforward to implement, making it accessible to a wider audience.

### Weaknesses
While this paper presents an intriguing approach, it suffers from a lack of strong motivation. The premise of the paper is that introducing non-linearity will improve the model’s ability to capture complex relationships, leading to more efficient fine-tuning and improved optimization. However, non-linearity is inherently captured within the layers of a neural network. The need for additional non-linearity at this stage is not well justified. For example, Equations (3) and (4) appear linear, which raises questions about the actual contribution of non-linearity in this context. In my view, Section 4.2 does not provide adequate motivation or justification for the approach.

Additionally, the theoretical analysis in Section 5 has limitations. Upon reviewing the proof, the main issue is that it simply establishes an equivalence between a LoRA update and a single-layer factorization using a ReLU function. Specifically, it (1) demonstrates equivalence, not a lower bound as shown, (2) does not extend beyond a single-layer factorization with ReLU, and finally (3) even if this inequality holds, it merely indicates that the loss on the training set could be lower than that achieved by LoRA, which operates with a lower rank. This, however, *does not* provide any insights into generalization, which is ultimately what we should be concerned with. Consequently, the conclusions drawn from this proposition remain unclear and seem limited in their broader applicability.

The method also appears sensitive to the choice of activation function and rank. Unlike LoRA, which requires only rank selection, NEAT’s additional requirement to choose an appropriate non-linearity introduces complexity. The paper would benefit from a more thorough discussion on this aspect.

A stronger argument could be made by exploring, for instance, the relationship between parameter updates and the model’s Taylor expansion when non-linearity is incorporated. The authors may wish to consider discussing related works, such as the concept of Neural Redshift [*], to provide a more compelling rationale. In its current form, the paper lacks consideration of alternative activations, which could strengthen the analysis.

Minor Comments

- Figure 1: The caption could be more descriptive, explaining the figure’s notations and providing a clearer overview of the proposed NEAT framework.
- Equation 3: The term L requires explicit definition, along with its parameters.

[*] Teney et al., “Neural Redshift: Random Networks are not Random Functions,” CVPR 2024

### Questions
1. Could you clarify the benefit of non-linearity in the context of parameter-efficient fine-tuning (PEFT)?
2. What does the proposition in Section 5 aim to establish?
3. How did you select the activation functions used in the experiments, and are there insights on choosing among them?
4. What outcomes result from using alternative activation functions such as sigmoid or tanh, which are not explored here?
5. A sensitivity analysis on the hyperparameters for the sinusoidal function and other activations would be insightful. Is this analysis available?

### Soundness
1

### Presentation
2

### Contribution
3
