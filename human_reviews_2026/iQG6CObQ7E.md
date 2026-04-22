# On the Scaling Theory of Multi-Layer Transformers

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 6, 2

## Abstract
The scaling law, a cornerstone of Large Language Model (LLM) development, predicts improvements in model performance with increasing computational resources. Yet, while empirically validated, its theoretical underpinnings remain poorly understood. This work formalizes the learning dynamics of transformer-based language models as an ordinary differential equation (ODE) system, then approximates this process to kernel behaviors. Departing from prior toy-model analyses, we rigorously analyze one-pass stochastic gradient descent (SGD) training for multi-layer transformers on sequence-to-sequence data with arbitrary data distribution, closely mirroring real-world conditions. Our analysis characterizes the convergence of generalization error to the irreducible risk as computational resources scale with data. We derive an excess risk of $\Theta(\mathsf{C}^{-1/8})$ for computational cost $\mathsf{C}$. The theory reveals a phase transition: under specific conditions, the generalization risk's upper bound drops sharply to $\exp(-\mathsf{C}^{1/4})$ before reverting to its original decay rate. This transition delineates three scaling regimes—*classical, over-parameterization, and data-limited*—which we analyze for their impact on scaling efficiency and the emergence of grokking.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper provides a theoretical framework for analyzing the scaling laws in LLMs by modeling the dynamics of a multi-layered transformer as an ODE. Then, using this framework and assuming one pass SGD training on sequence-to-sequence data and kernel behavior of LLMs, bounds are provided for excess loss risk during training. The obtained bound gives an excess risk bound in terms of the compute C, as well as shows a phase transition in the obtained upper bound that depends on the grokking coefficient epsilon.

### Strengths
- Unlike prior work working on toy settings, this work successfully models multi-layered transformers training dynamics using SGD by formulating it using kernels and using NTK. Even though there are assumptions to frame multi-layered transformers as an ODE, the obtained insights that is applicable for large scale transformers is useful
- The upper bound shows phase transition and also links it to the grokking coefficient at initialization, and is also validated empirically.

### Weaknesses
- In Def 5.2, the constraint on w seems too small, also m is too large, scaling exponentially with d. Can the authors please provide evidence from practical multi-layered transformers to check if these are practical assumptions?
- The theory is based on decoder-only transformers (line 71) but all experiments are on vision transformers, which are not decoder only transformers. Can you please clarify if the theory holds for vision transformers and how or provide experiments with decoder only LLMs?
- Scaling rate C^{-1/8} is very slow compared to empirical scaling rate such as in Chinchilla paper. To that end, the phase transition observed in the bound, could that be just a property of the bound and not necessarily LLM behavior (which has different scaling rate than the bound?)
- Grokking coefficient: Could the authors point to initialization schemes that uses the Grokking coefficient, I am not aware of this being a standard practice like Xavier/Kaiming initializations.

### Questions
Please see the weaknesses section for questions

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
2

### Summary
This paper provides the first rigorous theoretical analysis of scaling laws for multi-layer transformer-based language models, establishing convergence guarantees for generalization error with a rate of $\Theta(C^{-1/8})$ where $C$ is computational cost. The authors formalize training dynamics as an ODE system under the Neural Tangent Kernel regime and derive a three-stage scaling theory that transitions from classical scaling to exponential convergence (grokking phenomenon) and finally to data-limited regime. Experimental validation on image classification tasks demonstrates alignment between theoretical predictions and empirical scaling behavior.

I am not an expert for such a theoretical paper.

### Strengths
1. The paper builds on multi-layer transformers that is not toy setting as previous works.
2. This paper has a good and rigorious theory induction. Moreover, this theory explains grokking phenomenon to some degree. Some empirical experiments are also conducted for validation.

### Weaknesses
1. The proven rate of Θ(C^{-1/8}) is relatively slow compared to empirical scaling laws (e.g., Kaplan et al. 2020 report power laws closer to $C^{-0.5}$). Moreover, I think the coefficient here (-1/8)  is variable with dataset, experimental settings in real-world LLM pretraining. How would the coefficient be fixed?

2. Some settings could be still far away from real-world LLM training, and I think some of them including
- a. AdamW vs. SGD optimizer
- b. Positional Encoding: RoPE vs. NoPE
- c. Pretraining on very large and diverse datasets vs. Finetuning on small datasets like MNIST, CIFAR-10/100

I know it's hard and this paper has advanced one step. But these inconsisitency could still lead to unexpected gaps between theory and real-world practice.

### Questions
See above please.

### Soundness
3

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
4

### Summary
The paper studies the scaling laws of attention based models extending the analysis in the kernel regime calculations from the deep networks. It derives a

### Strengths
The paper derives the convergence and scaling laws in the kernel regime for transformers, it extends the convergence and generalization from the deep networks using NTK and derive kernel based bounds and rates.  

The works goes beyond toy models for laying foundations of large language models. 

With certains assumptions on the initial scales of the model, width of the MLPs, the paper dervies a scaling law for the convergence of excess risk.

### Weaknesses
The lazy regime does not correspond to the practical learning setup which operated in rich or feature learning regime. Hence, the results derived are not broadly applicable. 

It would be nice to get more empirical validation for the scaling law the paper derives beyond vision transformers.

### Questions
Q1) I do not completely understand the distinction between the over-parameterized and data limited regimes, what is the precise difference

Q2) How is the convergence speed exponential in the over-parameterized regime, isn't it still bounded by $O(1/C^{1/8})$, also it is hard to decipher how this phase corresponds to grokking and faster generalization speed.  
 
Q3) In the proof sketch of theorem 6.1, the authors cite Lemma 11 of Schmidt-Hieber 2017 which do not exist, this citation needs to be revisited.

### Soundness
3

### Presentation
2

### Contribution
2
