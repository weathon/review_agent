# FRUGAL: Memory-Efficient Optimization by Reducing State Overhead for Scalable Training

- Decision: Reject
- Scores: 5, 5, 6, 5, 5

## Abstract
With the increase in the number of parameters in large language models, the process of pre-training and fine-tuning increasingly demands larger volumes of GPU memory. A significant portion of this memory is typically consumed by the optimizer state. To overcome this challenge, recent approaches such as low-rank adaptation (LoRA (Hu et al., 2021)), low-rank gradient projection (GaLore (Zhao
et al., 2024)), and block-wise optimization (BAdam (Luo et al., 2024)) have been proposed. However, in all these algorithms, the effective rank of the weight updates remains low-rank, which can lead to a substantial loss of information from the gradient. This loss can be critically important, especially during the pre-training stage. In this paper, we introduce **FRUGAL**; (**F**ull-**R**ank **U**pdates with **G**r**A**dient sp**L**itting),, a new memory-efficient optimization framework. The framework leverages gradient splitting to perform low-rank updates using advanced optimization algorithms (such as Adam), while updates along the remaining directions are
executed via state-free methods like SGD or signSGD. Our framework can be integrated with various low-rank update selection techniques, including GaLore and BAdam. We provide theoretical convergence guarantees for our framework when
using SGDM for low-rank updates and SGD for state-free updates. Additionally, our method consistently outperforms concurrent approaches across various fixed memory budgets, achieving state-of-the-art results in pre-training and fine-tuning
tasks while balancing memory efficiency and perplexity targets.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper introduces FRUGAL. The fundamental idea is that during the backward pass, we will take a subset of parameters (a block) to perform stateful Adam updates and for the rest parameters (with blockwise selection) or the gradient residuals (with low-rank gradient projection), we use stateless signSGD updates. The memory efficiency of FRUGAL is achieved by reducing the optimizer states. The authors provide a convergence rate similar to SGD momentum's usual rate under nonconvex optimization. The authors also perform experiments with Llama pretraining on C4 and RoBerta-base fine-tuning on GLUE tasks. The baselines are primarily Galore and BAdam.

### Strengths
1. FRUGAL's convergence rate is provided and it can recover the rate of standard SGD(M). 

2. The experiment execution is strong and the results are convincing. The hyperparameter details are well disclosed and the implementation is provided.

### Weaknesses
**Major concern**:

1. The idea of FRUGAL is fairly simple (as a combination of signSGD, Adam, and a gradient projector) but the empirical and theoretical support behind FRUGAL is not solid enough. FRUGAL's stateful optimizer is basically either Galore or BAdam. The main contribution is therefore stateless optimizer part (signSGD), and such effectivenss relies on the finding that stateless optimizers are sufficient to optimize most parameters in LLM (linear weight matrices). The authors only provide a single ablation study in Table 3 without further empirical or theoretical insights on the stateless optimizer part. This evidence alone is not convincing enough on an assured generalization to other non-Llama architectures. So it appears to me that the contribution of this paper is insufficient for an ICLR paper. 

2. The motivation (Figure 2) of FRUGAL is that low-rank gradient projection is similar, and random or blockwise selection can cover the whole space. Figure 2 justifies that the top gradient directions across timestep is similar, but *is insufficient to show that random or blockwise selection is always/necessarily better. It is highly likely that after a certain threshold, the role of randomly selected parameters/blocks of parameters have worse performance than top gradient directions. An ablation study on projector type versus stateful optimization density $\rho$ is definitely needed.

**Minor concern**:

1. The presentation of the Algorithm needs to be clearer. It is hard to understand the exact algorithm (which is actually simple) in the first time of reading Algorithm 1 and Section 3.


I consider the first major weakness as critical and I would vote for a borderline reject score at this moment.

### Questions
I don't have other questions. All major weaknesses are listed above.

### Soundness
3

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
This work proposed a new memory-efficient training methods that allows part of the parameters being optimization with optimization states within a compact space while other parameters are optimizated in the original space without optimization states. Results on serveral pre-training and fine-tuning tasks demonstrates the effectiveness of the proposed methods.

### Strengths
- Plenty of experiments are conducted to evaluate FRUGAL where FRUGAL demonstrates significant improvments against GaLore.

- Both empirical and theoretically justification are provided to validate the effectiveness of FRUGAL.

### Weaknesses
- The GLUE benchmarks is little bit outdated, more recent tasks like common-sense reasoning, mt-bench would further improve this work.

- Is there any explanations about which part of the parameters can be directly optimized with SGD type optimizer with other requires adam and why?

- For $\rho=0$ in Table 2, is it equals to fully optimized with SGD? Does it controdict with recent works that demonstrates that transformers can not be effectively optimzied with SGD? [1]

- The concepts of state-full and state-free subspace in line80/82 is hard to understand, it's better to formally define these two concepts. 

- line 192: "Surprisingly, we found that although SVD decomposition delivers an initial boost, subsequent training with random projection yields significant improvements", this sequence make it a little bit confusing whether the "Low-rank Random" in Table 1 is training of entire random projection or first SVD and later random.

- it's better to define the meaning of K in the inputs of algorithm 1, as well as s.


[1] Why Transformers Need Adam: A Hessian Perspective

### Questions
Please refer to the weakness

### Soundness
3

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
2

### Summary
This paper introduces a novel memory-efficient optimization method. Unlike other state-of-the-art approaches, such as LoRA and GaLore, that have low-rank updates, this method maintains a full-rank update structure. The experimental results demonstrate its superior performance, highlighting its potential advantages in both efficiency and effectiveness over competing methods.

### Strengths
**Well-structured Presentation:** The paper is well-structured and easy to follow, with a clear presentation of concepts and methodology.

**Practical Impact:** The method is straightforward to implement and has broad applicability, making it valuable for practical use in various settings.

### Weaknesses
**Lack of Discussion on Limitations:** The paper would benefit from a discussion of the method's limitations and potential failure modes. Addressing these aspects would provide a more balanced view of the approach's applicability and constraints.

**Vague Terminology:** Given the importance of "state-full" and "state-free" in the proposed method, the paper should offer clearer definitions of these terms. Precise terminology is essential to fully understand the mechanics and implications of the approach.

### Questions
**Formal Definitions of Full and Free States:** Could the authors provide formal definitions of "full" and "free" states as used in the method? A clearer understanding of these terms would improve the paper’s theoretical foundation.

**Main Limitations:** What are the primary limitations of this approach? A discussion on the constraints or situations where the method might be less effective would help clarify its scope and potential trade-offs.

**Running Time Comparisons:** Beyond memory efficiency, how does the method’s running time compare to that of other baseline approaches? Performance in terms of speed is crucial for practical deployment, so direct comparisons would provide a more complete picture of the method’s efficiency.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper introduces FRUGAL (Full-Rank Updates with GrAdient spLitting) that reduces memory consumption by splitting gradient updates into two subspaces. A *state-full* subspace is updated using advanced optimization algorithms like Adam, while a *state-free* subspace is updated using stateless and memory-efficient methods like SGD or signSGD. The framework allows for a flexible choice of optimizers and projection methods. FRUGAL achieves state-of-the-art results in pre-training and fine-tuning tasks, outperforming existing memory-efficient algorithms while maintaining a similar memory budget.

### Strengths
-1) The paper presents a novel approach to improving memory efficiency while performing updates using full-rank information. 
-2) The proposed method is flexible, supporting various choices for both stateful and stateless optimizers as well as different projection methods. It offers convergence guarantees for FRUGAL within a specified framework and consistently outperforms existing memory-efficient algorithms, such as GaLore and BAdam, achieving performance levels close to the memory-intensive Adam optimizer. 
-3) Additionally, the paper provides valuable insights into the learning dynamics of transformer models.

### Weaknesses
-1) The paper's structure would greatly benefit from a clearer organization. Currently, some analysis and experimental results appear within the Methods section, which disrupts the logical flow and makes it challenging for readers to follow the methodology. Reorganizing the paper and dedicating specific sections to distinct aspects of the research could significantly enhance readability and impact.

-2) Several notations (e.g., g~) are introduced without proper definitions, which assumes too much prior knowledge from readers. Additionally, concepts like smoothness and unbiasedness are only vaguely referenced and would benefit from clearer definitions. The theory section should be expanded to explicitly define each notation and assumption, as well as to contextualize them within a more general setting relevant to the proposed method.

-3) Including a full-parameter fine-tuning baseline in Table 4 would provide a valuable benchmark, offering a clearer context for evaluating the results.

-4) Definitions for Full-Rank SVD/Random and Low-Rank SVD/Random are scattered across Table 1 and lack clear differentiation. Consolidating these explanations into a concise paragraph would improve clarity and reader comprehension.

-5) Lastly, there are deviations from the primary algorithm, such as using column-wise projection instead of block-wise projection. For completeness, it would be beneficial to include results using the original proposed approach alongside the variations in the experiments.

-6) By solving this issues in the revision, especially following a more structured writing style and lowering the jumps, the paper would definitely level up.

### Questions
- 1) Including more experiments comparing the method with various stateful and stateless optimizers would enhance the paper. 

- 2) Testing models with larger sizes (e.g., 3B and 7B) could further demonstrate the generalizability of the proposed method. 

- 3) Please clarify the reasons for selecting the specific optimizers in the theoretical section. They appear restrictive and differ from those used in the main algorithm. Additional details and guarantees would help generalize this proof. 

- 4) While it’s mentioned that stateless optimizers typically underperform with transformer architectures, the paper doesn’t explain why FRUGAL with $\rho=0$ achieves optimal performance in certain scenarios. Providing more details and comparisons would clarify this.
Expanding the dataset and incorporating diverse architectures could strengthen the argument for FRUGAL's superior characteristics.

### Soundness
3

### Presentation
1

### Contribution
3

---

## Human Reviewer 5

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The work proposes a memory efficient training method called FRUGAL which is essentially a combination of full-rank updates with gradient splitting. The authors partition the parameters and update using advanced optimizers (like Adam) for low-dimensional updates and state-free methods (like SGD or signSGD) for remaining directions. Additionally, the authors provide theoretical convergence guarantees and validate FRUGAL’s effectiveness through experiments on models like LLaMA.

### Strengths
1. The combination of state-free optimizers with advanced ones, like SGD and Adam, for memory efficient training is a novel idea.
2. The  empirical results show that FRUGAL does better than other methods in terms of memory use and perplexity,
3. The paper includes sufficient ablation studies and it helps to see how FRUGAL works in different situations and settings.

### Weaknesses
Line 249 introduces state-free and stateful parameters but could provide more explicit explanation on the selection criteria. Are parameters randomly selected to each category? In that case the assumption is all the parameters are equally important for that iteration. The work could benefit from more detailed study on how to choose the parameters for state free updates. 

The purpose of the density parameter ($\rho$) is not thoroughly explained, especially in relation to zero-density training. Please clarify whether zero-density training implies all parameters are state-free (i.e., trained exclusively with SGD). The selection of $\rho$ is not mentioned in the algorithm as well.

### Questions
GaLore theoretically prove that gradient is low-rank and a study in BlockLLM (https://arxiv.org/pdf/2406.17296)  show that only a few parameters are updated during the training. A few other recent works also seem to suggest that the low rank structure exists in the network. But this paper seems to suggest the opposite. Do you see a space where these two ideas coexist? For example, low rank for certain tasks vs full rank for other tasks? 

Minor:
- Introduce abbreviations for better readability. For example SGD as Stochastic Gradient Descent. 
- Missing references Adam-mini and BlockLLM

### Soundness
2

### Presentation
2

### Contribution
3
