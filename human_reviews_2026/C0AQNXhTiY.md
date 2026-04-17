# Memory Determines Learning Direction: A Theory of Gradient-Based Optimization in State Space Models

- Decision: Reject
- Scores: 4, 2, 4, 8

## Abstract
State space models (SSMs) have gained attention by showing potential to outperform Transformers.
However, previous studies have not sufficiently addressed the mechanisms underlying their high performance
owing to a lack of theoretical explanation of SSMs' learning dynamics.
In this study, we provide such an explanation and propose an improved training strategy.
The memory capacity of SSMs can be evaluated by examining how input time series are stored in their current state.
Such an examination reveals a tradeoff between memory accuracy and length,
as well as the theoretical equivalence between the structured state space sequence model (S4) and a simplified S4 with diagonal recurrent weights.
This theoretical foundation allows us to elucidate the learning dynamics, proving the importance of initial parameters.
Our analytical results suggest that successful learning requires the initial memory structure to be the longest possible
even if memory accuracy may deteriorate
or the gradient lose the teacher information.
Experiments on tasks requiring long memory confirmed that extending memory is difficult, emphasizing the importance of initialization.
Furthermore, we found that fixing recurrent weights can be more advantageous than adapting them
because it achieves comparable or even higher performance with faster convergence.
Our results provide a new theoretical foundation for SSMs and potentially offer a novel optimization strategy.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper develops a theoretical framework explaining how memory determines the learning dynamics of state space models. By analyzing how input sequences are encoded in model states, the authors show that the memory function can quantify a model’s capacity to store and utilize past information. They prove the theoretical equivalence between the structured state space sequence model and its diagonalized variant and demonstrate that the initial eigenvalue configuration—which governs memory length—is crucial for successful training. Their analysis reveals that longer initial memory, even at the cost of accuracy, leads to more effective gradient-based learning. Building on this insight, they propose a training strategy with fixed recurrent weights, which preserves the optimal memory structure, reduces overfitting, and accelerates convergence. Experiments on the Long Range Arena benchmark validate the theory and highlight the practical benefits of memory-aware initialization and training.

### Strengths
This paper makes a clear and original theoretical contribution to understanding the learning dynamics of state space models. The authors rigorously define memory capacity through the lens of the memory function and use it to derive analytical insights into gradient-based optimization—a direction that has not been formally established in prior work. The study is conceptually elegant and technically solid, offering a mathematically grounded explanation of why initialization plays a decisive role in successful training.

It is particularly interesting that the paper reemphasizes the importance of initialization, echoing but also extending the intuition behind HiPPO by connecting it explicitly to gradient flow and memory preservation. The proofs are logically consistent and supported by well-designed empirical validation on long-sequence benchmarks. The exposition is clear and well-organized, making complex ideas accessible.

### Weaknesses
While the theoretical framework is rigorous and illuminating for linear or non-gated state space models, it does not generalize easily to gated architectures such as Mamba or Gated DeltaNet. In these newer models, the system dynamics—and therefore the eigenvalues—depend on the input, breaking the assumption of fixed linear recurrence that underlies the present analysis. Extending the theory to handle input-dependent dynamics remains a key open challenge. Developing a suitable mathematical framework to describe and analyze such cases would be essential for broadening the applicability of the paper’s results to the full family of modern state-space architectures.

### Questions
It is important to discuss the performance gap between S4 and mamba / GDN (because gated delta net has been adopted in large scale open source models.) There should be some discussion why GDN is better than S4.

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
Deep state-space models have received significant interest recently due to their empirical performance. However, their learning dynamics are still poorly understood theoretically. This paper aims to bridge this gap by investigating them from the perspective of the memory function. This approach enables highlighting the importance of initialization for successful learning of these architectures. The theoretical results are complemented with empirical analysis of the memory function for different types of initializations, as well as results emphasizing the benefits of freezing the SSM eigenvalues in terms of learning stability and thus learning performance.

### Strengths
- Learning dynamics of SSMs are a complicated process; gaining understanding of them is an important direction that this paper pursues.
- The memory function perspective the paper leverages is in principle an interesting lens through which to look at SSMs.
- The empirical results showing the stability benefits of RC on SSMs are interesting.

### Weaknesses
The theory introduced in the paper reads to me as sophisticated mathematics to arrive at conclusions that could be simply derived from the vanishing gradient perspective: initialization is indeed critical in RNNs to ensure that correlations between distant time steps can be learned. It is unclear to me that the theory adds anything beyond that.

To my understanding, this may occur because previous results from the literature are misrepresented. While I found many inaccuracies in citations of previous work, I'll highlight the most important ones below:
- **The problem the authors study is the vanishing gradient problem**. In lines 461-462, the authors write "Vanishing gradients may not occur because the only case in which G(y) = 0 […]". Vanishing gradients are not about the total gradient being close to 0, but about error signals transmitted over τ time steps decreasing exponentially with τ.
- **The benefits of diagonality are theoretically relatively well understood**. In line 50, it is written that "theoretical explanation justifying [diagonality] is still lacking". This is not accurate. The S4D and LRU papers argue about the (almost sure) equivalence of these classes in terms of representation. Orvieto et al. 2024 shows the benefits of complex-valued over real-valued diagonal recurrence in terms of reachability by gradient descent. Zucchet & Orvieto 2024 shows the optimization benefits of the normalization/reparametrization that diagonality enables. While precise understanding of the learning dynamics remains poorly understood, the existing understanding already enables explaining most (if not all) of the theoretical conclusions of the paper.

### Questions
The results reported in Table 1 for S4D variants sometimes differ significantly from those reported in the S4D paper, bridging the gap with the RC results. Can the authors elaborate on that?

In line 312 it is mentioned that only the final weights are learned in the RC experiments, whereas line 465 says that 10% of the parameters are not learned with RC (which suggests only freezing eigenvalues and learning the rest of the network). Which one is correct?

Could the authors elaborate on why equation 5 holds? The left term is an infinite sum of terms that are smaller than 1; why is that smaller than $N$?

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents a theoretical analysis of gradient-based learning in State Space Models (SSMs), using the memory function (MF) and memory capacity (MC) concepts originally developed in reservoir computing (RC).

The authors first restate that the dense S4 SSM layer is theoretically equivalent to its diagonalized version (S4D). They then analyze the MF of diagonal S4 variants and argue that these architectures achieve better reconstruction of delayed inputs than naïvely designed SSMs. They further relate the gradient of the loss with respect to the output parameters to the MF, both in the linear and nonlinear output cases, showing that learning dynamics depend on the initial MF profile. Based on these analyses and empirical observations, they argue that training the recurrent weights is unnecessary and potentially harmful.

Empirical evaluations on the Long Range Arena (LRA) benchmark confirm the theoretical claims: freezing the recurrence (RC-style training) improves convergence speed and generalization, and structured diagonal SSMs (S4Dinv, S4Dlin) outperform simpler baselines.

### Strengths
- Establishes an interesting conceptual link between state-space models and the memory function analyses used in reservoir computing.

- The empirical computation of MF for S4D variants is novel and clearly shows their advantage in preserving long-term dependencies.

- The experiments show evidence that freezing the recurrence can improve stability and generalization.

### Weaknesses
- **Lack of clarity in the theoretical part.** The theoretical statements are all expressed in dense natural language, making it hard to isolate the main claims and their assumptions. There is a lack of formal structure (e.g., lemmas or theorems) that would help organize the theoretical contributions. In particular, $\alpha_\tau$, a key quantity in the analysis, is never formally defined. It is unclear whether it corresponds to the classical convolution kernel of the SSM or to another object. It is introduced (L172) after a citation to *Dambre et al.* (2012), but the precise result being referenced from that work is not specified.  

  - **S4D is equivalent to S4.** The assumptions underlying this claim are not discussed. It seems to rely on the uniform i.i.d. input assumption, which should be made explicit and justified. Moreover, according to the empirical results reported in the paper (and in previous literature on S4 and its diagonal variants), S4 outperforms S4D. It would therefore be important to clarify under which conditions this theoretical equivalence is supposed to hold. The authors also state that “since the MF $M[u_{t-\tau}]$ is statistically equivalent to the norm of $\alpha_\tau$” (L176) without citing where this result originates from.  

  - **Initialization for successful learning.** The derived gradient formula (Eq. 12) is not clearly connected to the final claim that one should prioritize MF profiles with longer delays (even if their amplitudes are small). If this is a central claim, it could be empirically tested by comparing systems along the Pareto front of MF profiles (accuracy vs. long-delay memory) to verify whether emphasizing long-delay memory indeed improves learning.

- **Fragile link between theory and experiments.** The main empirical claim—that freezing the recurrence improves performance—is only loosely justified by the theory. The explanation of an “unintended extension of the MF” (L302) would be stronger if formalized or tested empirically. If the underlying mechanism is instead regularization or overfitting mitigation, the authors should compare against standard techniques (weight decay, dropout, etc.).

- **Reservoir computing motivation.** The appeal to reservoir computing (RC) as motivation for not training the recurrence is weak. In RC, the motivation is computational efficiency (avoiding backpropagation through time), not improved accuracy. Here, since gradients are still computed through the recurrence, this rationale is only partly convincing. A more convincing motivation would be to analyze how to identify and remedy the pathologies that arise when training the recurrence.


Minor issues and typos:
- L086: “It **builts** on continuous-time linear SSMs” → *builds*  
- L117: “The emulation is conducted” → not clear what *emulation* means here
- L174: “using a **constatnt** sequence” → *constant*  
- L199: “index **indentifying** a certain SSM” → *identifying*  
- L214: "here T is the output dimension of SSM" → based on the context, it seems that T is the sequence length, not the output dimension
- L427: “**suggestting** that learning eigenvalues…” → *suggesting*  
- L466: “**Altough** the reduction is small” → *Although*  
- L632: “uniform **randomo** distribution” → *random*  
- L674: “one SSM **lyaer**” → *layer*  
- L683: “numerical **unstability**” → *instability*  
- L698: “structured eigenvalues **indicates** significantly more…” → plural agreement → *indicate*

### Questions
1. **Scope of gradient analysis**  
   The theoretical analysis focuses on gradients with respect to the output-layer parameters, even though the models are trained with full backpropagation through the recurrence. Why not also analyze gradients with respect to the **input** weights, which could reveal how input encoding interacts with the memory structure? Do you expect such an analysis to yield additional insights?

2. **Effect of regularization tuning**  
   Did you test whether the reported improvements with frozen recurrent weights persist when regularization hyperparameters (e.g., weight decay, dropout) are more carefully tuned in the fully trainable setting?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper presents an analysis of the learning dynamics of (non-selective) state-space models like S4. They analyze the gradient of an SSM to show that the model's "memory function" (MF) multiplies a key term in the gradient calculation. If a model's MF is zero for a long time delay at initialization, the corresponding component of the gradient will also be zero, making it impossible to learn the long-range dependency. They argue that these initial weights are then of critical importance; training them can "destroy" the memory (but also possibly expand it?). They propose a "reservoir computing" setting (RC) instead, where the recurrent weights/eigenvalues are kept fixed throughout training. They show this actually works better than fully training the model, converging faster, reaching lower loss, and having better validation accuracy.

### Strengths
- Strong theoretical justification for why the initialization of SSMs is so critical: the finding that the MF multiplies the teacher information in the gradient seems important and relatively intuitive and moves beyond the usual vanishing/exploding arguments
- The RC setting is non-obvious and compelling given the analyses
- The empirical support is quite strong for the particular problem of modeling long sequences (in synthetic settings); that it fails with random initializations provides a natural additional baseline. Generally nice experiment design.
- The proof of equivalence between S4 and S4D seems like an important contribution

### Weaknesses
- The analyses are for linear SSMs like S4/S4D and cannot be easily extended to more practical models like Mamba (but I suspect the contribution is relatively transferrable)
- Performance degradation on the Listops task that seems to be unexplained

### Questions
The idea that the initialization is critical for ensuring the gradient as non-zero signal for long-range dependencies seems related to other empirical findings such as [0], which initializes Mamba's $A$ matrix for "full history" to improve recall and copying performance. Do the authors think that their theory could help explain why this works, even if the particular math doesn't apply to the Mamba setting?

Do you have any other speculation on if this might apply to larger-scale selective SSMs as well? (To be clear, I don't think larger-scale experiments are necessary to warrant acceptance; the paper seems well-scoped.)

Did you attempt to find any more optimal initializations, beyond the additional random/linear schemes?

[0] Trockman, A., Harutyunyan, H., Kolter, J. Z., Kumar, S., & Bhojanapalli, S. (2024). Mimetic initialization helps state space models learn to recall. arXiv preprint arXiv:2410.11135.

### Soundness
3

### Presentation
4

### Contribution
4
