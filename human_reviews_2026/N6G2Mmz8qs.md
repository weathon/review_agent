# ESNv2: Resurrecting Reservoir Computing in the Deep Learning era

- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
Reservoir Computing (RC) has established itself as an efficient paradigm for temporal processing, yet its scalability remains severely constrained by the necessity of processing temporal data sequentially. In this work, we revisit RC through the lens of structured operators and state space modeling, introducing Parallel Echo State Network (ParalESN), a framework that enables the construction of efficient reservoirs with diagonal linear recurrence in the complex space that can be parallelized during training. We provide a theoretical analysis demonstrating that ParalESN preserves the Echo State Property and the universality guarantees of classical Echo State Networks while admitting an equivalent representation of arbitrary linear reservoirs in the complex diagonal form. Empirically, ParalESN attains comparable predictive accuracy to traditional RC on memory and forecasting benchmarks, while delivering substantial gains in training efficiency. On 1-D pixel-level classification tasks, the model achieves competitive accuracy with fully trainable networks, reducing computational costs and energy consumption. Overall, ParalESN offers a promising, scalable, and principled pathway for integrating RC within the deep learning landscape.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work introduces ESNv2, revisits ESN with a diagonal complex-valued recurrence to enable parallelism, but offers limited real innovation beyond reinterpreting existing state-space and recurrent ideas. 

The main contribution includes: 1. using a diagonal complex-valued linear recurrence that allows parallel computation. 2. adding a nonlinear mixing layer for expressivity while keeping the readout as the only trainable part.

### Strengths
1. This work has a clear motivation. The author figure out the the lack of parallelism in RCs and attempt to address it.

2. This work is well structured and easy to follow.

### Weaknesses
Although the method achieves faster training by eliminating sequential dependencies, this improvement comes at the expense of reduced model expressiveness, since the diagonal recurrence structure limits the network’s ability to capture complex temporal interactions.


The main problem is that the experiments only demonstrated mainly on simplistic synthetic benchmarks such as memory and forecasting tasks; moreover, comparisons with modern deep models like Transformers or LRUs appear shallow and unconvincing, as ESNv2 lacks the flexibility and scalability required for real-world applications. 

Since the author has been benchmarking against the concept of SSM, I suggest the author test the real capability on text datasets.

Therefore, I believe this work does not meet the acceptance criteria

### Questions
1.  Please use more solid and credible experiments to demonstrate the effectiveness of the proposed solution. The currently used dataset is too old and too small. It's even a benchmark Lorenz96 published in 1996. While the largest classification dataset is MINST. This cannot represent the latest research progress at all.

2. These mini datasets cannot demonstrate the efficient and parallelized effects that the author claims to have proposed. In the current situation, I suggest the author consider ImageNet[1] as a baseline for classification task and PILE[2] for sequence modelling. 


[1] Deng, Jia, et al. "Imagenet: A large-scale hierarchical image database." 2009 IEEE conference on computer vision and pattern recognition. Ieee, 2009.

[2] Gao, Leo, et al. "The pile: An 800gb dataset of diverse text for language modeling." arXiv preprint arXiv:2101.00027 (2020).

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose a new type of RC architecture that can have multiple layers and can process sequence data in parallel. They show the new architecture, ESNv2, can achieve comparable performance to ESN on many tasks while being computationally more efficient.

### Strengths
The idea of combining classical dynamical systems-inspired learning and contemporary large-scale modeling to bring reservoir computing into the deep learning regime is intriguing. If successful, it has the potential to scale up reservoir computing to tackle tasks that were previously out of reach. The problem is well motivated and the paper is well written.

### Weaknesses
* ESN and RC are most commonly used in time-series forecasting tasks. On this crucial task, ESNv2 does not improve on existing architectures in terms of performance.
* The benchmark on forecasting focuses on three very simple systems (Lorenz96, Mackey-Glass, NARMA), which may not be representative enough to draw robust conclusions about the forecasting capability of ESNv2 in general.
* The theoretical results seem to be direct applications of existing results.
* Things like the fading memory property are not defined.

### Questions
* Why is ESNv2 better than ESN at certain tasks (e.g., 1-D pixel-level classification) but not other tasks (e.g., forecasting chaotic systems)?
* One fundamental limitation of the traditional ESN is that the size of the reservoir cannot be scaled to billions of parameters due to the poor scaling of matrix inversion used in the training process (e.g., Ridge regression). Does ESNv2 address this limitation in any way? Is it possible to have a much larger reservoir in ESNv2 than in ESN?
* Figure 5 seems to be partially incompatible with Figure 3. Why do ESN and ESN (deep) have high efficiency in Figure 5 but low efficiency in Figure 3?
* It was mentioned that "Observe that even ESNv2 (deep), despite consisting of multiple reservoir layers, trains faster than a traditional, shallow ESN consisting of just one layer." My understanding is that both ESNv2 and ESN train through Ridge regression. So how does ESNv2 train faster than ESN with the same number of trainable parameters (i.e., when the dimension of the regression problem is the same)?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The present manuscript proposes a structured method of Reservoir Computing, called ESNv2, that parallelize the input processing maintaining the Echo State Property, provided that the spectral radius of the diagonal weights is bounded, as proved in Theorem 1. 
Moreover, for every ESN an equivalent ESNv2 in terms of expressivity can be found. 
Evaluation on benchmarks is provided, comparing the proposed model with the traditional ESN and with popular recurrent models such as LSTM and Transformers.

### Strengths
The paper is well organized and easily readable.
The proposed model is supported by a sound, although simple, theoretical characterization (Theorem 1 and Proposition1). 
The metrics evaluated in the experimental framework give a broad overview of the performance of the proposed model, also comparing with deep learning alternatives such as LSTMs and Transformers.

### Weaknesses
- Structured transforms in Reservoir Computing have already been proposed and investigated [1,2], but the present manuscript is totally missing a part of literature review in this regard, and therefore it lacks of a consequent comparison , in terms of performances and computational cost, with these structured methods. 
- at line 99 it is claimed that the weight matrices "are generally sampled from a uniform distribution"; the claim lacks of bibliography references; moreover, in earlier works [3], it is suggested that the weights are initialized from Gaussian distributions;
- A reference and a consequent discussion to standard Deep Reservoir Computing is missing [4]
- The title, in my opinion, is disregarding the huge recent literature of works that are considering reservoir computing as a sound recurrent alternative to deep (backpropagation-trained) learning models for what concern hardware implementations [5]. 
Minor comments:
- In the proof of Theorem 1, there is a B in place of what it should be W_{in}; I believe it is so, because otherwise the proof wouldn't work.

[1] Dong, J., Ohana, R., Rafayelyan, M., & Krzakala, F. (2020). Reservoir computing meets recurrent kernels and structured transforms. Advances in Neural Information Processing Systems, 33, 16785-16796.
[2] D’Inverno, G. A., & Dong, J. (2025). Comparison of Reservoir Computing topologies using the Recurrent Kernel approach. Neurocomputing, 611, 128679.
[3] Verstraeten, D., Schrauwen, B., d’Haene, M., & Stroobandt, D. (2007). An experimental unification of reservoir computing methods. Neural networks, 20(3), 391-403.
[4] Gallicchio, C., Micheli, A., & Pedrelli, L. (2017). Deep reservoir computing: A critical experimental analysis. Neurocomputing, 268, 87-99.
[5] Gallicchio, C., & Soriano, M. C. (2025). Hardware friendly deep reservoir computing. Neural Networks, 108079.

### Questions
In view of what already said, my suggestion for the authors are listed as follows:
- the authors should integrate a substantial comparison with already existing structured transforms for RC in terms of formulation, theoretical guarantees and experimental validation;
- the authors may try to validate experimentally the performances in correspondence of Gaussian weight initialization;
- I would suggest either to revise the title, or to integrate the manuscript with a convincing argument to support the current title;
- all the missing references must be integrated in the manuscript.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents a framework, ESNv2, which introduces the diagonal linear recurrence in the complex space into traditional RC systems. The paper also provides a theoretical analysis and empirical validation, demonstrating the model's efficiency and competitive performance compared to both classical RC and some deep learning models (LSTM, Transformer, LRU) on sequential MNIST tasks.

### Strengths
1. The introduction of diagonal linear recurrence in the complex space is a contribution to RC that allows for parallelization during training.
2. The paper includes a solid theoretical foundation, proving that ESNv2 preserves the Echo State Property (ESP) and universality guarantees, which strengthens its credibility.
3. The paper is well-structured, with clear explanations and logical flow from theoretical concepts to practical experimentation.

### Weaknesses
1. The name "ESNv2" does not adequately reflect the core innovations of the model. A more descriptive name that captures the essence of the diagonal linear recurrence and its parallelization capabilities would enhance clarity and impact.
2. While the paper compares ESNv2 with various models, a more detailed comparative analysis with a wider range of state-of-the-art deep learning models, particularly in terms of specific applications, could enhance the discussion.
3. The work does not deeply survey and discuss related works on ESN variants and Deep ESN, which could provide valuable context and highlight the novelty of ESNv2.
4. The absence of a comparison with Mamba and related state space models is a notable gap. Including this comparison would enrich the findings and establish ESNv2's position more clearly within the landscape of contemporary models.
5. The paper does not discuss how key parameters from traditional ESNs, such as input scaling and spectral radius, are set and their influence on the new model (missing in the methodology section). A detailed examination of these parameters would improve understanding of ESNv2's behavior and performance.
6. The paper lacks a comparison of ESNv2's performance on real-world time series prediction tasks. Adding such analysis would provide practical insights into the model's applicability and effectiveness in real-world scenarios.
7. The method of hyperparameter tuning is mentioned, but more details on the specific impact of different hyperparameters on performance could be beneficial for reproducibility and practical implementation.

### Questions
In the part of Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
