# COSMOS: A Hybrid Adaptive Optimizer for Efficient Training of Large Language Models

- Avg Score: 4.80
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4, 4

## Abstract
Large Language Models (LLMs) have demonstrated remarkable success across various domains, yet their optimization remains a significant challenge due to the complex and high-dimensional loss landscapes they inhabit. While adaptive optimizers such as AdamW are widely used, they suffer from critical limitations, including an inability to capture interdependencies between coordinates and high memory consumption. Subsequent research, exemplified by SOAP, attempts to better capture coordinate interdependence but incurs greater memory overhead, limiting scalability for massive LLMs. An alternative approach aims to reduce memory consumption through low-dimensional projection, but these methods lose the gradient information in the residual space, resulting in less effective optimization. In this paper, we propose COSMOS, a novel hybrid optimizer that leverages the varying importance of eigensubspaces in the gradient matrix to achieve memory efficiency without compromising optimization performance. The design of COSMOS is motivated by our empirical insights and practical considerations. Specifically, COSMOS applies SOAP to the leading eigensubspace, which captures the primary optimization dynamics, and MUON to the remaining eigensubspace, which is less critical but computationally expensive to handle with SOAP. This hybrid strategy significantly reduces memory consumption while maintaining robust optimization performance, making it particularly suitable for massive LLMs. Numerical experiments on various datasets and transformer architectures are provided to demonstrate the effectiveness of COSMOS.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces an efficient optimizer for pre-training LLMs by combining two previously successful optimization methods. Specifically, it applies SOAP to the leading eigenspace of the gradient matrix and Muon to its orthogonal complement. This hybrid approach is claimed to yield a more memory-efficient and faster optimizer than SOAP, while also outperforming Muon. The authors validate their method through LLM training experiments on standard benchmarks, demonstrating improvements over baselines.

### Strengths
- The idea of applying SOAP to the leading eigenspace and Muon to its orthogonal complement is useful and promising. 

 - Leverages the strength of a stepwise effective yet computationally expensive optimizer by restricting its use to a lower-dimensional subspace, and hence improving memory and computational efficiency. 

- This approach could inspire further exploration of hybrid methods that combine different optimizers using their principle. 

- The paper presents pretraining experiments across standard model sizes and datasets, demonstrating improvements of the proposed method.

### Weaknesses
- The main concern is that the paper does not provide sufficient wall-clock time experiments, making the claimed advantage over Muon questionable. The only such experiment involves a 1B LLaMA model. However, due to the large gradient accumulation, I think the forward and backward passes dominate the computation time, making the optimizer’s step time negligible. As a result, the reported plot may effectively reflect only a stepwise comparison rather than a true wall-clock evaluation.

### Questions
- (1) and (2) in the introduction are somewhat vague. In (2), you mention a limitation related to storing adaptive learning rates —what do you mean here? It would be more precise to state that the optimizer stores both the first and second moments, which increases memory consumption.
- On the line 167 you say G-S procedure. Do you implement your own QR with G-S? Presumable, you use torch.qr here which doesn't use G-S. 
- Line 220, regarding Muon's scaling, I think the right scaling for learning rate transfer is $\sqrt{\frac{m}{n}}$. Am I missing something?
- In table 1 would be good to see COSMOS memory usage dependence on $r$ as well.
- In Figure 1 and subsequent experiments, why do you observe SOAP to be worse then COSMOS. Isn't it counterintuitive?
- I suspect forward/backward takes most of the time in Table5. Can you separate out the optimizer step time in that plot? That will give a fair comparison. 
- Can you provide wall-clock plots for nano-gpt experiments?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a new optimizer named COSMOS, which applies the update of SOAP on leading eigensubspace of parameters and applies Muon for the rest. The motivation for the method is to reduce the memory cost of storing second moment of SOAP. Numerical results show that COSMOS is able to achieve comparable convergence as SOAP in the task of pre-training LLMs under Llama and GPT-2 models, while costs less memory.

### Strengths
* The proposed method avoids the storage of second moment in SOAP and is memory-efficient.
* The method achieves comparable convergence speed as SOAP in pre-training Llama/GPT-2 models of different scales, and is shown to be robust to newly introduced hyper-parameters, e.g. $\gamma$.

### Weaknesses
* The main motivation for the design of COSMOS is not well presented. It is unclear why replacing the top eigenspace's update in Muon with SOAP will accelerate the convergence. It has been shown that SOAP exhibits comparable in-iteration convergence as Muon [1].

* The acceleration of COSMOS over Muon is marginal. Given COSMOS' higher memory cost and training FLOPs, the practical benefit is a bit marginal compared to Muon.

[1] Muon: An optimizer for hidden layers in neural networks

### Questions
* Could the authors provide more comments on why setting $\gamma=\frac{\eta}{\eta_0}$? The $\gamma$ determines the weight of Muon's update; it is unclear why this is related to the learning rate in embedding/language modeling head layers.

* Given the fact that COSMOS is faster than Muon, it would be interesting to see the effect of applying SOAP on different range of eigenspaces rather than the top eigenspace, which helps understanding the algorithm more comprehensively.

### Soundness
2

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
4

### Summary
This paper proposes COSMOS, a hybrid optimizer that combines the strengths of SOAP and MUON for large language model (LLM) training.  
The core idea is to decompose the gradient matrix into leading eigensubspace and residual subspace:  
- A SOAP-like update is applied to the leading subspace to capture major optimization dynamics.  
- A MUON-like update is used for the residual subspace to reduce memory and computation costs.  
This approach greatly reduces memory usage (about 19% of SOAP) without noticeable performance loss.  

Experiments on LLaMA, GPT-2, and NanoGPT show that COSMOS outperforms MUON in token and memory efficiency while achieving performance comparable to SOAP.

### Strengths
Innovative idea: A hybrid low-rank optimization framework that balances performance and resource cost.

Comprehensive experiments: Covers multiple models, datasets, and baselines.

Detailed analysis of memory and efficiency: Provides quantitative evidence of significant savings.

Strong robustness: Insensitive to rank and hyperparameter variations.

### Weaknesses
Insufficient methodological details:
Lacks analysis on rank selection and the dynamics of subspace changes during training.

Weak theoretical explanation:
Unclear whether subspace variation causes instability in the optimization direction.

Limited experimental analysis:
Lacks mechanistic or spectral analysis to explain why COSMOS outperforms SOAP.
The fundamental advantage of COSMOS over SOAP and MUON remains unclear, and the observed performance degradation with higher ranks is only heuristically explained.

### Questions
How is the rank determined? Is it based on an energy ratio or an information-based criterion?

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
See below

### Strengths
See below

### Weaknesses
See below

### Questions
The paper proposed a new pre-training optimizer called COSMOS, which has similar performance as SOA  but has cheaper memory requirement than SOAP. The key idea is to compress the SOAP’s second-moment matrix using its dominant eigenspace and track it in a low-dimensional subspace.

The idea is interesting. The reduction in the complexity of SOAP is potentially useful.   The presentation is clear. 

I have concerns for several experimental settings adopted in the current script, which I think might not fully reflect the performance of the proposed method. I suggest the following setup changes and ablations. 

1. **Non-zero weight decay.** The authors use weight dacay = 0. What if we use non-zero weight decay such as weight decay = 0.1? The choice of weight decay can be crucial to the optimizer behaviors, as pointed out by recent benchmark [1].

2. **More training tokens.** For Fig 1: 5k step (5B tokens) is roughly 1x chinchilla. How would the methods perform under an overtrained setting, such as 4x or 8x chinchilla in the recent work [1]? Will the advantage of COSMOS degenerate, or will it be surpassed by Muon or other methods? It would be helpful to see a new version of Fig. 1 under 4x or 8x chinchilla setting. 

3. **More ablation on batchsize.** Matrix-based methods usually can afford a larger batchsize than Adam-type methods.  I think the ablation on batch size is insufficient in the current script. It is possible that the optimizer ranking becomes quite different under different batchsize settings. It is also possible that the proposed method performs even better in a larger batch setting. 

4. **Higher-quality datasets.**  C4 is popular,  but the data quality is relatively poor and noisy from the current standard. I think the ablation on the more SOTA dataset is needed. I understand it might be too expensive to re-run all the experiments on Fineweb, but I think at least some ablation on the small-scaled experiments is needed.  For the smallest experiments in the paper (125M model), can the authors provide additional ablation on the dataset by changing the dataset to Fineweb and re-evaluate all the methods?  Will the data quality affect the optimizer rankings? 

I am happy to raise my score if my concerns are addressed. 

[1] Fantastic Pretraining Optimizers and Where to Find Them

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
COSMOS presents an innovative hybrid optimization approach that cleverly combines SOAP and MUON optimizers through eigenspace decomposition. The authors tackle the important challenge of reducing memory consumption in LLM training while maintaining optimization quality. By applying SOAP to the leading eigensubspace and MUON to the remaining space, the method achieves notable efficiency gains with reported memory savings of 82% and training time reduction of 91%.

### Strengths
-- Addresses a critical problem: Memory efficiency in LLM training is a major bottleneck, and this work makes genuine progress

-- Creative solution: The eigenspace decomposition approach shows good intuition about balancing computational cost and optimization quality

-- Promising empirical results: The reported efficiency gains are substantial and could have real practical impact

-- Clear presentation of core ideas: The authors effectively communicate their hybrid strategy

### Weaknesses
-- Theoretical gaps: The paper would greatly benefit from convergence analysis and theoretical justification for the eigenspace splitting strategy. Adding even basic convergence guarantees would strengthen the contribution significantly

-- Limited scope validation: Expanding beyond transformer architectures to CNNs or other models would demonstrate broader applicability

-- Hyperparameter sensitivity: The rank r and weight γ selection needs more principled guidelines - perhaps adaptive schemes could be explored

-- Missing key ablations: Comparing different eigenspace decomposition strategies or investigating the optimal split ratio would provide valuable insights

### Questions
-- Could you provide theoretical analysis showing when COSMOS converges faster than pure SOAP/MUON?

-- Have you explored adaptive rank selection based on the eigenvalue spectrum?

-- Could you test on at least one non-transformer architecture to demonstrate generalizability?

-- Is there a principled way to choose γ without the η/η0 heuristic?

### Soundness
2

### Presentation
3

### Contribution
2
