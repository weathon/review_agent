# NoLoCo: No-all-reduce Low Communication Training Method for Large Language Models

- Decision: Reject
- Scores: 2, 6, 4

## Abstract
Training large language models is generally done via optimization methods on clusters containing tens of thousands of accelerators, communicating over a high-bandwidth interconnect. Scaling up these clusters is expensive and can become impractical, imposing limits on the size of models that can be trained. 
Several recent studies have proposed training methods that are less communication intensive, avoiding the need for a highly connected compute cluster.  These state-of-the-art low communication training methods still employ a synchronization step for model parameters, which, when performed over all model replicas, can become costly on a low-bandwidth network.

In this work, we propose a novel optimization method, NoLoCo, that does not explicitly synchronize all model parameters during training and, as a result, does not require any collective communication. NoLoCo implicitly synchronizes model weights via a novel variant of the Nesterov momentum optimizer by partially averaging model weights with a randomly selected other one. We provide both a theoretical convergence analysis for our proposed optimizer as well as empirical results from language model training.

We benchmark NoLoCo on a wide range of accelerator counts and model sizes, between 125M to 6.8B parameters. Our method requires significantly less communication overhead than fully sharded data parallel training or even widely used low communication training method, DiLoCo. The synchronization step itself is estimated to be one magnitude faster than the all-reduce used in DiLoCo for few hundred accelerators training over the internet. We also do not have any global blocking communication that reduces accelerator idling time. Compared to DiLoCo, we also observe up to $4\%$ faster convergence rate with wide range of model sizes and accelerator counts.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper try to eliminating global collective communication to reduce the e2e training communication overhead. This can be useful in extreme cases where cross-node network bandwidth is limited.

The authors proposed NoLoCo method, which does dynamic routing on PP dimension (similar as dynamic routing in MoE expert tokens), but do need model synchronization on DP dimension. However, loss divergence issue has been observed.

### Strengths
1. The proposed idea is interesting (i.e. doing dynamic routing in PP stages). Also picking PP is practical since it has minimum communication volumes compared to other parallelism strategies. 

2. A detailed convergence analysis in appendix is a plus.

### Weaknesses
1. In general, eliminating global communication will lead to model divergence. In practice, training a model has much worse model serving accuracy, this training itself will be a complete waste of compute resources. Therefore, only try to optimizing/eliminating communcation overhead is not a good strategy. 

2. For experiments, the paper does not mention any details on which LLama vesion is used? e.g. LLama2? llama3? which can be a huge difference on perplexity or loss.

3. the results of Figure 2, it shows there is noticeable and significant loss/PPL gaps between FSDP and proposed NoLoCo strategy, which makes the proposed method not practical in real model training scenarios.  

4. Although the paper has some "proof" of model convergence with dynamic PP routing, as shown in later result (e.g. Table2 and Figure 2), the model divergence issue is quite severe. Therefore, it makes the whole NoLoCo scheme less practical, mainly because the loss divergence issue.

### Questions
1. In Figure 2, why larger models and more PP stages will make PPL differnce less compared wth FSDP?

2. In practice, nowadays it is very rare to see communication bandwidth is less than 1GB/s. So the Table 3 comparison number on 100Mb/s ( 12.5MB/s) and 1Gb/s (125MB/s) are less practical and less convincing. 

3. Theoretically, doing dynamic routing on PP stages will definitely cause loss divergence issue. The high level idea is, difference PP stages of a single PP group are training on different input data at the same iteration, this will definitely introduce errors and wrong momentum.

### Soundness
2

### Presentation
3

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
This paper proposes NoLoCo, a decentralized training method for large language models that eliminates global all-reduce synchronization. Building on DiLoCo, which reduces communication by performing infrequent global synchronizations between local training phases, NoLoCo further minimizes overhead by synchronizing only random pairs of workers using a modified Nesterov momentum optimizer that maintains model stability. Combined with dynamic pipeline routing, this approach enables efficient large-scale training on low-bandwidth or high-latency networks. Experiments on different sizes of LLaMA models show that NoLoCo achieves up to 4% faster convergence than DiLoCo while greatly reducing communication cost.

### Strengths
•	The key innovation of this work is to remove global all-reduce synchronization by introducing local, pairwise averaging.
•	The authors propose a new variant of Nesterov momentum with an additional local averaging term to prevent divergence, supported by theoretical convergence analysis.
•	The authors provide mathematical proofs showing convergence and variance behavior, linking stability to the inner learning rate.

### Weaknesses
•	Only empirically compare NoLoCo against the original DiLoCo baseline, despite acknowledging multiple improved variants.
•	Convergence proof assumes independence between replicas and IID data, which may not hold in real heterogeneous clusters.
•	Largely reused learning rates and batch sizes from prior FSDP studies and fixed outer-loop settings without systematic or sensitivity analysis.
•	The authors may want to separately test pairwise averaging, modified Nesterov, and random routing to show each component’s contribution.

### Questions
•	The paper references multiple improved versions but compares only with the original DiLoCo; including these baselines would better show whether NoLoCo truly advances beyond recent low-communication optimizers. If cannot make the comparison, the authors may want to provide a detailed explanation of why these variants were excluded and how NoLoCo differs from them in design or communication efficiency.
•	The paper integrates three innovations: pairwise averaging, modified Nesterov momentum, and random pipeline routing. However, do not disentangle their effects. The authors may want to conduct ablation studies that isolate each component if possible to clarify which element primarily drives convergence speed, stability, and communication efficiency.
•	NoLoCo introduces several new hyperparameters (like outer momentum, outer learning rate, local averaging strength, subgroup size, and outer step frequency), but the paper provides little analysis of how these parameters influence stability, convergence, or communication cost. Could the authors discuss which hyperparameters are most critical, how sensitive the method is to their choice, and how much tuning effort practitioners should expect compared to standard distributed training?

### Soundness
3

### Presentation
3

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
This paper proposes NoLoCo, a novel optimization method designed to eliminate the need for collective communication during distributed training of large language models (LLMs). Instead of explicitly synchronizing model parameters, NoLoCo achieves implicit synchronization through a variant of the Nesterov momentum optimizer. Specifically, model weights are partially averaged with those of a randomly selected peer, allowing for communication-efficient and scalable model training. The authors provide both theoretical convergence guarantees for the proposed algorithm and empirical experiments demonstrating its efficiency in large-scale language model training.  

While the idea is innovative and the theoretical aspect is solid, the experimental studies are incomplete, especially regarding the ablation and sensitivity analyses needed to confirm the importance of the paper’s key design choices. I will raise my score to positive if the authors address my concerns.

### Strengths
1. The paper provides both theoretical and empirical validation for the proposed optimization method, which strengthens its overall contribution and credibility.  
2. NoLoCo effectively reduces communication overhead compared to state-of-the-art methods, particularly DiLoCo, while achieving faster convergence in language model training.  
3. The idea of replacing global synchronization with randomized local interactions is novel and practically meaningful, offering a potential improvement for large-scale distributed training systems.

### Weaknesses
1. Ablation studies are insufficient. The paper lacks experiments isolating the impact of the “outer optimizer step with modified Nesterov momentum,” which is one of the paper’s core contributions. Specifically, it is unclear how performance would change if the original (unmodified) Nesterov momentum were used instead of the modified version described in Equation (2). Without further ablation to isolate the effect of the modified Nesterov momentum, it remains unclear whether this component is indeed critical to NoLoCo’s improved performance.  
2. The paper omits ablations on critical hyperparameters such as:  
   - The group size (n) in section 3.2, which likely influences both communication cost and model performance (e.g., perplexity).  
   - The number of inner optimizer steps, which differs between NoLoCo (50) and DiLoCo (100) in the main comparisons, but the rationale for this choice is not adequately explained.  
     Including these analyses would provide deeper insight into NoLoCo’s behavior and fairness in comparison.  
3. The implementation details in Section 3 are not sufficiently clear, making it challenging to fully grasp how NoLoCo is practically realized.

### Questions
1. What would happen if the modified Nesterov momentum (Equation 2) were replaced with the original Nesterov momentum? How does this change impact convergence and model performance?  
2. How does group size (n) affect communication efficiency and model perplexity? Can the authors provide an ablation study?  
3. How does varying the number of inner optimizer steps influence both communication cost and performance?  
4. During the outer optimizer step, are the local groups fixed or re-sampled randomly at each round?  
5. Since each step only performs partial synchronization within groups, is there a final global synchronization among all subgroups at the end of training? If not, wouldn’t this result in $\frac{N}{n}$ slightly different model replicas?  
6. In Figure 4, the variable *n* is annotated as “world size,” but in Section 3.2 it denotes “group size.” Are these the same variable or different? The notation may be confusing.

### Soundness
2

### Presentation
2

### Contribution
3
