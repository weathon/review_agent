# EDiT: A Local-SGD-Based Efficient Distributed Training Method for Large Language Models

- Decision: Accept (Poster)
- Scores: 6, 6, 5

## Abstract
Distributed training methods are crucial for large language models (LLMs). However, existing distributed training methods often suffer from communication bottlenecks, stragglers, and limited elasticity, particularly in heterogeneous or large-scale environments. Local SGD methods have been proposed to address these issues, but their effectiveness remains limited to small-scale training due to additional memory overhead and lack of concerns on efficiency and stability. To tackle these issues, we propose EDiT, an innovative Efficient Distributed Training method that combines a tailored Local SGD approach with model sharding techniques to enhance large-scale training efficiency. EDiT performs layer-wise parameter synchronization during forward pass, reducing communication and memory overhead and enabling overlap. Besides, EDiT employs a pseudo gradient penalty strategy to suppress loss spikes, which ensures training stability and improves performance. Additionally, we introduce A-EDiT, a fully asynchronous variant of EDiT that accommodates heterogeneous clusters. Building on EDiT/A-EDiT, we conduct a series of experiments to validate large-scale asynchronous training for LLMs, accompanied by comprehensive analyses. Experimental results demonstrate the superior performance of EDiT/A-EDiT, establishing them as robust solutions for distributed LLM training in diverse computational ecosystems. The code is available at Atorch codebase: https://github.com/intelligent-machine-learning/atorch/tree/main/atorch/local_sgd.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposed a distributed training approach which arranges the participating workers into a two-dimensional device mesh, i.e., model replica group and the model shard group. It also introduces a pseudo gradient penalty mechanism that eliminates the significantly anomalous worker and averages and clips the gradients. It also introduces an asynchronous variant of the proposed scheme. Experiments have been conducted to evaluate the performance of the proposed scheme on the distributed training of LLMs.

### Strengths
It is interesting and novel idea of arranging the participating workers into a two-dimensional device mesh, i.e., model replica group and the model shard group. The experimental results also demonstrate the effectiveness of the proposed pseudo gradient penalty mechanism and asynchronous version of the proposed scheme. The presentation of this paper is fair and easy to follow.

### Weaknesses
1. The novelty of this paper is limited. The author claimed that this paper is the first to integrate Local SGD with modern distributed strategies. What does "modern distributed strategies" refers to here is not clear. Is't local SGD a modern distributed strategy. The way to eliminate the significantly anomalous worker and averages and clips the gradients is not a new idea. 
2. The  hierarchical distributed training method on a 2-D device mesh may be problematic. Since each worker within model shard group runs a subpart of the model, it may need to wait for the backward gradients of the same batch before processing the next batch, leading to significant idle time and slowing down the training. 
3. The improvement over the existing work since quite insignificant as shown in Figure 4. The performance of the proposed scheme is only compared to baseline, which is not sufficient to demonstrate its superiority. The author should compare it with a more advance asynchronous scheme, such as following work:
[1] Nguyen J, Malik K, Zhan H, et al. Federated learning with buffered asynchronous aggregation[C]//International Conference on Artificial Intelligence and Statistics. PMLR, 2022: 3581-3607.

### Questions
1. The proposed hierarchical distributed training method on a 2-D device mesh should be better explained, like how workers within a model shard group coordinate with each other. What are the messages transmit between them (should be intermediate results instead of parameters?)
2. Is there any design in the proposed method addressing the concern that each worker within model shard group may need to wait for the backward gradients of each batch from the next worker in the same group? 
3. How this proposed hierarchical distributed training method accelerates the distributed training? It should be evaluated in the numerical results. 
4. In Section 4.5, it seems suggesting that DiLoCo scheme with the pseudo gradient penalty mechanism is the proposed scheme EDiT?

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
This paper introduces EDiT, a distributed training method leveraging local SGD to reduce communication overhead when training LLMs. The method relies on a "2D-grid" graph topology with a total of $K= M \times N$ data parallel models. On highly connected GPUs (e.g., on the same compute node in a cluster), $N$ models are sharded and communicate at each optimizer step. On the other hand, $M$ of these islands of highly connected machines are connected through possibly lower bandwidth communication links (e.g., on separate compute nodes), and leverage the lower communication frequency needs of local SGD to alleviate communication bottlenecks. To counter the destabilizing effect of using lower batch sizes in local SGD, a gradient penalty method is introduced. Finally, an asynchronous version of the method is also introduced to mitigate the effect of stragglers.

### Strengths
* **The novel gradient penalty method seems effective**: In the experiment on the in-house dataset (Fig 4.d), the unstable behavior of standard local-SGD methods is highlighted, and the gradient penalty introduced seems to be an effective solution to alleviate it.
* **Advantages of a proper distributed implementation of a local SGD algorithm are highlighted**: Section 4.3 displays the practical advantages (in terms of throughput) given by a proper implementation of local SGD algorithms using modern distributed methods (such as ZeRO-3/FSDP) when communication links between distributed workers have limited bandwidth.
* **Asynchronous extension**: In the presence of stragglers, the advantage (in terms of throughput) of using an asynchronous extension of EDiT are highlighted in Sec.4.3.

### Weaknesses
* **Claiming novelty on the distributed implementation of a local SGD algorithm seems too strong:** As the sharding strategy itself cannot be claimed as novel (see the ZeRO series, FSDP), the “distributed implementation” cannot either. In fact, apart from the gradient penalty method, the Algorithm seems to be a straightforward and natural way of implementing local SGD for LLM training (using ZeRO-3) and not particularly novel (previous local SGD papers such as CO2 [[Sun et al., 2024]](https://arxiv.org/pdf/2401.16265) seem to already make use of ZeRO for their Autoregressive Language Modeling task, and SlowMo [[Wang et al., 2020]](https://arxiv.org/pdf/1910.00643 ) already consider all GPUs inside a cluster node to be a single local worker in their experiments so that each local worker reaches larger batch sizes through data-parallel). Thus, although having a *detailed* pseudo-code for this implementation is welcome, making it the main contribution of the paper and dedicating so much space for explaining it weakens the paper for me. For instance, as the stabilizing effect of the gradient penalty is empirically demonstrated, focusing on this novel observation and contribution seems more relevant to me. 
* **No comparison with CO2**: while experiments with SlowMo/DiLoCo are performed, no comparison with state-of-the art method CO2  [[Sun et al., 2024]](https://arxiv.org/pdf/2401.16265) is done.
* **Hyper-parameters introduced:** EDiT introduces 3 novel hyper-parameters $\alpha, \delta, \phi$, but the impact of their values on the method is not discussed.
* **Lack of clarity and imprecision in the writing:**
  * It was not clear at first that the $N$ models were *data parallel sharded* model replicas and **not** model-parallel splits. Maybe clarifying this distinction early in the paper could avoid potential confusion.
  * **line 21:** *“ensure training stability and improve performance.”* this is a bit cryptic. Maybe detailing the reason for these instabilities (as done lines 210-213) early in the paper would help understand the challenges tackled here more clearly.
  * **line 50:** *“Current Local SGD methods do not integrate well with modern distributed strategies”*. Why? Can you explain your point? For instance, the CO2 paper claims that their algorithm is compatible with ZeRO-series optimizers.
  * **lines 73-75:** the problem described is exactly the one the CO2 paper aims at solving, so saying *“current local-SGD method”* seems to be an overstatement here.
  * **line 116:** *“DiLoCo extends the slow momentum in SlowMo to the outer optimizer.”* this statement does not seem accurate: the momentum in SlowMo is already in the outer-loop, and seems to be roughly equivalent to DiLoCo (with nesterov SGD as the outer optimizer, which is also noted line 752). Can you clarify in which way DiLoCo extends SlowMo in this context?
  * **line 170:** *“model synchronization”*. What does this mean? Can you provide a clear definition or explanation of this?

### Questions
* **Intra node for model shards:** on standard GPU clusters using nodes with 8 A100 or H100 GPUs with 80GB of memory, what size of model can be loaded & how many intra-node model replicas does it lead to?
* **line 168:** a warm up phase is used as in post-local SGD. However, DiLoCo finds that it is not really helpful (cf their Fig.3), do you observe a different behavior here?
* **Tab.1 & Fig.4**: Why not experiment with A-EDiT on the in-house dataset?
* **Fig 5.a)**: can you explain the reason why A-EDiT and EDiT exhibit the same throughput in the random-straggler scenario?
* **In ablation Fig 7.a):** the Gradient Clipping doesn’t seem to have much effect in the validation PPL curve, is it really necessary to keep it along with WA and anomaly detection (which both seem to smooth the spikes observed in the validation PPL curve during training)?

=========REBUTTAL==========

The authors correctly addressed my concerns and provided further precision on their method, I raise my score.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper presents EDiT (Efficient Distributed Training) and its asynchronous variant A-EDiT, which aim to improve the efficiency of distributed training for large language models (LLMs). EDiT solves issues in existing distributed training such as communication bottlenecks, straggler delays, and limited scalability in heterogeneous environments. A pseudo-gradient penalty strategy is introduced to enhance training stability. Experimental results suggest EDiT and A-EDiT addresses the straggler issue and is more stable compared to baseline like DiLoCo.

### Strengths
* The paper introduces an innovative approach that combines Local SGD with asynchrony and gradient penalty,, which addresses communication overhead and resource elasticity.
* The paper rigorously evaluates EDiT and A-EDiT on multiple benchmarks, demonstrating improved performance in training speed, stability, and generalization compared to state-of-the-art methods.

### Weaknesses
* EDiT and A-EDiT rely on a range of hyperparameters, how should we think about and choose $\alpha, \phi, \beta$ on a new training task?
* The authors propose using gradient norm as a metric for anomaly elimination and gradient penalty. However, in llm training, gradients often have outliers on some of the examples. If we ignore examples with large gradient norms, will it create bias on training?
* It would also help if the authors can provide convergence analysis for the proposed EDiT method. Do EDiT and A-EDiT have convergence guarantee?

### Questions
Please see weakness section.

### Soundness
3

### Presentation
3

### Contribution
2
