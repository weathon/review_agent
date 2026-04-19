# Ada-K Routing: Boosting the Efficiency of MoE-based LLMs

- Decision: Accept (Poster)
- Scores: 8, 6, 6, 6, 6

## Abstract
In the era of Large Language Models (LLMs), Mixture-of-Experts (MoE) architectures offer a promising approach to managing computational costs while scaling up model parameters. Conventional MoE-based LLMs typically employ static Top-K routing, which activates a fixed and equal number of experts for each token regardless of their significance within the context. In this paper, we propose a novel Ada-K routing strategy that dynamically adjusts the number of activated experts for each token, thereby improving the balance between computational efficiency and model performance. Specifically, our strategy incorporates learnable and lightweight allocator modules that decide customized expert resource allocation tailored to the contextual needs for each token. These allocators are designed to be fully pluggable, making it broadly applicable across all mainstream MoE-based LLMs. We leverage the Proximal Policy Optimization (PPO) algorithm to facilitate an end-to-end learning process for this non-differentiable decision-making framework. Extensive evaluations on four popular baseline models demonstrate that our Ada-K routing method significantly outperforms conventional Top-K routing. Compared to Top-K, our method achieves over 25% reduction in FLOPs and more than 20% inference speedup while still improving performance across various benchmarks. Moreover, the training of Ada-K is highly efficient. Even for Mixtral-8x22B, a MoE-based LLM with more than 140B parameters, the training time is limited to 8 hours. Detailed analysis shows that harder tasks, middle layers, and content words tend to activate more experts, providing valuable insights for future adaptive MoE system designs. Both the training code and model checkpoints will be publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents Ada-K routing, an additional adapter on mixture-of-expert model that can decide the number of experts activated for each token. The dynamic number of experts helps reducing the inference cost of the model, while still keeping a descent performance. The authors used proximal policy optimization algorithm to make the ada-K adapter trainable, and evaluated it on four MoE models at different scales. The authors additionally run a sweep of experiments to study the effect of Ada-K at different scale of MoE, the trade-off between accuracy and efficiency, and the ablation study of different hyper-parameters. The authors further introduced visualization of the expert allocation pattern, which helps understand the MoE architecture's dynamism.

### Strengths
- The idea of an adapter to control the number of experts at the granularity of token and layer can be universally migrated to most of the MoE models, and is simple to implement.
- The paper presents detailed optimizations to make the reinforcement learning of adapter more stable.
- The experiment is abundant and at a large scale, including:
  - models at different scale of both parameters and activated parameters, as well as different baseline number of activated experts. 
  - the pareto frontier between performance and the cost reduction
  - comparing Ada-K with existing heuristic-based dynamic expert allocation algorithms
  - ablation study on dataset, regularization, and warmup strategy
- The visualization of the result is novel and inspiring.

### Weaknesses
- Although Ada-K significantly reduces the total FLOPs at inference, an efficient implementation could be nontrivial, which limits the use of Ada-K technique in practice. Providing experiment data on the average latency/throughput could help make the inference cost improvement more significant.
- Figure 2 suggests that the hyperparameter $\lambda$ is crucial to the balance between performance maintenance and the inference cost reduction. However, the corresponding paragraph ("Trade-off between performance and activation reduction rate") lacks enough information on how to find the $\lambda$. What is the range of $\lambda$ in the figure? Is the best $\lambda$ generally applied to all the four models, or should a user of Ada-K also finetune on a sweep of different $\lambda$'s? Providing a plan for the user to find a good $\lambda$ could help improve Ada-K's usability.

### Questions
- I noticed that the authors claim that all conclusions in the ablation study is generally applied to all the four models. Providing the trade-off point of all other three models (or are they also 3e-3?) could answer my doubt on the second point of the weakness section.

The paper is well written in general. The question below will not influence my scoring.
- In section 3.3, paragraph PPO Loss, "Eq.(4) could be simplified as ...". I'd suppose it is "The expected return could be simplified as ..."

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a new Ada-K routing strategy for MoE-based large language models (LLMs), which dynamically adjusts the number of activated experts based on token importance. Due to the non-differentiable nature of the decision, the paper leverages the Proximal Policy Optimization (PPO) algorithm to facilitate the end-to-end learning process. The proposed method improves model performance while reducing computational costs. Extensive evaluations on various benchmarks, along with comprehensive ablation studies, demonstrate the effectiveness of Ada-K routing compared to traditional Top-K routing strategies.

### Strengths
1. The paper investigated a timely problem.
2. The routing strategy proposed in this paper is a new approach that dynamically adjusts the number of activated experts.
3. The paper presents a comprehensive set of experiments and analyses.

### Weaknesses
1. The structure of the Method section can be improved. Section 3.1 describes the traditional routing method of MoE, which would be more appropriately placed in the Related Work section. The Method section needs more elaboration. At present, it is rather brief, and expanding on key details would greatly improve the clarity and comprehensibility of the research.
2. This method has certain limitations, particularly in its application to models that use top-1 routing, such as Switch Transformer, making optimization more challenging.
3. Regarding the experiments in Section 4.2, a direct comparison between the warm-up Ada-k routing and the baseline top-k routing with different k values may be somewhat unfair. The models are likely to have different performance levels even before training due to the differences in routing methods. Providing a loss curve during training could better demonstrate the effectiveness of the proposed method.
4. Figure 8 shows that the use of the Ada-k strategy leads to performance degradation on simpler tasks while achieving better results on more complex tasks. This seems counterintuitive based on prior experience. Perhaps the authors could provide a more plausible explanation for this phenomenon.

### Questions
In addition to the issues mentioned in the Weaknesses section, there are a few other concerns:

1. What does $i$ represent in Equation (8)? Perhaps it should be $n$?
2. The authors should provide more details on how the policy $\pi$ was designed in this work and the rationale behind this design choice? Additionally, how was the number of training parameters calculated?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This work proposes to introduce an adaptive computation budget for MoE LLMs. Specifically, the proposed method fine-tunes a pretrained MoE LLM to activate the adaptive number of experts with PPO training and a trainable allocator layer. The proposed model achieves comparable performance with baselines but uses less computation.

### Strengths
1) Well-motivated. It is well-known that MoE LLMs are very effective and promising, but the efficiency of MoE LLM deployment is limited due to the huge amount of trainable parameters. It is good to improve the efficiency of LLMs.
2) Clear writing and comprehensive ablation studies.

### Weaknesses
1) An important baseline is missing -> Mixture of Depth (https://arxiv.org/abs/2404.02258). Due to the layer skip in this paper, the computation cost for each token is adaptive as well,
2) Due to the imbalanced computation cost in different layers, the pipeline parallelism is more difficult and challenging to use, during both training and inference.
3) There are many other ways to introduce adaptive computation budget, e.g. ACT algorithm in universal transformer (https://arxiv.org/abs/1807.03819), PonderNet (https://arxiv.org/abs/2107.05407). Need to discuss and compare. Why do you select PPO to train the allocator? More justification is required. It seems that the model is unnecessarily complex. Why not just ACT or so? Any algorithm or training difficulty? Introducing the PPO in such an early stage will make the LLM training pipeline much more complicated, which may make this approach not that useful, even if it is effective to some extent.

### Questions
1) What is the setting of LLM inference speedup? Batch inference or batch size == 1? How does the inference speed trend if we ablate the batch size? And are you using expert parallelism during training and inference?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces the Ada-K Routing method, which incorporates an additional RL agent into pre-trained MoE models to dynamically control the top-k in MoE routing. This approach enables dynamic adjustment of top-k at a lower cost, thereby enhancing the inference efficiency of MoE models. The effectiveness of the proposed method has been validated on multiple open-source models, demonstrating improved accuracy compared to existing threshold-based methods when achieving similar acceleration effects. Additionally, the paper presents detailed ablation studies and analytical experiments that provide valuable insights into the research on MoE models.

### Strengths
- Innovatively proposes to introduce an RL agent for controlling top-k in pre-trained MoE models, optimizing the allocation of computational resources and improving inference efficiency. The method has a low training cost and shows robustness to training data, outperforming existing methods across various downstream tasks.
- Thorough ablation studies demonstrate the relationship between acceleration effects and accuracy, providing practical guidance. It also validates the effectiveness of activation regularization and warm-up strategies.
- Performs meticulous analysis revealing that intermediate layers of trained models require the activation of more experts, reaffirming that more challenging tasks necessitate the activation of more experts.

### Weaknesses
- Some comparisons with baseline methods are not entirely reasonable. Existing work suggests that freezing the router yields better results when tuning MoE models. Ada-K freezes the router and introduces additional agent parameters for training, while other comparison methods only train the router. The reviewer recommends supplementing the results by (1) conducting full fine-tuning of the model with a frozen router and comparing the effects of introducing only threshold methods versus Ada-K; or (2) maintaining the current Ada-K setup and adding LoRA parameters to the threshold-based method while freezing the router for tuning.
- The paper "AdaMOE: Token-Adaptive Routing with Null Experts for Mixture-of-Experts Language Models" (ArXiv: 2024-06-19), which has a very similar title, also achieves dynamic adjustment of top-k during post-training and improves accuracy while saving FLOPs during the tuning phase. Given the short time interval, the lack of comparison with this baseline can be understood, but the reviewer suggests discussing it.

### Questions
- The reported speedup in the paper is based on the actual inference time during evaluation tasks, which likely applies to scenarios with batch sizes of 1 or smaller. However, MoE models have large parameter volumes, and their efficiency typically becomes evident with larger batch sizes during actual deployment. The reviewer is concerned that the RL-based method of controlling top-k for each token might lead to imbalance in batch inference, potentially affecting the acceleration effect. Therefore, the reviewer would like to know the relationship between acceleration effect and batch size.
- Would combining Ada-K with the standard SFT setting yield better results? Or, under the control of Ada-K, would the performance of SFT be affected?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper studies the dynamic routing strategy in MoE architectures. Conventional MoE architectures use a static Top-K routing, activating a fixed number of experts regardless of the token's complexity or importance. They propose the Ada-K routing strategy. Ada-K routing dynamically adjusts the number of activated experts based on the significance of each token, balancing efficiency and performance. The allocator module is based on RL. Ada-K achieves over 25% reduction in FLOPs and 20% faster inference speeds, with performance improvements across various benchmarks.

### Strengths
MoE is a scalable solution that balances parameter increase with computational cost. Targeting the limitations of prior efforts with a fixed number of experts, this paper works to make the expert number dynamic, which can bring additional efficiency and potential performance gains. The experiments include studies across multiple scales of models to test the effectiveness of the proposed method. An advantage of Ada-K is that it is pluggable, making it applicable across different MoE-based LLMs.

### Weaknesses
1. The Ada-k routing design works for the post-training of MoE-based LLMs. For post-training, as the gate is already trained, a question is whether it is necessary to use RL to learn the gate selection again, which complicates the overall design. For instance, a simple solution is to distill Top-k selection into binary selection with a separate gate, similar to the MoD design. The paper doesn't include comparison studies with this naive Top-k selection distillation, making it hard to say whether RL is necessary. I also didn't see a concrete motivation for using DRL.

2. Some parts of the paper writing need to be improved. For instance, for each layer $l$ equipped with the allocator, the state design and action design for the DRL agent are unclear.

### Questions
1. Is the usage of DRL necessary? How is the performance of the proposed method compared to naive Top-k selection distillation?

2. What is the state space of the DRL agent?

3. As each layer $l$ is paired with a DRL agent, are all DRL agents trained together or in a layer-by-layer manner? Will the selection for a layer $l$ influence the selection of layer $k$, where $k\neq l$? If so, how is the influence taken into account in the desigm?

### Soundness
2

### Presentation
2

### Contribution
3
