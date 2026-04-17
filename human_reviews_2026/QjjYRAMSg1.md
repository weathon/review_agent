# Mediater: Memory-efficient LLM Merging with Less Parameter Conflicts and Uncertainty Based Routing

- Decision: Reject
- Scores: 2, 4, 4, 6

## Abstract
Model merging aggregates Large Language Models (LLMs) finetuned on different tasks into a stronger one. However, parameter conflicts between models leads to performance degradation in averaging. While model routing addresses this issue by selecting individual models during inference, it imposes excessive storage and compute costs, and fails to leverage the common knowledge from different models. 
In this work, we observe that different layers exhibit varying levels of parameter conflicts. Building on this insight, we  average layers with minimal parameter conflicts and use a novel task-level expert routing for layers with significant conflicts.
To further reduce storage costs, inspired by task arithmetic sparsity, we decouple multiple fine-tuned experts into a dense expert and several sparse experts. Then, we select and merge appropriate experts based on the task uncertainty of the input data. 
We conduct extensive experiments on both LLaMA and Qwen with varying parameter scales, and evaluate on real-world reasoning tasks. Results demonstrate that our method consistently achieves significant performance improvements while requiring less system cost.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a framework for efficiently merging multiple fine-tuned large language models. It observes that parameter conflicts vary across layers, so it averages layers with low conflicts and routes high-conflict layers using task-specific experts to retain unique knowledge while minimizing interference. Mediator further compresses these experts through sparse decomposition and uncertainty-based expert selection, achieving superior performance and reduced system cost compared to prior merging methods on models like LLaMA and Qwen.

### Strengths
1. The proposed method is reasonable.
2. The experiments are substantial and promising.

### Weaknesses
1. Writing should be improved. I notice authors really like to use very long and complex phrases such as „observing“ or „considering“. Actually this lowers readability and make readers read in pain. I have to read some sentences for several times to grab ideas. And due to this, the logic flow of some parts becomes pretty bad. Also there are numerous grammar problems.
2. Unclear technical details. Some parts in Section 3 and 4 are hard to follow. Authors introduce lots of ideas without giving structured, concise and simple explanations. In my humble opinion, it would be better to keep the contents in the main paper easy to follow.

### Questions
1. Line 105. Should say next token prediction.
2. Line 126-128. I have no idea what do you want to say with the sentence starting with „considering“. There is even no verb in this sentence.
3. I don’t get why did you do parameter denoising, even after reading Appendix F1. Could you give more details and kindly with some example?
4. From Fig. 3, I think 4B models don’t not show huge conflict ratio in its last layers. So your claim should be reconsidered.
5. Line 249-259. Could you give more explanations about the conclusion you observe from task arithmetic and model parameters? I could not understand what is its relation to sparsifying model.
6. How did you create CoT data? I also read appendix and it seems you are not generating CoT data and is just reusing the ones provided by original datasets. Solely compiling a CoT data split from all data is not a contribution.

### Soundness
3

### Presentation
1

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
This work proposes a new method named Mediator for merging LLMs finetuned on different tasks. The authors first observe that the parameter conflicts between models lead to performance degradation in the conventional averaging strategy. To mitigate this, Mediator performs an average over layers with minimum conflicts and keeps these layers with distinct behaviors, which can be selected via an uncertainty-based router later. Through this way, they decouple multiple fine-tuned experts into one dense expert and several sparse experts, which can reduce memory cost while achieving high accuracy. Experimental results of different LLMs and tasks confirm the effectiveness and efficiency of the proposed method.

### Strengths
* The motivation is clear. This work aims to propose a memory-efficient LLM merging method that accounts for the layer conflicts
* The structure is good. The authors first identified a merging issue due to the parameter conflicts, and then proposed the layer-wise merging and routing. To reduce the overhead and handle OOD cases, appropriate methods have been introduced. Each part is supported by good analyses and empirical studies
* Extensive experiments across LLMs and tasks validate that Mediator can outperform other baselines while being more efficient

### Weaknesses
* The proposed method introduces significant complexity. Mediator consists of several components: Adaptive Merging and Routing, Expert Decomposition, and uncertainty-based routing (which requires further training samples). Adding these components together makes the method complex. I would suggest these directions for simplifying the current methods: (1) learn a router to select the best model. According to the table 1, $\Phi_{SEL}$ is able to outperform $\Phi_{AVG}$, we can learn a lightweight router to allocate the different input samples.  I understand this routing consumes more memory usage, but it is simple and easy to manage  (2) dynamical averaging. Based on my understanding, Mediator serves as a refined averaging strategy, i.e., averaging layers with similar distributions while performing routing for top-k distinct layers. Actually, the routing here is another averaging, and the weights are derived from uncertainty. I would unify the two-step averaging by using a dynamical averaging. The key point here is how to assign proper weight to each layer for a specific input. I am not convinced by the Mediator methods and expect to see the results of the baselines I suggest.
* It is not clear which component of Mediator contributes most in the ablation study. Can the authors clarify this?

### Questions
* how to calculate the layer conflict $d_{l}$ across different fine-tuned models?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies the model merging task, i.e. merging various fine-tuned model weights to create a superior LLM that retain the advantages of fine-tuned ones avoiding parameter conflicts. Specifically, this paper first shares findings on the patterns of parameter conflicts: front and last layer exhibit the highest level of conflict yet central layers have lower level of conflict. This paper introduces an adaptive framework, Mediator, which averages layers with lower conflicts and uses expert routing for layers with higher level of conflicts.

### Strengths
This paper has a clear motivation, i.e. to preserve the task-specific knowledge from each fine-tuned weights and to avoid parameter conflicts, and proposes a practical framework Mediator to achieve such goal. 

The experiments are extensive, across various datasets and evaluation tasks. Mediator takes lower post-training time, inference time and memory cost compared with other baseline models. Mediator also shows better model merging performance compared with other baseline methods.

### Weaknesses
One of the weaknesses is about the writing quality. The Introduction section could be written more coherently. The current paragraphs feel inconsistent and may confuse readers. For example, it is unclear what components are actually included in Mediator. The discussion starting around Line 59 introduces adaptive merging, but then suddenly shifts to model compression without clear motivation or transition. Afterwards, the topic changes to different levels of routing and out-of-distribution (OOD) samples, still without sufficient introduction or logical connection. This lack of flow makes it difficult for readers to follow the authors' reasoning and understand the overall contribution of the paper. Also, in lines 77–78, the topic shifts from OOD samples to finetuning on CoT datasets without proper connections, which can confuse the readers.

There are similar issues in sub-sections in Section 4. It would be better to organize each part of the method in a more logical and consistent way to help the reader better understand the ideas.

Some details of the method can be clarified more clearly. For instance, which layers in the used models are treated as front layers, central layers, and which threshold is used to classify layers with low and high conflicts.

Another concern is about the model choice. The models used in this study are within the 3–8B range and published more than 1 year ago. It would be better to validate the framework on bigger and more recent models to further prove its effectiveness.

### Questions
Please see weakness above

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studies LLM merging where they merge different fine-tuned domain models to build a superior LLMs. The paper proposes an adaptive method where it only merges the layers whose parameter conflict rate is smaller than a threshold and keeps layers with significant parameter conflicts with inference-time routing. The paper conducts comprehensive experiments across model families and scales for verifying the method’s effectiveness.

### Strengths
- The paper studies an interesting direction of merging different fine-tuned LLMs for obtaining a more powerful general LLM. The idea is clear and is proved with informative analysis. 

- The paper provides detailed studies to help understand the parameter conflicts in different LLMs. This motivates its proposed method quite well.

### Weaknesses
- The paper focuses on the LLMs with different domain data. Such fine-tuned LLMs can be quite a lot (could be hundreds and even more in real world). Although the proposed method considers this issue with expert decomposition, its scalability is still unclear. The reviewer expects to see experiments with much more domain experts.

- The writing needs to be improved even given the page limit. The paper defers too many details into Appendix, which makes the main paper itself not technically clear enough.

### Questions
- Is this approach also applicable to MoE LLMs?

### Soundness
3

### Presentation
3

### Contribution
3
