## Human Reviewer 1

### Summary
This paper proposes Mixture of Shards (MOS), a LoRA-based method designed to reduce trainable parameters while maintaining performance for LLMs. MOS combines inter and intra layer sharing mechanisms with MoE-like routing system for parameter selection. It also introduces four “differentiation strategies”: subset selection, pair dissociation, vector sharding, and shard privatization (to add diversity and prevent performance degradation from parameter sharing). The authors claim that MOS achieves about an 8x reduction in parameters compared to LoRA while retaining competitive performance.

MOS method proposes leveraging both VeRA-like inter-layer parameter sharing and PRoLoRA-like intra-layer parameter sharing. It is proposing “Global Sharing Scheme” where each adapted layer across the Transformer creates its low-rank matrices (A and B) using shards from a globally shared pool selected by MoE-like routing.

“Differentiation Strategies” used in MOS:

-Subset selection - selects a subset of vector pairs per transformer block

-Pair dissociation - separates vector pairs into two different pools to create unique combinations for each block.

-Vector sharding - breaks down vectors into smaller shards, which are then concatenated.

-Shard privatization - divides the global pool into public and private sections.

### Strengths
-The paper has solid motivation.

-Authors propose an intuitively sound idea of using a combination of inter layer and intra layer sharing with MoE-like routing.

-The authors claim that MOS is the first method to use an MoE-like mechanism for parameter-efficient fine-tuning in a single-task LoRA.

-The comparisons with LoRA, VeRA, and PRoLoRA are relevant baselines for MOS performance.

-I like the design of the initial experiment (Table 1) - it's good to back up and motivate the method (though I have some comments mentioned in the weaknesses).

### Weaknesses
I think that the idea behind MOS is interesting and worth exploring, but it needs more rigorous experiments and analysis, along with a clearer explanation of the method and design choices. Without the code, reproducibility is challenging. 

I would like to understand why we would select this method over simpler methods like VeRA that provide good results. Does this method offer enough benefits to justify its complexity? The complexity should be justified, and any impact on finetuning overhead should be mentioned.

Weaknesses:

-The motivation for the specific “differentiation” strategies could be clearer. The authors mention that these strategies help maintain representational power, but this is very high-level and lacks theoretical support.

-The MoE-like routing mechanism for parameter selection isn’t clearly explained, making it hard to reproduce. What exactly is the routing algorithm? Were other approaches tested?

-The paper only evaluates MOS on instruction-following tasks.

-The comparison with VeRA isn’t entirely fair, as MOS uses more parameters than VeRA. I understand that VeRA can have practical limitations in increasing parameters, but could we reduce the MOS parameter count to match VeRA?

-Standard deviations are not provided.

-The initial experiment (Table 1) is interesting, but the conclusion about random scaling doesn’t seem fully justified - this strategy shows very minimal improvement and might not be statistically significant. For subset selection, could more models and seeds be tested to confirm the results?

-Could we add some additional models? Even smaller ones could help validate MOS’s performance. The current results are limited to LLaMA2-7B and LLaMA2-13B, with minimal gains for the latter, which may not justify MOS complexity.

### Questions
Questions:

-What’s the reason behind choosing the specific differentiation strategies? How was each expected to impact performance, and was the decision based on empirical results?

-How exactly are vectors selected from the global sharing pool?

-Does the MOS method affect finetuning time? (given the added complexity of MOS)

-Do you plan to release the code? This is a complex framework, and code would be very useful

-Can you elaborate on the statement that differentiation “reverses the detrimental effects of sharing”? Is there any theoretical support for MOS’s design?

-Why was standard deviation not provided for the averages?

-“Differentiation” is typically associated with gradient computation in deep learning, which might cause confusion. I’d consider a different name.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
3

---

## Human Reviewer 2

### Summary
This paper introduces Mixture of Shards (MoS), a sharded adaptation of LoRA designed to achieve greater parameter efficiency by leveraging parameter sharing across layers and within layers. MoS not only reduces the number of parameters required compared to traditional LoRA but also mitigates the potential performance degradation associated with excessive sharing. This is achieved through four strategies: subset selection, pair dissociation, vector sharding, and shard privatization, which ensure that each shared parameter can adapt to specific model requirements. MoS demonstrates a further reduction in trainable parametric usage, allowing more scalable deployment of LoRA-based models.

### Strengths
1. MoS combines subset selection, pair dissociation, vector sharding, and shard privatization to reduce parameters while maintaining performance.
2. Demonstrates an eightfold parameter reduction compared to traditional LoRA with empirical support.
3. Provides insights into the contributions of each differentiation strategy.

### Weaknesses
1. Although the paper introduces subset selection, it lacks criteria for choosing subsets; the selection is randomly initialized and fixed throughout training.
2. The MoS approach is primarily a combination of various techniques rather than a cohesive, unified method.
3. The MoS approach introduces significant randomness, making it challenging to determine if the improvements result from the design or from random variations. A test of significance could strengthen these claims.
4. The paper includes limited ablation studies for MoS, making it difficult to isolate and understand the contributions of each individual strategy in the overall design.

### Questions
1. For the experiments conducted with two runs using seeds 0 and 1, could you provide the individual performance results for each run? Additionally, were any further experiments conducted with different seeds to assess the robustness of the results?
2. How does the random initialization impact the performance of MoS? Given the reliance on randomness, are there specific initialization settings or hyperparameters that consistently yield better results?
3. What criteria, if any, were used to decide the number of shards in the global pool, and how sensitive is the model’s performance to this choice?
4. Were there any specific cases where certain differentiation strategies (e.g., subset selection, pair dissociation) proved more beneficial than others?
5. How does the computational overhead of MoS compare to traditional LoRA during training and inference, especially with regard to memory usage and GPU hours?
6. Since MoS integrates multiple strategies, are there any known trade-offs between parameter savings and performance across tasks?

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
6

### Confidence
4

---

## Human Reviewer 3

### Summary
The paper introduces a novel fine-tuning method called **Mixture of Shards (MoS)**, which aims to significantly improve parameter efficiency in adapting large language models for customized applications. As large language models (LLMs) continue to scale, there is a growing need for parameter-efficient fine-tuning techniques to manage the high GPU memory overhead associated with serving multiple customized models simultaneously. Traditional approaches, such as Low-Rank Adaptation (LoRA), reduce resource consumption by updating pretrained weights with trainable low-rank matrices, but they still encounter scalability and memory limitations when applied to large models and extensive user customization. MoS offers a solution that retains the advantages of LoRA while achieving greater parameter efficiency through innovative parameter sharing and differentiation mechanisms.

The central concept behind MoS is to combine **inter-layer and intra-layer parameter sharing** in a single framework. This sharing is further enhanced by four lightweight differentiation strategies designed to counteract potential performance degradation from pure parameter sharing. These strategies include **subset selection**, **pair dissociation**, **vector sharding**, and **shard privatization**, each providing unique ways to increase the diversity and exclusivity of shared parameters across layers. By using a **Mixture-of-Experts (MoE)-like routing mechanism**, MoS selects and concatenates specific shards from a global parameter pool, thereby achieving efficient memory usage while maintaining high model performance.

In terms of experimental validation, the paper presents extensive evaluations on various NLP tasks, including factual knowledge (MMLU), multilingual question-answering (TyDi QA), mathematical reasoning (GSM8K), multi-step reasoning (BBH), and coding (HumanEval). The experiments demonstrate that MoS outperforms LoRA and other baseline methods in parameter efficiency, particularly under limited parameter budgets. MoS achieves approximately eightfold parameter savings compared to standard LoRA configurations, making it a promising approach for scenarios requiring numerous custom models.

An ablation study further examines the importance of each differentiation strategy, showing that components like pair dissociation and shard privatization provide substantial gains in efficiency, while vector sharding offers incremental improvements. The study reinforces the necessity of each differentiation strategy in achieving the performance and efficiency benefits observed with MoS. Additionally, a scalability analysis using the larger LLaMA2-13B model demonstrates that MoS maintains its advantages on a larger scale, further underscoring its robustness and suitability for high-capacity models.

The paper positions MoS as an important step forward in parameter-efficient fine-tuning. MoS’s compatibility with LoRA-based infrastructure and its ability to serve multiple customized models simultaneously without substantial memory overhead make it practical for real-world deployment. The findings provide insights into the trade-offs and design considerations of parameter sharing, offering a valuable resource for researchers and practitioners working on efficient model adaptation techniques. The paper’s detailed methodology, comprehensive experimentation, and focus on parameter efficiency contribute meaningfully to the broader research area of resource-efficient machine learning, addressing critical scalability issues as the field advances.

### Strengths
The paper provides solid technical grounding for the Mixture of Shards (MoS) method, with each component—inter-layer and intra-layer sharing and differentiation strategies. The Mixture of Shards (MoS) approach is a novel, well-motivated response to the growing need for efficient fine-tuning techniques for large models. By blending inter-layer and intra-layer sharing with lightweight differentiation strategies, the paper introduces a resource-efficient method that extends beyond existing parameter-sharing methods like LoRA, VeRA, and PRoLoRA. This innovative combination of techniques in MoS is practically a valuable approach. 

The experimental design is comprehensive and addresses key aspects of parameter efficiency, memory usage, and model performance across a range of NLP benchmarks (e.g., MMLU, GSM8K, TyDi QA). The thorough ablation study underscores the necessity of each differentiation strategy (subset selection, pair dissociation, vector sharding, and shard privatization) and supports the paper’s claims about MoS’s efficiency. The paper also includes scalability tests, demonstrating MoS’s robustness on larger models, such as LLaMA2-13B, reinforcing its applicability to current large model architectures.

MoS integrates four nearly cost-free differentiation strategies—subset selection, pair dissociation, vector sharding, and shard privatization—to counteract the performance limitations of pure parameter sharing. These strategies is carefully designed to enhance the diversity and exclusivity of shared parameters, which contributes to the robustness and performance of the method. 

The paper includes rigorous experimentation across diverse NLP benchmarks, such as MMLU ( Massive Multitask Language Understanding for factual knowledge), TyDi QA (multilingual question-answering), GSM8K (for mathematical reasoning), BBH (Big-Bench-Hard for multi-step reasoning), and HumanEval. These benchmarks test the model on factual knowledge, multilingual capabilities, mathematical reasoning, general reasoning, and coding. The results demonstrate MoS’s parameter efficiency and effectiveness compared to baseline methods, making a strong case for its practical utility. The parameter savings—approximately eightfold compared to standard LoRA—are significant, supporting the method’s scalability. This reduction substantially alleviates the memory burden, enabling more efficient model customization and serving without sacrificing performance, which is particularly valuable in settings requiring multiple concurrent models.

The paper is well-structured, with a logical flow that introduces the problem, presents the MoS solution, and discusses experimental results comprehensively. The clarity of the writing is generally good, though the differentiation strategies could benefit from additional diagrams or illustrations to aid in understanding for a wider audience.

### Weaknesses
While MoS is evaluated on a range of NLP tasks, the paper does not sufficiently analyze the method’s performance across various model architectures or specific task categories (e.g., multilingual tasks, code generation) where parameter efficiency and differentiation strategies could have different impacts. A breakdown showing how MoS performs on individual tasks, especially ones that are highly memory-intensive, would offer a clearer picture of its advantages and limitations across diverse NLP applications. 

The ablation study is a strong point but could be enhanced by further exploring each differentiation strategy’s scaling potential. For instance, while the study confirms the individual benefits of subset selection, pair dissociation, vector sharding, and shard privatization, it doesn’t analyze the interactions or scalability of these strategies as model or task complexity increases. Additional experiments showing the performance impact of these strategies in larger configurations or different combinations would make the study more informative for readers looking to fine-tune MoS to specific needs.

The paper would benefit from a section discussing the potential limitations of MoS in specific scenarios. For instance, the effectiveness of MoS might be reduced when applied to tasks with low data diversity or high variance in representational needs across layers. Discussing scenarios where MoS might underperform or require adaptation would provide a more balanced view and help users assess when MoS is a suitable choice.

### Questions
Have you considered how MoS might interact with techniques like quantization, pruning, or dropout? Many practical deployments of large models use these methods in conjunction to manage resource constraints, and understanding how MoS might complement them would add value. If feasible, a brief experimental analysis or discussion on this integration would enhance the paper’s relevance for real-world applications.

Could you discuss potential limitations of MoS, such as scenarios where the method may underperform or require additional tuning? For example, does MoS have any specific limitations when applied to domains with high variability in representation needs across layers? A discussion on this would offer a more balanced perspective, helping readers assess the suitability of MoS in various contexts.

Has MoS been evaluated for its impact on inference latency, especially in multi-model serving scenarios?  

Are there limitations to the size or complexity of models that MoS can handle effectively? For example, do the benefits of MoS start to diminish for models larger than LLaMA2-13B, or do you anticipate any challenges in scaling it to models with trillions of parameters?

### Soundness
4

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
5

---

## Human Reviewer 4

### Summary
The paper investigates a more lightweight solution than LoRA in order to serve a large number of finetuned models at the same time. Based on a finding that excessive sharing may hinder model performance, the authors believe that differentiation is necessary to reverse the detrimental effects of pure sharing. The paper proposes Mixture of Shards (MoS) that incorporates both inter-layer and intra-layer sharing with four differentiation strategies.

- Subset selection: randomly choose r pairs of vectors at the beginning for each layer
- Pair dissociation: separate each pair into two pools, where each vector in a pair will be sampled independently
- Vector sharding: shard the global pool into n parts and concatenate sampled vectors from each shard
- Shard privatization: reserve a private segment for exclusive use for each matrix

### Strengths
- The method is more general than LoRA, making LoRA a special case when there is no global pool.
- The authors provide ablation study for each of the differentiation strategies (except subset selection), showing the efficacy of each strategy.
- Overall, I find the finding about sharing & differentiation makes sense and the motivation is clear. Each differentiation strategy is proposed to keep the number of parameters unchanged but increase level of differentiation between each layer.

### Weaknesses
Overall, the paper is well written. There are some minor details that can be improved.
- Figure 2 can be more accurate and following the main text better. There is no mentioning of router "R" in the main text. Notations like $A^{pub}$, $A^{pri}$, $B$, $I$, $m_{ij}$ can be used to make it clearer.
- Index(.) could be replaced by a better notation since .index(.) can be understood as the index of some element in an array.

### Questions
- In Section 3.3, is $I_a^k \in \mathbb{R}^r$ or $\mathbb{N}^r$
- What do the 4/8, 16/32 (or "increasing the rank to 4 or 8") in Table 2 mean?
- Many implementation details are missing - what's the pool size, how many shards, breakdown of private & public segments, etc.

### Soundness
4

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
3