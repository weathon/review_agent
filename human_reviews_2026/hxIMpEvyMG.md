# MLLM-CL: Continual Learning for Multimodal Large Language Models

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Recent Multimodal Large Language Models (MLLMs) excel in vision-language understanding but face challenges in adapting to dynamic real-world scenarios that require continuous integration of new knowledge and skills. While continual learning (CL) offers a potential solution, existing benchmarks and methods suffer from critical limitations. In this paper, we introduce MLLM-CL, a novel benchmark encompassing domain and ability continual learning, where the former focuses on independently and identically distributed (IID) evaluation across evolving mainstream domains, whereas the latter evaluates on non-IID scenarios with new model abilities. Methodologically, we propose preventing catastrophic interference through parameter isolation and an MLLM-based routing mechanism. Extensive experiments demonstrate that our approach can integrate domain-specific knowledge and functional abilities with minimal forgetting, significantly outperforming existing methods. Our benchmark and code will be publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces a continual learning benchmark and methodology to tackle continual in more realistic settings: where general-domain knowledge and expert knowledge might be required, and tasks might not be similar to the training regime (non-IID, as the authors put it). They demonstrate that a simple baseline with a LoRA router performs almost like an oracle owning to the router's performance.

### Strengths
* Many baselines were presented, giving readers some confidence that findings are solid. Though not a domain expert, it seems that the baselines cover the basics from recent literature.
* Prescient problem in contemporary AI, as current models forget new knowledge when switching between prompts
* Intuitive ideas, both for baseline and benchmark
* Solid presentation and writing: logic was more or less easy to follow from the first read, although writing could be tightened here and there.

### Weaknesses
* Main weakness: experts and router are fixed based on the knowledge of the tasks/datasets. What is a task-agnostic way to construct such a pipeline? Why wasn't it explored? Moreover, the router's job is easy in this constrained environment, as it has to select between a small number of experts. A more realistic setting would perhaps contain many more experts, making the task much harder for the router. I do want to acknowledge that a pipeline as the authors present it might still be useful in practice for some settings, but I do not expect the task to be so easy in general.
* I understand the computational constraints with such models, but a permutation of the tasks would also be interesting to see to get a sense of whether results are generalizable. This could also refer to previous works showing consistency across task order, though it wouldn't make the point as strongly.
* Why was only accuracy shown? How unbalanced are the datasets, and is accuracy (over F1 or other metrics) reflective of relative performance? It would be useful to see the same settings with a different metric, which is hopefully easy to compute if the authors have kept proper logs of their experiments.

---

I wanna note that I am more than willing to take the response of the authors seriously and improve my rating after a satisfactory rebuttal, so I want to encourage them to provide a thoughtful response.

### Questions
* Replay-based baselines not explained
* Section 5.1 would benefit from more detail in the main paper itself. Figure 7 is particularly helpful, although I expect it may be slightly difficult for some readers to understand.
* Big tables might be difficult to parse, please experiment with adding alternative modes of presentation, at least in the appendix.
* The finding that experts other than the expected one improving performance is a very interesting and intuitive finding. I think it would be useful for the authors to add a section in the paper exploring this more, e.g., when router "fails", how and where does performance increase.
* It would be helpful to briefly discuss complexities and number of trained parameters for each method.

### Soundness
2

### Presentation
4

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
The paper introduces the MLLM-CL benchmark and MR-LoRA method for enabling CL for MLLMs. The authors design a benchmark first which consists of two settings: (i) Domain Continual Learning (DCL), where models sequentially learn domain-specific knowledge across heterogeneous domains under IID conditions and (ii) Ability Continual Learning (ACL), which evaluates the acquisition of heterogeneous skills in non-IID environments that mimic real-world adaptation. The authors also propose MR-LoRA algorithm which combines parameter isolation via domain- or ability-specific LoRA adapters with a multimodal router that selects the most suitable expert for a given vision-text input using only a few samples. Experiments on LLaVA-v1.5-7B and InternVL-Chat-V1.0 VLMs demonstrate that MR-LoRA achieves near-oracle accuracy on DCL and significant improvements on ACL while eliminating catastrophic forgetting, outperforming several existing baselines.

### Strengths
1. The paper is well-written and can be followed easily.

2. The idea of using a two stage inference using the router and ability-specific modules is interesting and from experiments seems effective.

3. Experiments include studying hyperparameters which provide helpful insight about the proposed method.

4.

### Weaknesses
1. The MLLM-CL benchmark primarily consists of existing benchmarks that are combined. As a result, the contribution in terms of  introducing a new benchmark is weak.

2. Comparisons are limited and include a handful of recent methods. However, there are other CL methods for VLMs with public codebases that can be included to demonstrate that the proposed method is competitive. 

3. The code and the benchmark are not provided which makes judgment about reproducibility challenging. The authors have promised to release but at this point, they are not provided. 

4. Although the router performs near-perfectly on the specific benchmark of the papers, it assumes clear domain boundaries and stable task definitions. In practice, new tasks may not fit into existing categories, leading to ambiguous routing or expert overlap. The router itself is another trainable component requiring careful tuning and maintenance; its misclassification can directly degrade output quality. It is not clear that the method would work well on other benchmarks.

5. The benchmark has five domains and four abilities and hence, it is unclear how MR-LoRA would perform on unseen domains or real-time streaming data where task boundaries are unknown. The model assumes discrete, sequential task learning rather than fully online continual adaptation, limiting its generality for practical applications.

### Questions
1. There are several existing benchmarks for CL with VLMs. I agree the heterogeneity in the existing datasets are less but why MR-LoRA  is not tested on existing benchmarks? It is important to demonstrate that MR-LoRA  remains competitive on existing benchmarks. Moreover, many existing methods report their performances on the existing datasets which makes comparisons to a broader range of methods straightforward.

2. Some performance values on tables are quite close. Why standard deviation values is not reported? Without them, it is difficult to conclude that the differences are statistically meaningful.

3. I was wondering if the method would be effective for a larger number of domains or skills? Would the router still be able to select the right adapter module with high accuracy? Wouldn't continual accumulation of adapter module lead to memory overhead?

4. I was wondering whether the two stage inference leads to significant inference time overhead? In CL, real-time inference is also very important and this aspect has not been studied in the paper.

### Soundness
3

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
This paper studies continual learning for multimodal large language models. It proposes a new benchmark called MLLM-CL which incorporates domain continual learning and ability continual learning. A method is also proposed to boost the CL capability of MMCL through low-rank adaptation and parameter selection.

### Strengths
1. It is good to extend the continual learning task in traditional deep learning to MLLM.
2. The contributed dataset could be helpful to the community.
3. The proposed method is simple, and works as shown in experiments.

### Weaknesses
1. There exist many ways to make MLLM adapt to new tasks or domains, e.g, in context learning, or retrieval augmented generation. Given a base MLLM model with strong generalization capability, a training free strategy could be more valuable.
2. The classification of DCL and ACL should be further justified. Some tasks in DCL could also be regarded as ACL, e.g., identifying is acid present in medical images. A fuzzy classification would degrade the importance of the dataset.
3. Another concern is that the proposed method is more like a trick and lacks novelty or elegance in methodology. Lora and expert selection are commonly used strategies. Continuously adapting to new domains or new abilities would increase the complexity of the model. 
4. The two-stage inference also introduces extra computational complexities.

### Questions
1. Discuss or compare with other methods that could make MLLM adapt to new tasks or domains. 
2. Further justify the classification of DCL and ACL.
3. Evaluate the efficiency.

### Soundness
2

### Presentation
3

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
This paper addresses the critical challenge of catastrophic forgetting and limited plasticity in continual learning scenarios for Multimodal Large Language Models (MLLMs). The authors propose **MR-LoRA**, a novel framework that introduces two key components: (1) training a fresh LoRA adapter from scratch for each new task to preserve model plasticity and avoid negative transfer from previous task weights, and (2) employing a few-shot MLLM-based router that dynamically selects the most suitable expert based on the input modality and query semantics. Additionally, the router is fine-tuned incrementally using a small number of samples from each learned task, enabling it to adapt to new tasks while retaining knowledge of old ones. The method is evaluated on both domain-level and ability-level continual learning benchmarks, demonstrating superior performance in terms of average accuracy, final performance, and backward transfer compared to existing baselines.

### Strengths
1.  **Clear and Well-Motivated Problem Formulation:** The paper clearly identifies the dual challenges of stability and plasticity in MLLM continual learning.
    
2.  **Simple yet Effective Core Idea:** The proposal to train a _fresh_ LoRA from scratch for each task is conceptually simple but effective. This design choice directly tackles the issue of weight interference caused by reusing previous adapters, leading to better new-task performance.
    
3.  **Innovative Use of MLLM as a Router:** Leveraging the MLLM itself as an intelligent, few-shot router for expert selection is a elegant solution, which is more robust than feature-similarity-based routing.

### Weaknesses
1.  **Linear Parameter Growth and Scalability Concerns:** The most significant limitation is the linear increase in the number of stored LoRA modules with the number of tasks. While LoRA is parameter-efficient, storing hundreds or thousands of adapters could become impractical in long-term or open-ended continual learning scenarios. The paper does not discuss potential strategies to mitigate this.
    
2.  **Limited Discussion on Task Similarity and Negative Transfer:** The paper assumes tasks are distinct enough to warrant separate experts. However, it does not explore scenarios where tasks are highly similar or overlapping. In such cases, having separate LoRAs might lead to redundant learning, and the router might struggle to make consistent decisions.
    
3.  **Evaluation on Synthetic vs. Real-World Task Sequences:** The experimental setup uses predefined task sequences. A more realistic evaluation would involve open-world scenarios. The generalization capability of the router to completely unseen task categories is not tested.

4. **Novelty Limited:** All content revolves around a central point: adding the LoRA module. Similar approaches exist in continuous learning tasks for standard models. Overall, while effective, the innovation is relatively limited, resembling more of an engineering refinement.

### Questions
See Weakness

### Soundness
3

### Presentation
3

### Contribution
3
