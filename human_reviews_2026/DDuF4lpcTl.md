# HarmoMoE: Unifying Domain-Specialized Experts into a Mixture-of-Experts Model under Privacy Constraints

- Decision: Reject
- Scores: 6, 6, 4, 4

## Abstract
Mixture-of-Experts (MoE) models offer a powerful way to scale capacity, but existing designs typically assume centralized access to all training data. In many real-world scenarios, however, data is distributed across clients from different domains and cannot be shared due to privacy constraints, making it challenging to build a unified and generalizable MoE. We propose HarmoMoE, a framework that unifies domain-specialized experts into a single MoE without sharing private data. HarmoMoE combines relevance-weighted DPP proxy selection with a context-aware router, ensuring that experts trained on both private and proxy data remain compatible and effectively coordinated. Experiments on CV and NLP show that HarmoMoE consistently outperforms recent methods such as BTX and FlexOlmo.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a framework for enabling data-distributed training without sacrificing data owners’ privacy. Their approach is to train diverse experts using local data for each client, and then construct proxy samples for fine-tuning private models and router learning. The method has shown superior performance over SOTA baselines.

### Strengths
- This paper provides a novel method to merge domain-specific experts into a versatile expert, with a special focus on data privacy. The method is promising. 

- The authors introduce several key elements, compared to the existing baselines: DPP proxy data sampling, context-aware router, and proxy-aligned expert training. Each component is ablated rigorously and is proven empirically to have contributed to the performance improvement.

### Weaknesses
- An immediate dropback is that what if there are no similar data to $D_p$ in the public dataset to construct $\hat{D}_p$? Does the procedure still work?

- There is no documentation of the computational cost in addition to BTM. Is the enhanced performance coming at a greater cost?

### Questions
- In Table 3, it seems the domain-specific experts are not performing the best in the corresponding domains?

- There are several other works with similar purposes. How does your method compare to [1][2]?

- To get a complete understanding of the method, could you please replicate the experiment from Section 4.4 on the vision tasks and the experiment from Section 4.5 on the NLP tasks?


[1] On-Device Collaborative Language Modeling via a Mixture of Generalists and Specialists

[2] Mixture-of-LoRAs: An Efficient Multitask Tuning for Large Language Models

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The manuscript proposes HarmoMoE, a framework designed to unify expert models trained on private data without requiring coordinated retraining. HarmoMoE uses a relevance-based determinantal point process to select diversified and domain-representative proxy samples, allowing the router to be trained in a harmonized manner using abundant approximate data. A context-aware router further refines the overall design.

### Strengths
1. The baseline selection is up to date.
2. The motivation for unifying expert models is well explained.

### Weaknesses
1. The main concern lies in the privacy-preserving claim. HarmoMoE uses relevance-weighted DPP to select proxy data that represent private data. However, if the proxy data are highly similar to the private data, wouldn’t this constitute a form of data exposure? If not, how does this differ from a vanilla DPP? I suggest adding more discussion about the privacy–utility trade-off involved in using such public proxy data.
2. HarmoMoE focuses on unifying full-rank experts, but extending the approach to low-rank adapters seems both more feasible and practical in many real-world settings.

### Questions
1. Given the assumption that D_0 contains sufficient public data that are representative of private client data, why not simply allow the cloud to train directly on the entire D_0? This baseline should be included to highlight the unique effectiveness of HarmoMoE.
2. Please discuss the relationship between HarmoMoE and low-rank adaptation unification methods (e.g., LoRASuite, NeurIPS 2025). While additional experiments are not mandatory, even a small-scale or illustrative experiment could strengthen the empirical validation.
3. What is the proportion of public versus private data used in the experiments?

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
This paper proposes a privacy-preserving training method for sparse Mixture-of-Experts models. To unify multiple expert models, each trained on separate private data, into a single MoE model, the paper proposes a proxy-data selection strategy (weighted DPP) and a context-aware router-training strategy to train the router within the unified model. The empirical results presented in the paper demonstrate that the proposed method outperforms previous baselines in privacy-preserving unification to MoE models.

### Strengths
1. The proposed method is logically designed

2. The empirical results outperform previous baselines

3. The paper is well-structured and easy to follow

### Weaknesses
1. **Underperformance**: The proposed method outperforms previous privacy-preserving unification baselines. But it still underperforms compared to separately finetuned models on private data. My concern is that, if the proposed unification method does not improve results after unification, what is the advantage of unification? In that case, each client can use their respective finetuned model and enjoy better performance.

2. **Potential suboptimal design**: The paper incorporated the context-aware router training, where the input tokens contain a component average over all the tokens of the input sequence, to capture the input context. Although the design choice improves performance over router training without the context-aware component, the design choice may be suboptimal.

### Questions
1. Can the authors explain why the proposed technique is advantageous despite having lower performance than the individually trained models on private data?

2. Can the authors discuss why the proposed context-aware design is optimal?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors present HarmoMoE, an approach to training a unified MoE model leveraging local client expertise while respecting data residency constraints. The approach addresses common challenges in training across heterogeneous data sources by introducing a proxy dataset that ensures commonality between local experts during training, enabling more effective model unification when the model is assembled and the router is trained. The approach differs from conventional federated learning in that it trains each local expert fully without coordinated optimization.  The work is similar to FlexOlmo with the main innovation being a proxy data selection method that enforces diversity, yielding better representativeness across the clients and their local training sets.

### Strengths
* Well-motivated and situated against prior work in this area.
* Good experimental setup and empirical validation.
* Strong clarity of presentation and results.

### Weaknesses
* The proxy selection approach is perhaps a marginal improvement over FlexOlmo. 
* Privacy benefits are limited or illusory - see my comments below.

### Questions
1. It is a bit of a stretch to claim privacy with this setup. It is true that you enforce data residency constraints, but by training and transmitting a local expert on the private data you are essentially communicating a compressed version of the private data that is highly vulnerable to attack. Many papers play fast and loose with this idea of privacy, but it would not meet criteria for privacy compliance in settings where this matters.

2. What is the relative computational cost of DPP vs the similarity-based method in FlexOlmo?

3. What assumptions are necessary about the proxy data?  What is the impact of having a client with strictly OOD data relative to the proxy set?

4. What client signals are needed for training the router? Is it just a question of minimizing the loss on the proxy data assuming frozen client experts?

5. Discuss the absence of an FL-based baseline evaluation.

6. As an additional nice-to-have baseline it might be interesting to train solely on the proxy data. 

7. Do you have an ablation where you don't perform final fine-tuning? How important is that step?

8. Are there any concerns about catastrophic forgetting in the final fine-tuning phase? How do you protect against this?  

9. Briefly clarify the difference between the two CLIP models tested.

10. You could get away with moving the large table of CLIP /32 results to the appendix as it doesn't add a lot to the discussion. Likewise for Llama-3b

11. In the experiment comparing with DPP- what is the baseline? Is it random sampling of proxy examples? I get the impression a slightly different set of experiments are depicted in Table 5 vs Fig 2- eg Table 5 has a row for FlexOlmo + DPP but Fig 2 has a figure for FlexOlmo with similarity-based sampling.

### Soundness
3

### Presentation
4

### Contribution
2
