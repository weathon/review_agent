# $1+1<1$? Breaking the Standalone Barrier in Federated Fine-Tuning of Multimodal Large Language Models under Non-IID Data

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Federated fine-tuning of multimodal large language models faces significant challenges in communication costs, which can be addressed by Low-Rank Adaptation (LoRA). Existing methods typically allow all clients to collaboratively learn and share a single LoRA adapter. However, we identify a long-overlooked issue: under non-IID data, federated fine-tuning can even underperform standalone local training ("$1+1<1$"). Strikingly, much of the literature still focuses on surpassing SOTA Federated Learning (FL) methods, while neglecting the more fundamental requirement that any effective FL approach should at least outperform standalone local training. To address this, we propose a novel method termed Federated Mixture of LoRA Experts (Fed-MoLE). It adopts a hybrid mixture-of-LoRA-experts architecture with an alternating disentanglement–alignment mechanism. This design enables the model to disentangle diverse instance-level variations through dynamically routed LoRA experts, and then align cross-client knowledge into a unified global representation, thus enhancing robustness under non-IID data. Extensive experiments on two benchmarks show that Fed-MoLE consistently surpasses both SOAT FL baselines and standalone local training, effectively breaking the "$1+1<1$" barrier in federated fine-tuning of multimodal large language models under non-IID data.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper focuses on a fundamental issue in federated fine-tuning: under non-IID data, some federated fine-tuning methods perform even worse than standalone local training. To enhance the effectiveness of federated learning, the authors introduce a Mixture of Experts (MoE) framework. Specifically, each client includes both shared and specialized experts, which are trained alternately, the specialized experts capture instance-level variations, while the shared expert integrates common knowledge across clients. Experiments on multiple datasets demonstrate that the proposed method not only significantly improves federated fine-tuning performance but also successfully overcomes the limitation where federated training underperforms local training.

### Strengths
This method reveals that existing federated fine-tuning approaches can sometimes perform worse than standalone local training, a finding that is highly significant for federated learning.

### Weaknesses
1. The method shows limited innovation, as applying MoE to federated LoRA fine-tuning has already been explored in several studies (e.g., [a–d]); the main difference here seems to lie only in the use of an alternating training strategy.

[a] Le, Khiem, et al. "FLAME: Towards Federated Fine-Tuning Large Language Models Through Adaptive SMoE." arXiv preprint arXiv:2506.16600 (2025).
[b] Wang, Lei, et al. "Adaptive LoRA Experts Allocation and Selection for Federated Fine-Tuning." arXiv preprint arXiv:2509.15087 (2025).
[c] Hu, Gang, et al. "FFT-MoE: Efficient Federated Fine-Tuning for Foundation Models via Large-scale Sparse MoE under Heterogeneous Edge." arXiv preprint arXiv:2508.18663 (2025).
[d] Chen, Fahao, et al. "Federated Fine-Tuning of Sparsely-Activated Large Language Models on Resource-Constrained Devices." arXiv preprint arXiv:2508.19078 (2025).

2. As shown in Table 1, the performance improvement over FedSA-LoRA is relatively small. The introduction of MoE in clients provides only marginal gains while increasing computational and communication costs.
3. Although the discovery of the “1 + 1 < 1” phenomenon is interesting, the paper lacks further investigation into its causes — for example, whether data heterogeneity or LoRA configuration contributes to the inferior performance of federated LoRA fine-tuning compared to local training.
4. Additional ablation studies are recommended for the client-side MoE design in LoRA, such as analyzing the impact of the number of specialized experts and the rank of LoRA on model performance.

### Questions
The main issue lies in the lack of methodological innovation. Many studies have already applied MoE to federated LoRA fine-tuning, and this work merely adds an alternating training strategy. The performance improvement is limited, and the ablation studies are insufficient.

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
4

### Summary
In federated fine-tuning of MLLMs , the substantial communication overhead can be effectively mitigated through LoRA. Existing approaches typically allow all clients to share and collaboratively train a unified LoRA adapter. However, this paper finds that under non-IID conditions, federated fine-tuning often performs worse than training locally on each client, a phenomenon the authors refer to as “1+1 < 1.” To address this issue, the authors propose Fed-MoLE, a framework that employs dynamically routed LoRA experts to capture instance-level variations while utilizing a shared LoRA across all clients to extract a unified global representation. The two types of LoRA are trained in an alternating manner to achieve a balance between shared and personalized learning. Experiments on Fed-VQA and FedMed datasets demonstrate that Fed-MoLE achieves superior performance compared with existing methods.

### Strengths
1. Given the growing trend of data decentralization and the rapid development of large models, federated fine-tuning for MLLMs is a research topic of reasonable relevance.
2. The paper is generally well-structured, starting from an observed phenomenon and gradually introducing the proposed method in a logically coherent way.
3. The experiments are relatively complete, including multiple baseline comparisons that provide some validation for the proposed model’s effectiveness.

### Weaknesses
See Questions section.

### Questions
**Novelty**

- The logical connection between the reported “1+1 < 1” phenomenon and the proposed method is not sufficiently strong. The proposed framework still largely relies on the concept of shared parameters; overall, it appears more like introducing a direct comparison with *Standalone* in the experiments to highlight performance gains. It would be helpful for the authors to further clarify the causal relationship between this phenomenon and the design of their method.
- Although all local samples pass through the shared LoRA module, the paper does not provide a clear mechanism or experiment verifying that this module indeed captures cross-sample shared features. Likewise, there is no direct evidence showing how the routed LoRA models instance-level variations. The authors are encouraged to demonstrate the differences between the features captured by the two types of LoRA modules.
- Experimental results show that GFL typically underperforms *Standalone*, while PFL methods generally outperform *Standalone*. This suggests that under non-IID conditions, PFL methods may serve as a more appropriate comparison baseline. In this case, the emphasis on the “1+1 < 1” phenomenon may be less critical for evaluating the proposed method’s contribution.

**Cost**

- In line 268, the authors state, “This sparse structure significantly reduces the additional computational overhead when *N* is larger.” It is recommended to include quantitative results showing how computational cost varies with the number of experts $N$ and $K_r$.

**Details**

- The paper states that Fed-MoLE is applied only to the up_proj layer, while other layers use standard LoRA implementations. It would be helpful if the authors could explain the reasoning behind this design choice.
- Equation (6) may contain a typographical error; it should likely be written as $\sum^N_{i=1}\rho_iB^{r,t}_{k,i}A^{r,t}_{k,i}X$

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
5

### Summary
The paper identifies a key issue in federated fine-tuning of multimodal large language models that under non-IID data, standard approaches can perform worse than local training alone, a phenomenon referred to as the “1+1<1” problem. To address this, the authors propose Fed-MoLE, which employs a mixture of LoRA experts with dynamic routing to capture client-specific variations and integrates an alternating disentanglement–alignment mechanism to unify cross-client knowledge. Experiments demonstrate that Fed-MoLE consistently outperforms both state-of-the-art federated learning methods and standalone local training, effectively overcoming the “1+1<1” barrier.

### Strengths
1.Highlights and rigorously addresses the underappreciated “1+1<1” failure mode in federated fine-tuning—ensuring FL methods at least beat local baselines.
2.Introduces a hybrid mixture-of-LoRA-experts design that balances personalization and collaboration, where the alternating disentanglement–alignment strategy effectively handles instance-level heterogeneity while promoting global consistency.
3.Strong experimental results on multimodal benchmarks showing consistent gains over both SOTA FL methods and local training under non-IID settings.

### Weaknesses
1.Regarding the “1+1<1” issue: most existing methods include ablation studies on the number of participating clients, typically concluding that performance improves as more clients join, which directly contradicts the authors’ claim of a “1+1<1” phenomenon.
2.The combination of Mixture-of-Experts (MoE) and LoRA has been widely explored in prior work. The proposed Fed-MoLE method appears to lack novelty, as it merely integrates MoE with LoRA, and the alternating disentanglement–alignment strategy is relatively simple and generic.
3.The paper claims that the alternating disentanglement–alignment mechanism reduces client drift and enhances global representation. However, it only explains that disentanglement captures shared patterns across samples, without clarifying how this mechanism specifically mitigates client drift.
4.In Table 1, which compares Fed-MoLE against state-of-the-art methods, its performance is nearly identical to, or even slightly worse than that of FedSA-LoRA. The authors do not provide any analysis or justification for this marginal (or negative) performance gap.

### Questions
1.In Figure 1, the authors compare the performance of the standalone (independent) approach against other FedLoRA methods. How was the test set split for this comparison? Can the fairness of this comparison be guaranteed? Please also explain Weakness 1.

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
The paper studies federated fine-tuning of multimodal large language models under non-IID conditions. The authors argue that the common “share a single LoRA adapter” approach can yield “1+1<1” in strongly non-IID settings, federated training may even underperform standalone fine-tuning on each client. They propose Fed-MoLE: at each target layer, introduce a mixture-of-experts structure with a shared LoRA plus multiple routed LoRA experts. Training alternates in a decoupled manner, first update routed experts and the router, then only update the shared LoRA; only the shared LoRA is aggregated. On Fed-VQA (5 clients) and Fed-Med (3 clients with task heterogeneity), the method reportedly outperforms Standalone and several FL baselines, with no extra communication cost and only modest VRAM overhead.

### Strengths
1. Clear and practically meaningful problem setup. The paper emphasizes a sensible baseline principle—federated fine-tuning should at least not underperform Standalone—surfacing an evaluation pitfall overlooked by many works. The “client drift / representation misalignment” diagnosis (variance and norm traces) is intuitive and useful. 
2. Simple and effective method. Aggregate only the shared LoRA, leaving instance-level heterogeneity to sparse routed experts. Alternating freezing reduces gradient conflicts; the engineering recipe looks implementable (with pseudocode and practical notes).
3. Efficiency considerations. Communication and VRAM comparisons are laid out; communication matches standard LoRA-FL and VRAM grows only modestly.

### Weaknesses
1. Insufficient theoretical support. The paper does not clearly state modeling assumptions for the non-IID setting (e.g., bounds on gradient dissimilarity or a “shared subspace + client-specific shift” decomposition), making it difficult to delineate under what conditions “aggregating only the shared LoRA” is effective; the training dynamics lack convergence/stability guarantees—especially under federated settings with partial participation, asynchrony, and distribution shift—so error bounds and the radius of convergence remain unclear; the interaction between the shared LoRA and expert LoRA, as well as the impact of routing misassignment rates on optimization and generalization, is not quantitatively analyzed; moreover, the absence of generalization-error or sample-complexity bounds leaves the expected gains under task heterogeneity and data imbalance unexplained.

2. Notational issues. At least in Eq. (6), the routed expert term appears as B^r B^r X,which likely should be B^r A^r X under LoRA. 	Clarifications needed on design and evaluation details. The paper under-specifies the router’s architecture, its inputs, the regularization terms, and any perturbation strategies; moreover, it does not analyze the stability of Top-K sparse routing under distribution shift. The claim of “no extra communication cost” also requires stricter accounting—including the sizes of shared vs. local parameters, the actual transmitted bytes under sparse activation, and the impact of optional quantization/compression. As presented, the evidence is largely qualitative and based on single-round measurements. 
3. Comparisons to personalized/clustered/expert-style FL are incomplete. The paper mainly compares with global/personalized LoRA-FL variants; however, MoE or clustering-style personalization is not new in FL. Missing strong baselines from this line dilutes the novelty claim.

### Questions
see weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
