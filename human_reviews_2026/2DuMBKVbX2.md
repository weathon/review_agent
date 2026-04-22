# Mitigating Non-IID Drift in Zeroth-Order Federated LLM Fine-Tuning with Transferable Sparsity

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 2, 6, 6

## Abstract
Federated Learning enables collaborative fine-tuning of Large Language Models (LLMs) across decentralized Non-Independent and Identically Distributed (Non-IID) clients, but such models' massive parameter sizes lead to significant memory and communication challenges. This work introduces Meerkat, a sparse zeroth-order optimization (ZO) method designed for federated LLM fine-tuning. By limiting fine-tuning to a transferable, static, extremely sparse subset of parameters, Meerkat achieves remarkable communication efficiency, enabling cost-effective high-frequency synchronization. With theoretical analysis and experiments, we show that this high-frequency communication effectively mitigates Non-IID data challenges and leads to superior performance compared to full-parameter ZO. Furthermore, experimental results show that Meerkat outperforms existing sparsity baselines with better performance at the same communication frequency. To further handle Non-IID drift, Meerkat leverages traceable local updates and forms a virtual path for each client. This virtual path mechanism reveals the GradIP phenomenon: the inner products between LLM pre-training gradients maintained by server and client gradients estimated via ZO converge for extreme Non-IID clients but oscillate for IID ones. This distinct behavior provides a signal for identifying clients with extreme data heterogeneity.
Using this signal, Meerkat-vp is proposed to analyze GradIP trajectories to identify extreme Non-IID clients and applies early stopping to enhance aggregated model quality. Experiments confirm that Meerkat and Meerkat-vp significantly improve the efficiency and effectiveness of ZO federated LLM fine-tuning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes MEERKAT, a novel framework for federated fine-tuning of LLMs that combines Zeroth-Order optimization with extreme sparsity. The core idea is that this combination is extremely memory-efficient and communication-efficient. The authors claim this efficiency enables high-frequency client-server synchronization, which in turn is the key to mitigating Non-IID client drift. Furthermore, the paper introduces a "virtual path" mechanism, where the server can reconstruct client update trajectories, and use"GradIP", a metric based on the inner product of client and pre-training gradients, that can identify clients with extreme Non-IID data. The authors then propose MEERKAT-VP, an extension that uses this signal to apply early stopping to these problematic clients, further improving global model performance.

### Strengths
1. **Strong Empirical Performance:** The method is well-validated empirically. Empirical results clearly show that MEERKAT consistently and significantly outperforms full-parameter ZO and LoRA-based ZO baselines, especially in Non-IID settings. 
2. **Comprehensive Analysis:** The paper provides strong support for its claims through both theoretical convergence analysis and extensive empirical ablations. The validation is structured clearly around three main claims, and each is convincingly argued with evidence.
3. **Novel Mechanism for Heterogeneity:** The discovery of the "GradIP" phenomenon is a novel contribution. Using this signal, which is enabled by the virtual path, to identify and manage extreme Non-IID clients (MEERKAT-VP) is a clever and effective FL-native strategy for handling heterogeneity.

### Weaknesses
The paper's primary strategy for handling extreme Non-IID clients (MEERKAT-VP) is purely defensive. It aims to "mitigate their adverse impact" by identifying them via GradIP and applying early stopping. This implies that the base MEERKAT method is still vulnerable to significant client drift from these clients. This strategy is inherently limited: it focuses on reducing the participation of "bad" clients rather than developing a mechanism to effectively learn from their (potentially valuable) information.

### Questions
The core optimization (sparse ZO on a static mask) seems to be a general technique, not specialized for FL. Are there any other properties of this sparse ZO method that make it uniquely suited for the federated setting, beyond the communication efficiency gains and the virtual path mechanisms they enable?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes MEERKAT, a sparse zeroth-order optimization method for federated LLM fine-tuning. By updating a static, highly sparse subset of parameters, MEERKAT reduces communication costs and enables frequent synchronization to alleviate Non-IID drift. It also introduces a “virtual path” mechanism that uncovers the GradIP phenomenon—the inner product between server and client gradients varies across data heterogeneity. Building on this, MEERKAT-VP detects extreme Non-IID clients and mitigates their impact through early stopping. The paper provides theoretical convergence results and validates the method with extensive experiments.

### Strengths
* The paper is clearly presented, systematically proposing three distinct claims and validating them one by one.
* It includes a detailed convergence analysis for the proposed methods.
* The experimental evaluation is comprehensive, covering various benchmarks and recent LLMs.

### Weaknesses
* The use of sparse updates for communication efficiency in FL is a well-explored area, often framed as model compression. The paper should provide a more thorough comparison with this body of work to clarify the unique contributions of applying sparsity specifically to ZO-based FL fine-tuning. Applying sparsity to new FL models (here is federated fine-tuning) seems not a strong contribution.
* Methods for tackling the Non-IID issue in compressed FL have also been developed, such as the one in [a]. The paper would be stronger if it compared its GradIP-aware solution not just with other non-sparse fine-tuning methods, but also with compression-aware FL baselines that explicitly address Non-IID data, even if they are not designed for LLMs.
* The rationale for setting the number of local training steps to one for identified Non-IID clients is not fully convincing. These clients may hold important minority data that is crucial for preventing the global model from becoming biased. You should not move it only because it is minority and hard to converge into the global model. The paper should better justify why this aggressive early stopping is preferable to other strategies that might still leverage the clients' unique data more fully. 

[a] Huang, Xinmeng, Ping Li, and Xiaoyun Li. "Stochastic Controlled Averaging for Federated Learning with Communication Compression." The Twelfth International Conference on Learning Representations.

### Questions
* The client identification mechanism in MEERKAT-VP uses an "initial phase" and a "later phase" of local training to analyze GradIP scores. What is the motivation for this two-phase design?
* Figure 3 shows the GradIP trajectories. To make the effect of the proposed solution more intuitive, could the authors add a plot showing the trajectory for a client being managed by MEERKAT-VP (i.e., after mitigating the global weight changes after the design)?

### Soundness
2

### Presentation
3

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
This paper introduces MEERKAT and its extension MEERKAT-VP, a novel methodology to tackle the twin challenges of high communication overhead and Non-IID data drift when fine-tuning LLMs in a Federated Learning setting. MEERKAT addresses efficiency by using a sparse Zeroth-Order Optimization method that limits fine-tuning to an extremely sparse (e.g., <0.1%), static subset of parameters, drastically reducing communication costs and enabling high-frequency server synchronization to mitigate drift. Building on this, MEERKAT-VP uses the concept of a virtual path based on local updates to detect the GradIP phenomenon in extreme Non-IID clients, subsequently applying early stopping to restrict these clients' local steps, thereby improving global model quality. The experiments confirm that MEERKAT, through its combined approach of extreme sparsity and strategic client management, effectively mitigates Non-IID data challenges and achieves superior performance and efficiency compared to existing baselines.

### Strengths
i.MEERKAT employs an extremely sparse and static subset of parameters for fine-tuning. This drastically reduces the communication load and memory consumption on client devices, addressing the primary scalability bottleneck of LLMs in FL.

ii.The extreme sparsity enables cost-effective high-frequency client-server synchronization. This high synchronization rate is key to effectively suppressing the client drift caused by Non-IID data distributions across decentralized clients.

iii.The paper proposes using a static sparsity mask identified through gradients from pre-training data. This transferable sparsity ensures that the chosen small subset of parameters is highly effective for downstream tasks, allowing for consistent performance throughout the FL process.

iv.MEERKAT-VP introduces the Virtual Path mechanism, which allows the server to diagnose the severity of Non-IID data without accessing the raw client data.It utilizes the discovered GradIP phenomenon to strategically identify and apply early stopping to severely Non-IID clients. This ensures robustness and convergence quality while maintaining data privacy.

v.The work provides strong experimental evidence demonstrating that MEERKAT consistently outperforms full-parameter Zeroth-Order Optimization and other SOTA sparse fine-tuning methods under various Non-IID settings.

vi.The use of Zeroth-Order Optimization , combined with extreme sparsity, makes the approach highly suitable for clients with limited computational resources, as it avoids the need to calculate full, complex second-order gradient information.

### Weaknesses
i.In the description of the steps in Figure 1 of the paper, the word "Aggregrate" appears, and the correct spelling should be "Aggregate".In the phrase on lines 153–154, "(3) Sever aggregates and initiate the next round...", shouldn't the word "Sever" actually be spelled "Server"?Line 165 contains a redundant repetition of the article "a" in the phrase "and a a new seed list."

ii.Although the paper validates the effectiveness of the extreme sparsity level of 0.1%, it fails to deeply analyze the performance impact of varying sparsity levels (such as 0.01%, 0.5%, 1%) across different scales of LLMs (e.g., larger models like 7B/13B) and different task types (e.g., complex reasoning tasks versus simple classification tasks). Furthermore, the transferability of the sparsity mask was only tested on a few domain shift datasets (C4, Wiki, Code datasets) and was not verified for its adaptability in highly specialized domains (such as legal), thus making it difficult to support a conclusion of broader applicability.

iii.The paper compares MEERKAT/MEERKAT-VP against Full-FedZO, LoRA-FedZO, and the baseline FedDYN improvement method. However, it fails to include state-of-the-art Federated LLM fine-tuning methods specifically designed for Non-IID scenarios. These missing baselines include established methods adaptable for LLMs, such as FedAvgM and SCAFFOLD, or FedSparse. This omission makes it difficult to ascertain whether the performance advantages of the proposed method truly surpass the latest research achievements.

### Questions
i.Would the performance of MEERKAT/MEERKAT-VP significantly decline as the number of clients increases to a larger magnitude? For example, would the server-side virtual path reconstruction incur a higher computational overhead with an increased number of clients, and has the potential reduction in the efficiency of MEERKAT-VP's early stopping strategy been considered if a large number of clients are categorized as "extreme Non-IID"?

ii.The early stopping strategy of  MEERKAT-VP relies on a threshold for identifying the GradIP phenomenon. How is this threshold determined? Should the paper analyze the model performance's sensitivity to this threshold choice, and potentially propose an automatic or adaptive mechanism for threshold selection?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a sparse zeroth order federated fine tuning framework for LLMs that is both theoretically grounded and highly practical. MEERKAT uses a transferable 0.1 percent sensitivity mask and frequent synchronization to tame memory and bandwidth while reducing Non IID drift. Virtual path reconstruction and the GradIP score then allow the server to detect extreme Non IID clients and actively limit their damage through MEERKAT VP. Experiments across multiple open LLMs and multiple classification and reasoning benchmarks show consistent gains over strong baselines like Full FedZO and LoRA FedZO at matched communication frequency, along with dramatic savings in memory and bandwidth.

### Strengths
1. Virtual path and GradIP are novel ideas. By reconstructing each client’s local update trajectory, the server gains visibility into how that client is moving in parameter space without seeing private data. GradIP then becomes a quantitative signal that reveals which clients are extreme Non IID.

2. The paper provides convergence analyses for both MEERKAT and MEERKAT VP.

2. Strong empirical validation. The experiments cover multiple open LLMs (Llama 3.2 1B, Qwen2 1.5B, Gemma2 2B), multiple tasks (SST2, AgNews, Yelp polarity, BoolQ, RTE, WSC, WiC), both IID and strongly Non IID Dirichlet splits, and multiple baselines

### Weaknesses
1. hyperparameters in MEERKAT VP : The early stopping rule uses thresholds on GradIP phase ratios and quiescent duration. Although a small sensitivity study is reported, it is still unclear how practitioners should pick these thresholds for new tasks with no oracle labels. 


2. Baselines in experiments:  There is active work on federated LoRA under heterogeneous clients, which explicitly tackles aggregation noise, knowledge contamination, and aggregation distortion by separating global and client specific structure or by using rank adaptive aggregation. The paper mentions LoRA FedZO but does not deeply compare against these newer structured aggregation approaches, so it is hard to see whether MEERKAT VP replaces them, complements them, or could be combined with them. Existing works are listed below: 

[1] Zhe Li, Bicheng Ying, Zidong Liu, Chaosheng Dong, and Haibo Yang. Achieving dimension-free
communication in federated learning via zeroth-order optimization. In The Thirteenth International
Conference on Learning Representations, 2025.

[2] Youbang Sun, Zitao Li, Yaliang Li, and Bolin Ding. Improving loRA in privacy-preserving federated
learning. In The Twelfth International Conference on Learning Representations, 2024.

[3] Pengxin Guo, Shuang Zeng, Yanran Wang, Huijie Fan, Feifei Wang, and Liangqiong Qu. Selective
aggregation for low-rank adaptation in federated learning. In The Thirteenth International
Conference on Learning Representations, 2025.

3. Lack of explicit adversarial evaluation: MEERKAT VP is essentially throttling clients whose updates are harmful. This is similar to defending against malicious or poisoned clients. However, the experiments do not include adversarial threat models like label flipping or Byzantine behavior. Showing that GradIP based early stopping also mitigates such attacks would significantly strengthen the robustness story.

### Questions
see weakness 1, 2, 3

### Soundness
4

### Presentation
4

### Contribution
4
