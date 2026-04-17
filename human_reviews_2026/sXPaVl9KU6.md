# Heterogeneous Federated Fine-Tuning with Parallel One-Rank Adaptation

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
Large Language Models (LLMs) have demonstrated remarkable effectiveness in adapting to downstream tasks through fine-tuning. Federated Learning (FL) extends this capability by enabling collaborative fine-tuning across distributed clients using Low-Rank Adaptation (LoRA), while preserving data privacy by avoiding raw data sharing. However, practical deployments face challenges when clients have heterogeneous resources and thus adopt different LoRA ranks, leading to substantial initialization and aggregation noise that undermines performance. To address these challenges, we propose Fed-PLoRA, a novel lightweight heterogeneous federated fine-tuning (FFT) framework. Fed-PLoRA introduces Parallel One-Rank Adaptation (PLoRA), a new LoRA variant that replaces the classic multi-rank LoRA module with multiple parallel one-rank modules, and a novel Select-N-Fold strategy that folds untrained PLoRA modules into the pre-trained weights before local training, thereby accommodating heterogeneous client resources. We provide a unified analysis of initialization and aggregation noise of Fed-PLoRA and demonstrate how it addresses the limitations of state-of-the-art methods. Extensive experiments on diverse LLM fine-tuning tasks demonstrate that Fed-PLoRA consistently outperforms existing methods in both accuracy and efficiency. The code is available at \url{https://github.com/TNI-playground/Fed-PLoRA}.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes Fed-PLoRA, a framework for heterogeneous federated fine-tuning (FFT) of large language models. The key idea is to decompose each LoRA module into multiple parallel one-rank modules (PLoRA) and introduce a Select-N-Fold strategy that folds untrained modules into the frozen backbone to mitigate initialization and aggregation noise. The paper provides a unified noise analysis and reports performance gains over prior methods such as FLoRA, FlexLoRA, and HETLoRA on GLUE, Natural Instructions, and other datasets.

### Strengths
This paper addresses a timely problem of heterogeneous resource constraints in federated fine-tuning of LLMs. It proposes a modular reformulation, PLoRA, that can in principle generalize to other LoRA-based methods, and introduces a Select-N-Fold strategy to mitigate initialization and aggregation noise. The experiments span multiple models and benchmarks, providing empirical evidence of the proposed method’s efficiency.

### Weaknesses
1. The paper overlooks the training FLOPs and storage overhead of the frozen base model on the client side. From my understanding, each client fine-tunes only its LoRA adapters while keeping the base model parameters fixed. However, even though the base model is frozen, it is still fully involved in the forward passes. When the base model is large (e.g., LLaMA-3.1-8B), its forward FLOPs can dominate the total computation, likely exceeding the resource capacity of many local clients. This makes the proposed setup impractical for real-world federated environments.

2. The implementation code is currently unavailable. Releasing it would greatly improve reproducibility and allow the community to validate its efficiency.

### Questions
1. Please describe in more detail how the non-IID data distribution is constructed, and the definition of heterogeneity ratio.
2. For a fair comparison, the number of trainable parameters should be explicitly reported in the main experimental results. For instance, what are the trainable parameter counts for each baseline method presented in Table 1, Table 2, and Table 3? This information is critical for interpreting both the performance and communication efficiency.
3. How is the rank $r_i$ for the $i$-th client determined? Can $r_i$ be adaptively assigned according to the complexity of the client’s local task, or it only depends on the client resources. 
4. In Table 2, FLoRA performs near random guessing on the CoLA benchmark. Could the authors provide an explanation for this? Even though there exists large initialization noise in the stage of broadcast and initialization. 
5. What is the definition of the communication cost in Figure 5, please clarify it or provide the clear reference. Does the communication occur every local training iteration？

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper tackles the critical challenge of heterogeneous LoRA ranks in Federated Fine-Tuning (FFT). The authors identify that current methods suffer from initialization noise and aggregation noise due to rank mismatches. Their proposed solution, Fed-PLORA, uses PLORA (Parallel One-Rank Adaptation) to re-parameterize LoRA modules as a sum of parallel rank-1 components . This enables a novel Select-N-Fold strategy, where resource-constrained clients train a random subset of modules and "fold" the rest into the frozen weights. This method is theoretically shown to achieve zero initialization noise and is empirically demonstrated to consistently outperform existing heterogeneous FFT methods like FLORA, HETLORA, and FlexLoRA.

### Strengths
- **Clear and Compelling Motivation:** The paper is exceptionally well-motivated. It formalizes the *specific failure modes* of prior art: initialization noise and aggregation noise. The entire paper is a clear and focused effort to solve these two problems.
- **Strong Theoretical Analysis:** The noise analysis theoretically proves that the proposed Fed-PLORA framework eliminates initialization noise and provides a powerful and fundamental justification for the method's design.
- **Simple and Effective Methodology:** The proposed solution is both elegant and practical.
- **Comprehensive Empirical Validation:** The experimental results are strong and thorough. The authors test on a wide variety of models and diverse datasets.

### Weaknesses
- Adding pseudocode for the algorithm would improve clarity.
- It appears that PLoRA requires downloading the entire global LoRA model. In contrast, other methods—if rank is publicly available—can use much smaller downloads. Although Section 4.2 addresses this, the R − ri downlink cost could be quite high when R is large and ri is small, potentially causing synchronization issues.

### Questions
PLoRA folds the remaining R − ri untrained rank modules into the pretrained weight. Is this different from sparse LoRA tuning where the corresponding R − ri untrained rows are frozen?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper targets a problem of resource heterogeneity in Federated Learning (FL) for fine-tuning LLMs. A common solution for federated fine tuning for clients with different computational resources is to adapt LoRA modules of different ranks. The authors argue that this heterogeneity introduces two primary issues: initialization noise (when low-resource clients must truncate or discard parts of the global model) and aggregation noise (when the server attempts to combine modules of different dimensions).

To solve this, the paper proposes Fed-PLORA, a novel framework built on two core ideas: 1) Parallel One-Rank Adaptation  or PLORA where instead of having a standard rank-R LoRA module, they propose utilizing the  sum of R parallel one-rank modules. 2) Select-N-Fold Strategy which is A new initialization and training protocol.

The authors claim that the "Select-N-Fold" strategy completely eliminates initialization noise, as no information from the global model is 
discarded. They analyze the remaining aggregation noise and argue it is minimal. Through extensive experiments, they demonstrate that Fed-PLORA outperforms existing heterogeneous FFT methods.

### Strengths
* The paper is well-written and the authors did a good job explaining the existing problems. 

* Through various settings and different empirical results the authors show the merits of their algorithms. 

* The authors did a proper ablation study, explaining the importance of each component.

### Weaknesses
One important aspect of the paper is the Downlink Communication cost. The "Select-N-Fold" strategy has one clear limitation that is understated: downlink communication cost.

### Questions
Can you make a table (at least for one setting) to show all the new costs of your method and compare with the prior works?

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
1

### Summary
Fed-PLoRA addresses federated fine-tuning with heterogeneous LoRA ranks by proposing a new framework to mitigate both issues where rank mismatches introduce substantial initialization and aggregation noise. It introduces PLoRA, which replaces a single rank-R LoRA module with R parallel rank-1 modules that are mathematically equivalent to standard LoRA. Combined with a Select-N-Fold strategy, the method achieves zero initialization noise and reduces aggregation noise under heterogeneous budgets. Experiments across multiple LLM fine-tuning tasks (e.g., GLUE and instruction-following) show consistent accuracy gains over FLoRA, FlexLoRA, and HETLoRA, while avoiding the heavy SVD overhead that hurts communication and training time. It’s easy to adopt in practice though broadcasting all R modules does add some downlink cost.

### Strengths
1. This paper precisely defines initialization and aggregation noise in heterogeneous LoRA settings and shows Fed-PLoRA removes the former and reduces the latter.
2.  PLoRA’s parallel rank-1 decomposition is mathematically equivalent to standard LoRA yet naturally supports heterogeneity; paired with Select-N-Fold, it guarantees zero initialization noise while curbing aggregation noise. 
3. Strong and robust empirical results. Fed-PLoRA consistently outperforms FLoRA/FlexLoRA/HETLoRA across tasks (e.g., GLUE), and remains robust as client counts and rank distributions vary, including challenging non-IID scenarios.

### Weaknesses
1. The method broadcasts all R parallel rank-1 modules to every client and asks clients to keep folded modules, downlink traffic and on-device storage could become non-trivial in weak-network or mobile scenarios.
2. The paper’s empirical validation relies on relatively small or outdated and non-unified base models, which limits generalizability; it would be stronger to standardize on modern backbones like Qwen3 and Llama 3.2 across multiple sizes.
3. Its benchmark suite skews toward easier tasks (e.g., GLUE and basic instruction following) and should incorporate more rigorous reasoning/knowledge evaluations such as MMLU-Pro, GPQA, MuSR, MATH, IFEval, and BBH.
4. The authors do not clearly report the untuned base-model performance, obscuring absolute gains.

### Questions
See weakness.

### Soundness
3

### Presentation
3

### Contribution
2
