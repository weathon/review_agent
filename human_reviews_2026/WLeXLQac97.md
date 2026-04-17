# Branching Memory: Task-Specific Expansion for Continual Learning in Large Language Models

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 2, 2

## Abstract
Large Language Models (LLMs) face the challenge of catastrophic forgetting in continual learning scenarios, where learning new tasks often overwrites previously acquired knowledge, leading to performance degradation and limiting their applicability in dynamic task environments. Existing approaches can be categorized into rehearsal-based, regularization-based, and architecture-based methods. Among these, architecture-based methods are more suitable for LLMs as they dynamically adjust model structures to handle large-scale parameters and task interference. However, existing methods often struggle with parameter efficiency and fail to fully leverage the Transformer architecture's characteristics.
In this work, we propose Branching Memory, a novel method that leverages the organization of knowledge within transformer models. By modeling knowledge as key-value (KV) representations within the FFN layers, our approach dynamically allocates dedicated capacity for new tasks, allowing the model to store and integrate task-specific knowledge without overwriting existing information. To further improve knowledge retention and reduce task interference, we employ an orthogonality-based regularization strategy to stabilize training and minimize parameter conflicts.
Experimental results on standard continual learning benchmarks demonstrate that Branching Memory achieves superior performance with enhanced parameter efficiency. On short-sequence tasks with T5-Large, Branching Memory with regularization achieves 76.6\% average accuracy, outperforming baseline methods. Extended evaluations on LLaMA2-7B and 15-task long sequences validate the method's scalability and effectiveness across different model architectures and task lengths. The method's practical advantage lies in its balanced trade-off between performance, parameter efficiency, and inference simplicity in continual learning scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes Branching Memory, a continual learning framework designed to alleviate catastrophic forgetting in large language models by introducing task-specific branches within the Feed-Forward Network (FFN). The approach reinterprets the FFN as a Key-Value memory module, enabling each task to store and retrieve its own representations independently. It further employs orthogonality regularization to maintain separation between task-specific parameters and preserve prior knowledge. Experiments demonstrate that Branching Memory achieves performance gain across multiple continual learning benchmarks.

### Strengths
- The paper clearly articulates the motivation behind its study by emphasizing the limitations of existing continual learning methods, such as catastrophic forgetting and task interference when applied to large language models.
- Comprehensive experiments with nine baselines and some variants (initialization and regularization).

### Weaknesses
- **Insufficient contribution from the existing methods**
  - The paper does not clearly differentiate the proposed Branching Memory mechanism from existing LoRA-based adaptations applied to FFN layers.
  - Both approaches introduce low-rank or task-specific parameter additions within the network, and LoRA can already be flexibly applied to both MHA and FFN modules depending on the setting. While the paper reports performance improvements over LoRA based methods, it lacks a theoretical or structural analysis explaining where these gains stem from.
  - Also interpreting FFN as Key-Value memory is not new, Geva et al. (2020) had already introduced the concept.
- **Missing report of Forward Transfer (FWT)**
  - The paper reports only Average Accuracy (AA) and Backward Transfer (BWT).
  - Since each branch is completely isolated, the features learned in previous branches cannot be reused or adapted by subsequent tasks. This creates some concerns that the separation would make positive transfer to be difficult, which should be observed through the FWT report.
- **Limited advantages over conventional methods**
  - Unlike other conventional approaches, this work freezes Multi-Head Attention (MHA) layers, which the model cannot perform fine-grained adjustments at the attention level during downstream fine-tuning.
- **Unclear criterion for setting λ and δ parameters**
  - The paper compares variants with initialization (δ parameter) and regularization (λ parameter) strategies, however does not specify the details, which makes it difficult to assess the fairness and robustness of the experiment.
  - λ varies across different task sequences, but does not explain how these values are determined. E.g., why λ=0.001 for tasks 3-4 in Order 1-3, but λ=0.0005 for tasks 3+ in Long-Order?
  - The paper mentions using QR decomposition followed by scaling with "a diagonal matrix with hyperparameter δ" but never specifies what δ values were used or how they were selected.
- **Questionable claims of “task diversity”**
  - The paper claims to evaluate on "diverse tasks", but exclusively tests on text classification problems.
  - Therefore fails to articulate what makes their benchmark a challenging continual learning problem, since all tasks are essentially text classification with similar processing requirements from general web domains.

### Questions
Figure 3's caption incorrectly refers to "short-sequence continual learning benchmark" which actually is long-sequence experiments.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes the Branching Memory method to address catastrophic forgetting by dynamically allocating additional key-value (KV) pairs per task, expanding the intermediate dimension of the FFN. By incorporating orthogonal regularization and initialization strategies, the method effectively reduces interference between tasks.

### Strengths
The paper is clearly written and easy to follow.

### Weaknesses
The main issue with this paper is that the proposed method closely resembles O-LoRA. The core improvements in this paper are:

1. The introduction of the "lightweight memory branch" concept
2. An initialization strategy
3. A regularization strategy.

However, points 1 and 3 appear to be similar to O-LoRA, and I also have some concerns regarding point 2, as outlined below:

* For point 1: the "lightweight memory branch" concept is essentially a dynamic architecture-based method. Previous approaches, such as the O-LoRA baseline you compare against, also allocate task-specific trainable parameters (LoRA) and do not differ significantly from your approach, aside from the fact that the new parameters are added in the attention layer in your method, whereas O-LoRA adds them to the MLP layer.

* For point 3: the regularization strategy in this paper is not novel, as similar techniques have been used in previous work.

* For point 2: regarding the initialization strategy, the proposed orthogonal initialization may limit the learning space for new tasks, especially when tasks are similar. Enforcing strict orthogonality could prevent the model from exploring the optimal parameter space. It would be useful to consider whether a less restrictive initialization approach could improve performance, particularly for related tasks.

* Additionally, you do not analyze different initialization strategies in the experiments.

* The baseline comparisons are outdated; the most recent baseline you compare to, O-LoRA, is from 2023, making it difficult to effectively demonstrate the method's validity. It would be helpful to compare against more recent work.

* I also suggest comparing your proposed dynamic architecture-based method with fixed capacity-based methods, such as model-merging-based methods (e.g., memory-free TaSL [1] and memory-based Recurrent-KIF [2] ), where model parameters do not expand with the addition of new tasks.

[1] Tasl: Continual dialog state tracking via task skill localization and consolidation

[2] Kif: Knowledge identification and fusion for language model continual learning

### Questions
All the comments have been raised in the weaknesses section, please refer to the points above.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes Branching Memory, a novel framework for scalable and parameter-efficient continual learning. The key idea is to reinterpret the Feed-Forward Network (FFN) layer as a key–value representation, allowing task-specific knowledge to be integrated through lightweight branch expansion without overwriting existing information in the model. In addition, orthogonality-based regularization and task-aware initialization methods are employed to improve training stability and overall performance. The proposed method is evaluated on a short-sequence task scenario with five datasets and a long-sequence task scenario with fifteen datasets.

### Strengths
- This paper is easy to read.
- The proposed method achieves efficiency by freezing the existing model weights and only training task-specific lightweight FFN branches.
- The experimental results show performance gain compared with existing methods, including O-LoRA, SeqLoRa, and L2P.

### Weaknesses
W1. Lack of Quantitative Evidence for Parameter Efficiency
- The paper emphasizes parameter efficiency as one of its key advantages; however, no quantitative analysis is provided to support this claim.
- It would strengthen the argument if the authors included concrete measurements of branch memory size, computational overhead, and training/inference time, compared with existing continual learning methods. Such evidence is essential to validate the proposed framework’s claimed efficiency.

W2. Scalability Concerns Regarding Task-Specific Memory Expansion
- The proposed approach introduces a separate memory branch for each task, resulting in parameter growth that scales linearly with the number of tasks.
- Without clear numerical analysis of computational and memory costs, it remains uncertain whether the method can scale effectively to long-term continual learning scenarios involving dozens or hundreds of tasks. This lack of scalability assessment limits confidence in its practical applicability.

W3. Unclear Inference Procedure for Branch Selection
- The paper does not clearly specify how the model selects or activates the appropriate memory branch during inference.
- Since each task maintains an independent branch, it remains unclear how the system determines which branch to use for unseen or mixed-task inputs, leaving an essential component of the framework underdefined.

W4. Outdated and Insufficient Baseline Methods
- The paper evaluates Branching Memory against MTL, SeqFT, SeqLoRA, IncLoRA, Replay, EWC, LFPT5, L2P, and O-LoRA.
- However, many of these methods are outdated and do not represent the current state of the art in continual learning.
Recent approaches such as InfLoRA, GainLoRA, and CLoRA exhibit significantly stronger and more stable performance, yet are not included in the comparison.
- Moreover, while comparisons with O-LoRA are provided, Branching Memory does not consistently outperform O-LoRA, particularly in its regularized form, and no analysis is given for the marginal performance differences.
The absence of stronger baselines (e.g., Mixture-of-Experts, MoE-based continual learners) limits the credibility of the superiority claims.

W5. Limited Experimental Rigor and Missing Evaluation Metrics
- Sections 5.2.2 and 5.2.3 contain restricted baseline diversity and omit several listed baselines entirely.
- Additionally, the Backward Transfer (BWT) metric for Branching Memory is not reported, and the omission is not justified.
- These issues collectively weaken the empirical rigor and make it difficult to assess the robustness of the proposed approach.

W6. Outdated and Narrow Benchmark Selection
- The evaluation relies solely on text classification datasets (AG News, DBpedia, Amazon/Yelp Reviews, Yahoo Answers, GLUE/SuperGLUE subsets).
- While standard in earlier continual learning research, these datasets are relatively outdated given current LLM capabilities.
- Including more diverse and challenging benchmarks—such as commonsense reasoning, mathematical reasoning, or multimodal tasks—would better demonstrate the method’s generalizability and robustness.

W7. Marginal Overall Performance Gains
- Although the proposed Branching Memory method achieves improvements over several baselines, the absolute gains are modest—typically within one to two percentage points of strong baselines such as O-LoRA.
- These limited improvements make it difficult to argue for substantial advancement over existing approaches.

### Questions
- It would be better to include comparisons with more recent and stronger methods such as InfLoRA, GainLoRA, and CLoRA to provide a fair and up-to-date evaluation.
- It would be better to expand the experimental evaluation to include more baselines and other recent state-of-the-art continual learning methods.
- It would be better to report the BWT results in Section 5.2.3, or provide an explanation for missing metrics to ensure a complete evaluation.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes an architecture-based continual learning approach, BM, to mitigate catastrophic forgetting in LLMs. Building on prior work that views FFNs as key–value memory structures, BM expands this memory by freezing the pretrained backbone and adding task-specific lightweight branches. To further reduce task interference, the method applies orthogonality-based initialization and regularization, improving knowledge retention across tasks.

### Strengths
1. The paper extends prior work that interprets Transformer FFNs as key–value memory structures and introduces a simple yet effective architecture that isolates task-specific knowledge through lightweight memory branches while keeping the pretrained backbone frozen.
2. This design achieves clear task decoupling and provides a concise solution for continual learning in large language models.

### Weaknesses
1. The paper lacks a clear explanation of how the proposed mechanisms align with its stated objectives, and several details of the experimental setup remain ambiguous.
2. The evaluation is limited to short-sequence benchmarks and primarily conducted on the T5-Large model (≈770M parameters), which restricts the generalizability to larger LLMs.
3. While scalability and computational cost become potential concerns as new tasks introduce additional parameters, the paper provides no analysis of the trade-off between performance and these costs.
4. The absence of statistical significance testing weakens the reliability of the reported results. Moreover, critical hyperparameters in the initialization strategy (e.g., $\sigma^{2}$ and $\delta_{i}$) and their selection criteria are omitted, hindering reproducibility.
5. Minor inconsistencies are present: the dimension notation ‘$d$’ in Table 3 appears to correspond to ‘$r$’ in Section 4.1, and Figure 3’s caption incorrectly labels a long-sequence result as ‘short-sequence’.

### Questions
1. It is unclear whether the LoRA rank and overall parameter count were matched between BM and baseline methods such as IncLoRA, which is essential for a fair comparison.
2. Since BM (Init.) and BM (Reg.) represent complementary strategies, it would be valuable to examine their combined effect (BM + Init. + Reg.). Why was this joint configuration not considered?

### Soundness
2

### Presentation
2

### Contribution
2
