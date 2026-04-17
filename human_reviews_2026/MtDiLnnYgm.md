# HippoTune: A Hippocampal Associative Loop–Inspired Fine-Tuning Method for Continual Learning

- Decision: Accept (Poster)
- Scores: 8, 8, 4, 6

## Abstract
Studies have shown that catastrophic forgetting primarily stems from the difficulty of reactivating old memories; although parameter-efficient fine-tuning can mitigate forgetting while keeping most model parameters frozen, it still falls short in fully reawakening knowledge of prior tasks. In contrast, humans can efficiently retrieve and flexibly integrate existing experiences when learning new tasks, thereby maintaining stable performance on earlier ones. During cognition, the hippocampal EC–DG–CA3–CA1 circuit engages in multiple rounds of associative recall, and its pattern-separation and memory-completion mechanisms excel at activating historical information. Inspired by this mechanism, we propose HippoTune, a latent-space iterative retrieval strategy that embeds a query–retrieve–feedback loop within each Transformer layer. Starting from the hidden state as an initial query, the model performs a few rounds of soft key–value retrieval, projects the retrieved signals back into the query, and updates it iteratively until convergence or a preset iteration limit. Theoretically, we show this process implements a Krylov-style polynomial approximation, equivalent to a differentiable second-order preconditioner, thereby deepening retrieval in a principled way. Empirically, HippoTune outperforms classical buffer-free PEFT-CL methods by 5–8\% in accuracy across three vision benchmarks, while reducing training FLOPs by 50\%, effectively mitigating forgetting under tight compute constraints. 
Code is available at: https://github.com/yan4xi1/HippoTune.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
Recently, several continual learning (CL) methods have emerged that utilise pre-trained models through parameter-efficient fine-tuning. One category of these methods is based on soft prompts, which involve training a set of prompts to guide the representations of each input. Typically, these methods select prompts based on a feature vector of the input, computed via an additional forward pass through the model. In this work, the authors address the inefficiency of this process by proposing a new method for selecting prompts for each layer of the model, based on the representations from the previous layer. This approach reduces the number of FLOPs during training by eliminating the need for an initial forward pass to identify prompts. Inspired by the hippocampus, they introduce HippoTune, an iterative retrieval strategy designed for transformer layers. The paper includes results presented across three visual benchmarks, along with a comprehensive ablation analysis.

### Strengths
- The paper effectively highlights a limitation of several CL methods that use soft prompts: they perform a double forward pass per example (prompt selection and classification). Using the representation of the previous layer for prompt selection is an interesting idea that has been used in other areas, but it is presented in a novel way in the CL scenario.
- Performing multiple retrieval steps is an interesting idea, showing theoretically and empirically that it works.
- A comprehensive ablation study of the components of the proposed method is conducted.

### Weaknesses
- The paper discusses a framework for PEFT-CL methods, but it applies only to methods that use soft prompts. LoRA-based methods do not necessarily include a retrieval component, so they cannot be extended to the framework. It would be good to make it clear that the proposed framework applies only to methods that use a key-value selection of prompts.
    - Also, it is not clear why the proposed framework is necessary. I understand that it helps to explain the method better, but it does not necessarily help to understand the other methods better.
- There are many hyperparameters to search over (T_max, loss coefficients, T, plus the common ones for this type of method). This can increase the method's complexity.

### Questions
- By concatenating the values of "v" from each iteration (Eq. 6), a better representation of what is retrieved is achieved. Although the proposal achieves fewer FLOPs, this concatenation increases the batch size, since each batch element adds several elements to the sequence, resulting in a larger batch size than previous methods.
    - What percentage of the batch is used by these vectors? How much can this affect scenarios where GPU size may be an issue?
- Despite showing better results with multiple iterations (T_max >1), can you provide insight into why this improves performance?
    - Is it related to increasing the concatenated values to v?
- One problem with prompt-based methods is the low generalisation of vectors to datasets outside the distribution of the pre-training set.
    - Were experiments on datasets where these types of methods fail conducted?
    - It would be interesting to see how the proposed method would perform.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors propose a computationally efficient recurrent method for continual learning in the context of parameter-efficient fine-tuning. They draw an analogy between their method and a memory recall mechanism in the hippocampus. Proving properties of the recurrence enables the authors to provide practical guidelines for hyper-parameter selection. Although results are not state of the art in all settings, they are typically very strong vis-a-vis baselines, and the approach is computationally more efficient than the most competitive baselines. The focus here on the (exhaustive) recall of previously learned tasks is refreshing and a novel perspective in the literature.

### Strengths
The paper is well written, the theoretical analysis is insightful, and practical implications of the analysis are fleshed out. The evaluation is largely thorough. Results are reported on a variety of datasets with a number of baselines. Ablation studies are performed to illustrate the contributions of the (several) components of the method. Complete code appears to have been provided, although this reviewer was not able to review the code thoroughly.

### Weaknesses
It appears to be the case that the evaluations are done using task-incremental learning [1]. Evaluations of the method in domain-incremental or class-incremental settings are not included. The provided code is perhaps too voluminous and is certainly too much for a reviewer to digest. A typical useful repository accompanying a paper enables a reviewer to inspect the implementation of the key details of the method. 

[1] https://www.nature.com/articles/s42256-022-00568-3

### Questions
* Is my impression that all evaluations are performed only in a task-incremental learning setting correct?
* Can you report results in domain- or class-incremental settings in the appendix?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces HippoTune, a continual learning approach inspired by the hippocampal memory circuit (EC–DG–CA3–CA1). Unlike traditional parameter-efficient fine-tuning (PEFT) methods that perform single-step prompt retrieval, HippoTune embeds a latent deliberation loop within each Transformer layer, enabling multi-round associative memory retrieval and feedback to alleviate catastrophic forgetting. The method integrates orthogonal and entropy regularization to stabilize learning, and theoretical analysis shows that the iterative retrieval approximates a second-order preconditioner through a Krylov subspace expansion. Experiments on Seq-CIFAR100, Seq-ImageNet-R, and Seq-CUB200 demonstrate that HippoTune achieves higher accuracy (≈80%) with roughly half the computational cost of prior PEFT-CL baselines, maintaining efficiency and strong memory retention without rehearsal buffers.

### Strengths
1. The paper’s primary strength is its strong linkage between biological mechanisms and algorithmic design, modeling the hippocampal memory circuit (EC–DG–CA3–CA1) as a multi-round associative retrieval process within Transformer layers.
This biologically grounded framework makes the method both conceptually interpretable and practically effective in mitigating catastrophic forgetting.
2. The authors formalize existing prompt-pool approaches as a single key–value retrieval framework and situate L2P, DualPrompt, CoDA-Prompt, and HiDe-Prompt as special cases, which clarifies trade-offs and motivates deeper, iterative retrieval.
3. HippoTune  matches or outperforms leading buffer-free PEFT baselines on Seq-CIFAR100, Seq-ImageNet-R, and Seq-CUB200.

### Weaknesses
1. While the paper includes ablation studies, it does not clearly isolate the individual contributions of the recurrent loop, orthogonal regularization, and entropy term to the overall performance gain.
2. The model introduces several new hyperparameters (loop depth, temperature, $\alpha$, Top-k), yet lacks an analysis of stability or sensitivity across different settings.
3. The writing style feels overly formulaic and AI-generated, with excessive use of em dashes and uniform sentence patterns.
4. All studies use a frozen ViT-B/16 with only PEFT modules trained, which limits adaptability and generalization; including experiments on other architectures such as Swin Transformer as well as partially unfrozen ViT variants, would better demonstrate the method’s robustness and general applicability.
5. The paper only reports accuracy as the evaluation metric, but does not provide any quantitative measures of forgetting, such as BWT, FWT, or average forgetting over time.

### Questions
1. What is the precise novelty over prior PEFT-CL prompt-pool methods beyond adding iterative retrieval; what can HippoTune do that they fundamentally cannot?
2. What is the exact buffer size used for the classical continual learning baselines (e.g., LwF, DER++)?
3. Could you clarify what the function $\phi(x; \theta)$ specifically represents in your formulation (line 188)?

### Soundness
2

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
This work introduces a biologically inspired approach to mitigate catastrophic forgetting in continual learning. Traditional parameter-efficient fine-tuning (PEFT) methods, such as adapters and prompts, reduce computation and storage costs but still fail to retrieve and integrate previously learned knowledge effectively. Drawing inspiration from the hippocampal EC–DG–CA3–CA1 loop in the human brain, the authors propose HippoTune, a latent deliberation mechanism that iteratively reactivates past representations to enhance memory retention during fine-tuning. In this model, each transformer layer is augmented with a small associative retrieval loop: the input state initializes a query that undergoes pattern separation (DG), auto-associative completion (CA3), and integrative fusion (CA1) across multiple iterations. This process allows the network to refine representations through recursive interaction within a learned latent space, mimicking hippocampal recall dynamics. Theoretically, the iterative update of HippoTune is shown to approximate a second-order preconditioning step in the Krylov subspace, providing curvature-aware optimization without explicit Hessian computation. The method also maintains stability across varying task numbers, demonstrating its scalability and robustness.

### Strengths
1. This work propose HippoTune, a continual learning method that bridges biological insight and machine learning theory by translating hippocampal associative memory mechanisms into a PEFT framework, offering a novel perspective on memory consolidation and retrieval in neural models for continual learning.
2. The paper presents a relatively novel and biologically inspired analogy that grounds its internal updating mechanism in the hippocampal associative loop. This connection gives the proposed architecture a plausible neurobiological motivation, and the accompanying theoretical analysis adds a layer of methodological soundness.
3. In addition, the partial update mechanism (Eq. 5) effectively preserves previously acquired knowledge while allowing new information to be injected into multiple parameter layers, achieving a desirable balance between plasticity and stability. Empirically, the method demonstrates strong performance across continual learning benchmarks, surpassing existing PEFT-based baselines by a meaningful margin.

### Weaknesses
However, despite the originality of its biological analogy, it should be acknowledged that the neuroscience of long-term memory formation remains far from fully understood. Thus, while the hippocampal metaphor adds interpretability, it does not constitute a rigorously biomimetic design in a scientific sense. Fundamentally, the proposed HippoTune module still functions as a multi-layer retrieval and partial parameter update process, consistent with mainstream continual learning paradigms rather than a radically new mechanism of memory consolidation. Consequently, the work’s strength lies more in its conceptual integration of biological inspiration and computational practicality than in demonstrating a genuinely new class of biologically faithful learning architecture.

### Questions
In, eq5, you mentioned a layer-specific linear transforamtion fuction P, would you mind to elaborate it a littble more?
How is the process of eq5 can be seen as a minimal abstraction of the CA3 mechanism? Is there more previous work to enlight us on this issue?

### Soundness
3

### Presentation
4

### Contribution
4
