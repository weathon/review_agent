# Task-Aware Model Merging via Fisher-Weighted Median

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 4, 2

## Abstract
Fine-tuning large language models (LLMs) on task-specific data provides strong in-domain performance but limits generalization and requires storage of many specialized models. Retraining a unified multitask model is often infeasible, as it demands task-specific training data that may be unavailable, raise privacy concerns, or incur prohibitive computational costs. Model merging has been proposed as an alternative solution that effectively integrates the distinct strengths of several fine-tuned models into a single, comprehensive model. The majority of model merging approaches rely on performing arithmetic operations directly on model parameters.  Although research in model merging has expanded significantly in recent years, two distinct approaches have become dominant: 1) techniques that mitigate interference from redundant parameters and sign conflicts, and 2) techniques that account for the varying sensitivity of individual parameters. However, these two approaches operate independently without considering each other's strengths and remain disconnected from each other. In this work, we aim to unify these two well-established yet currently disconnected approaches by integrating insights from both the approaches.
 We propose DRIFT-MEDIAN, a unified framework for merging models that leverages Fisher information to assign appropriate weights to the task vectors.
 Our contribution lies in the development of a closed-form solution of loss function grounded in the Fisher-weighted median. The formulation ensures that parameter contributions reflect both sensitivity and relevance, leading to more robust model merging.
This mechanism prioritizes parameters with high task-specific sensitivity in the merged representation, while naturally diminishing the influence of less important parameters. 
Comprehensive experiments on Llama-3.1-8B, Llama-3.2-3B, Llama-2-7b, GPT-2, CLIP-ViT-B/32 models across mathematics, coding, multilingual reasoning, safety, instruction following, GLUE benchmark and vision tasks demonstrate that DRIFT-MEDIAN outperforms existing model merging methods.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes DRIFT-MEDIAN, a new parameter-space model merging framework for combining LLMs across different tasks without retraining. The authors introduce a Fisher-weighted median aggregation mechanism for merging parameters robustly. Experiments are conducted on GPT-2, Llama-3.1-8B, Llama-3.2-3B, and Llama-2-7B, covering diverse domains such as math reasoning, coding, multilingual understanding, instruction following, safety, and GLUE benchmarks.

### Strengths
The authors evaluate across a wide range of model architectures and task types across language models.

This paper is well-written and easy to follow.

### Weaknesses
Although the combination of Fisher weighting and interference mitigation is new, each individual component (sign pruning, Top-K selection, Fisher weighting) already exists. The paper’s novelty is limited.

The method’s theoretical advantages are not quantitatively demonstrated. For instance, it would help to show why the Fisher-weighted median is superior to Fisher-weighted mean beyond robustness claims.

Fisher Information estimation for large models (8B parameters) is computationally expensive—reported as “~1 hour per domain on A100” (Appendix E). The practicality for extremely large-scale merging such as 14B or 32B or real-world continual learning scenarios remains questionable.

Although the proposed method is applied to language tasks. It is recommended to provide results on vision-language models, such as merging eight CLIP models over 8 classification datasets.

While λ is analyzed, the keep ratio κ (which controls sparsity) is not systematically explored. Its effect on stability and generalization should be reported.

### Questions
See weakness. I think the major one is the limited novelty.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes DRIFT-MEDIAN, a framework for combining multiple fine-tuned LLMs into a single multitask model. The method integrates key advances from both parameter interference reduction and sensitivity-aware merging. To achieve this, it employs a multi-stage process that first mitigates interference using sign resolution and coordinate-wise Top-K filtering. The refined parameters are then aggregated via a Fisher-weighted median. Comprehensive experiments on GPT-2 and Llama variants across a diverse range of benchmarks demonstrate that DRIFT-MEDIAN consistently outperforms existing model merging baselines.

### Strengths
1. The paper is clearly written and well-organized. The architectural pipeline illustrated in Figure 1 provides a clear step-by-step overview of the methodology, making a complex multi-stage process easy to understand.
2. The proposed method is thoroughly evaluated across a diverse set of LLM architectures (GPT-2, Llama-2, and two Llama-3 variants) and a wide range of tasks, including mathematics, coding, multilingual reasoning, and safety. This comprehensive evaluation increases confidence in the generalizability and robustness of the results.
3. The authors provide a detailed description of the experimental setup, including baseline methods, datasets, and hyperparameter settings. This significantly aids in the reproducibility of the work.

### Weaknesses
1. **Limited Novelty**: The main weakness is the incremental nature of the contribution. The proposed framework is largely a synthesis of well-established techniques:

  - The sign resolution step is directly adopted from TIES-merging [1].
  - The use of Fisher information to weight parameter importance is the core idea of Fisher-merging [2].
  - Top-K filtering is a key component of TIES-merging [1] and SCE-merging [3], which the authors adapt to a coordinate-wise application.
  - The final scaling of the merged task vector is a standard technique used in Task Arithmetic [4] and its successors.

While the combination is effective, the paper primarily rearranges and refines existing building blocks rather than introducing a new, foundational concept.

2. **Potential Overclaiming of Contribution**: The abstract (lines 22-25) and introduction state that interference-reduction and sensitivity-based approaches have remained "largely disconnected" and that this work aims to "bridge the gap". This claim may be too strong. Recent works such as SCE-merging [3] and PCB-merging [5], also implicitly consider both interference and parameter importance. The authors should more carefully contextualize their work with respect to these methods.

**References**:

[1] Ties-merging: Resolving interference when merging models. (Yadav, et al., NeurIPS 2023)

[2] Merging models with fisher-weighted averaging. (Matena, et al., NeurIPS 2022)

[3] Fusechat: Knowledge fusion of chat models. (Wan, et al., Arxiv 2024)

[4] Editing models with task arithmetic. (Ilharco G, et al., ICLR 2023)

[5] Parameter competition balancing for model merging. (Du, et al., NeurIPS 2024)

### Questions
1. Could you please elaborate on what you see as the primary conceptual contribution of this work beyond the successful integration of existing methods? What is the core, fundamental problem in model merging that DRIFT-MEDIAN solves in a way that prior art has not?
2. The visualization in Figure 2 contrasts the "model-wise" Top-K selection of TIES with your "coordinate-wise" approach. This "row-wise" vs. "column-wise" distinction, however, seems contingent on the arbitrary representation of parameters and models as matrix axes. Could you provide a more fundamental rationale for why applying Top-K filtering independently at *each parameter coordinate* is superior? A more detailed justification or supplementary analysis would strengthen this design choice.
3. Have you investigated the interplay between the keep-ratio $K$ and scaling factor $\lambda$? For instance, does a more aggressive pruning (smaller $K$) necessitate a larger scaling factor  $\lambda$ to compensate? A 2D sensitivity plot for these two parameters would be insightful.
4. While the method shows strong average performance, a qualitative analysis of its limitations would be valuable. For instance, the paper reports performance degradation on certain tasks (e.g., Multilingual in Table 2, Maths in Table 3). Could the authors provide an analysis of these failure cases? What might such an analysis reveal about the method's underlying assumptions or potential weaknesses?
5. The paper provides limited detail regarding the computation of the Fisher information matrix, which is a critical component. Could you please clarify the following:

- **Scalability:** How does the computational cost of this step scale with model size? Is it linear, or does it pose a significant bottleneck for even larger models?

- **Sample Selection:** What criteria were used to select the samples for this computation? What is the composition of the "validation data" mentioned in line 978, how was it constructed, and is it fully disjoint from the test sets used for final evaluation?

- **Domain Sensitivity:** Have you analyzed how the domain coverage and diversity of the dataset used for Fisher estimation affect the final merging performance? For example, would using a more general-purpose dataset instead of a task-specific one alter the results?

6. I am willing to raise my score if the authors can satisfactorily address the weaknesses and questions I have raised.

### Soundness
2

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
This paper proposes a task-aware model merging method that aims to enhance the robustness and adaptability of parameter-space merging across diverse tasks. The approach can be viewed as a parameter-level weighted merging strategy that, similar to FisherMerging, leverages task-specific data characteristics to guide the parameter combination process. Conceptually, the overall workflow can be regarded as a linear combination of TiesMerging and FisherMerging: it integrates the conflict-mitigation mechanism from TiesMerging with the parameter-sensitivity estimation from FisherMerging. A distinctive feature of the proposed method is the introduction of a cross-model redundancy pruning strategy, which differs from TiesMerging that performs pruning within a single model. The paper further explores several design choices—such as the use of the Fisher matrix, sign resolution, and top-K pruning—and evaluates the method on multiple tasks, including Natural Language Understanding, mathematics, and coding benchmarks. Experimental results demonstrate that the method can effectively improve merged model performance in many cases.

### Strengths
+ The paper provides a clear motivation for addressing the limitations of current parameter-space merging methods and positions its work within the context of task interference and parameter sensitivity.

+ By integrating the principles of TiesMerging and FisherMerging, the proposed approach attempts to balance parameter alignment and task sensitivity, offering a new perspective for parameter-space model merging.

+ The introduction of a cross-model redundancy pruning strategy represents an interesting attempt to reduce interference between tasks, distinguishing this method from prior single-model pruning approaches.

+ The paper follows a clear methodological pipeline and conducts extensive experiments across various task domains.

+ The method demonstrates adaptability across multiple types of tasks, suggesting its potential applicability to broader multi-task or multi-domain model merging scenarios, especially in generative tasks.

### Weaknesses
- The abstract claims that the proposed method bridges the gap between interference-handling approaches and parameter-sensitivity-aware approaches. However, since the two processes are combined in a serial manner, the method does not truly consider interference and parameter sensitivity simultaneously. These steps are independent. For example, important sensitive parameters may be pruned during the sign resolution step before the Fisher matrix is even computed.

- The proposed method is mainly derived from TiesMerging and FisherMerging, but it does not clearly state the specific improvements or innovations compared with these methods.

- The method is not data-free, as it still requires task data to compute the Fisher matrix. Therefore, it cannot be applied in completely data-free scenarios.

- The method workflow first computes the Fisher matrix and then performs cross-model pruning. However, if a parameter is highly sensitive for a given task but has only small update magnitudes due to the characteristics of that task, this parameter might be overshadowed by a less critical parameter from another task that has a larger update magnitude.

- The process of determining unified signs may already eliminate many offsets of task vectors at the corresponding parameter positions. The subsequent cross-model pruning step further removes parameters from some task vectors. As a result, performance might be dominated by a few tasks with large update magnitudes. This may be acceptable for NLP tasks, where tasks with small offsets can often be handled well by the base model itself. However, for vision-related tasks, where task offsets are generally larger, performance may not be as stable. It is therefore suggested to include more vision-domain tasks to verify this issue and further evaluate the cross-modal applicability of the proposed method.

- The key coefficient K for cross-model pruning is not thoroughly discussed in terms of how different values affect performance.

- The transition from Equation (2) to Equation (3) is not described in sufficient detail, although it is crucial for understanding the subsequent steps.

- Different hyperparameters are searched for each base model, and the reported results correspond to the optimal values found for each. However, according to the appendix, other baselines seem to use the same hyperparameters across different base models, which may make the comparison unfair. In addition, the claim that performance is insensitive to hyperparameter variations near the optimal value lacks theoretical or experimental support.

- The computation of the Fisher matrix requires significant additional computational resources and time, as well as extra storage space to retain the corresponding matrices.

- The paper lacks a discussion on how the performance of the proposed method changes as the number of tasks increases, which is crucial for verifying its ability to handle task conflicts and to effectively measure task-sensitive parameters.

- The work lacks comparisons with the latest methods. Although several recent approaches are introduced in the paper, they are not included in the experiments. For instance, PCB Merging appears only in Table 4 but is not included in other experiments. Moreover, AdaMerging (2024), which is also a task-aware method, is not compared.

- In the experimental section, most of the content focuses on reporting results, while deeper analysis and discussion of observed phenomena are missing. For example, the paper does not explain why the method performs worse on some tasks than simple task arithmetic.

- The paper lacks detailed descriptions of the specific task configurations and experimental setups used for the in-domain and out-of-domain experiments shown in Figure 3.

- The reference formatting is inconsistent. For instance, there is inconsistency in whether journal names are italicized, whether URLs and DOIs are provided, and in the abbreviation styles of journal or conference names. At the very least, papers from the same venue should follow a consistent citation format.

### Questions
Q1. In the first paragraph of the Introduction, it is stated that “current parameter-space merging methods exhibit fundamental limitations that constrain their effectiveness.” What exactly are these limitations?

Q2.The Fisher matrix already measures the sensitivity of parameters for a given task and reduces the influence of less important parameters. Why does the subsequent TOP-K selection use the magnitude of parameter updates as the criterion, instead of jointly considering parameter sensitivity? Could TOP-K possibly prune out parameters that are actually sensitive?

Q3. Compared with previous methods, this approach adds one more step of task vector decomposition and combination, making it a total of two steps. Is this necessary?

Q4. In Table 3, it can be observed that the proposed method performs worse than simple task arithmetic on mathematical and coding tasks. What is the reason for this?

Q5. In Table 4, the reproduced fine-tuned model performance is reported. During reproduction, were the parameters and environment exactly the same as those used in the first row of fine-tuning? Are the results in the last row obtained by merging your own fine-tuned models rather than those in the first row?

Q6. In the ablation study, after removing some key components such as sign resolution, the performance on some task types (e.g., Maths and Multilingual) even increases significantly. Why does this happen?

Q7. Why does the out-of-domain performance decrease as the merging coefficient increases? What might be the underlying reason for this trend?

Q8. In Figure 1(b), what does the symbol located between the two formulas at the bottom represent? It seems that this symbol is not used in the subsequent equations or steps.

Q9. For the scaling coefficient used in merging task vectors, what is the specific granularity used during the search?"

### Soundness
2

### Presentation
3

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
This paper proposes DRIFT-MEDIAN, a model merging method that aims to unify two dominant paradigms in the field: techniques that mitigate parameter interference (e.g., sign conflicts and redundancy) and those that account for parameter sensitivity to task performance. The method first resolves task vector conflicts via sign alignment, then quantifies parameter importance using Fisher information. It subsequently introduces a coordinate-wise Top-K selection strategy to retain the most relevant task vectors across models, and finally aggregates them using a Fisher-weighted median, which has a closed-form solution. The authors conduct experiments on various large language models (Llama family, GPT-2) across a range of tasks, reporting that their method outperforms existing baselines.

### Strengths
1. The paper is well-motivated. It correctly identifies two important yet often disconnected lines of research in model merging—parameter interference resolution and sensitivity-based weighting—and sets a valuable goal of unifying them into a single framework.
2. The proposed DRIFT-MEDIAN integrates multiple components (sign resolution, Top-K selection, Fisher-weighted median aggregation) into a logically coherent pipeline where each step serves a clear purpose.
3. The paper assesses the method on multiple LLMs of varying scales and architectures, covering a diverse set of tasks including mathematics, coding, multilingual understanding, and safety, which provides a relatively comprehensive testbed for its effectiveness.

### Weaknesses
1. The major concern is the limited novelty. Its core components—sign resolution, Top-K selection, and Fisher weighting—are heavily inspired by or directly adopted from existing works like TIES-merging and Fisher-merging. While the contribution lies in their combination, the assembly feels somewhat straightforward and lacks a deeper, more innovative mechanism.

2. The motivation and advantage of the coordinate-wise Top-K selection are not well-substantiated. The paper claims its coordinate-wise (row-wise) selection is superior to TIES's model-wise (column-wise) approach at preventing "parameter crowding and scarcity," but this claim is primarily supported by a simple schematic (Figure 2) and lacks more rigorous theoretical analysis or empirical evidence to prove its necessity and superiority.

3. The experimental results are not consistently convincing. On certain key tasks, the proposed method performs worse than some baselines. For instance, in Table 2, its PRR on the Maths task (85.19) is significantly lower than TIES (96.44) and Task Arithmetic (93.85). Similarly, in Table 3 on the Maths task, its PRR (65.77) is also lower than Task Arithmetic (72.40) and DARE (70.34). These inconsistent outcomes weaken the central claim of "consistently outperforming" prior methods.

### Questions
Please see the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
