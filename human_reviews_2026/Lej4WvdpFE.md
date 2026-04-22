# Detect, Decide, Unlearn: A Transfer-Aware Framework for Continual Learning

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 6

## Abstract
Continual learning (CL) aims to continuously learn new tasks from data streams. While most CL research focuses on mitigating catastrophic forgetting, memorizing outdated knowledge can cause negative transfer, where irrelevant prior knowledge interferes with new task learning and impairs adaptability. Inspired by how the human brain selectively unlearns unimportant information to prioritize learning and to recall relevant knowledge, we explore the intuition that effective CL should not only preserve but also selectively unlearn prior knowledge that hinders adaptation. We introduce DEtect, Decide, Unlearn in Continual lEarning (DEDUCE), a novel CL framework that dynamically detects negative transfer and mitigates it by a hybrid unlearning mechanism. Specifically, we investigate two complementary negative transfer detection strategies: transferability bound and gradient conflict analysis. Based on this detection, the model decides whether to activate a Local Unlearning Module (LUM) to filter outdated knowledge before learning new task. Additionally, a Global Unlearning Module (GUM) periodically reclaims model capacity to enhance plasticity. Our experiments demonstrate that DEDUCE effectively mitigates task interference and improves overall accuracy with an average gain of up to 4.55\% over state-of-the-art baselines.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces DEDUCE, a continual learning framework that detects negative transfer between tasks and mitigates it through selective unlearning. The core idea is that not all prior knowledge should be preserved - some interfering knowledge should be unlearned to improve adaptation to new tasks. The framework uses two detection strategies (transferability bounds and gradient conflict analysis) and two unlearning modules (local and global) to dynamically balance knowledge retention and plasticity. Experiments show consistent improvements across multiple datasets and baselines.

### Strengths
1. The paper addresses an underexplored aspect of continual learning - that blindly preserving all knowledge can hurt performance. This perspective is refreshing and the neuroscience motivation about selective forgetting is compelling.

2. The experimental evaluation is comprehensive, testing across multiple datasets (CIFAR-100, CIFAR-10, Tiny-ImageNet, CORE-50) and showing that DEDUCE works as a plug-in enhancement for various existing methods. The consistency of improvements across different baselines suggests the approach has broad applicability.

3. The paper provides two complementary detection strategies, which is useful given different computational and accuracy trade-offs. The ablation studies are thorough and help understand the contribution of different components.

### Weaknesses
1. The author may ignore the optimization-generalization gap. There's a fundamental conceptual issue with the gradient conflict detection mechanism. Gradient conflict is an optimization phenomenon, while negative transfer concerns generalization. The paper doesn't justify why opposing gradients during training would necessarily indicate poor test-time transfer between tasks. This conflation undermines the theoretical foundation of one of the main detection strategies.

2. The implementation of the transferability bound also has gaps. The use of LEEP scores as a proxy for λ isn't well justified, especially since LEEP was designed for offline transfer learning with fixed source models, not continual learning where models evolve. The connection between Eq.4 and the actual bound is hand-wavy at best.

3. The paper doesn't clearly explain what constitutes **interfering knowledge** versus **useful knowledge**. How does maximizing cross-entropy loss (Eq. 8) specifically target interference rather than just degrading performance broadly? The mechanism seems too coarse-grained for selective unlearning.

4. The global unlearning module appears disconnected from the negative transfer detection - it periodically resets neurons regardless of whether negative transfer is detected. This seems more like a generic plasticity mechanism than targeted unlearning.

5. The paper claims that unlearning improves plasticity but provides no direct evidence for this. While they show accuracy improvements on new tasks, they don't demonstrate that this is due to increased plasticity rather than other factors. Including analysis of network capacity in [1] could help.

[1] Dohare S, Hernandez-Garcia J F, Lan Q, et al. Loss of plasticity in deep continual learning[J]. Nature, 2024, 632(8026): 768-774.

### Questions
1. Can you provide theoretical or empirical evidence that gradient conflict during optimization actually correlates with negative transfer at test time?

2. How exactly does maximizing cross-entropy loss lead to selective unlearning of only interfering knowledge? What prevents this from removing useful knowledge as well?

3. In Tab. 13, the LUM activation frequency varies dramatically across datasets (92% for CIFAR-100 epoch-level vs 36% for batch-level). What does this suggest about the reliability of the detection mechanism?

4. Could the performance improvements be explained by regularization effects rather than actually addressing negative transfer or improving plasticity?

### Soundness
3

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
This paper addresses the problem of negative transfer in continual learning (CL) by proposing a 3-stage framework, Detect, Decide, and Unlearn, which can be plugged into existing CL methods to improve the performance. It presented two strategies, task-level transferability bound and batch-level gradient conflict analysis, to detect negative transfer. If the interference between old and new tasks is detected, it performs unlearn-learn update to future batches. This approach was tested on several existing CL approaches with standard CL benchmarks, showing its effectiveness of increasing the CL performance.

### Strengths
The paper has a novel contribution to CL, i.e., focusing on negative transfer, while many others are concerned with catastrophic forgetting.

The two different mechanisms of detecting negative transfers are designed at different levels and can complement each other. 

The LUM is designed to balance between removing the interference (negative transfer) and important parameters (prior knowledge). The GUM can help keep those low activated but important neurons.

### Weaknesses
This detect-decide-unlearn process introduces significant computational costs, and the partial of the costs is mentioned in the appendix. The work needs to be further analyze the extra costs and be optimized to make this the process lightweight. 

In Eq8, it is unclear why L_{unlearn} is defined in that way. Maximizing the error on the new task's current batch does not necessary mean to unlearn the old task. It could be unlearning the new task as well. 

The LUM relies on a diagonal FIM to compute the importance of parameters, which makes it inherits all the known limitations of diagonal FIM, such as inaccurate approximation,  scalability issue of storing FIM, and saturation of FIM with more tasks are coming. 

The experiment of the paper should include several more recent CL baselines, including those leveraging pre-trained models, to give a more comprehensive evaluation.

### Questions
1. Clarify Eq8, i.e., how does maximizing the loss on the current batch can reliably unlearn prior interfering knowledge, rather than unlearning new knowledge
2. Provide a more detailed analysis on the extra computational cost and propose/discuss optimiziation to make the framework more lightweight 
3. How to address the limitations of using diagonal FIM? 
4. How will the proposed framework perform with more recent CL including those using pre-trained models?

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
This paper proposes a novel approach to continual learning that aims to effectively balance stability and plasticity. The key idea is to detect interference from new tasks on past task knowledge. When such interference is detected, a Local Unlearning Module selectively removes conflicting knowledge while preserving critical parameters. A Global Unlearning Module identifies less important network components to allocate more learning capacity for new tasks, thereby reducing forgetting. The method is supported by theoretical justifications and evaluated empirically on standard CL benchmarks.

### Strengths
- A novel and well-motivated approach that integrates neuron importance estimation with targeted unlearning based on interference signals.

- Strong theoretical foundation supporting the role of local and global unlearning.

- The proposed method indicates a meaningful contribution to continual learning research.

### Weaknesses
The evaluation setup requires more clarity. See details below:
- It is unclear whether all methods (including the proposed one) are trained in a fully online continual learning scenario.

- The buffer size used in Table 1 should be explicitly stated.

- The DER++ baseline results differ noticeably from previously published online CL work (e.g., OnPro). I suggest re-running DER++ and comparing under the OnPro (ICCV 2023) experimental protocol for fairness:

  - OnPro: Online Prototype Learning for Online Continual Learning, ICCV 2023

- Results for the STAR baseline also differ from what has been reported in prior work. How is STAR implemented here?
  - According to the STAR paper, STAR > DER++ in most settings, so STAR being significantly weaker raises concerns of evaluation fairness.
  - When comparing with STAR, shouldn’t the comparison be against X-DER + STAR?

### Questions
- What is the explanation for the results where the bound-based variant (OUR(B)) underperforms the gradient-based variant (OUR(G))?

- Why does backward transfer (Figure 4) decrease for HAL when combined with the proposed method?

- Why does EWC perform worse than the simple fine-tuning baseline in your experiments?

Also refer to the weakness section above.


**Additional Remarks:**

I am open to increasing my score if the concerns about evaluation fairness and clarity are addressed in the rebuttal.

### Soundness
3

### Presentation
3

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
The paper proposes DEDUCE, a transfer-aware framework for continual learning that dynamically detects and mitigates negative transfer, where outdated or irrelevant prior knowledge interferes with learning new tasks. The framework combines two detection strategies: a transferability bound derived from domain adaptation theory to estimate when prior knowledge harms generalization, and gradient conflict analysis to identify real-time interference between old and new task gradients. Experiments across CIFAR100, TinyImageNet, and CORE50 show consistent improvements over strong CL baselines.

### Strengths
1. Introduces a novel approach to detect and decide when to unlearn, bridging machine unlearning and continual learning communities.
2. Dual-strategy detection plus hybrid unlearning are well-motivated and mathematically backed.

### Weaknesses
1. Relies on stored exemplars and known task identities, and the applicability to online task-free CL remains unclear.
2. All experiments use vision classification benchmarks, which may limit the paper's generality.
3. The framework adds two auxiliary modules, which may cause additional compute complexity.

### Questions
What is the computational overhead of the proposed method compared to other methods?

### Soundness
2

### Presentation
3

### Contribution
3
