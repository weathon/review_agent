# CellDuality: Unlocking Biological Reasoning in LLMs with Self-Supervised RLVR

- Decision: Accept (Poster)
- Scores: 2, 6, 6

## Abstract
\begin{abstract}
Developing generalist large language models (LLMs) capable of complex biological reasoning is a central challenge in computational biology. While existing LLMs excel at predictive tasks like cell type annotation and logically-constrained problems, enabling open-ended and mechanistic reasoning remains a challenge. A promising direction is Reinforcement Learning from Verifiable Rewards (RLVR), which has been shown to significantly enhance complex reasoning in general domains like mathematics and code synthesis. However, its application in biology is hindered, as most biological outcomes are non-verifiable. For example, verifying a generated gene sequence is usually infeasible. In this paper, we introduce CellDuality, a self-supervised framework that enables LLM agents for robust reasoning in single-cell biology. Our framework is built on the principle of complementary task duality, a self-verification process that leverages a bidirectional reasoning loop. First, the model performs a forward reasoning task by predicting a biological outcome (e.g., a cell's response to a drug). Then, in a complementary inverse task, it must reason backward from its own prediction to reconstruct the initial conditions (e.g., the original drug perturbation). The fidelity of this reconstruction serves as an intrinsic reward signal, creating a feedback loop that enforces logical and biological consistency. We use these intrinsic rewards to align the base LLM via reinforcement learning, without requiring ground-truth verification labels. We demonstrate that CellDuality achieves state-of-the-art performance and provides coherent biological explanations across a diverse suite of single-cell reasoning tasks. Critically, on the challenging out-of-distribution perturbation prediction benchmark, our self-supervised approach significantly outperforms the standard fine-tuning baseline and narrows the performance gap to a supervised RLVR baseline. Our work showcases a new path toward scalable training of biological foundation models.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper proposes CellDuality, a self-supervised RL-framework that enables large language models (LLMs) to perform open-ended biological reasoning in single-cell analysis. The key idea is to make sure each biological reasoning task such as drug response prediction is paired with an inverse task such as  recovering perturbation from response, forming a bidirectional reasoning loop. The consistency between the forward and inverse predictions is used as an intrinsic reward signal without labeling or annotation, enabling RL alignment without ground-truth verification labels. Experiments across four representative single-cell reasoning tasks demonstrate that CellDuality achieves competitive or slightly better performance compared to SFT baselines and narrows the gap to fully label-supervised RLVR models.

### Strengths
- The proposed self-supervised RL training paradigm is training-efficient and agnostic to the choice of policy optimization algorithm, showing that consistent rewards can provide alignment signals without external supervision.
- The Complementary Task Duality concept is clearly formulated and grounded in prior work (eg., DuPO), providing a natural extension to domains where canonical verifiable rewards are not available or expensive to acquire.

### Weaknesses
- The self-consistency reward effectively measures semantic or reconstruction fidelity rather than true biological correctness. While this provides a stable and well-defined signal, it may not inject new knowledge or enable deeper reasoning as canonical verifiers do in math, code, or symbolic domains. In this sense, the biological reasoning learned may remain shallow.
- Across multiple evaluation cases (Tab 2-6), CellDuality shows seemingly marginal gains over its SFT baseline and does not consistently outperform strong supervised or specialized models. This raises doubts about the real effectiveness of the proposed self-supervised alignment compared to simpler fine-tuning approaches and how is applied to other relevant domains in biology.
- The methodological novelty appears limited relative to DuPO (She et al., 2025), with CellDuality serving primarily as an instantiation in the single-cell domain rather than a substantial conceptual advancement.

### Questions
- How many samples or optimization steps are passed for a complete RL stage? Could the authors annotate such information in Figure 2 (a)(b)? 
- Regarding the differences between the proposed self-consistency reward and ground-truth reward, what if the model predict the wrong label (X <> Y) during rollout sampling? Should the model learn useful knowledge or will this cause the model being enforced to predict the wrong label? 
- What is the base model and performance for CellDuality in Table 2-5? Line 377 indicates this is LLama-3.2-3B. Could the authors add the performance of base model before SFT for better comparison in these tables? 
- Could the authors attach the exact text prompts for constructing the CellDuality in the appendix, as a common practice, to enhance the reproducibility?
- Could you also show some failure modes when the model exhibits degenerated consistency (eg., mutually consistent but biologically incorrect predictions).

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces CellDuality, a self-supervised reinforcement learning framework for training large language models to perform biological reasoning in single-cell analysis. The method leverages Complementary Task Duality, where forward (primal) and backward (dual) reasoning tasks generate intrinsic rewards that enable alignment without ground-truth supervision. Extensive benchmarking on classification and generative tasks is also present.

### Strengths
1) The paper presents a technically sound and well-motivated framework that adapts reinforcement learning with verifiable rewards to inherently non-verifiable biological problems.
2) The principle of complementary task duality is conceptually elegant, providing a self-consistent mechanism for intrinsic reward generation without labeled data.
3) The experimental results are extensive, covering multiple benchmark datasets, and the ablation studies convincingly isolate the contribution of the self-supervised RL stage.

### Weaknesses
1) The study lacks an explicit evaluation of biological interpretability: while metrics demonstrate numerical improvements, there is no assessment of whether reasoning outputs capture biologically meaningful mechanisms.
2) The framework’s scalability and computational cost are not detailed, especially regarding the RL alignment phase, which could be a practical limitation.
3) The proposed approach is validated only on transcriptomic data; extending it to multimodal biological datasets would strengthen its generality claims.

### Questions
I would ask the authors to address the weaknesses listed above:
1) Please include biologically grounded evaluations, such as verifying whether the model’s reasoning outputs (for example, perturbation responses) are consistent with known pathways or regulatory relationships.
2) Provide an analysis of computational cost and scalability, including training time and hardware requirements for the RL alignment stage.
3) Please discuss how the framework could be extended to multimodal datasets (for instance, integrating ATAC-seq or proteomic data) to further validate the generality of the approach.

### Soundness
3

### Presentation
3

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
Summary:
This paper introduces CellDuality, a self-superivsed framework to enhance the biological reasoning ability of LLM. It tries to address the challenge that it’s hard to build reward functions for most biological outcomes as it is non-verifiable. The framework uses complementary task duality. The method has achieved SOTA performance without needing ground-truth verification labels.

### Strengths
Pros:
- The paper introduces CellDuality, a new self-supervised method to enhanve biological reasoning ability of LLMs
- The method has achieved SOTA performance compared with baseline methods.

### Weaknesses
Cons:
- The “cell sentences” representation used in the method is a simplified representation discarding the actual expression value. There should be better way to encode single-cell transcripomics data.
- The claim of unified framework may be a bit too broad, as it only includes 4 downstream tasks with limited settings. 
- The proposed method is not consistently outperform baselines methods plus the additional reinforcement learning only bring very marginal benefits.

### Questions
See Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3
