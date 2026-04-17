# Towards Better Branching Policies: Leveraging the Sequential Nature of Branch-and-Bound Tree

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
The branch-and-bound (B\&B) method is a dominant exact algorithm for solving Mixed-Integer Linear Programming problems (MILPs). While recent deep learning approaches have shown promise in learning branching policies using instance-independent features, they often struggle to capture the sequential decision-making nature of B\&B, particularly over long horizons with complex inter-step dependencies and intra-step variable interactions. To address these challenges, we propose Mamba-Branching, a novel learning-based branching policy that leverages the Mamba architecture for efficient long-sequence modeling, enabling effective capture of temporal dynamics across B\&B steps. Additionally, we introduce a contrastive learning strategy to pre-train discriminative embeddings for candidate branching variables, significantly enhancing Mamba's performance. Experimental results demonstrate that Mamba-Branching outperforms all previous neural branching policies on real-world MILP instances and achieves superior computational efficiency compared to the advanced open-source solver SCIP. The source code can be accessed at https://github.com/doctor-watson626/Mamba-Branching/.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
### This paper presents ​Mamba-Branching, a novel neural branching policy for Mixed-Integer Linear Programming (MILP) that leverages the Mamba architecture to model the sequential nature of Branch-and-Bound (B&B) trees. The approach incorporates contrastive learning, which is a novel embedding layer trained with a contrastive loss to improve discriminability between candidate variables, which often have similar features. Experiments on heterogeneous MILP benchmarks (MILP-S and MILP-L) show that Mamba-Branching outperforms previous neural policies.

### Strengths
* This is the first work to explicitly model the sequential nature of B&B using a state-space model (Mamba). The idea of treating the search as a "branching path" is intuitive and well-motivated.
* The method is well-designed, combining contrastive learning for representation learning with sequential modeling for decision-making. The use of Mamba is justified given the long-sequence nature of the problem.
* The paper provides extensive experiments on real-world benchmarks, including easy and difficult instances. The results are interesting, demonstrating clear improvements over neural baselines.

### Weaknesses
1.  The description of the input sequence is confusing. It appears that at each step t, the model processes all candidate variable embeddings (e_t^1, ..., e_t^|C_t|) plus the expert's chosen variable embedding e_t^{a_t}. This leads to a sequence length of Σ(|C_t| + 1)over T steps. 
    * Concern 1:​​ Why include the expert action e_t^{a_t}in the input during imitation learning? During inference, the expert action is unavailable. This creates a train-test mismatch. The model might be learning to rely on seeing the expert's choice in the history.
    * Concern 1:​​ The massive sequence length (Σ|C_t| + T) is acknowledged, but it's unclear how this is handled practically. How many tokens are typical for T=100? The paper mentions "tens of thousands," which seems prohibitive even for Mamba. Please provide statistics on average sequence lengths in training/inference.
2. The comparison to ​Transformer-Branching​ is unfair. Transformer-Branching is trained with only T = 9 steps due to hardware constraints, whereas Mamba-Branching uses T = 100. The performance gap is likely dominated by this difference in historical context rather than the architecture itself.
3. The rationale for choosing relpscostover strong branching as the expert is crucial but relegated to the appendix. The argument that TreeGate features are more compatible with relpscost is interesting, but needs a more straightforward explanation in the main text 
4. The experimental setting on the difficult problem is limited. I would like to see more challenging instances being solved and compared to obtain a statistically persuasive experiment.

### Questions
1. During inference, does the input sequence contain the model's own previous predictions or the expert's actions? If it's the latter, how do you justify this setup for a practical scenario?
2. The results indicate that on standard ("easy") instances, all neural branching policies, including Mamba-Branching, still underperform the expert rule-based relpscost method. This raises two important questions: what explains this performance gap, and what would be needed for neural methods to become practically viable?
3. Why choose the difficult instances shown in Table 17? Is there justification for that, given the numerous difficult instances in MIPLIB 2017?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Mamba-Branching, a novel method for the Branch and Bound (B&B) algorithm in Mixed Integer Linear Programming (MILP) solvers. The authors' key insight is that existing neural branching methods (e.g., TreeGate and T-BranT) fail to effectively exploit the sequential nature of B&B tree expansion. To address this, they propose using the Mamba architecture (a selective state-space model) to model the complete branching path from the root to the current node. The method treats the historical decisions and tree states along the path as a long sequence. Experiments on the MILP-S and MILP-L datasets show that Mamba-Branching outperforms existing neural branching methods (TreeGate, T-BranT) in solving time, primal-dual integral (PD Integral), and number of nodes, and significantly surpasses Transformer-based models in computational efficiency.

### Strengths
+ Insightful problem formulation: The observation that traditional neural methods "generally overlook the sequential nature of B&B tree expansion" is a key insight. Modeling the "branching path" as a sequence is a novel and well-motivated idea that aligns with the intrinsic logic of the B&B process.
+ Outstanding computational efficiency: The paper clearly demonstrates the memory and speed advantages of Mamba-Branching in training and inference through Table 13. Compared to Transformer-Branching, it consumes significantly less memory and achieves an 80x faster inference speed, a remarkable advantage crucial for practical applications.

### Weaknesses
+ **Insufficient validation of the contrastive learning module**: The effectiveness of the contrastive learning (CL) module is inadequately validated. Although the paper introduces CL for embedding layer training in Section 4.1 and presents its effect via t-SNE visualization in Figure 6, the argument remains weak. The authors only compare with the "without CL" setting, which only proves that CL itself is helpful, but fails to demonstrate that their specific CL design is superior or necessary. To more comprehensively assess the value of their approach, the authors should compare their CL framework against several standard and widely-used contrastive learning loss functions and paradigms, such as InfoNCE.
+ **Ambiguous innovation attribution of Mamba**: The paper's demonstration of computational efficiency (Table 13, Figure 3) is excellent, strongly proving Mamba's significant advantage over Transformers in handling long sequential branching histories. However, these advantages primarily stem from the inherent properties of the Mamba architecture as an existing advanced model (e.g., linear complexity, selective state spaces), and the paper does not introduce any structural or algorithmic modifications or optimizations to the Mamba model itself. Therefore, the core innovation of the work leans more towards "successfully applying the cutting-edge sequence model Mamba to B&B."

### Questions
+ Are Mamba-Branching and baselines (e.g., TreeGate) compared using identical input features? It is recommended to disclose the feature list to ensure a fair comparison.
+ Has the method's generalization been validated on datasets beyond MILP-S/L, such as ML4CO 2021?
+ How does model performance vary with the length of the B&B path? Are there any observed performance bottlenecks for very long sequences?
+ Could the authors compare their approach with recent learning to branch methods [1, 2] to better position their contribution?
+ What is the specific selection logic for the instances used in the experiments?

**If the authors can convincingly demonstrate the method's generalization capability and the effectiveness of the contrastive learning component through additional experiments, I would be willing to update my score accordingly**.

[1] Zhang C, Ouyang W, Yuan H, et al. Towards imitation learning to branch for mip: A hybrid reinforcement learning based sample augmentation approach[C]// ICLR 2024.

[2] Parsonson C W F, Laterre A, Barrett T D. Reinforcement learning for branch-and-bound optimisation using retrospective trajectories[C]// AAAI 2023.

### Soundness
3

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
3

### Summary
The paper observes that existing GNN-based branching policies treat the B&B tree as an unordered graph and therefore ignore the sequential nature of branching decisions. The authors propose Mamba-Branching:
1) Employ the Mamba architecture (linear-time state-space model) to encode long branching paths (100+ steps);
2) Pre-train variable embeddings via contrastive learning so that the selected variable is closest to an anchor in embedding space;
3) Use imitation learning with SCIP’s relpscost as the expert.
On MILP-S and MILP-L datasets Mamba-Branching beats TreeGate, T-BranT and Transformer-Branching in number of nodes and solving time, and surpasses relpscost on difficult instances.

### Strengths
1) Fresh perspective: sequential modelling of branching history matches the dynamic nature of B&B.
2) Sound engineering choice: Mamba’s linear complexity enables 100-step history without memory explosion; experiments confirm faster inference than Transformer.
3) Strong empirical results on 73 heterogeneous instances; no problem-specific features required.

### Weaknesses
1) Baseline comparison limited to relpscost; no hybrid strong-branching/pseudocost rules used in commercial solvers are included.
2) The anchor in contrastive learning is obtained by max-pooling, which can be sensitive to outliers; alternative aggregations are not evaluated.
3) Sequence length (99 train / 24 test) is chosen empirically; no study on optimal truncation or attention-window size.
4) No theoretical analysis, e.g., sample complexity or regret bounds for the sequential policy.

### Questions
1) Does back-propagation through the entire Mamba hidden state remain memory-efficient if the sequence grows to hundreds of steps?
2) For highly irregular instances, does the order of nodes in the sequence (DFS, BFS, or other) affect convergence, and have you tried alternative orderings?
3) Contrastive pre-training introduces extra hyper-parameters (margin, temperature); do they need re-tuning when transferring across problem classes?

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
3

### Summary
This paper introduces Mamba-Branching, a novel deep learning-based policy for selecting branching variables within the Branch-and-Bound (B&B) algorithm used to solve Mixed-Integer Linear Programming (MILP) problems. The core insight is that existing learning-based methods fail to capture the sequential nature of the B&B process, where the path of decisions from the root of the search tree to the current node influences the next best choice.
The focus here is on heterogenous problems and instances.

Sequential Modeling with Mamba: It models the "branching path"—the history of states and decisions—as a long sequence. It employs the Mamba architecture, a recent State Space Model (SSM), which can efficiently process these long sequences with linear-time complexity, unlike Transformers which have quadratic complexity.
Contrastive Learning for Embeddings: To help the model distinguish between highly similar candidate branching variables, the paper proposes a contrastive learning strategy. This pre-training step learns discriminative feature embeddings by maximizing the similarity between a context anchor and the expert-chosen variable while minimizing similarity with other candidates.

Experiments show that it achieves superior computational efficiency compared to SCIP, a state-of-the-art open-source solver, on particularly difficult problem instances.

### Strengths
Novel Conceptualization: A key strength of this work lies in its novel conceptualization of the Branch-and-Bound (B&B) branching process. By framing it as a sequential decision-making problem, the authors effectively model the entire "branching path," a perspective that captures the historical context of the search in a way prior methods have not.

Appropriate Architectural Choice: The selection of the Mamba architecture is exceptionally well-justified and proves to be highly effective for modeling the long B&B sequences where Transformers are computationally infeasible due to their quadratic complexity.

Effective Embedding Strategy: The use of contrastive learning to generate more discriminative embeddings for candidate variables is an insightful solution to a significant challenge in this domain. The ablation studies compellingly confirm that this strategy has a significant and positive impact on performance.

Performance: The proposed method establishes a new state-of-the-art, demonstrating superior performance over existing neural branching policies and even surpassing the highly optimized relpscost heuristic in the SCIP solver on challenging instances. ( heterogeneous)

Evaluation: The experimental setup is robust and comprehensive, featuring multiple datasets, a strong suite of baselines, appropriate evaluation metrics, and detailed ablation studies that clearly validate the efficacy of the proposed components.

Code is shared.

### Weaknesses
1. Line 341: "To isolate the study of branching policies and eliminate interference from other solver components,
we disable all primal heuristics and provide each test instance with a known optimal solution value as
a cutoff.".

How realistic is this setting? Can the authors justify the impact of the above( providing optimal value before hand). It might be done in literature,  but what is the impact of this setting? Justification of this setting is important.

### Questions
1. The authors do report result as geometric mean across 5 random seeds, however, variance is not present in the paper. Request the authors to add variance studies.

### Soundness
3

### Presentation
3

### Contribution
3
