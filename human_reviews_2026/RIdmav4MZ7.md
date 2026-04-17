# Freeze, Prompt, and Adapt: A Framework for Source-free Unsupervised GNN Prompting

- Decision: Reject
- Scores: 2, 6, 2, 8

## Abstract
Prompt tuning has become a key mechanism for adapting pre-trained Graph Neural Networks (GNNs) to new downstream tasks. However, existing approaches are predominantly supervised, relying on labeled data to optimize the prompting
parameters and typically fine-tuning a task-specific prediction head—practices that undermine the promise of parameter-efficient adaptation. We propose Unsupervised Graph Prompting Problem (UGPP), a challenging new setting where
the pre-trained GNN is kept entirely frozen, labels on the target domain are unavailable, the source data is inaccessible, and the target distribution exhibits co-variate shift. To address this, we propose UGPROMPT, the first fully unsupervised GNN prompting framework. UGPROMPT leverages consistency regularization and pseudo-labeling to train a prompting function, complemented with
diversity and domain regularization to mitigate class imbalance and distribution mismatch. Our extensive experiments demonstrate that UGPROMPT consistently outperforms state-of-the-art supervised prompting methods with access to labeled
data, demonstrating the viability of unsupervised prompting as a practical adaptation paradigm for GNNs.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces the Unsupervised Graph Prompting Problem (UGPP). To handle this problem, the authors propose UGPROMPT, a fully unsupervised prompting framework based on consistency regularization and pseudo-labeling.
Extensive experiments across both graph- and node-level classification tasks demonstrate that UGPROMPT outperforms graph prompting baselines.

### Strengths
1. The proposed method is clearly presented.
2. Extensive experiments demonstrate the effectiveness of the proposed method.

### Weaknesses
1. The most critic weakness of this paper is the problem formulation. The studied problem UGPP in this paper is more like domain adaptation instead of graph prompting. As the comparison between this study and source-free domain adaptation (SFDA) in Introduction, the only difference is the adopted techniques: previous SFDA studies achieve this by learning GNN parameters while this work achieves this through learnable prompts. However, UGPP is essentially different from graph prompting in terms of accessible data (with/withour label information) and learning paradigms (unsupervised/supervised) for pre-training and adaptation. Hence, I encourage the authors to rewrite this paper as a study of SFDA through learnable prompts. 
2. Theoretical analysis is encouraged to be provided in the paper.

### Questions
See the above Weaknesses.

### Soundness
2

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
2

### Summary
This paper proposes a new problem setting called Unsupervised Graph Prompting Problem (UGPP), where the goal is to adapt a frozen pre-trained GNN to  inaccessible source data and unavailable target domain with covariate shift. To address this scenario, this paper proposes UGPROMPT, the first fully unsupervised GNN prompting framework. UGPROMPT leverages consistency regularization and pseudo-labeling to train a learnable prompting function, complemented with diversity and domain regularization to mitigate class imbalance and distribution mismatch. Extensive experiments show that UGPROMPT consistently outperforms state-of-the-art supervised prompting methods—even those using 25% to 100% labeled data—demonstrating the feasibility and effectiveness of unsupervised adaptation in the context of GNNs.

### Strengths
* Novel Problem Formulation: The UGPP setting is well-motivated and defined. This aligns with the parameter-efficient adaptation paradigm of LLM prompting but addresses unique challenges in graphs.
* Technical Innovation: The proposed UGPROMPT framework creatively combines consistency regularization, pseudo-labeling, and novel regularization techniques to enable effective unsupervised prompting.
* Comprehensive Evaluation: The paper provides thorough experiments across multiple datasets, tasks (node and graph classification), and base GNN architectures, demonstrating consistent improvements over supervised baselines.
* Reproducibility: Code is provided, hyperparameters are detailed, and experimental protocols (e.g., 50 runs per setting) ensure statistical reliability.

### Weaknesses
* Assumption Limitation: The assumption $P_T(Y \mid X)=P_S(Y \mid X)$,  (i.e., label-conditional distribution remains unchanged) limits applicability. In practice, label distribution shift may occur, which could severely degrade performance. This constraint needs more discussion as a key limitation.
* Augmentation Sensitivity: Results in Table 3 suggest that high feature masking rates can hurt performance on datasets with continuous features (e.g., ENZYMES, DHFR). This indicates that the choice of augmentation must align carefully with data characteristics, potentially limiting generalizability. More discussion on how to choose or design augmentations would strengthen the paper.
* Computational Overhead: Training involves multiple components (discriminator, augmentations), leading to higher training time than some baselines (though inference is efficient). This trade-off is noted but not thoroughly analyzed for resource-constrained settings.

### Questions
* How does UGPROMPT's performance scale with the degree of distribution shift between source and target domains? Are there theoretical guarantees on when the method would fail?
* UGPROMPT relies heavily on the pre-trained GNN’s ability to generalize knowledge from the source domain. If the source and target domains differ significantly in semantics (e.g., molecular graphs vs. social networks), do you expect the method to still work? Would stronger pre-training objectives or architectures improve robustness in such extreme transfer scenarios?
* How is UGPROMPT's scalability to larger graphs (e.g., billions of nodes)?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a challenging new problem setting called the Unsupervised Graph Prompting Problem (UGPP), which requires adapting a completely frozen, pre-trained GNN to new, unlabeled, and distribution-shifted data without access to the original source data. To solve this, the authors develop UGPROMPT, the first fully unsupervised GNN prompting framework, which uniquely utilizes a combination of consistency regularization, pseudo-labeling, and domain regularization. Experiments are conducted to verify its effectiveness.

### Strengths
- Extensive experiments regarding both node classification and graph classification are conducted.
- It’s novel trial for trying to establish a label-free graph prompting paradigm.
- Code and data is provided for reproducibility.

### Weaknesses
**Major Concerns:**

1. **Clarity and Significance of Motivations:** The motivation for the proposed UGPP setting could be strengthened.
    - The introduction (Line 47) frames the problem by contrasting it with methods that "rely heavily on labeled data." This characterization may not fully capture the current state of graph prompting, as many state-of-the-art methods [1-3] or other models benchmarked in [4],  are already designed for label-scarce (e.g., 1-shot or 5-shot) scenarios. The authors are encouraged to clarify how UGPP's "zero-label" requirement offers a significant practical advantage over these "few-label" settings.
    - The paper's second motivation (Lines 48-49), regarding the use of "truly frozen" models, also requires further clarification. The argument that projection heads prevent a GNN from being "truly frozen" is not immediately clear. The authors should elaborate on this distinction and, more importantly, substantiate the tangible benefits of a "truly frozen" GNN over other parameter-efficient adaptation approaches.
2. **Related Work:** The literature review appears to conclude around early 2024. The authors should consider including and discussing several more recent and highly relevant contributions, for example [1–3], to situate the proposed framework within the most current research landscape.
3. **Experimental Comparisons:**
    - **Baselines:** To fully validate the "state-of-the-art" claims, the experimental comparison would benefit from the inclusion of several stronger, more recent baselines.
    - **Result Discrepancies:** There appear to be discrepancies in the reported baseline results. The paper states it uses the code from [4], but the results in Table 2 differ notably from those in the original [4] paper. Furthermore, the performance reported here (with a 25% label ratio) seems to underperform the 1-shot or 3-shot results from [4]. The authors are requested to clarify the reasons for this difference to ensure a fair and reliable comparison.
4. **Reliance on pseudo-labeling:**
    - The method's reliance on pseudo-labeling (e.g., `FixMatch`) acts as a "semantic anchor." This mechanism conceptually requires the target graph to share the identical class space as the source graph, tethering the solution to the specific pre-trained task.
    - While the goal of prompting is often to adapt a model to diverse downstream tasks of various domains, this reliance on shared semantics seems to limit the method to adapting the model to relevant semantic domains and tasks. It seems that the proposed method does not transfer general structural comprehension;
    - Further, the authors claim other graph prompting methods that rely on few downstream labels and projection heads are sub-optimal. While those methods may have costs, their application is still practical. In contrast, this model requires the match of the label **space. Is this rigid requirement not significantly less practical?
5. **Conceptual Framing (Prompting vs. Domain Adaptation):** A conceptual concern arises regarding the framework's classification as a new "prompting" paradigm, distinct from Unsupervised Source-Free Domain Adaptation (SFDA).
    - The authors argue that their method "fundamentally differs" from SFDA because the GNN remains frozen. This distinction, however, appears to be more of an *implementation* (parameter-efficient) choice rather than a *conceptual* one. Both approaches solve the same problem (adapting a fixed task) using the same core constraint (a shared class space, enforced by pseudo-labels).
    - Therefore, the claim that this setting "firmly places" the work in a new paradigm may be overstated. It might be more accurate to position this as a novel, parameter-efficient *variant* of SFDA. Clarifying this positioning would strengthen the paper's claims.

**Minor Concerns:**

- Please review the citation formatting; parenthetical citations should generally use `\citep`, reserving `\citet` for in-text references.

[1] All in One and One for All: A Simple yet Effective Method towards Cross-domain Graph Pretraining, KDD 2024.

[2] Non-homophilic graph pre-training and prompt learning, KDD 2025.

[3] DAGPrompT: Pushing the limits of graph prompting with a distribution-aware graph prompt tuning approach, WWW 2025.

[4] ProG: A Graph Prompt Learning Benchmark, NeurIPS 2024.

### Questions
Please see the weakness section.

### Soundness
1

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper introduces a parameter-efficient graph adaptation method that freezes the pre-trained GNN and learns only small prompts without labeled data to adapt to new graph datasets/tasks.

### Strengths
1. It's impressive UGPrompt outperform competitive baselines without leveraging labeled data for adaptation.
2. Prompts are small and modular, easy to plug into different backbones.
3. Clear and well-defined setting, stricter than standard source-free adaptation, which is well-motivated.
4. The paper is well written and easy to follow.

### Weaknesses
1. The proposed method heavily depends on pseudo-label confidence; when the backbone is poorly calibrated under distribution shift, performance may decline.
2. Node classification requires ego-subgraph extraction, adding preprocessing overhead.
3. The proposed method cannot handle label distribution shift, since the classifier head remains frozen.

### Questions
It would be beneficial if other important graph learning tasks (regression, generation) could also be evaluated.

### Soundness
3

### Presentation
3

### Contribution
3
