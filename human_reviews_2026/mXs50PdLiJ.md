# Bridging Neural Learning and Symbolic Reasoning: A Differentiable Fuzzy Description Logic Framework

- Decision: Reject
- Scores: 6, 2, 6, 6

## Abstract
Injecting logical reasoning as a structural prior into deep learning models is a central goal of Neural-Symbolic AI. A key challenge is 
making expressive formalisms like the Description Logic (DL) $\mathcal{ALC}$ compatible with gradient-based optimization without sacrificing their rich semantics. While the theory of fuzzy $\mathcal{ALC}$ offers a path, a significant methodological gap has persisted: although operator-agnostic fuzzy reasoning has been explored in first-order logic-based frameworks, these methods face fundamental computational barriers when scaling to real-world ontologies, leaving the choice of fuzzy operators for DLs largely unexplored. This paper introduces NeSy$\mathcal{ALC}$, an end-to-end learning framework designed to bridge this gap by transforming fuzzy $\mathcal{ALC}$ from a theoretical construct into a practical tool for representation learning at scale. Our framework is operator-agnostic within the DL setting, allowing us to conduct the first systematic empirical analysis of how different t-norms and fuzzy implications impact learning on diverse knowledge bases. We find that no single operator is universally optimal, motivating our second contribution: a novel adaptive dual-loss optimization strategy that dynamically adjusts its objective based on the logical structure of the knowledge base, enhancing learning robustness. Through extensive experiments on ontology completion and semantic image interpretation tasks, we demonstrate that NeSy$\mathcal{ALC}$ consistently and statistically significantly outperforms established baselines. Our work operationalizes fuzzy $\mathcal{ALC}$ for modern machine learning, providing a practical and robust framework for injecting rich symbolic knowledge into neural models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a differentiable version of the description logic fuzzy ALC. Towards this end, it explores different choices of fuzzy operators and employs a dual loss that takes both T-Box and A-Box axioms into account. For evaluation, the concept membership function is partially randomized and it is evaluated how many T-Box axioms are fulfilled by the solution (compared to the perfect solution without randomization).

### Strengths
- Timely, original topic of making fuzzy ALC differentiable
- Extensive experiments on 8 datasets
- Proofs of theoretical properties of the differentiable operators in Appendix
- The approach is clearly presented
- The paper goes beyond previous approaches that mainly focused on EL (instead of ALC)

### Weaknesses
- The main result table (Table 1) is not clearly described. What exactly does "rule-based models" mean? What exactly does "hierarchy models" mean? I assume their loss is different. This should be made more explicit
- The use of fuzzy operators in the evaluation seems inconsistent: Why does Yager only appear under rule-based models (Table 1)? Why does Hamacher not appear under "adaptive dual loss"?

### Questions
see Weaknesses above

### Soundness
3

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
4

### Summary
The paper proposes NeSy ALC, a neurosymbolic framework based on the description logic ALC with fuzzy semantics for differentiability.
NeSy ALC claims to be more flexible and better performing than the state-of-the-art in neuro-symbolic AI.

### Strengths
- Description logics is a language formalism relevant to many benchmarks and tasks related to ontologies.
- The presentation of the framework itself is quite clear.

### Weaknesses
The paper overclaims at several occasions.
1. This is not the first fuzzy framework to be operator-agnostic. While LTN suggests default operators for their experiments, the LTN papers (Badreddine et al., 2020; Serafini and Garcez, 2016) clearly define their Real Logic semantics as operator-agnostic. Differentiable Fuzzy Logics (van Krieken et al, 2020) also defines semantics with many fuzzy operators. If “frameworks” is meant as code libraries rather than semantics, the LTN library also implements most operators presented in this paper. There are other examples: LYRICS (Marra et al, 2019), SBR (Diligenti et al, 2017), ...
2. The proposed modification in Section 3.3.2 does not solve Reasoning Shortcuts. There seems to be a misunderstanding on what RSs are. In C->D, a reasoning shortcut can still occur for C=D=1. If anything, this loss design encourages convergence to more crisp solutions. But it cannot help identifying which logical optima are correct or incorrect in the real world (this is impossible from a purely loss-centric approach). This claim is also not backed by experiments on RS benchmarks like Bortoletti 2024, or grounded in any theory properly connected to RSs (Marconato 2023, Marconato 2025).
3. About the dual-loss objective: having different weights for different rules is not new (Marra et al, 2019). The difference here is the heuristic used to calculate the weights, which is interesting. Unfortunately it’s quite ad-hoc and I don’t think there is a comparison with a baseline where we simply sum the two losses without weights (unless I missed something?).

I am also concerned with the lack of experimental details. The paper does not report the knowledge used for each experiment and baseline, nor the hyperparameters of the baselines. For example, NeSy-ALC with Product and S-implication is quite similar to LTN with Stable Product semantics except for the semantic gate. So I am surprised that NeSy-ALC gets twice the performance on, e.g., Nihss. I would like to see more details on how the baselines are implemented.

There is also no comparison with probabilistic frameworks.

In general, the novelty feels quite limited for ICLR. Fuzzy ALC systems like FALCON already exist. I think the difference between this framework and the existing sota NeSy framework is not properly identified. Overall, it’s not clear why one should use this framework over other -very similar- fuzzy NeSy frameworks except for the fact that it supports another way to write knowledge.

- Serafini and Garcez, 2016: Logic Tensor Networks: Deep Learning and Logical Reasoning from Data and Knowledge
- Diligenti et al, 2017: Semantic-based regularization for learning and inference
- van Krieken et al, 2020: Analyzing Differentiable Fuzzy Logic Operators
- Marra et al, 2019: LYRICS: a General Interface Layer to Integrate AI and Deep Learning
- Badreddine et al, 2020: Logic Tensor Networks
- Bortolotti et al, 2024: A neuro-symbolic benchmark suite for concept quality and reasoning shortcuts.
- Marconato et al, 2023: Not all neuro-symbolic concepts are created equal: Analysis and mitigation of reasoning shortcuts
- Marconato et al, 2025: Symbol Grounding in Neuro-Symbolic AI: A Gentle Introduction to Reasoning Shortcuts.

### Questions
1. “We find that no single operator is universally optimal, motivating our second
key contribution: a novel adaptive dual-loss optimization strategy that dynamically adjusts its objective based on the logical structure of the knowledge base, enhancing learning robustness.” How is the dual-loss strategy a consequence of no operators being universally optimal?
2. What are the knowledgebases you used for each experiment and each baseline?
3. What hyperparameters did you use for the baselines?
4. Do you have an ablation of the semantic gate?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces NeSyALC, an end-to-end learning framework that bridges the gap between theory and practice by transforming fuzzy ALC from a purely theoretical construct into a practical tool for representation learning. The authors note that no single operator is universally optimal, motivating their second major contribution—an adaptive dual-loss optimization strategy that dynamically adjusts its objectives according to the logical structure of the knowledge base, thereby enhancing learning robustness. Overall, this work operationalizes fuzzy ALC for modern machine learning, offering a practical and resilient framework for integrating rich symbolic knowledge into neural models.

### Strengths
The paper proposes a novel foundational approach to integrating description logic into deep learning. The authors conduct extensive experiments to determine which t-norm is best suited for the task. They further introduce a loss function derived from the axioms and the logical relations among them.

### Weaknesses
The paper provides limited discussion on how description logic is selected and applied in realistic scenarios involving neural networks. For example, it remains unclear how the axioms and their concepts are defined in practical datasets.

In addition, the overall presentation of the paper could be further improved. For instance, introducing an independent Preliminaries section to formally describe well-defined concepts, such as fuzzy interpretation, would enhance clarity and readability.

### Questions
1.	In Line 187, the authors mention that different t-norms can be combined. What is the specific method used for combining different t-norms, and under what conditions is such a combination chosen? However, according to Table 1, the results indicate that only a single t-norm was used for testing. Therefore, the related statements appear to be inconsistent or unclear.
2.	When applying the ALC ontology and rules to neural networks, how is the loss function determined, including the proposed total loss function ${L}_\text{total}$ presented in the paper? Furthermore, how are the weights between the proposed adaptive dual-loss and the original neural network loss defined or balanced?
3.	Some prior works have already explored differentiable first-order logic, applying t-norm into neural networks, and applied first-order language to neural networks. What is the motivation for using description logic instead of first-order logic in this work?

### Soundness
3

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
5

### Summary
This paper introduces NeSyALC, a differentiable framework for integrating a fuzzy Description Logic (ALC) with neural learning. The framework is operator-agnostic, supporting various t-norms and implications, and includes an adaptive dual-loss mechanism that dynamically weights hierarchical and rule-based objectives according to ontology structure. The paper provides formal proofs of soundness and semantic preservation and evaluates the framework on eight ontologies for ontology completion and query answering tasks. Experimental results show that NeSyALC performs consistently well across datasets and generally outperforms existing neuro-symbolic baselines.

### Strengths
⁠Well-motivated introduction: The paper addresses a clear broad motivation for making fuzzy ALC compatible with gradient-based learning.

•⁠  ⁠Sound theoretical foundation: The formal results (soundness and semantic preservation) are clearly presented and relevant.

•⁠  ⁠Systematic and diverse evaluation: Uses multiple ontologies and fuzzy operators, producing informative and reproducible results.

•⁠  ⁠Clarity and organization: The paper is clearly written, with logical structure and appropriate technical depth.

•⁠  The appendix including complete proofs, ablations, and reproducibility materials that enhance transparency and rigor.

### Weaknesses
-the conceptual novelty is modest. The approach  primarily integrates existing fuzzy ALC semantics with differentiable optimization (and the fuzzy implications had been already well studied in (Van Krieken et al 2022). So I am surprised to not see it as a benchmark e.g., sigmoidal implication. Related to this two issues: 1) how ontologies are described in LTN is not clear. 2) Moreover BoxEL and variants as benchmarks:  Apparently they have limited expressivity due to light fragments like EL. So does comparing it in more expressive ontologies fair to these approaches?    Overall,  it looks  to me more like a careful engineering of known elements than a conceptual breakthrough.

- Another weakness/limitation is that while it claims to bride perception and reasoning, the tasks (masked ABox revision and conjunctive_query answering) are still synthetic and ontology-centric. In other words symbolic. What is presented is rather approximate fuzzy ALC rather than full NeSy. One of the biggest promise of NeSy approaches are grounding the symbols in sensory data e.g., images. And in this paper,  there’s no such perceptual integration (such as neural vision–logic tasks).  In the end, the paper claims robustness “across diverse ontologies,” but they are all  still within a narrow symbolic reasoning domain. It’s not clear how this framework performs in more complex, multimodal, or real-world noisy environments. 


- While the framework is operator-agnostic, which is nice,  it doesn’t provide a principled criteria to choose or optimize operators automatically. The paper left it  as future work, granted but this also weakens the claim of “bridging” fuzzy ALC theory with practice.


- Focus  is mainly on F1 score and success rate, but I am missing some  statistical significance tests.

### Questions
Please feel free to respond any of the listed weaknesses.

In addition two questions:

1.⁠ ⁠Could the authors discuss whether the adaptive dual-loss mechanism ever destabilizes optimization, e.g. in ontologies with extreme imbalance in quantified vs. atomic axioms?

2.⁠ ⁠Is is possible to generalize the function behaviour of NeSyALC with respect to any set of fuzzy operators (e.g., different t-norms and t-conorms) based on the results of the families tested? For the parametrized t-norm families (Yager, Hamacher, ...), could the parameters be learned instead of set?

### Soundness
3

### Presentation
3

### Contribution
2
