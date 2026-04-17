# Data Diversity for Compositional Generalization

- Decision: Reject
- Scores: 4, 2, 6, 2

## Abstract
Human cognition excels at understanding complex concepts by combining simpler, learned elements, enabling efficient learning and generalization to novel scenarios.
Recent work suggests that machine learning models may exhibit a similar capability, generalizing to novel scenarios by first acquiring fundamental components and then recombining them.
Data serves as the driving force behind this process, and the diversity of training data plays a crucial role in shaping a model's ability to generalize.
In this work, we introduce a framework that disentangles the multifaceted notion of diversity and formalize its impact on model performance and generalization ability from different perspectives.
Through both theoretical analysis and empirical validation, we demonstrate that increasing diversity without a principled strategy does not necessarily lead to optimal generalization ability.
Instead, a deeper understanding of data diversity is required.
Building on this insight, we propose a high-level guideline for dataset designing and preparing that facilitate more efficient learning and enable improved generalization to unseen compositions.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies the two data dimensionalities of semantic diversity and structural diversity individually, with Rademacher complexity and covering numbers as theoretical tools, respectively. The experiments on synthetic datasets and real-world datasets with fixed data budget confirm the theoretical findings and provide insights like the importance of a principled strategy with respect to data diversity to improve compositional generalization.

### Strengths
- The theoretical findings, e.g., bounds of generalization, are sound. The formulations are well-defined and tools are established (i.e., Rademacher complexity and covering numbers), though there are some typos. 
- The experiments on datasets, including synthetic datasets and real-world datasets, are comprehensive and the conclusions are insightful.

### Weaknesses
- The theoretical results still hinge on the validity of the assumptions. The embedding of all compositional functions into a d-dimensional Euclidean space. How well does this idealized geometry capture the true "conceptual space" of real-world problems? If the training structural combinations are confined to a narrow space in this d-dimensional space, the generalization performance is expected to be very low (i.e., a large radius $\epsilon$ to cover all possible combinations). 
- The scope of this work is limited to algebraic circuits. Thus, only structural and semantic diversities are considered, which, based on my understanding, refer to "how different the training samples are" and "how many combinations are in each structure". The contribution could be further improved by generalizing the ideas to other mathematical settings, which may introduce other types of composlitionality. 
- As pointed out by the author, the equation 5 ignores the potential coupling interactions. How does this correlation affect the generalization error?  
- There are some typos in the equations, e.g., in line 215, $|\phi(c_1)-\phi(x_2)|$ should be $|\phi(c_1)-\phi(c_2)|$.

### Questions
- Why in figure 4 and 5, M is small (i.e., the maximum examined M is 5), compared with figure 3 (i.e., the minimum examined M is 10)? What happens when M is 10 for GPT-2-XL and Mistral? What will the figure change if increasing the data budget from 4500 to a larger value like 10000?
- I am curious about the impact of different training strategies. If using parameter-efficient tuning instead of fine-tuning, will the effects of the two types of diversity be different?​​
- Why is it necessary to define $s(\cdot)$ as the sub-circuit in $c(\cdot)$? This notation seems not to be used in the rest of this paper.

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
3

### Summary
This paper investigates the role of *compositional generalization* under limited data budgets and introduces a theoretical framework that decomposes data diversity into two orthogonal dimensions: **semantic diversity (M: component variations)** and **structural diversity (N: compositional variations)**. The authors provide generalization bounds for intra- and inter-compositional learning and propose an optimal allocation strategy between M and N. The theoretical claims are validated on synthetic algebraic circuit datasets using both Transformers and large language models (e.g., GPT-2-XL, Mistral 7B), and further extended to real-world reasoning tasks such as GSM8K.

### Strengths
The paper is generally well-structured, and the exposition is clear enough to follow the main ideas.

The theoretical formulation provides a reasonable attempt to justify the proposed perspective.

### Weaknesses
Several important related works on compositionality and LLM-based structure-sensitive generalization are missing. For example:
	•	“Compositional Semantic Parsing with Large Language Models” (ICLR 2023)
	•	“Does Data Scaling Lead to Visual Compositional Generalization?” (ICML 2025).

The evaluation is limited to GSM8K, which is a relatively shallow reasoning dataset where compositional structure is implicitly assumed rather than explicitly grounded. Thus, it remains unclear whether the observations hold in more rigorous compositional benchmarks (e.g., SCAN, COGS, PCFG-based datasets).
	

No comparison is made against standard baselines such as common data augmentation techniques or existing data selection strategies. Therefore, it is unclear whether the reported gains originate from the specific “semantic vs. structural balance” or simply from paraphrase-based data expansion.


The experiments are conducted with only two models (“GPT-2-XL (1.5B)” and “Mistral-7B”), which is insufficient to support general claims about scaling trends or the universality of the proposed findings.

### Questions
pls see the Weaknesses.

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
4

### Summary
This paper studies data diversity for compositional generalization and argues that “more diverse data” is not uniformly better. It formally separates semantic diversity (many surface realizations of the same structure) from structural diversity (many distinct compositional structures), derives generalization bounds for each, and shows that under a fixed data budget K=M ⋅ N, increasing one kind of diversity can hurt if the other is the real bottleneck. The theory (via Rademacher complexity for intra-compositional generalization and covering-number arguments for inter-compositional generalization) yields an error bound with an interior optimum in N, explaining why we need to balance semantic and structural diversity. Experiments on synthetic algebraic circuits and on GSM8K (via GPT-4o–generated variants) confirm that when component complexity is high and compositional complexity is low, semantic diversity wins; when components are simple and recombination is the challenge, structural diversity wins.

### Strengths
**1. Intuitive and Robust Theoretical Upperbound of compositional generalization error**

-- Authors study both semantic diversity and structural diversity to disentangle data factors that actually influence compositionality. They established theoretical upper bounds for both intra-compositional and inter-compositional generalization error, which corresponds to the two types of diversity.

-- The overall upper bound of generalization error reveals a crucial fact: increasing diversity without a principled strategy does not necessarily lead to optimal generalization ability, and the training dataset requires a balanced allocation between semantic and structural diversity to achieve data efficiency.

**2. Solid experiment design and strong empirical results**

-- The designed experiments align well with the theory to be tested. The synthetic circuit setup is well controlled and directly tests the theoretical stationary point 

-- Real-data validation on GSM8K that shows semantic-augmented subsets can match or beat full-data fine-tuning under the same budget.

### Weaknesses
**-- Missing citations:** A few previous works [1,2] also discussed the importance of data diversity for compositional generalization.

[1] Zhou, Xiang, Yichen Jiang, and Mohit Bansal. "Data factors for better compositional generalization." EMNLP 2023.
[2] Akyürek, Ekin, and Jacob Andreas. "LexSym: Compositionality as lexical symmetry." ACL 2023.

**-- Undefined notation of semantic/structure complexity:** See my question below on distinguishing between semantic/structure complexity (d,r) and semantic/structure diversity (M/N).

### Questions
-- A few citation format mistakes: For example, authors should use \citep in line 156.

-- When the notations of “semantic/structure complexity (d,r)” are first discussed in Theorem 3.3, they are not properly defined. In Theorem 3.1, it’s not immediately clear why the dimension of $\mathcal{A}$ “characterizes the complexity of the underlying components”. Similarly, in Theorem 3.2, authors mentioned space dimension d characterize structure complexity. Since these two terms are so important for the theoretical contribution of this paper, they should be properly introduced as authors did for semantic/structure diversity (N,M).

-- Most previous work studying compositional generalization trains Transformer networks from scratch rather than from a pretrained checkpoint to eliminate the influence of pretraining data (e.g., a pretrained model may have already learned to do arithmetic compositionally). While Sec 4.1.1 focuses on training encoders from scratch, in Sec 4.1.2, the authors should at least discuss the potential effect of using pretrained models.

-- Can the authors propose a practical estimator of (r,d), even proxy-level, so that the balance between semantic vs structural diversity can be decided automatically for a new task?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The work studies compositional generalization in the algebraic circuits context. The work argues for making a distinction between semantic and structural diversity, emhpasizing that semantic diversity by itself may not be sufficient for achieving compositional generalization. An attempt is made to come up with a theoretical framework to include these diversity conditions into a compositional generalization framework.

### Strengths
1. The problem of compositional generalization is relevant. The scope (algebraic circuits in controlled settings) is easy to understand and reasonable.
2. The writing is clear

### Weaknesses
1. Generalization results (Sections 3.2, 3.3) read like speculation. The analysis essentially extrapolates iid generalization arguments to an __assumed__ configuration within Euclidean space. That is not a genuine distribution-shift generalization proof. I don't see any justification of the assumptions, which makes the results disconnected. 
2. Empirical evaluations are tiny, e.g. Fig 3, arguably the graph that should contain most important results, only shows the training progress of three configurations. 
3. Related work has made a distinction between diversity and pure scale and its impact on compositional generalization (see, e.g., [1, 2]); these should be discussed and positioned within this work. 

[1] Uselis, Arnas et al. “Does Data Scaling Lead to Visual Compositional Generalization?” arXiv:2507.07102.    
[2] Zhou, Xiang et al. “Data Factors for Better Compositional Generalization.” arXiv:2311.04420.

### Questions
1. Can the analysis be extended to a larger set of variations of of N, M? Current Fig. 3 only shows three configurations.

### Soundness
1

### Presentation
3

### Contribution
1
