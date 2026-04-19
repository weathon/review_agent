# Exact Certification of (Graph) Neural Networks Against Label Poisoning

- Decision: Accept (Spotlight)
- Scores: 8, 8, 8, 6

## Abstract
Machine learning models are highly vulnerable to label flipping, i.e., the adversarial modification (poisoning) of training labels to compromise performance. Thus, deriving robustness certificates is important to guarantee that test predictions remain unaffected and to understand worst-case robustness behavior. However, for Graph Neural Networks (GNNs), the problem of certifying label flipping has so far been unsolved. We change this by introducing an exact certification method, deriving both sample-wise and collective certificates. Our method leverages the Neural Tangent Kernel (NTK) to capture the training dynamics of wide networks enabling us to reformulate the bilevel optimization problem representing label flipping into a Mixed-Integer Linear Program (MILP). We apply our method to certify a broad range of GNN architectures in node classification tasks. Thereby, concerning the worst-case robustness to label flipping: $(i)$ we establish hierarchies of GNNs on different benchmark graphs; $(ii)$ quantify the effect of architectural choices such as activations, depth and skip-connections; and surprisingly, $(iii)$ uncover a novel phenomenon of the robustness plateauing for intermediate perturbation budgets across all investigated datasets and architectures. While we focus on GNNs, our certificates are applicable to sufficiently wide NNs in general through their NTK. Thus, our work presents the first exact certificate to a poisoning attack ever derived for neural networks, which could be of independent interest. The code is available at https://github.com/saper0/qpcert.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper introduces exact certificates for graph neural networks against label poisoning. The author reformulate the certification problem into a Mixed Integer Linear Program (MILP) using  Neural Tangent Kernel (NTK), deriving both sample-wise and collective certificates for GNNs in node classification tasks. The method is tested on various GNNs and datasets.

### Strengths
- The paper is well-written and easy to follow. It introduces first exact certificates for graph neural networks against label poisoning. 
- The evaluation presents insights that guide future designs and improvements in GNNs regarding label poisoning.

### Weaknesses
+ The NTK is data dependent, but the paper only evaluates two real-world datasets, Cora-ML and Citeseer. This raises concerns about the generalizability of the findings. It would strengthen the paper to evaluate on a broader variety of real-world datasets.
+ Given that the core technique of NTK-based MILP formulation is already established [1], the contribution seems incremental.

References:

[1] Gosch, Lukas, et al. "Provable Robustness of (Graph) Neural Networks Against Data Poisoning and Backdoor Attacks." arXiv preprint arXiv:2407.10867 (2024).

### Questions
In addition to the concerns mentioned in the weaknesses, some figures are not very reader-friendly, such as Figures 2c 3c. Consider adjusting the colors and legends for better clarity.

### Soundness
4

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper presents a novel approach to certify Graph Neural Networks (GNNs) against label poisoning attacks by introducing exact sample-wise and collective robustness certificates. The authors leverage the Neural Tangent Kernel (NTK) to capture training dynamics, reformulating the label-flipping problem as a Mixed-Integer Linear Program (MILP). This method allows for exact robustness certification across multiple GNN architectures in node classification tasks, establishing robustness hierarchies and investigating the effect of architecture-specific choices on resilience to label flipping. A key contribution is the discovery of a plateau effect in robustness for intermediate perturbation budgets, which has implications for understanding adversarial robustness in GNNs.

### Strengths
**The Problem Addressed is Highly Relevant**

Poisoning attacks, of which label flipping is one variant, are highly relevant in machine learning and pose a significant security risk. While empirical defenses can sometimes weaken this effect, it is prudent to develop methods that can guarantee their absence and accurately measure their influence. The paper thus addresses a key challenge to develop more secure and trustworthy models.

**The Proposed Method is Sound Under The Infinite-Width Assumption**

The proposed method is sound under the infinite-width assumption. While this assumption does not hold in practice (see weaknesses), it gives a good theoretical basis for reasoning about model behavior. Computing exact (and not just sound but incomplete) certificates has several advantages, including a better comparison of different models’ robustness.

**Extensive Evaluation**

Considering the method's limited scalability, the evaluation is thorough, spanning several models and datasets. I find the finding of plateaus especially intriguing, which has potential for additional investigations in the future.

**Good Reproducibility**

The authors provide many implementation details in the paper and the appendix, and include their code for reproducibility.

### Weaknesses
**The “Exact” Certificates are Approximations**

The authors claim exact certificates (e.g., L68), which means the certificate has neither false positives nor false negatives. However, this only holds under the assumption of infinite-width models, and not for finite-width models used in practice. While I am not an expert in NTKs, my understanding is that there can be significant deviations in training behavior, especially for deeper models. I would therefore argue that the framing of the certificates should be adjusted to account for this, putting less emphasis on the certificates being “exact” for neural networks and highlighting more the fact that they are approximations. The authors acknowledge this in the conclusion; however, it should be reflected in the claims (e.g., section 1, abstract) as well.

**Limited Scalability**

The second major limitation is the method’s inherent limited scalability to small-scale models and graphs. This is common to all certification methods, but especially prevalent in exact methods. I thus don’t see this as a reason against acceptance, but it does limit the method’s practical applicability and raises questions of whether the findings hold in larger-scale settings.

### Questions
L132: The NTK Q_{ij} is defined as the expectation over all model initialisations. Does this mean the certificate also holds wrt. the expectation over model initialisations? If so, it would be good to highlight this, as it deviates from most work on certification which computes certificates wrt. the concrete parameter initialisation used for the particular model training.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper "Exact Certification of Graph Neural Networks Against Label Poisoning" introduces the first exact certification method, for deriving complete sample-wise and collective robustness certificates for Graph Neural Networks, against adversarial label flipping. 
Using the Neural Tangent Kernel (NTK), the authors leverage the dynamics of wide networks in order to recast the bilevel optimization problem of label flipping, as a Mixed-Integer Linear Program (MILP). In this work, the authors establish hierarchies of GNNs on different benchmark graphs, and quantify the effect of architectural options to the worst-case robustness flipping.

### Strengths
1.   The authors introduce a novel exact robustness certification method for label flipping attacks on neural networks, particularly Graph Neural Networks (GNNs). This work is impactful given that exact certification for poisoning attacks is generally unsolved for GNNs​.
2. By leveraging the Neural Tangent Kernel (NTK), the paper rigorously reformulates the robustness certification problem as a Mixed-Integer Linear Program (MILP). The authors were able to extend previous work [1], deriving complete certificates, and collective certification.
3. The approach offers wide applicability, as it can extend beyond GNNs to general neural networks, offering a framework that could potentially influence robustness research across various domains in machine learning​.
4. The authors identified a novel phenomenon of robustness plateauing for intermediate perturbations.


[1]. Lukas Gosch, Mahalakshmi Sabanayagam, Debarghya Ghoshdastidar, and Stephan G¨unnemann. Provable robustness of (graph) neural networks against data poisoning and backdoor attacks. arXiv preprint arXiv:2407.10867, 2024.

### Weaknesses
1. The high complexity of Mixed-Integer Linear Program (MILP)  could introduce computational overhead, which may be prohibitive for real-time or large-scale applications. 
2. Additionally, the certification method is largely tested on synthetic datasets. Although effective on these datasets, it’s unclear how well it would generalize to real-world graphs.
3. Finally, the paper focuses solely on exact certification. The authors are encouraged to consider trade-offs between exactness and scalability.

### Questions
1. Scalability and Complexity: Given the computational complexity of your MILP-based certification, have you considered or tested any strategies to approximate or simplify the certification process for larger datasets? Would relaxing the exactness of the certification improve scalability while retaining useful guarantees?

2. Dataset Dependence: Your results highlight that robustness hierarchies among GNN architectures vary across datasets. Can you elaborate on why certain architectures may perform better on specific datasets? What dataset characteristics influence these hierarchies?

3. Intermediate Perturbation Plateau: You observed a plateau in robustness for intermediate perturbation budgets across several GNN architectures and datasets. Do the authors have any hypotheses about the causes of this phenomenon, and do you plan further analysis to understand it more deeply?

4. Extension Beyond GNNs: While the method can theoretically be applied to other types of neural networks, have you tested it on non-GNN architectures? If so, how did the performance and scalability compare?

5. Unaddressed Adaptability to Dynamic Graphs: Real-world graphs often change over time, yet the certification process assumes a static graph. Can your methodology adapt to provide certification for dynamic or evolving graphs?

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
This work investigates the exact robustness certification for GNN against label poisoning attack. They use the NTK to approximate the wide GNN, and reformulate the bilevel optimization problem into a MILP. The authors believe that their framework can be applied to other GNNs through its NTK. Finally, they conduct experiments to show their certification for various GNNs over several datasets.

### Strengths
1. The robustness certification is a vital problem for GNN and other NNs. It is hard to obtain an exact robustness certification.
2. The author proposes a novel framework for robustness certification against label poisoning, and their main idea is to approximate the model by its NTK. It seems to be easy to apply this framework to other models.
3. The computational complexity of solving the MILP is not very large. The authors conduct experiments on several datasets and GNNs to show its empirical performance.

### Weaknesses
1. My main concern is about the approximation error when using NTK to approximate GNN. The approximation error exists unless the model is infinitely wide, especially when we use a pooling layer. However, the authors aim to obtain exact certificates, but the approximation error is not considered in this work.
2. A limitation of this framework is that it requires the width of the model to be sufficiently large, and it cannot be used for narrow NNs. 
3. An important recent work is [a], which studies the robustness certification of GNN against data poisoning. I find that the techniques and framework in this work are very similar to [a], which limits the contribution of this work.
4. The authors only show the computational complexity of solving the MILP. However, since it approximates GNN by its NTK, it is necessary to show the computational complexity of computing the NTK of GNN.

[a] Lukas Gosch, Mahalakshmi Sabanayagam, Debarghya Ghoshdastidar, Stephan Günnemann. Provable Robustness of (Graph) Neural Networks Against Data Poisoning and Backdoor Attacks. Arxiv 2024.

### Questions
1. When considering the approximation error in this framework, is this method an exact certification method? (especially when the width of GNN is not sufficiently wide).
2. How to evaluate the tightness of the certifies computed by this method?
3. What is the total computational complexity of the proposed method?
4. What is the novelty of the techniques in this work?

### Soundness
3

### Presentation
3

### Contribution
2
