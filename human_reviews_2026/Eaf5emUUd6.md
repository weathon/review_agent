# Towards Understanding Feature Learning in Parameter Transfer

- Decision: Reject
- Scores: 2, 4, 8, 6

## Abstract
Parameter transfer is a central paradigm in transfer learning, enabling knowledge reuse across tasks and domains by sharing model parameters between upstream and downstream models. However, when only a subset of parameters from the upstream model is transferred to the downstream model, there remains a lack of theoretical understanding of the conditions under which such partial parameter reuse is beneficial and of the factors that govern its effectiveness. To address this gap, we analyze a setting in which both the upstream and downstream models are ReLU convolutional neural networks (CNNs). Within this theoretical framework, we characterize how the inherited parameters act as carriers of universal knowledge and identify key factors that amplify their beneficial impact on the target task. Furthermore, our analysis provides insight into why, in certain cases, transferring parameters can lead to lower test accuracy on the target task than training a new model from scratch. Numerical experiments and real-world data experiments are conducted to empirically validate our theoretical findings.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper develops a dynamic-theory framework for parameter transfer in which an $\alpha$-proportion of weights from an upstream two-layer ReLU CNN are inherited by a downstream model, formalizing how inherited parameters convey universal knowledge and when transfer improves over training from scratch. It identifies key factors—shared-signal strength, source sample size, noise levels, and dimension—and derives sharp conditions (captured by an interpretable scalar $\Gamma$) that delineate beneficial transfer from negative transfer, offering mechanistic explanations for both outcomes. Controlled simulations and CIFAR-10/100 experiments with ResNet, VGG, and DeiT corroborate these predictions.

### Strengths
1. **Rigorous analysis of feature learning under partial parameter transfer.** The paper builds a concrete, analyzable setup that tracks training dynamics beyond purely lazy/NTK views, explicitly decomposing shared signal versus task-specific noise and characterizing how these components evolve during pretraining and downstream fine-tuning.
    
2. **Testable predictions via an interpretable scalar criterion.** The analysis aggregates key factors—inheritance ratio $\alpha$, source data size $N_1$, shared-signal strength $|u|$, noise levels, and dimension $d$—into a compact indicator that predicts phase-like behavior between beneficial transfer and negative transfer; notably, these qualitative predictions align with controlled synthetic studies and with trends observed on standard vision benchmarks using canonical architectures.
    
3. **Mechanistic explanation of negative transfer with qualitative guidance for mitigation.** By identifying how weak shared signal coupled with inherited filters can inadvertently amplify non-shared noise, the work clarifies why negative transfer arises and suggests levers—such as moderating $\alpha$ or strengthening regularization during fine-tuning—that can reduce risk; while not a full recipe, this mechanism-level understanding usefully narrows the space of practical interventions.

### Weaknesses
1. **The core assumptions, especially $\alpha$-proportion random parameter sampling, appear misaligned with practical deployment.** From an application standpoint, it is uncommon that “some arbitrary subset of weights” is available while others are not; more realistic constraints expose only certain layers or interfaces (e.g., a subset of layers or layer outputs). The motivation provided for adopting random sampling does not convincingly reflect these scenarios, and the paper does not clearly justify why random sampling is the right abstraction. Please clarify whether this choice is primarily for analytical tractability, and explicitly discuss how the conclusions would change under more realistic constraints such as layer-wise availability or fixed adapter interfaces.
    
2. **The claimed practical guidance remains vague and lacks actionable procedures.** While the paper states that its analysis can guide practice, it does not articulate how one would operationalize the findings in a real pipeline. The qualitative dependence of transfer effectiveness on data scale and source–target relatedness is well known and intuitive; the hard part is how to _measure_ a model’s potential transferability and how to _attain_ the best transfer in situ. As presented, the theory does not specify a concrete, data-driven procedure for estimating transferability on a new target (e.g., from a small validation split) nor a clear decision rule for when to prefer partial transfer over full fine-tuning. Please make explicit what operational steps a practitioner should follow—what to compute, how to select $\alpha$, and how to decide between full fine-tuning, partial initialization, or alternative adaptation mechanisms.
    
3. **The positioning with respect to prior theory on transfer learning is incomplete and the exposition obscures the contribution.** I am not a theory specialist, but a cursory read of the transfer-learning portion of the related work did not surface theoretical analyses directly comparable to this study, which is unexpected for a paper centered on transfer-learning theory. The omission makes it difficult to assess novelty and significance, and the writing in other sections compounds the issue: from the abstract and introduction alone, it is hard to quickly grasp the precise value and scope of the contribution. Please expand and structure the related work to situate this analysis among prior theoretical efforts on transfer, and revise the abstract/introduction to foreground the problem setup, key assumptions (including $\alpha$-sampling), and the main takeaways in a way that is immediately accessible.

### Questions
See weaknesses.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper presents a theoretical analysis of partial parameter transfer, a setting where a downstream model inherits only a subset of parameters from a pre-trained upstream model. The study is performed within a specific theoretical framework: both upstream and downstream models are two-layer ReLU convolutional neural networks (CNNs), and the data for both tasks is generated with a shared "universal" signal component and task-specific signal components. The major contribution is the theoretical framework and link to negative transfer.

### Strengths
* The paper provides an important theoretical contribution of partial parameter transfer as well as links to negative transfer which i find really interesting. This work is highly relevant, as architectural mismatches (e.g., different model sizes, new input/output heads) are the norm in practical transfer learning, yet most theory assumes a full-model transfer.

* The paper claims to be one of the first to analyze the training dynamics of this process, moving beyond static generalization bounds.

* well-designed experiments on classical ResNets and Transformer models.

### Weaknesses
W1: **Writing**: The paper provides important theoretical contributions but i found the paper to not be very readable,  finding sare presented as large, dense, and complex mathematical conditions that are extremely difficult to parse for a non-expert in this specific theoretical subfield

L:132: $\Omega$ $\Theta$ definition is confusing at first they seem like variable but are conditions. 

Specific Instances: L282, cannot really understand what is the condition for negative transfer

W2: Missing Citations, Highly relevant papers which were not cited:

[1]Characterizing and Avoiding Negative Transfer, wang et al CVPR 2019.

[2] Representation Alignment in Neural Networks, Imani et al. TMLR 2024

[3] Identification of Negative Transfers in Multitask Learning Using Surrogate Models, Li et al. TMLR 2023

 W3: **Problematic and wrong Citations:**

Correct citation for  Vershynin R. et al is:  High-Dimensional Probability: An Introduction with Applications in Data Science. Cambridge University Press; 2018. The Authors did not include the book title in the citations. Url: https://www.cambridge.org/core/books/highdimensional-probability/797C466DA29743D2C8213493BD2D2102


Made up citation!!: **JIANG, Z. ET AL. (2022). Transfer learning with pre-trained models: A survey. arXiv preprint arXiv:2209.01791**, I cannot find it. Is this a hallucinated citation?


Minor: Typo in Section G of the appendix title, Discusstion -> Discussion. 

Overall, i believe that the paper should go through another rewriting to avoid these mistakes and make the paper more readable for a non-expert audience as well.

### Questions
Q1: Proposition 4.4 (part 2)  seems to imply that negative transfer is most likely when one transfers from a very large but poorly-aligned task. Is this a correct interpretation?

Q2 Generally, transfer is not random but based on first $m$ layer or $n-1$ layer,s where $n$ is the total number of layers, how does the current work take that into account? 

Q3 Can a connection to [1] and [2] be established?

S1 Suggestion: Have a uniform citation format. Sometimes, the page number is included in NeurIPS; sometimes it's not, sometimes conference names are abbreviated, and sometimes written in full.

Q4 Is this theory only valid for classification, or canit  be generalized to regression and other tasks somehow?


[1]Characterizing and Avoiding Negative Transfer, wang et al CVPR 2019.

[2] Representation Alignment in Neural Networks, Imani et al. TMLR 2024

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper provides a theoretical analysis of feature learning in parameter transfer, a core mechanism in transfer learning where parts of a pretrained model are reused for downstream tasks. Focusing on ReLU convolutional neural networks, the authors study when and why partial parameter reuse is beneficial. Their framework reveals how transferred parameters encode universal knowledge that can enhance learning efficiency under certain conditions, while also explaining cases where parameter transfer may hurt downstream performance compared to training from scratch. Theoretical insights are supported by both numerical simulations and real-world experiments, offering a clearer understanding of the mechanisms governing successful parameter transfer.

### Strengths
The theoretical derivations appear rigorous, and the topic—providing a theoretical explanation for transfer learning—is both timely and important. The use of feature learning analysis to study parameter transfer offers a novel and interesting perspective.

### Weaknesses
Some assumptions are a bit restrictive, such as assuming $u$ is orthogonal to both $v_1$ and $v_2$. In addition, certain terminologies, including "Bayesian optimal" and "sub-Bayesian optimal" as used in Theorems 4.2 and 4.3, require clearer definitions or explanations to ensure accessibility for a broader audience.

### Questions
1. In Definitions 3.1 and 3.2, the covariance matrices of the noise are given as $\sigma_{p,1}(I - uu^\top / \|u\|^2 - v_1v_1^\top / \|v_1\|^2)$ and $\sigma_{p,2}(I - uu^\top / \|u\|^2 - v_2v_2^\top / \|v_2\|^2)$, respectively. However, the last paragraph of Page 3 states that "the noise variances in Task 1 and Task 2 are $\sigma_{p,1}$ and $\sigma_{p,2}$." These descriptions seem inconsistent—please clarify the relationship between them.  
2. Still in Definitions 3.1 and 3.2, the covariance matrices imply that different elements of the one noise vector may be correlated, with correlations depending on $u$ and $v$. This modeling choice requires justification. Moreover, it should be verified (or stated) that the covariance matrices are always positive semidefinite, as required for valid covariance definitions.  
3. When $d = 1$, it is impossible to have $u \perp v_1$ and $u \perp v_2$. It should therefore be noted somewhere that $d \geq 2$ is assumed throughout the analysis.

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
The paper investigates the role of parameters shared between upstream and downstream models. Inheriting parameters can serve as a carrier of general knowledge and is sometimes beneficial for the target task; however, in certain cases, transferring parameters leads to lower accuracy on the target task than training from scratch.

### Strengths
1.The paper is highly readable.
2.The theoretical derivations are complete, lending strong credibility.
3.It substantiates a key conclusion: inheriting more parameters, using larger upstream training datasets, and having less noise in upstream tasks can improve downstream model performance. The conclusions in the contributions section are insightful.

### Weaknesses
Please refer to the Questions section below.

### Questions
1.The case would be stronger with experiments on ViT-based models or VLMs.
2.There is a lack of cross-dataset experiments across different tasks.
3.The paper does not provide sufficient comparisons with existing parameter-transfer or transfer-learning methods.

### Soundness
3

### Presentation
3

### Contribution
2
