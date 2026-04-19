# DPD-LoRA: Dynamic Prompt-Driven Low-Rank Adaptation for Improved Generalization

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 5, 5, 5

## Abstract
Fine-tuning large models presents technical challenges such as catastrophic forgetting and parameter inefficiency. Low-rank Adaptation (LoRA) and Propmt Learning can help address some of these challenges by providing more compact and flexible representations. However, Low-rank approximation is susceptible to outliers and relies on the assumption of a global low-rank structure, which can be suboptimal. Additionally, Prompt learning can overfit to specific downstream tasks, reducing its effectiveness when adapting to new tasks. In this paper, we introduce $\textbf{Dynamic Prompt-Driven Low-Rank Adaptation (DPD-LoRA)}$, a novel framework that seamlessly integrates task-specific guidance using hierarchical prompt tokens and parameter-efficient adaptation. Unlike traditional methods, task-aware prompts in the DPD-LoRA dynamically influences low-rank updates in the model's parameters, thus enabling robust adaptation and generalization across diverse tasks and mitigating the forgetting issues. We further improve the learning capabilities of the model by breaking down the standard LoRA into multiple low-rank sub-matrices, without adding additional parameters. Further, we use an adaptive loss function to guarantee alignment with the distribution of the pre-trained model. Specifically, we introduce a self-regulated mechanism to improve stability, and a soft-gated selection mechanism to decide when to activate adaptation modules to improve performance on unseen categories. Extensive experiments on 11 benchmark datasets demonstrate that DPD-LoRA significantly outperforms state-of-the-art methods in both accuracy and generalization, offering a comprehensive solution to the challenges of fine-tuning large-scale models.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This manuscript introduces DPD-LoRA, a novel framework that aims to improve the generalization capability of large models by integrating dynamic prompt-driven low-rank adaptation. This method combines hierarchical prompt tokens and parameter-efficient adaptation to incorporate task-specific guidance, demonstrating superior performance over existing techniques across multiple benchmark datasets.

### Strengths
1. The proposal of the DPD-LoRA framework is innovative as it integrates prompt learning and low-rank adaptation to enhance the model's generalization capabilities. The introduction of adaptive loss functions and soft-gated selection mechanisms (PCGM) adds to the novelty of the approach.
2. The method designed by the authors has been applied to three different tasks: Base-to-novel class generalization, Cross-dataset evaluation, and Few-shot learning, showing promising results across the board, which speaks to the effectiveness of the approach.
3. The authors have conducted extensive experiments on 11 benchmark datasets, which helps to substantiate the effectiveness of the proposed method.
4. The overall structure of the paper is relatively clear, with proper introductions to various techniques, facilitating the reader's understanding of the content.

### Weaknesses
1. The paper employs a variety of techniques and methods, including prompt learning, LoRA, gating mechanisms, and loss design, with five points listed in the INTRODUCTION under contributions and five in the METHOD section. This can seem a bit cluttered and redundant; a more concise summary and consolidation of related content would be beneficial.

2. Section 3.2 is titled "PROMPT LEARNING WITH LOW RANK ADAPTATION IN TRANSFORMERS," yet the explanation seems to treat prompts and LoRA separately, although the appendix provides a detailed explanation of their combined effect. As this is a crucial part of the paper, more clarity and detail in the main body of the text would be necessary.

3. The title of Section 3.3 is "HIERARCHICAL INTERACTION AND EXPANDED SUBSPACES," but the content first introduces expanded subspaces and then hierarchical interactions. The order of introduction and the content should correspond to the title.

4. The paper slightly lacks in-depth analysis of the synergistic effects between LoRA and prompt learning. Although an ablation study is conducted, showing experimental results under different conditions, a deeper analysis of how these components interact and contribute to performance improvements is needed, especially considering this is the core and key of the paper.

### Questions
1. In the hierarchical interaction section, both prompt tokens and LoRA layers establish connections between the current layer and the previous one to prevent information loss across layers. However, different weight allocation methods are used: α and 1-α for prompt tokens, and β and γ for LoRA layers. It would be beneficial to explain the rationale and necessity for using different methods when their purposes are aligned.
2. The paper sets a considerable number of hyperparameters, including learning rates, weight factors, deep prompt tokens, etc., and uses a fixed rank r and quantity m for LoRSS configurations across three different tasks. The paper does not seem to discuss the rationale behind these settings or how different ranks and quantities might impact the results. An explanation for these fixed values would be necessary.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
DPD-LoRA uses task-specific prompts to dynamically influence the low-rank updates of model parameters, enhancing the model's adaptability across diverse tasks and mitigating forgetting issues. By decomposing the standard low-rank adaptation into multiple low-rank sub-matrices, the method retains flexibility without adding additional parameters, thus improving the model’s learning capacity. An adaptive loss function is introduced to ensure alignment between the adapted distribution and the pre-trained model, thereby enhancing learning effectiveness and stability. A self-regulating mechanism is used to further improve model stability, along with a soft-gating mechanism to determine when to activate adaptation modules, ensuring improved performance on new categories.

### Strengths
(1) The proposed methods are relatively comprehensive, using several points to improve existing problems.

(2) The writing is clear and easy to understand.

### Weaknesses
(1): In line 041, “LLaVa” should be revised to “LLaVA” for consistent terminology throughout the document, avoiding unnecessary visual inconsistency.

(2): The related work section lacks references to significant LoRA extensions, such as DoRA, SVFT, PISSA, and LoRA-XS. It is recommended to include these studies and discuss how the proposed method compares to or builds upon these prior approaches. Specifically, it would be helpful to highlight the innovations of this work and the advantages it has over these extensions.

(3): The method incorporates a distillation-like Self-Constrain Loss, but there is no evaluation of training time, GPU resource consumption, or other efficiency-related metrics. Providing specific efficiency metrics, such as training time per epoch, peak GPU memory usage, and FLOPs, would substantiate the claims of being resource-efficient. Including a comparison of these metrics to baseline methods would further support the efficiency claims.

(4): The ablation study section only presents the individual performance of each component without evaluating the performance of their combinations. Adding experiments that evaluate different component combinations (e.g., two, three, and all four components) would provide a more comprehensive view of the model's performance. Including a table or figure showing these combinations or using an approach like forward selection to systematically evaluate the synergies between components would be very informative.

(5): The comparative experiments do not include related LoRA methods, such as DoRA and VeRA. Including comparisons with these methods would more clearly demonstrate the advantages of the proposed approach. It is suggested to add a specific experiment or table comparing the proposed method to DoRA, VeRA, and other relevant LoRA variants on key metrics or datasets to provide a clearer demonstration of its benefits.

### Questions
(1): The phrase “without any additional models prior” in lines 110-111 is somewhat ambiguous. Typically, Parameter-Efficient Fine-Tuning (PEFT) builds on pre-trained models, so it is recommended to clarify whether this refers to the absence of model priors or additional model parameters.

(2): The abbreviation “PEFT” is used for both Prompt-based Efficient Fine-Tuning and Parameter-Efficient Fine-Tuning, which may lead to confusion. It is advisable to select distinct abbreviations to improve clarity.

(3): The term “PLoRA” in line 378 is confusing, as its specific reference is unclear. Further definition or clarification of this acronym is recommended for improved reader comprehension.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper proposes the DPD-LoRA algorithm, which integrates prompt learning to guide the LoRA learning distribution. By incorporating modules such as Hierarchical Interaction, the Prompt-Conditioned Gating Mechanism (PCGM), and the Self-Regularized Lower-Rank Subspace (LoRSS), the proposed DPD-LoRA achieves strong performance across 11 benchmark datasets.

### Strengths
- The paper is well-structured and relatively easy to understand.
- Detailed experiments provide convincing evidence of the effectiveness of the proposed algorithm.

### Weaknesses
- Overall, the proposed algorithm involves numerous modules. I strongly suggest the authors consider identifying and focusing on the core components of their method.
- It is unclear how Eqn (4) is optimized. Are $s_i$ and $A_iB_i$ learned simultaneously? How many sub-LoRAs $m$ are used, and why is it imperative to decompose a single LoRA into multiple sub-LoRAs essential? Do the learnable $S_i$ and $G(P)$ share any functional overlap?
- Why doesn't the weighting form in Eqn (6) match that in Eqn (5) (e.g., setting $\gamma=1-\beta$) ? This discrepancy should be clarified.
- In Eqn (8), why does the orthogonal regularization prevent overfitting and encourage diversity in the learned LoRA? If this assertion is based on findings from other studies, supporting citations would strengthen the claim.
- I would like to see a memory cost comparison between the DPD-LoRA and SOTA methods. DPD-LoRA requires storing $m$ LoRAs per layer (Eqn (4)) and also duplicates each encoder in both branches while retaining unprompted inputs, which appears to impose a substantial memory cost.

### Questions
See weakness 2-5

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper presents a dynamic prompt-guided LoRA approach that integrates several key modules: Hierarchical Interaction, a Prompt-Conditioned Gating Mechanism (PCGM), and a Self-Regularized Lower-Rank Subspace. The proposed method is evaluated on 11 benchmark datasets, demonstrating its effectiveness.

### Strengths
+ The integration of prompts with LoRA represents an innovative exploration in this domain.
+ The authors conducted extensive experiments to substantiate the performance improvements of the proposed algorithm.

### Weaknesses
+ The motivation for using prompts to guide LoRA learning is not entirely intuitive. The authors should clarify why applying a weight to each $A_i B_i$​ in the LoRA layer solely through gating prompt tokens is expected to be effective.

+ The explanation of the Gating function requires clarification. Does $G(P)$ apply a weight before each $iA_i B_i$? How does this differ from directly learning $S_i$, and could it potentially overlap in function? Additionally, it is unclear how $G(P)$ interacts with the Hierarchical Interaction—does it apply weighting to $A_i B_i$ at layer $l−1$ as well?

+ Given the complexity of the proposed method and its multiple components, the current ablation study feels insufficient. For example, what is the rationale for decomposing a single LoRA into multiple sub-LoRAs? How are hyperparameters $\alpha$, $\beta$, $\gamma$, $\lambda_1$, $\lambda_2$​, and $\lambda_3$ set, and what is their impact on the final performance?

+ How does the addition of orthogonal regularization prevent overfitting? More details on this would clarify the choice and its benefits.

### Questions
Please refer to the weakness section.

### Soundness
2

### Presentation
3

### Contribution
2
