# Redirection for Erasing Memory (REM): Towards a universal unlearning method for corrupted data

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 4

## Abstract
Machine unlearning is studied for a multitude of tasks, but specialization of unlearning methods to particular tasks has made their systematic comparison challenging. To address this issue, we propose a conceptual space to characterize diverse corrupted data unlearning tasks in vision classifiers. This space is described by two dimensions, the discovery rate (the fraction of the corrupted data that are known at unlearning time) and the statistical regularity of the corrupted data (from random exemplars to shared concepts). Methods proposed previously have been targeted at portions of this space and, as we show, fail predictably outside these regions. We propose Redirection for Erasing Memory (REM), whose key feature is that corrupted data are redirected to dedicated neurons introduced at unlearning time and then discarded or deactivated to suppress the influence of corrupted data. REM performs strongly across the space of tasks, in contrast to prior SOTA methods that fail outside the regions for which they were designed.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper addresses the challenge posed by the specialization of machine unlearning (UL) methods to particular tasks, which has complicated their systematic comparison and resulted in unpredictable failures in practice. To characterize diverse corrupted data unlearning tasks in vision classifiers, the authors propose a conceptual taxonomy described by two key dimensions: the **discovery rate** (the fraction of corrupted data known at the time of unlearning) and the **statistical regularity** of the corrupted data. Previous State-of-the-Art (SOTA) methods were typically designed for specific regions within this 2D space and are shown to fail predictably and catastrophically outside of these targeted areas. To overcome this limitation and provide a universal solution, the authors introduce **Redirection for Erasing Memory (REM)**. The core feature of REM involves redirecting corrupted data to dedicated neurons that are newly introduced to the model during the unlearning process. Subsequently, these dedicated neurons are discarded or deactivated to suppress the influence of the corrupted data. Experimental results demonstrate that REM is the first unlearning algorithm that performs strongly across the entire spectrum of tasks within this defined 2D space, setting a new SOTA and proving to be a safe choice for practical applications.

### Strengths
- Proposes a clear 2D taxonomy (discovery rate × statistical regularity) that explains why prior unlearning methods succeed only in narrow slices and fail predictably elsewhere. 

- Introduces REM, a simple yet novel “redirection” mechanism that routes corrupted data into added neurons and then drops them—yielding strong performance across the entire task space. 

- Backs claims with broad, rigorous benchmarks (CIFAR-10/SVHN; ResNet-9/ViT), showing SOTA scores and practical efficiency.

### Weaknesses
- The stated motivation is that existing unlearning methods sometimes fail unpredictably an that they lack a principled understanding of the causes.Yet the paper offers no deep causal analysis; it primarily introduces a taxonomy (but valuable).

- Step 2 states that the original model is updated using $L_{step2} (L_{remove})$. However, in Step 3 the model is updated using $L_{step3}$, where a minus sign is placed in front of $L_{remove}$. Even if Step 2 is indeed the unlearning phase, the current wording is ambiguous.

- The problem setting is well-motivated and realistic. That said, the assumption of full access to the entire training set is difficult to justify in practice. Although the authors defend this design choice as a means to preserve model utility, it would be better that exploring alternatives that operate with only a subset of the training data.

- The loss function notation seems inconsistent, $L$ and $\mathcal{L}$ are used confusingly.

### Questions
- Most prior unlearning research treats a fully retrained model as the gold-standard, by the definition of unlearning. However, this paper considers scenarios in which retraining is not a valid gold-standard. Under that premise, can this work still be considered, in a fundamental sense, a contribution to machine unlearning?

- I'm wondering whether the Utility$\times$Healed metric is essential. Would separate reporting convey the trade-off more transparently in ViT case?

- For LLM unlearnings, what are practical ways to capture or quantify "statistical regularity"?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper focuses on a particular machine unlearning setting, where the data is corrupted and the user only partially knows the corrupted data. In other words, some corrupted data still remain in the retain set. The authors propose a method to handle this setting, which can potentially handle cases with a high fraction of unknown corrupted data, as well as corruptions with high regularity. The method is based on Example-Tied Dropout (ETD), which introduces new parameters, and introduce random dropout/masking that is fixed for each example, such that the example-specific knowledge is encoded in these parameters (which can then be removed). The proposed method is largely similar to ETD, although there are certain differences. Results are reported on CIFAR10 with ResNet-9 and ViT with SVHN.

### Strengths
This paper has clearly identified a gap in the literature of machine unlearning, where the corrupted data discovery rate and regularity can pose a challenging scenario.

The proposed method shows good improvements under this corrupted data setting (Table 1).

### Weaknesses
The exact differences from ETD, and why these differences matter, is not very well explained. It seems to me that there are several differences, but I am not able to grasp well these implications, and why these differences are substantial or important.

The experiment results are compared against methods that are rather outdated. SCRUB and ETD seem to be the latest works in Table 1, but they are both works from 2023. I strongly suggest that the authors compare their works with more recent publications and methods. This would make their claims more convincing and substantial.

More specific questions are placed in the Questions section.

### Questions
The details of the NPO optimization method is unclear, and how it is applied to this classification is also not clear to me. Also, since the NPO optimization is to remove the impact of the forget data from the general parameters, can this be replaced by a more traditional unlearning method?

I am not familiar with the ETD method. How is it that so many parameters can be dropped out at test time, yet the performance is not severely affected? Is there some intuition for this?

It is unclear how the proposed method can handle cases of corrupted data with high regularity. Is it something to do with the masking strategy?

Could the authors clarify how they measure regularity exactly? Also, how do they vary the regularity in the data?

Did the authors experiment with different ways of adding parameters onto the model? 
Also, how many parameters were added to the model (which are then removed)?


Overall, I currently recommend a borderline accept. However, I am not familiar with several key concepts in this paper as well as ETD, which is strongly related to the proposed method. I will revise my score based on the author’s responses.

### Soundness
3

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
3

### Summary
This paper introduces a method for unlearning of mislabelled data in the context of classification tasks. Authors first provide a problem definition and a analyse pros and cons of pre-existing works under a taxonomy covering two axes: levels of discovery of corrupted data and corruption patterns. Following this analysis, they propose a two-step unlearning method that borrows ideas from anti-memorisation technology to redirect corrupted information to a new set of dedicated weights. The approach shows more robustness across different scenarios than other methods.

### Strengths
The introduced taxonomy, analysis different forms of unlearning tasks in the context of classification problems is a nice framing of the problem. The analysis of a set of unlearning methods, including their strengths and reasons why they fail in certain scenarios was well written and helpful to a non expert in unlearning. Authors provide sensible and well justified arguments. 

The proposed method heavily relies on prior work, but makes intelligent use of ideas developed in those work, which are designed for other problems, and adapts it to the problem at hand. The method is further designed in a way that builds on the analysis provided in the taxonomy section, making the paper more coherent as a whole.

### Weaknesses
-	One weakness refers to the inconsistent presentation quality of the paper, certain sections (such as section 4) are much better presented than others (such as section 5). The abstract and introduction in particular try to be too general, making it more difficult to understand the purpose and setting of the paper until section 2. Similarly, the paper mentioning a “universal unlearning method” yields expectations that the proposed work is not restricted to the classification problem.  Certain terms are discussed before being introduced, such as healing in the related works section. 
Figures could also be improved, particularly figure 1 which requires accessing the appendix to be readable. In general, there are too many references to content in the appendix that is needed to read the main paper properly. Please refrain from including experiments in the main paper referring to results reported only in the supplementary materials. The main paper should be self-contained. Ablations are an important part of evaluating new methodology and results should be in the main paper.
-	Certain aspects of the methodology are confusing notably how the second fine-tuning stage prevents re-introducing corrupted information in the model. The claim that “undiscovered corrupted examples may be partially caught due to their masks partially overlapping” seems to be more wishful thinking than a rigorous scientific hypothesis. 
-	In general, the methodological novelty is somewhat limited, being closer to an orchestration of pre-existing work.

### Questions
-	In general, I think this is an interesting contribution to the field of unlearning. The most interesting part of this work is the unlearning taxonomy and analysis of pre-existing work. I would recommend revising presentation in intro/abstract to facilitate reader understanding. 
-	Why is NPO not discussed in section 4? It is a crucial part of the methodology in the following section. It would be great to have an explanation of how this approach works conceptually, as it is currently not discussed at all in the paper. 
-	Could authors clarify what they mean when mentioning 50% and 100% capacity models? Is it just smaller models?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies machine unlearning for corrupted data in vision classifiers. Existing unlearning methods tend to be task-specific and fail when the corruption pattern changes. To diagnose these failures, the authors propose a 2-D taxonomy for unlearning tasks along two axes: Discovery Rate and Statistical Regularity. Through extensive experiments, the authors show that all prior unlearning algorithms succeed only in narrow regions of this 2D space, and fail catastrophically outside their specialized slices.To address this, the paper introduces REM (Redirection for Erasing Memory), redirecting all corrupted data into newly added neurons during unlearning, then drops these neurons so that all corruption effects are removed from the final model.

### Strengths
(1)The authors introduce 2D taxonomy to identify statistical regularity as a missing dimension in unlearning research.

(2)The authors proposed REM to redirect corrupted data into dedicated parameters and drop them, inspired by the existing method ETD.

(3)REM is the only method that avoids collapse across both dimensions, compared with baseline methods.

### Weaknesses
(1)EM doubles the model size and is not viable for real-world models. From the perspective of efficiency, the core design of REM expands the model by adding a full set of new parameters. This may be feasible for small CIFAR-scale networks, but it is not realistic for large scale models. Even the VIT structure meets the OOM problem. The paper does not discuss scalability, memory cost, or efficiency, which makes REM difficult to use in real-world unlearning scenarios.

(2)While the empirical results indicate that REM achieves strong performance across discovery levels and regularities, the paper does not provide any network-level evidence confirming that the proposed mechanism, i.e., redirecting corrupted information into θ₀₂ while keeping θ₀₁ clean, actually occurs inside the model.

(3)Incomplete reporting of ViT metrics. In Table 1, the authors report full results (Healed, Utility, and Utility×Healed) for ResNet-9, but only report the aggregated metric (Utility×Healed) for ViT. The omission of the individual Healed and Utility values for ViT makes the evaluation incomplete and prevents readers from assessing the unlearning–utility trade-off on a second architecture.

(4)Questionable OOM explanation for Potion–ViT experiments. The paper states that the Potion–ViT experiment resulted in Out-of-Memory on an A100 40GB GPU. This justification is not entirely convincing: Potion is not significantly more memory-intensive than several other baselines (e.g., SCRUB, BadT, NPO), all of which were successfully evaluated on ViT under the same setup. Moreover, techniques such as gradient checkpointing, reduced batch size, or accumulation are typically sufficient to avoid OOM on CIFAR-10–scale ViT models.

### Questions
(1)The main conceptual contribution of REM is the claim that corrupted examples can be explicitly routed into a newly added parameter partition. However, the paper provides no network-level analysis verifying that this mechanism actually occurs.

(2)Masking scheme lacks theoretical guarantees and may cause pathway overlap. The redirection mechanism relies on a mask assignment: all corrupted samples share one mask and each clean sample receives a random mask. However, the paper does not justify why this ensures correct sample separation.

### Soundness
3

### Presentation
3

### Contribution
2
