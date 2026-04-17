# Task Matrices: Linear Maps for Cross-Model Finetuning Transfer

- Decision: Reject
- Scores: 2, 4, 2, 2

## Abstract
Results in interpretability suggest that large vision and language models learn implicit linear encodings when models are biased by in-context prompting. However, the existence of similar linear representations in more general adaptation regimes has not yet been demonstrated. In this work, we develop the concept of a task matrix, a linear transformation from a base to finetuned embedding state. We demonstrate that for vision and text models and ten different datasets, a base model augmented with a task matrix achieves results surpassing linear probes, sometimes approaching finetuned levels. We show that linear encoding in transformer embedding spaces exists between pretrained and finetuned architectures, and can be readily exploited through task matrices. These matrices incur low computational costs, and are both data-efficient and generalizable in multiple domains. We make our implementation publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The work proposes learning a linear map between the representations of the base model and those of its finetunings, and show that this allows obtaining close-to-finetuning performance just by equipping the base with this extra adapter layer. Fitting the map requires task-specific data, and the manuscript show results both using 100% of the task data as well as a fraction (20%), showing a graceful degradation. The proposed approach is compared with linear probes trained with the same data, showing the learned base-to-finetune linear maps to obtain better performance, sometimes getting close to that of the finetuned model. Experiments are performed with RoBERTa and all-MiniLM-L12-v2 for text, and ViT, DeIT and DINOv3 for vision.

### Strengths
- Although very simple, the idea of fitting an adapting from the base model to its finetuning makes intuitive sense, and might actually be useful in several practical use cases. Indeed, having a whole finetuning compressed into a single vector is what made task vectors [1] particularly enticing, and in this case the dimension of the task "digest" would be much smaller than that of the whole network.

- The results, if confirmed, seem promising and interesting. To the best of my knowledge, this analysis is also novel. Differences between finetuned and base models have been mostly studied in the weight-space, with this work focusing instead on the representation space.

[1] Ilharco, Gabriel, et al. "Editing models with task arithmetic." The Eleventh International Conference on Learning Representations.

### Weaknesses
- The whole “interpretability suggests that vision and language models learn implicit linear encodings”, repeated several times throughout the manuscript as main motivation, is somewhat vague and misleading. What the cited prior work claims is that relation decoding is (sometimes) well approximated by a linear transformation, not that the overall function computed by a transformer block, or a whole transformer, can be approximated thus. In this perspective, approximating a whole finetuning by a linear transformation over some chosen layer seems arbitrary.

- The CLIP experiments employ a trainable classifier instead of doing open vocabulary classification with the frozen text encoder. This is different than what is usually done for open vocabulay classification on multi-modal architectures such as CLIP. This makes it hard to assess how much the finetuned head contributes to the final accuracy.
    - In this matter, I find the results in table 4 and 5 intuitive as the finetuned head expects different statistics than those presented by the activations obtained from the base, but this does not really rule out that the benefit stems from the finetuned head.
    - To actually rule this one out, one would have to use CLIP’s frozen text encoder as in most works. I tried myself implementing the proposed method using CLIP’s text encoder but I obtain results close to random-chance.

- No code is provided. Given the lacking of theoretical justification and the surprising nature of the results, I cannot reliably judge this paper without being able to reproduce its results. To this end, I performed a best-effort implementation of the method with the only difference being the usage of the frozen CLIP text encoder instead of an ad-hoc classifier as done in the paper. I could not reproduce the results; in fact, I could not get anywhere close to the reported scores.

- The merging experiment needs some baselines for comparison. With a quick comparison to recent literature, the method does not seem competitive to merging techniques: the single task matrix in figure 4 obtains ~80% normalized average accuracy, below modern merging techniques which surpass 90% [2, 3].

- Writing and figures seem a bit rushed. There is a dangling acknowledgment section stating “We thank ….”.

Overall, I feel like the idea is interesting and worth exploring in depth, but the uncertainty of the empirical evidence and the overall lack of motivation suggest the paper not to be ready for publication at its current stage. 

[2] Gargiulo, Antonio Andrea, et al. "Task singular vectors: Reducing task interference in model merging." *Proceedings of the Computer Vision and Pattern Recognition Conference*. 2025.

[3] Marczak, Daniel, et al. "No Task Left Behind: Isotropic Model Merging with Common and Task-Specific Subspaces." ICML 2025.

### Questions
- There is no clear insight on when the method will work and when not. Can you provide experimental evidence or theoretical discussion about when and why to expect it to work in practice? Studying the structure of the learned map may help here. e.g. is it low-rank? is it sparse?

-  Did you try aggregating the linear maps instead of fitting one linear map for all the tasks? 

- Why wasn't the CLIP text encoder used? I couldn't get it to work. Are there any reasons why this method should only work with linear classifier heads instead of frozen text encoders in an open vocabulary setting?

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents and defines the concept of **Task Matrix**, a linear transformation that maps an intermediate embedding state of a pretrained base model to the corresponding finetuned embedding state. The authors empirically show that such a linear encoding exists between pretrained and finetuned architectures, and that this relationship can be exploited using task matrices as an alternative to linear probing. Experiments on both vision and text tasks demonstrate that a base pretrained model augmented with a task matrix can achieve results that are comparable to, and in several cases surpass, linear probes, sometimes approaching finetuned accuracy. The proposed method is conceptually simple, providing a compact way to transfer finetuned behavior without retraining model parameters.

### Strengths
Method simplicity: the proposed task matrix construction method is straightforward and easy to implement. It consists essentially of a least-squares regression that learns a linear map between base and finetuned hidden representations.

Generalization across multiple datasets and models: the approach is shown to generalize well to both vision and text models, and even across multiple datasets simultaneously. The authors demonstrate with extensive experiments that a single task matrix can support multiple classification tasks with only marginal accuracy drops.

Data efficiency and robustness: Task matrices can be trained with very small amounts of data, sometimes below 1% of the training set, while maintaining strong performance with respect to linear probing perfomed on the same amounts of data. This suggests potential advantages in low-data or privacy-restricted scenarios.

### Weaknesses
No clear conceptual advantage over existing baselines: while the approach is elegant and achieves competitive performance, its advantages over standard techniques such as linear probing or low-rank adaptation remain unclear in the full-data setting.
The authors describe task matrices as “lightweight” and “low-cost,” but do not provide runtime, FLOPs, or parameter count comparisons against LoRA, adapters, or full finetuning baselines.

Potential equivalence to linear probing under specific conditions: if the optimal base embedding for learning the task matrix W corresponds to the final layer of the base model, the method effectively collapses to standard linear probing, since both rely on the same linear relationship between final-layer embeddings and the output space. Empirically, this appears to be the case for vision tasks, where the best-performing task matrices are consistently derived from the final layer.

Readability and notation issues: although the general concept is clear, the paper suffers from minor inconsistencies in notation and some missing definitions. For example, in Section 3.2 (lines 168–173), the set indices (S) are inconsistently indexed by (n) instead of (k). The overall exposition of the multitask classification setting (Section 5.4) lacks detail on how the base and final layer embeddings are sampled from the joint dataset.

Dependence on fine-tuned model embeddings: the method requires access to the embeddings of the target fine-tuned model in order to construct the task matrix. This dependency substantially limits the practicality and claimed efficiency of the approach, since obtaining such embeddings presupposes the existence of a fine-tuned model. In effect, the method relies on the same model it seeks to approximate or replace, which undermines its utility for scenarios where fine-tuning is computationally infeasible or where fine-tuned weights are unavailable.

Statistical reporting ambiguity: throughout the figures and tables, the notation “n=5, CI=95%” is used. I suppose this is intended to mean that results are averaged over five independent runs (n=5) and reported with a 95% confidence interval, but this is not explicitly stated in the paper.

### Questions
1) In multitask training, how are datasets balanced or sampled when constructing a shared task matrix?
2. Can task matrices trained on one task be transferred or composed for related downstream tasks?
3. How does the approach behave when the finetuned model diverges substantially from the base model (e.g., nonlinear adaptation or large domain shift)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces Task Matrices, a simple but intriguing method for transferring the effects of fine-tuning across models via a learned linear transformation between embedding spaces. Specifically, the authors posit that for a base model and its fine-tuned counterpart, there exists a linear mapping $W$ such that $Wx\approx y$, where $x$ and $y$ are hidden representations from the base and fine-tuned models respectively. They show empirically that this assumption holds across multiple architectures (RoBERTa, CLIP ViT, DeiT, DINOv3) and domains (text and vision). The task matrix is learned via regression and can then be used to approximate fine-tuned performance with minimal data and compute cost. Results show that task matrices outperform linear probes and sometimes approach fine-tuned accuracy.

### Strengths
**Conceptual novelty:** The central idea -- that fine-tuning can be approximated as a linear transformation in representation space -- is novel and elegant. It reframes fine-tuning as a geometric relation rather than an optimization process, opening potential directions for efficient adaptation and cross-model understanding.

**Empirical coverage:** The authors conduct an extensive empirical study spanning both vision and language domains, multiple model families, and diverse datasets.
    
**Ablations:** The paper includes ablation studies disentangling the role of the task matrix from classifier heads and fine-tuned weights. These experiments convincingly show that the performance gains come from the learned transformation itself rather than from trivial parameter reuse.

### Weaknesses
**Experimental discussion and clarity.** While the empirical coverage is broad, the presentation and discussion of results are weak. Tables are numerous and scattered across the main and supplementary sections, but the commentary does not synthesize the findings into a coherent take-away. For instance, the reader is never clearly told \textit{whether} and \textit{under what conditions} the task matrices work best. A concise summary of results (e.g., performance trends across depth or domains) is missing. Such discussion is critical for a paper that is primarily empirical in nature.

**Practical usefulness:** It remains unclear in what concrete scenarios the method is beneficial. The requirement of having a fine-tuned model to construct the task matrix largely defeats its potential use as a parameter-efficient fine-tuning (PEFT) alternative. It does not reduce the computational cost of fine-tuning, nor does it serve as an effective distillation mechanism — in vision settings, it even increases model size by often appending an additional linear layer. The discussion around multi-task classification could provide a compelling application, but it is presented vaguely and without quantitative results, leaving its value difficult to assess.  

**Layer selection ambiguity.** The paper reports results as a function of layer ID, but the criteria for selecting the "best layer" are not properly explained. If the choice is based on validation accuracy, the best-performing layer would trivially tend to be the final one. Moreover, since models of different depths are compared, layer indices are not directly interpretable -- expressing layer position as a fraction of total depth would have been clearer and more comparable.
    
**Lack of theoretical analysis**: The central linearity assumption, while empirically explored, lacks theoretical grounding. The paper would benefit from a deeper analysis of when and why such linear mappings hold — for example, as a function of layer depth, task similarity, or representation alignment. Without this, the method risks being a descriptive observation rather than a generalizable principle.
    
**Limited comparisons:** The paper only compares the proposed approach against linear probing, which is a relatively weak baseline, moreover it's not explained why this should be an interesting comparison. To substantiate the claimed advantages, it would be important to include comparisons with stronger and more relevant alternatives — such as modern knowledge distillation or parameter-efficient fine-tuning methods. Without these, it is difficult to assess the practical competitiveness of the proposed method.
    
**Decontextualized experiments.** Some experimental setups (e.g., data-scarce, multi-task) are introduced without sufficient justification or connection to the paper’s stated goals. After each, the reader is left wondering: "why is this experiment important?" The results are not tied back to an overarching research question or potential application. This gives the impression of a collection of loosely related tests rather than a coherent investigation.
    
**Writing and presentation.** The exposition requires substantial improvement, especially from Section 5 onward, where the paper becomes quite confusing (for instance, lines 404–408 are unclear). The main claim is buried in the text rather than stated explicitly in the introduction or conclusions. Notation is inconsistent — for example, lines 154–160 introduce symbols in a confusing way, and this notation is never reused later. Equations are unnumbered, and figures and tables are poorly designed (e.g., Figure 2 is almost unreadable in grayscale). Table 4 appears incomplete, as it contains only a single row, and the large number of tables without consistent headers makes it difficult to interpret the results. Overall, the paper would benefit from a clearer structure, stronger motivation, and more precise phrasing.
\end{itemize}

**Minor issues, not influencing the final assessment:**

- Line 40: "offer lightweight and effective approximations", should be "offers" (subject "model states" is singular).
- Line 95–96: "while in text, while in text" repetition.
- Line 122: "have subsequently provided", should be "has subsequently provided" (subject "body" is singular).
- Line 154-155: "be $H^L$ be $H_{ft}$" repetition
- Line 174: Equation brackets misaligned: "V ($\hat{h}^L$))": remove extra parenthesis: "V ($\hat{h}^L$)".
- Line 240-241 "state-of-art" should be "state-of-the-art".
- Line 248 "\textit{eight} diverse NLP benchmarks", only seven text datasets are presented
- Line 260-261 "casual influence" should be "causal influence"

### Questions
No questions.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces the concept of a "task matrix," a linear transformation learned via least-squares regression that maps the intermediate representations of a pretrained (base) model to the final-layer representations of a fine-tuned model. The goal is to create a lightweight and data-efficient method for domain adaptation that avoids the costs of full fine-tuning. The authors demonstrate that for a variety of vision and text datasets, augmenting a base model with a task matrix can outperform linear probing and, in some cases, approach the performance of a fully fine-tuned model. The paper also explores the properties of task matrices, showing they are robust to data scarcity and can be generalized to multitask settings.

### Strengths
1. The experiments, conducted across multiple vision (CLIP ViT-B/32) and text (RoBERTa-large) models on ten different datasets, are comprehensive. The results consistently show that the task matrix approach surpasses the performance of linear probing, a standard baseline.
2. A key strength of the proposed method is its performance in data-scarce environments. The experiments show that task matrices maintain a significant performance advantage over linear probes when trained on only 20% of the data.

### Weaknesses
1.  **Insufficient Positioning Against Prior Work:** The core idea of using linear maps between embedding spaces has deep roots in prior work on concept learning (e.g., Paccanaro & Hinton, 2001; Mikolov et al., 2013). While the paper applies this to the new context of finetuning transfer, its novelty relative to this established literature is not clearly articulated. Furthermore, the connection to related methods like task arithmetic [Ilharco et al., 2022], which also manipulates model states, is only briefly mentioned and warrants a more thorough discussion to better frame the paper's contribution.

2.  **Mismatch Between Optimization and Inference Objectives:** There is a potential disconnect between the method's training objective and its ultimate goal. The task matrix $W*$ is optimized by minimizing the least-squares error $\|Wh^i - h^f\|^2$, which is the Euclidean distance between the transformed base embedding and the finetuned embedding. However, the inference goal is to match the final classification prediction, i.e., $arg max V(Wh^i) = arg max V(h^f)$. The paper does not provide a theoretical or empirical justification for why minimizing L2 distance in the embedding space is a suitable or optimal proxy for maximizing classification agreement.

3.  **Lack of Guidance on Hyperparameter Selection:** The choice of which intermediate layer from the base model to use is a critical hyperparameter that significantly impacts performance. Figure 2 shows that accuracy can vary dramatically depending on the selected layer (e.g., RoBERTa-SNLI performance fluctuates significantly across layers). The paper offers no principled method, heuristic, or analysis for selecting the optimal layer a priori, which would require an exhaustive and computationally expensive search in practice.

4.  **Limited Comparison to Other PEFT Methods:** The paper's primary baseline is linear probing. While this is a reasonable starting point, the introduction positions the work as an alternative to other prominent parameter-efficient fine-tuning (PEFT) methods like LoRA, BitFit, and Prefix-Tuning. However, no direct comparisons are made. Without this context, it is difficult to assess where task matrices stand in the broader PEFT landscape regarding the trade-offs between adaptation accuracy, memory/storage costs, and computational overhead.

5.  **Unclear and Potentially Incorrect Notation:** Some of the mathematical notation is confusing. For example, the final representation $h^L$ is described as belonging to $H^L \in R^N$, where $N$ is the number of classes. This appears to incorrectly use the number of classes for the hidden dimension, which is defined as $d$ elsewhere. Similarly, the definition of the decoder head as $V ∈ R^{d \times N}$ seems to be transposed from its conventional use in $z = Vh$, where $h$ is $d \times 1$. These inconsistencies make the technical approach harder to follow.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
