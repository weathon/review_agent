# Reconciling In-Context and In-Weight Learning: A Dual-Space Modeling Perspective

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 4, 6

## Abstract
In-context learning (ICL) is a valuable capability exhibited by Transformers pretrained on diverse sequence tasks. However, prior studies have observed that ICL often exhibits a conflict with the model’s inherent in-weight learning (IWL) capability. In this work, we aim to reconcile ICL and IWL by disentangling the model’s encoding spaces for context and input samples. To do so, we first propose a dual-space modeling framework, explicitly modeling a task representation space via the dual space of the sample representation space. Such a dual-space structure can be derived from the linear representation hypothesis and, as we theoretically prove, is conducive to ICL by representation learning. Furthermore, we show that the standard Transformer architecture with softmax self-attention is inherently limited in realizing this structure. Building on this insight, we introduce CoQE, a Transformer architecture with separate context-query encoding, to realize the disentanglement between context and sample representations. Through experiments on both regression and classification tasks, we demonstrate that CoQE not only achieves lower ICL error compared to the standard Transformers, but also successfully reconciles ICL and IWL under diverse data distributions.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper aims to reconcile the discrepancy between In-Context Learning (ICL) and In-Weight Learning (IWL) by disentangling the model’s encoding spaces for context and input samples. To achieve this, the authors propose a dual-space modeling framework that explicitly constructs a task representation space as the dual of the sample representation space. They demonstrate that such a structure facilitates ICL through improved representation learning. Building upon this insight, the authors introduce CoQE, a Transformer-based architecture with separate context and query encoders, effectively achieving the desired disentanglement between context and sample representations.

### Strengths
1. The paper addresses an important and meaningful research question—the conflict between ICL and IWL.
2. The proposed CoQE architecture introduces a novel design. Beyond empirical validation, the authors also provide a detailed theoretical analysis to support their claims.

### Weaknesses
The theoretical presentation lacks clarity and organization:

1. The purpose and implication of **Theorem 3.7** are unclear. It is not evident whether the dual-space formulation achieves a tighter bound than the one presented in this theorem.
2. Although the paper claims to reconcile the conflict between ICL and IWL, **Theorem 3.10** only establishes their entanglement, rather than addressing the nature of their conflict. In fact, such entanglement might allow mutual reinforcement instead of opposition.

Additionally, the computational efficiency of the proposed method is not discussed. The paper omits key details such as the parameter scale and the computational cost associated with the new operations.

Finally, the empirical evaluation only compares CoQE against standard Transformers. It remains uncertain whether the proposed structure outperforms other advanced Transformer variants.

### Questions
How to assure that the task representation actually learn the representation of tasks instead of others?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper addresses the important challenge of designing autoregressive models that support both in-context learning (ICL) and in-weight learning (IWL). The paper proposes a dual-space modeling framework to allow both ICL and IWL capabilities, providing a theoretical support using linear representation hypothesis. They explicitly model task representation space via the dual space of the sample representation space. To implement this, the paper proposes a new architecture that encodes context and query separately to resolve the representation entanglement, which is identified as the main cause of conflict. The proposed architecture attempts to implement the dual-space theory two spaces interact through inner products. The proposed method is evaluated on a regression task using synthetic data and classification task with Omniglot data. Results show improved results on both ICL and IWL performance.

### Strengths
1. The conflict between ICL and IWL in autoregressive transformer models is a relevant topic of research. The idea that samples and tasks can operate in separate but dual representation spaces and the connection with Riesz representation theorem is conceptually novel. The dual-space theoretical framework is well defined covering required formal definitions and assumptions.

2. Results show clear improvements with improved ICL performance without hurting IWL. To support the proposed theory, they further demonstrate empirically that separating context and query stabilizes ICL in both in-distribution and OOD cases. This is shown for both regression and classification tasks.

3. Paper is well-structured and easy to follow.

### Weaknesses
1. Authors proposed CoQE architecture to structurally  separate the spaces to resolve the ICL and IWL conflict. However, the dependence on gaussian noise for the classification is very strong to avoid collapse, where performance is close to chance level without the noise injection. The dependence of gaussian noise regularizer parallels with the usage of l2-regularization in [Chan et al. 2022] to balance ICL and IWL, implying that the CoQE benefits step more from regularization than from its theoretical design. This weakens the theoretical claim and advantage of CoQE. 

2. Theorem 3.10 equates non-linearity of softmax with the non-existence of factorized dual-space. Non-linearity does not rule out the existence of an equivalent linear form under a suitable feature mapping. The reasoning is limited to a simple setup such as a single softmax-attention layer and doesn’t include a mathematical proof. Empirical studies in the literature such as Han et al. 2025 [a], show that attention-based transformers can exhibit linearly separable task vectors for distinct in-context tasks. This weakens the theorem. 
- [a] Emergence and Effectiveness of Task Vectors in In-Context Learning : An Encoder Decoder Perspective, Han et al. 2025.

3. The assumption of a shared linear sample representation across tasks is conceptually good but too strong for real-world multi-task settings. Many real-world tasks in natural language and vision exhibit nonlinear mapping between input and output spaces. This limits the practical applicability of the proposed framework.

Please feel free to clarify if I misunderstood anything mentioned above.

### Questions
1. I suggest that the authors demonstrate the advantage of the CoQE architecture with more ICL-IWL tasks which are not dependent on such strong regularizations. This would clarify whether the observed improvements come from the architecture or the regularization.

2. Abstract claims that transformer with softmax self-attention is limiting for the dual-space structure. However, this limitation is not demonstrated clearly in the main text. Authors should clarify what specific theoretical or empirical evidence supports this claim.

3. The paper would benefit from a better connection to previous works in the literature. A comparative discussion with recent studies mentioned below should be included to better position this work. (a) Toward Understanding In-context vs. In-weight Learning (Chan et al. 2025), (b) What Matters for In-Context Learning: A Balancing Act of Look-up and In-Weight Learning (Bratulic et al. 2025), (c) Dual Process Learning: Controlling Use of In-Context vs. In-Weights Strategies with Weight Forgetting (Anand et al. 2025). 

4. The paper lacks discussion about certain points such as implications of oversimplified assumptions like shared linear sample space across tasks and limitations of currently used empirical tasks.

Minor remark:
- Figure and table captions should be self-contained clearly conveying the necessary information to understand them.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper tackles the observed conflict between in-context learning and in-weight learning . The authors propose that this tension arises because standard Transformers entangle the encoding of context and samples. To address this, they introduce a dual-space modeling framework, where the task representation space is the dual space of the sample representation space. Building on this, they propose CoQE, an architecture that explicitly separates context and query encoding: Predictions are computed via an inner product between task and sample representations.

### Strengths
1.	The paper presents a novel theoretical framework to explain the underlying conflict between in-context learning (ICL) and in-weight learning (IWL).
2.	Building on this theory, the authors propose a new architectural design (CoQE) that improves performance and reconciles the trade-off between ICL and IWL.
3.	The figures and visualizations are clear and well-designed.

### Weaknesses
1.	The CoQE architecture resembles a dual-tower model, which may compromise the model’s capability for open-ended generation. This limitation reduces its applicability to a broader range of tasks, and additional training would likely be required for different task types.
2.	The evaluation of CoQE is limited to relatively simple regression and classification tasks, which may not be sufficient to demonstrate its effectiveness on more complex or real-world tasks.
3.	The proposed approach relies heavily on a strong linear representation hypothesis, assuming that the task and sample representation spaces form dual linear spaces. The paper lacks sufficient empirical evidence or justification to support this assumption.

### Questions
1.	In Section 3.1, the authors state that “each basis often corresponds to an independent attribute or concept.” However, bases in a linear space can be chosen arbitrarily. It is unclear why each basis necessarily corresponds to a specific attribute. Could the authors provide concrete examples to clarify this point?
2.	Could the authors include experiments evaluating CoQE on more complex tasks, such as standard NLP benchmarks, to demonstrate its scalability and general applicability?
3.	Could the authors further explain why a task function can be regarded as a linear function of the sample representation, given that many real-world tasks—especially reasoning tasks requiring chain-of-thought (CoT)—exhibit strong nonlinearity?

### Soundness
2

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
3

### Summary
This paper studies the interplay between In-Context Learning (ICL) and In-Weight Learning (IWL) in Transformers.  
It proposes a dual-space modeling framework that represents these two processes in a task representation space and a sample representation space, linked through the Riesz representation theorem.  
The authors argue that the conflict between ICL and IWL originates from the entanglement between context and query encoding in standard self-attention.  
To address this issue, they introduce CoQE (Context–Query Encoding Transformer), which explicitly separates context and query encoding pathways.  
Experiments on regression and few-shot classification tasks show consistent improvements across both ICL and IWL metrics.

### Strengths
- The paper provides a mathematically grounded view of ICL and IWL, introducing a dual-space theoretical framework that distinguishes the task space from the weight (sample) space. This perspective is conceptually novel.  
- Theoretical reasoning directly motivates an architectural design (CoQE), forming a coherent pipeline from theory → architecture → experiment.

### Weaknesses
- The concept of In-Weight Learning (IWL) should be more explicitly defined within the paper, instead of relying mainly on external references.  
- The proposed CoQE structure appears to apply an additional embedding to the context input. From Equation (11), it is unclear how this achieves a true *separation* between context and query encodings, since both still pass through a shared encoder.  
- The mapping between ID/OOD performance and IWL/ICL capability is indirect. It is unclear why the authors did not include a no-context baseline (weight-only inference) to explicitly test IWL.

### Questions
1. Nonlinear Sample Spaces  
   The theoretical assumptions are quite strong and rely on the linearity of the sample space \( $M_F$ \).  
   How would the proposed dual-space framework extend to nonlinear or non-convex representation spaces?

2. Memory Interpretation  

   I am wondering whether it is appropriate to interpret the interaction between ICL and IWL as analogous to short-term and long-term memory mechanisms.
    Can CoQE be understood as a framework that aims to enhance both types of memory simultaneously, or rather as one that mitigates interference between them?

3. Regression Function Families  
   In Section 4.1, the authors write:  
   *“Specifically, we use the following four classes of functions F: linear functions, sparse linear functions, two-layer ReLU networks, and combination functions.”*  
   Are these referring to target functions used for data generation or to model architectures?  
   Figure 3(a) suggests they are target functions, but this should be stated explicitly.

4. Theory–Experiment Gap  
   The theoretical part mainly analyzes the entanglement between ICL and IWL and argues that CoQE supports both within the dual-space framework.  
   However, the experiments (especially regression and few-shot classification) primarily test ICL generalization, particularly under OOD settings.  
   - Are the experiments intended to verify ICL generalization rather than true ICL–IWL coexistence?  
   - If claiming “simultaneous improvement,” how is IWL concretely evaluated?  
   - Would adding a no-context control (i.e., weight-only inference) better demonstrate CoQE’s preservation of IWL capability?

### Soundness
2

### Presentation
2

### Contribution
2
