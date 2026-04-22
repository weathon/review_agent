# Conflict-Aware Representation Editing for Robust Retrieval-Augmented Generation

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Large Language Models (LLMs) may not always provide accurate responses to user queries, owing to the staleness of training data and the presence of noise. To address this, Retrieval-Augmented Generation (RAG) has been widely adopted, enabling LLMs to ground their responses in external knowledge sources. Nonetheless, recent studies show that conflicts between the retrieved external knowledge and the model’s parametric knowledge can lead to hallucinatory outputs, and this problem is exacerbated when the retrieved documents contain noise. In this work, we propose Conflict-Aware Representation Editing (CARE), a method designed to generate robust responses even when the retrieval includes documents with low relevance to the query. CARE aims to produce conflict-resilient responses by editing the internal representations of the model. Assuming that LLMs encode distinguishable internal patterns indicative of knowledge conflicts, we introduce an autoencoder into the model’s internal layers to identify such regions. We then modulate neuron activations accordingly, steering the model to generate responses unaffected by knowledge conflicts. We evaluate CARE across six Question Answering (QA) benchmarks and four LLMs, demonstrating its superior performance over existing methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces **conflict-aware representation editing (CARE)**, a representation-editing-based method to address the challenge of knowledge conflict in retrieval-augmented generation (RAG).

CARE is based on autoencoders and resolve knowledge conflicts by (1) carefully curating training data consisting of positive answers and negative answers (including parametric and noise answers) (2) editing in the autoencoder latent space.
Data curation allows the autoencoder to distinguish between knowledge aligned/conflicted answers, while latent space editing shifts model representations towards the conflict-free direction.

Experiment results validate that CARE outperforms RAG baselines, conflict resolution methods and truthfulness representation editing methods on six standard and multi-hop QA tasks.
Further experiments including ablation study, hyperparameter sensitivity and visualization support that CARE is a robust enhancement to RAG.

### Strengths
1. Originality

   The paper is novel by using representation editing to solve the knowledge conflict challenges of RAG.
   This approach is superior to RAG-enhancement methods and conflict resolutions methods in terms of QA performance and efficiency, outperforms truthfulness representation editing methods (e.g. ITI) in the setting of RAG, and it is better suited for RAG tasks than knowledge editing methods for its ability to perform on-the-fly knowledge adjustment.
2. Quality

   The paper is technically sound.
   The method is clearly explained, experiments are well-designed and results are convincing.
3. Clarity

   Overall, this paper is well written by clearly communicating the motivation, key insight, and how it contributes to the field.
4. Significance

   First, this work contributes to the field of RAG.
   It shows that representation editing is both effective and efficient when compared with RAG methods like SURE and conflict resolution methods such as IRCAN. This work could lead to a paradigm shift towards more robust RAG.

   Second, this work is also meaningful to the field of representation editing.
   It shows concrete evidence that representation editing could be effective in long-context and knowledge-extensive QA tasks.

### Weaknesses
(Authors do **not** need to refer to points raised in this section since the main points are already mentioned in *"Questions" section*.)
1. W1: Reproducibility

   The authors have provided extensive details in the paper that could facilitate reproduction of main results. However, code implementations are not provided either as supplementary material or present in an anonymous link.
2. W2: Discussions on potential limitations

   The paper does not thoroughly discuss limitations of this work.
3. W3: Related work

   The paper does not take into account broader literature in mechanistic interpretability, particularly studies on sparse dictionary learning that train sparse autoencoders to reconstruct model representations (details in "Questions" section).

### Questions
**Major questions (that could affect rating)**
1. **Question 1**: Reproducibility

   Do authors have plans of open-sourcing their code? If so, it would be helpful to the research community; if not, please elaborate since it may adversely impact the paper's contribution to the field.
2. **Question 2**: Explanation for suboptimal performance on PopQA in Table 1

   Table 1 shows that although CARE generally outperforms most baseline methods, it consistently fails to yield optimal performance on PopQA. What might be the reason for this performance gap? Is it due to certain intrinsic characteristics of the PopQA task, or does it stem from the method CARE itself?
3. **Question 3**: Discussions on potential limitations

   The paper could expand on potential limitations, which would show the authors' in-depth insights in this field and would be meaningful in inspiring future work.
4. **Question 4**: Related work

   The paper does not take into account broader literature in mechanistic interpretability, particularly those on sparse dictionary learning (SDL)[1-3]. These works are related to this paper since autoencoders are trained to reconstruct model representations.
   [1-3] are different from the method of CARE since [1-3] use sparse autoencoders, use different auxiliary losses and have much larger latent space sizes.

   Nevertheless, incorporating SDL into the picture could help the paper set a broader context for potential readers, especially considering that this work touches on knowledge-related mechanisms in representation space (lines 82-85).
5. **Question 5**: Is CARE resolving knowledge conflicts or eliciting truthful answers?

   Truthfulness representation editing methods such as ITI and TruthX are introduced as baselines. Despite the narrative of this paper that CARE is designed to solve knowledge conflicts, it *might* be the case that CARE is eliciting truthful answers.

   Therefore this question is a conceptual and fundamental one: Is CARE directly improving truthfulness, or is it indirectly improving truthfulness by resolving knowledge conflicts?

   Could current evidence of Table 4 (Analysis of knowledge conflict resolution with external knowledge) help answer this question?
   Furthermore, could this question be answered by decomposing the negative sample distribution of Figure 4 into two distributions, one for parametric negative answers and another for noise negative answers?
6. **Question 6**: Effect of latent space dimension on performance

   The paper has shown ablation studies with respect to training loss functions and directional coefficients.
   Beyond these aspects, additional information regarding the latent space dimension could be useful in terms of the trade-off between computational efficiency and task performance.
   This concern is motivated by previous works on linear representation steering where 1D linear vectors are sufficient for good task performance[4-6].


**Minor questions and suggestions (that are not considered to affect rating)**
1. **Minor question 1**: GPU model specs

   It is mentioned multiple times in the paper that "A6000 40GB" GPUs are used. However, as far as I know the A6000 model only has the 48GB configuration. Therefore I am confused if this is simply a typo.
2. **Minor question 2**: MLP non-linear activation function

   What is the activation function used for encoder/decoder MLPs (ReLU/GeLU/...)?
   This information is useful since I am interested in *how much non-linearity* in introduced into the CARE autoencoder, which could be expressed as the following question: can CARE achieve the same level of performance with an entirely linear autoencoder?
3. **Minor question 3**: Inconsistency between text and figure.

   The paper states at lines 202-204 that the dimension of the autoencoder latent space is *smaller* than representation space.
   However, in Figure 1, the latent space is *larger* representation space, judging by the shapes of the encoder and decoder.
   Therefore authors might need to modify the figure to help with understanding.
4. **Suggestion 1**: Efficiency-related results and discussions

   Efficiency of RAG enhancements is a critical concern. The authors have done a good job by including efficiency analysis in Section G.3, showing that CARE introduces negligible latency with respect to the "With Retrieval" baseline.
   This result is important and deserves mentioning in the main body to highlight efficiency advantages.
5. **Suggestion 2**: Variance and statistical significance

   Experiment results could be enhanced with variance across seeds and statistical significance. These information could shed light on whether CARE is stable across training runs.


References:

[1] Towards monosemanticity: Decomposing language models with dictionary learning. (2023)  
[2] Sparse autoencoders find highly interpretable features in language models. (ICLR 2024)  
[3] Scaling and evaluating sparse autoencoders. (ICLR 2025)  
[4] Inference-Time Intervention: Eliciting Truthful Answers from a Language Model (NeurIPS 2023)  
[5] Steering llama 2 via contrastive activation addition. (ACL 2024)  
[6] Activation addition: Steering language models without optimization. (2023)

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes Conflict-Aware Representation Editing (CARE), a method to improve the robustness of Retrieval-Augmented Generation (RAG) systems under knowledge conflicts and noisy retrievals. The central idea is that large language models (LLMs) exhibit distinguishable internal activation patterns when external context contradicts parametric knowledge. CARE introduces an autoencoder trained on positive (correct) and negative (conflicted/noisy) QA pairs to identify such patterns and learns a latent “conflict-mitigation direction.” During inference, it adjusts neuron activations along this learned direction in specific layers to suppress conflict signals and enhance reliability. Experiments across six QA benchmarks (NQ, TriviaQA, PopQA, SQuAD, HotpotQA, 2Wiki) and four LLMs (LLaMA-2/3, Qwen-2.5) show consistent improvements over prior RAG and representation-editing baselines such as CAD, IRCAN, and TruthX.

### Strengths
See Summary.

### Weaknesses
1.	Conceptual validity of training supervision — The autoencoder is trained using externally labeled correct/incorrect answers rather than the model’s internal conflict cases. This design may not truly capture internal conflict representations, making it unclear why the learned δ direction should correspond to factual reliability.
2.	Ambiguous knowledge-source attribution — The method can identify conflict-related activation patterns but cannot determine which knowledge (parametric or retrieved) is correct. Without this distinction, editing representations might sometimes steer the model toward incorrect sources.
3.	Lack of interpretability and mechanism insight — The paper does not analyze what features the autoencoder encodes, or whether its latent dimensions correspond to known interpretable features, as done in sparse-autoencoder-based interpretability works such as Cunningham, Hoagy et al., “Sparse Autoencoders Find Highly Interpretable Features in Language Models” (arXiv:2309.08600, 2023) and Zhao, Yicheng et al., “Steering Knowledge Selection Behaviours in LLMs via SAE-based Representation Engineering” (arXiv:2410.15999, 2024).
4.	Limited novelty — The approach closely parallels existing SAE-based representation editing and truthfulness-control studies (e.g., the two works above); the paper neither cites nor compares with them, reducing originality and theoretical differentiation.
5.	Reproducibility issues — Code and trained modules are not open-sourced, despite heavy reliance on architectural tuning, dataset preprocessing, and hyperparameter sensitivity. This limits independent verification and reuse.

### Questions
1.	How can the authors ensure that the learned latent direction δ corresponds to more reliable knowledge rather than merely the dominant or majority pattern in the training data?
2.	Since both “positive” and “negative” examples rely on externally provided answers, how can you confirm that the autoencoder captures internal knowledge conflict rather than dataset-level bias or label correlation?
3.	Could the learned latent subspace be analyzed (e.g., via activation attribution or neuron probing) to demonstrate interpretability similar to the sparse autoencoder findings in “Sparse Autoencoders Find Highly Interpretable Features in Language Models” (Cunningham et al., 2023)?
4.	How does CARE differ concretely from the SAE-based editing approach proposed in “Steering Knowledge Selection Behaviours in LLMs via SAE-based Representation Engineering” (Zhao et al., 2024), which also manipulates latent features to adjust model knowledge utilization?
5.	Would incorporating explicit sparse or disentangled representation constraints (e.g., sparse coding or orthogonal regularization) improve the interpretability and stability of the learned conflict-aware latent directions?

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
3

### Summary
This paper addresses two critical challenges in RAG: knowledge conflicts between LLM parametric knowledge and retrieved external information, and interference from noisy retrieval results.  The authors propose CARE, a method that leverages autoencoders to learn distinguishable neuron activation patterns for knowledge-aligned and conflict-prone representations in LLMs’ internal layers. Comprehensive experiments across six QA benchmarks (including single-hop and multi-hop tasks) and four LLMs (LLaMA-2 series, LLaMA3.1, Qwen-2.5) demonstrate that CARE outperforms existing RAG optimization and conflict resolution baselines in answer accuracy (EM/F1 scores) while maintaining comparable inference efficiency.

### Strengths
1. This paper proposes a representation editing method to address the knowledge conflicts and interference from noisy within RAG system, differing from prior work that focuses on input filtering or decoding adjustment.

2. The authors evaluate across diverse QA tasks  and model architectures, ensuring generalizability of the proposed method.

3. Ablation studies systematically verify the contribution of each component, and supplementary analyses strengthen the work’s reliability.

### Weaknesses
1. It is recommended that the authors further clarify how the proposed method differs from previous representation editing-based RAG [1] and conflict-aware RAG [2] approaches. CtrlA [1] also employs representation editing during inference to optimize RAG systems. What are the key improvements of CARE over CtrlA in terms of motivation and methodology? Please elaborate in detail.

[1]CtrlA: Adaptive Retrieval-Augmented Generation via Inherent Control.
[2]Tug-of-war between knowledge: Exploring and resolving knowledge conflicts in retrieval-augmented language models

2. The authors analyzed the performance of the proposed method when the number of retrieved documents ranges from 1 to 9. However, since the number of retrieved documents often determines the degree of knowledge conflict, it is recommended that the authors include additional experiments and analyses under settings with a larger number of retrieved documents to further demonstrate the effectiveness of the proposed method.

3.Can the degree of knowledge conflict and the level of knowledge noise be intuitively measured or evaluated? How can the authors demonstrate that the effectiveness of the proposed method indeed stems from its ability to mitigate knowledge conflict and noise? One possible approach is to compare the model’s performance under varying degrees of knowledge conflict or noise interference.

4.The authors select the top 5 layers based on AUC scores, but there is no theoretical explanation for why these layers are more critical for conflict resolution.  Additionally, the impact of layer selection on different model architectures is not deeply discussed.

### Questions
1.On datasets such as PopQA, the proposed method performs worse than existing approaches. It is recommended that the authors further analyze the underlying reasons to clarify the applicability and limitations of the proposed method. Could this be attributed to the unique characteristics of the dataset?

2.The negative samples considered in this work cover only two types—parametric knowledge errors and retrieval noise–induced errors—while real-world RAG conflicts may involve more complex scenarios (e.g., partial overlap between parametric and external knowledge, or contradictions among multiple retrieved documents). Does the proposed method have the potential to address such more complex types of conflicts?

3.For the layer selection strategy: The AUC analysis shows that upper layers have better separability (Figure 5), but why are specific layers (e.g., LLaMA-3.1-8B-Instruct uses layers 3, 12, 18, 20, 31) chosen instead of the top 5 consecutive upper layers?  Is there a functional difference between these layers in encoding conflict information?

4.Does CARE risk losing useful parametric knowledge during representation editing?  For example, when retrieved documents are partially incorrect but parametric knowledge is accurate, can CARE preserve the correct parametric information?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper aims to address the issue of internal–external knowledge conflicts in retrieval-augmented generation (RAG). It proposes a method called CARE, which trains an autoencoder and applies it during inference to alleviate conflicts in the hidden representation space. Experiments are conducted on multiple QA datasets (e.g., NQ, HotpotQA, 2Wiki) and different models (LLaMA-2/3, Qwen-2.5). The method does not require training the base model or retriever and provides a new perspective for handling knowledge conflicts in RAG systems.

### Strengths
1. The paper proposes a representation-level editing approach to address internal–external knowledge conflicts in RAG, which can be applied without fine-tuning the retriever or generator, making it practically meaningful.

2. The method is clearly described and validated on multiple datasets and models, with convincing results.

### Weaknesses
Please refer to the questions section for detailed weaknesses and concerns.

### Questions
1. The paper assumes the existence of a global linear direction \delta that can consistently shift model representations from a “conflict” state to an “aligned” state. However, this direction is derived from positive and negative samples constructed from factual datasets such as NQ and evaluated in the same domain. Can this approach generalize to other domains such as medical, financial, or legal QA, where knowledge structures differ significantly?

2. During inference, the paper applies the learned \delta to all inputs uniformly. Would this unconditional editing introduce interference or degradation on non-conflicting samples?

3. The necessity of introducing the autoencoder (AE) for alignment in hidden space is unclear. For instance, if the same positive/negative pairs used for AE training were instead used for DPO-style preference optimization, how would the results compare? Further clarification from the authors would be helpful.

4. How effective is the proposed method on large reasoning models (LRMs) such as Qwen3 (Thinking), whose internal hidden spaces are more complex and dynamically routed?

5. The experimental comparison mainly includes baselines from 2024, but omits more recent 2025 works such as 《ASTUTE RAG: Overcoming Imperfect Retrieval Augmentation and Knowledge Conflicts for Large Language Models》. Incorporating or discussing such advanced baselines would strengthen the paper’s evaluation.

### Soundness
2

### Presentation
3

### Contribution
3
