# InfoDisent: Explainability of Image Classification Models by Information Disentanglement

- Decision: Reject
- Scores: 4, 2, 2, 6

## Abstract
In this work, we introduce InfoDisent, a hybrid approach to explainability based on the information bottleneck principle. InfoDisent enables the disentanglement of information in the final layer of any pretrained model into atomic concepts, which can be interpreted as prototypical parts. This approach merges the flexibility of post-hoc methods with the concept-level modeling capabilities of self-explainable neural networks, such as ProtoPNets. We demonstrate the effectiveness of InfoDisent through computational experiments and user studies across various datasets using modern backbones such as ViTs and convolutional networks. While InfoDisent achieves competitive performance within the class of interpretable models, we observe an accuracy-interpretability trade-off when compared to black-box counterparts, especially visible in CNNs. Notably, InfoDisent generalizes the prototypical parts approach to novel domains (ImageNet).

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents InfoDisent, a post-hoc explainability method that disentangles features from a frozen pretrained backbone through an orthogonal decomposition of the channels. The goal is to identify prototypical parts that explain the model's decisions. The paper provides multiple experiments showcasing that the approach is applicable to a wide range of models, from CNNs to Transformer architectures, with classification performance on several datasets. It also includes an evaluation of interpretability on the FunnyBirds dataset and two user studies assessing confidence in predictions and ambiguity in explanations.

### Strengths
- **Significance**: 
	- The proposed approach is flexible and can be effectively applied to various pretrained backbones, which is a significant advantage over methods limited to specific architectures or that needs to retrain the model.
	- The results demonstrate that the method largely preserves the accuracy of the original pretrained backbone, especially compared to some other prototypical part methods.
- **Originality**: The decomposition into prototypical parts through the orthogonal matrix and its optimization process, is original. The reformulation of the optimization problem using skew-symmetric matrices is clever and an interesting technical contribution, although hidden  and only mentioned in the appendix.

### Weaknesses
- **Clarity**: 
	- The paper's overall structure and clarity needs to be improved. Several key discussions and technical details, such as the full optimization process, are briefly mentioned or only detailed in the appendix. Describing the overall training and optimization process more comprehensively in the main paper would significantly enhance understanding.
	- The positioning of tables and figures within the paper is confusing, and detracts from the reading flow.
- **Quality**: The supplementary material includes a section labeled "ablation study" which, in its current form, does not present a true ablation of the method's components. The method involves multiple design choices beyond a "default classification head", the orthogonal decomposition, max pooling, and the non-negative coefficients matrix. A proper ablation study for each of these components is missing, making it difficult to assess their individual importance and contribution to the method.
- **Significance**: While the method offers flexibility across architectures, the quality of interpretation, from the evidence presented, appears to be similar to existing prototypical part methods. The paper could better articulate how InfoDisent's interpretability provides a distinct advantage or deeper insights beyond its architectural flexibility.

### Questions
- How consistent are the prototypes found across different classes and samples? Does the same channel consistently link to the same "concept" or "prototype," or do these interpretations vary significantly?
- It is mentioned in the section about "optimizing U and W" that the goal is to find a product of a sparse and unitary matrix. Are there any explicit sparsity constraints or regularization terms added to the optimization objective, other than optimizing for an orthogonal matrix, to encourage this desired sparsity?
- Can the method be effectively applied when transferring from one dataset to another (e.g., using a backbone trained on ImageNet and applying InfoDisent to explain predictions on a medical imaging dataset)?

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The proposed method InfoDisent is a post-training classification head designed to enhance the interpretability of frozen image-based model backbones.
The underlying idea that pretrained features already contain the necessary atomic concepts, is plausible and well motivated.
The method combines an orthogonal transformation, sparse pooling of the most positive and negative activations per channel, and a nonnegative linear classifier, to isolate channel-wise representations that correspond to salient semantic cues.
The authors interpret this mechanism as an information bottleneck that disentangles latent features into concept-like channels and enables prototype-based explanations in a single forward pass.
Experiments on several vision datasets demonstrate comparable accuracy and visually coherent explanations.

### Strengths
The proposed method is technically simple and broadly applicable. It can be attached to any frozen backbone without finetuning, offering post-training interpretability with minimal computational overhead. The model produces explanations in a single forward pass, which is computationally efficient compared to gradient-based saliency approaches and aligns with the idea of self-explainable architectures. The qualitative results are intuitive and visually consistent across datasets, and the experiments demonstrate reasonable generality of the method.

### Weaknesses
**1.**
L57: The paper claims that InfoDisent “leverages the information bottleneck principle” to ensure each channel encodes an atomic concept, but no mutual information quantity (e.g., (I(X;T)), (I(T;Y))) is defined or measured. The idea of “information bottleneck” is used rhetorically, not mathematically.

**2.**
L155: The method section introduces “extremely sparse pooling” and “unitary map (U)” yet provides no justification for how these operations realize an information bottleneck. There is also no ablation showing the effect of each component.

**3.**
L164: The explanation of sparse pooling uses notation (K \to \text{mx_pool}(K) = \max(\text{ReLU}(K)) - \max(\text{ReLU}(-K))). The presentation is algebraically correct but conceptually vague. 

**4.**
L175: The notation (I = (I_{rs})*{rs} \to J = (U I*{rs}) \to v_J = \text{mx_pool}(J)) is confusing. Indices (I, J, r, s) are inconsistently used to denote both spatial positions and transformed tensors. This ambiguity makes even simple tensor operations difficult to follow.

**5.**
L64, Table 3: ImageNet experiments are introduced as the first generalization of prototypical parts to large-scale datasets, yet Table 3 
shows a performance drop. Although the performance–explainability tradeoff is understandable, the observed performance gap between CNN and ViT backbones on ImageNet should be further investigated. The method yields a noticeably larger accuracy drop for CNNs, while transformers retain most of their baseline performance. This difference likely reflects how the proposed orthogonal transformation and sparse pooling interact with backbone representations, but the paper provides no analysis or hypothesis. A deeper examination of this effect would strengthen the claims about scalability and general applicability.

**6.**
Much of the essential methodological and quantitative content resides in the appendix, while the main text remains largely descriptive. Core analyses, e.g.,  disentanglement metrics and quantitative evaluation of interpretability, are only referenced but not summarized. As a result, the paper’s main body does not provide enough self-contained evidence for the claims. The authors should bring key results and analyses into the main text to ensure scientific transparency.

**7.**
Heatmaps are compared qualitatively, but the text asserts that “our approach yields more focused heatmaps.” This claim is unsupported by any numeric localization score or overlap measure.

**8.**
L1509: Theoretical explanation of sparsity seems important but no ablation or analysis demonstrates causal relation between orthogonalization and sparsity. Orthogonalization alone does not guarantee sparsity, and the claim remains speculative without quantitative evidence.

**9.**
The claimed novelty of InfoDisent is somewhat overstated. Similar prototype-based interpretability approaches already explore post-hoc or semi-post-hoc explanations using frozen backbones. The main contribution here lies in a streamlined architectural design, which is technically neat but conceptually incremental. The introduction would benefit from a clearer articulation of what gap in interpretability research this specific formulation addresses.

**10.**
The term “atomic concept” is commonly used in the interpretability literature to denote minimal, human-recognizable visual units. I accept this usage as conventional. However, in this paper, the notion is treated as an achieved property rather than a descriptive goal. The authors should clarify that “atomic” here refers to qualitative observations, not a formally verified characteristic, and avoid implying theoretical guarantees.

**11.**
Figures are numerous but lack quantitative caption summaries. The prose often mixes conceptual justification with qualitative description, making it difficult to distinguish evidence from interpretation.

### Questions
**Q1.**  You state that InfoDisent “leverages the information bottleneck principle,” yet no mutual information quantity (e.g., (I(X;T)), (I(T;Y))) is defined or measured.
Can you formally specify what information quantity is being constrained, and how your loss function or architecture approximates an information bottleneck objective?
If the term is used metaphorically, please clarify this explicitly.


**Q2.**
The method section introduces the “extremely sparse pooling” and “unitary map (U)” but provides no justification for how these operations achieve an information bottleneck.
Could you provide a theoretical argument or an ablation experiment showing how removing either component affects information compression or interpretability?


**Q3.** 
Your sparse pooling function (L165) selects only two scalars per channel.
What statistical or information-theoretic reasoning supports this choice?
Have you compared it to alternatives such as top-(k) pooling or soft log-sum-exp pooling, and can you report how these affect sparsity and accuracy?


**Q4.** 
The notation in L177-184 is hard to follow.
Can you provide a clearer tensor formulation, explicitly defining the dimensions of each variable to make the method reproducible?


**Q5.**
The ImageNet experiments show a noticeable performance drop, particularly for CNNs.
Although some tradeoff between interpretability and accuracy is expected, could you investigate why the degradation differs between CNN and ViT backbones?
For example, does orthogonal transformation interact differently with convolutional versus attention features?
Please include a diagnostic or ablation study to analyze this effect.


**Q6.**
You claim that InfoDisent “matches classical models,” but no error bars or variance estimates are shown.
Could you report statistical confidence intervals or repeat runs to substantiate the claim of comparable performance?


**Q7.**
Much of the methodological and quantitative content is relegated to the appendix.
Can you summarize key results (e.g., disentanglement metrics, quantitative interpretability scores) in the main paper to make it self-contained?
In particular, which quantitative evidence best supports your central claims?


**Q8.**
The text claims that InfoDisent produces more focused heatmaps than Grad-CAM or LRP, but no numeric localization or overlap metrics are shown.
Could you provide quantitative comparisons (e.g., IoU, pointing-game accuracy, or deletion/insertion scores) to support this statement?


**Q9.**
You attribute sparsity to the combination of a frozen backbone and an orthogonal transformation, but no analysis demonstrates this causality.
Could you add an ablation experiment showing activation sparsity with and without orthogonalization, or visualize activation histograms to quantify the effect?


**Q10.**
Prototype-based interpretability methods such as ProtoPNet and PIP-Net already explore post-hoc interpretability using frozen backbones.
Can you clarify what precise gap InfoDisent fills beyond simplified architecture?
In what aspect does your approach advance the understanding of interpretability rather than streamline prior designs?


**Q11.**
The paper frequently refers to “atomic concepts” as if they are achieved properties.
Could you clarify how you define atomicity in measurable terms?
Is there any quantitative analysis (e.g., concept purity, mutual information, or ablation sensitivity) showing that channels correspond to minimal, non-overlapping semantic units?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces InfoDisent, a method for explainability (XAI) that disentangles information in the final layer of pretrained models into interpretable atomic concepts. It’s a hybrid approach combining the flexibility of post-hoc methods with the concept-based transparency of ante-hoc models (like ProtoPNet). The method applies an orthogonal transformation and information bottleneck (using sparse pooling) to produce prototype-like channel activations without retraining the backbone. It demonstrates interpretability and competitive performance on datasets including CUB-200-2011, Stanford Cars, Stanford Dogs, and ImageNet, and includes user studies for human interpretability.

### Strengths
1. It unifies the interpretability of ProtoPNet-like models with the flexibility of post-hoc XAI, taking the best of the both worlds.
2. Evaluation spans five datasets, including the challenging ImageNet, where most prototype-based methods fail.

### Weaknesses
1. First of all, the authors should cite concept bottleneck models - related papers. That includes fully interpretable concept bottleneck models to post hoc-based concept bottleneck models.

2. One big critcism is how to relate these prototypes to human-readable concepts. Often, these prototypes resemble to something unusual which does not match any known concept. I am coming from a medical imaging background, and I have seen this a lot in medical images where these prototypes often select a patch that is not related to any human anatomy. The interpretability community need to solve this challenge to actually produce plausible interpretation of a deep model. This is both true at a local and global level.

3. The paper's central concept, "information disentanglement", is not well-defined or rigorously evaluated. The authors claim the channels represent "atomic concepts" , but this is a subjective claim based on cherry-picked qualitative examples (e.g., Fig 1's "strawberry texture" ). The primary quantitative metric for disentanglement (RV coefficient) is relegated to the appendix (Table 14)  and shows mixed results, with InfoDisent performing worse than the baseline in several cases (e.g., ConvNeXt-L on ImageNet). For a paper with "Disentanglement" in the title, this core concept needs a much stronger theoretical grounding and more convincing quantitative support.

4. The paper introduces a trainable orthogonal transformation ($U$) and a sparse pooling operation, claiming this leads to "atomic concepts". However, it lacks a dedicated loss term to enforce this. There is no equation equivalent to a variational bottleneck's KL-divergence (like in $\beta$-VAE) or a mutual information maximization term that would mathematically compel the channels to represent independent, non-overlapping concepts. 

5. The paper never formally defines what "disentanglement" means in this context. It is used as a qualitative descriptor for the desired outcome (interpretable prototypes), but the process of how the transform and bottleneck actually achieve this separation of information is not rigorously explained or validated.

6. The main paper presents Figure 2 as the model architecture. This figure is incomplete and misleading, as it shows a simple max(ReLU(...)) operation, which is non-differentiable and cannot be trained as-is. As you correctly point out, Figure 9 (in the Appendix) reveals the actual training architecture. SO, Fig 9 has to be in the main section.

7. The authors are not fully transparent about the performance-interpretability trade-off. Table 3  clearly shows that InfoDisent's accuracy is lower than the original (non-interpretable) backbone. For example, ResNet-50 drops from 76.1% to 67.8%, and Swin-S drops from 83.4% to 81.4%. This is a critical trade-off. However, the authors claim their approach "not only matches the performance of these classical models but also offers enhanced explainability". This is verifiably false; it doesn't match the performance. This trade-off is the cost of interpretability and should be discussed as such, not to be deemed as a non-issue. Also, Table 1 and Table 2 shows info-dissect falls short to many competitive methods.

### Questions
1. Could you please provide a formal, technical definition of "disentanglement" as it applies to this work?
2. There appears to be a significant discrepancy between the method's presentation in the main paper and the appendix. Figure 2 shows a simple, non-differentiable \textt{mx_pool} operation , while Figure 9 in the appendix reveals the actual, more complex training-time architecture, which relies on a Gumbel-Softmax estimator. This is a critical, non-obvious detail. Can you confirm that the Gumbel-Softmax-based architecture in Figure 9 is, in fact, the reproducible method, and will you move this to the main paper in a final version?

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
The authors proposed InfoDisent, an interpretable image classification framework based on information disentanglement and sparse concept reasoning.
It applies an orthogonal transformation to decorrelate feature channels and a deterministic max–min pooling mechanism to extract positive and negative evidential activations, forming disentangled atomic concepts.
These concepts serve as interpretable units that link model decisions to localized object parts while maintaining scalability through channel sparsity and a frozen backbone.
Unlike conventional prototype models that rely on large prototype sets or end-to-end fine-tuning, InfoDisent achieves interpretability and efficiency simultaneously across both CNN and transformer backbones.
Comprehensive experiments, including quantitative interpretability benchmarks, user studies, and robustness tests, demonstrate that InfoDisent yields human-recognizable explanations and reasonable classification consistency.

### Strengths
- **S1. Scalable and sparse prototype reasoning**

The proposed method effectively addresses the scalability problem common in prototype-based XAI methods, where an excessive number of prototypes limits applicability to large datasets.
Through an orthogonal transformation that decorrelates feature channels and a deterministic max–min pooling bottleneck, it prunes redundant or non-discriminative activations, leaving only a compact set of meaningful concept channels.
This sparse and disentangled design reduces the effective number of prototypes while preserving interpretability and classification consistency.
Empirical results confirm that InfoDisent uses far fewer active channels than prior models yet maintains competitive accuracy, showing that scalability and interpretability can be achieved simultaneously.

- **S2. Comprehensive and rigorous experimental design**

The current InfoDisent framework demonstrates a clear improvement in experimental scope and rigor.
Across Sec 4 and 5 and the appendix, the authors integrate a broad range of evaluations covering classification performance, interpretability, robustness, and efficiency.
Beyond standard accuracy tests, the study includes multi-backbone benchmarking, user-study validation, and quantitative interpretability metrics using the FunnyBirds and Spatial Misalignment datasets.
Additional analyses in the appendix--such as ablation, sparsity, efficiency, and robustness tests--further confirm the model’s stability and scalability.
Together, these additions transform the evaluation into a well-rounded and reproducible experimental framework supporting the method’s reliability and generality.

### Weaknesses
- **W1. Moderate performance gap, backbone sensitivity, and residual background influence**

The proposed method achieves strong interpretability and sparsity, yet its classification accuracy remains slightly below that of the best-performing prototype and concept-based approaches, as reported in Tables 1 and 2.
Its performance also appears dependent on the underlying backbone: results are generally higher with ResNet, where localized convolutional priors align well with the method’s spatial disentanglement, but less competitive with transformer architectures such as ViT or Swin, which rely on global attention patterns.

Furthermore, the FunnyBirds evaluation (Fig. 7) indicates that while the model obtains the highest overall interpretability score, its Background Independence (B.I.) metric is somewhat lower than that of certain baselines. This may suggest that a portion of the learned concepts still captures contextual or texture cues inherited from the frozen backbone--such as background elements around target objects--rather than purely object-centric information.

My questions would be:
- To what extent is the observed accuracy and backbone variation driven by architectural inductive bias, with convolutional locality favoring InfoDisent’s pooling and disentanglement design?
- Could the orthogonal disentanglement mechanism interfere with globally distributed representations in ViTs?
- Does the frozen-backbone configuration limit adaptability to transformer-based features?
- Would incorporating explicit background suppression mechanisms--such as object masks, counterfactual interventions, or background-invariance regularizers--help improve B.I. while preserving sparsity and accuracy?

### Questions
Most of my main concerns or questions have been outlined in the Weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
3
