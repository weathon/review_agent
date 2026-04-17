# Concept-level Multimodal Reasoning via Semantic Representation for Intent Recognition

- Decision: Reject
- Scores: 4, 6, 2

## Abstract
Multimodal intent recognition is a fundamental task in understanding human communication, aiming to infer intent from heterogeneous modalities and serving as a cornerstone for developing human-centric systems. However, existing methods face two key challenges. First, they rely on entangled and modality-specific features, which hinder the derivation of interpretable representations across modalities. Second, they lack explicit reasoning mechanisms, making it difficult to capture high-level semantic dependencies and systematically link multimodal evidence to complex intents. To address these issues, we propose a novel method (ConMR) that conducts concept-level multimodal reasoning by jointly learning semantic concept representations and modeling concept relations. Specifically, we first leverage the Large Language Model (LLM) to generate high-quality intent-related concepts, providing explicit semantic anchors beyond shallow features. By supervising multimodal feature mapping through activation alignment, these concepts yield interpretable and discriminative representations. Building on this foundation, the concept-level multimodal reasoning module models concept-to-intent relations through LLM-guided relevance scores and infers inter-concept relations from activation patterns. By jointly exploiting these relations, it guides transparent reasoning paths from concepts to intents, thereby enhancing both accuracy and interpretability. Extensive experiments on two challenging datasets show that ConMR outperforms state-of-the-art methods with superior robustness and interpretability, laying a new paradigm for multimodal intent recognition.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes a Concept-level Multimodal Reasoning framework for intent recognition named ConMR. Specifically, ConMR includes two core modules: Concept Representation Learning and Concept-level Multimodal Reasoning. The Concept Representation Learning module leverages a large language model to automatically generate and filter intent-related concepts for text, video, and audio modalities, and maps multimodal features into a unified, interpretable concept space under activation alignment supervision from pretrained modality-specific encoders. The Concept-level Multimodal Reasoning module models both concept-to-intent relevance and inter-concept relations derived from activation patterns, enabling structured reasoning paths from concepts to intents.

The experimental evaluation in this paper assessed the performance of the proposed ConMR on two challenging multimodal intent recognition benchmarks (MIntRec and MIntRec2.0), comparing it with various state-of-the-art methods. The results indicate that the proposed ConMR achieved consistent performance improvements and superior interpretability, with notable gains in accuracy and F1-score across diverse intent categories.

### Strengths
- The attempt to transform feature-level patterns into the explicit concept-level paradigm is valuable.
- The experimental analysis is overall thorough; even the appendix is well-organized.

### Weaknesses
- The quality of the entire concept space and the evaluation of intent relevance rely heavily on the capability and stability of the large language model. If the LLM’s understanding is inaccurate in specific domains or performs poorly in cross-lingual and cross-cultural scenarios, it may introduce noisy concepts and biased relevance scores.
- The ablation study on the concept selection strategy is somewhat incomplete. For example, whether similarity filtering is necessary, whether submodular selection is applied, or how varying the number of concepts affects performance.
- The paper lacks discussion on why SBERT, XCLIP, and CLAP were specifically chosen to compute association scores.
- Concept-level reasoning mainly focuses on constructing and reasoning within the same modality; however, cross-modal concept connections often better reflect real intent (e.g., contradictions or alignments between visual expression and verbal tone). The current fusion mainly relies on concatenation and shared weights, lacking explicit modeling of cross-modal concept interactions.
- Considering that the model itself requires an LLM and involves multiple encoders (SBERT, XCLIP, CLAP, PLM, Swin Transformer, WavLM), there is a complex coupling between these components, making it difficult to analyze the contribution of each model or conduct ablations on different combinations. Moreover, the combination of multiple models introduces additional computational burdens.

### Questions
- It seems that the concept set is usually fixed before training and cannot be dynamically adjusted based on samples or contexts. Does this imply that a separate model must be prepared for each benchmark, thereby posing challenges for open-domain scenarios?

### Soundness
3

### Presentation
2

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
This paper presents ConMR, a novel framework for multimodal intent recognition that elevates reasoning from the feature level to the concept level. The core idea is to leverage LLM-generated concepts as semantic anchors, learn explicit concept representations from multimodal features, and then perform structured reasoning over concept-intent and inter-concept relations. The authors demonstrate state-of-the-art performance on two established benchmarks (MIntRec and MIntRec2.0) and provide extensive experiments, including ablation studies and case analyses, to validate their design choices.

### Strengths
This paper's principal strengths lie in its conceptual contribution. It proposes a concept level reasoning approach for multimodal intent recognition, effectively bridging the gap between low-level features and high-level intents to enhance both performance and, crucially, model interpretability. The proposed ConMR framework is meticulously designed, integrating LLM-based automatic concept generation, a supervised feature-to-concept transformation, and a dual-path reasoning module that models both concept-to-intent and inter-concept relations. This rigorous design is supported by compelling empirical evidence, including state-of-the-art results on two benchmarks, ablation studies that validate each component, and case analyses that demonstrate its transparent decision-making. Furthermore, the framework's robustness is confirmed by its consistent performance across different LLM, solidifying its value as a significant advancement towards trustworthy multimodal AI.

### Weaknesses
see questions

### Questions
1. How would the framework perform in a low-resource setting where access to powerful LLMs like Gemini-2.5 is limited? Is there a fallback strategy or a lighter-weight alternative for concept generation?
2. The failure case analysis in Appendix E.3 is excellent. Could the framework be extended to incorporate a feedback loop that uses such mis-predictions to iteratively refine the concept set?

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
3

### Summary
The paper introduces ConMR, a concept-level multimodal reasoning model that uses LLM-generated concepts and relevance scores to supervise intent recognition training. The framework is clearly presented and shows consistent empirical improvements, with potential benefits in reducing inference time and cost compared to full MLLMs.

However, its true novelty is minimal—the method mainly distills MLLM priors through an additional MSE loss, while the underlying structure remains a standard CBM.
The claimed "reasoning" ability is in fact semantic alignment, not autonomous reasoning, and the paper overstates its originality.
Therefore, this is a practical but not theoretically innovative work whose main value may lie in efficiency rather than conceptual contribution.

### Strengths
1. The paper proposes a two-stage intent understanding framework consisting of Concept Representation Learning and Concept-level Multimodal Reasoning. It transforms entangled multimodal features into explicit concepts and reasons over their relations, using LLM-generated semantic concepts as interpretable anchors. This design improves interpretability, enabling the model to produce concept-level activations that correspond to human-understandable semantic units.
2. The experiments are conducted on two widely used benchmarks (MIntRec and MIntRec2.0), and the results are consistent across multiple evaluation metrics. The proposed method achieve the best performance compared with fusion-based methods and contrastive learning based methods.
3. The manuscript is logically structured, clearly written, and easy to follow despite some overcomplicated exposition.

### Weaknesses
1. Given that the proposed method already relies on LLMs for concept supervision and intent-relevance supervision, it is unclear why the authors do not directly employ LLMs for multimodal intent recognition (e.g., via zero-shot or fine-tuning approaches) instead of introducing additional concept reasoning layers.
Importantly, the experiments do not include comparisons with LGSRR[1], which, according to the authors, is the first work leveraging LLMs for multimodal intent recognition in Line 54.
This paper only compares with traditional fusion-based methods (MulT, MAG-BERT, MISA, etc.) and contrastive approaches (TCL-MAP, MVCL-DAF, etc.), but lacks LLM-based baselines, including:
* MLLM zero-shot/few-shot baselines, e.g., Gemini-2.5 directly predicting intents from multimodal inputs;

* MLLM pseudo-labeled baselines, where MLLMs label training samples with / without CoTs and a lightweight model is then trained on those labels.

Such comparisons should be included to convincingly demonstrate that the improvement of ConMR comes from its structured semantic reasoning mechanism rather than merely relying on LLM priors.

[1] Llm-guided semantic relational reasoning for multimodal intent recognition, Qianrui Zhou, et. al. 2025. arXiv 

2. Modern multimodal LLMs (e.g., Gemini 2.5, Qwen-VL-Series) are already capable of: (1) directly handling text, audio, and visual inputs,(2) producing explicit reasoning chains via prompting, and (3) generating concept-level explanations. Therefore, ConMR’s approach appears to distillate LLM capabilities to ConMR’s training stage.
The authors need to clarify what concrete advantages the ConMR provides in terms of performance or interpretability beyond what a capable MLLM can already achieve.

3. This is a concern of the motiation.

Lines 56–61:

> "First, existing methods predominantly operate at the feature level, relying on entangled and abstract representations that leave a substantial gap between low-level multimodal signals and the nuanced semantics of human intent. Second, they lack explicit and structured multimodal reasoning mechanisms capable of modeling the interplay between high-level semantic representations, which makes it difficult to construct transparent and discriminative paths that bridge raw inputs to complex intents. "

However, these challenges are no longer valid in the LLM era, since current LLMs already perform feature disentanglement and explicit chain-of-thought reasoning for multimodal tasks. Thus, this argument cannot serve as a justification for proposing ConMR.

Additionally, lines 54–56: 

> "while LGSRR Zhou et al. (2025) represents the first attempt to leverage Large Language Models (LLMs) to guide multimodal intent recognition. "

If so, the authors must clearly articulate what limitations of LGSRR or other LLM-based approaches remain unsolved.
The two points raised in lines 56–61 describe issues of pre-LLM fusion methods, not of LLM-based methods.
Therefore, the introduction currently fails to identify the real gap that ConMR aims to address in the post-LLM context.

4. The novelty is overstated. 

Line 136-138:

> " Although CBM research has made notable strides in interpretability, it remains confined to surface-level contribution scores of concepts rather than capturing their intrinsic semantics, which fundamentally constrains its performance."

This claim is somewhat overstated.The real methodological novelty of ConMR lies only in introducing LLM-generated concept–intent relevance scores as explicit supervision via the MSE loss. Traditional Concept Bottleneck Models (CBMs) indeed learn a concept layer followed by a linear classifier to map the concepts and intent labels, where concepts are usually human-annotated or similarity-based matching. ConMR differs mainly in that: (1) the concept supervision is generated by an LLM (2) the concept-to-intent supervision also comes from the LLM’s semantic similarity scores. Therefore, for the method proposed in this paper, the structure is CBM-style, while the semantics come from the LLM. It is also explicitly shown:

* KL $\rightarrow$ concept-semantic consistency
* MSE $\rightarrow$ LLM-prior consistency
* CE $\rightarrow$ task-label consistency

Only the LLM-based semantic supervision is genuinely new. This is a practical innovation, not a theoretical one. It can be describe as a LLM-guided CBM method. This paper over-claims its methodological novelty. 

5. Ablation results largely reflect dependence on LLM priors rather than genuine reasoning ability. In Table 2, the ablation outcomes can be directly explained by removing or preserving access to the LLM-derived priors. 

Line 369-372:

> "Besides, a severe degradation is observed when the learnable transformation W is replaced with linear layers with activations (w/o W), with metrics on MIntRec2.0 dropping by more than 7%, which highlights the critical role in generating robust concept representations."

Line 372-374:

> "In the concept-level multimodal reasoning module, removing LMSE (w/o LMSE) results in performance drops from 0.49% to 2.28% across all metrics on both datasets, confirming the importance of LLM-based intent relevance score supervision."

Line 374-375:

> "Furthermore, ablating the concept-to-intent pathway (w/o Zconcept) causes a severe collapse, with accuracy on MIntRec dropping to 36.36% and F1 on MIntRec2.0 falling to 46.99%."

W learns LLM-provided "semantic ground truth", and it is essentially a distillation of the LLM prior, not a discovery of new relationships among the data. 

Similarly, line 372-375:

> "For concept-to-intent reasoning, intent-conditioned relevance scores generated by Gemini-2.5 are leveraged to guide a weighting network through MSE loss to selectively reinforce concept features."

It shows that removing $𝓛_{MSE}$ or the concept-to-intent pathway $Z_{concept}$ leads to drastic performance drops (up to total collapse). The fact may be that removing these components effectively removes the LLM semantic channel. Therefore, the ablation study does not demonstrate autonomous concept-level reasoning, however, it shows that ConMR’s performance is driven by dependence on LLM-generated priors rather than self-learned reasoning capacity.

### Questions
NA

### Soundness
1

### Presentation
1

### Contribution
2
