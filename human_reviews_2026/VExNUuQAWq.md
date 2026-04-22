# A Neuroscience-inspired Framework for Tri-modality Alignment of Brain Signals, Vision, and Language

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 4, 2

## Abstract
Visual retrieval from brain signals is a key challenge in Brain-Computer Interfaces (BCIs). Existing methods mainly rely on direct cross-modality mapping, yet they often overlook the neural mechanisms of visual processing, which leads to three major limitations. First, a feature physiology mismatch arises because high-level semantic features extracted by image encoders do not align with the low-level neural responses evoked by rapid visual stimulation. Second, most approaches emphasize cross-modality alignment while neglecting the similarity of neural representations within the same category, which results in poor intra-modality semantic consistency. Third, brain-image alignment typically depends on static image-text semantic spaces and therefore lacks dynamic semantic priors that interact with brain activity. We introduce NeuroAlign, the first neuroscience-inspired framework for brain-vision alignment. NeuroAlign mitigates the feature physiology mismatch by integrating bottom-up structural perception with top-down semantic modulation, enhances semantic consistency through intra-modality self-supervision and cross-modality intra-class constraints, and leverages large language models (LLMs) to provide dynamic semantic signals that interact dynamically with brain responses. Extensive experiments demonstrate that NeuroAlign achieves state-of-the-art performance on both intra-subject and inter-subject retrieval tasks, which validates the effectiveness of this neuroscience-guided alignment strategy.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces NeuroAlign, a neuroscience-inspired framework for aligning brain (EEG), vision, and language modalities to improve visual retrieval from brain signals. It mimics the brain’s bottom-up and top-down visual processing through a Visual Saliency Extraction module for physiologically consistent image features, a Semantic Guidance Alignment module using LLM-generated text priors, and a Dynamic Loss Adjustment mechanism for adaptive multi-objective learning. Experiments on THINGS-EEG2 and THINGS-MEG datasets show that NeuroAlign achieves state-of-the-art zero-shot retrieval performance, significantly outperforming previous brain-vision alignment models. Overall, the framework bridges neuroscience and multimodal learning, offering both stronger decoding accuracy and greater interpretability of brain–vision–language correspondence.

### Strengths
1. The tri-modality setup (EEG–vision–language) with KL-based semantic guidance is methodologically clear and addresses specific bottlenecks in prior EEG decoding approaches (e.g., NICE, BraVL, UBP).

2. NeuroAlign consistently surpasses recent baselines on both intra- and inter-subject retrieval across two datasets. The inter-subject improvement, though modest, is statistically meaningful given the difficulty of EEG generalization.

3. The analysis (temporal saliency, t-SNE category clustering) convincingly demonstrates alignment between neural and visual-semantic representations. The results support the claim that NeuroAlign better mirrors real cortical processing dynamics.

### Weaknesses
1. The method’s dependence on LLM-generated captions may raise reproducibility and stability issues. It is unclear how much the results rely on specific text priors versus inherent EEG-vision correlations.

2. The work focuses heavily on visual saliency and semantic analysis but provides little insight into which EEG features drive cross-modal alignment. Some interpretability analysis (e.g., channel/time contribution) would improve clarity.

3. Regarding the dynamic loss adjustment component, the authors are encouraged to provide the curve of the adjustable hyperparameter during the training to see how it learns during the optimization.

4. The analysis section lacks of in-depth analysis towards the theoretical contribution of this work. The authors are suggested to enrich the description in the analysis section and provide more information regarding why their proposed method works better.

5. The proposed method consists of three major components, i.e., VSE, SGA, and DLA. What are the relationship among these components? Can these components help each other?

6. In the experiment section, the authors are encouraged to add more ablation experiments of the proposed method. VSE, SGA, and DLA should be removed one-by-one to validate their individual performance contributed to the NeuroAlign model.

### Questions
1. How sensitive is NeuroAlign’s performance to the specific choice of LLM-generated captions?

2. Have the authors evaluated whether different captioning models or prompting strategies affect the alignment results and reproducibility?

3. Can the model still perform well when no textual priors or simpler labels are used?

4. Which EEG channels or temporal segments contribute most to the cross-modal alignment?

5. Could the authors provide visualizations or attention maps showing how EEG features correspond to visual or semantic elements?

6. Could the authors provide the evolution curves of the adaptive weights (w1, w2, w3) during training to illustrate how the model balances objectives over time?

7. How stable is the DLA mechanism across different random seeds or subjects? Does DLA always converge to similar weighting behavior across runs?

8. What theoretical insights explain why integrating VSE, SGA, and DLA leads to better cross-modal alignment? Have the authors conducted ablation studies by removing VSE, SGA, or DLA individually to quantify their individual contributions?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces NeuroAlign, a neuroscience-inspired framework for visual retrieval from brain signals. It argues that prior approaches suffer from three issues:  feature–physiology mismatch, weak intra-modality semantic consistency and reliance on static image–text spaces that lack dynamic semantic priors. NeuroAlign addresses these by first extract salient object visual features and align them with EEG  embedding. It then conducts semantic guidance alignment and dynamic loss adjustment to achieve dynamic brain-to-vision alignment.

### Strengths
- This paper is well-written, with three motivations clearly illustrated.
- The experiments demonstrate the effectiveness over the baseline model.

### Weaknesses
- From the introduction, the authors aim to align EEG embeddings with low-level features. In lines 205–206, they mention that “directly aligning EEG signals with raw RGB images often leads to overfitting on high-frequency details,” citing UBP. However, high-frequency details actually correspond to low-level features, and UBP is designed to remove these components, which appears contradictory to the approach taken in this paper. 

- The methodology and motivation of Center-surround Antagonism part is not well-explained.

- Some key ablation studies are missing. What’s the decoding accuracy when ablating the semantic guidance alignment? From Figure 6, the VSE module visualization does not clearly demonstrate enhanced attention to specific image regions. How does it compare with UBP, which applies the Fovea Blur mechanism? Additional experimental analysis is recommended. 
- In Figure 1, several anatomical terms such as PFC, LGN, and PPC are presented. What is their specific contribution to explaining the core principle of the proposed method? Please clarify their relevance to the framework. In addition, the design appears quite similar to [1]; this work should be properly cited.

[1] Choi M, Han K, Wang X, et al. A dual-stream neural network explains the functional segregation of dorsal and ventral visual pathways in human brains[J]. Advances in Neural Information Processing Systems, 2023, 36: 50408-50428.

### Questions
- In the main experiments on the EEG dataset, how was the intra-subject result of UBP obtained? The reported Avg acc do not match the official benchmark (Top-1 = 50.9, Top-5 = 79.7), which are notably higher.
- What does “low-level neural responses” in introduction section refer to? Does it mean the low-level components within the neural signals themselves, or the neural responses elicited by low-level visual stimuli?
- What’s the meaning of E_d in Equation 1.  Besides, the parentheses in Equation 1 are mismatched.
- Equation (5) seems to contain a possible typo: the image representation should be denoted as v rather than t.

Please refer to weaknesses for other questions.

### Soundness
2

### Presentation
3

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
This paper focuses on the cross-modal retrieval task between EEG signals and images. The authors propose their method based on several, somewhat scattered motivations, which they summarize as follows:

1. Existing deep models do not align well with human visual mechanisms;

2. The modality alignment strategy in CLIP overlooks the correlations among unpaired data;

3. The human visual mechanism is dynamic and bidirectional, while existing methods are static and unidirectional (which is actually a subset of motivation 1).

At the methods level, the authors introduce three main designs:

1. Based on the low-level visual features of images, they compute a pixel-wise mask or attention matrix;

2. They design a semantic-guided alignment loss (this approach has already been mentioned in other related works—see the Limitation section);

3. They propose a dynamic loss weighting strategy.

The authors evaluate their method on two brain–image datasets for the cross-modal retrieval task.

In my opinion, the contribution of this paper to the field is limited and does not introduce much that is genuinely novel.

### Strengths
1. The manuscript is well-structured with clear formatting.

### Weaknesses
1. The motivation of this paper is not sufficiently in-depth. In my view, the authors merely list some common issues—for example, that deep networks do not align with human visual mechanisms, and that CLIP’s cross-modal alignment ignores the correlations among unpaired data. These motivations are applicable to many tasks and are not specific to the EEG–image retrieval task.

2. The semantic-guided alignment strategy proposed by the authors in Section 3.3 has already been introduced and applied in previous brain–image retrieval studies [1]. The authors should properly cite these previous works and discuss the similarities and differences.

3. The experimental evaluation is not sufficient. The authors only conducted evaluations on the retrieval task, whereas many recent studies based on the Things-EEG and Things-MEG datasets have also performed evaluations on image reconstruction tasks.




[1] Qiongyi Zhou et al. CLIP-MUSED: CLIP-guided multi-subject visual neural information semantic decoding. ICLR 2024.

### Questions
1. In line 82, the authors mention that deep learning–based visual encoders extract high-level visual features, which leads to misalignment. However, I notice that even though the authors compute a pixel-wise attention map based on low-level visual features to obtain 𝐼_𝑎, they still feed 𝐼_𝑎 into such a visual encoder. This approach does not seem to address the misalignment problem, as it still relies on extracting high-level visual features?

2. I am curious about what the computed pixel-wise attention 𝑅_𝑎 looks like. Could the authors provide a visualization of it? Additionally, the subsequently computed 𝐼_𝑎 might also benefit from being visualized.

3. I don’t quite understand how the proposed dynamic loss weighting strategy can address the bottleneck 3 mentioned by the authors.

4. I couldn’t seem to find the definition of 𝐸_𝑑 in line 217.

5. Some citation formats need to be corrected, for example, in line 262.

### Soundness
2

### Presentation
2

### Contribution
1
