# Cocktail-Party at the MUSEUM: Referring Audio-Visual Segmentation requires Augmentation

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 6

## Abstract
Recent advances in Referring Audio-Visual Segmentation (Ref-AVS) have significantly progressed, with the development of multimodal fusion methods and Multimodal Large Language Models (MLLM). However, their modality-specific performance is underexplored, and the effectiveness of audio perception remains unclear. We find that current methods often fail to identify the correct sounding object with audio expressions (e.g., $\textit{loudest sounding object}$), especially at the cocktail-party (i.e., mixed audio source). In addition, MLLM methods tend to memorize through visual-text patterns due to their weaker audio understanding capabilities. To this end, we first propose $\textbf{MISA}$: $\textbf{M}$usical-audio $\textbf{I}$nstructed $\textbf{S}$egmentation $\textbf{A}$ssistant, with an integration of specialized musical-audio encoder MERT, and a musical-specific dataset for alignment to enhance audio tokens' representation. To mitigate the lack of variation of mixed-source signals, we introduce $\textbf{MUSEUM}$, a musical-audio augmentation pipeline consisting of three stages: $\textbf{MU}$sical $\textbf{S}$ourc$\textbf{E}$, A$\textbf{U}$gment, and $\textbf{M}$ix, to respectively perform source separation, sampling from extra musical datasets, and audio augmentation. Our proposed augmentation enriches the mixture of audio signals in the existing training dataset, which facilitates the model learning with diverse samples. Moreover, we refine the existing benchmark as $\textbf{C-Ref-AVSBench}$ that categorizes expressions into Audio-Centric (audio cues), AV-Grounded (audio and visual cues), and Visual-Centric (visual cues), in order to perform modality-specific evaluation. Our approach achieves state-of-the-art performance on both Ref-AVSBench and C-Ref-AVSBench, particularly with the Audio-Centric expressions.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose an integration of the specialized musical-audio encoder and a musical-audio augmentation pipeline for Ref-AVS.

### Strengths
1. Good visualization for readability
2. Clear and easy to understand writing
3. The result seems promising.

### Weaknesses
1. About the MUSEUM method: I am confused about the scope of sources addressed by MUSEUM. Ref-AVS includes not only vocals and musical instruments but also many complex general sounds. The paper does not seem to discuss how MUSEUM handles such non-musical sources. Could you provide analyses or experiments that cover these broader sound categories?
2. Contribution of the augmentation method: There are well-established audio augmentation techniques commonly used in training, such as room impulse responses (RIR), filtering, codecs, and SNR-based noise addition. See, for example:
- Torchaudio’s augmentation tutorial: https://docs.pytorch.org/audio/main/tutorials/audio_data_augmentation_tutorial.html
- Audiomentations: https://github.com/iver56/audiomentations
- You also mention MixUp (Line 199). 
The manuscript describes your augmentation stages but does not compare them against these prior augmentations or their combinations. Without such comparisons, it is hard to assess the incremental value of the proposed augmentation pipeline. Please include controlled ablations and head-to-head baselines (e.g., RIR convolution, SNR-controlled additive noise, codec perturbations, filters) to quantify performance gains.
3. On the generality vs. specificity of MUSEUM. “MUSEUM, a musical-audio augmentation pipeline consisting of MUsical SourcE, AUgment, and Mix stages” appears broadly applicable beyond Ref-AVS—for example, to visually guided audio separation, music source separation, and singing voice tasks. What makes MUSEUM specifically suitable or necessary for Ref-AVS? Please clarify what is specialized to Ref-AVS (e.g., source selection, conditioning, mixing policy, or evaluation alignment), and, if general, discuss expected transfer to other tasks with either evidence or a clear rationale.
4. Separation quality and its impact. The paper uses htdemucs (2023), which has known limitations, and source separation itself remains challenging. How do you ensure that separation quality is sufficient for your pipeline? To what extent do separation errors propagate and affect final performance? Please report: Sensitivity analysis across different separation models, versions, and checkpoints. Metrics of separation quality (e.g., SDR/SIR) vs. downstream task performance.
Any robustness mechanisms (e.g., confidence-based filtering, remix consistency, data selection) to mitigate poor separations.
5. Figure 4 formatting. The typography in Figure 4 is distracting: the letter “D” is disproportionately large compared to the other labels, and the templates are too small to interpret. Please standardize font sizes across panels and enlarge the templates for readability.

### Questions
My main concern is the novelty and comparison of the paper. I will reconsider my score after the rebuttal.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper investigates the Referring Audio-Visual Segmentation (Ref-AVS) task, which aims to enable large language models (LLMs) to perform reasoning over video data. It categorizes the learning scenarios into three distinct types: Audio-Centric (audio cues), AV-Grounded (audio and visual cues), and Visual-Centric (visual cues). The study demonstrates that a well-designed Ref-AVS model can effectively interpret each modality and determine which modalities to leverage for accurate segmentation. However, the authors identify that prior methods inadequately address this challenge. To overcome these limitations, the paper introduces MISA—a novel learning framework—and MUSEUM, a musical-audio augmentation pipeline comprising three stages: Musical Source, Augment, and Mix. The proposed approach achieves state-of-the-art performance on both Ref-AVSBench and the refined C-Ref-AVSBench benchmarks.

### Strengths
The topic of effectively incorporating multiple modalities is both timely and compelling.

The proposed benchmark emphasizes three distinct evaluation directions to systematically assess model performance.

The paper presents qualitative results that illustrate the effectiveness of the proposed methods.

### Weaknesses
- The readability of the paper could be enhanced, as certain descriptions lack specificity. For instance, a concise explanation of the purpose (i.e., audio captioning) behind using $L_{txt}$ would improve clarity regarding its role in representation learning. Additionally, in line 246, the rejection supervision training objective is introduced abruptly without sufficient context or motivation, which may lead to confusion for the reader.

- The technical innovation presented in the paper is relatively limited. The overall concept of using synthesized data to enhance segmentation performance is not novel and has already been explored within the audio-visual learning domain. For instance, [a] employs synthetic data to augment audio-visual segmentation datasets, resulting in improved model performance. Moreover, the MISA framework largely builds upon existing training strategies without introducing substantial new insights.

- The baseline results presented in Table 1 may not be directly comparable. For instance, the only method that shares both the same MLLM and segmentation architecture is "Sa2VA-1B (finetuned)", whereas the other entries differ either in the multimodal language model architecture or the segmentation framework, limiting the fairness of the comparison. When compared with "Sa2VA-1B (finetuned)", the proposed method demonstrates only marginal improvement. It is important to note that "Sa2VA-1B (finetuned)" is solely a vision-language model, without incorporating audio modality, which further highlights the limited gain achieved by the proposed approach. I hope the author could make direct comprasion with Mutlimodal-based methods to demonstrate the effectiveness of the propose method.

- The paper also lacks evaluation on the AVSBench dataset, which serves as a more comprehensive benchmark for assessing audio-visual grounding performance.


[a]  Liu, J., Wang, Y., Ju, C., Ma, C., Zhang, Y., & Xie, W. (2024). Annotation-free audio-visual segmentation. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (pp. 5604-5614).

### Questions
n/a

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
In this work, the authors tackle an important but neglected problem in the referring audio-visual segmentation task, the inadequate use of audio information. They propose a data augmentation pipeline (MUSEUM) and a new audio-centric benchmark (C-Ref-AVSBench) from the original Ref-AVS dataset. Extensive evaluation on datasets demonstrate the effectiveness of this framework.

### Strengths
The strengths of this work are shown below:

- First, this paper is well motivated as it points out a potential problem in MLLM when performing the Ref-AVS task, the visual-text shortcut issue. The modality bias could be essential for this field's progress. The authors also curate a new subset from the original dataset to systematically evaluate this issue. 
- The results are impressive. In Table 1 and Table 2, MISA + MUSEUM achieves remarkable progress over the baseline methods, demonstrating the effectiveness of this framework. 
- This paper is well-written and easy to follow.

### Weaknesses
The weaknesses of this work are summarized below:

- As shown in Sec. 3.2, the MUSEUM pipeline is explicitly driven by keywords, such as 'loudest/fastest', and training and testing datasets share the same referring templates for the target object. This raises my concern about the generality of this method, that is, the model is not learning a general, robust concept of "loudness" or "rhythm," but rather a trivial dictionary mapping. I would like to know whether the performance would drop significantly under expressions that are similar in meaning but do not explicitly contain the keywords, e.g., 'the instrument that is drowning out the other one'.
- The paper argues it targets at "cocktail-party" problem, but the entire proposed solution is highly specialized for musical scenarios. There is no evidence that this framework would generalize to non-musical "cocktail-party" scenes involving overlapping human speech or other object sounds.
- In Table 1, the performance improvement from the base "Sa2VA-1B" (J&F 49.2) to the "Sa2VA-1B (finetuned)" (J&F 80.3) is remarkable. This implies that standard fine-tuning on the original dataset (without MUSEUM) already achieves a very high SOTA baseline. This needs to be clarified, as it helps to properly evaluate the contribution of this pipeline.

### Questions
Please refer to the weakness part above.

### Soundness
3

### Presentation
3

### Contribution
3
