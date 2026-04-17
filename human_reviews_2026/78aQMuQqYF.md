# Bridging Audio-Visual Semantics with Language-Guided Synthesis

- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
One of the underlying assumptions behind audio-visual learning models is that the two modalities convey overlapping information. However, this assumption is widely violated in practice, which results in degraded performance. To address this problem, we propose to replace mismatched audio-visual signals using cross-modal generative models. Our approach uses language-based supervision to perform this generation. We show that data synthetically generated through this process is well-suited for a variety of representation learning methods. The features that we learn this way outperform those trained solely on real data for a range of downstream tasks, including audio classification, audio-visual retrieval, and visual sound localization.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a data generation framework that uses language as an intermediate bridge to create semantically consistent audio-visual pairs, solving the problem of misaligned or low-quality data that often degrades multimodal learning models. By generating synthetic images conditioned on audio and synthetic audio conditioned on images, the method repairs corrupted real-world data and generates a highly aligned dataset. The primary contribution is demonstrating that features learned using representations trained on this resulting mixed dataset which combines filtered real data with the high-quality synthetic data outperform models trained solely on the original real data across tasks like audio classification, cross-modal retrieval, and visual sound localization. Human and automatic evaluations confirm that the synthesized audio-visual pairs are more semantically aligned than those found in the original dataset.

### Strengths
- The framework made progress in addressing the issue of misaligned or mismatched audio-visual signals in real-world datasets by using cross-modal generative models to replace corrupted data. Features learned using the resulting mixed dataset (real and synthetic data) lead to better results across a variety of downstream tasks, including audio classification, audio-visual retrieval, and visual sound localization
- The method introduces a filtering process and mixing strategy that combines the best-aligned real data with synthetic data, which is found to outperform models trained purely on either real or synthetic data
- The synthetic data is suitable for a wide range of representation learning algorithms, including contrastive learning methods (like CAVP) and reconstruction-based methods (like CAV-MAE)

### Weaknesses
- The method's core technical strategy involves chaining together several existing, powerful models. While the application of this pipeline to repair audio-visual mismatch is novel, the individual components and their mechanisms are off-the-shelf. Moreover, Language-as-a-Bridge is an established concept. models like ImageBind align multiple modalities by connecting them to frozen vision–language embeddings, and LG-CAV-MAE employs captioning to associate signals with text. I think authors should better clarify that the novelty lies in leveraging this bridge to generate new, corrected data rather than simply aligning existing data.
- The framework uses individual video frames as the visual signal, rather than generating or processing full video sequences, a choice made because generating full video is a challenging open problem. This limits the model's ability to utilize the temporal visual information available in a full video clip.
- The reported experiments focus mainly on high-level downstream tasks, such as overall scene understanding (e.g., classification and retrieval), which are well-suited to the language-based generation approach employed.

### Questions
- Is a fine-grained search of the balance data mixing ratio necessary for every new dataset used for pretraining?
- The caption rewriting strategy was carefully developed to avoid hallucination by the language model and semantic misalignment by altering only the environment and perspective. How was the threshold for modifying captions determined, and what specific metrics or internal evaluations were used to quantify the risk of semantic collapse before choosing the final, constrained approach?
- What specific limitations or performance degradations would the authors expect if this language-guided framework were applied to more fine-grained, low-level audio analysis tasks (e.g., detecting quick transients or specific frequency changes) where language descriptions might lack the necessary acoustic detail?

### Soundness
2

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
5

### Summary
The paper aims to address audio-visual misalignment in web video by replacing the mismatched modality with language-guided synthetic counterparts (audio↔image) and mixing them with the best-aligned real pairs via embedding-space filtering. Train standard audio-visual self-supervised learners on this “repaired” corpus. After using these synthetic data together with real data, different methods achieve consistent gains on audio classification, AV retrieval, and visual sound localization.

### Strengths
1. The paper is generally well-written and the core idea is simple and seems effective.

2. Consistent gains are seen across multiple tasks (Audio CLS (ESC-50 / FSD-50K / Urban8K), ) after using the synthesized data for models training, and helps all models to improve (CAV-MAE, CAVP, AudioCLIP).

3. The paper also compares different synthetic strategies and shows their advantage over the others in the ablations on audio cls and sound localization task.

### Weaknesses
1. The synthetic data is generated using rather weak models, why not use stronger frontier-models like these omni models like Qwen-Omni, MiniCPM-o-2.6, or even like Gemini / GPT ones. The paper should ablate different model's impact on synthetic data quality.

2. The benchmark evaluation lacks an important dataset, which is AudioSet, they can be used for both audio-visual classification and retrieval, the paper should definitely include the results. VGGSound dataset is still considered very small for these tasks.

3. Lack of important citations related to audio-visual representation learning, e.g., [1] and [2].

4. The proposed method is still constrained within the scope of semantic alignment, if extend to video temporal alignment, then the proposed pipeline would be considered as limited.


[1] From vision to audio and beyond: A unified model for audio-visual representation and generation, ICML 2024.

[2] CAV-MAE Sync: Improving Contrastive Audio-Visual Mask Autoencoders via Fine-Grained Alignment, CVPR 2025.

### Questions
See weaknesses above.

### Soundness
3

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
This work proposes a method to resolve the issue of mismatched audio–visual signals using a cross-modal generative model. Specifically, either an audio caption or an image caption is first predicted from the original signal using off-the-shelf models, refined through a large language model, and then fed into a text-to-audio or text-to-image generative model to synthesize a semantically matched pair.
To evaluate the effectiveness of this synthesis approach, the authors learn several audio–visual representations using the synthesized (or combined) datasets and apply these representations to various downstream tasks, such as audio classification and visual sound localization. Experimental results show that incorporating synthetic data during training consistently improves performance on downstream tasks compared to using only real datasets.

### Strengths
- The issue of modality mismatch is an important challenge, and approaching it through a generative model is a promising direction.

- The empirical experiments are comprehensive, covering diverse downstream tasks to validate the proposed approach.

- The ablations are well designed to understand the effectiveness of the method.

- The writing is clear and easy to follow.

### Weaknesses
- While the results demonstrate that synthetic data improves downstream performance, it remains unclear how well the combined real–synthetic dataset generalizes to more recent or stronger baseline models. For example, methods such as “Sound Source Localization is All about Cross-Modal Alignment, ICCV 2023” already surpass the quantitative results shown in Table 6 without using synthetic data. Including additional experiments on more recent models would help verify the generalizability and lasting impact of the proposed approach of generating a synthetic dataset and using it for downstream tasks.

- Some aspects of the method description lack clarity and reproducibility, such as the design of prompts for caption refinement or diversity control. Including example prompts or generation templates would make the method more transparent and reproducible.

- It is also unclear how detailed and diverse the generated images are. How do the authors control for the noisiness, bias, or stylistic artifacts introduced by the text-to-image or text-to-audio generative models?

- Audio classification results on VGGSound are not reported. While the paper includes audio–visual classification on the VGGSound test set, a direct audio-only classification experiment (similar to Table 3) would provide a more complete understanding of the learned audio representations.

- The coverage of long-tail or rare events using synthetic data is not analyzed. Since many mismatched scenarios occur in rare or underrepresented events, synthetic generation could be particularly beneficial here. A detailed analysis of which categories or domains show performance degradation—and where synthetic augmentation provides improvement—would provide valuable insight into the strengths and limitations of the approach.

### Questions
### Questions and Suggestions for Further Experiments
To more thoroughly validate the contribution, the following additional experiments and analyses are recommended:

- Evaluating generalization to more recent or stronger models will confirm the consistent effectiveness of synthetic data augmentation.

- Including audio classification results on the VGGSound test set will complement the existing audio–visual experiments.

- Conducting an error analysis on the data synthesis pipeline, including potential failure cases of the generative models and their effects on downstream performance.

- Analyzing category- or domain-level performance, highlighting where the current model struggles and how synthetic data alleviates these weaknesses, will support the effectiveness of the synthetic dataset.

### Minor Questions and Suggestions

- L279, L379, L408: Use \citep{} instead of \cite{} for consistency with citation style.

- Please capitalize “Figure” and “Table” throughout the paper (e.g., L313: fi.g2 → Figure 2).

- In Table 7, bold the ESC-50 result for the Direct method for readability.

- Provide more details on prompt design for both image and audio generation. For instance, how are environmental variations introduced? Are there structured templates, constraints, or diversity prompts used?

-For tasks that are audio‐only (e.g., ESC-50 classification), how does adding synthetic images help? Do you see gains purely because the audio is unchanged, but training data context changes? Please clarify the mechanism.

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
The paper proposes a new method for audio-visual representation learning. The core idea is to "repair" or "enhance" misaligned parts in the original audio-visual data through a language-guided cross-modal generation model (e.g., text-to-image, text-to-audio) to improve the effectiveness of self-supervised learning. The authors constructed a hybrid dataset containing both synthetic and real data and validated its effectiveness on several downstream tasks (such as audio classification, audio-visual retrieval, sound source localization).

### Strengths
By using text as a bridge to generate images from audio or audio from images, the method repairs misalignments in the original data. The effectiveness of the synthetic data was verified through human perception evaluation, automated semantic similarity evaluation, and comparisons with various representation learning methods. The method's effectiveness was validated on multiple benchmark datasets, with models trained on synthetic data outperforming those trained only on real data.

The paper has a certain degree of originality. While using synthetic data for visual representation learning has been studied, applying it to audio-visual alignment and using language-guided generation to repair misaligned data is a novel and practically valuable direction. The expression and layout of the paper still have room for improvement, such as the complex representation of evaluation mechanisms and the data construction pipeline in the charts. The paper holds a certain level of significance.

### Weaknesses
1. Further assessment of data quality is needed. For example, existing contrastive learning models could be used to evaluate semantic cosine similarity, especially in comparison with datasets constructed by other methods, such as Ex-MCR's retrieval-based dataset construction approach.

2. There is a lack of direct performance comparisons with other data-driven methods, such as ImageBind and LanguageBind.

Ex-MCR: https://proceedings.neurips.cc/paper_files/paper/2024/file/a71df365f872a39e58475f1fa7950879-Paper-Conference.pdf

ImageBind : https://arxiv.org/abs/2305.05665

LanguageBind : https://arxiv.org/abs/2310.01852

### Questions
1. I noticed the paper's discussion on the distributional inconsistency between generated data and real data. Could you elaborate on the data construction pipeline used during fine-tuning? Specifically, what are the sources of the images, text, and audio in the fine-tuning dataset?

2. The current generation models mentioned in the paper, such as FLUX and Stable Audio Open, also rely on contrastive learning models for prompt understanding (these models include pre-trained contrastive learning models in their structures). Essentially, can the paper's method be considered as a distillation and integration of semantic modality alignment within existing generative models? If so, can the paper's method be compared in terms of model performance with methods like Ex-MCR that also integrate existing contrastive learning models?

FLUX: https://arxiv.org/abs/2210.02747

Stable Audio Open: https://arxiv.org/abs/2407.14358

### Soundness
2

### Presentation
2

### Contribution
2
