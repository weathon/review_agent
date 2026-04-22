# Motion-Aligned Word Embeddings for Text-to-Motion Generation

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 8

## Abstract
Existing text-to-motion (T2M) generation models typically rely on pretrained large language models to encode textual inputs. However, these models, trained on generic text corpora, lack explicit alignment between motion-related words (e.g., "clockwise'', "quickly'') and human skeletal movements. This misalignment, fundamentally rooted in the word embedding layers, severely limits the ability of T2M models to understand and generalize fine-grained motion semantics. To tackle this issue, we propose Motion-Aligned Text Encoding (MATE), a novel framework that explicitly incorporates motion semantics into the word embedding layers of large language models to enhance text-motion alignment for motion generation. To address the challenge of inherent semantic entanglement in motion sequences, MATE introduces two key components: 1) a motion localization strategy that establishes localized correspondences between sub-texts and motion segments, enabling soft attention guidance for semantic localization; and 2) a motion disentanglement module that isolates word-specific motion semantics via contrastive kinematic prototypes, ensuring word-level alignment between linguistic and kinematic representations. Remarkably, language models enhanced with MATE can be seamlessly integrated into existing T2M methods, significantly surpassing state-of-the-art performance on two standard benchmarks with minimal modifications.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces Motion-Aligned Text Encoding (MATE), which improves text-to-motion generation by fine-tuning only the token-embedding layer of a frozen language model to encode motion semantics. The framework has two parts: (1) a text–motion joint segmentation that temporally localizes word meaning within motion sequences, and (2) word-guided motion disentanglement based on dataset-wide kinematic prototypes with self- and cross-disentanglement losses. Plugging the MATE-enhanced encoder into existing T2M backbones yields consistent gains on HumanML3D and KIT. Ablation studies attribute improvements to each loss component and show that restricting updates to embeddings mitigates overfitting compared with tuning deeper layers or using LoRA adapters.

### Strengths
1. The plug-and-play strategy of injecting motion semantics into word embeddings is simple, effective, and likely useful for future T2M and broader multimodal systems.
2. The module delivers consistent improvements across multiple T2M backbones and on both HumanML3D and KIT, demonstrating robustness and generalizability.
3. Comprehensive ablations clearly isolate the effect of each component and provide strong evidence for the overall design.

### Weaknesses
1. The prototype-based contrastive alignment improves stability and semantic discrimination, but may trade off representational consistency against contextual adaptability. The reported generalization to unseen words seems to rely on strong contextual cues and semantic proximity to known terms.
2. The evaluation centers on CLIP/DistilBERT encoders and does not assess more popular decoder-only language models. Testing MATE with T5-style architectures (e.g., MotionGPT [1], M3-GPT [2]) or LLaMA-based models (e.g., SoLaMi [3]) would provide a stronger case for paradigm-agnostic generalization.

>[1] Jiang, Biao, et al. "Motiongpt: Human motion as a foreign language." *Advances in Neural Information Processing Systems* 36 (2023): 20067-20079.

>[2] Luo, Mingshuang, et al. "M $^ 3$ GPT: An Advanced Multimodal, Multitask Framework for Motion Comprehension and Generation." *arXiv preprint arXiv:2405.16273* (2024).

>[3] Jiang, Jianping, et al. "Solami: Social vision-language-action modeling for immersive interaction with 3d autonomous characters." *Proceedings of the Computer Vision and Pattern Recognition Conference*. 2025.

### Questions
1. Relative to training the T2M backbone, what fraction of wall-clock time and compute does training the motion-aligned embeddings add? 
2. The authors decompose each description with ChatGPT, which can introduce noise. Why not use datasets with human action and sentence-level annotations (e.g., BABEL [4])? 

>[4] Punnakkal, Abhinanda R., et al. "BABEL: Bodies, action and behavior with english labels." *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*. 2021.

### Soundness
3

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
4

### Summary
This paper proposes MATE, a novel framework that explicitly incorporates motion semantics into the word embedding layers of large language models to enhance text-motion alignment for motion generation.

### Strengths
1.The paper proposes a novel framework (MATE) that explicitly addresses text-motion misalignment rooted in LLM word embeddings, a critical limitation in prior T2M methods.

2.The paper’s structure is logical, with clear problem formulation, method description, and experimental analysis. 

3.Several experimental results show the effectiveness of this method.

### Weaknesses
1. The issue of misalignment between textual semantics and motion semantics has been raised in several works, such as LaMP. However, the authors neither compare their method with these works in the baseline nor discuss the differences between their solution and others.

2. There are numerous motion annotation errors in HumanML3D and KIT-ML datasets. For example, the motion shows the left hand being raised, while the textual annotation states the right hand. Such noisy data will have a significant impact on the method proposed by the authors.

3. The baseline lacks comparisons with LLM-based T2M methods, such as MotionGPT and MotionGPT2. The authors also fail to discuss the differences between their work and these methods.

4. The overall writing is slightly rough, and the framework diagram is not clear enough.

### Questions
1. Do the authors have any solutions for data with annotation errors? Such data may have many negative impacts on the training results.

2. In the demo, I am curious why the generated motion makes a larger circle when "quickly" appears in the prompt. Is this because the semantics of some actions and words have not been completely disentangled?

3.Given the failure of the re-weighting strategy for word frequency imbalance, could the authors elaborate on the differences between T2M word embedding alignment and standard long-tail learning? Are there adaptive update strategies (e.g., dynamic weighting based on prototype stability) that could mitigate this issue?

4.Could the authors compare MATE with recent LLM-based T2M methods (e.g., MotionGPT2, MotionGPT) in terms of model complexity, training cost, and fine-grained semantic alignment ability? This would better highlight MATE’s advantages.

5.Author should discuss with other works on the misalignment between motion and language question.

I hope author can increase the presentation quality and answer my questions.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper addresses a fundamental challenge in text-to-motion (T2M) generation. Existing approaches that rely on pretrained large language models (LLMs) often suffer from a semantic misalignment between motion-related vocabulary—such as “clockwise” or “quickly”—and the corresponding human skeletal movements. This issue stems from the generic nature of text embeddings in LLMs, which are not inherently designed to capture fine-grained kinematic semantics. As a result, the ability of T2M models to accurately interpret and generalize nuanced motion concepts remains limited.

To tackle this problem, the authors propose a novel framework named Motion-Aligned Text Encoding (MATE), which aims to explicitly inject motion semantics into the word embedding layer of LLMs, thereby enhancing the alignment between linguistic expressions and motion representations.

The MATE framework consists of two core components:

* A motion localization strategy, which jointly decomposes paired text descriptions and motion sequences into semantically aligned sub-units. This process establishes a soft attention prior that facilitates the temporal grounding of word-level semantics in motion sequences.

* A motion disentanglement module, which isolates word-specific motion semantics through two complementary mechanisms: 
    * self-disentanglement, which extracts shared semantics across related motions using contrastive kinematic prototypes; 
    * cross-disentanglement, which explicitly excludes irrelevant semantics to ensure discriminability between different motion words. 

The disentangled motion semantics are then aligned with their corresponding word embeddings, effectively mitigating the word-level misalignment inherent in conventional LLMs.

Extensive experiments validate the effectiveness of the MATE framework. The authors demonstrate that MATE-enhanced language models can be seamlessly integrated into existing T2M pipelines, leading to significant performance gains over state-of-the-art methods across multiple standard benchmarks.

### Strengths
The problem this paper focuses is critical for text-to-motion generation task. This paper is well-structured. In addition, the paper provide extensive experiments to prove the effectiveness of the proposed method. The proposed MATE is integrated into several existing T2M pipelines to validate its effectiveness.

### Weaknesses
## Major Concern:
My primary concern pertains to the foundational text-to-motion matching strategy outlined in Section 3.2. The proposed method relies on a pre-trained dual encoder to partition motion sequences into sub-clips by minimizing a matching loss. While this approach is intuitively reasonable, its effectiveness is critically dependent on the capacity and quality of the pre-trained encoders.

The dual encoder adopted from HumanML3D, in my view, may possess inherent limitations for fine-grained semantic alignment. These limitations stem from its relatively simple architecture, limited model scale, and—most importantly—the nature of its pre-training data. 
The HumanML3D dataset provides only sentence-level motion-text pairs, lacking explicit, fine-grained alignment between sub-motion segments and their corresponding word-level descriptions. Consequently, I have significant doubts regarding the encoder's ability to accurately infer the optimal start and end boundaries for sub-motion clips based solely on weak, sentence-level supervision. 
An inaccurate partitioning in this initial stage could propagate errors throughout the entire disentanglement and alignment pipeline.

This challenge is particularly acute in the domain of human motion generation, which is notoriously data-scarce, especially for fine-grained annotations. The situation stands in stark contrast to other domains like action detection or fine-grained action recognition, where similar matching strategies have succeeded. In those fields, methods typically benefit from powerful, large-scale dual encoders (e.g., CLIP) trained on massive, diverse datasets, which provide a robust semantic foundation. 
A similar replication of this paradigm in the data-poor motion generation domain, without a similarly robust backbone, represents a significant and potentially under-addressed risk in the current work.

## Other Concern
* **Novelty:** As indicated in my major concern, the core technical components—including fine-grained clip recognition, text-segment pairing, and contrastive learning for alignment—have been extensively explored and validated in other research areas (e.g., video-text alignment). While the integration and application of these ideas to the T2M generation task is valuable, the conceptual framework itself does not appear to be fundamentally novel. Therefore, I am inclined to assign a more moderate score regarding methodological innovation.
* **Discussion of Related Work:** To better position this work within the broader research landscape, I would recommend the authors to include a discussion of contrastive learning-based methods from related fields.
* **Clarification of Figure 2:** Figure 2 (3) is a bit unclear.

### Questions
Please refer to the weakness.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents Motion-Aligned Text Encoding (MATE), a novel framework aimed at improving the semantic alignment between textual descriptions and human motion in text-to-motion (T2M) generation tasks. The authors introduce a lightweight fine-tuning strategy that updates only the word embedding layer of a pre-trained language model, while keeping all other layers frozen. This enables the embedding space to better capture motion-relevant semantics.

MATE consists of two core components:
- Motion localization, which aligns motion sub-sequences with corresponding text spans through soft attention guidance; and
- Motion disentanglement, which employs contrastive learning with kinematic prototypes and cross-word negative sampling to disentangle word-specific motion features.

Experimental results show that MATE can serve as a plug-and-play module for various T2M frameworks, significantly enhancing the performance of state-of-the-art models such as MoMask and MotionDiffuse on HumanML3D and KIT benchmarks. Additional ablation, visualization, and user studies further validate the effectiveness and generalizability of the proposed approach.

### Strengths
1. The paper effectively identifies a major limitation in existing T2M approaches, word-level semantic misalignment, and addresses it through a simple yet effective embedding-layer fine-tuning mechanism.
It is the first to tackle the systemic mismatch between linguistic and motion semantics from the word embedding perspective, offering both scientific motivation and engineering value.

2. By restricting fine-tuning to the word embeddings, the method remains computationally efficient and easily integrable into existing architectures.

3. The proposed motion localization and disentanglement modules enhance the interpretability of semantic learning, and the contrastive objectives are elegantly designed.

4. The experimental section is comprehensive, which These experiments strongly support the method’s efficiency.
- Evaluation on multiple T2M baselines (e.g., MoMask, MMM, MotionDiffuse) demonstrate consistent improvements on HumanML3D and KIT datasets across metrics such as R-Precision and FID.
- Visualization of embedding distributions further illustrates improved discrimination of action-related terms such as “left/right” and “quickly/slowly.”
- The ablation studies thoroughly examine loss design, parameter settings, module compatibility, attention mechanisms, and semantic disentanglement.
- Supplementary experiments—including integration with different LLMs, user evaluations, and cross-task tests (retrieval, editing, completion)—confirm the stability and versatility of MATE.

### Weaknesses
1. Text segmentation using ChatGPT may introduce stochasticity and hinder replication.

2. The reliance on a fixed, limited set of prototypes could constrain generalization, potentially biasing the learned representations toward dataset-specific distributions.

3. Direct fine-tuning of the text encoder might reduce its capacity to retain rich or nuanced semantics, especially for ambiguous expressions.

4. While embeddings are improved, the inference stage still does not explicitly address temporal misalignment, leaving potential issues with word ordering or motion duration.

### Questions
1. If ChatGPT-assisted segmentation is unstable, how does the model handle inconsistent segmentation results? 
Additionally, When a motion segment corresponds to multiple keywords, how does the system ensure these cues jointly guide generation? Does it produce two motion sequences (each with its own representation $f^m$) or merge them into a unified representation?

2. Line 246 states "we predefine a set of motion-word prototypes consisting of K learnable
vectors". Does “predefine” mean that the prototypes are manually specified or that only the number K is fixed while the prototypes are learned automatically? If the former, how are the prototype words selected?

3. Was any negative transfer observed for non-motion-related words  during word embedding adjustment? Are there mechanisms to mitigate this?

4. Could MATE’s capability of cross-domain generalization be evaluated through few-shot or zero-shot tasks, i.e., testing on unseen verbs, domains, or motion styles, rather than relying solely on the original training distribution?

### Soundness
4

### Presentation
4

### Contribution
3
