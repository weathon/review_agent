# NeSy-MMCAD: A Neuro-Symbolic Multimodal Framework for Child-Abusive Meme Detection and Explanation with Emotion Consistency

- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Child-abusive memes pose a serious online safety threat by combining imagery, overlaid text, and humor to mask coercive or exploitative cues. Standard multimodal classifiers, while effective on surface features, often fail in subtle or low-resource cases. We present NeSy-MMCAD, a neuro-symbolic multimodal framework for child-abusive meme detection and explanation with emotion consistency. Our architecture integrates neural perception with symbolic reasoning: neural modules extract probabilistic predicates from images and text, capturing child/adult presence, nudity, violence, toxic language, coercion, and affective signals, while domain-informed rules encode commonsense constraints. A differentiable rule loss is jointly optimized with the classification loss, enforcing symbolic consistency while retaining flexibility to learn from data. Emotion-aware rules capture affective incongruities, and mitigation rules reduce false positives in benign contexts. To support this work, we curate DACAM (Dataset for Analysis of Child-Abusive Memes), a benchmark resource for evaluating harmful content detection. Experiments on DACAM demonstrate improvements in classification accuracy and interpretability over baseline multimodal models. Importantly, rule activations provide transparent explanations that link predictions to explicit constraints. These results demonstrate the effectiveness of combining neuro-symbolic reasoning, multimodal representation learning, and emotion consistency to enhance the reliability and accountability of AI systems for socially critical tasks such as child-abuse detection.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces MM-CAD, a neuro-symbolic multimodal framework designed to detect and explain child-abusive memes, a critically underexplored yet highly sensitive area of harmful content moderation. The approach integrates visual cues from CLIP, textual features from OCR and LLM encoders, and a Quantum-inspired Embedding Enhancement (Q-EE) module that maps multimodal features into a higher-dimensional Hilbert space to better capture subtle, entangled abuse patterns. To support the task, the authors curate DACAM, the first benchmark dataset specifically focused on child-abusive memes, with balanced labels and strong annotator agreement.

### Strengths
- The proposed work introduces the first dedicated framework and dataset for detecting child-abusive memes, addressing an important yet underexplored safety problem.
- The work combines neuro-symbolic reasoning, CLIP vision features, OCR text, and quantum-inspired embedding to achieve robust and interpretable detection.

### Weaknesses
- DACAM focuses narrowly on child-abusive memes and may not generalize to broader abusive or multimodal harm categories [a, b].
- The Q-EE component is empirically useful but lacks a deeper explanation of why quantum-inspired embeddings outperform standard high-dimensional mapping.
- The pipeline relies heavily on OCR quality; noisy or stylized text could degrade performance and reduce robustness.
- Although rationales are generated, the paper provides limited analysis of whether these explanations are reliable, faithful, or helpful for real moderation workflows [c].

[a] "Beneath the Surface: Unveiling Harmful Memes with Multimodal Reasoning Distilled from Large Language Models." The 2023 Conference on Empirical Methods in Natural Language Processing.
[b] "Pro-cap: Leveraging a frozen vision-language model for hateful meme detection." Proceedings of the 31st ACM international conference on multimedia. 2023.
[c] "Towards explainable harmful meme detection through multimodal debate between large language models." Proceedings of the ACM Web Conference 2024. 2024.

### Questions
- Can the authors provide stronger evidence that Q-EE captures meaningful “quantum-like” interactions rather than simply acting as a high-dimensional projection layer?
- To what extent might DACAM’s limited scope introduce dataset bias, and how would the model behave on broader abusive-meme domains like GOAT-Bench?

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
The paper makes a socially important contribution by focusing on child-abusive memes and providing a dedicated dataset and a detect-explain pipeline. The quantum-inspired embedding enhancement also appears to yield consistent, though modest, gains across several LLM backbones, and the empirical sweep on a single dataset is relatively thorough. However, several high-impact issues remain unresolved: the entire evidence base is on one small, in-house dataset with no cross-benchmark validation; there is no comparison to established multimodal meme/hate detectors, so the paper’s position in the literature is unclear; the reported gains are not backed by statistical significance or multi-seed reporting, which weakens the Q-EE claim; and dataset sourcing/release details are too loose for a sensitive domain. This paper will have much higher chances if it is submitted to a dataset track.

### Strengths
S1: This paper addresses a high-impact, underserved harm category (child-abusive memes) with clear societal relevance and links to real moderation needs.

S2: Introduces a dedicated, IRB-reviewed dataset (DACAM) specifically for child-abusive meme detection, with balanced abusive/non-abusive samples and annotated modality information.

S3: Shows that the quantum-inspired embedding enhancement consistently improves both classification F1 and explanation quality across several open LLM backbones.

S4: Provides a comparatively broad empirical sweep on the dataset (zero-shot, few-shot, fine-tuning; text-only vs multimodal; multiple ablations), which strengthens the evidence for the design.

S5: Includes human evaluation on explanations (fluency, consistency, informativeness) over 200 abusive memes, supporting the interpretability claim.

### Weaknesses
W1: Results are shown only on a single, relatively small in-house dataset (2,103 memes), so it is hard to tell how well the method would transfer to broader meme/hate benchmarks or real-world distribution shifts, even though the dataset itself is well curated.

W2: Despite the multimodal design that pulls in image, OCR, and title text, the claimed robustness to incomplete modalities is not actually stress-tested; the dataset has no image-only cases and overlapping text modalities, so we cannot see performance under genuinely missing inputs.

W3: Even though the quantum-inspired embedding enhancement component improves scores across several backbones, the paper does not provide a strong classical control to prove that the gains come from the “quantum-inspired” mechanism rather than from a generic non-linear projection.

W4: Lacks comparisons to established multimodal hateful-meme or harmful-content models/datasets (e.g., Hateful Memes, SemEval/MAMI-style tasks), which makes the positioning of the approach within existing literature unclear.

W5: Although a human evaluation of explanations is provided, the section is under-specified (annotator profiles, agreement, protocol), which weakens the strength of the interpretability claim.

W6: The overall pipeline is fairly heavy (CLIP + OCR + LLM + Q-EE + explanation); without an inference-time or resource/latency analysis, it is unclear whether this otherwise practical detect-explain design can be deployed in real moderation settings.

W7: Data collection and release details are only loosely described, even though the dataset is IRB-reviewed, sources, licensing, and handling of potentially illegal CSAM-like material are not spelled out, making reproduction and safe sharing harder.

### Questions
Q1: Is there a reason why no other harmful meme datasets are benchmarked?

Q2: Can the authors provide more information about the human evaluation? Expand the human-evaluation section with annotator profiles (number, background), agreement measures, task instructions, and an example rubric, so readers can assess the reliability of the 3.48–3.55 scores.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a database of memes that relate to or allude to child abuse messaging. The dataset also contains 50% non-abusive memes. The paper then presents a method for detecting such memes and compares the results to several LLM-based baselines. However, I think the two parts of the paper are not very well combined -- I see no reason to use this very specific methodology in this specific context.

### Strengths
- Important topic in the context of data moderation.
- Considerable manual work relating to the dataset creation.

### Weaknesses
- I am not sure the embedding present is superior to the embeddings of the picture, or even for ingesting the picture directly (in a VLM model). The comparison only shows improvement compared to a text-only baseline. This is likely because some of the signal comes from the photo, but the specific methodology presented isn't validated by the experiments presented.

- I would have liked to see how the accuracy changes if the dataset is added to a larger meme dataset, showing a more challenging, yet more realistic setting, in which this type of meme is just one type of abusive memes to be detected and taken down. 

- In the context of ethics, I would like the authors to discuss more of how they see their work being used.

### Questions
Can you improve the paper in relation to the weaknesses pointed out above?

### Soundness
2

### Presentation
3

### Contribution
2
