# Modal Aphasia: Can Unified Multimodal Models Describe Images From Memory?

- Decision: Accept (Poster)
- Scores: 4, 2, 6, 8, 6

## Abstract
We present *modal aphasia*, a systematic dissociation in which current unified multimodal models accurately memorize concepts visually but fail to articulate them in writing, despite being trained on images and text simultaneously. For one, we show that leading frontier models can generate near-perfect reproductions of iconic movie artwork, but confuse crucial details when asked for textual descriptions. We corroborate those findings through controlled experiments on synthetic datasets in multiple architectures. Our experiments confirm that modal aphasia reliably emerges as a fundamental property of current unified multimodal models, not just as a training artifact. In practice, modal aphasia can introduce vulnerabilities in AI safety frameworks, as safeguards applied to one modality may leave harmful concepts accessible in other modalities. We demonstrate this risk by showing how a model aligned solely on text remains capable of generating unsafe images.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper analyzes the phenomenon of modal aphasia, where unified multimodal LLMs models may be able to illustrate concepts that they fail to accurately describe in text. The paper systematically demonstrates the existence of this phenomenon in both proprietary and open-source multi-modal LLMs, as well as showing a consequence of this phenomenon in creating vulnerabilities in multi-modal LLMs aligned to prevent harmful outputs.

### Strengths
The concept of modal aphasia is intuitive and cleverly formulated, and well-motivated by connections to studies of visual and verbal cognition in humans and phenomena such as aphantasia.

It is an interesting and potentially surprising finding that unified multimodal LLMs taught new visual concepts fail to describe them accurately in text. It is clear how this is a drawback of these models and important to understand in order to mitigate such misalignments. The connection to work on visualizing concepts for reasoning is also interesting.

The experiments on open-weights models are appreciated for reproducible science.

Overall the paper is clear and well-written.

### Weaknesses
I find the motivation from frontier models (L41–47) and tests on GPT-5 (Sec 3) unconvincing, because it is likely that GPT-5 routes image generation requests to a separate image generation model rather than specifically training the LLM and image generator in parallel. The [GPT-5 system card](https://cdn.openai.com/gpt-5-system-card.pdf) only mentions GPT-5 accepting textual or image input, but noticeably does not discuss image generation.

It’s not obvious to me that performance when fine-tuning on a new visual concept would be comparable to zero-shot evaluation of the pretrained model’s knowledge of visual concepts. It could be that fine-tuning on a single concept alone degrades the LLM’s general reasoning abilities and differs from the dynamics of pretraining. Why not also perform a similar zero-shot test to that in Sec 3 applied to open models?

I’m not sure if the effect demonstrated in Sec 5 is due to modal aphasia. Modal aphasia was previously defined (L32-33) as the inability to access knowledge in text which can be expressed in images, while Sec 5 discusses the case where a visual concept can still be described in text using a circumlocution.

Overall, I think there is a valuable conceptual and methodological contribution here, and I'm willing to reconsider my score if the significant points above are addressed.

### Questions
The paper focuses on modal aphasia where a concept can be illustrated but not described in text. Have you considered the reverse phenomenon? (Models being able to describe concepts in words that they struggle to illustrate.) Or whether this could generalize to other modalities which are now being processed by unified multimodal LLMs?

There is some evidence that images may provide supervision for language models that are trained multimodally, reflected in visual reasoning in text, beyond what is learned from text alone. [1–3] On the other hand, this paper’s results suggest that textual description of visual concepts is not readily learned from text-image pairs. How should we interpret these findings in this context?

Does the effect also hold when the model’s visual parameters are trainable? (vs. L298)

Should “ChatGPT-5” be “GPT-5”?

Figs 4–5 are missing a visualization of the baseline (random) performance, making it harder to visually interpret the results.

[1] Zhang et al. Visual Commonsense in Pretrained Unimodal and Multimodal Models. NAACL 2022

[2] Liu et al. Things not Written in Text: Exploring Spatial Commonsense from Visual Signals. ACL 2022

[3] Alper et al. Is BERT Blind? Exploring the Effect of Vision-and-Language Pretraining on Visual Language Understanding. CVPR 2023

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
4

### Summary
The paper introduces the concept of modal aphasia, where unified multimodal models accurately generate images yet fail to generate the same knowledge through text. The authors find that this phenomenon persists across different model architectures and training procedures. They further highlight its implications for AI safety: safety interventions applied to one modality do not reliably transfer to another (e.g., a model aligned in text may still produce harmful images).

### Strengths
* The paper addresses a fundamental question in unified multimodal modeling: whether such models can express the same knowledge consistently across modalities. 
* It also provides a comparative analysis of both proprietary and open-source models to support the generality of its claims.

### Weaknesses
* The experimental design is questionable. The authors train their unified multimodal models *only* on the image generation task—learning to produce images from text prompts. This setup naturally biases the model toward visual generation without ensuring it can express the same concepts in language, since it is not trained on captioning tasks. A more appropriate design would jointly train the model on both image generation and image captioning using the same image–text pairs, then compare its performance across modalities.

* The paper conflates modal aphasia with a prompt engineering loophole. Regarding safety risks (Section 5), the authors interpret reduced refusal on real words ("feet") vs. rare expression prompts ("secondary balance units") as evidence of modal aphasia. However, this may not necessarily be due to a modality gap. It may simply be that the text input pipeline is not robust to rephrasing / prompt engineering. Moreover, the experiment is based on a single example ("feet" vs. "secondary balance units"), making its findings too limited in scope to support a general claim.

* The assumption that modern unified architectures train image and language jointly "from scratch" is invalid. Section 3 reports initial experiments on ChatGPT, whose model details are undisclosed. For instance, the model may not be a single jointly trained vision-language architecture, but rather a combination of a separate vision-language model and an image generation model accessible through a unified chat interface. Furthermore, this statement does not hold for the two open-sourced models analyzed in Section 4, Janus-Pro and Harmon, neither of which is trained jointly from scratch. Harmon combines an image generation model (MAR) and LLM (Qwen2.5-1.5B-Instruct), which are trained separately on large-scale datasets describing distinct modalities and concepts that far exceed the scale of data used for vision–language alignment. The same applies to Janus-Pro, which combines separately trained components (DeepSeek-LLM and SigLIP). Consequently, the image and language components capture different visual and linguistic concepts, making modal aphasia an expected, not novel, outcome.

### Questions
The paper poses a very interesting question, but the experiments are poorly designed and lack generalizability. As mentioned in the weaknesses, it would be important to examine what happens when the models are trained on both image generation and captioning tasks using the same image–text pairs. Would the same trend persist under this training setup?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper first points out that current unified multimodal models exhibit a systematic dissociation, termed modal aphasia, where they can generate highly accurate visual content but fail to access the same knowledge through text queries. To address this, the paper proposes a series of controlled experiments using open-weight unified models to analyze how this dissociation emerges across architectures and training setups. The study aims to reveal the inherent limitations of current multimodal knowledge transfer mechanisms and highlight the potential safety risks of modality-specific alignment, providing insights for developing models with genuinely unified cross-modal understanding.

### Strengths
* The paper introduces the new concept of modal aphasia, identifying a systematic dissociation between visual and textual understanding in unified multimodal models. This is an original and theoretically significant contribution that reframes existing assumptions about cross-modal knowledge transfer.

* The authors demonstrate modal aphasia not only in frontier models such as ChatGPT-5, but also in controlled experiments with open-weight models (Janus-Pro, Harmon). This dual-level validation strengthens the robustness and generality of the findings.

* The authors release code, data, and detailed experimental procedures, enhancing transparency and reproducibility. Their unified rubric-based evaluation also provides a standardized way to measure multimodal consistency.

### Weaknesses
* The paper successfully identifies modal aphasia as a systematic failure of multimodal models, but it does not offer a clear theoretical explanation or model-level mechanism to account for this behavior. The contribution remains largely descriptive rather than explanatory.

* Most experiments are conducted on controlled synthetic datasets such as fictional faces and geometric patterns. While these setups enable variable control, they do not demonstrate whether modal aphasia persists in realistic multimodal tasks such as captioning or retrieval. This limits the external validity of the findings.

* Although the authors link modal aphasia to potential safety risks, the provided evidence is based on a single case study involving a narrow concept category. The safety implications would be more convincing if supported by broader empirical evaluation across multiple harmful concept types or adversarial prompt settings.

* The conclusion suggests that allowing models to visualize concepts during reasoning may mitigate modal aphasia, but the paper does not propose any concrete implementation or experimental verification of this idea. This limits the practical contribution of the work.

I would like to discuss these points with the authors during the rebuttal stage and will adjust my score based on their responses and the feedback from other reviewers.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper investigates a problem in MLLMs called Modal Aphasia, a phenomenon that MLLMs can accurately memorize concepts visually but fail to articulate them in writing. This work starts with a case study of poster generation with GPT5 to reveal this problem. And then created two synthetic datasets to further validate the existence of this issue. Related AI safety concerns are raised with a case study.

### Strengths
S1: This paper is clearly written and easy to follow

S2: The found problem of modality imbalance in image/text generation fidelity is of great importance, and the naming is fun and accurate.

S3: The experiments to quantify and validate modal aphasia are well designed

### Weaknesses
W1: This work focuses on the modality imbalance problem of MLLMs in image/text generation. There are related studies/benchmarks in image/text understanding about modality imbalance in VLMs, which are worth discussing to better position this work.

W2: Since GPT5 is a proprietary model, there are rumors that its image generation is routed through another “sub-model” of GPT5. If so, the modality imbalance problem in image/text generation is kind of expected because of such a mismatch. I would like to know the authors’ thoughts on this.

### Questions
Please refer to W2.

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents an interesting phenomenon called modal aphasia, where leading unified multimodal models can faithfully generate almost perfect visual outputs but fail to describe details and concepts verbally. The authors demonstrate modal aphasia through observational experiments on GPT-5 and controlled fine-tuning on open-weight models. They also conduct a targeted case study to demonstrate bypassing safeguards by exploiting the modal aphasia.

### Strengths
This paper is well written. It starts with a clear motivation that derives from our daily interactions with commercial multimodal models. Then, the authors employ an interesting study of generating visually vs. describing verbally upon movie posters in frontier models, showing the prevalence of the modal aphasia. Beyond observational experiments, the authors design crisp synthetic data and fine-tuning experiments to show that modal aphasia can stem from more than just naive image memorization, but a systematic discrepancy of knowledge and concept understanding across modalities. Finally, they exploit a harmful use case if modal aphasia is not properly addressed. Overall, this paper is coherent and a joy to read.

### Weaknesses
1. The controlled experiments on open-source models only examine two open-source image generators. Also, the scale of test data is relatively small (below 200).

2. In the controlled fine-tuning for tracing the origin of modal aphasia, activating only the LLM backbone while freezing other components may not reflect real-world training.

### Questions
See Weaknesses. Also,
1. I am curious to see what happens with chain-of-thought prompting in these unified multimodal models? For example, instead of describing features in text independently, what will happen if the model is explicitly asked to "visualize then describe"?

2. The definition of modal aphasia sounds relevant to modality imbalance, which has already been well-identified and mechanistically understood in quite a few previous works. How do the authors view the similarity and difference compared with modality imbalance in general VLMs or multimodal models?

In general, the paper’s claims and findings are self-contained, and weaknesses are relatively minor. I’d be happy to raise the score if the questions are adequately addressed.

### Soundness
3

### Presentation
4

### Contribution
3
