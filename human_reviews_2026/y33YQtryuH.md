# Fake-in-Facext: Towards Fine-Grained Explainable DeepFake Analysis

- Decision: Reject
- Scores: 6, 6, 6, 2

## Abstract
The advancement of Multimodal Large Language Models (MLLMs) has bridged the gap between vision and language tasks, enabling the implementation of Explainable DeepFake Analysis (XDFA). However, current methods suffer from a lack of fine-grained awareness: the description of artifacts in data annotation is unreliable and coarse-grained, and the models fail to support the output of connections between textual forgery explanations and the visual evidence of artifacts, as well as the input of queries for arbitrary facial regions. As a result, their responses are not sufficiently grounded in Face Visual Context (Facext). To address this limitation, we propose the Fake-in-Facext (FiFa) framework, with contributions focusing on data annotation and model construction. We first define a Facial Image Concept Tree (FICT) to divide facial images into fine-grained regional concepts, thereby obtaining a more reliable data annotation pipeline, FiFa-Annotator, for forgery explanation. Based on this dedicated data annotation, we introduce a novel Artifact-Grounding Explanation (AGE) task, which generates textual forgery explanations interleaved with segmentation masks of manipulated artifacts. We propose a unified multi-task learning architecture, FiFa-MLLM, to simultaneously support abundant multimodal inputs and outputs for fine-grained Explainable DeepFake Analysis. With multiple auxiliary supervision tasks, FiFa-MLLM can outperform strong baselines on the AGE task and achieve SOTA performance on existing XDFA datasets. The code and data will be made open-source.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes an MLLM with multiple input streams and multiple outputs (natural language, RGB and segmentation/bounding box) for deepfake detection, localisation and explanability. To achieve this, the paper first proposes an annotation method and creates a 1+M image dataset, with fine-grained annotations and hierarchical labels. Then, to standardise and ensure safety, a benchmarking for the new task is proposed. Finally, an MLLM is trained on the task using the proposed dataset and evaluated against the baseline, and evaluated on the proposed benchmark and DDVQA.

### Strengths
The paper is well written and easy to follow with little, if any, ambiguity. The problem identified is largely with existing datasets (i.e. they largely have binary labels), and the authors fix it by proposing a new annotation method, constructing a very large-scale dataset and showing that MLLMs can adequately learn and perform in the downstream tasks of deepfake detection and localisation. Contrary to previous works (eg, DDVQA, which is the main comparison in the paper), the MLLM has multiple outputs to enhance natural language explanations, which is both novel in the task using MLLMs and intuitive (natural language and visual explanations are complementary).

### Weaknesses
The main weakness identified is the lack of experiments on previously established and standard deepfake detection datasets (eg, FF++, DFDC, Wild DeepFake, etc), and subsequent comparison with non-MLLM methods on these. This is essential both for context and to assess the generalisation capabilities of the proposed MLLM, particularly as the explanation in sec. 2.2, step 1 implies that the forgery methods used are rather limited.

Some minor comments:
- Table 4 is unclear whether the MLLM is trained from scratch on DDVQA, fine-tuned or if this is cross-dataset performance.
- In the Multi-task processing, the sections describing samples are confusing: "For the sample Det., T_in is the detection embedding [...]" -> this and the subsequent sentences are not making a lot of sense; it would be best to rephrase for clarity.
- In method 2.2, step 2, again, the explanation for atomic/parent concepts is a little unclear. Consider maybe breaking down and expanding on the conditions for clarity.
- Some Qualitative examples and Human evaluation results on the natural language output would strengthen the paper

### Questions
- What is the cross-dataset and zero-shot performance of the model on standard DF detection datasets?
- Are the Table 4 results cross-dataset, finetuned or trained from scratch?
- How do humans rate the natural language output of the MLLM?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents Fake-in-Facext (FiFa), a comprehensive framework for fine-grained Explainable DeepFake Analysis (XDFA). The authors introduce a Facial Image Concept Tree (FICT) to model facial regions for fine-grained artifact localization, an automated annotation pipeline to generate image-text pairs with explanations, and a unified multi-task model FiFa-MLLM for joint predictions of textual forgery explanations and segmentation masks. Extensive experiments show that FiFa-MLLM achieves state-of-the-art performance in artifact grounding and localization.

### Strengths
1. The work makes progress toward fine-grained explainability in deepfake analysis through textual and visual artifact reasoning.

2. The hierarchical concept tree and annotation pipeline are well-motivated and improve annotation precision.

3. The unified architecture for multimodal framework is well-designed, which integrates multi-task learning without requiring multiple encoders.

4. Experimental results across multiple datasets demonstrates the framework’s effectiveness and data reliability.

### Weaknesses
Major Weaknesses

1.	The semantic consistency between generated textual explanations and segmentation masks is not deeply validated. While the multi-task decoders output both modalities, the linguistic and visual results may be unaligned.

2.	The FiFa-Annotator pipeline heavily relies on GPT-4o and ChatGPT for generating explanations, which may introduce linguistic or conceptual bias. The authors claim reliability improvement via prior knowledge but do not quantify annotation quality beyond model performance.

3.	The qualitative results for Artifact-Grounding Explanation (AGE) are not extensively illustrated. The paper would benefit from more visual samples showing how textual and segmentation outputs correspond at different facial content.

Minor Weaknesses

1.	Some concept names (e.g., “Facext”, “Artifact-Grounding Explanation”) can be clearly defined for better readability.

2.	The introduction section could clarify the novelty of FiFa-MLLM relative to prior segmentation-aware MLLMs.

### Questions
1. How do the authors verify that textual explanations and segmentation masks are semantically aligned?

2. How is the annotation quality from GPT-4o and ChatGPT controlled or measured in FiFa-Annotator?

3. Could the authors add more visual examples to show text–mask correspondence in the AGE results?

4. How does FiFa-MLLM differ architecturally from prior segmentation-aware MLLMs such as LISA[1] or GLaMM[2]?

[1] Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, and Jiaya Jia. LISA: reasoning segmentation via large language model. In CVPR, pp. 9579–9589. IEEE, 2024.

[2] Hanoona Abdul Rasheed, Muhammad Maaz, Sahal Shaji Mullappilly, Abdelrahman M. Shaker, Salman H. Khan, Hisham Cholakkal, Rao Muhammad Anwer, Eric P. Xing, Ming-Hsuan Yang, and Fahad Shahbaz Khan. Glamm: Pixel grounding large multimodal model. In CVPR, pp. 13009–13018. IEEE, 2024.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes the Fake-in-Facext (FiFa) framework to address the lack of fine-grained awareness in existing multimodal large language models for explainable deepfake analysis. The framework introduces a hierarchical Facial Image Concept Tree with 112 atomic concepts for reliable data annotation, defines 11 tasks (FiFa-11) including a novel Artifact-Grounding Explanation task that interleaves textual explanations with segmentation masks, and constructs FiFa-Instruct-1M, the largest training dataset (1.38M samples) in the XDFA field.

### Strengths
1. The paper leverages multimodal large language models for deepfake annotation and considers diverse tasks including bounding box-level queries, which advances the explainability of deepfake detection. The multi-granularity task design (image-level, region-level, and box-level) enables fine-grained forgery analysis.

2. The authors contribute a datasetwith comprehensive annotations covering 11 different tasks, including novel artifact-grounding explanations that interleave textual descriptions with segmentation masks. This represents the largest training dataset in the explainable deepfake analysis field to date (1.38M QA-pairs).

3. The paper proposes a reasonable baseline model (FiFa-MLLM) for comprehensive evaluation. The unified architecture with a single visual encoder for both LLM input and mask prediction demonstrates efficiency, and the ablation studies validate the effectiveness of key components such as auxiliary supervision tasks.

### Weaknesses
1. The paper only considers attribute manipulation techniques (primarily FaceApp) for creating fake samples, excluding other common deepfake types such as identity swapping, expression swapping, and entire face synthesis in the data annotation pipeline (FiFa-Annotator). This narrow focus on a single forgery method may hinder the model's generalization capability to diverse deepfake techniques encountered in real-world scenarios.

2. The data generation approach (using masks + large language models) appears similar to the CVPR 2025 paper "Towards General Visual-Linguistic Face Forgery Detection", which also analyzes fine-grained facial features and uses them for large model context learning. While the proposed Facial Image Concept Tree provides more detailed annotations (112 atomic concepts vs. coarser regions), it remains unclear whether this increased granularity translates to meaningful performance gains. I recommend the authors conduct a direct comparison with this method, given the similar annotation paradigms, to demonstrate the added value of the fine-grained concept hierarchy.

3. The evaluation is relatively limited and narrow in scope. The experiments primarily focus on performance on the authors' own dataset (FiFa-Bench) with limited baseline comparisons (mainly GLaMM). Critically, there is insufficient evaluation of generalization performance on out-of-distribution datasets or real-world deployment scenarios. Given the narrow data range (only FaceApp for training AGE/TOE/Loc tasks), I have significant concerns about whether the proposed method can generalize to other deepfake generation techniques, diverse manipulation methods, or in-the-wild forgery detection. Additional cross-dataset evaluation (e.g., on FaceForensics++, Celeb-DF, or DFDC) would strengthen the paper's claims.

4. The paper claims SOTA results on DD-VQA and DFA-Bench (Tables 4-5), but these improvements are relatively modest. More diverse baseline comparisons with recent XDFA methods (FFAA, FakeShield, etc.) would better contextualize the contributions.

Despite the aforementioned issues, I acknowledge that the authors' proposed comprehensive multimodal evaluation framework, particularly the inclusion of bounding box-level regression tasks, makes a meaningful contribution to this field. I hope the author can address my concerns.

### Questions
1. Why 112 atomic concepts specifically? No ablation on the granularity of FICT (e.g., 50 vs. 112 vs. 200 concepts)

2. This paper replacing CLIP+SAM encoders with a single FaRL encoder. It's unclear whether improvements come from the unified encoder design, face-specific pretraining (FaRL), or simply better training data

3. Table 6 shows region mask prediction helps, but: Improvements are relatively modest (Loc: 30.0→30.9 mIoU, AGE: 30.0→30.4 mIoU)
Det/Cls auxiliary tasks actually hurt some metrics (TOE METEOR drops from 21.3 to 20.9 without region mask)

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper introduces a benchmark that moves beyond simple detection to fine-grained, explainations. The authors propose an automated pipeline, FiFa-Annotator, to generate a large-scale dataset where forged facial images are annotated with detailed textual explanations and corresponding segmentation masks that pinpoint the manipulated regions. They also introduce FiFa-MLLM, a unified multimodal model designed to perform a new "Artifact-Grounding Explanation" task, which involves generating natural language descriptions of forgeries that are visually grounded by these masks.

### Strengths
- The paper makes a commendable attempt to advance DeepFake analysis beyond simple binary real vs. fake detection to fine-grained, explainable localization and description.
- The proposed FiFa-MLLM architecture is a thoughtful effort to create a unified, multi-task model that can handle diverse inputs like bounding box queries.

### Weaknesses
- FiFa is designed specifically for DeepFakes created using "attribute manipulation" techniques. The authors state that this is because other methods, like identity or expression swapping, create pixel-level changes that are not localized to the artifact, making their artifact detection method unreliable. This limits the training data for fine-grained explanation to a single class of forgery. In fact there are several methods in the literature that can detect and localize identity or expression swaps. So, if existing methods can detect and/or localize them, making these annotations a part of the benchmark is crucial.
- The annotations are generated using proprietary models like GPT-4o. This assumes that GPT-4o is a good DeepFake detector and reasoning model, which is wrong. MLLMs are inherently built for global semantics rather than capturing subtle inconsistencies that occur in DeepFakes. Papers like [1, 2] show that subtle changes are ignored, and these models are broadly good in overall content understanding. This means that the annotations are highly unreliable and noisy.
- Artifact masks are created by identifying the top 5% of pixel intensity differences between a real and fake image. This heuristic may not be robust enough to capture subtle manipulations or forgery types that involve smoother, less distinct alterations. The choice of a fixed 5% threshold is also not justified and may be suboptimal. If pixel intensity differences were enough for DeepFake detections, there would not be a need for complicated pipelines for DeepFake detection that are present in the literature.
- Since the FiFa-MLLM is trained and evaluated on a benchmark (FiFa-Bench) created by its own FiFa-Annotator pipeline, there is a risk that the model is learning the specific patterns of the annotation process rather than generalizable features of DeepFakes, through a "shortcut" learning method. There are no qualitative results that show that the generated explanations are actually grounded in visual cues.

[1] Tong, Shengbang, et al. "Eyes wide shut? exploring the visual shortcomings of multimodal llms." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.
[2] Huynh, Ngoc Dung, et al. "Vision-Language Models Can't See the Obvious." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2025.

### Questions
- Why was a fixed 5% threshold chosen for creating artifact masks? Were alternative, more adaptive methods for identifying manipulated regions explored? And why does it make sense to flag artifacts solely based on this concept of pixel intensity?
- What specific methods were used to verify the factual accuracy of the forgery descriptions generated by GPT-4o? How was the claim of achieving "fewer hallucinations" quantified?
- How well does the model perform its primary AGE task on forgeries it was not trained on for this task, such as identity swaps, expression swaps, or entire face synthesis, i,e, in cross-data settings?
- The addition of auxiliary supervision for region masks improved localization but slightly hurt text generation scores. Could the authors elaborate on this potential trade-off between visual grounding accuracy and textual explanation quality?

### Soundness
1

### Presentation
3

### Contribution
1
