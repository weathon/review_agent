# Gaze-VLM: Eye-Tracking Informed Multimodal Large Language Models

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 4, 4

## Abstract
Vision-Language Models (VLMs) have demonstrated remarkable capabilities in visual question answering (VQA), yet they often struggle with referential ambiguity when multiple objects in an image could satisfy a given query. To address this challenge, we present Gaze-VLM, a novel training-free approach that uses eye-tracking data in real-time as an external alignment signal to resolve ambiguity in open-ended VQA. Through a comprehensive user study with 500 unique image-question pairs, we demonstrate that fixations closest to the time participants start verbally asking their questions are the most informative for disambiguation in Multimodal Large Language Models (MLLMs), more than doubling the accuracy of responses on ambiguous questions (from 35.2\% to 77.2\%) while maintaining performance on unambiguous queries. We evaluate our approach across state-of-the-art VLMs, showing consistent improvements when gaze data is incorporated in ambiguous image-question pairs, regardless of architectural differences. To facilitate future research in gaze-informed VQA, we release a new benchmark dataset to use eye movement data for disambiguated VQA, a novel real-time interactive protocol, and an evaluation suite. Our findings demonstrate that human visual attention signals can effectively guide VLMs toward intended referents in ambiguous contexts without requiring model retraining or architectural changes.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes GAZE-VLM: a framework that enhances Vision-Language Models (VLMs) by incorporating real-time eye-tracking data to resolve referential ambiguity in vqa. When users ask ambiguous questions about images with multiple similar objects, Gaze-VLM captures these fixations and overlays them on the image as visual cues for the VLM, guiding it to the correct object without training model.

### Strengths
- The temporal analysis of gaze provides insights to the gaze behaviour regarding question asking.
- The paper shows that if a question is paired with human gaze when the question was asked, VLMs can accurately judge if the answer is correct or not.

### Weaknesses
- The setting now requires real-time gaze to solve the ambiguity, i.e. external eye tracking devices and the speech onset timestamps are needed. This means for a dataset without gaze and question pair, it will not work. The proposed framework can't generalize to other vqa datasets.
- The prompt for accuracy evaluation is seriously flawed, it contains the answer: 'the same image with the referent object in question indicated by a white X sign'. This creates information leakage in evaluation.
- The baselines only contain VLMs, no other vqa methods.
- There are no ablation studies on human gaze integration. For instance, using a cropped image around gaze to query VLMs or using segmentation/bounding boxes the gaze fixated on.

### Questions
- Why not directly get ground truth from participants? This would be the most accurate answer.
- From a more practical point of view, if the human asks questions to an AI assistant, he/she can avoid ambiguity by asking question without referential ambiguity.
- "What is that?" is more like ambiguity in the question, not ambiguous referent. For example, three completely different objects in the scene with the question What is that.

### Soundness
1

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the problem of referential ambiguity in Vision-Language Models (VLMs), where a model cannot identify the correct object of interest when a query could apply to multiple items in an image. The proposed solution, Gaze-VLM, is a training-free framework that uses real-time human eye-tracking data as an external signal to resolve this ambiguity. The core method involves capturing a user's gaze fixations within a critical time window around speech onset and providing this information to the VLM at inference time.
Based on a comprehensive user study, the authors report that this approach more than doubles accuracy on ambiguous questions (from 35.2% to 77.2%) across 10 state-of-the-art VLMs, without requiring architectural changes or retraining. The work's primary contributions are the method itself, a temporal analysis pinpointing the most informative gaze signals, and a new benchmark dataset for gaze-informed VQA.

### Strengths
- The paper's primary originality lies in its novel, training-free, and real-time approach to resolving referential ambiguity in VQA by leveraging human gaze. Unlike prior work that often requires architectural modifications or model retraining, this method operates at inference time as an external alignment signal, making it a creative and "plug-and-play" solution. Furthermore, the systematic investigation of the temporal dynamics of gaze, which empirically identifies the time window around speech onset as the most informative, is a novel and valuable contribution that provides a theoretically-grounded insight for interaction design.

- The quality of the work is exceptionally high, anchored by a comprehensive user study that generated a new, high-fidelity dataset of 500 synchronized speech, gaze, and image pairs. This bespoke data collection is a significant strength, ensuring perfect alignment with the research question. The empirical results are compelling: more than doubling the accuracy on ambiguous questions (from 35.2% to 77.2%) is a substantial improvement. The evaluation is made robust by testing across 10 diverse, state-of-the-art VLMs, demonstrating the method's architectural agnosticism and generalizability. Finally, the detailed qualitative and quantitative error analysis adds a layer of scientific rigor.

### Weaknesses
- The method's success hinges on the precise temporal alignment of high-quality gaze data with the detected speech onset. While this is achievable in a controlled lab setting, real-world applications (e.g., on a mobile AR device) will face challenges from noisy audio environments (affecting voice activity detection) and less accurate eye-trackers. The paper does not evaluate the method's robustness to such noise.

- The approach relies on a spatial filter based on the median of fixations within the temporal window, assuming a single, compact point of interest. This may fail when a user scans a larger object or fixates back and forth between two objects of interest, as the median could fall on an irrelevant location. The paper notes this leads to zero retained fixations in 10.8% of trials, which is a non-trivial failure rate for the core filtering mechanism.

### Questions
Plz answer my concerns in the weakness section.

### Soundness
2

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
4

### Summary
The Gaze-VLM method proposed in this paper addresses the problem of referential ambiguity faced by Vision-Language Models (VLMs) in Visual Question Answering (VQA). It innovatively introduces real-time eye-tracking data as an external alignment signal without requiring model training or architectural modifications, making the research entry point highly practically significant. The paper verifies the effectiveness of the method through systematic experimental design (500 image-question pairs, validated on 10 mainstream VLMs). The overall research logic is complete, and the experimental data are relatively solid. It provides new ideas for the field of VQA ambiguity resolution and has certain academic value and application potential.

### Strengths
1. Accurately capture the core pain point of VLMs in multi-object scenarios - referential ambiguity (such as the inability to locate target objects in queries like "What is that"), and introduce human eye movement as a natural behavioral signal, which is consistent with the theoretical basis in cognitive science that "eye movements are associated with linguistic intentions".
2. The paper adopts a "no-training" design, which does not require modifying the parameters or architecture of existing VLMs. It can improve performance only by inputting gaze data in real-time. Experiments have verified that this method can achieve accuracy improvement on 10 SOTA VLMs with different architectures, with outstanding compatibility and generalization, reducing the threshold for practical applications.

### Weaknesses
1. Only 10 subjects were recruited, all of whom were students aged 19 to 26 (with an average age of 22), resulting in a lack of demographic diversity. People of different ages, occupations, and visual experiences may have different eye movement patterns. The existing sample may limit the generalization of the results and make it difficult to cover a wider range of actual user scenarios.
2. Regarding the "Eye Data Error" (occurring in 11.6% of Image + Gaze trials), only the statement "retaining all fixation points after duration filtering can improve accuracy by 1.5%" is mentioned, without an in-depth analysis of the causes of the error—whether it is due to the participants' scattered gaze (such as looking away from the target when asking questions), unreasonable parameter settings of the filtering algorithm (spatiotemporal filtering), or fluctuations in the correlation between fixation points and speech intent. There is a lack of discussion on targeted optimization directions.
3. Semantic similarity calculation uses OpenAI's sentence encoder, but the specific version of the encoder is not specified. The embedding effects of different versions vary, which may affect the reproducibility of results; the 88% consistency between the automatic evaluator and manual evaluation does not specify the sample scope of manual evaluation—whether it covers all 500 image-question pairs or is a sample evaluation? If it is a sample, the sampling ratio and randomization method are not mentioned, and the credibility of the evaluation results needs more details to support.

### Questions
This paper proposes an innovative training-free method in the field of VQA disambiguation, which has high academic value and application potential. However, there are issues such as a single test sample and insufficient depth of error analysis. It is necessary to improve the completeness and generalization of the research by modifying and supplementing the above content.

### Soundness
3

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
3

### Summary
This paper addresses the challenge of **referential ambiguity** in Vision-Language Models (VLMs)  where a query might plausibly refer to multiple objects in an image (e.g., "What color is that car?" when multiple cars are present)

The authors propose **Gaze-VLM**, a novel **training-free** and **inference-time** approach that uses human eye-tracking data as a real-time signal to disambiguate the user's intent.

The primary contribution is demonstrating that this method drastically improves VQA accuracy on ambiguous questions (reportedly from **35.2% to 77.2%**) without harming performance on unambiguous ones. This improvement is shown to be consistent across 10 different SOTA VLMs, demonstrating its generality. The authors also contribute a new benchmark dataset and interactive protocol for gaze-informed VQA.

### Strengths
- **High Generalizability:** The method requires no architectural changes or model-specific fine-tuning The demonstration that it consistently improves performance across 10 diverse VLM architectures (including open and closed-source models) is a very strong result.

 - **Strong Cognitive Grounding & Analysis:** The paper doesn't just use gaze; it investigates *when* gaze is informative. The temporal analysis in Figure 5, which pinpoints the window around speech onset as being optimal provides a strong link to cognitive science literature and validates the design choices.

### Weaknesses
- **Small-scale user experiment and insufficient dataset richness**
The study's claims are based on a small sample size, involving only 10 participants and 40 ambiguous images. While this generated 500 unique image-question pairs, the low diversity of both subjects and stimuli limits the generalizability of the finding. Describing this as a "comprehensive user study" or a "rich dataset" is an overstatement.
- **Limited method innovation and lack of necessary ablation studies**
 The core method relies on a simple heuristic: overlaying white crosses on black circles to denote gaze and prompting the VLM to use these "white X signs".This is an intuitive first step, but its optimality is not proven. The paper lacks a necessary ablation study comparing this visual prompting choice to other plausible alternatives, such as using heatmaps, drawing bounding boxes around the median gaze cluster, or feeding gaze coordinates directly as text.
- **Marginal gain of the complex temporal filtering**
  The temporal analysis in Figure 5A shows that the "optimal" filtered window (peak similarity $\approx 0.65$) performs only marginally better than the simple "All Fixations" baseline (similarity $\approx 0.615$). This small performance gap calls into question the necessity of the complex speech-onset synchronization component.This component adds system complexity (requiring real-time voice activity detection) , while simply aggregating all fixations might be a sufficiently powerful and much simpler heuristic.

### Questions
1. The paper uses Gemini-2.5-Flash as an automated evaluator, reporting an 88% agreement with human judgment. This 12% error rate is not low. If the 35.2% (Image Only) result is manual and the 49.7% (Image Only) result from Table 1 is automated18181818, this implies the automated evaluator is *significantly* inflating the baseline performance. Have you considered using a more powerful model for evaluation and quantifying the margin of error this automated evaluation introduces?
2.  The appendix reports an "Eye Data Error" rate of 11.6%. This is defined as cases where the VLM responds correctly given the "perfect" mouse-click LOI, but fails when given the filtered gaze data. This failure rate seems high, suggesting the filtering heuristic cannot capture user intent in over 1 in 10 cases. Have you analyzed the root causes for these failures (e.g., is the gaze too diffuse, is the speech onset detection inaccurate, or is the median filter a poor choice) ? How could the filtering method be improved to reduce this error?
3. Can you provide a stronger justification for using "white crosses on black circles" as the method to feed gaze data to the VLM? As raised in the weaknesses, have you experimented with alternative representations, such as heatmaps or textual coordinate inputs?
If my problem would be addressed, I would consider to raise my score.

### Soundness
2

### Presentation
3

### Contribution
2
