# How Well Can General Vision-Language Models Learn Medicine By Watching Public Educational Videos?

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 2, 4, 6

## Abstract
Publicly available biomedical videos, such as those on YouTube, serve as valuable educational resources for medical students. Unlike standard machine learning datasets, these videos are designed for human learners, often mixing medical imagery with narration, explanatory diagrams, and contextual framing. In this work, we investigate whether such pedagogically rich, yet non-standardized and heterogeneous videos can effectively teach general-domain vision-language models biomedical knowledge. To this end, we introduce OpenBiomedVid, a biomedical video instruction tuning dataset comprising 1031 hours of video-caption and Q/A pairs, curated through a multi-step human-in-the-loop pipeline. Diverse biomedical video datasets are rare, and OpenBiomedVid fills an important gap by providing instruction-style supervision grounded in real-world educational content. Surprisingly, despite the informal and heterogeneous nature of these videos, the fine-tuned Qwen-2-VL models exhibit substantial performance improvements across most benchmarks. The 2B model achieves gains of 98.7% on video tasks, 71.2% on image tasks, and 0.2% on text tasks. The 7B model shows improvements of 37.09% on video and 11.2% on image tasks, with a slight degradation of 2.7% on text tasks compared to their respective base models. To address the lack of standardized biomedical video evaluation datasets, we also introduce two new expert curated benchmarks, MIMICEchoQA and SurgeryVideoQA. On these benchmarks, the 2B model achieves gains of 99.1% and 98.1%, while the 7B model shows gains of 22.5% and 52.1%, respectively, demonstrating the models' ability to generalize and perform biomedical video understanding on cleaner and more standardized datasets than those seen during training. These results suggest that educational videos created for human learning offer a surprisingly effective training signal for biomedical VLMs. We release OpenBiomedVid, MIMICEchoQA, SurgeryVideoQA, the fine-tuned models, and the complete codebase to support future research.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces OpenBiomedVid, a biomedical video-text dataset (1,031 hours) curated from YouTube educational videos, along with two expert-curated benchmarks—SurgeryVideoQA and MIMICEchoQA—for evaluating biomedical video understanding. The authors further fine-tune Qwen2-VL and InternVL3 models on this dataset, demonstrating improvements on both video and image benchmarks.

### Strengths
1. Clear presentation: The paper clearly describes the construction of instruction data, evaluation datasets, and model training procedures, making the methodology easy to follow.

2. Thoughtful data curation: The dataset creation process includes several reasonable and interesting design choices, such as leveraging SigLIP-Medical and Whisper models, to ensure the reliability and quality of the collected data.

3. Practical and novel contribution: While prior works have explored YouTube data for research, the focus on video instruction tuning and the introduction of corresponding evaluation benchmarks fills an important gap in the current biomedical multimodal landscape. The dataset and benchmarks are likely to be of practical use to the community.

### Weaknesses
1. Although the paper targets video-level instruction tuning, prior works (e.g., Quilt-1M) have already shown success with image-level instruction data in the medical domain. It remains unclear whether video-level supervision provides a substantial advantage over image-text data for medical VQA tasks. A valuable follow-up experiment could involve creating a subset of the dataset where temporal information is minimized (e.g., few-frame clips or frame-level QA) to empirically assess whether videos or images contribute to the performance gains.
2. In lines 343–346, the authors state that “performance on video benchmarks remains significantly lower than on text and image benchmarks.” However, the reported results show only modest improvements on text benchmarks (with MedQA even decreasing) and limited gains for PathVQA with the 7B model. The discussion should be more nuanced.
3. The dataset mainly focuses on videos, but I noticed certain performance improvements on the image-text datasets VQA-RAD and SLAKE. If possible, I encourage the authors to derive and release a medical image instruction tuning dataset from the existing collection. For example, by selecting clips with fewer frames or converting segments that do not require strict temporal encoding into multi-image samples, since many QA pairs may not rely on temporal information. This would further advance the development of medical vision-language models.

### Questions
See weaknesses. I may consider raising my score depending on the authors’ response, as I am genuinely interested in this work.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper investigates the potential of general-purpose vision-language models to be directed towards effective biomedical video understanding by training them on publicly available educational content, primarily sourced from YouTube. The authors have constructed a substantial instruction-tuning corpus, named OpenBiomedVid, which comprises approximately 1,031 hours of clinician-guided biomedical videos, meticulously cleaned captions, and GPT-assisted yet human-verified question-answer pairs. Additionally, they introduce two more challenging expert-curated benchmarks—MIMICEchoQA (echo) and SurgeryVideoQA (surgery)—to ensure that evaluations are not conducted on the same noisy distribution as the training data. Fine-tuning Qwen2-VL (2B/7B) using this dataset results in significant relative improvements in biomedical video question answering (QA) and notable advancements in image visual question answering (VQA). In some instances, performance approaches or even surpasses that of stronger general models when applied to echo-style videos. This finding suggests that "videos made for humans" can still serve as an effective supervisory signal for medical vision-language models. However, it is important to note that results on the surgical benchmark remain distinctly lower. This indicates that long and heterogeneous procedural videos continue to pose challenges and highlights concerns regarding stylistic inconsistencies and potential hallucination risks introduced by LLM-in-the-loop curation—a concern the authors attempt to address through human verification. In summary, this work presents a well-motivated dataset and benchmark package along with compelling empirical evidence supporting the notion that public educational videos represent a viable resource for domain adaptation of open vision-language models.

### Strengths
- Clear Motivation & Well-Defined Problem: The authors identify an important gap: most biomedical video datasets are either small, narrowly focused, or unsuitable for large-scale multimodal learning. They correctly observe that public educational videos are a massive, untapped source for such efforts.
- Novel Dataset and Benchmark Creation: The curation of a 1031-hour biomedical video-text dataset from public sources is commendable. The pipeline is systematic, leveraging both expert clinical input and multi-stage LLM filtering. The inclusion of 79,367 Q/A pairs and structured metadata adds substantial utility and diversity.
- Expert-Curated Evaluation: The introduction of the MIMICEchoQA and SurgeryVideoQA benchmarks is a valuable contribution, offering much-needed standardized, clinically relevant tests for biomedical VLMs. The expert review and curation of questions are a particular highlight.

### Weaknesses
The paper creates a same-model, same-style loop: GPT-4o is used to clean captions and generate the training/eval Q&A and metadata, and then the very same GPT-4o is used as the automatic judge to score open-ended answers (binary 0/1). The authors even note the risk of “stylistic bias” and only add a one-off check with Gemini-2.0-Flash, but GPT-4o remains the primary scorer. This tightly couples data generation and evaluation to one model’s style, making gains plausibly evaluation-protocol-driven rather than true capability. 

- Lack of evaluation against domain-specific medical VLMs. The paper compares mainly with general-purpose multimodal backbones (Qwen2-VL at different scales, InternVL3-8B, GPT-4o, Gemini-2.0-Flash, etc.) on the proposed biomedical video QA benchmarks, but does not report results for the most directly relevant medical instruction-tuned VLMs such as LLaVA-Med, Med-Flamingo (medical adaptation of Flamingo), or widely used medical multi-image models (e.g., MedCLIP), even though many of them are cited in Related Work. Because these models target clinical/biomedical vision-language understanding on VQA-RAD, PathVQA, SLAKE and related tasks, including at least a frame-based or image-only adaptation of them on the authors’ benchmarks would make the claimed gains over “prior medical VLMs” easier to judge. The current tables therefore make it hard to tell whether the improvement comes from the proposed video-centric data and pipeline or simply from using stronger general backbones. (We acknowledge that some models, e.g. MedCLIP or MedGemini, are image-centric or not fully open, but even a best-effort comparison or a discussion of feasibility would strengthen the evaluation.)
- Reliance on LLM-as-a-judge without human grounding. For the open-ended SurgeryVideoQA benchmark, the paper evaluates all models with GPT-4o as the automatic grader, assigning binary correctness against the reference answer. Since GPT-4o was also involved in earlier stages of caption refinement and QA generation, this creates a potential style / phrasing bias toward GPT-4o-like answers. The authors do run a second pass with Gemini-2.0-Flash and report that the ranking is broadly consistent, which is helpful, but both judges are frontier LLMs with unknown overlap in training data and alignment objectives, and no human adjudication, inter-rater agreement, or adversarial stress tests are provided. As a result, part of the reported gains on SurgeryVideoQA could still be influenced by judge-specific preferences rather than purely by better video understanding.
- Potential residual data-quality and provenance issues. Although the paper presents a multi-stage, human-in-the-loop curation pipeline (YouTube retrieval → GPT-labeled frame filtering with a fine-tuned SigLIP → GPT-4o caption refinement → GPT-4o Q/A generation → human verification) and even reports ~95% agreement on the frame-filtering stage, many of the quality controls are described only at a high level. In particular, the paper does not spell out how deep the human verification went (per-clip vs per-QA, random spot checks vs full passes), how many annotators were involved, or what the inter-annotator agreement was beyond the small sample quoted. Moreover, because GPT-4o is used both to clean captions and to synthesize Q/A pairs, typical LLM failure modes—overly generic answers, temporal misalignment with the actual video segment, or medical hallucination—are plausible but not systematically audited or reported. This matters especially because the paper itself shows that models trained on the noisy YouTube-derived corpus still perform noticeably worse on the cleaner, expert-curated benchmarks (MIMICEchoQA, SurgeryVideoQA), suggesting that remaining noise in the large-scale training split may be a limiting factor. A more explicit error analysis (e.g., failed frame segmentation, ambiguous captions, low-quality or hallucinated Q/A) would make the dataset contribution stronger.
- Underdeveloped video baselines on the proposed benchmarks. Because the core claim of the paper is about video-centric biomedical understanding, the experimental setup on MIMICEchoQA and SurgeryVideoQA would be more convincing if it included stronger, publicly available video–language models beyond the authors’own Qwen2-VL fine-tunes. At minimum, prior general-purpose video chat models (e.g., Video-ChatGPT, Video-LLaVA, PALIGemma-style video variants) could be run in a frame-sampling or short-clip regime to provide external points of reference, even if they are not domain-tuned. Their absence makes the reported gains look partly relative to the authors’ chosen baselines rather than to the broader video–language landscape, and it also hides whether the new benchmarks are genuinely “hard” for off-the-shelf video models or only for image-first VLMs.

### Questions
1. You cite several domain-specific multimodal/medical VLMs in Related Work (e.g., LLaVA-Med, Med-Flamingo, MedCLIP), but they do not appear in the main comparison tables. Can you clarify which of these models you actually attempted to run, and what prevented you from including their results?

2. Right now the improvements are mostly against Qwen2-VL/InternVL-style backbones. How can we be sure the gains come from your video-centric data/pipeline rather than simply from using a stronger base model?

3. The evaluation of SurgeryVideoQA was conducted using GPT-4o as the automatic grading system. Given that GPT-4o was also employed for the refinement of captions and question-and-answer pairs, what measures were taken to mitigate potential style bias towards outputs resembling those generated by GPT-4o?

4. You mention using Gemini-2.0-Flash as a second judge and finding similar rankings. Could you provide the agreement numbers between GPT-4o and Gemini on this benchmark?

5. You report ~95% agreement on frame filtering, but on how many samples, with how many annotators, and at which stage of the pipeline was this measured (video-level vs clip-level vs QA-level)?

6. Is human verification applied to every GPT-generated Q/A pair, or only to a sampled subset? If sampled, what was the sampling strategy and coverage?

7. Would it be feasible to run them in a constrained setting (e.g., fixed frame sampling, short 8–16 frame clips) just to position your benchmarks relative to the broader video-VLM ecosystem?

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
This paper investigates whether general-purpose vision-language models can effectively learn biomedical knowledge from publicly available educational videos on platforms like YouTube. It introduces OpenBiomedVid, a large-scale, diverse instruction-tuning dataset comprising 1,031 hours of biomedical video content, curated through a multi-step human-in-the-loop pipeline that filters frames, refines captions, and generates question-answer pairs. For evaluation, two new benchmarks are released: Surgery VideoQA and MIMICEchoQA.

### Strengths
The effort to create these resources is commendable. The field lacks large-scale biomedical video datasets, and OpenBiomedVid, along with the two new benchmarks, represents a significant contribution.

### Weaknesses
The paper is undermined by several major flaws in its methodology and results, which make the central claim (that the models are "learning medicine") unconvincing.

1.	Potential Data Contamination and LLM Bias: A major concern is the pervasive use of GPT-4o throughout the pipeline (frame annotation, caption refinement, Q/A generation, and evaluation). Although the authors implement safeguards, there is a non-trivial risk of the model learning a "stylistic bias" or patterns specific to GPT-4o's output, rather than genuine biomedical reasoning. The evaluation with Gemini mitigates this but does not fully eliminate the concern, as the fine-tuned models' training data was itself shaped by GPT-4o. 

2.	Absolute Performance is Still Low: Despite impressive relative gains, the absolute performance on the video benchmarks remains low (e.g., 25.1% for the 7B model on SurgeryVideoQA). The paper correctly notes that these tasks are challenging, but the low scores highlight that the problem is far from solved and that the models are not yet reliable. This should be more prominently discussed in the context of the claim that models can "learn medicine." 

3.	The paper claims the models are "learning medicine," but their performance on the standard MedQA text benchmark significantly decreased after fine-tuning. For example, the Qwen-2-VL-7B-Biomed model’s accuracy dropped from a baseline of 52.6% to 47.4%. This suggests the informal, "noisy" knowledge from the videos may be conflicting with, or causing the model to "forget," formal textbook knowledge. This directly contradicts the paper's primary narrative.

4.	Limited Analysis of "Why" and Failure Modes: The paper excellently demonstrates that the method works but provides less insight into why and how it fails. A deeper analysis of the types of questions or video segments where the model struggles (e.g., temporal reasoning in long surgical videos vs. static frame understanding in echocardiograms) would be highly valuable. The qualitative examples are good but are primarily success cases. 

5.	Clarity on Training-Test Split and Overlap: While the authors state there is no video ID overlap between OpenBiomedVid and SurgeryVideoQA, both are sourced from YouTube. There is a potential for concept or stylistic overlap. A more detailed discussion on how the "cleanliness" and focus of the evaluation set differ from the training data would clarify the generalization claim.

### Questions
please see the weakness

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents **OpenBiomedVid**, a large-scale dataset of **1,031 hours of biomedical educational videos** from YouTube, containing **22K clips** and **79K Q/A pairs**. The dataset is created through a **human-in-the-loop pipeline** that combines Whisper transcription, GPT-4o caption refinement, and expert verification. The authors also introduce two benchmarks, **SurgeryVideoQA** and **MIMICEchoQA**, for biomedical video-language evaluation. Fine-tuning **Qwen2-VL (2B/7B)** models on OpenBiomedVid yields strong performance gains across biomedical video and image benchmarks, showing that open educational videos can be an effective training signal.

### Strengths
1. Addresses an underexplored area: **biomedical video-language modeling**.  
2. The dataset is **large, diverse, and well-organized**, potentially a valuable community contribution.  
3. The **data curation pipeline** is systematic, combining LLM-based processing with human verification.  
4.  Empirical results are consistent across multiple benchmarks and architectures.

### Weaknesses
1.  **Evaluation bias** — heavy dependence on GPT-4o for caption refinement, Q/A generation, and evaluation creates potential bias and reproducibility concerns.  
2. **Lack of deep analysis** — minimal discussion of failure cases, temporal reasoning, or cross-domain generalization.  
3. **Ethical and legal clarity** — the discussion of data licensing, PHI risk, and content ownership is insufficient.  
4. **Incremental insight** — while scale is impressive, the core finding (“educational videos help”) feels somewhat intuitive and under-analyzed.

### Questions
1. **Will the dataset and benchmarks be fully released?** If so, under what license and with what access restrictions?  
2. How is **personal or sensitive information** (e.g., PHI, identifiable subjects) detected and filtered in the YouTube videos?  
3. Does the fine-tuned model show **transferability** to unseen clinical or institutional datasets?  
4. What is the ratio of educational diagrams/narration-only videos to true clinical imaging?  
5. Are there **quantitative measures of data quality**, such as annotation agreement or verification accuracy?

### Soundness
3

### Presentation
2

### Contribution
2
