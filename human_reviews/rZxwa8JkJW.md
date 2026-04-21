# Fusing Visual and Textual Cues for Sequential Image Difference Captioning

- Avg Score: 5.25
- Decision: Reject
- Scores: 8, 3, 5, 5

## Abstract
We present FVTC - a technique for image difference captioning that is able to benefit from additional visual and/or textual inputs. FVTC is able to succinctly summarize multiple manipulations that were applied to an image in a sequence. Optionally, it can take several intermediate thumbnails of the image editing sequence as input, as well as coarse machine-generated annotations of the individual manipulations. We demonstrate that the presence of intermediate images and/or auxiliary textual information improves the model's captioning performance. To train FVTC, we introduce METS - a new dataset of image editing sequences, with textual machine annotations of each editorial step and human edit summarization captions after the 5th, 10th and 15th manipulation.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper presents Fusing Visual and Textual Cues (FVTC), a novel technique for image difference captioning that summarizes multiple manipulations applied to an image in sequence.  FVTC leverages intermediate image thumbnails and machine-generated annotations of individual edits, improving captioning performance.  To support FVTC, the authors also introduce METS (Multiple Edits and Textual Summaries), a new dataset of image editing sequences with machine annotations for each edit and human summaries at specific points.  This addresses the limitations in existing datasets, which often focus on image pairs or short, non-destructive editing sequences.

### Strengths
The paper introduces a novel task: image difference captioning with multiple inputs, including intermediate editing stages and textual annotations.  This expands the scope of existing image difference captioning and addresses real-world scenarios in creative editing workflows. The METS dataset, with its focus on longer editing sequences and destructive changes, is a significant contribution to the field.

The data generation process for METS appears robust, incorporating generative and pixel-level edits with GPT-3.5 generated prompts and human labeling for edit summaries.  The proposed FVTC model, a multi-modal LLM with multiple visual inputs, is well-suited to the task.

Addressing the challenge of summarizing multi-step image edits, particularly those with destructive changes, is a significant step for image provenance and content authenticity verification. The METS dataset and FVTC model offer valuable resources for research in this area.

### Weaknesses
The experimental evaluation could be strengthened by including comparisons to a wider range of models, including closed models like Claude and Gemini, as well as open-source models capable of handling multiple images, such as certain LLaVA versions.  The lack of comparison to other models in the MagicBrush evaluation (Table 1) is a notable omission.

### Questions
Could you elaborate on the choice of 1000 sequences for the METS dataset? Is there a risk that this dataset size is insufficient for training robust models?

Why weren't other models, such as those used in the METS experiments (GPT3.5, GPT4-V), included in the MagicBrush comparison (Table 1)?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper creates a new dataset for image difference captioning with multi-step editing, which contains more steps than existing benchmarks, by automatically generating difference captions for each step and adding human annotations every 5 editing steps. Additionally, the authors train a multimodal model on the edited image sequence.

### Strengths
The idea of leveraging multiple images from intermediate editing stages to generate difference captions and assessing whether this approach outperforms using only the original and final images is intriguing. Furthermore, the paper demonstrates that the introduced METS dataset may be valuable for adapting models to datasets like PSBattles.

### Weaknesses
There are several areas in the current version of the paper that could benefit from further improvement.

**Motivation** 

The scenario presented in the paper, which assumes access to both the original and the final edited images along with all intermediate editing steps, seems less applicable to real-world situations where only the original and final images are typically available. Relying on intermediate editing steps—and even more so on the auxiliary text that describes each edit—feels more like a limitation than a strength in practical applications.

**Evaluations**

In Table 1, the authors compare their method to zero-shot GPT-4V but limit the evaluation to their new METS dataset without comparing it to MagicBrush, which is only used for comparing FVTC-2 and FVTC-4. Additionally, the use of n-gram-based metrics, which are highly sensitive to caption phrasing and length, may skew the results. For instance, if GPT-4V accurately captures the edits but generates a longer caption, it would score poorly on n-gram similarity. These metrics are mostly reflective of quality when the models compared are trained on the same datasets, and so have similar phrasing. Incorporating an LLM-as-a-Judge evaluation would improve the reliability and robustness of the results. It would be great if the authors could provide a comparison on MagicBrush, as welll as evaluation with LLM-as-a-Judge.

Moreover, in Table 2, the authors compare their model against several older baselines, all of which use weaker backbones compared to FVTC, which is built on LLaMA-2. Since FVTC adopts a standard LLaVA-like architecture [1] and follows conventional training procedures, the contribution of the proposed model appears limited. Moreover, Table 2 does not include an evaluation on MagicBrush, which is the standard benchmark for this task. This omission limits the comparability and relevance of the results. Adding a comparison on MagicBrush would greatly help.

[1] Liu et al. Visual Instruction Tuning. NeurIPS 2023

### Questions
How was text-only GPT-3.5 prompted in the Table 1 experiments? Its poor performance is surprising, given that it had access to all editing steps and only needed to summarize them. Providing a few-shot example in the prompt would likely lead to a significant improvement in its results, could the authors clarify whether this was considered or tested, and if not conduct such experiment?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper focuses on the task of image difference captioning. It first introduces a dataset, METS, which contains image sequences generated by pixel-level edits and generative manipulations using inpainting models. Then, this paper trained a Fusing Visual and Textual Cues, which can take multiple images and additional textual information when generating captions. The proposed model is compared with baselines on several image difference captioning tasks.

### Strengths
The proposed synthetic image sequence generation pipeline is technically sound. The paper is easy to follow.

### Weaknesses
1. **Evaluation Metrics.** This is the main concern of this paper, which makes it hard to evaluate the effectiveness of the proposed method. The paper chooses n-gram-based metrics, which are too strict and cannot capture the semantics between two sentences. For example, even if two sentences express a similar meaning, the different wording, paraphrases, and sequence length will result in a low score. This is reflected in Table 1, where the best model only gets an 8.2 BLEU score, meaning none of the models works here. In addition, those scores are not consistent, e.g., in Table 2, FTVC has no big difference with baselines in some metrics, but will suddenly be much higher/lower on some metrics. For captioning tasks, evaluation is challenging. Only relying on those automatic metrics will finally lead to a biased model that overfits the output style of a certain dataset. I suggest doing some human evaluation or using an LLM as the judge, which can be a more robust metric to quantify models' performance.

2. **Lack of Ablations and Analyses.** This paper only shows the main results on the evaluation datasets without ablating different components of the system. There are a lot of interesting analyses to do in the data generation and model training parts, e.g., will the number of generated image sequences affect the final performance? Why use Eq (2) to decide on the manipulation?

3. **Missing Baselines.** Adding the recent MLLMs (like LLAVA and QwenVL) as baselines will provide context on those benchmarks. In addition, fine-tuning those open-source MLLMs on your created METS datasets could help to disentangle the impact of data and modeling.

### Questions
1. Why is GPT-4V not evaluated on all other tasks?

2. Why not add METS when training on CLEVER and SPOT-THE-DIFF?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper presents Fusing Visual and Textual Cues (FVTC), a method for image difference captioning that leverages multiple sequential inputs. The method includes images and optional textual descriptions, to generate descriptive captions summarizing changes across multiple editing steps. The authors introduce the METS (Multiple Edits and Textual Summaries) dataset, comprising image sequences with detailed annotations and human-generated edit summaries. The study evaluates FVTC’s performance on established benchmarks and demonstrates superior results with multiple input images and textual data, outperforming baselines in captioning accuracy.

### Strengths
1. The FVTC model’s capability to incorporate multiple images and auxiliary text to improve difference captioning is novel and applicable to real-world scenarios where tracking sequential changes is crucial.
2. The new dataset is worth collecting. METS fills a gap by offering a dataset with longer, realistic editing sequences, supporting nuanced understanding of progressive edits.
3. FVTC is evaluated against multiple datasets with various metrics (e.g., BLEU, CIDEr, METEOR), showing robustness and adaptability across tasks.

### Weaknesses
1. The method depends on auxiliary textual data. While auxiliary data enhances model performance, it may limit FVTC’s application to cases where such data isn’t available, impacting generalizability. Please discuss or evaluate how FVTC performs without auxiliary textual data. Could these data be generated automatically in scenarios where the data is not readily available?

2. Concerns with task definition:

(1) The task of image difference captioning, as defined in this paper, focuses on scenarios where two images have nearly identical backgrounds with slight variations. However, this scenario is relatively limited in practical real-world applications, where more diverse changes are often encountered. For example, there are many image pairs in the Visual Storytelling Dataset (VIST) with a slight difference and shots from different viewing angles. Could these pairs be used in this method?

(2) Using semantic maps to highlight changes directly would be more explicit and interpretable for identifying differences between two images. For example, performing the change detection first and then describing the part of the change. Please include a comparison with semantic map methods if feasible. Captioning, by contrast, involves a degree of semantic encoding that translates visual differences into linguistic descriptions, making it difficult to accurately assess the model's true ability to detect and interpret nuanced changes. 

3. Concerns with methodology: The paper relies heavily on synthetic data for training, which raises questions about the model's adaptability to real-world distributions. The extensive use of synthetic data may cause the model to become overly specialized in handling synthetic image patterns, which could differ significantly from real-world image characteristics, potentially limiting its generalization and robustness in practical scenarios. The authors could include experiments or analysis of the model's performance on real-world datasets. For example, some pairs could be constructed from MSCOCO or LAION-2B datasets. The authors could also discuss strategies for mitigating potential overfitting to synthetic data patterns.

### Questions
Please address my concerns above.

### Soundness
3

### Presentation
2

### Contribution
2
