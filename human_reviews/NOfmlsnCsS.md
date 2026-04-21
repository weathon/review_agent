# Teach Multimodal LLMs to Comprehend Electrocardiographic Images

- Avg Score: 5.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 3, 8, 6

## Abstract
The electrocardiogram (ECG) is an essential non-invasive diagnostic tool for assessing cardiac conditions. Existing automatic interpretation methods suffer from limited generalizability, focusing on a narrow range of cardiac conditions, and typically depend on raw physiological signals, which may not be readily available in resource-limited settings where only printed or digital ECG images are accessible. Recent advancements in multimodal large language models (MLLMs) present promising opportunities for addressing these challenges. However, the application of MLLMs to ECG image interpretation remains challenging due to the lack of instruction tuning datasets and well-established ECG image benchmarks for quantitative evaluation. To address these challenges, we introduce ECGInstruct, a comprehensive ECG image instruction tuning dataset of over 1 million samples, covering a wide range of ECG-related tasks from diverse data sources. Using ECGInstruct, we develop PULSE, a fine-tuned MLLM tailored for ECG image interpretation. In addition, we curate ECGBench, a new evaluation benchmark covering four key ECG image interpretation tasks. Our experiments show
that PULSE sets a new state-of-the-art, outperforming general MLLMs with an average accuracy improvement of 15% to 30%. This work highlights the potential of PULSE to enhance ECG interpretation in clinical practice

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This work proposes an impressive framework for instruction-tuning ECG images, evaluates it across multiple datasets, and introduces the first benchmark for ECG image instruction tuning.

### Strengths
- The first work to treat ECG signals as visual signals rather than time series.
- Builds the first benchmark for ECG instruction tuning with image input.

To the best of the reviewer's knowledge, this is the first work converting ECG signals into visual input, allowing ECG analysis to benefit from the advancements in the Visual Language Model (VLM) community. Additionally, the authors establish a comprehensive benchmark for evaluating ECG image instruction tuning, opening a new domain for ECG analysis.

### Weaknesses
Even though this work is well-executed and evaluated on multiple datasets, it still has some weaknesses:

- **Reliance on Accuracy and LLM Scoring**: The evaluation metrics are based solely on accuracy or LLM scoring. Incorporating human evaluation could enhance benchmark quality, as current LLMs are not specialized for ECG-related text understanding. I suggest using human evaluation for all open-ended tasks.

- **ViT Model Limitations**: The Vision Transformer (ViT) model is frozen, and while LLAVA uses ViT, it is not specifically tuned for ECG image features. This may degrade performance on ECG-specific tasks.

- **Translation Issues in Report Generation**: For report generation, the PTB-XL dataset originally includes German, Danish, and other European languages, not English. Translating these reports into English could lead to inaccuracies or loss of meaning.

- **Evaluation Metrics in Report Generation**: In the report generation task, only GPT is used as the LLM judge, without any lexical or clinical efficacy metrics. Clinical efficacy metrics are commonly used in other report generation tasks, such as in [1].

- **Relevance of Abnormality Detection as an LLM Task**: Is abnormality detection a necessary task for an LLM? According to Section 3.2, this task appears similar to classification. Additionally, LLMs can generate synonyms for disease names, raising the question of how to handle variations in terminology.


[1]Tanida, Tim, et al. "Interactive and explainable region-guided radiology report generation." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.

### Questions
See weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper proposes a fine-tuned multi-modal LLM tailored for ECG image interpretation. Additionally, the paper introduces a large ECG image instruction tuning dataset with 1 million examples, covering a wide range of ECG-related tasks from diverse data sources. A new evaluation benchmark for ECG image interpretation tasks is also proposed.

### Strengths
The paper's strengths include:

- many SOTA MLLMs as baselines
- Evaluation on a wide range of tasks
- Preparation of a benchmark that could be used in future works. This may be relevant for the field.
- Preparation of an instruction-tuning dataset that could be used in future works. Again, this may be relevant for the field.

### Weaknesses
The weaknesses include:

- Motivation: while there may be cases where only printouts of ECGs are available, the need for AI models on ECG printouts is still questionable.
- Related works: Current multimodally trained ECG models are neither discussed nor compared against, e.g.:
Radhakrishnan et al. (2023), "Cross-modal autoencoder framework learns holistic representations of cardiovascular state", https://www.nature.com/articles/s41467-023-38125-0, Turgut et al. (2023), "Unlocking the Diagnostic Potential of ECG through Knowledge Transfer from Cardiac MRI", https://arxiv.org/pdf/2308.05764
- Current foundational time series models are neither discussed nor compared against, e.g.: Yang et al. (2023), "BIOT: Biosignal Transformer for Cross-data Learning in the Wild", https://proceedings.neurips.cc/paper_files/paper/2023/hash/f6b30f3e2dd9cb53bbf2024402d02295-Abstract-Conference.html, - It would have been interesting to see how much using the ECG-images instead of the raw ECGs limits the performance.
- An ablation study using an ECG encoder instead of the image model would be interesting
- An ablation study using the image encoder directly on ECG images, without the LLMs would be interesting
- To provide robustness guarantees, have the authors evaluated their methods as well as the baselines across multiple seeds set during fine-tuning?Goswami et al. (2024), "MOMENT: A Family of Open Time-series Foundation Models", https://arxiv.org/pdf/2402.03885
- Statements such as "limitations of traditional ECG models" (ll 64-65) are made without elaborating and providing references.
- Experiments / Evaluation: For fair comparison, Table 3 and 4 should include the ECG baselines mentioned above. This would highlight the differences between imaging and time series modality with respect to downstream performance.
- Furthermore, domain-specific methods in Table 3 and 4 have not been tuned for the reported downstream tasks. The authors make the effort to tune multiple MLLMs on the applications under investigation, however, they only report scores from the original paper for the domain-specific methods.
- In Table 4, MERL achieves best performance on the CPSC 2018 task, however, the authors mistakenly highlight their own method PULSE as the best performing one.
- In Table 3 and 4, the authors report that certain methods are not applicable or not designed for certain tasks, which is why they opt to not report scores in those cases. However, for the open-source MLLMs that are not designed for ECG image analysis either, the authors tune the models and report performance scores. This clearly is an unfair comparison, further highlighting the need for tuning the domain-specific methods and reporting their performance.
- All LLM-based baselines have not been trained on ECGs / ECG-images, while ECG baselines are not able to handle all tasks. So for some of the tasks (the text-based ones) no meaningful baseline is provided. While such baselines may not be available, the quality of the model could still be compared with strong baselines by splitting it into subtasks, i.e. use established standard models to extract relevant features from a given ECG (heart rate, irregularities, abnormalities, …). Then include the extracted features in a prompt and ask a general or medical LLM to provide an answer. This could handle most of the tasks and thus provide a meaningful baseline.
- Limited novelty: This seems to be a very applied paper without any novelty in the method (note that the provided instruction tuning and evaluation datasets are still relevant for the field)

### Questions
- Are there many use cases, where ECGs are not available in electronic form, but where scanners and AI models are available and could be integrated?
- Even if only printouts are available, is directly interpreting them as images the best solution? Wouldn’t the more reliable solution be to convert these images into electronic ECGs first (the problem seems relatively simply due to the high contrast of the plots and often clearly visible reference lines) and the use established ECG models?
- Motivation lacks reasoning: Why would the authors transform readily available ECG recordings, e.g. 800k samples from MIMIC-ECG, into images before analysing them? The majority of the transformed image represents background, i.e. noise. Is the patch projector of the imaging encoder (ViT) capable of learning a meaningful signal from the data?
- It would have been interesting to see how much using the ECG-images instead of the raw ECGs limits the performance.
- An ablation study using an ECG encoder instead of the image model would be interesting
- An ablation study using the image encoder directly on ECG images, without the LLMs would be interesting
- To provide robustness guarantees, have the authors evaluated their methods as well as the baselines across multiple seeds set during fine-tuning?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces PULSE, a multimodal large language model (MLLM) tailored for interpreting electrocardiogram (ECG) images, addressing the limitations of existing methods which often rely on raw physiological signals and lack generalizability. The authors contribute a new dataset, ECGInstruct, comprising over one million ECG image-text samples to fine-tune PULSE, and ECGBench. Experiments demonstrate that PULSE outperforms both proprietary and open-source MLLMs, with an average accuracy improvement of 15% to 30%.

### Strengths
This work contributes a comprehensive ECG image instruction tuning dataset from diverse data sources, which facilitates the advancement of MLLMs in understanding ECG images.
The proposed PULSE model surpassed proprietary and open-source MLLMs by a large range in diverse tasks related to ECG image comprehension. The model, data and code have been released.

### Weaknesses
As stated in the discussion, the dataset contains few multistep instructions, which could undermine the multistep reasoning capability of the PULSE model and limits the report generation performance.

### Questions
According to table 4, the performance gain in arena score which reflects the model’s instruction-following ability is limited compared with Claude 3.5 Sonnet in arena score. The authors could include more discussion about the results to help improve multistep reasoning capabilities in future research.

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
4

### Summary
This study presents the creation of a fine-tuning dataset and benchmark for ECG in multi-language language models (MLLM), which appears to hold value for advancing medical AI. The dataset is synthetically generated with various distortion noises, and Llama 3 was used for ECGInstruct and quality checking; however, the accuracy of these implementations has not been assessed, raising concerns about potential LLM bias. Therefore, evaluation on independent downstream task datasets would be essential.

### Strengths
The study’s strengths include the development of a large-scale dataset and the introduction of extensive benchmark tasks, both of which contribute positively to the field.

### Weaknesses
The study lacks a strong technical novelty and technical details.
For a fairer comparison, it would be preferable to include fine-tuning results against other LLMs. 
Additionally, human accuracy should be evaluated, at least partially, for the ECGBench tasks. 
It is also recommended to assess performance on external validation tasks beyond the ECGBench dataset.

### Questions
Please provide technical details.
For a fairer comparison, it would be preferable to include fine-tuning results against other LLMs. 
Additionally, human accuracy should be evaluated, at least partially, for the ECGBench tasks. 
It is also recommended to assess performance on external validation tasks beyond the ECGBench dataset.

### Soundness
3

### Presentation
3

### Contribution
3
