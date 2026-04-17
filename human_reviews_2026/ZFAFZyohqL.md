# CH-CEMS: A Chinese Multi-Concept Benchmark Dataset Towards Explainable Multi-Modal Sentiment Analysis

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 6, 4

## Abstract
Explainable Multimodal Sentiment Analysis (EMSA) is a booming research area aimed at advancing robust and faithful multimodal language understanding. Recent explainable datasets and methods based on multimodal large language models (MLLMs) have introduced a new paradigm that produces chain-of-thought–style explanations within affective computing. However, high-quality data resources for EMSA remain scarce, largely because annotating reliable reasoning cues is costly and difficult. To address this gap, we introduce CH-CEMS, the first multimodal sentiment dataset for explainable multimodal sentiment analysis. It contains 3,715 curated video segments with polarity and intensity annotations. In addition, we annotate three semantic concepts for each sample (i.e., speaking style, tone of voice, and facial expression), which serve as explicit reasoning cues to enable process-level supervision. To fully leverage these concept cues, we propose a concept-guided reinforcement learning framework with Group Relative Policy Optimization (GRPO) for MLLMs, in which concept-level supervision explicitly constrains cross-modal semantic relations and guides the model to infer sentiment from verifiable concepts. We further establish baselines with state-of-the-art multimodal machine learning methods and MLLMs via zero-shot inference and supervised fine-tuning. Experiments show that MLLMs outperform feature-based methods, typically by 4–12\% in accuracy for three-class sentiment analysis, and that our concept-guided GRPO yields a further 8.5\% improvement, even surpassing closed-source model such as GPT-5. We believe CH-CEMS and the benchmark will facilitate future research on explainable multimodal sentiment analysis. The dataset and codes are avaliable for use at https://anonymous.4open.science/r/CH-CEMS-C34F.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper focuses on explainable multimodal emotion recognition, introducing a dataset (CH-CEMS) and a GRPO-based pipeline to address this task. The key novelty lies in leveraging semantic concepts, where the dataset annotates these concepts and the framework uses their recognition accuracy as rewards during reinforcement learning.

### Strengths
1.	A new dataset for explainable multimodal emotion recognition.

2.	Incorporation of semantic concepts during both dataset construction and reward design.

3.	Performance improvements compared to existing MLLMs.

### Weaknesses
1.	The paper claims that CH-CEMS is “the first multimodal sentiment dataset for explainable multimodal sentiment analysis” and “the first Chinese dataset for explainable multimodal sentiment analysis”. However, Explainable Multimodal Sentiment Analysis is an active research area with prior datasets (e.g., EMER and MERR) that also provide reasoning cues. Thus, the novelty claim is overstated.

2.	“EMER curates multimodal emotion–reasoning pairs (Liu et al., 2023a) and MERR provides a multimodal emotion description and reasoning benchmark (Zhang et al., 2024). PanoSent introduces a multimodal conversational ABSA benchmark for panoptic sextuple extraction and sentiment ﬂipping with causal rationales (Luo et al., 2024). However, to the best of our knowledge, these resources primarily target emotion recognition or aspect-based sentiment analysis rather than user-centric sentiment reasoning.” => The paper states that EMER and MERR focus on emotion recognition or aspect-based sentiment analysis rather than user-centric sentiment reasoning. However, to the best of my knowledge, these datasets do involve user-centric sentiment reasoning.

3.	The paper classifies semantic concepts into only three categories (speaking style, tone of voice, and facial expression). However, emotion reasoning also depends on other cues (e.g., gestures, events, background, linguistic content). Focusing on such a narrow set may reduce the completeness of emotion reasoning.

4.	As shown in Figures 2 and 4, the authors categorize semantic concepts (e.g., speaking style, tone of voice) into fixed labels. However, these concepts are complex and context-dependent, making a fixed taxonomy potentially inaccurate or incomplete.

5.	Some abbreviations (e.g., SP, SI, EC, adverts in Table 1) are not clearly defined.

6.	There are also citation errors in the paper. Proofreading is recommended to improve clarity and accuracy.

7.	Prior emotion reasoning frameworks (e.g., R1-Omni, Emotion-LLaMA, AffectGPT) already use emotion descriptions containing semantic concepts (e.g., speaking style, tone of voice, facial expression). This paper proposes explicitly extracting these concepts from descriptions. However, it is unclear how this differs from or improves upon traditional description-generation methods. From my perspective, previous description generation methods can cover more semantic concepts but this paper only focuses on limited semantic concepts. I cannot figure out the benefit of this approach. Therefore, a discussion on the advantages of a concept-driven approach is needed.

8.	"Thus We accordingly annotate" → Should be "Thus, we accordingly annotate"(lowercase "we")

### Questions
See weakness.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces CH-CEMS, a novel Chinese multimodal sentiment analysis dataset designed for explainability. The dataset comprises 3,715 carefully curated video segments that are manually annotated for emotional polarity and intensity, as well as three key semantic concepts: speaking style, tone of voice, and facial expression. These semantic concepts are engineered as explicit reasoning cues to enable process-level supervision of the model's inference, thereby overcoming the scarcity of high-quality data for explainable multimodal sentiment analysis (EMSA). Furthermore, we propose a reinforcement learning framework, GRPO (Guided Reinforcement Learning with Policy Optimization), for Multimodal Large Language Models (MLLMs), which implements concept-level supervision as verifiable rewards.

### Strengths
1. CH-CEMS is the first multimodal sentiment dataset for the Chinese language that supports explainability and possesses process-level supervision capability, thereby filling a significant gap in the interpretability of existing Chinese multimodal sentiment analysis datasets.

2. By annotating the three fine-grained semantic concepts—speaking style, tone of voice, and facial expression—as explicit reasoning cues, the dataset enables direct supervision of the model's inference process.

### Weaknesses
1. 3715 samples are relatively small for a multimodal dataset used for deep learning benchmark testing. Although high-quality data is crucial, a small sample size may limit the training and generalization ability of large multimodal models (MLLMs). The authors may need to argue in the main text why high-quality annotation of reasoning clues can compensate for the insufficient sample size.

2. The authors need to further argue the sufficiency of these three concepts as 'interpretability'. For example, whether there are other important cultural or social reasoning clues that have not been captured in the Chinese context.

3. The baseline method framework based on SFT (Supervised Fine Tuning) and combined with GRPO (Guided Reinforcement Learning with Policy Optimization) for reinforcement learning proposed by the author is a relatively standard paradigm in current MLLM research. It lacks sufficient novelty and uniqueness at the methodological level.

### Questions
1. The paper provides kappa scores for Tone, Expression, Speaking Style, and sentiment, but they are all between 0.45-0.55, indicating a moderate level of annotation consistency. Therefore, there is controversy over the paper's claim that this dataset is of high quality. A high score of Inter-Annotator Agreement is the key to successful process level supervision.

2. Please provide more detailed statistical information on data distribution in the paper. For example, the category distribution and co-occurrence of three semantic concepts (style, intonation, expression) (such as whether a certain "speaking style" is always accompanied by a certain "facial expression").

3. In Section 5, was the performance difference of the model in emotion prediction and explanatory generation discussed when using and not using these three explicit inference clues for process supervision? If not, this should be presented as a core experimental result to highlight the unique value of the CH-CEMS dataset.

4. Why do you use feature-based methods only to conduct regression task? What about LLMs?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper proposes multi-concept datasets for explainable multimodal sentiment analysis task and conduct concept-based GRPO with effective fine-grained reasoning process. Experiments on MOSI/MOSEI are conducted to show effectiveness of the proposed model.

### Strengths
1.	Comprehensive dataset.
2.	Experiments are sufficient.

### Weaknesses
1.	More recent baseline models with small model such as RoBERTa, DeBERTa should be conducted to show fair comparison.
2.	Multi-person cases should be provided.
3.	The SFT version of the proposed model should be included for comparison. Cold start performance should be reported to better show multi-stage training efficiency.
4.	The most number of neutral class may reduce the effectiveness of the presented dataset.

### Questions
1.	How to ensure the correctness of reasoning process? Any human value alignment stage in annotation?
2.	Why does GRPO not help for Qwen2.5-Omni compared with SFT version?
3.	Since the description of concept could be different with similar semantics, does reward computation considering such situations?
4.	Why does the neutral class emotion with 0 regressive value have the most numbers, much more than other classes which show intensity of emotion?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents CH-CEMS, a new Chinese multimodal dataset (text-audio-video) for explainable sentiment reasoning. Each clip (3,715 total) is annotated with sentiment polarity, intensity, and three intermediate concepts — speaking style, tone of voice, and facial expression — plus corresponding chain-of-thought traces.
The authors also propose a concept-guided GRPO (Group Relative Policy Optimization) framework that rewards models for correct structure, sentiment prediction, and concept alignment. Experiments are conducted on MLLMs and feature-based baselines, reporting up to 73.6% accuracy on 3-class sentiment classification, claimed to surpass several closed-source large models.

### Strengths
Process-level supervision: Incorporating three interpretable concept dimensions (style, tone, expression) provides more transparent reasoning supervision than typical black-box multimodal sentiment datasets.

Methodical RL extension: The concept-guided GRPO is a reasonable and reproducible adaptation of RLHF/GRPO to emotion reasoning, with clear reward decomposition.

Transparency: The paper provides dataset splits, inter-annotator statistics, prompts, and ethical disclosure, which improve reproducibility.

Timely topic: Explainable multimodal emotion reasoning is increasingly important in the era of MLLMs and aligns well with ICLR’s focus on interpretable and grounded learning.

### Weaknesses
1. Motivation and “Why Chinese?”

The motivation for focusing on Chinese data remains weak.
While the authors argue that Chinese multimodal sentiment data are under-represented, there is no empirical demonstration that Mandarin introduces unique challenges (e.g., tonal prosody, discourse politeness, idiomatic affect) that justify a separate dataset.
Without comparative cross-lingual experiments (e.g., transfer CH-CEMS to other commonly known datasets), the argument reads as geographical novelty rather than scientific necessity.

2. Dataset Soundness

The dataset’s size (3,715 clips) is modest for training MLLMs and may limit generalization. The total length is only 7 minutes, which is confusing. Inter-annotator κ ≈ 0.46 – 0.54 indicates moderate agreement; noise at this level restricts the theoretical ceiling of achievable accuracy. 

3. . Evaluation Scope

Evaluation is limited to the authors’ dataset; there are no cross-dataset generalization tests (e.g., MER, EmoSet). Faithfulness of the generated reasoning is measured only via concept match, not by human or counterfactual verification.

4. Missing Key Baselines and Model Comparisons

The literature review is significantly incomplete, omitting all major recent models in multimodal emotion reasoning: EmoVIT (CVPR 2024) – a unified visual-language transformer with affective grounding, establishing a strong multimodal benchmark. AffectGPT (ICML 2025) – a generative multimodal LLM fine-tuned for emotion reasoning and affective dialogue understanding. EmoSet (ICCV 2023) – employs attribute-based reasoning on a large-scale fine-grained emotion dataset, conceptually similar to this work. OV-MER (ICML 2025) – supports open-vocabulary multimodal emotion reasoning with > 200 continuous labels and powers the MER 2025 Challenge benchmark. Without discussing or comparing against these, the paper’s claimed novelty in concept-guided multimodal emotion reasoning is undermined. For ICLR, inclusion of these baselines or at least a thorough discussion is mandatory.

Others:

✓ symbols missing: In Table 1, the check marks indicating modality and concept availability are incomplete.
The number format in the table is not consistent.
Typos can be observed, e.g., in 214.

### Questions
In general, this is a good paper, but the authors need to address two main concerns before it can reach the bar of ICLR. 1. It is necessary to prove the generalizability of the proposed method on the language aspect (not limited to Chinese). 2. It's vital and beneficial to verify the trained model on other existing datasets, to show the gain of introducing reasoning and multi-attribute in the emotion understanding task.

I will adjust the rating accordingly.

### Soundness
3

### Presentation
2

### Contribution
2
