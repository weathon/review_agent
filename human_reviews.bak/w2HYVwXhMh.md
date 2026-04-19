# Unlocking Exocentric Video-Language Data for Egocentric Video Representation Learning

- Decision: Reject
- Scores: 6, 6, 6, 5

## Abstract
We present EMBED (Egocentric Models Built with Exocentric Data), a framework designed to mine video-language data from exocentric sources for egocentric video representation learning. Large-scale exocentric data covers diverse activities with significant potential for egocentric learning, but inherent disparities between egocentric and exocentric data pose challenges in utilizing one view for the other seamlessly. In this study, we propose leveraging hand-object interactions and language narratives as cues to incorporate exocentric data into egocentric training. Specifically, we focus on identifying specific video clips that emphasize hand-object interactions and pairing them with action-focused language narrations. By applying our framework to exocentric datasets such as HowTo100M, we construct datasets thar are effective for egocentric video-language pretraining. Our extensive evaluations reveal that EMBED achieves state-of-the-art performance across various egocentric downstream tasks, including a 4.7\% absolute improvement in multi-instance retrieval on the Epic-Kitchens-100 benchmark and a 6.2\% improvement in classification on the EGTEA benchmark in zero-shot settings. Furthermore, EMBED enables egocentric video-language models to perform competitively in exocentric tasks. Finally, we showcase EMBED's application across various exocentric datasets, exhibiting strong generalization capabilities when applied to different exocentric datasets.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes EMBED, a data generation framework that leverages exocentric datasets for egocentric video-language pretraining. Unlike previous works that primarily learn from paired ego-exo data or unpaired full-scale exocentric videos (e.g., Ego-Exo, EgoExoNCE), EMBED focuses on learning HOI-related representations from zoomed-in, HOI-specific exocentric videos and egocentric-style text descriptions. The authors conduct comprehensive experiments across various egocentric and some simple exocentric benchmarks that demonstrates promising results.

### Strengths
- **Comprehensive Experimental Evaluation of EMBED**: I appreciate the authors' efforts in conducting diverse and thorough evaluations. They primarily validate their exocentric video-text pairs from the HTM-AA dataset across a wide range of commonly used egocentric benchmarks, as well as simpler exocentric benchmarks like HMDB and UCF101. They also attempt to demonstrate the effectiveness of EMBED using other third-person datasets like Kinetics-700, though this yields only minor improvements.
- **Promising Experimental Results**: EMBED improves LaViLa-B and LaViLa-L when continue to pretrain on Ego4D and HTM, offering a promising approach for egocentric video understanding by using filtered exocentric data (particularly HTM-AA) that highlights hand-object interactions.

### Weaknesses
- **Limited Technical Contribution:** The tools (HOI Detector, Narrator), data (Howto100M), and architecture (LaViLa) are mostly well-known in the egocentric domain. From prior works like Ego-Exo[CVPR2021], EgoVideo [CVPR 2024 EgoVis challenge champions] and EgoExoNCE [CVPR 2024], I've seen that incorporating exocentric data can offer some benefits. The approach of extracting HOI regions for training is somewhat similar to GroupRandomSizedCrop (if the crop just happens in HOI region and zoomed-in), a useful augmentation in the egocentric domain. Additionally, rephrasing exocentric text into egocentric text is an obvious and naive way to address the text domain gap. Overall, I don’t find much novelty here.
- **Concerns About Using the Exocentric Dataset HTM-AA:** While I appreciate the extensive experiments, the improvements seem to stem primarily from using the clean version of the HowTo100M dataset (HTM-AA). HTM-AA offers cleaner data and includes a significant amount of kitchen activities, which overlap with scenarios in Epic-Kitchens and EGTEA, likely contributing to the larger improvements in these downstream tasks. Table 5 further shows that EMBED gains little and incurs high costs when pretraining with additional noisy datasets like K700, strenghthening my concerns about the overall effectiveness of EMBED.
- **Concerns About Initialization with LaViLa:** I have concerns of the approach to initialize with LaViLa (mentioned in Lines 356-357) rather than pretraining from scratch when fine-tuning all parameters. By initializing with LaViLa, the overall computational cost—combining the training of LaViLa on Ego4D and fine-tuning on Ego4D+HTM—becomes twice that of LaViLa alone, which raises concerns about a potentially unfair comparison in Table 1.

### Questions
- **About Selecting Visually Aligned Sentences:** Why is the selection conducted by classifying the text alone? Wouldn't the mismatch between video and sentences be determined by both the video and the text? It seems that the classification focuses on distinguishing HOI from non-HOI sentences. Please correct me if I’m wrong.
- **Limited Improvement with SSv2:** Despite SSv2 being more similar to egocentric visual content, the improvement is even smaller than with the noisier K700 dataset. I'm curious about the underlying reasons for this.
- **Obvious Typo in Abstract, Line 20-21**: ""..we construct datasets thar are..."", thar-->that.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper presents EMBED, a framework for enhancing egocentric video representation learning using exocentric video-language data. The method involves: 1.Video Clip Curation: Selecting video clips from exocentric datasets that emphasize hand-object interactions and refining them spatially to focus on these regions. 3.Language Narration Generation: Using an exo-to-ego rephraser to transform existing narrations and an ego narrator to generate new ones in egocentric style. The experimental results indicate that this method achieves performance improvements across egocentric tasks.

### Strengths
1.This method has achieved a significant performance improvement across several tasks.
2.The authors conduct thorough ablation experiments.
3.The authors propose a framework that utilizes a large amount of third-person video data to assist in understanding egocentric video data, which is significant for advancing first-person video understanding.

### Weaknesses
1. It is unclear whether the authors intend to release their models and datasets.
2. There have already been some works on refining annotations using LLMs and leveraging exocentric videos to assist with egocentric videos, making the entire framework not novel.
3. How is the quality of annotations on exocentric videos using an HOI detector and how do the authors handle erroneous data?

### Questions
See weaknesses above.

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
4

### Summary
Large-scale web-based datasets are commonly available; however, high-quality, ego-centric video data remains challenging to obtain. While datasets like EGO4D have been introduced, their scale still falls short in meeting the demands of scaling laws in Multi-modal Large Language Models (MLLMs). 

In this work, they propose a method to extract ego-centric clips from web videos, primarily leveraging the HowTo100M dataset, which contains a substantial collection of instructional videos. This approach serves as an efficient way to retrieve valuable ego-centric clips from existing large-scale video data, expanding the resource pool for ego-centric research. Beyond common tasks explored in related ego-centric works, the author also conduct a series of experiments on HOI set and show inspiring improvement.

### Strengths
1. The motivation is clear: there remains a challenge in accessing high-quality ego-centric video data.

2. LaViLa-B demonstrates a clear improvement over both EgoVLP and EgoVLP V2 on the EK-100 MIR and EgoMCQ tasks.

3. The approach is straightforward and well-presented, making it easy to understand and practical for real-world applications.

4. Unlike previous ego-centric models, this paper proposes evaluating the model in a human-object interaction (HOI) setting.

### Weaknesses
1. The primary concern is the comparison and discussion of the advantages over retrieval-augmented methods, such as the Neural Data Server series. The critical aspect of EMBED lies in its need to access original data, denoted as X^{exo} in this work.

2. EMBED requires multiple offline models and must iterate through all candidates in a large corpus, such as HT100M in this work, making it time-intensive. Additionally, LANGUAGE NARRATION GENERATION relies on off-the-shelf LLMs, like LLAMA2. These steps involve substantial engineering efforts in data filtering, rather than being directly learned or optimized as part of the core model training.

3. From Tab4 we observe original noisy HT100M also boosts the performance a lot. This shows the improvement bring by EMBED is limited on UCF.

### Questions
1. The HT100M consists a large part of instruction video during data collection. So it is natural that this work can select suitable clip. If possible this work also works in other datasets like K600, Moments In Time etc.?

2. The HTM-AA dataset  is 247K videos, how about the samples number of HTM-EMBED in Tab.4 ?

3. Even though the EMBED data selection need to run only once for multiple experiments. Could you add the time comparison for LaViLa-B with LaViLa-B+EMBED.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper presents EMBED (Egocentric Models Built with Exocentric Data), a novel approach for adapting exocentric video-language data to improve egocentric video representation learning. Although exocentric data is rich and varied, it lacks the close-up hand-object interactions and narratives central to egocentric perspectives.  EMBED bridges this gap by identifying specific video clips that emphasize hand-object interactions and pairing them with action-focused language narrations. Experiments show EMBED's effectiveness, achieving state-of-the-art performance with a 4.7% improvement on Epic-Kitchens-100 multi-instance retrieval and a 6.2% increase on EGTEA classification in zero-shot settings. Furthermore, EMBED enhances egocentric model performance on exocentric tasks and demonstrates strong generalization across diverse exocentric datasets, showcasing its potential to unify egocentric and exocentric video learning and capitalize on the unique strengths of exocentric data for egocentric applications.

### Strengths
(1) This paper iseasy to follow and  well-written.
(2) The proposed approach of using HOI to filter exocentric videos for egocentric learning is compelling, meanwhile the exo-to-ego rephraser and ego narrator prove effective. The paper provides extensive experiments across diverse datasets and benchmarks, offering strong support for its methodology.

### Weaknesses
The technical contribution of this work appears limited, as it primarily proposes a data filtering process for exocentric videos, utilizing pretrained LLMs to refine HowTo100 captions into an egocentric format and to generate additional egocentric narrations—an approach already widely explored in previous research. These methods seem more suited for the Ego4D workshop and competition, raising questions about their suitability for an ICLR submission.

Another key concern is the reliance on egocentric retrieval and classification benchmarks, which have been heavily used in recent years. It’s widely acknowledged that improvements on these benchmarks may not necessarily indicate a truly understanding of egocentric content or applicability in real-world scenarios. If the authors could provide additional experiments demonstrating consistent improvements on more challenging tasks, such as grounding or manipulation tasks, I would be inclined to reconsider my rating.

### Questions
Please refer to the weakness.

### Soundness
2

### Presentation
3

### Contribution
2
