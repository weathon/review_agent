# NatureLM-audio: an Audio-Language Foundation Model for Bioacoustics

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 3, 5, 8

## Abstract
Large language models (LLMs) prompted with text and audio have achieved state-of-the-art performance across various auditory tasks, including speech, music, and general audio, showing emergent abilities on unseen tasks. However, their potential has yet to be fully demonstrated in bioacoustics tasks, such as detecting animal vocalizations in large recordings, classifying rare and endangered species, and labeling context and behavior—tasks that are crucial for conservation, biodiversity monitoring, and animal behavior studies. In this work, we present NatureLM-audio, the first audio-language foundation model specifically designed for bioacoustics. Our training dataset consists of carefully curated text-audio pairs spanning bioacoustics, speech, and music, designed to address the field's limited availability of annotated data. We demonstrate successful transfer of learned representations from music and speech to bioacoustics, and our model shows promising generalization to unseen taxa and tasks. We evaluate NatureLM-audio on a novel benchmark (BEANS-Zero) and it sets a new state of the art on several bioacoustics tasks, including zero-shot classification of unseen species. To advance bioacoustics research, we release our model weights, benchmark data, and open-source the code for training and benchmark data generation and model training.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
The paper introduces NatureLM, an audio-language model specifically designed for bioacoustics, focusing on animal sounds. NatureLM leverages a diverse dataset combining bioacoustics, human speech, and music to enhance generalization across various species and tasks. NatureLM demonstrates strong in-context learning capabilities, enabling zero-shot classification for unseen species.

Additionally, the paper presents BEANS-Zero, a benchmark for bioacoustics that includes several tasks beyond species classification, such as call-type prediction, life-stage classification, and individual counting, aiming to push the boundaries of bioacoustic research.

### Strengths
1. The introduction of NatureLM, the first audio-language model specifically designed for bioacoustics, represents a promising new direction for incorporating language models into biodiversity monitoring.

2. The development of the BEANS-Zero benchmark extends the original BEANS benchmark by introducing new tasks, such as call-type prediction, life-stage classification, individual counting, and open-ended audio captioning. These additions have the potential to advance bioacoustics research and enable more detailed acoustic monitoring of species.

3. The paper is easy to follow, and the related works are thoroughly referenced.

### Weaknesses
1. Incorrect Terminology


1.1.  The introduction describes BioLingual as self-supervised; however, the supervision is derived from text generated based on class labels. I recommend referring to it as supervised learning with language-based supervision for greater clarity and accuracy.

2.1. Both BioLingual and AVES are described in the paper as foundation models, but this classification may be misleading. BioLingual and AVES are trained on datasets with less than 2 million samples, while models trained on AudioSet with 2 million samples are not typically considered foundation models. BioLingual is evaluated on classification and retrieval tasks, while AVES is evaluated on classification and detection tasks. Typically, a foundation model is a very large model trained on extremely large datasets so that properties emerge that enable it to handle a wide variety of tasks across different domains. For improved clarity, I suggest either using alternative descriptions like self-supervised (for AVES) and audio-language contrastive (for BioLingual) to more accurately describe them, or providing a clear definition of what you consider to be a foundation model if you wish to retain this terminology. This will help avoid confusion and ensure that readers more precisely understand the capabilities of these models.

2. Overstatement of Results


2.1. The analysis of NatureLM-audio's performance on the cbi dataset (Table 4, Section 4.2) is potentially misleading due to data overlap. Since BirdNet, Perch, and NatureLM-audio are trained on Xeno-Canto, and the cbi evaluation dataset is a subset of Xeno-Canto, there is overlap between the training and evaluation data. This compromises the claim of state-of-the-art performance, especially that Perch significantly outperforms NatureLM in enabirds.

2.2. In Table 5, Section 4.3, the authors claim that NatureLM-audio demonstrates generalization to completely unseen species by outperforming CLAP, a contrastive audio-language model trained on non-bioacoustic data. However, this conclusion seems too strong. The results primarily show that a model trained on bioacoustic data performs better than one trained on non-bioacoustic data, which does not necessarily indicate true generalization to unseen species. Additionally, BioLingual, which was trained similarly to CLAP but on bioacoustic data, significantly outperforms NatureLM-audio. I suggest rephrasing the conclusion to more accurately reflect these findings.

2.3. In Section 4.4, the paper claims state-of-the-art performance in the captioning task. However, this comparison is based solely on the SALMONN model, which was not trained on bioacoustic data and is not specifically designed for this task. To support the claim of state-of-the-art performance, I recommend including a comparison with a captioning model trained on bioacoustic data, as this would provide a more accurate and meaningful evaluation.

### Questions
1. In Section 3.1.1, the method for hard negatives sampling for species detection is briefly mentioned, but the details are unclear. Could you provide a more detailed explanation of the strategy for sampling these hard negative samples?

2. Could you provide more details in Section 3.2 about the selection strategy for the held-out species in the AnimalSpeak dataset for the unseen-cmn and unseen-sci subsets? Even if the selection was random, it's important to clarify the process.

3. In Section 4.5, you state that including speech and music in training has a positive impact on counting zebra-finch individuals. However, while the ablation study includes speech, it is unclear whether music was also ablated. Could you clarify whether the effect of music was tested? Additionally, conducting ablation on all downstream tasks, not just individual identification, could provide more comprehensive insights into whether speech and music data enhance performance across other tasks as well. This could help clarify which types of training data are most beneficial for specific applications.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The authors present a large audio-language model (Lalm) trained and evaluated on bioacoustics data. First, a dataset of text-audio pairs is compiled, partially with the use of LLMs. This dataset is then used to train an architecture similar to SALMONN. It is consisting of a LLM (Llama 3.1-8b), an audio encoder (BEATs), and a Q-Former to connect the computed audio embeddings to the LLM. During training the LLM stays frozen while the audio encoder and Q-Former are trained after a curriculum (first species classification than also detection, captioning, life stage prediction, call-type detection). The model is evaluated on the self introduced BEANS-Zero benchmark which extends BEANS with additional tasks and a zero shot evaluation protocol. The model achieves the best results compared to non bioacoustics

### Strengths
1. Addresses a important topic from both the ML research community ( since audio and especially computational bioacoustics is a hard problem) and societal importance.
2. Collects a comprehensive training dataset and extends an existing evaluation benchmark  with additional tasks.
3. The performance improvements compared to a model not trained on bioacoustics data (SALMONN) supports the claim that this domain is in the need for a own foundation model.

### Weaknesses
1. Soundness of results: Your presented results only show a minor improvement compared to BioLingual (which also presents zero shot results on BEANS, there numbers differ sometimes why?), so whats the benefit of your approach and more particularly does integrating a LLM has a benefit? Or is it the different training dataset? Or the audio encoder (BEATs vs. HTS-AT)?
2. No further details for replication of the experiments are given, e.g. pretrained models or the list of species which were hold out.
3. Difficult covariate shift from focal training to soundscape test data is not evaluated, which is one of the major challenges in bioacoustics. E.g. for birds see Stowell,2022 or BirdSet,2024.
4. Call-type task: birds usually have more than two call-types per species. A binary classification task ignoring the specie might have few practical applications.
5. L. 428-219: The cbi dataset consists of XC recordings, your model should have the same advantage as BirdNet and Perch, so the comparison should be fair.
6. Line 071-072: How do you support the claim that the generalization of BirdNet is limited? Did not Ghani et al. claim the opposite?

---

### Language

1. l. 190-192 strong repetition
2. Inconsistent use of abbreviations (e.g. state of the art)

### Questions
1. Could you add ablation studies to access the influence of each part of your approach? What happens if the audio encoder stays frozen? Are the embeddings that BEATs generates already good enough for taking bioacoustics tasks?
2. What SoTA model is used in the call-type comparison? I could not find that in the text.
3. How do you evaluate the LLM without audio and why is it outperforming your model on the gibbons dataset?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
An excellent paper providing a valuable source of bioacoustic data and processing methods to the community. An attempt to comprehensively collect much of the available domain data and curate it into a usable dataset. Plans for distribution and general availability of the dataset/code to the community are missing.

### Strengths
An incredible collection of datasets, and careful curation.
A lot of ancillary code for use of the data in various learning tasks.

### Weaknesses
Unclear from the presentation if the authors intend to make the dataset widely available, and under what license.

### Questions
This is excellent and careful work. I particularly liked the SoTA results in classification

### Soundness
4

### Presentation
4

### Contribution
3
