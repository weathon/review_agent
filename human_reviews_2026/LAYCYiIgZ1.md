# Aurelius: Relation Aware Text-to-Audio Generation At Scale

- Decision: Accept (Poster)
- Scores: 4, 4, 4, 4

## Abstract
We present  Aurelius, a new framework that enables relation aware text-to-audio (TTA) generation research at scale. Given the lack of essential audio event and relation corpora, Aurelius contributes a large-scale audio event corpus AudioEventSet and another large-scale relation corpus AudioRelSet. Comprising 110 event categories, AudioEventSet maximally covers all commonly heard audio events and each event is unique, realistic and of high-quality. AudioRelSet consists of 100 relations, comprehensively covering the relations that present in the physical world or can be neatly described by text.  As the two corpora provide audio event and relation independently, they can be combined to create massive <text,audio> pairs with our pair generation strategy to support relation aware TTA investigation at scale. We comprehensively benchmark all existing TTA models from both general and relation aware evaluation perspective. We further provide an in-depth investigation into scaling existing TTA models' relation aware generation by either training from scratch or leveraging cross-domain general TTA knowledge. The introduced corpora and the findings from investigation potentially facilitate future research on relation aware TTA generation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a relation-aware text-to-audio generation framework and introduces two newly developed datasets, AudioEventSet and AudioRelSet, to improve the modeling of relational structures between audio events. The datasets are organized in a tree-structured hierarchy and are designed to enhance compositionality and relation understanding in audio generation. The authors argue that these datasets can serve as new benchmarks for relation-aware audio generation.

### Strengths
The paper focuses on an important and relatively underexplored topic — relation-aware text-to-audio generation, which aims to capture richer dependencies between sound events beyond conventional text-to-audio mapping.

The proposed datasets include explicit relation-level annotations (such as “arity,” count, and compositional structure), which provide more fine-grained relational information than most existing datasets.

The authors attempt to model hierarchical relations between sound events, potentially offering insights into how complex auditory scenes could be represented in structured data formats.

### Weaknesses
The main contribution lies almost entirely in dataset creation, with no substantial methodological innovation or new generation framework beyond existing relation-aware approaches (e.g., RiTTA or CompA).

The datasets are not released, and only high-level descriptions are given. Without access to examples or samples, it is impossible to verify the data quality or reproducibility.

The tree structure design (depth 3 for AudioEventSet and depth 2 for AudioRelSet) lacks theoretical or empirical justification. It is unclear why this specific hierarchy benefits the relation-aware task or how the root/leaf nodes are defined.

The data generation process appears mostly manual or GPT-assisted, rather than automatic. This makes it resource-intensive, potentially inconsistent, and difficult to scale. The resulting compositions may sound unnatural or ambiguous, especially since all clips are fixed to 10 seconds regardless of content or event duration.

The evaluation methodology is largely inherited from general text-to-audio works and RiTTA, without introducing new metrics to evaluate the proposed hierarchical or relation-specific structures.

Overall, the paper reads more like an engineering dataset report than a research paper introducing conceptual or algorithmic innovation.

### Questions
Previous works such as RiTTA and CompA also address relation-aware text-to-audio generation. Beyond the newly developed datasets, what is the core conceptual or technical difference between this work and prior ones?

How exactly were the datasets collected — are all audio-text pairs manually constructed or verified? If so, how was data distinctiveness and quality control ensured?

Did the authors listen to and validate all audio compositions to ensure naturalness and correct temporal ordering of events?

The proposed tree structure seems to play no role in evaluation — could the authors propose or experiment with metrics that explicitly utilize this structure?

Why was the relation-aware improvement only tested on the Tango model family? Would results generalize to other text-to-audio architectures (e.g., AudioLDM, Make-An-Audio)?

Is there any other way to improve the performance on relation aware?  Not just training on the "new-proposed dataset".

Any demo or public data for the dataset (waveform)?

### Soundness
3

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
4

### Summary
This paper introduces Aurelius, a new framework designed to enable relation-aware text-to-audio (TTA) generation research at scale. The authors also propose two accompanying datasets, AudioRealSet and AudioEventSet, which demonstrate effectiveness in audio generation tasks. Overall, the proposed datasets are valuable contributions to the audio research community. However, the paper lacks novel methodological and technical innovations beyond dataset construction. (But the dataset is a great one!)

### Strengths
- The release of a large-scale, well-curated dataset is always beneficial to the audio research community.

- The authors conduct comprehensive benchmarking of existing TTA models to evaluate the proposed datasets.

- The distinction between AudioEventSet and AudioRealSet is clearly defined. In particular, AudioRealSet provides rich attribute-level annotations for sound events, which is a valuable addition.

### Weaknesses
- My main concern is that, despite the usefulness of the dataset, there are no novel methodological or technical contributions. The paper reads primarily as a dataset paper rather than a technical paper.

- In the Introduction, the term “relation modeling” is not clearly defined.

- In Section 3.1, the phrase “audio events potentially present in the 3D physical world is unclear” is ambiguous — clarification is needed.

- The authors claim that AudioEventSet is more distinctive than AudioSet, but no supporting evidence or analysis is provided. Since AudioEventSet is also manually designed, how do the authors ensure that the 110 sound classes are well-separated and not confusing?

- The size of AudioEventSet should be reported more clearly. As it is built from Freesound and FSD50K (which contains only ~50k clips), it is unclear how large the final dataset actually is or how scalability is achieved.

- In Section 3.2, is AudioRealSet a subset of AudioEventSet? How are the relations labeled — automatically or manually? The writing in this section could be improved for clarity.

- The experimental section lacks comprehensive evaluation of how the proposed datasets improve performance. For instance, how much improvement is observed when models are trained with these datasets compared to without them?

### Questions
N.A.

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
3

### Summary
The paper presents Aurelius, which contains:
(1) AudioEventSet, which contains clean clips for audio events
(2) AudioRelSet, which contains the relative relationship of audio events.

The authors show that, by combining these two resources, we can generate theoretically unlimited relation-aware text-audio pairs. 

The authors also present the evaluation method (benchmarks) for the relation-aware audio.

It also shows that training existing TTA models on the proposed dataset can benefit the relation-aware TTA performance.

The relation-awareness is an important property in current TTA, and this paper is a very good resource.

### Strengths
The paper comprehensively discusses the relation-aware TTA generation. It contains good resources, benchmarks and sufficient discussion.

The relation-awareness is of wide interest in the TTA community. The paper also reveals that the current TTA is not good enough in this direction.

### Weaknesses
(1) The paper is mostly about the resources (data, benchmark) of building relation-ware TTA. For this kind of resource paper, I wonder if the author would make it public.
(2) In section 3, although the author provides a comprehensive design in the data content, they don't mention (1) why the designs are reasonable and (2) how they ensure the intended design philosophy is well implemented in practice. e.g., how they ensure the audio clips in AudioEventSet are precise and clean enough; why such designs in AudioRelSet are reasonable. Such missing information would compromise the contribution of the work.
(3) I'm a bit confused about the mAMSR metric: in the Table 2 caption, you mention its range is [0, 1], but numbers in Table 3 are beyond this range.
(4) Even with this carefully designed data pipeline, the overall mAPre, mARel, etc, are still absolutely low. (e.g., <30%). This very tailor-made dataset seems not to solve the issue very well.
(5) The authors claim that unlimited data simulation is feasible, but scaling up the simulated data is not very effective, as shown in Figure 6.

### Questions
In general, whether the resources would be made public is an important metric for the paper evaluation. Would the authors release them?

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
2

### Summary
This paper introduces Aurelius, a large-scale benchmark framework with two corpora, AudioEventSet and AudioRelSet, that enables systematic evaluation and development of relation-aware text-to-audio generation at scale.

### Strengths
- The paper tackles the well-established limitation of current TTA models in generating audio with accurate temporal ordering and relational structures, which is an interesting and important research problem.
- The framework's approach of combining relation templates with audio events to generate numerous <text, audio> pairs provides excellent flexibility and scalability. The adoption of the "Head-Modifier Structure with Progressive Verb Form" (e.g., "door bell ringing audio" rather than "ringing door bell") ensures syntactic consistency across the dataset.

### Weaknesses
- The GPT-generated templates or synonyms are not always accurate, and some generated texts may not properly correspond to the actual audio events, leading to potential noise in the dataset.
- Since the generated sounds are synthetic, there is no clear way to assess or guarantee their perceptual quality.
- Compared to the existing datasets shown in Table 1, the improvement in performance is not clearly demonstrated. There is no head-to-head comparison between models trained on existing datasets and those trained on Aurelius, making it difficult to quantitatively verify the superiority of the proposed dataset.

### Questions
Please refer to the weaknesses mentioned above.

### Soundness
2

### Presentation
2

### Contribution
2
