# Revisiting Audio-language Pretraining for Learning General-purpose Audio Representation

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 6, 2, 2

## Abstract
Audio-language pretraining holds promise for leraning general-purpose audio representation, yet remains underexplored compared to its vision counterpart. 
Crucially, there is no consensus on whether audio–language models can build effective general-purpose audio encoders, nor a systematic understanding of how pretraining objectives behave across diverse audio processing tasks and scales.
We identify three key barriers: limited large-scale audio-text corpora, insufficient caption diversity, and lack of systematic exploration and evaluation.
To fill this gap, we present the first principled empirical study of audio–language pretraining.
To this end, we introduce CaptionStew, a 10.7M caption dataset aggregating diverse open-source audio-text corpora across multiple domains and captioning styles.
Using this resource, we conduct the first comprehensive evaluation comparing contrastive and captioning objectives for audio representation learning across speech, music, and environmental sound tasks.
Our results not only demonstrate that audio-language pretraining yields competitive, transferable representations, but also reveal critical trade-offs: contrastive learning offers superior data efficiency, while captioning exhibits better scalability.
Furthermore, we find that supervised initialization provides diminishing returns at scale, challenging common practices.
By grounding these claims in empirical evidence, we establish a viable pathway toward general-purpose audio representation learning, guiding future research. To accelerate progress, we will release data preparation recipes, training protocols, and pretrained models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a comprehensive empirical study revisiting audio-language pretraining as a pathway to general-purpose audio representations. The authors identify key challenges in the field: data scarcity, limited caption diversity, and a narrow focus on contrastive learning for retrieval. To address these, they introduce CaptionStew, a large-scale (10.7M captions, 9.3M audio samples) aggregation of diverse, open-source audio-text datasets spanning environmental sounds, music, and speech. Using this dataset, they conduct a systematic comparison of two pretraining objectives: the dominant contrastive learning approach and an underexplored captioning (generative) objective. The evaluation is extensive, covering linear probing on discriminative tasks, audio-language alignment (retrieval, captioning), and open-form question answering. Key findings include: 1) Contrastive learning is more data-efficient and excels at discriminative tasks, while captioning shows promise for language-involved tasks and better scalability in some settings; 2) Supervised initialization from AudioSet provides benefits but diminishes for tasks unrelated to its ontology; 3) Audio-language pretraining can produce competitive, transferable representations across diverse audio domains, establishing its viability for general-purpose audio representation learning.

### Strengths
1. The systematic and comprehensive evaluation is a major strength of the paper. The authors go beyond the standard audio-text retrieval benchmark and evaluate representations across a wide suite of tasks (audio event classification, speaker ID, emotion recognition, music tagging, sound event detection, QA) and protocols (linear probing, retrieval, captioning, LLM-based QA). This provides a holistic and convincing view of the models' capabilities and transferability.
2. The paper provides the first systematic, controlled comparison of contrastive and captioning objectives for audio representation learning. The insights are well-supported by the results across different evaluation protocols and data scales.
3. The commitment to releasing data preparation recipes, training scripts, evaluation protocols, and pretrained models is highly commendable and will significantly accelerate future research in this area.

### Weaknesses
1. The work's foundational elements lack significant innovation. The CaptionStew dataset, while large-scale and practically useful, is an aggregation of existing public datasets without a novel data creation or curation methodology. Similarly, the two primary technical insights—that contrastive learning is highly data-efficient and that generative (captioning) objectives scale well with more data—are well-established principles in machine learning, particularly in vision and language domains. Their demonstration in the audio-language context is valuable but does not constitute a novel finding. The observation that the benefits of supervised pre-training diminish with scaling data also aligns with broader trends in representation learning.
2. The paper's primary contribution is a thorough empirical study rather than a methodological breakthrough. It expertly combines existing techniques (contrastive loss, mixed autoregressive/parallel decoding from CapPa) and resources (public datasets) to perform a systematic comparison. While this is highly valuable for the community, it means the paper does not introduce a model, training method, data (aggregation of existing public datasets), or insights (most are common sense).

### Questions
1. You selected Zipformer for its efficiency and preliminary cross-domain performance. Did you conduct ablation studies with other popular audio backbones (e.g., HTSAT, BEATs, EAT) to ensure the observed conclusions are not architecture-specific? Would you expect the ranking between contrastive and captioning objectives to change with a different encoder?
2. The paper presents a clear dichotomy between contrastive and captioning objectives. It does not explore hybrid approaches (e.g., multi-task learning combining both objectives). Will it show promise and could it combine the strengths of both paradigms?

### Soundness
4

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
3

### Summary
This work revisits audio-language pretraining as a pathway to general-purpose audio representations. The authors introduce CaptionStew, a large-scale dataset created by aggregating multiple existing audio caption datasets. They conduct a comprehensive empirical study comparing two pretraining objectives—contrastive learning and captioning—across a wide range of downstream tasks. Their experiments systematically evaluate the impact of data scale, model initialization, and objective design. The results demonstrate that audio-language pretraining can produce competitive and transferable representations across speech, music, and environmental sound domains, positioning it as a viable approach for universal audio understanding.

### Strengths
The experimental setup is a major strength of this work. The evaluation is extensive, covering linear probing, audio-language tasks, and open-form QA. The controlled experiments on key variables, pretraining objectives (contrastive vs. captioning), data scaling (from 400K to 10M pairs), and initialization (from scratch vs. supervised pretraining)—are exceptionally thorough and provide robust, convincing evidence for the claims. The paper offers several key findings that challenge common practices, such as the diminishing returns of supervised AudioSet initialization on tasks unrelated to its ontology, and the complementary scaling behaviors of contrastive and captioning objectives. These insights are highly valuable for practitioners and researchers in the field.

### Weaknesses
The method for creating the core contribution, CaptionStew, is arguably too simplistic. While aggregating existing datasets is a low-cost solution to the data scarcity problem, it inherits the potential quality issues of the source datasets (e.g., LLM-synthesized captions in WavCaps and AudioSetCaps). The authors acknowledge the limited linguistic diversity in the analysis, but the approach feels unrefined and may limit the upper bound of what the pretrained models can learn.

### Questions
You identify limited linguistic diversity as a key constraint in CaptionStew. Beyond simple aggregation, what more sophisticated curation strategies (e.g., keyword-based filtering, diversity sampling, or using LLMs to rewrite captions for style variation) did you consider, and why were they not adopted?

Given the varying quality of source datasets (e.g., human-annotated AudioCaps vs. LLM-synthesized WavCaps), did you observe any correlation between the source of a caption and downstream task performance? For instance, do captions from ParaSpeechCaps lead to better performance on speaker-related tasks?

The mixed autoregressive/parallel decoding (inspired by CapPa) is a key component of your captioning objective. Could you ablate the contribution of each mode? Does the parallel decoding, which forces stronger audio dependency, contribute more to the learned representation quality than the standard autoregressive mode?
The results suggest captioning has an advantage in open-form QA, a more generative task. Do you think this is because the captioning objective inherently trains the audio encoder to be more compatible with a language model's reasoning process, or simply that it learns a richer set of semantic concepts?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper explores the potential of audio-language pretraining (ALP) as a pathway to develop general-purpose audio representations that can transfer effectively across diverse domains, including speech, music, and environmental sounds. The authors argue that ALP—by grounding audio perception in natural language—offers a unified learning framework with strong promise, though existing models have primarily excelled in retrieval tasks and lag behind vision-language models like CLIP in serving as general-purpose encoders. They identify three key limitations hindering progress in ALP: the scarcity of large-scale audio-text corpora, limited caption diversity, and the absence of systematic evaluations of pretraining objectives.

To address these challenges, the authors introduce CaptionStew (CS10M), a large-scale dataset comprising 10.7 million captions paired with 9.3 million audio samples spanning over 37,000 hours across multiple domains. CaptionStew aggregates and harmonizes several open-source corpora such as AudioCaps, WavCaps, MusicCaps, and ParaSpeechCaps, thereby enhancing the diversity of captioning styles—from concise human annotations to detailed, LLM-generated narratives and domain-specific descriptors. Although CaptionStew significantly broadens the vocabulary (56,586 unique words), its lexical diversity remains lower than that of mature image-text corpora, potentially constraining scaling gains for certain fine-grained tasks.

The paper also presents a systematic comparison of pretraining objectives, contrasting contrastive learning (a discriminative, InfoNCE-based approach) with captioning objectives (generative, via autoregressive or parallel prediction). Contrastive learning demonstrates superior data efficiency at smaller scales and stronger performance on linear probing tasks such as audio event classification and speaker identification. In contrast, the captioning objective scales better with larger datasets and performs slightly better on language-involved tasks, such as open-ended question answering and caption generation. Notably, the performance gap between these objectives narrows considerably when adaptive pooling mechanisms (e.g., multi-head attention pooling) replace mean pooling, underscoring the influence of downstream architectural choices.

Further experiments examine the effects of initialization and data scaling. While initializing the audio encoder with supervised models pretrained on AudioSet (AS SL) yields immediate performance boosts—particularly for discriminative objectives—these benefits diminish or vanish at larger data scales. This suggests that diverse caption-based supervision can rival or even surpass domain-specific supervised initialization when training for general-purpose representation learning. Scaling analyses reveal steady performance improvements across most tasks with increasing data, except in sound event detection, where performance declines as caption data grows, possibly due to conflicts between global semantic supervision and the temporal precision required for event localization.

In conclusion, the study demonstrates that audio-language pretraining produces competitive and transferable audio representations, marking it as a promising strategy toward universal audio understanding. Conceptually, general-purpose audio representation can be likened to a universal dictionary for sounds: while traditional supervised methods create specialized, domain-limited dictionaries, ALP—empowered by diverse descriptive language from CaptionStew—builds a flexible, semantic scaffold capable of defining both a "voice timbre" and a "rhythmic pattern." In this analogy, contrastive learning efficiently constructs a strong core vocabulary, whereas generative captioning gradually cultivates a broader and more scalable comprehension of sound when given sufficient data.

### Strengths
The paper demonstrates several notable strengths, particularly through its systematic approach to addressing existing limitations and its comprehensive empirical validation of audio-language pretraining (ALP) as a pathway to developing general-purpose audio representations. One of its most significant contributions is the introduction of the CaptionStew (CS10M) dataset, a large-scale and diverse resource specifically designed to overcome the scarcity of extensive audio-text corpora that has long hindered progress in ALP. CaptionStew comprises 10.7 million captions paired with 9.3 million audio samples, totaling over 37,000 hours of audio—an order of magnitude larger than prior datasets. Its diversity stems from the aggregation of multiple open-source corpora across speech, music, and environmental sounds, coupled with varying captioning styles, including human-annotated descriptions, LLM-generated narratives, and fine-grained annotations of speaking or musical attributes. The paper further validates this diversity through t-SNE visualizations of sentence embeddings, demonstrating the unique descriptive focuses of each corpus and highlighting how their complementary linguistic perspectives enrich model generalization.

Beyond data creation, the study conducts the first systematic comparison between contrastive and captioning pretraining objectives across multiple audio domains. This comprehensive evaluation reveals clear complementary strengths: contrastive learning proves more data-efficient and excels in linear probing tasks like classification, whereas captioning objectives exhibit better scalability and stronger performance in language-dependent tasks such as open-form question answering. Importantly, the authors show that the performance gap between these objectives narrows considerably when adaptive pooling mechanisms, such as multi-head attention pooling, are used instead of simpler mean pooling—underscoring the critical role of downstream architecture in fair representation assessment.

The paper also challenges common practices in model initialization, particularly the reliance on supervised pretraining from datasets such as AudioSet (AS SL). While supervised initialization offers early-stage benefits, the results demonstrate that these advantages diminish or vanish entirely at larger data scales. Moreover, such pretraining can bias models toward event-level semantics, potentially hindering performance on tasks requiring fine-grained acoustic understanding, such as speaker identification or music tagging. This finding questions the necessity of domain-specific supervision and highlights the potential of large-scale, diverse caption data to serve as a self-sufficient foundation for semantic learning.

### Weaknesses
A major contribution of the paper, the CaptionStew (CS10M) dataset, also represents a few limitation. While its scale and diversity mark a substantial advance for audio-language pretraining, the authors recognize several issues inherent in its design. First, CaptionStew’s reliance on LLM-synthesized and augmented captions introduces potential biases and artifacts, as no extensive human verification or quality control was conducted across the dataset. This could affect the fidelity and reliability of learned representations. Second, despite an expanded vocabulary of over 56,000 unique words, the dataset exhibits low lexical diversity overall. Aggregating multiple corpora does not inherently ensure rich linguistic variation, and CaptionStew’s Distinct-n metrics remain considerably lower than those observed in mature image-text datasets. This limitation, partly attributable to certain constituent datasets with low linguistic diversity, may have contributed to weaker scaling behavior in tasks like emotion recognition and musical instrument classification, where nuanced language is essential for robust representation learning.

Beyond dataset concerns, the paper’s technical novelty is relatively limited. The work primarily serves as an empirical and systematic study, combining existing methods such as contrastive learning, captioning objectives, and large-scale data aggregation rather than introducing fundamentally new algorithms or architectures. Techniques like the mixed autoregressive/parallel training approach for captioning are adapted directly from prior vision-language models (e.g., CapPa), and the architectural foundations follow well-established conventions. The authors justify this approach by emphasizing their aim to provide a comprehensive comparative analysis rather than a methodological innovation—an important contribution in a field that previously lacked rigorous cross-objective evaluations.

Finally, the study’s scalability constraints limit the generalizability of its conclusions. The experiments are conducted on a dataset of roughly 10 million audio-text pairs using relatively modest model sizes, far smaller than the billion-scale datasets and architectures typical in modern vision-language research. As a result, the findings may underestimate the full potential of audio-language pretraining, particularly for the captioning objective, which appears to benefit disproportionately from increased scale. The authors note that achieving parity between captioning and contrastive methods on certain benchmarks may require scaling to hundreds of millions of audio-text pairs. Additionally, the study does not explore more advanced architectures or integration with large language models, which could further enhance performance. These omissions stem not from oversight but from deliberate experimental control and computational limitations.

### Questions
Thanks for the paper, let's work on this paper collectively to make it a better publication. Kindly address all of these questions.

1. [Line 50] : "diverse audio modalities" - A bit vague, define what you wish to mention, what kind of diversity are you refereeing? And what are different audio modalitties? 

2. [Line 60] : audio-language pertaining : I recommend you to cite few other papers which are there in this field (Have previously reviewed these work) : 1) Sinha, Anshuman, et al. "Enhancing Audio-Language Models through Self-Supervised Post-Training with Text-Audio Pairs." arXiv preprint arXiv:2408.09269 (2024)., 2) Ghosh, Sreyan, et al. "Compa: Addressing the gap in compositional reasoning in audio-language models." arXiv preprint arXiv:2310.08753 (2023).

3. [Line 72] : Define the audio spectrum which you wish to mention.

4. [Line 83-96] : Refers the sections, figures, plots where you have addressed these four points in each bullet point. It helps the readers.

5. [ Line 141] : For captioning objective, how are you stating that this is a general objective? Do you think auto-regressive generation of captions from audio is a more general way of learning representations than a contrastive model? How do you define "generality" in terms of learning?

Tbh, the auto-regressive training which you're implementing is actually an imitation learning method (supervised learning). Do you think this method is more general than a self-supervised contrastive learning method?

6. [Line 143] : denser token-level supervision? What do you mean by density in this context? How do you claim this?

7. [Line 160] : Parallel mode enforces stringer dependency ... ; Is this a claim or a proposition? If it's a claim, do you have results supporting this claim? If it's your proposition, then what makes you can write "we assume/ think .... ; and further provide evidence of this effect; although the evidence need not be strong".

8. [Line 180-205] : Can you please get a figure of this section and replace the long texts; Usually other good publications have demonstrated a figure for their data strategy. Makes the life easier for a reader.

9. [Line 185] : What do you mean by single generation pipeline?

10. [Line 203] : Since this pre-training is done over a huge dataset which covers different aspects of audio, how do you know whether it's your model strategy or just the data coverage?

11. [Table 3] : 

(a) What is AEC, SID, etc? The core results of the model look a bit on the weaker side; You've compared the strengths of your work with a baseline (Zipformer) on which you've further post-trained on specialized datasets; right? Did we expect any other result than the one which have been posted?

(b) How do these results support your initial claim of making a "genera" purpose audio encoder (which essentially learn a general purpose audio representation.)?

(c) Put Section 4.4 in bullets; the crux of the paper shouldn't be in long paragraphs.

12. Where is the ablation study which compares the effects of adding different data sources, as you mentioned something about adding different kinds of data; now what effects does it have on your model? Or did you just simply added a mix and expected everything to work? (If we understand deep how deep learning works, a mix of everything would more often than not work; but your paper is not aimed just at that. You wish to show how to make a general purpose audio encoder.)

13 [Figure 3] The t-SNE plot show some grouping, but it does not reflect anything on your learning! Moreover do you have any data on how separate the groups are? Because to me only two of the group (JamendoMaxCaps and ParaSpeechCaps seem distinct).

That's all for now, let's discuss the above. All the best

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper conducts experiments to show that large-scale audio-language pretraining (with contrastive/captioning supervision) can yield good general-purpose audio representations. 

However, the technical novelty of this paper is limited. The effort on the dataset is just an aggregation of several existing open-source datasets, and it’s misleading to create a new name for this dataset, as there is no new content being added. The method used in this paper is also of limited novelty. The work in this paper is mostly experimental.

### Strengths
1. It's useful to see how model performance scale with the dataset size.

### Weaknesses
1. The paper has a lot of missing details in writing:
What do the settings “PSC” and “MC” mean in Table 3.b?
What do “AEC,” “SID,” “SER,” “MTAG,” “INST” and “SED” mean in Table 3.a? Please do not assume readers are already familiar with these acronyms. 
2. The paper's experimental setting is unclear. It shows a lot of settings, but the exact setting of the method is not mentioned. Please do not assume reader can understand these setting without explaination.
For example, the author list results of “Contrastive-scratch,” “Captioning-scratch,” etc. in the table, but there are no corresponding explanations of these settings. This can cause confusion for the reader. 
3. Please unify the formatting for w2v in Table 3.a
4. The paper mentioned they release the dataset and related model but it’s not found anywhere in the paper.
5. The overall technical contribution is not strong enough.

### Questions
Using captioning to perform audio language pretraining is not a common practice. I'm wondering why authors are interested in this scheme, especially considering that captioning pretraining seems to always perform worse than contrastive learning (Figure 2).

### Soundness
3

### Presentation
2

### Contribution
2
