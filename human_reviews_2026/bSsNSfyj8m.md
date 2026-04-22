# The Human Brain as a Dynamic Mixture of Expert Models in Video Understanding

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 6

## Abstract
The human brain is the most efficient and versatile system for processing dynamic visual input. By comparing representations from deep video models to brain activity, we can gain insights into mechanistic solutions for effective video processing, important to better understand the brain and to build better models. Current works in model-brain alignment primarily focus on fMRI measurements, leaving open questions about fine-grained dynamic processing. Here, we introduce the first large-scale model benchmarking on alignment to dynamic electroencephalography (EEG) recordings of short natural videos. We analyze 100+ models across the axes of temporal integration, classification task, architecture, and pretraining, using our proposed Cross-Temporal Representational Similarity Analysis (CT-RSA) which matches the best time-unfolded model features to dynamically evolving brain responses, distilling $10^7$ alignment scores. Our findings reveal novel insights on how continuous visual input is integrated in the brain, beyond the standard temporal processing hierarchy from low to high-level representations. After initial alignment to hierarchical static object processing, responses in posterior electrodes best align to mid-level temporally-integrative action features, showing high temporal correspondence to feature timings. In contrast, responses in frontal electrodes best align with high-level static action representations and show no temporal correspondence to the video. Additionally, temporally-integrating state-space models show superior alignment to intermediate posterior activity, in which self-supervised pretraining is also beneficial. We draw a metaphor to a dynamic mixture of expert models for the changing neural preference in tasks and temporal integration reflected in the alignment to different model types across time. We posit that a single best-aligned model would need such training and architecture as to allow combining and dynamically switching between these capacities.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
1

### Summary
This paper presents the first benchmark for the alignment between artificial image / video representations from neural networks and dynamic brain responses to natural videos, recorded using EEG.  This benchmark is carried out on a new dataset EEGMD (not public yet) that extends the Bold Moments Dataset (BMD) to EEG responses (to natural videos). To compare between models, the authors introduce a time-aware extension of the RSA method: Cross-Temporal Representational Similarity Analysis (CT-RSA), that allows computing dissimilarity between time-courses of EEG and model representational dissimilarity matrices. This allows them to make a number of significant findings about how the representations in artificial models and brains correlate across time and brain regions.

### Strengths
- The paper is well written, easy to follow, the methodology and metrics are clearly defined
- The assumption resulting from the analysis of the paper, that is the brain as a dynamic mixture of experts, is indeed supported by the results of the benchmark across the different architectures, time-aware / static models, model types, ...
- There are a number of conclusions that are new to the best of my knowledge (e.g. 'frontal activity being best capture by early static semantic action representations')

### Weaknesses
- The measurement of alignment is rather complex (UNC-scaled Spearman's rho) and the paper misses a discussion of why using this metric is superior to much simpler methods (e.g. just using Pearson's r across time). Since cluster analysis is absent, the advantage of using RDMs is not obvious.
- The paper makes a number of interesting findings but does not seem to delve into their interpretations (e.g. SSMs better aligning to posterior processing)

### Questions
- Apart from the already presented axis of analysis, could the authors present results correlating alignment with task-related performance of video / image embedding models on a given task ?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper presents the first large-scale benchmarking of over 100 deep neural networks against EEG recordings collected during natural video viewing. Using a novel Cross-Temporal Representational Similarity Analysis (CT-RSA), the authors analyze alignment dynamics between model features and brain responses. Results reveal distinct temporal processing stages in posterior and frontal electrodes, suggesting that brain activity during video perception reflects a dynamic mixture of expert models tuned to different representational demands.

### Strengths
* Interesting large-scale study linking deep video models with dynamic EEG signals.
* Introduction of CT-RSA as a principled extension of RSA to capture cross-temporal alignment.
* Rich, interpretable findings showing stage-wise specialization in brain responses and model correspondences.
* Strong relevance to both neuroscience and AI, offering biologically inspired insights for model design.

### Weaknesses
* While conceptually interesting, the empirical novelty beyond previous fMRI-based benchmarks feels incremental, focusing mainly on EEG substitution.
* The EEG dataset is limited to six subjects, which may restrict generalizability.
* Some claims about "mixture of experts" remain metaphorical rather than computationally substantiated.
* The paper does not analyze the robustness of CT-RSA to preprocessing or electrode partitioning choices, nor quantify its improvement over standard RSA or fMRI-based alignment methods, leaving the methodological advantage insufficiently demonstrated.
* To what extent do the findings generalize across individuals, given the small sample size?

### Questions
See above.

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
2

### Summary
This study compares deep learning models to brain activity (EEG) and finds that the brain processes videos like a "dynamic mixture of experts." Different brain regions align with different model typesposterior areas with mid-level dynamic features and frontal areas with high-level static actions. The findings suggest that a single, ideal AI model would need to flexibly combine these capabilities.

### Strengths
1. The paper is clearly written and well-organized, making it reader-friendly.

2. The article conducted a large number of rigorous experiments and collected new EEG data to explore this issue.

3. The article provides a reasonable explanation for all the results obtained from the experiment, and it is consistent with some existing conclusions in the field of neuroscience and model explanations.

### Weaknesses
1. From several articles about brain and model alignment (not limited to video understanding), they all stated that aligning research can help in designing better deep learning models. However, no improved models have been created based on these works. Each time a new model is developed, alignment and explanation are made based on it (e.g., Mixture of Experts, MoE has been used for several years, not based on this article). Even some brain activity explanations of BERT [1] have now been considered outdated by engineering standards. Could the author provide me with some examples of recent studies on brain activity that actually guide the design of deep learning models?

2. It is uncertain how reliable the research on EEG signal alignment is. In my view, EEG signals are very blurry and unstable. They contain a lot of noise, and even for the same subject, EEG responses to the same stimulus can vary when collected multiple times. Did the author first verify the reliability of the collected EEG signals? For example, conducting multiple collections to verify consistency, etc.

3. Furthermore, all the video clips of the experimental studies were only 3 seconds long. It is completely fine to explore the basic visual responses or decode visual signals within this length. But can we really explore the complex reactions of the brain within this time window? (For example, video understanding) Some of the conclusions in the article may have little connection with video understanding. For instance, the initial brain activity of each video segment might merely indicate that we have just viewed a new video. Among the fleeting content, this article not only studies comprehension but also examines brain activity and the changes in the model over time. I'm not sure if this is reliable.

4. Furthermore, current research mainly focuses on alignment, meaning which time points show changes in brain activity. However, we do not understand what these changes actually represent. Although the author provided an explanation, it does not serve as proof. In other words, these changes can be interpreted in various ways, and there is no way to tell which interpretation is correct. It is also unreasonable (no evidence) to assume that the model should match the EEG (brain) activity. In other words, we neither understand how the brain works nor grasp why the model functions. So, does alignment make any sense in this situation?

[1] BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding


Declaration: I lack profound background knowledge of brain-model alignment. But I have interests and questions for this field. If the questions raised are unreasonable or if there are common limitations in the field, the author can provide an explanation. Also, please ACs consider the weight of my questions as appropriate.

### Questions
Please see the Weaknesses section.

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
This paper presents the first large-scale representational alignment benchmarking study between human EEG recordings during natural video viewing and deep neural networks. They benchmark 110 architectures, including imaging models, video models, and state-space model and using a Cross-Temporal Representational Similarity Analysis (CT-RSA), that extends the classic RSA analysis along the temporal dimension. It works by building representational dissimilarity matrices (RDMs) for each EEG timepoints and for each model layer and timepoints across all videos, correlating them to form a cross-temporal map of brain-model correspondence. By extending RSA into the temporal domain, the method aims to reveal when and how neural network representations align with evolving brain activity during continuous visual perception.

### Strengths
The CT-RSA framework is an elegant methodology, allowing dynamic model-brain alignment to be explored at high temporal sampling rate using EEG datasets. The paper contributes to bridging fMRI-based alignment work (limited by slow dynamics in general) and fast EEG dynamics, offering complementary insights into temporal processing. The framework is generalisable and could serve as a benchmark pipeline for cross-modal studies comparing other neural modalities (MEG, ECoG, fMRI)

The paper is globally well-written paper, with thorough contextualisation (related work section), and an extensive discussion sections. 

The study covers an unusually wide model space (over 110 architectures).

The results uncover a meaningful temporal hierarchy: posterior EEG activity transitions from low-level visual encoding to mid-level integrative representations, while frontal activity remains stable and semantically oriented.

Interesting exploratory results between frontal and posterior electrodes encoding preference w.r.t various types of models (image, video) and training schemes (pre-training, finetuning only)

### Weaknesses
Despite the large-scale benchmarking, it looks like the neuroscientific interpretation remains underdeveloped/underexploited: 
- The paper provides little theoretical discussion linking the observed dynamics to known cortical hierarchies or cognitive models of video perception.
- It is not clear why this segregation (posterior/frontal) of electrodes was chosen. Was other cluster of electrodes tested? What is the limit of this choice? 

Regarding the methods, one could highlight a few blind spots in the methodology: 
- The temporal stages (I-IV) are defined by visual inspection rather than a formal statistical segmentation or theoretical motivations, which weakens their interpretability.
- While CT-RSA is a strong method, the heavy reliance on correlation-based RSA prevents causal or mechanistic insights; additional decoding analyses could strengthen the conclusions.
- The methodology does not account for the spatial-temporal organisation of brain activity or the existence of functional networks or prior anatomical organisation; instead, it effectively collapses the EEG pattern at each time step into a single similarity value, losing information about spatial interactions and network-level dynamics.
- It would have been interesting to evaluate differences between subjects (if any?) or mentioned it otherwise. 

Results: 
- The paper could benefit from a clearer distinction between what is novel and what is confirmatory from the previous fMRI work (Sartzetaki et al. (2025)).
- It is difficult to draw any clear conclusions regarding the results of different type of DL architecture and training schemes, and how it compares to previous work with fMRI.  

Limitations of the methods: 
- There is not enough discussion of the limits of using EEG for this kind of study (spatial resolution), despite the advantage of the temporal resolution. How do you approach this tradeoff? 
- Some part could benefit from more clarity: "Responses in posterior electrodes, after initial alignment to hierarchical static object processing, best align to mid-level representations of temporally-integrative actions and closely match the unfolding video content."
- Figure 1 could be more clear.

### Questions
- The alignment of CNNs with frontal EEG activity seems counterintuitive; could the authors expand on this? 

- How does correlation metrics evolve with relative models performance of image and video models on there respective benchmarks? Related to that, could the authors clarify how self-supervised models were selected and whether their pretraining objectives influence alignment differently than supervised models?

- How robust are the results to variations in video duration or temporal resolution - for example, would similar dynamics appear with longer or continuous movie stimuli? what do you expect? 

- How sensitive is CT-RSA to differences in model frame sampling rates? what would happen if video models were trained on the same sampling rate as the EEG signal is processed? 

- Could CT-RSA be extended to multimodal representations (audio-visual or language-conditioned models) to explore cross-modal dynamics in brain alignment? One could think of movie-watching experiments (mostly with fMRI) for this kind of studies as some aspects of cognition can only be unlocked from longer stimuli, especially with complex stimuli such as audio-visual. 

- It is not a strong opinion, but the title of the paper might be misleading with respect to the reference to "Mixture of Experts" (MoEs). The “mixture of experts” framing is conceptually interesting but remains speculative; the authors do not provide evidence of explicit modular switching mechanisms.

### Soundness
3

### Presentation
3

### Contribution
2
