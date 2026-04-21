# Vision-Language Integration in Multimodal Video Transformers (Partially) Aligns with the Brain

- Avg Score: 3.75
- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 1, 5, 6

## Abstract
Integrating information from multiple modalities is arguably one of the essential prerequisites for grounding artificial intelligence systems with an understanding of the real world. Recent advances in video transformers that jointly learn from vision, text, and sound over time have made some progress toward this goal, but the degree to which these models integrate information from modalities still remains unclear. In this work, we present a promising approach for probing a pre-trained multimodal video transformer model by leveraging neuroscientific evidence of multimodal information processing in the brain. Using brain recordings of participants watching a popular TV show, we analyze the effects of multi-modal connections and interactions in a pre-trained multi-modal video transformer on the alignment with uni- and multi-modal brain regions. We find evidence that vision enhances masked prediction performance during language processing, providing support that cross-modal representations in models can benefit individual modalities. However, we don't find evidence of brain-relevant information captured by the joint multi-modal transformer representations beyond that captured by all of the individual modalities. We finally show that the brain alignment of the pre-trained joint representation can be improved by fine-tuning using a task that requires vision-language inferences. Overall, our results paint an optimistic picture of the ability of multi-modal transformers to integrate vision and language in partially brain-relevant ways but also show that improving the brain alignment of these models may require new approaches.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper uses neuroscientific evidence, specifically brain recordings from people watching a TV show, to probe a pre-trained multimodal video transformer model. The authors uses the human brain as a benchmark, specifically brain recordings of participants watching a TV show, to analyze the alignment of a multi-modal video transformer with both uni- and multi-modal brain regions.The findings suggest that vision can enhance language processing within the model. However, the joint multi-modal transformer representations don't capture more brain-relevant information than individual modalities alone. The results indicate that multi-modal representations can significantly enhance alignment with language regions. The added alignment from visual input majorly benefits masked language prediction. However, the joint multi-modal transformer doesn't offer more brain-relevant information than individual modalities. Fine-tuning the model for a vision-language task enhances its brain alignment.

### Strengths
1. The paper has some interesting novel findings such as - a)no evidence of brain-relevant multimodal interactions encoded in the pre-trained model. b) Early and middle layers of the pre-trained and fine-tuned model are similarly brain aligned c)Task-dependent changes in top layers are not aligned with brain representations. d) Fine-tuning for vision-language question-answering improves brain alignment in some regions.

### Weaknesses
1. The paper suggest that while multi-modal transformers show promise in integrating vision and language, there's room for enhancement in aligning them with brain processes. But it is not clear what might be the potential approaches for better alignment
2. The experimentation settings lacks some clarity. For example, did all the participants watching the TV show actually seeing it first time? Or some of them might have seen this show before? Have they heard about the name of the show/synopsis of the show before?
3. The flow of the paper is hard to follow, the writing could be more concise
4. The paper has some interesting novel findings on brain alignment but it is not clear how findings like these are actually impactful

### Questions
1. Will it be possible to share some more details on the experimentation settings? For example, did all the participants watching the TV show actually seeing it first time? Or some of them might have seen this show before? Have they heard about the name of the show/synopsis of the show before?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
1: strong reject

### Rating Number
1

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The authors examine non-invasive functional magnetic resonance imaging (fMRI) measurements while participants watch a TV show. The authors use multimodal video transformer neural network models to attempt to fit the fMRI signals. Based on these analyses, the authors claim that vision data improves fitting performance during language but also argue that the multimodal transformers do not provide any better fitting compared to models that focus on the individual vision and language modalities.

### Strengths
The authors use state-of-the-art transformer models

The authors ask interesting questions about the potential for shared representation of vision and language information

The highly uncontrolled nature of the stimuli (TV shows) makes the interpretation complex but is also interesting in bringing the questions to real-world relevance.

### Weaknesses
I could not find any evidence in the paper that the fMRI signals provide either visual or language information during the task. The first figure after the initial definitions goes on to show correlations between the neural network representations and the fMRI signals but there is no indication of what those fMRI signals are actually representing. 

The overall results are extremely weak. In the best case scenarios the Pearson correlations are about 0.1 and in most cases, they hover between 0.01 and 0.05. The fraction of explained variance is the square of the Pearson correlation coefficient. With a correlation of 0.1, that means that the neural network, in the best-case scenario, can explain around 1% of the fMRI signals.  

To make matters worse, the conclusions that the authors are interested in drawing are based on comparisons between different conditions. Take for instance, the first two columns in the second panel in fig. 2 (pCingulate). Vision+audio yields a correlation of about 0.03 (that is, about 0.0009 of variance explained), whereas only audio yields a correlation of 0.02 (that is, 0.0004 of variance), and vision yields a slightly negative (!) correlation. Conclusions are drawn based on a difference of 5x1^(-4) in variance!

### Questions
There are many datasets available with higher quality data (two-photon calcium imaging, neurophysiological recordings). I suggest focusing on datasets with higher quality if the goal is to better understand brain function and build neural network algorithms that can correlate with neural representations.

### Soundness
1 poor

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The study delves into the mechanisms that drive the efficacy of multi-modal models, a topic still under exploration in the machine learning realm. The authors aim to understand this phenomenon by drawing parallels with the human brain, known for its adeptness at integrating multiple modalities. They bridge this gap by juxtaposing fMRI activity with the activations of a vision-language model when exposed to the same TV show stimuli. Through their investigation, they discovered that (1) Cross-modal models enhance brain alignment with individual modalities; (2) Visual inputs improve masked language prediction, notably in some brain regions such as the language-focused Angular Gyrus; (3) The pre-trained model lacked evidence of brain-relevant multi-modal interactions; (4) Fine-tuning for vision-language tasks improved the alignment in some brain regions.

### Strengths
Utilizing the brain as a benchmark for multi-modal integration provides an intriguing perspective, especially for the machine learning community. Over the past decade in neuroscience, spurred by seminal works like that of Yamins et al. (2014), there has been a surge in efforts to correlate deep neural networks with brain activity, yielding significant insights for neuroscience. In this study, the authors ingeniously invert this approach, seeking to extract insights from neuroscience to better comprehend expansive machine learning models. This innovative approach is not only captivating but also holds promise if appropriately substantiated.

### Weaknesses
While the premise of the study is intriguing, it could benefit from further refinement. A primary concern is the treatment of multi-modal integration in the human brain as a static rather than a dynamic process[2]. Human sensory processing occurs in sequences — visual information traverses from the retina through the thalamus to the visual cortex before reaching the prefrontal cortex. Audio processing follows a distinct timeline, and semantic systems kick in later. Additionally, fMRI signals exhibit a delay spanning several seconds. In contrast, vision-language models lack these temporal delay characteristics. The study's definition and measurement of alignment seem to bypass these temporal nuances, making it a potential oversight.

Furthermore, the research could have delved deeper into cross-comparisons of different fMRI datasets and vision-language models. It's imperative to discern whether findings remain consistent across various model choices. If one model aligns more closely with the brain than another, what implications arise from this? Does a higher alignment score for Model A over Model B necessarily denote its superiority? And if so, by what margin? An intriguing proposition would be to investigate the effects of fine-tuning a model based on its brain alignment score. What outcomes would this entail, and what do these outcomes signify?

Lastly, the authors' claim about audio-visual information being processed in the Angular Gyrus — a prominent language region — isn't groundbreaking. This observation is already documented in the neuroscience literature [1,2,3,4].

[1] Thakral, Preston P., Kevin P. Madore, and Daniel L. Schacter. "A role for the left angular gyrus in episodic simulation and memory." Journal of Neuroscience 37.34 (2017): 8142-8149.

[2] Chambers, Christopher D., et al. "Fast and slow parietal pathways mediate spatial attention." Nature neuroscience 7.3 (2004): 217-218.

[3] Fang, Mengting, et al. "Angular gyrus responses show joint statistical dependence with brain regions selective for different categories." Journal of Neuroscience 43.15 (2023): 2756-2766.

[4] Bonnici, Heidi M., et al. "Multimodal feature integration in the angular gyrus during episodic and semantic retrieval." Journal of Neuroscience 36.20 (2016): 5462-5471.

### Questions
Based on the weakness, I have the following questions: 
## 1
Given that multi-modal integration in the human brain is a dynamic process with distinct timelines for visual and audio information processing, and considering the inherent delay in fMRI signals, how do you account for these temporal nuances when defining and measuring alignment with vision-language models? Do you believe that the lack of such considerations could impact the study's outcomes, and if so, how?

## 2
You've provided alignment scores for the vision-language models in relation to brain activity, but how might these scores translate to practical implications? For instance, if Model A aligns more closely with the brain than Model B, does it necessarily signify a performance advantage? Furthermore, have you considered the impact of fine-tuning a model based on its brain alignment score, and what potential outcomes might you anticipate from such an endeavor?

## 3
The observation regarding the processing of audio-visual information in the Angular Gyrus is noted in several pieces of existing neuroscience literature [1,2,3,4]. Could you elaborate on how your findings differ or expand upon these previous studies, or provide novel insights that set your research apart from these established understandings?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
1: You are unable to assess this paper and have alerted the ACs to seek an opinion from different reviewers.

### Summary
This paper studies the alignments between human brain and multimodal video transformers. Two relations are studied: cross-modal connections and multi-modal interactions. Cross-modal connections is defined as the shared information that exists when different modalities are related. Multi-modal interactions is defined as the novel information that arises when these modalities are integrated. To do this study, video clips from a TV show are selected, where brain recordings are available. To obtain transformer representations, MERLOT Reserve is selected, which has three encoders for image, audio, and text, respectively, and a joint encoder to aggregate the information from these three modalities. Experimental results show that the multimodal video transformer partially aligns with vision-language integration in the brain. It is also shown that vision can contribute to language processing in the brain.

### Strengths
The topic and study are interesting. It is interesting to study to what extent the current large models align with human brain. This paper has also shown some interesting findings. For example, vision modality significantly improves brain alignment over language regions, etc.

### Weaknesses
I would need to say sorry, this paper is out of my domain. I am not able to provide a valid assessment, and I am unable to fully understand this paper. There are so many terms that I don't understand without background. For example, "repetition time", "voxel", "vision-language integration", etc.

In addition, I think the writing is unclear and unconcise to me, which might be one of the reasons that make me not able to understand this paper. There are many very long sentences. For example, "This improvement cannot be due to the further processing of language-specific information in the joint encoder since the depth of language input processing is the same in both conditions and is unlikely to be due to vision-only information since these regions are known to support language processing." These make it very hard to read and understand the whole paper in additional to unknown terms. If that is not my problem, then I think the paper writing needs to be improved.

I do have one suggestion: Section 3.1 "Model Representations" should be named as "Transformer Representations" or "Artificial Neural Network Representations". As this paper is across domains, it is better to clearly name each term for that specific domain. "Model" can represent many things. But here I think you are referring to the transformer.

### Questions
N/A

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good
