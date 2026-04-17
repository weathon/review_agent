# Deep neural network model of sound localization replicates “what” and “where” representations in auditory cortex

- Decision: Reject
- Scores: 2, 4, 2, 4

## Abstract
Unlike visual cortex, whether auditory cortex has parallel pathways for sound identification (“what”) and localization (“where”) is debated. It also lacks a topographic map of auditory space, like the retinotopy in visual cortex. Here, we examined auditory “what” and “where” representations in deep neural network models of sound localization. Surprisingly, the models learned well-separated clusters by sound type, but not by sound level. Sounds were further organized by spectrogram similarity, and these organizations were aligned with human spatial hearing. Sound type also determined whether the representations of sound locations organized into a map: maps were formed in both the horizontal and vertical planes when sounds contained binaural and monaural localization cues that were topographically organized relative to human ears. However, formation of a spatial map did not improve, but rather deteriorated, both the model’s and humans’ localization accuracy. Together, our model suggests that “what” cannot be dissociated from “where” in auditory cortex. A space map is created by spatially organized localization cues and is unnecessary for auditory cortex.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
In this work, the Authors invesigate the co-localization of spatial and contentual representations in auditory processing. Motivated by the question of how these representations are processed in the brain, they trained a neural network where they investiigated the relations between the representations of the content and location in the auditory processing. As a result, they provide a comprehensive display of these representations on different stages of auditory processing in their model.

### Strengths
- Thorough literature review

The paper offers a substantial literature review that clearly motivates the problem at hand and provides an account of the related work on it. Different theories / models are mentioned and their support / valifdity is thoroughly discussed. This offers a nice introdcution into the topic and the related research, clearly builing up to the gap that this work aims to address.

- Clear text, high-quality figures.

The text overall is well-structured. It clearly devivers the message of the work and substantially describes the steps taken within the research project, paired with the corresponding results. The figures are clear and intuitive, showing the work's results chiefly throug the visualized dimensionality reduction / clustering techniques, easy to visualize parse. The font are of human-readable size, which is rare among the submissions here.

- Well-founded approach

I am personally a big fan of this approach to understanding and interpreting the representations learned by deep-learning models on various levels. In auditory processing, this approach has been highly successful in the past: Applied to Transformer models trained for speech recognition (Toyota institute circa 2019), it helped determining which layers of the Transformer stack encode syllables, words, sentences, and semantic content shich, in turn, has allowed to optimize the models. Naturally, the same approach can be applied to the broader spectrum of auditory tasks, as done in this paper.

- Comprehensive investigation

As represented in the many figures of the paper, the representations of the sound have been compared to a large set of variables, resulting in the comprehensive investigation of the representations learned by the model.

### Weaknesses
While I find the overall approach taken in this paper to be proper and interesting, there are two considerations regarding to the way it's used here.

- Limited to one network architecture

While comprehensive, all the analysis is performed for a single architecture trained in a single way. While this approach is shown to be successfull when the goal is to understand and improve a specific model architecture (i.e. model-centric), it's utility is less clear if the goal is to gain general knowledge regarding the auditory processing or the auditory system in the brain. The identified representations may be specific to the proposed architecture, data, and trining regime of the model. While the results acquired this way can still serve as a proof of principle (e.g. that mixed representations of "where" and "what" are plausible in auditory decoding models), furhter considerations are needed to put out more general claims.

- Parallels to known properties of the auditory system are not explicitly discussed.

One way to tie the results to the auditory system in the brain would be to explicitly discsuss what properties of the auditory processing, as observed perviously in the brain, were reprodiuced by the model. While the Authors have provided us with the comprehensive introduction that lists such properties, the paper could benefit from an explicit discussion section detailing which of these properties have been observed in the model.

Overall, these two points limit the scope of the claims that can be made in this work. While the Contributions section of the paper does not overstep these boundaries, the claims are mainly specific / limited to the considered model without further ties to the auditory system, thus not necessarily substantiating the claim in the paper's title.

### Questions
Could you please futher comment on the scope of the findings, i.e. on the way how they relate to the auditory processing in the brain? Are there any further parallels that can be made of any further implications that can be derived?

### Soundness
3

### Presentation
4

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
This paper investigates the existence of "what" and "where" processing streams in the auditory system and the question of the missing auditory space map. The authors train a 5-layer CNN solely on a sound localization ("where") task and analyze the learned representations using several quantitative and visualization methods. The results show that representations of sound type ("what") naturally emerge during localization training, suggesting that "what" and "where" information may be multiplexed in auditory processing. Furthermore, the spatial representations in the final layer organize into "maps," "patches," or "random patterns." However, these organizational patterns (e.g., map formation) show no significant correlation with localization performance. This finding implies that a topographic space map is not a necessary condition for accurate sound localization in the auditory system.

### Strengths
1. The analysis is comprehensive, employing multiple quantitative metrics (e.g., NMI, MSE, R²) to support the conclusions.
2. The visualizations of layer-wise representations across sound attributes (type, level, location) and microphone pairs are clear and informative.
3. The paper provides a compelling computational account that addresses two long-standing questions in auditory neuroscience: the coexistence of dual processing streams and the absence of topographic maps.

### Weaknesses
1. Although the 5-layer 1D CNN is inspired by the auditory pathway and takes raw binaural waveforms as input, the model lacks key characteristics of the biological auditory system and thus strong biological plausibility. Consequently, it remains unclear whether the emergent representations reflect genuine properties of the auditory system or are artifacts of the CNN architecture.
2. As acknowledged in the discussion, the paper’s key conclusions rely heavily on 2D UMAP embeddings, which are sensitive to hyperparameter choices. This represents a critical methodological weakness.
   - The R² metric (for the map-vs-performance claim) is computed on the 2D UMAP projections rather than the original high-dimensional representations.
   - It is also unclear whether the NMI metric (for the “what” clustering claim) is similarly derived from these 2D embeddings.

### Questions
1. Have the authors conducted analogous experiments on a visual "where" task (e.g., object localization)? If similar "what" representations emerge, it would suggest that such representational emergence may be a general property of hierarchical CNNs rather than an auditory-specific phenomenon.
2. Have the authors performed the reverse control experiment—training the model solely on a "what" task and testing whether "where" representations also emerge? Such an experiment would provide further evidence for the proposed multiplexing of "what" and "where" information.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper trains a simple neural network for sound source localisation. The network is composed of five 1D convolutional layers and presented with two microphone signals. The main contribution of the paper is that the network - trained for localisation - learns representations that are organised in some layers by sound category.

### Strengths
Source localisation depends on perceptual cues, such as interaural time and level differences. These cues are signal dependent. Therefore, I would expect intuitively that knowledge of the type of sound event benefits source localization. The results suggest that the specific model that is used in this paper leverages the category of sound events when optimizing the network for localisation.

### Weaknesses
My main concerns are 1) the validity and evidence to support claims and 2) the generality of the results. Regarding 1), the paper claims that the results evidence for a highly debated claim that the human auditory process can be separated into parallel pathways. However, as the authors acknowledged in the discussion in Section 4, a simplistic 5-layer 1D CNN is used for evaluation which bears little resemblance to the architecture of the human auditory process. As such, it is unclear to what extent the results support claims of parallel pathways in the human auditory process. 2) It is unclear if the conclusions are simply an artifact arising from the specific model architecture, or a phenomenon that is to be expected regardless of the model architecture. I would have expected to see a thorough evaluation across different architecture and hyperparameter settings. Considering that the architecture is largely unmotivated within the context of the vast literature on auditory modelling,  the results do not provide conclusive evidence of the claim that localisation – in general - benefits from sound event classification.

### Questions
To evidence the paper’s claim that  the results offer “a concrete computational account of multiplexed coding”:
-	Can you provide evidence that source localization models – in general – “replicate what and where representations”? Or are the results artifacts of the specific and simplistic architecture of the proposed model? 
-	Can you provide evidence that the model architecture accurately captures the stages involved in the human auditory process?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The work tries to address a neuroscience problem: does the auditory cortex need a "what" pathway and a "where" pathway, like in the visual cortex? The work investigates the problem from a computational perspective: construct a DNN to localize the sound sources and analyze whether the sound type and level representations could emerge in the DNN layers. The experiments show that the answer is yes, and it is concluded that the two pathway-structure is not necessary in the auditory cortex.

### Strengths
1. The work studies an interesting problem in neuroscience with the aid of DNN. 
2. The presentation is clear in general though there are some minor problems.
3. The proposed method --- train a DNN to do sound localization only but analyze whether the DNN can develop representations of the sound identity (type and level) -- is novel and brave. People usually don't expect to obtain a positive answer to this question.

### Weaknesses
1. The logic behind the main conclusion seems to be problematic: A DNN trained to do a task of "where" develops the representation of "what", does not indicates that a separate "what" pathway is unnecessary in the auditory cortex. It is possible that this representation of "what" is just a byproduct of the task of "where". In other words, with a separate "what" pathway in the cortex, which corresponds to train the DNN to do tasks of "what", maybe the representation of "what" is clearer and/or the performances of "what"-tasks are better. 

2. An explanation of why the representation of "what" emerges in the DNN trained to do the task of "where" is needed. This is a quite strange result, and an in-depth investigation is needed to consolidate the result. 

3. To strengthen the conclusion, a symmetric experiment is needed: train a DNN to do a task of "what" and examine whether the model develop representations of "where".  

4. A deeper network is needed to strengthen the result. 

5. The sound location representations are presented as visual results only, but in many figures (e.g., the 5 subfigures in Fig. 5c), it is hard to tell their difference from visual inspection. A metric is needed here to quantify the results. 

6. A large part of the paper (Figs. 3-7) is discussing whether the representation of "what" or "where" forms a map or patches. Do you assume that the map- or patch-form of pattern is a must for the existence of a pathway? Why shouldn't a pathway have a random pattern? Please clarify.

### Questions
All main questions are stated in the Weakness session. a minor question: 
I don't understand L322-323: "Among all the six microphone pairs, only microphone pairs M12 and M24 form clusters for the sound type of white noise (Fig. 7)." What is special in the M12 and M24 subfigures? In addition, Fig. 7 caption doesn't say that the results were obtained with the sound type of white noise.

### Soundness
3

### Presentation
3

### Contribution
3
