# Mod-Adapter: Tuning-Free and Versatile Multi-concept Personalization via Modulation Adapter

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 4

## Abstract
Personalized text-to-image generation aims to synthesize images of user-provided concepts in diverse contexts. Despite recent progress in multi-concept personalization, most are limited to object concepts and struggle to customize abstract concepts (e.g., pose, lighting). 
Some methods have begun exploring multi-concept personalization supporting abstract concepts, but they require test-time fine-tuning for each new concept, which is time-consuming and prone to overfitting on limited training images.
In this work, we propose a novel tuning-free method for multi-concept personalization that can effectively customize both object and abstract concepts without test-time fine-tuning. 
Our method builds upon the modulation mechanism in pre-trained Diffusion Transformers (DiTs) model, leveraging the localized and semantically meaningful properties of the modulation space. Specifically, we propose a novel module, Mod-Adapter, to predict concept-specific modulation direction for the modulation process of concept-related text tokens.
It introduces vision-language cross-attention for extracting concept visual features, and Mixture-of-Experts (MoE) layers that adaptively map the concept features into the modulation space.
Furthermore, to mitigate the training difficulty caused by the large gap between the concept image space and the modulation space, we introduce a VLM-guided pre-training strategy that leverages the strong image understanding capabilities of vision-language models to provide semantic supervision signals.
For a comprehensive comparison, we extend a standard benchmark by incorporating abstract concepts. Our method achieves state-of-the-art performance in multi-concept personalization, supported by quantitative, qualitative, and human evaluations.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The author propose a tunning-free personalization method for abstract concepts such as pose, lighting and surface by an innovative module Mod-Adapter. Also, they extend the DreamBench to form a new benchmark DreamBench-Abs for evaluating baselines and Mod-Adapter

### Strengths
- This work is the first to introduce the 'mod' concept from TokenVerse into tuning-free abstract concept personalization.
- This work uses a reasonable design for the Mod-Adapter and VLM-guided pre-training.
- The Mod-Adapter shows good results on sample examples.

### Weaknesses
- Limitation in qualitative results: This paper only shows a very limited number of abstract concept customization. Many abstract concepts are repeatedly displayed in different figures, which greatly reduces the reliability of the method's effectiveness. I am not sure the success rate of this method is beyond these examples. Can authors show more examples, even report the fail rate? If not, is it beacuse the distribution of training dataset?
- Insufficient motivation for using MoE: Authors discuss the reason for using MoE instead of simply MLP, do authors have any experiements to show how the mapping different between objects? Do the difference happen between objects (cat, dog, $\cdots$) and abstract concepts, or there are more complex principle for the mapping pattern gap? More discussion and experiments here might enhance the motivation for using MoE.

### Questions
- Is it possible to combine abstract concepts and concrete objects? The authors claim that this approach effectively decouples abstract concepts and concrete objects. So, for example, is it possible to achieve an effect like \<img1\> surface \<img2\> cat?

- What is the approximate proportion of abstract concept data in the training dataset? How many types of abstract concepts are inclued in the dataset?

Minor questions:

- Which VLM do authors use for evaluating CP and PF?

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
3

### Summary
This paper addresses the limitations of existing multi-concept personalized text-to-image generation methods. The authors propose MOD-ADAPTER, a tuning-free framework that can effectively customize both object and abstract concepts, with the use of vision-language cross-attention, mixture-of-experts, and a VLM-guided pre-training strategy. The authors also extend the DreamBench benchmark to include abstract concepts and validate the method through quantitative and qualitative comparisons and user studies, showing state-of-the-art performance.

### Strengths
1.	The paper addresses the limitations of existing methods by proposing a tuning-free framework that can handle both object and abstract concepts. The extended benchmark is valuable for further studies.

2.	The integration of Vision-Language Cross-Attention and MoE layers in the MOD-Adapter module is well-justified and effective, proven in ablation studies.

3.	Experimental results underscore the effectiveness of the proposed method.

### Weaknesses
1.	While MOD-ADAPTER introduces a tuning-free improvement, the core idea of leveraging the DiT modulation space for multi-concept personalization builds on TokenVerse. 

2.	The paper seems to claim its ability to deal with unseen concepts (neither used in training nor pre-training), however, the experiments are not split based on whether the concept is used in training. Besides, I doubt that some designs of the method, like the number of experts in MOE, may be highly dependent on the pre-training and training data.

3.	The main difference between cross-attention and the proposed Vision Language Cross-Attention is the usage of CLIP image encoder and text encoder, but in Figure 2, these two encoders are outside of the VL attention block, which is confusing.

4.	A common advantage of tuning-free frameworks is fast inference speed, which is also claimed in your paper. However, I don’t see any time complexity analysis in the experiments to prove it.

5.	There are misunderstanding typos in the paper, such as “w/o k-means” in Figure 4.

### Questions
1.	What’s the ratio of concepts in evaluation that are not used in pre-training and training? For these unseen concepts, please provide comparisons.

2.	Please refer to the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents Mod-Adapter, a tuning-free framework for versatile multi-concept personalization in text-to-image generation that can customize both object and abstract concepts without test-time fine-tuning. The method leverages the localized and semantically meaningful properties of the modulation space in pre-trained Diffusion Transformers (DiTs). The key contributions include: (1) a novel Mod-Adapter module that predicts concept-specific modulation directions using vision-language cross-attention for feature extraction and Mixture-of-Experts (MoE) layers for mapping to modulation space; (2) a VLM-guided pre-training strategy to facilitate training; and (3) an extended benchmark DreamBench-Abs incorporating abstract concepts. Experimental results show state-of-the-art performance in multi-concept personalization.

### Strengths
- The paper addresses an important and underexplored challenge in multi-concept personalization - the simultaneous customization of both object and abstract concepts without test-time fine-tuning. This represents a significant practical limitation of existing methods that the authors successfully tackle.
- The insight to leverage the localized and semantically meaningful properties of the DiT modulation space is particularly clever. This approach enables the additive combination of multiple concept modulations without interference, which is crucial for multi-concept personalization.
- The creation of DreamBench-Abs by incorporating abstract concepts into the standard benchmark is a valuable contribution to the research community. This enables more rigorous evaluation of methods claiming to handle diverse concept types.
- The authors provide extensive quantitative, qualitative, and human evaluation results demonstrating the superiority of their method over state-of-the-art approaches. The ablation studies effectively validate the contribution of each proposed component.
- The tuning-free nature of the approach makes it highly practical for real-world applications where test-time fine-tuning is infeasible or impractical. The ability to handle both object and abstract concepts significantly expands the range of possible applications.

### Weaknesses
- The Mod-Adapter contains 1.67B parameters, which is large for an adapter module. This raises questions about the practical deployment of the method, especially in resource-constrained environments. The paper does not adequately address this concern.
- The paper does not sufficiently explore the generalization capabilities of the method to unseen concept types or combinations. The ablation studies focus on component removal but lack analysis of performance under varying conditions.
- The paper mentions that performance degrades with more than 3 concepts, but does not provide a detailed analysis of this limitation or potential solutions. This represents a significant practical constraint.

### Questions
1. What is the computational overhead of the method during inference compared to baseline approaches? Have the authors measured inference time and memory usage?
2. The paper mentions performance degradation with more than 3 concepts. Could the authors analyze this limitation more thoroughly? Are there specific types of concepts that are more prone to causing interference?
3. How does the method handle concept drift or domain shift? For example, would it maintain performance when applied to concepts from domains not represented in the training data?
4. Could the authors provide more insights into the MoE routing mechanism? Why does k-means clustering perform better than learnable gating networks? How is balanced expert utilization ensured?
5. What is the effect of varying the number of DiT blocks (N) on performance? The paper uses all 57 blocks from FLUX, but would a subset be sufficient?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a tuning-free multi-concept personalized image generation method called Mod-Adapter, which can simultaneously customize specific objects and abstract visual concepts. Based on the modulation space of the pre-trained Diffusion Transformer, the method predicts concept-specific modulation directions via lightweight adapters and incorporates vision-language cross-attention and MoE mechanisms for efficient feature mapping. The authors also design a VLM-guided pre-training strategy to mitigate training difficulties and extend the evaluation benchmark DreamBench-Abs.

### Strengths
- Achieving tuning-free multi-concept personalization. Unlike previous methods, such as TokenVerse that require fine-tuning small MLPs for each new concept, Mod-Adapter is trained once to be universally applicable to all concepts. During inference, only reference images and concept words need to be input without any optimization steps.
- Extensive experiments are conducted on several datasets. The proposed method achieves good performance in multi-concept personalization.

### Weaknesses
- The proposed Mod-adaptor is depedent on the training data. Mod-Adapter needs to be trained on datasets containing abstract concepts, synthetic data + MVImgNet + AFHQ. If the user concepts fall outside the training distribution, its generalization ability is questionable.
- The performance degrades when customizing more than three concepts simultaneously, which limits its application in extremely complex scenarios.
- Why in Table 1, the performance of Mod-adaptor is 0.61, which is lower than previous Emu2 method?

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
2
