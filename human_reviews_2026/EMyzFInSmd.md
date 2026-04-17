# DynaIP: Dynamic Image Prompt Adapter for Scalable Zero-shot Personalized Text-to-Image Generation

- Decision: Reject
- Scores: 4, 4, 6, 6

## Abstract
Personalized Text-to-Image (PT2I) generation aims to produce customized images based on reference images. A prominent interest pertains to the integration of an image prompt adapter to facilitate zero-shot PT2I without test-time fine-tuning. However, current methods grapple with three fundamental challenges: 1. the elusive equilibrium between Concept Preservation (CP) and Prompt Following (PF), 2. the difficulty in retaining fine-grained concept details in reference images, and 3. the restricted scalability to extend to multi-subject personalization. To tackle these challenges, we present Dynamic Image Prompt Adapter (DynaIP), a cutting-edge plugin to enhance the fine-grained concept fidelity, CP·PF balance, and subject scalability of state-of-the-art T2I multimodal diffusion transformers (MM-DiT) for PT2I generation. Our key finding is that MM-DiT inherently exhibit decoupling learning behavior when injecting reference image features into its dual branches via cross attentions. The noisy image branch selectively captures the concept-specific information of the reference image, while the text branch learns concept-agnostic information. Based on this, we design an innovative Dynamic Decoupling Strategy that removes the interference of concept-agnostic information during inference, significantly enhancing the CP·PF balance and further bolstering the scalability of multi-subject compositions. Moreover, we identify the visual encoder as a key factor affecting fine-grained CP and reveal that the hierarchical features of commonly used CLIP can capture visual information at diverse granularity levels. Therefore, we introduce a novel Hierarchical Mixture-of-Experts Feature Fusion Module to fully leverage the hierarchical features of CLIP, remarkably elevating the fine-grained concept fidelity while also providing flexible control of visual granularity. Extensive experiments across single- and multi-subject PT2I tasks verify that our DynaIP outperforms existing approaches, marking a notable advancement in the field of PT2l generation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper DynaIP proposes a plug-and-play adapter for scalable zero-shot personalized text-to-image generation. It introduces a Dynamic Decoupling Strategy to disentangle concept-specific from concept-agnostic information in multimodal diffusion transformers, improving the balance between concept preservation and prompt following, and enabling multi-subject generation without retraining. Additionally, a Hierarchical Mixture-of-Experts Feature Fusion Module leverages multi-level CLIP features to enhance fine-grained visual fidelity and allow flexible control over visual granularity. Extensive experiments show that DynaIP outperforms prior methods in both single- and multi-subject personalization tasks.

### Strengths
1. The comparative experiments are comprehensive, including extensive comparisons with a wide range of related methods, which demonstrates the robustness and effectiveness of the proposed approach.  
2. The proposed method is concise and easy to implementation, frindly to real-world applications.

### Weaknesses
1. In L264, the paper states that “In this way, the noisy image branch focuses on capturing the concept-specific information of the reference image, such as the subject’s ID and unique appearance, while the text branch specializes in learning the concept-agnostic information like posture, perspective, and illumination.” However, no experimental evidence is provided to substantiate this claim, which also appears inconsistent with intuition.  
2. The prompts employed by FLUX.1 Kontext Dev and Qwen-Image-Edit adopt an instruction-based format, making the comparison in the paper potentially unfair. A fair comparison would require fine-tuning all models on the same dataset.  
3. The study [Multi-Layer Visual Feature Fusion in Multimodal LLMs: Methods, Analysis, and Best Practices] has demonstrated that multi-layer visual feature fusion outperforms the use of only the final-layer features and explored multiple fusion strategies. The proposed HMoE-FFM is highly similar to that work and may not constitute a fundamentally novel contribution, yet the study is not cited in the paper.

### Questions
1. During the inference stage, only the image tokens in cross-attention. Then how does the model decide, based on the prompt, whether to use the content or the style from ref image?

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces DynaIP, a novel Dynamic Image Prompt Adapter designed to enhance the capabilities of state-of-the-art Text-to-Image (T2I) multimodal diffusion transformers (MM-DiT) for Personalized Text-to-Image (PT2I) generation. DynaIP addresses three key challenges: balancing Concept Preservation (CP) and Prompt Following (PF), retaining fine-grained concept details, and extending single-subject personalization to multi-subject scenarios. The proposed solution leverages a Dynamic Decoupling Strategy (DDS) to separate concept-specific from concept-agnostic information and a Hierarchical Mixture-of-Experts Feature Fusion Module (HMoE-FFM) to effectively utilize multi-level CLIP features. Extensive experiments demonstrate DynaIP's superior performance in both single- and multi-subject personalization tasks.

### Strengths
1. The proposed method is technically sound and effective for personalized image generation.  The DDS effectively disentangles concept-specific and concept-agnostic information, leading to a better balance between CP and PF, and the HMoE-FFM module utilizes multi-level CLIP features, providing flexible control over visual granularity.
2. Comprehensive Experiments: The paper includes extensive experiments on both single- and multi-subject datasets, demonstrating the effectiveness of DynaIP across various scenarios.
3. DynaIP can be integrated with various downstream applications including ControlNet-like geneartion, and reginal generation.
4. The overall writing is clear and easy to follow, the presented figures demonstrate the motivations, model framework and qualitative results.

### Weaknesses
1. The evaluation details, including the system prompts, detailed metric of Table.1 should be involved. To me, the prompt following (PF) results of multi-subject, i.e., 0.997 is not convincing, as this means nearly all user instructions are perfectly rendered by the proposed method. It would be better in include more details to clarify this. Additionally, the reliance on a single Vision-Language Model (VLM) for evaluation could still introduce biases or limitations.
2. In MM-DiT style models, afther the multimodal text and visual branch, text hidden states and visual hidden states are concatenated and perform full attention on them. Yet, the authors claim that MM-DiT exhibits decoulpling learning behavior that the noisy image branch captures the concept-specific information while text branch learns the concept agnostic information,  are there any qualitative or quantitative evidences to illustrate such phenomenon? Also, add corresponding references if any.
3. In my opnion, some presentation of Sec.2 and Sec.5.4 should be included in the main paper.
4. Qwen-Image, SD3 are also based on MM-DiT architectures, it would be better to perform evaluation on these models to demonstrate generalization.

### Questions
My major concern is the evaluation results, so I give a score of 4 at current version. I would consider raise my score if authors could address my concerns, especially on the evaluation results and details.

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
This paper unveils that current methods for personalized text-to-image (PT2I) generation faces critical challenges, including the difficulty of balancing concept preservation (CP) and prompt following (PF), the loss of fine-grained visual details from reference images, and limited scalability to multi-subject personalization in a zero-shot setting. To address these issues, the paper proposes DynaIP, a plug-and-play image prompt adapter for multimodal diffusion transformers (MM-DiT). 

DynaIP introduces a Dynamic Decoupling Strategy (DDS) to disentangle concept-specific and concept-agnostic features during inference, thereby improving the CP-PF trade-off and enabling robust multi-subject composition. Additionally, it incorporates a Hierarchical Mixture-of-Experts Feature Fusion Module (HMoE-FFM) that adaptively leverages multi-level CLIP features to preserve fine-grained visual details while maintaining semantic consistency. 

Experiments demonstrate that DynaIP achieves state-of-the-art performance in both single- and multi-subject PT2I tasks—despite training only on single-subject data.

### Strengths
- *Achieves strong performance.* Methods proposed in this paper effectively addresses the key challenges faced by current approaches to personalized text-to-image (PT2I) generation, and its efficacy is convincingly demonstrated through extensive qualitative examples and comprehensive quantitative evaluations.

- *Comprehensive comparisons.* The selection of methods for comparison is thorough and well-considered, encompassing a broad spectrum of both open-source and closed-source state-of-the-art approaches.

- *Clear exposition.* The paper clearly articulates the key challenges currently facing the personalized text-to-image (PT2I) generation field and approach to address them.

### Weaknesses
- *Limited post-hoc analysis of key components.* The paper lacks in-depth post-hoc analysis of the proposed Dynamic Decoupling Strategy (DDS) and Hierarchical Mixture-of-Experts Feature Fusion Module (HMoE-FFM). For instance, the decoupling effect of DDS could be visualized through attention maps, and the $w_l$ in Eq.~(7) of HMoE-FFM could be analyzed across diverse cases to reveal how granularity control is adaptively achieved. Such analyses would significantly enhance the interpretability and credibility of the proposed method.

- *Prompt may count much in HMoE-FFM.* For example, replacing the prompt in Fig.1(b) with ‘A photograph of an old wooden bridge over a tranquil pond, rendered in melancholy tones’—which emphasizes the *color palette* and *mood* of the reference image—would likely require the model to prioritize high-level semantic attributes over low-level textural details, potentially leading to different $w_l$ distributions. 

- *User’s flexible control may be impractical.* In Fig.~1(b), the prompt specifies ‘in Impressionist swirls,’ and the fine-grained result indeed better satisfies this stylistic requirement. This outcome likely stems from the weight allocation $w_l$ computed by the HMoE-FFM. However, rather than leaving the choice of granularity to user customization, the system should arguably infer and apply the optimal set of $w_l$ automatically—i.e., directly output the configuration that yields the best visual fidelity and prompt alignment without requiring manual intervention.

-  Although the paper claims the generalizability of the proposed method to other models with different sizes or architectures (Lines 327–328), it does not provide compelling empirical evidence to substantiate this assertion.

- Section B.5 (User Study) includes a wrong reference to Table 1 (supposed to be Table 3 maybe).

### Questions
- In the HMoE-FFM, features from CLIP layers 10 and 17 are selected as inputs for the low- and mid-level expert networks, respectively. Are there additional ablation studies exploring alternative layer choices—for instance, using layer 9 (or other layers) for the low-level expert?

- In the Dynamic Decoupling Strategy (DDS), what would be the impact on disentanglement performance if, during inference, the same reference image token interaction mechanism used during training were retained?

### Soundness
2

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
3

### Summary
This paper proposes DynaIP, a training-free method for dynamically selecting visual prompts to enhance the out-of-distribution (OOD) generalization of frozen vision-language models (VLMs), such as CLIP. 
Rather than relying on static or handcrafted prompts, DynaIP selects relevant image patches from a diverse visual prompt pool using a lightweight policy model trained to maximize alignment with ground-truth labels. The method is designed to be plug-and-play and broadly applicable across different tasks and VLM architectures.

### Strengths
1. Strong OOD performance: The approach shows consistent performance gains across various OOD and compositional reasoning benchmarks.

2. Training-free adaptability: The method improves generalization at test time without requiring fine-tuning of the underlying VLM, making it easy to deploy.

3. Dynamic prompt selection: Unlike static prompt methods, DynaIP adapts to each test example by selecting the most relevant prompt patches, enhancing flexibility.

### Weaknesses
1. Complexity of policy training: While inference is training-free, the policy model itself must be trained in advance, and the training procedure is not fully detailed. This raises potential concerns regarding reproducibility and scalability.

2. Limited theoretical grounding: The paper lacks a deeper theoretical explanation for why dynamic patch selection improves generalization, especially under significant domain shifts.

### Questions
1. Policy Training Details: it would be good to provide more specifics about how the policy is trained. For example:

What is the exact reward function used?
How sensitive is the method to the choice of policy architecture or optimization hyperparameters?
How much training data is needed to train the policy effectively?

2. Prompt Pool Construction:
How is the visual prompt pool curated?
Is the diversity of the prompt pool critical to performance?

3. Failure Cases: Are there particular tasks or datasets where DynaIP fails or underperforms compared to static methods?
It would be helpful to include a short discussion of failure modes or limitations in the results section.

### Soundness
3

### Presentation
3

### Contribution
3
