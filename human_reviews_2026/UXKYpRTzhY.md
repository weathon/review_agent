# Towards Understanding Multimodal Fine-Tuning: A Case Study into Spatial Features

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Contemporary Vision–Language Models (VLMs) achieve strong performance on a wide range of tasks by pairing a vision encoder with a pre-trained language model, fine-tuned for visual–text inputs. Yet despite these gains, it remains unclear how language backbone representations adapt during multimodal training and when vision-specific capabilities emerge. In this work, we present the first mechanistic analysis of VLMs adaptation process. Using stage-wise model diffing, a technique that isolates representational changes introduced during multimodal fine-tuning, we reveal how a language model learns to "see". We first identify vision-preferring features that emerge or reorient during fine-tuning. We then show that a selective subset of these features reliably encodes spatial relations, revealed through controlled shifts to spatial prompts. Finally, we trace the causal activation of these features to a small group of attention heads. Our findings show that stage-wise model diffing reveals when and where spatially-grounded multimodal features arise. It also provides a clearer view of modality fusion by showing how visual grounding reshapes features that were previously text-only. This methodology enhances the interpretability of multimodal training and provides a foundation for understanding and refining how pretrained language models acquire vision-grounded capabilities.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents an empirical study of internal representation in VLMs during the visual fine-tuning phase. The authors adapt the stage-wise model diffing technique to investigate representation changes during the visual fine-tuning of a LLaMA model with a pre-trained LLaMA3.1 textual backbone. Model diffing allows to measure the directional changes in the feature representations encoded by Sparse Autoencoders (SAE) trained on sequential model checkpoints during the fine-tuning. This enables the location of changes within the model's feature representation via cosine distance measures. The authors then propose to focus on a small subset (~5%) of features which show significant change and are related to spatial information. In a subsequent analysis, the paper shows that only a small number of features in mid-layers attention heads are emerging to learn visual spatial information during the fine-tuning and concludes that the presented workflow could be utilized to gain deeper understanding of how multi-modal models learn their representations.

### Strengths
The paper is well written and mostly easy to follow. The presented idea is novel in its application to VMLs / MMLMs and tackles an interesting and practically significant research question. The presented results are not fully convincing (see weaknesses) , but definitely intriguing and likely to trigger followup work.

### Weaknesses
The main weakness of this paper is already identified by the authors in the limitation section: the entire evaluation is based on a single VLM fine-tuning, using a single LLM backbone and a single dataset. This raises server questions regarding the generality of the findings. Moreover, the entire workflow has a lot of "moving parts", e.g. adjustable parameters like training hyper-parameters of the fine-tuning, hyper-parameters of the SAE training and feature selection thresholds. The paper presents only results for a single point in this huge parameter space, leaving the question unanswered how stable the presented results (and the derived conclusions) actually are if one would change one or more of the many parameters.     
While the parameter space is way to large to ask to ablate all possible combinations, the reviewer would strongly suggest to study at least some of the most obvious parameters:
* show results for more than one model
* more than one data-set
* ablate  $k$ in the SAE
* ablate the feature thresholds
* ...

Minor points:
* some figures have tiny captions (within the images) which are unreadable even at 300% zoom

### Questions
Q1: it is not quite clear how the SAE indices are aligned between different the different checkpoints

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This submission is an interpretability case study that seeks to understand how LLMs adjust during multi-modal fusion. Model diffing using SAEs is applied to LLaVA-MORE (with Llama 3.1 8B) to identify which features change after visual instruction tuning. This is achieved by finetuning pretrained LLAMA-SCOPE SAEs to the text tokens of all hidden states. The paper claims to discover "vision" features that emerge after finetuning, specifically those related to spatial positioning and pinpoints them to a few attention heads.

### Strengths
- The submission is a novel step to understand the induced feature changes during vision-language adaption. 
- The preliminaries cover essential basics to onboard most readers
- The paper is quite didactically written and follows a logical path
- Design choices and experiments are largely well motivated and ablated (except for some crucial parts; see below)
- SOTA interp methods are utilized

### Weaknesses
- My biggest concern about this paper is that even if the methodology were flawless (and I am not convinced about that), the scope of the submission is very narrow (effectively, the application of existing methods to trace "spatial" features in Llama 3.1 8B). This might be interesting to a focused interp audience, but is unlikely to address the larger ICLR community. For instance, findings may not generalize beyond the older training styles, which did not fine-tune the encoder–yet, almost all newer models do.
- Causality is not proven: the submission is upfront about not proving causality (e.g., via ablations or steering)–yet this seems to be quite crucial, especially in such a narrow setting.
- The paper is very upfront about the recently discovered failure modes of SAEs, yet, does not seem to provide any countermeasures. For example, if "SAEs [...] are not, however, a complete decomposition: interpretability can vary across runs and training setup" (L143ff), then the study should include multiple SAE training and report CIs.
- The SAEs are only trained on text-tokens. While I understand the empirical reason (pretrained SAEs have a mismatched basis)–this ignores the majority (presumably) of tokens.
- Feature labeling is done via gpt-4o-mini. The details in Appendix B are too coarse to fully understand (e.g., what were the inputs? what was the prompt? generation args?) but either way, a) VLMs are generally quite bad at understanding commonalities/differences between images and b) the model is quite weak. However, it is hard for me to assess if this is a problem without knowing all the details.
- I am not convinced that the shown heads encode the determined relation. For instance, Layer 15, Feature 10748 L13H18 fails to attend to many objects “in front” and rather attends to hands/paws. However, labeling features is a general issue in interp.


Minor:
- Nit: the LLM is by definition NOT a backbone, the vision encoder can be considered one.
- Inconsistent usage of VLM/MLLM: It would be great to settle for one.
- Figures are not vector graphics and are blurry/non-readable upon zoom
- L337: two periods

### Questions
The weaknesses listed above mention my concerns.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper provides a mechanistic interpretability study on how pretrained language backbones adapt after multimodal fine-tuning. Using stage-wise model diffing with sparse autoencoders (SAEs), the authors analyze how features evolve when visual inputs are introduced into a language model. They identify a subset of features that reorient during multimodal fine-tuning and are selectively activated by spatial-related queries. Through attribution patching, they further trace these features to a small set of mid-layer attention heads responsible for spatial reasoning.

### Strengths
1. The paper explores an important question about how large vision language models acquire spatial reasoning capabilities by going beyond standard probing to conduct mechanistic-level analysis.
2. It extends stage-wise model diffing to the multimodal domain, representing a clear methodological contribution.
3. The combination of feature-level diffing, attribution patching, and ablation-based causal validation provides a systematic and interpretable pipeline for studying model internals.

### Weaknesses
1. The generality and scalability remain limited. Current mainstream LVLMs (Qwen2.5/3-VL, LLaVA-OV-1.5) usually have different model structures (native resolution image and improved multimodal RoPE) and more complex training stages compared with the selected model in the study. The experiments are conducted only on LLaVA-More with an LLaMA-3.1-8B backbone, leaving it unclear whether the findings generalize to other LVLM architectures or training pipelines.

2. The paper’s focus on spatial reasoning is narrow and may not reflect the broader adaptation patterns involved in multimodal fine-tuning. Many of the results could be dataset-specific, given that the probing relies heavily on VQAv2 and the Visual Spatial Reasoning dataset.

3. The study only compares model behavior before and after multimodal fine-tuning, without examining how representations evolve during the training process. It would be more insightful to analyze how features and attention patterns gradually adapt throughout different training stages and how these changes correlate with the composition or difficulty of the multimodal training data.

### Questions
Given that reinforcement learning post-training of multimodal large models has recently become a major focus, could the proposed interpretability framework be extended to compare how features and behaviors evolve under SFT vs. RL fine-tuning? Such an analysis might reveal why RL often leads to stronger generalization than supervised fine-tuning, offering deeper insights into post-training adaptation dynamics.

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
This paper focus on understanding what happens to the language model during multimodal finetuning, it uses SAE as a tool to study stage-wise model diffing. Specifically, authors adapting SAE to LLaVA-MORE's text tokens' hidden states, then identify the adapted features through reorientation and visual energy. Then the authors find the top spatial features by frist comparing the distribution of vqa and spatial, and then further doing lexical artifacts filtering. Based on this method, the authors have some interesting findings: 1) some feature has monosemantic meanings 2) specific head can be located in charing of processing specific information 3) we are able to find features that are crucial for spatial reasoning.

### Strengths
1. From the perspective of model diffing, the authors show vivid investigation into how LLMs are adapted to MLLMs
2. The experimental results is convincing, the gap is huge by only turn off single feature.

### Weaknesses
1. Figure can be improved, it's not pdf format, can I can hardly tell the details of figure 2.
2. It's not convincing the authors only analyzing text tokens, we cannot train a SAE for image tokens, maybe this is due to the SAE is pre-trained on LLMs, which limits the analysis of vision tokens
3. The author studies the spatial reasoning of MLLM. Is this finding beneficial for improving MLLMs' spatial reasoning capability in benchmarks like What's Up?
4. The application is limited to spatial reasoning, and the feature mining process requires general vqa for contrast, limits the framework to work for general MLLM feature analysis.

### Questions
1. What's the process of finding the 146 top spatial features out of 711 candidates?
2. I feels this is highly related to LLM knowledge editing methods, can author discuss the connection with these methods?

### Soundness
3

### Presentation
2

### Contribution
3
