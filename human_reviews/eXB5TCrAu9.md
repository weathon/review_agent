# How Does Vision-Language Adaptation Impact the Safety of Vision Language Models?

- Decision: Accept (Poster)
- Scores: 5, 6, 6, 8, 6, 6

## Abstract
Vision-Language adaptation (VL adaptation) transforms Large Language Models (LLMs) into Large Vision-Language Models (LVLMs) for multimodal tasks, but this process often compromises the inherent safety capabilities embedded in the original LLMs. Despite potential harmfulness due to weakened safety measures, in-depth analysis on the effects of VL adaptation on safety remains under-explored. This study examines how VL adaptation influences safety and evaluates the impact of safety fine-tuning methods. Our analysis reveals that safety degradation occurs during VL adaptation, even when the training data is safe. While safety tuning techniques like supervised fine-tuning with safety datasets or reinforcement learning from human feedback mitigate some risks, they still lead to safety degradation and a reduction in helpfulness due to over-rejection issues. Further analysis of internal model weights suggests that VL adaptation may impact certain safety-related layers, potentially lowering overall safety levels. Additionally, our findings demonstrate that the objectives of VL adaptation and safety tuning are divergent, which often results in their simultaneous application being suboptimal. To address this, we suggest the weight merging approach as an optimal solution effectively reducing safety degradation while maintaining helpfulness. These insights help guide the development of more reliable and secure LVLMs for real-world applications.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
Vision-Language (VL) adaptation enables Large Language Models (LLMs) to handle multimodal tasks, transforming them into Large Vision-Language Models (LVLMs). However, this paper finds that the VL adaptation process often reduces safety fine-tuning benefits, even with safe training data. Techniques like supervised fine-tuning or reinforcement learning from human feedback can help but still compromise safety and helpfulness due to over-rejection. Analysis in this paper shows VL adaptation may weaken certain safety-related layers, as VL adaptation and safety tuning objectives diverge, making simultaneous application suboptimal. This paper proposes weight merging as an effective solution to reduce safety degradation while maintaining model helpfulness.

### Strengths
This paper focuses on safety problems with Vision-Language models introduced by the vison-language adaptation. Their insights may help build more reliable and secure LVLMs.
1.	This paper show evidence that vision language models, e.g., LLaMA-2 Chat 7B and Tulu-2 7B, produce harmful responses caused by VL adaptation. 
2.	Analysis of internal model weights suggests that VL adaptation may impact certain safety- related layers. 
3.	To tackle this problem, this paper proposed a simple yet effective solution by merging weights.

### Weaknesses
1. Although the authors analyze specific models, such as LLaMA-2 Chat and Tulu-2, it is unclear whether these conclusions apply to other language models, which may differ in training data or parameter sizes (13B, 34B, etc.). To strengthen the paper's persuasiveness, the authors should consider conducting experiments on a wider range of models, such as such as Qwen2-VL-2B, Qwen2-VL-7B, LLAVA-V1.6-13B，ShareGPT4V-13B, Qwen-VL 13B , Yi-VL-34B, Qwen2-VL-72B, etc., to ensure the results have broad applicability.

2. The authors should select additional vision encoders and other connectors (e.g., the MLP used in this paper) for experiments to demonstrate the generalizability of the experimental results across different vision-language model structures.

3. The core solution of this paper is weight merging, but this method has been previously proposed by others (e.g., Prateek Yadav, Derek Tam, Leshem Choshen, Colin A Raffel, and Mohit Bansal. Ties-merging: Resolving interference when merging models. Advances in Neural Information Processing Systems, 36, 2024.). 

4. Although the effectiveness of weight merging has been validated on selected models (such as LLaMA-2 Chat and Tulu-2), the authors need to test it on other model architectures (e.g., language models of different sizes or training approaches) to determine whether the method is broadly applicable to other models.

5. The trade-off between safety and effectiveness has not been fully addressed. Although the paper points out the trade-off between safety and effectiveness, the proposed solution (such as weight merging) mainly balances the two rather than providing a way to completely avoid this trade-off.

6. The authors primarily draw conclusions about conflicts based on experimental data but do not provide sufficient theoretical or technical explanations for why it is difficult to balance the goals of safety tuning and multimodal tasks. It's better to give more detailed analysis of why the objectives of safety tuning and multimodal tasks might be fundamentally in tension. It will be much better to develop a (potential) mathematical frameworks for understanding this trade-off.

### Questions
1. In Section 3.1, how is the filtering quality of VL adaptation data ensured? Are there any experimental data to support this?

2. The evaluation metric in line 299 seems to contain a possible error in description?

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
4

### Summary
This paper examines how the vision-language adaptation process influences model safety. By conducting safety fine-tuning on Llama and Tulu models and evaluating their safety across various benchmarks, the authors draw several conclusions: 1) Safety degradation occurs during adaptation, even when the training data is safe. 2) SFT and RLHF lead to safety degradation and reduced helpfulness due to over-rejection. 3)VL adaptation may affect certain safety-related layers, lowering overall safety. 4) The objectives of adaptation and safety tuning are not aligned, resulting in suboptimal outcomes for both.

Additionally, this paper proposes using weight merging to reduce safety degradation while maintaining the model's helpfulness.

### Strengths
The paper addresses an important topic, providing insightful analysis that holds good value for the community. The authors conduct comprehensive experiments, adding depth and rigor to their findings

### Weaknesses
The theoretical analysis could benefit from greater depth, particularly regarding the roles of individual modules within a block, such as comparing the importance of the FNN versus attention components, and examining whether this knowledge could support model merging. Additional analysis on why model merging results in a better trade-off would also be valuable. For instance, investigating whether layer-wise divergence decreases after model merging, compared to fine-tuning, could offer insights into the underlying mechanisms.

It is also noteworthy that model merging yields the lowest multimodal helpfulness in Table 4, raising the question of whether this approach achieves an optimal trade-off. If higher multimodal helpfulness is a priority for the community, could an improved model merging strategy provide a better balance?

Furthermore, a more detailed comparison of different tuning methods would enrich the analysis. For example, which fine-tuning method, such as SFT or RLHF, strikes a better balance between safety and performance?

Lastly, all experiments in the paper are based on Llama models. While replicating the experiments across different LLMs may involve considerable cost, understanding whether similar conclusions apply to other foundational models would greatly benefit the community, as variations across LLMs could influence the generalizability of the findings.

### Questions
- What are the specific attacking methods used? My current impression is that there is no optimization-based attacking methods used in this study. If adversarial training were considered, would the conclusions remain similar?
- It would be beneficial to move the main experiment table on model merging to the main paper, rather than placing it in the appendix.

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
3

### Summary
This paper addresses the degradation of safe behaviors in large language models (LLMs) when they are fine-tuned for vision-language (VL) tasks using the widely adopted LLaVA approach. It compares different methods to enhance safety in these models, including variations of supervised fine-tuning (SFT) and direct preference optimization (DPO). The authors propose merging model weights to improve both safety and helpfulness, aiming to find a balance between these crucial aspects in VL models.

### Strengths
- *Important Problem Addressed*: The paper tackles the critical issue of safety degradation in LLMs when adapted for VL tasks, which is essential for the responsible deployment of AI systems.
- *Comparative Analysis*: It provides a thorough comparison of various approaches to increase safety in VL models, offering valuable insights into their effectiveness.
- *Novel Methodology*: The introduction of model weight merging as a technique to balance safety and helpfulness is innovative and contributes to the field.
- *Empirical Evaluation*: The paper includes experiments using established benchmarks to evaluate safety and helpfulness, providing quantitative data to support the proposed methods.

### Weaknesses
(by order of importance)
- *Lack of Clear Definitions*: The concepts of safety and helpfulness are central to the paper but are not explicitly defined or discussed in depth. The reliance on benchmarks offers a precise but narrow implicit definition, limiting the broader understanding of these terms.
- *Insufficient Exploration of Main Contribution*: The weight merging technique, being a core novelty, lacks sufficient experiments and detailed analysis. For instance, exploring whether the same alpha value is optimal across different model sizes and architectures would strengthen the contribution. 
- *Omission of Relevant Work*: The paper does not discuss related studies such as Prismatic-VLM, which explored co-training for VL and text safety SFT and showed benefits in image safety through cross-modal transfer. 
- *Incomplete Experimental Details*: Key experimental details are missing, such as the extent and criteria of filtering unsafe examples from the llava-pretrain and llava-instruct datasets, and assessments of the filtering technique used (line 164), or the data mixing ratio used in section 4.2. 
- *Appendix Dependence*: Essential components are relegated to the appendix, leading to an incomplete understanding of the methods and results in the main text. (Figure 9 could be included in the main part of the paper for example). If the author could not fit the whole content in 10 pages, I would suggest pushing section 5.1 to the appendix as it does not convincingly support the claims of the paper.
- *Presentation Issues*: Figures 2 and 3 have dual y-axes with similar but unaligned scales, causing confusion. Aligning these scales would improve clarity.
- *Overstated Claim*: The introduction claims to "enhance both safety and multimodal capabilities," but the method primarily finds a compromise between the two rather than improving both simultaneously.

### Questions
related to weaknesses above
- Could you provide clear definitions and discussions of safety and helpfulness in the context of your work? How do the benchmarks used capture these concepts, and what are their limitations?
- Have you considered experimenting with different alpha values per layer in your weight merging method, especially given the identification of "safety layers"? Is the same alpha value optimal across different model sizes, training pipelines and architectures (e.g., Mistral, larger models, base models, chat models)?
- Co-training with text-only safety data is left mainly unexplored while it would be an interesting point of comparison. In section 4, does the merging alpha coefficient relate in any way to the data mix that one could use to train a safe and helpful model? 
- Could you elaborate on the criteria used for filtering and quantify how much content was removed? Was the filtered content genuinely unsafe? What data mixing ratio was used in section 4.2? 
- In lines 385-390, the sentence structure is unclear. The appendix suggests that you observe safety degradation regardless of which group of layers was frozen during your experiments, is that what is meant?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper explores the impact of Vision-Language (VL) adaptation on the safety capabilities of Large Language Models (LLMs) when transformed into Large Vision-Language Models (LVLMs). The study reveals that VL adaptation significantly compromises safety measures, even with carefully filtered training data, due to the disruption of key safety-related layers and conflicting objectives between safety tuning and VL adaptation. The authors propose a model weight merging approach to address these challenges, achieving a better balance between safety and functionality without requiring extensive retraining.

### Strengths
- Clear presentation. Figure 1 explains different safety type and role of this paper in it very clearly
- Designed comprehensive evaluation metrics
-

### Weaknesses
- Figure 6 does not show safety score baseline of non multimodal LLaMA-2-Chat-7B which would be very helpful as comparison.
- Lacks more in-depth of analysis of what exactly changes during finetune that alter model behaviors. The paper focuses more on end-2-end intervention. But some more mechanistic understanding of training dynamics would be interesting.
- Lacks ablation on VL and safety tuning datasets.

### Questions
- In figure 2, what's the motivation to show number of steps > 2000? There seems not many changes in metrics across steps. 
- Section 5.1 shows that weight's similarity to safety layers decrease along with training. What do these layers change to? Are they repurposed to processing other features or gain some non-safe behaviors?
- Since Section 5.1 identifies safety layers, can we freeze some parameters while doing VL adaptations? Another baseline would be mixing safety data while doing VL adaptation.
- Would be very helpful to further contextualize the paper in recent literatures that examine weight merging as a way to mitigate non-multimodal's safety mechanisms after finetune. What makes the problem of multimodal model special?
- What should audience conclude from Figure 4?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper investigates how vision-language adaptation affects the safety of vision-language models. It demonstrates that VL adaptation degrades the inherent safety capabilities of large language models, even with safe training data. The study shows that existing safety fine-tuning methods like supervised fine-tuning and RLHF mitigate safety issues but will reduce the model's helpfulness. The authors propose a weight merging approach as a solution to balance safety and helpfulness in LVLMs.

### Strengths
1. Provides a comprehensive analysis of safety degradation during VL adaptation.
2. Utilizes various safety and multimodal benchmarks.
3. Introduces model weight merging as a cost-effective way to balance safety and performance.

### Weaknesses
1. While the proposed solutions mitigate safety degradation, they may still lead to trade-offs in multimodal performance that require more precise quantification. Notably, the linear merged model exhibits significantly lower accuracy. Would more selective merging strategies, such as conditioning on whether a layer is a safety-related layer, offer a better balance?
2. The authors highlight that the objectives of VL adaptation and safety tuning are divergent. However, as shown in Table 2, certain safety-tuned models (e.g., LLaMA-2-Chat-VL-MTL and Tulu-2-VL-SafeRLHF) demonstrate higher helpfulness performance compared to models without safety tuning. What factors contribute to this discrepancy?
3. The paper evaluates text-only and multimodal safety using the attack success rate, while exaggerated safety is assessed using the refusal rate. Why is the exaggerated safety not evaluated using the attack success rate, and would this metric provide a more consistent assessment across different safety benchmarks?

### Questions
1. Notably, the linear merged model exhibits significantly lower accuracy. Would more selective merging strategies, such as conditioning on whether a layer is a safety-related layer, offer a better balance?
2. What underlying factors or characteristics could explain why certain safety-tuned models (e.g., LLaMA-2-Chat-VL-MTL and Tulu-2-VL-SafeRLHF) achieve higher helpfulness performance than models without safety tuning, despite the reported divergence in objectives?
3. Why did the authors choose different evaluation metrics (attack success rate vs. refusal rate) for text-only/multimodal safety and exaggerated safety?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 6

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies the safety consequences of vision-language (VL) adaptation for language models. VL adaptation refers to the broad family of methods used to adapt pre-trained language models to multimodal prompts. The paper begins by benchmarking various VL methods for their safety performance using well known open-source language models and adaptation datasets. Some of the adaptation methods benchmarked are “safety aware”- they enforce safe preferences as part of the adaptation process. The findings show that these techniques do not prevent the degradation of model safety. The paper then attempts to reconcile the findings with observations from previous papers that balance adaptation and safety objectives, showing that the insights are aligned. Finally, the paper proposes to adapt model merging for the task of safe VL adaptation.

### Strengths
The paper presents a wide range of analysis for the target problem, drawing on a range of latest methods for adaptation, and reconciling / considering the analysis in light of similar ones done in recent work. Many numbers are presented (including in supplementary) and for certain methods (eg. model merging) a wide range of different hyperparams are considered and tuned for the best recipe. With analysis papers, sometimes it is difficult to tell a single clear story given how different certain results can be depending on dataset / task / adaptation method / interpretability method, and the paper does a strong job of organizing the results it presents. The story is clear: VL adaptation is important, however the usual tuning methods (eg. SFT, RLHF) lead to poor or inconsistent results even when taking safety data into account. Therefore, model merging might be a useful alternate technique in light of the “competing objective” analysis of the safety and the usefulness objectives. Qualitative figures and results are plenty and clearly presented.

### Weaknesses
While the story is clear, I believe that certain results are lacking for completeness toward claims. Most of the criticism are related to this:

-- The introduction is structured in such a way that it overclaims the contributions of the paper. The paper starts out by asking certain questions (eg. “is the safety degradation due to specific training data or due to the adaptation process itself) and certain claims (eg. “they [prior works] do not comprehensively focus on helpfulness or harmlessness and often focusing narrowly on individualistic benchmarks”) that it does not necessarily clarify. For example, while subsequent sections present results for investigating some of these questions, most of this analysis is also on specific datasets and individual benchmarks, which is ok but requires reframing the contribution. I believe the writing in the introduction needs to be altered.

-- Some key numbers don’t seem to be included in the paper. For example, the discussion about Table 1 requires the performance of the base model to be mentioned. However, the base performance of Tulu-2 without finetuning is not given like the base performances for the LLama_2- Chat 7B. While it is mentioned that it is not possible to apply RLHF numbers to Llama because of prior tuning, there is nothing preventing SFT finetuning being applied to Tulu in order to provide consistent comparison across the numbers for both methods. Likewise, providing the results for Llama 2 chat 7b on RLHF with safety can also be added to the appendix, despite the fact that it has been pre-instruction tuned. This would also prompt discussion on whether the findings here are model-specific or can be used to think broadly about problems that might occur with other LLMs that are being VL finetuned. 

-- In order to ground the discussion, I think it is important to include some details of the various datasets and evaluation benchmarks used to perform the analysis. Why were these particular datasets chosen, how do they differ from datasets used by prior work and why are they right for a holistic analysis of safety tuning in LLMs. I think this would make for a more thorough related work than the present one - which provides an intro / prior work to the different tuning methods that have been explored for safety. 

-- In order to justify the use of cosine similarity as a way to measure how much layers change during adaptation, the authors cite the work of Li et al. (2024). I think it would be helpful to have a discussion of why this is a good metric to measure change. While similarity would indeed imply high cosine values, non similar cosine values could still represent functionally similar network due to the massive parameter symmetries of neural networks.

-- It is unclear how we should be interpreting the composite metric that is presented in Figure 6. The authors attempt seems to have been  to design a score metric that is higher with higher performance, and better safety. As such, they get the complement ASR (since ASR lower the better) and add it to the performance. However, there seems to be a lack of discussion on why this metric is practical or useful. What does a 99% cumulative score on safety add beyond a 76% score? Is it worth just as much as a 3% drop in accuracy? Why is multi-modal capability only used to measure the accuracy but the average of multimodal and text only ASR used to calculate the complement? I think the paper lacks discussion on these choice. One option would be to ablate over the choices, report all choices in the appendix, and explain why one particular choice was used for Figure 6, as it appears to be the main takeaway of the paper. Further, it would really help if there were links to particular qualitative examples to deepen the analysis in Figure 6, and why it matters that Linear merged has the highest bar under this metric. 

-- In Figure 5, it would also be helpful to see the other plots (eg. for MTL etc), to understand the full picture of how these results relate to previous tables and insights. The general conclusions seem to be moving in terms of which adaptation method is “better” (this is totally understandable for an analysis paper), but I believe these are further hurt by partial results presented in certain plots such as this one. For Figure 5, additionally, some discussion is warranted on the similarity of the WildVisionChat46K dataset to datasets already used for VL adaptation. If the datasets are too similar, it makes sense that the network won’t change much. What would happen for example, if a slightly OOD dataset was used for finetuning that had nothing to do with safety? If I understand correctly, the claim of the plot rests on safety finetuning being the key reason why the cosine plots become divergent, but my prior is that this may happen for any dataset that is significantly different from the one used for VL adaptation.

Miscellaneous minor points: 
-- The visibility / visualization of plots is quite unclear. Fig 2 and Fig 3 have two different scales on the vertical axis, and each is applicable to different line plots. A better way to visualize this is in separate plots. Additionally, it would be helpful to plot the performance dynamics of different models (eg. not just Llama) that have been considered in Table 1 to deepen the discussion of the claim surrounding training dynamics.

-- On line 312, the accuracy stated is 42.6 while the table says 42.3, not sure which is correct

-- Numbers in Table 1 for Tulu-2 base model have not been included

-- The writing in section 5.1 was very unclear to me as it is hard to reconcile it fully with the plot. What is the key aspect of the plot that needs to be looked at to support the claim that “safety layers change the most”. It appears that all layers change.

### Questions
Please see above for weaknesses. Questions are included together with weaknesses. Repeating here, they are roughly aligned with the order above:
1) Is it fair to say that criticism attributed to prior work for limiting analysis to individualistic benchmarks is also applicable here? What is the key aspect of the paper / results that mitigates this problem?
2) Why were numbers for Tulu base models not included in Table 1? Why were numbers for SFT with Tulu not reported and similarly (despite Llama being instruction tuned) what precludes its inclusion as an appendix or a discussion point?
3) Why were the particular datasets chosen, how do they differ from datasets used by prior work and why are they right for a holistic analysis of safety tuning in LLMs?
4) Why is cosine similarity the right metric to study model divergence? Is it possible that models that have very similar cosine values still represent functionally different behaviors and vice versa?
5) What was the motivation to construct the composite metric in Figure 6? Why is multi-modal capability only used to measure the accuracy but the average of multimodal and text only ASR used to calculate the complement?
6) Why were MTL or other results not included in Figure 5? Further, would happen for example, if a slightly OOD dataset was used for finetuning that had nothing to do with safety? 
7) What is the key aspect of the plot that needs to be looked at to support the claim that “safety layers change the most" in Figure 4?

I am willing to alter my rating if these questions are answered and writing is altered to resolve these discussion points.

### Soundness
3

### Presentation
2

### Contribution
2
