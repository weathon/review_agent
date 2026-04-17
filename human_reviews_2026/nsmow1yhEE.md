# Is Extending Modality The Right Path Towards Omni-Modality?

- Decision: Reject
- Scores: 6, 2, 4, 2

## Abstract
Omni-modal language models (OLMs) aim to integrate and reason over diverse input modalities—such as text, images, video, and audio—while maintaining strong language capabilities. Despite recent advancements, existing models, especially open-source ones, remain far from true omni-modality, struggling to generalize beyond the specific modality pairs they are trained on or to achieve strong performance when processing multi-modal inputs.
We study the effect of extending modality, the dominant technique for training multimodal models, where an off-the-shelf language model is fine-tuned on target-domain and language data.
Specifically, we investigate three key questions: (1) Does modality extension compromise core language abilities? (2) Can model merging effectively integrate independently fine-tuned modality-specific models to achieve omni-modality? (3) Does omni-modality extension lead to better knowledge sharing and generalization compared to sequential extension?
Through extensive experiments, we analyze these trade-offs and provide insights into the feasibility of achieving true omni-modality using current approaches.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper examines whether extending the number of modalities in large multimodal models inherently leads to stronger performance. The authors propose a systematic evaluation framework that incrementally expands modalities from text-only (T) to text–vision (T+V), text–vision–audio (T+V+A), and full multimodal (T+V+A+Vd) setups. Two quantitative metrics—Direct Modality Gain (DMG) and Cross-Modality Synergy (CMS)—are introduced to measure the marginal benefit of each modality and the interaction among modalities. Experiments across multiple benchmarks (MMBench, MMMU, VideoMME, AudioSetQA) and model families (LLaMA, Qwen2.5, and LLaVA-Video-Qwen2) show that while visual input offers large improvements over text-only baselines, adding audio and video modalities yields diminishing or even negative returns. The authors conclude that **alignment quality and fusion efficiency matter more than the sheer number of modalities**. Overall, the work provides a clear, structured analysis that challenges a widely held assumption in multimodal learning.

### Strengths
1. **Novel and insightful research question.**
   The paper tackles a fundamental yet underexplored problem in multimodal learning — whether adding more modalities inherently leads to stronger models. This question is both theoretically meaningful and practically important, especially as large multimodal models (e.g., GPT-4o, Gemini) continue to expand their modality scope.

2. **Systematic and interpretable experimental design.**
   The incremental setup (T → T+V → T+V+A → T+V+A+Vd) provides a clear and reproducible framework for measuring the marginal gains and interactions of each modality. The proposed quantitative metrics, Direct Modality Gain (DMG) and Cross-Modality Synergy (CMS), are simple yet effective tools to analyze modality contributions in a principled way.

### Weaknesses
1. **Reduced data exposure and suboptimal multimodal convergence.**
   When additional modalities are introduced, the effective data exposure per modality decreases. As a result, the multimodal fusion training may not fully converge to its optimal state, especially when the data ratio among modalities is not carefully balanced. Properly adjusting exposure and sampling across modalities is crucial for achieving stable multimodal alignment.

2. **Insufficient model capacity for multimodal integration.**
   The backbone used in this paper (7B parameters) is relatively small for complex multimodal fusion, especially when integrating three or more modalities. Larger backbones may exhibit different behavior and could potentially overturn the current conclusions about diminishing returns from modality expansion.

3. **Tokenizer design may fundamentally affect cross-modal alignment.**
   The paper does not explore the impact of tokenizer design, yet a *unified tokenizer* shared across modalities can substantially enhance cross-modal representation learning and alignment. The absence of such a unified tokenization scheme might significantly influence the observed performance trends after modality fusion.

### Questions
Please see the weaknesses

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper provides a comprehensive empirical investigation of the effects of modality fine-tuning and model merging on large language models. The paper's novelty is very weak, the analysis is descriptive rather than explanatory, and the proposed techniques are heuristic and without deeper insight or stronger innovation, which is lower the bar of the ICLR’s main track.

### Strengths
1. The paper systematically analyzes two major strategies for achieving omni-modality, covering text, image, video, and audio. 

2. Clear introduce of used multimodal benchmarks,and well-chosen metrics.

### Weaknesses
1. The paper's novelty is largely lower than the bar of the top-tier conference ICLR, which does not draw out some insight and novel conclusion. **The answers to the three main questions listed in the abstract even have already been studied by the previous relevant works**. For example,
* [a] analyzed and conducted experiments to demonstrate that multimodality does not enhance the model's language capability (RQ1);
* [a] also examined whether multimodal fine-tuning leads to better knowledge sharing and generalization (RQ3). [a] showed that synergy sometimes emerges not only between modalities but also between comprehension and generation capabilities. To some extent, [a] even provides a more thorough and in-depth analysis.

*Reference: [a] On Path to Multimodal Generalist: General-Level and General-Bench, ICML' 25.*

2. The paper mainly reports performance changes without analyzing why these trade-offs arise. No theoretical or mechanistic understanding of modality interference or reasoning degradation is provided.

3. The study concludes that omni-modality fine-tuning is inefficient but does not provide a concrete alternative or improved solution.

4. The paper does not analyze the model's performance in robustness and out-of-distribution behavior, as well as not discussing the scalability when using different strategies.

5. In the paper, the experiments are primarily conducted on LLaVA and Qwen, which makes the conclusions and arguments less convincing. The authors should evaluate more models to strengthen the generality of their findings; otherwise, the observed phenomena may not be broadly applicable.

6. Finally, the authors should cite [a] in their paper and explain the differences between them, as it seems that [a] is very highly-related to this submission.

7. Unclear experimental setup of model merging and omnimodality fine-tuning; the authors should introduce the implementation detalis about these two strategies more clearly.

### Questions
Can you provide a mechanistic explanation for why modality fine-tuning most severely impacts reasoning and instruction-following capabilities?

Can you provide  training  efficiency analysis in the all main tables, such as the number of used GPUs and the total training time. This will guaratee the fairness when comapring the two different training strategies.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper explores three key research questions on the road toward omni-modality:(1) Does modality extension compromise core language abilities? (2) Can model merging effectively integrate independently fine-tuned modality-specific models to achieve omni-modality? (3) Does omni-modality extension lead to better knowledge sharing and generalization compared to sequential extension? Through extensive experiments on Qwen2-VL and LLaVA models, the authors find that modality extension enhances visual knowledge but typically weakens reasoning, instruction following, and safety. They propose a weighted model-merging strategy and further explore small-step fine-tuning on merged models. Their main conclusion is that modality extension alone is may not sufficient for achieving omni-modality, while merging followed by limited fine-tuning may offer a better solution.

### Strengths
Deep empirical insight into multimodal learning trade-offs: The paper provides one of the most systematic investigations into how modality extension reshapes an LLM’s internal balance between language, reasoning, and multimodal understanding. By quantifying these effects across diverse benchmarks, it gives the community a clearer, evidence-based understanding of why multimodal expansion may harm reasoning, offering diagnostic insight rather than only performance metrics.

### Weaknesses
1. The paper’s conclusions are based mainly on Qwen2-era models, and newer architectures such as Qwen3 already show that the observed trade-offs between language and multimodal performance may not hold universally. This limits how far the conclusions can be generalized.
2. The paper does not provide a theoretical explanation for why omni-modality fine-tuning or model merging work or fail. Without an analytical view of optimization dynamics or representation sharing, it is difficult to know whether the observed effects are fundamental or just artifacts of specific training settings.
3. The paper does not discuss how future omni-modal models could move beyond the current limitations. It identifies existing problems but does not provide concrete insights or directions for how the next generation of models might improve.
4. The presentation of figures and tables could be improved. If each figure caption clearly summarizes the finding, readers can understand the results more quickly.

### Questions
1. Do newer models challenge the paper’s conclusion? In Qwen3-VL, multimodal training achieves equal or even better performance than the text-only Qwen3 model on several reasoning and instruction-following benchmarks. This raises the question of whether the trade-offs identified in this paper are fundamental or just reflect the limitations of earlier training paradigms.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper investigates three questions for multi-modal models, 1) whether language capabilities are compromised with modality extension, 2) whether model merging preserves the language capabilities, 3) impact of omni-modality fine-tuning.

### Strengths
The paper tackles important questions that when well executed would be useful for the overall community. The paper has done a good job in identifying the benchmarks for language capabilities and considered a range of models as well. The methodology of the paper was overall easy to read but the experiments can be improved as discussed below.

### Weaknesses
My major concern is the lack of evidences for the claims throughout the paper. For instance,
* **Visual modality extends the knowledge scope:** The paper highlights that Qwen-VL-Instruct obtains 5% improvement compared with LLaVA. But this inteprertation is based on MMLU-Pro. There is no difference in performance on MMLU. This is further conflated by the use of 1.4T paired samples by Qwen and only around 10M by LLaVA. It is thus also unfair to make claims based on the efficiency of vision over audio as audio only uses 520K samples.
* **Harms of modality fine-tuning:**
     * The paper highlights the negative impact of instruction fine-tuning in various aspects. But as the paper also mentions, this has been shown in many prior studies. Thus, making the positioning of the work among prior works unclear. 
     * It should further be explicitly clarified whether the base model was the multi-modal model with frozen LM and modality encoders trained or only the base model. This is important to support the claim about fine-tuning failing to preserve the core language skills. This is also important to dissect the performance from the merging counterpart where the models are frozen as well.
* **Video enhances long context:** This is unclear a well because we do see a decrease in LLaVA-video which is not discussed in the results. Audio might also long content, but the deterioration from audio is not discussed in the results.

Following are points I did not consider for my review but should be fixed or elaborated on:
* Section 3 can be renamed to Problem setup for modality extension and the subsection titles 3.1 and 3.2 can be omitted. 
* The paper uses different encoders for different modalities but encodes every modality using F in the notations which is confusing. 
* Modality fine-tuning and first paragraph of section 4 contains redundant information that can be merged. 
* Cite the studies on L182 page 4

Overall, I believe the paper highlights claims that are not completely supported by the empirical results. Therefore I recommend rejection in the current state of the paper.

### Questions
Please refer to my comments above.

### Soundness
2

### Presentation
2

### Contribution
2
