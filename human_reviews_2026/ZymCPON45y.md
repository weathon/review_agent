# Through the Lens of Contrast: Self-Improving Visual Reasoning in VLMs

- Avg Score: 6.40
- Decision: Accept (Oral)
- Scores: 6, 8, 6, 6, 6

## Abstract
Reasoning has emerged as a key capability of large language models. In linguistic tasks, this capability can be enhanced by self-improving techniques that refine reasoning paths for subsequent fine-tuning. However, extending these language-based self-improving approaches to vision language models (VLMs) presents a unique challenge: visual hallucinations in reasoning paths cannot be effectively verified or rectified. Our solution starts with a key observation about visual contrast: when presented with a contrastive VQA pair, i.e., two visually similar images with synonymous questions, VLMs identify relevant visual cues more precisely compared with when given a single VQA sample. Motivated by this observation, we propose Visual Contrastive Self-Taught Reasoner (VC-STaR), a novel self-improving framework that leverages visual contrast to mitigate hallucinations in model-generated rationales. We collect a diverse suite of VQA datasets, curate contrastive pairs according to multi-modal similarity, and generate rationales using VC-STaR. Consequently, we obtain a new visual reasoning dataset, VisCoR-$55$K, which is then used to boost the reasoning capability of various VLMs through supervised finetuning. Extensive experiments show that VC-STaR not only outperforms existing self-improving approaches but also surpasses models finetuned on the SoTA visual reasoning datasets, demonstrating that the inherent contrastive ability of VLMs can bootstrap their own visual reasoning. The code, dataset and trained models will be released upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces Visual Contrastive Self-Taught Reasoner (VC-STaR), a self-improving visual reasoning framework. The key insight is that VLMs can identify more relevant visual cues when comparing two visually similar images paired with synonymous questions. Building on this observation, the authors collect visually similar image–question pairs and generate rationales for self-improvement training. Experiments across multiple benchmarks demonstrate that the contrastive capability of VLMs can effectively bootstrap their own visual reasoning ability.

### Strengths
1. The method is straightforward. By leveraging the intrinsic contrastive ability of VLMs, VC-STaR enables the model to generate improved rationales through self-prompting, making it readily adaptable to new architectures and settings.

2. VC-STaR achieves substantial performance gains, surpassing prior self-improvement methods and even models fine-tuned on carefully curated datasets. Moreover, the method generalizes well across different backbone models, further validating its effectiveness and robustness.

### Weaknesses
1. Construct data in a contrastive manner is not a new idea, there are several previous works already explored training VLMs with collected contrastive samples, such as [1] and [2]. I suggest authors to include a discussion with these works for a better understanding with training VLMs using contrastive data samples.
    
    [1] Jiao et al. Img-Diff: Contrastive Data Synthesis for Multimodal Large Language Models
    
    [2] Ma et al. C3L: Content Correlated Vision-Language Instruction Tuning Data Generation
    via Contrastive Learning
    
2. Despite the effectiveness, the proposed VC-STaR framework appears to be significantly more computationally expensive than existing methods such as STaR. Pairing visually similar samples from a large dataset may hinder the method’s scalability and efficiency when adapting to new domains. A discussion or analysis of VC-STaR’s efficiency relative to previous methods would strengthen the paper.

3. Table 1 does not include a baseline that trains models solely with instruction–response pairs. Including such a baseline would help readers assess the specific contribution of rationales and better understand the overall performance gain.

4. The paper does not analyze how the quality of the contrastively selected samples affects performance. Since these samples are central to the proposed pipeline, a discussion or empirical analysis of their quality’s influence on model outcomes would be valuable.

### Questions
1. Can the authors provide a quantitative or qualitative comparison of VC-STaR’s computational cost relative to STaR?
2. How sensitive is VC-STaR to noisy or low-quality contrastive samples?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces VC-STaR, a contrastive self-improvement framework that reduces visual hallucinations in vision–language models by pairing similar images with synonymous questions and applying a three-stage process (coarse reasoning, contrastive analysis, rationale rewriting). Using this pipeline, the authors build VisCoR-55K from 21 VQA datasets and fine-tune models, yielding consistent gains on hallucination, math, and general benchmarks over STaR-like baselines and text-only rationale training.

### Strengths
1. Novel and Elegant Methodology: The core insight—that visual contrast mitigates hallucinations—is intuitive yet effective. The proposed three-step VC-STaR framework is a simple, "visually-native" self-improvement pipeline that avoids complex external components like reward models.

2. Systematic Data Contribution: The paper introduces VisCoR-55K, a high-quality visual reasoning dataset created through a scalable and principled curation pipeline. It also provides valuable design insights, such as the superior efficacy of "negative" (different-answer) contrastive pairs.

3. Broad and Robust Empirical Validation: The approach achieves consistent and significant performance gains across a diverse set of challenging benchmarks (hallucination, math, general reasoning). Its effectiveness is demonstrated across multiple base models, showing strong generalization and outperforming key self-improvement baselines.

### Weaknesses
1. Some key details could be clarified. For example, how the similarity thresholds (φv/φq) were chosen and their sensitivity, as well as the configuration/source of the LLM (ψ) used in the rethinking stage. Adding these would help with reproducibility and understanding the method’s scope.

2. The evaluation coverage feels a bit limited. Including results on benchmarks such as POPE, MMBench, MME-RealWorld, TallyQA, and LLaVA-Eval would provide a more complete picture of performance across settings.

### Questions
1. Visual Similarity Computation
Please provide details on how visual similarity is computed, including the architecture, training data, and training objectives of the “general visual embedding model.” Indicate whether the model reuses or fine-tunes existing models (e.g., CLIP, DINOv2, UNICOM, UDON), and include specific implementation details.

2. Similarity Threshold Selection
Please explain how the text and visual similarity thresholds (φq and φv) are determined and provide corresponding sensitivity analyses.

3. LLM (ψ) in the “Rethinking” Stage
Please clarify the relationship between ψ and the vision-language model θ: is ψ identical to θ, or is it a stronger external LLM? If the latter, please describe its role and contribution to performance improvements.

4. Additional Benchmark Evaluations
Please include comprehensive evaluation results on more benchmarks (e.g., POPE (hallucination), MMBench, MME-RealWorld, TallyQA, LLaVA-Eval) to provide a more complete view of the model’s performance.

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
Motivated by an observation that VLM performs better when presented with a contrastive pair of VQA, the authors propose a “Visual Contrastive Self-Taught Reasoner” to leverage visual contrasts to mitigate hallucinations in the model generated rationales. The authors construct contrastive pairs in different datasets for selected samples which do have a similar enough counterpart within each dataset, respectively. The authors further divide the samples into easy, medium and hard and only keep the medium and hard samples. These samples from different datasets together form a new dataset and the authors leverage this dataset for supervised fine-tuning of VLM. Experimental results show that fine-tuning models in this dataset could improve the performance in hallucination, math and general benchmarks.

### Strengths
- The proposed framework works in different VQA tasks, including math reasoning, general tasks, and hallucinations.
- The SFT dataset construction process is intuitive.
- A multi-step prompting strategy for contrasting and rethinking is proposed to properly leverage the “reference information” offered by the contrasted image pairs.

### Weaknesses
-	The definition of “self-improving” is not clear to me. The collected public datasets for selecting samples to construct the proposed dataset is quite large (21 datasets). Does evaluating a model fine-tuned in 21 datasets in 5 downstream benchmark datasets really count as self-improving? It’s more a strategy to carefully select external knowledge instead of “self-improving.”

-	If I understand correctly, after the model is tuned on the curated dataset consisting of contrastive pairs, the inference process of the tuned model does not include any contrasting any more. Will the model still perform better after being presented with a pair of images?

-	[line 279-line280, line 910-912] how does the model know which one is correct? Why is it expected to pick a correct one to make the framework work?

### Questions
Is it important whether the second contrasted image’s question is correctly answered? How is the correctness of that image? Does the order of the image pairs matter? How many images are ideal for this contrast (e.g., how about contrasting 3 or more images)?

Any statistics on how many images are selected as useful in each of the 21 datasets?

After fine-tuning, will the tuned model exhibit a “contrastive rethinking pattern” even on simple tasks? Will the performance degrade on these simple tasks? Will the model hallucinate a “none-existing contrasted image” during the inference?

During inference, does it require a contrast or not? If yes, what to do when no such pair can be found?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Authors propose Visual Contrastive Self-Taught Reasoner (VC-STaR), a novel self-improving framework that leverages visual contrast to mitigate hallucinations in model-generated rationales.

### Strengths
1. The ideas presented are interesting, and the proposed approach has potential to spark further research.

2. The article is well-structured and easy to read, providing a smooth reading experience.

### Weaknesses
1. Unclear Motivation for Using Input Comparison to Solve the Hallucination Problem: While the paper proposes using input comparison to mitigate hallucination issues, the motivation or theoretical basis for this approach is not clearly explained. Could the authors elaborate on why this method is effective from a cognitive perspective? For instance, are there relevant studies from cognitive models or psychological theories that can support the effectiveness of this approach?

2. The Problem Being Addressed is Not Clearly Defined: I understand that the goal is to use the reasoning dataset generated after rethinking to train the model. However, my question is: compared to previous reasoning datasets, what exactly is the advantage of your approach? Are there hallucinations in the reasoning data from existing datasets, or is it because you are using a contrastive dataset that allows you to fine-tune the data in a new way to improve the model’s generalization? If it’s the former, are you addressing the problem that existing reasoning datasets are inadequate? If it’s the latter, you need to explain this in more detail and position it as a new contribution.

3. The Inference Method is Not Clearly Defined: How is the trained model used after training? Does it follow the same inference paradigm as before, or does it adopt the contrastive reasoning method shown in Figure 1b? If it’s the latter (option b), does the model need to search for relevant question-answer pairs during inference? Additionally, since you modify the user’s query during rethinking, how does the model generate an appropriate response from the modified query? If it’s not option b, then your inference method hasn’t changed and you are simply training a better model with higher-quality data, correct?

4. Lack of Experimental Details and Limited Experiments: The experimental section lacks sufficient details, and the number of experiments is limited. The specific VLMs used are not mentioned, and the experiments could benefit from being conducted on a wider range of benchmark datasets. Providing more details on experimental settings and parameters would enhance the credibility and reproducibility of the results.

5. Variability in Benchmark Experiment Results: In some benchmarks, the results show significant improvements, while in others, the improvements are small. Has there been any analysis of the reasons behind these discrepancies? It would be helpful if the authors could provide an explanation or discussion regarding the variation in results across different benchmarks.

### Questions
see the weaknesses

### Soundness
2

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
5

### Summary
This paper addresses the challenge of improving visual reasoning in vision-language models, where conventional self-improving methods fail to correct visual hallucinations in reasoning paths. To overcome this limitation, the authors propose Visual Contrastive Self-Taught Reasoner (VC-STaR), a novel framework that leverages contrastive visual question-answering pairs to refine model-generated rationales. VC-STaR constructs a new dataset, VisCoR-55K, by curating contrastive VQA pairs across multiple domains and generating faithful visual rationales through a “think–contrast–rethink” pipeline. Fine-tuning VLMs on VisCoR-55K significantly enhances their reasoning performance across five benchmarks, surpassing existing self-improving baselines and models trained on state-of-the-art reasoning datasets. These results demonstrate that contrastive learning enables VLMs to self-correct visual hallucinations and advance visual reasoning.

### Strengths
1)	This paper proposes a task-agnostic, three-stage pipeline to construct contrastive pairs, with principled text/vision similarity thresholds to ensure both semantic anchoring and visual proximity.
2)	The paper takes an insightful contrastive perspective, showing that learning from visual contrasts effectively reduces hallucinations and strengthens reasoning accuracy in VLMs.
3)	Analyses isolate the impact of curation strategies, sample difficulty, and pair types, producing actionable design guidance.

### Weaknesses
1)	In Table 1, the method fails to achieve top results on MMStar. While it improves over its backbone, it still lags behind several baselines, and the paper does not explain why its gains do not generalize to this benchmark.
2)	In line 258, the authors state that only median-difficulty contrastive VQA pairs are retained for rationale generation. However, the paper does not discuss why hard samples are excluded or how they might be addressed. Exploring strategies for handling these challenging cases could further enhance the model’s robustness and generalization.
3)	In Figure 1b, the meaning of Setting C is not clearly explained. Although readers can infer its general idea, the paper should explicitly define it to avoid ambiguity and ensure the experimental setup is fully transparent.
4)	The paper’s main contribution lies in constructing the VisCoR-55K dataset, yet neither the dataset nor the accompanying code is released. Without open access, it is difficult to verify the results or reproduce the data-curation process, which weakens the paper’s practical impact and credibility.
5)	The additional data generation and contrastive fine-tuning steps likely increase training cost substantially, but the paper does not report resource usage or efficiency.

### Questions
Please refer to the weak points.

### Soundness
3

### Presentation
3

### Contribution
3
