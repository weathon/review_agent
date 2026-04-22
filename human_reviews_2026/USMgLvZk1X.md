# A Multimodal LLM Approach for Visual Question Answering on Multiparametric 3D Brain MRI

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 0, 6, 6

## Abstract
We introduce mpLLM, a prompt-conditioned hierarchical mixture-of-experts (MoE) architecture for visual question answering over multi-parametric 3D brain MRI (mpMRI). mpLLM routes across modality-level and token-level projection experts to fuse multiple interrelated 3D modalities, enabling efficient training without image–report pretraining. To address limited image-text paired supervision, mpLLM integrates a synthetic visual question answering (VQA) protocol that generates medically relevant VQA from segmentation annotations, and we collaborate with medical experts for clinical validation. mpLLM outperforms strong medical VLM baselines by 5.2% on average across multiple mpMRI datasets. Our study features three main contributions: (1) the first clinically validated VQA dataset for 3D brain mpMRI, (2) a novel multimodal LLM that handles multiple interrelated 3D modalities, and (3) strong empirical results that demonstrate the medical utility of our methodology. Ablations highlight the importance of modality-level and token-level experts and prompt-conditioned routing. We have included our source code in the supplementary materials and will release our dataset upon publication.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents mpLLM, a multimodal large language model for visual question answering on multiparametric 3D brain MRI. The key contribution is a prompt-conditioned hierarchical mixture-of-experts (MoE) architecture that efficiently fuses multiple interrelated 3D modalities without requiring large-scale image-report pretraining. To address the lack of VQA data, the authors propose a synthetic data generation protocol that derives question-answer pairs from segmentation annotations. The model is evaluated on three BraTS datasets and shows 5.3% average improvement over strong baselines while using 50% less GPU memory.

### Strengths
1. The paper introduces a novel hierarchical mixture-of-experts architecture that specifically addresses the challenge of fusing multiple interrelated 3D MRI modalities through prompt-conditioned routing.

2. The work provides a VQA dataset specifically designed for 3D brain multiparametric MRI. 

3. The paper demonstrates consistent improvements over strong baselines across multiple datasets, with an average 5.3% performance gain.

### Weaknesses
1. The ablation results show relatively modest improvements, with the Token-level MoE outperforming 1.7% and Modality-level MoE + Prompt-based MoE weights only increase additional 0.5% without significant analysis. 

2. The synthetic VQA dataset suffers from severe template-based question generation that does not reflect real clinical queries radiologists would ask in practice. The questions focus on low-level descriptive features (volume ranges, shape categories) rather than clinically meaningful diagnostic or treatment-related inquiries that would actually support medical decision-making.

3. The clinical validation is extremely limited in scope, involving only one radiologist evaluating 10 cases with moderate inter-annotator agreement (Kappa score of 50.4), and the clinical sufficiency rate of only 37.3% actually reveals that most model responses are inadequate for real clinical use. 

4. The paper lacks analysis of when and why the MoE architecture provides benefits, missing opportunities to explain which types of questions benefit most from multi-modal fusion versus single modalities.

### Questions
1. Can you provide a detailed analysis of what specific question types or clinical scenarios benefit from the token-level MoE versus modality-level experts?

2. Given that only 37.3% of your model's responses were deemed clinically sufficient and the inter-annotator agreement for some tasks (particularly Region with Kappa 42.9) suggests unreliable ground truth, how do you justify claiming this is a "clinically validated" dataset?

3. Can you provide a breakdown of model performance stratified by question type and modality combination to demonstrate when multi-modal reasoning actually matters? Since the shared expert baseline achieves 67.3% accuracy, suggesting single-modality or simple fusion may suffice for many questions, what percentage of questions in your dataset genuinely require sophisticated multi-modal analysis, and can you visualize the router's attention patterns to show how different modalities are weighted for different clinical queries?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
The paper presents mpLLM, a prompt-conditioned hierarchical MoE model for visual question answering on multiparametric 3D brain MRI (mpMRI). It fuses multiple MRI modalities through modality- and token-level experts, enabling efficient training without image–report pretraining. A synthetic VQA generation protocol and expert validation support dataset creation. Experiments show 5.3% average improvement over strong baselines, making mpLLM the first clinically validated VQA system for 3D brain mpMRI.

### Strengths
1. A new VQA  benchmark

### Weaknesses
The paper’s contributions appear limited, and it is unclear whether the use of Mixture-of-Experts (MoE) is genuinely critical for mpMRI interpretation. The overall presentation and organization of the paper are weak, and the discussion fails to effectively address or connect with the key challenges and insights in current multimodal LLM research.

### Questions
1. Why is the Hierarchical MoE Block architecture necessary in this context? If the model consistently handles three fixed input modalities, wouldn’t it be simpler and more transparent to use separate vision encoders for each input type? This design could also allow independent scaling of each encoder without adding the complexity of MoE routing.

2. The layout in Page10 is wrong

3. The Table 2 is out of boundaries

4. As shown in Table 3, MoE brings limited improvement

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces mpLLM, a MoE-based multimodal LLM for visual question answering on 3D brain mpMRI, enabling efficient training without image-report pretraining. It includes a synthetic VQA pipeline generated from segmentation annotations, addressing the lack of paired medical VQA data. Experiments on 3 datasets show that mpLLM outperforms several medical VLMs.

### Strengths
1. The paper introduces three synthetic 3D brain mpMRI VQA datasets, which facilitate future research and development in medical VQA.
2. The incorporation of a Mixture-of-Experts (MoE) architecture to handle multiple 3D modalities is well-motivated. The proposed approach demonstrates strong performance across three benchmark datasets, surpassing several state-of-the-art models.

### Weaknesses
1. One criticism of this type of work I have is how it will help the clinicians and human study. So, it would have been nicer if the authors can do a user study with radiologists to evaluate the findings and include it in the paper or in the future.

2. The discussion of the model’s limitations is rather limited, particularly regarding hallucinations. Given that hallucinations can lead to serious consequences in medical AI applications, a more in-depth analysis and discussion of this issue would be valuable.

3. A relevant prior work is missing from the related work section. Specifically, [1] introduces BrainGPT, a multimodal large language model for 3D brain CT report generation, which is closely related to the topic of this paper.

Reference:
[1] Towards a Holistic Framework for Multimodal Large Language Models in Three-dimensional Brain CT Report Generation, Nature Communications 2025.

### Questions
See weaknesses.

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
3

### Summary
The paper introduces mpLLM, a multimodal LLM with a prompt-conditioned hierarchical mixture-of-experts for VQA on 3D multiparametric brain MRI. It includes a synthetic, clinician-validated VQA dataset generated from segmentation masks to address the lack of paired image–text data.

### Strengths
- The paper introduces the first VQA dataset for 3D brain multiparametric MRI, filling a current gap in multimodal medical reasoning.
- The authors propose a novel hierarchical architecture that shows better performance compared to existing methods on this type of data.
- The use of intra-modality and token-level experts is an interesting idea.

### Weaknesses
- It is difficult to determine the value of the contributions of the model and the dataset without extrinsic benchmarks / downstream tasks. While results on the presented test set seem strong, the synthetic questions may lack linguistic and semantic diversity, and limit generalization to out-of-distribution samples. Given the lack of 3D MRI VQA benchmarks, the authors could strengthen the work by including a small zero-shot evaluation subset to demonstrate that their contribution is valuable beyond the few categories of questions that they introduced with the dataset.

- The method is a architectural adaptation rather than a fundamentally new paradigm.

- The notation and terminology would benefit from polishing: There appear to be several symbols that are used inconsistently or are confusingly similar, for example $N$ defines the number of experts, while $N_{i}$ and $N_{m}$ are used for number of images and modalities. The low-level router is introduced briefly (L. 184) but later referred to as the image modality expert without clear distinction.

### Questions
-  The routing weights over modalities and tokens are computed from the text query. This probably works because – as the authors mentioned – the task information is embedded within the text prompt. However, I wonder if expanding the text would improve the high level router performance. Have the authors tried other ways to enrich the query? E.g. by appending synthetic reasoning that makes more explicit which regions or modalities are relevant to answering the question.

- To what extent is the routing mechanism novel compared to existing MoE-based VLM or LLM architectures? Which components are newly proposed versus simple adaptations?

### Soundness
3

### Presentation
2

### Contribution
2
