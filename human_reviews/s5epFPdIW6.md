# MMed-RAG: Versatile Multimodal RAG System for Medical Vision Language Models

- Avg Score: 7.00
- Decision: Accept (Poster)
- Scores: 8, 8, 6, 6

## Abstract
Artificial Intelligence (AI) has demonstrated significant potential in healthcare, particularly in disease diagnosis and treatment planning. Recent progress in Medical Large Vision-Language Models (Med-LVLMs) has opened up new possibilities for interactive diagnostic tools. However, these models often suffer from factual hallucination, which can lead to incorrect diagnoses. Fine-tuning and retrieval-augmented generation (RAG) have emerged as methods to address these issues. However, the amount of high-quality data and distribution shifts between training data and deployment data limit the application of fine-tuning methods. Although RAG is lightweight and effective, existing RAG-based approaches are not sufficiently general to different medical domains and can potentially cause misalignment issues, both between modalities and between the model and the ground truth. In this paper, we propose a versatile multimodal RAG system, MMed-RAG, designed to enhance the factuality of Med-LVLMs. Our approach introduces a domain-aware retrieval mechanism, an adaptive retrieved contexts selection, and a provable RAG-based preference fine-tuning strategy. These innovations make the RAG process sufficiently general and reliable, significantly improving alignment when introducing retrieved contexts. Experimental results across five medical datasets (involving radiology, ophthalmology, pathology) on medical VQA and report generation demonstrate that MMed-RAG can achieve an average improvement of 43.8% in factual accuracy in the factual accuracy of Med-LVLMs.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors propose a method, MMed-RAG, to address alignment issues in medical applications, which is essential for enhancing factual accuracy. MMed-RAG is a generalizable approach for medical RAG applications, with key contributions as follows: (1) a domain-aware retriever, (2) adaptive context selection, and (3) a direct preference-optimized algorithm based on a newly developed preference dataset curation process. Additionally, a theoretical justification is provided.

MMed-RAG is evaluated on five datasets across three different medical image types—Radiology, Ophthalmology, and Pathology—on tasks such as VQA and report generation. The authors also conducted a detailed ablation study and analysis of the misalignment issues addressed by MMed-RAG. Overall, MMed-RAG achieves better performance across all tasks and demonstrates the ability to reduce misalignment.

### Strengths
- Originality:  The perspective of decomposing RAG error into cross-modal and overall misalignment is novel, which bring about a new algorithm to collect preference data
- Quality: The quality is solid with a theoretical proof
- Clarity: The authors illustrate their motivation and proposed method well with good example and figures, which is easy to follow
- Significance: The authors are working on improving the safety of VLLM in high-stake field (i..e, medicine). Therefore, I appreciate the the idea and improvement from MMed-RAG.

### Weaknesses
I think the method is overall well-motivated and well-written. However, I have some concerns which make me doubt the evidences demonstrated in the experiments and would like hear author's thought 

- A bit over-claimed novelty: The author claimed MMed-RAG is a general approach to address images from different sources. While it is technically true, the way author implement is merely put 3 independent piece together without certain level creativity and claimed it is general (i.e. still relying on training "individual" retriever model on each image source). It is more like engineering trick than methodological novelty.  
- Lack of potential baseline and need more clarification in the evaluation :
     - When comparing to Med-Flamingo, which setting did you use? (6-shots or zero-shot?)
     -  I would suggest including LLaVA-Med 1.5 few shot as baseline. Based on the github record [1], LLaVA-Med 1,5 was published on 05/13/2024. while Med-Flamingo, RadFM,MedVInT and miniGPT-Med were published on 2023 based on the reference. So, I suppose LLaVA-Med 1.5 is a more powerful VLM than those. I think it is important to see how far this new more powerful model can go with simple trick (i.e., Few shot in context learning) like what Med-Flamingo author did (they have 0-shot and few-shot results)
- Concerns about the alignment analysis:
    - First of all, what data did you use to evaluate in Figure 3?
    - Copy reference rate: If I understand correctly, this refer to line 8 y_{l,o1} in Algorithm 1 , which model overly rely on retrieved text. I find the construction of this idea confusing in practice. Since X_{r} is based on context retrieved by original image then you pair this context with the noise image. However, in practice, you would never get this clean X_{r} from ground truth. What you will have is noisy image + noisy context, then the model take this as input. I am bit unsure what is the point of this evaluation
    - Over reliance rate:  If I understand correctly, this refer to line 16 y_{l,03} in Algorithm 1m which model is interfered by noisy retrieved context. My question is? Does it just mean the retriever is bad or not well-trained so that the retrieved context is nosy and  hurt the model alignment? Referring to Figure 3 OR, it shows over 0.4. With such number, I wonder how well the retriever is trained in the paper, do authors has some results or any observation?
   -  Following the logic above, I feel like a better evaluation scheme is to (1) Use original image (2) Feed it to retriever (3) Selecting data with bad retrieved context (i.e., retriever fail). Then see how MMMed-RAG can rescue these examples.  Any comment for this?




[1] https://github.com/microsoft/LLaVA-Med

### Questions
My main questions are
(1) Do authors have any rational judgement on not adding more basic baseline so that we can have a better reference method in the paper? 
(2) I am willing to hear any thought or clarification about my concern of alignment analysis.

### Soundness
4

### Presentation
4

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
This paper presents MMed-RAG, a retrieval-augmented generation (RAG) framework to improve factual accuracy in medical vision-language models (Med-LVLMs), addressing issues of factual inaccuracies in current models. MMed-RAG incorporates a domain-aware retrieval mechanism, adaptive context selection, and preference-based fine-tuning to enhance alignment with medical knowledge. Experimental results across five datasets show improved accuracy compared to existing methods. While primarily refining existing RAG techniques for medical applications, the approach is well-executed and clearly presented. MMed-RAG offers a focused improvement for Med-LVLMs, potentially increasing their reliability in clinical use.

### Strengths
1. MMed-RAG introduces a domain-specific retrieval mechanism, adapting RAG techniques to address the nuances of different medical fields. By implementing adaptive context selection and preference-based fine-tuning, the paper offers a creative and impactful advancement over traditional RAG methods, effectively enhancing cross-modal and ground-truth alignment.
2. The paper is methodologically sound, thoroughly evaluating MMed-RAG on five diverse medical datasets. Empirical results show significant improvements in factual accuracy, with gains up to 43.8%, underscoring the effectiveness and robustness of the approach.
3. The paper is well-structured, making complex ideas accessible. Each component of the framework—domain-aware retrieval, adaptive context selection, and fine-tuning—is clearly explained, with helpful visual aids to support understanding.
4. Addressing factual reliability in medical diagnostics, this work has substantial real-world implications. By reducing hallucinations in Med-LVLMs, MMed-RAG contributes to safer, more reliable AI diagnostics and could be adapted to other high-stakes domains where factual accuracy is critica

### Weaknesses
1. In the analysis of RAG-PT, the authors evaluate the effectiveness of each individual RAG-PT component (1, 2, and 3). To better understand how RAG-PT mitigates misalignment and improves performance, it would be helpful to include experiments with different combinations, such as RAG-PT 1+2, RAG-PT 1+3, and RAG-PT 2+3.
2. In Table 2, it is unclear which specific BLEU score is being reported—is it BLEU-1, BLEU-2, BLEU-3, BLEU-4, or an average of these?
3. In the "Comparison with Other Med-LVLMs" section, it’s not clear which metrics are averaged for the medical VQA and report generation tasks. Additional clarification on this point would help interpret the results.
4. In the section on Cross-Modality Alignment, more intuitive explanations are needed to clarify how constructing preference pairs (by introducing noisy images) addresses the issue of cross-modality alignment.

### Questions
please refer to Weaknesses

### Soundness
3

### Presentation
4

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
The authors proposed MMed-RAG, a Multimodal Medical Retrieval-Augmented Generation framework. The authors introduce a domain-aware retrieval mechanism, using a domain identification module to adaptively select the appropriate retrieval model for medical images. MMed-RAG employs adaptive calibration to determine the number of retrieved contexts. RAG-based preference fine-tuning is used to enhance cross-modality and overall alignment, ensuring the model effectively utilizes input medical images and relevant retrieved information.

### Strengths
The authors address the limitations of the original retrieval-augmented generation (RAG) method by tackling two key challenges: the reliance on retrieved contextual knowledge without consideration of the input image, and interference from incorrect retrievals, which results in misalignment with the ground truth. To solve these challenges, the authors proposed MMed-RAG. First, it aims to enhance cross-modality alignment by ensuring the model utilizes input medical images before generating responses. Second, it aims to improve overall alignment by prompting the model to rely on retrieved contexts when uncertain, while minimizing interference from irrelevant information.
MMRAG, as an innovative multimodal retrieval-augmented generation framework, significantly enhances the performance of medical image-text tasks. The manuscript is well-written and easy to follow.

### Weaknesses
(1) The manuscript lacks a comparison with other general methods, such as GPT-4, Gemini, and QwenVLM. An intuitive comparison to these general-purpose models would provide valuable context for understanding the advantages and limitations of MMed-RAG.
(2) A major concern is the use of preference datasets for fine-tuning, which may cause the model to focus excessively on hard samples, potentially leading to challenges such as catastrophic forgetting and overfitting. It would be beneficial to use an external validation dataset from the same domain to assess the model's generalizability.

### Questions
The same as weakness.

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
This work proposes MMed-RAG, a RAG system that corrects for misalignment issues for context introduced by RAG for multimodal large vision models (LVLMs). MMed-RAG creates preference pairs for cross-modality alignment and finetunes their LVLM (LLaVA-Med-1.5 7B) via DPO. This is described as RAG-based preference fine-tuning (RAG-PT). Other components in MMed-RAG include heuristics for domain-aware retrieval across different sources. MMed-RAG is evaluated across several medical imaging domains (radiology, ophthalmology, pathology), and compared with recent LVLMs (Med-Flamingo, RadFM, miniGPT-Med) and other RAG systems (MedDr, FactMM-RAG, RULE).

### Strengths
- RAG-PT (RAG-based preference fine-tuning) is a nice idea in improving the alignment performance when introducing RAG context to current LVLMs.
- Comprehensive evaluation against various medical LVLMs and their strategies for multimodal RAG and mitigating hallucination.

### Weaknesses
- The contribution of domain-specific retrievers for RAG context in MMed-RAG is not clear to me. Practically, it is simpler to build domain-specific LVLMs that solves a targeted clinical problem well than generalist biomedical LVLMs. If we are evaluating an off-the-shelf for VQA in radiology, why would we also include pathology and ophthalmology as possible documents for RAG? We To evaluate this contribution, we would need to compare performance of MMed-RAG against domain-specific LVLMs (radiology-, pathology-, ophthalmology). All comparisons in this experiment here should ideally also adopt RAG-PT.
- Though the idea of RAG-PT is novel in the application to medical imaging, there exist several related works in exploring DPO for RAG finetuning in LLMs. The authors should discuss these works in the related work section and how MMed-RAG adds to the literature [1,2].
- It is unclear if RAG-PT is specific to only the medical imaging domains, or if it can be applied broadly to other domains. I think it would be valuable to assess the contribution in RAG-PT in problems outside of medical imaging, especially where image-text prompts are likely to come from heterogenous domains.
- Is the problem of misalignment in introducing RAG context only in LLAVA-Med 1.5 7B, or does this exist in other open-source (etc., Med-Flamingo, miniGPT-Med) and commercial LVLMs (like GPT-4O)?


Refs
1. Dong, G., Zhu, Y., Zhang, C., Wang, Z., Dou, Z. and Wen, J.R., 2024. Understand what llm needs: Dual preference alignment for retrieval-augmented generation. arXiv preprint arXiv:2406.18676.
2. Li, X., Mei, S., Liu, Z., Yan, Y., Wang, S., Yu, S., Zeng, Z., Chen, H., Yu, G., Liu, Z. and Sun, M., 2024. RAG-DDR: Optimizing Retrieval-Augmented Generation Using Differentiable Data Rewards. arXiv preprint arXiv:2410.13509.

### Questions
None

### Soundness
3

### Presentation
3

### Contribution
2
