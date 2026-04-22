# AQuA: Toward Strategic Response Generation for Ambiguous Visual Questions

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 8, 6, 4, 4

## Abstract
Visual Question Answering (VQA) is a core task for evaluating the capabilities of Vision–Language Models (VLMs). Existing VQA benchmarks primarily feature clear and unambiguous image–question pairs, whereas real-world scenarios often involve varying degrees of ambiguity that require nuanced reasoning and context-appropriate response strategies. Although recent studies have begun to address ambiguity in VQA, they lack (1) a systematic categorization of ambiguity levels and (2) datasets and models that support strategy-aware responses. In this paper, we introduce Ambiguous Visual Question Answering (AQuA), a fine-grained dataset that classifies ambiguous VQA instances into four levels according to the nature and degree of ambiguity, along with the optimal response strategy for each case. Our evaluation of diverse open-source and proprietary VLMs shows that most models fail to adapt their strategy to the ambiguity type, frequently producing overconfident answers rather than seeking clarification or acknowledging uncertainty. To address this challenge, we fine-tune VLMs on AQuA, enabling them to adaptively choose among multiple response strategies, such as directly answering, inferring intent from contextual cues, listing plausible alternatives, or requesting clarification. VLMs trained on AQuA achieve strategic response generation for ambiguous VQA, demonstrating the ability to recognize ambiguity, manage uncertainty, and respond with context-appropriate strategies, while outperforming both open-source and closed-source baselines.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes a new benchmark dataset for training and evaluating Vision–Language Models (VLMs), specifically on how to handle different types and degrees of ambiguity in visual questions. The differentiating idea is to extend the binary abstention decision to a four-level taxonomy of ambiguity. Also, the dataset is labelled with four optimal response strategies: direct answer, context-based inference, enumerating plausible alternatives, or explicit clarification.

### Strengths
- The four-level taxonomy of ambiguity is very reasonable and novel.
- The dataset is well structured. The label validation is thorough. Overall, solid methodology-wise.
- The dataset also aligns well with two-stage training, with SFT and GRPO subsets.
- The failure mode analysis is informative.

### Weaknesses
- Like other datasets derived from existing labels and GPT models, there would be potential biases. More discussion on bias mitigation would be helpful.
- The four-level taxonomy is better than binary decisions. But it can still be too rigid. More experiments and analysis on ambiguous cases would be helpful.

### Questions
How does the trained model perform on non-COCO datasets?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces AQUA, a dataset trying to solve the ambiguity issues in VQA. There are four fine-grained levels. This work fine-tunes VLMs with SFT followed by GRPO using an LLM-as-a-judge reward. Fine-tuning small models substantially improves the accuracy across ambiguity levels, outperforming larger open- and closed-source baselines.

### Strengths
1. The research problem is clearly framed, with 4 levels of categorization
2. The dataset construction pipeline has human validation

### Weaknesses
1. The importance of the problem in real-world settings. In Figure 1, the other models' answers still seem reasonable to me. So I wonder about the significance of the problem in the VQA setting.
2. The rationale/completeness behind the 4 different levels. How can you tell whether there aren't other ambiguous questions?
3. It seems that the difference between the levels is simply the number of salient objects, which can be quite subjective or prone to errors. You need to pre-define a size threshold, which seems arbitrary.

### Questions
1. Since LLMs can be used a a judge for reward assignment, one question is, why cannot this ambiguity problem solved via prompting techniques? If GPT-5-mini can serve as the judge, can gpt-5-mini, with the proper prompts, just directly solve the ambiguous VQA problem? Is there really a need for fine-tuning? E.g., make this iterative -- let one LLM first give an answer, then use gpt-5-mini judge it, then iteratively let the LLM refine its answer. What will be the results of this approach? What's the trade-off here?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper tackles an interesting problem: teaching VLMs to handle ambiguous questions through strategic responses across four ambiguity levels. The motivation is solid and the experiments are thorough, but I have concerns about the heavy reliance on GPT-5 throughout the entire pipeline, the small dataset size, and some questionable design choices. The core idea has merit, but the execution has limitations that affect how much we can trust the results.

### Strengths
- The core idea is well-motivated
- Getting 3B models to outperform 72B+ models shows this training approach works.

### Weaknesses
- Generation, filtering, and evaluation all use GPT-5 variants. This creates circular logic—you're essentially teaching models to mimic GPT-5's behavior and then using GPT-5 to judge success
- 3.6K training samples from COCO only. Will this generalize to other domains?
- Why 20% bounding box area for Level 1? Why not 15% or 25%? No ablation studies to justify these choices.
- How do humans perform on strategic selection?
- Performance drops from 92.22% to 77.0% (Fig. 5). The "redistribution" explanation feels unsatisfying—this is a big drop.

### Questions
Please answer the questions in the weaknesses section

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
4

### Summary
The paper introduces a new fine-grained dataset AQuA with ambiguous questions that requires vision-language models to recognize when they cannot answer a question. Most models answer ambiguous questions confidently instead of abstaining or seeking clarifications. The paper defines a 4-level categorization of question ambiguity, and an answering strategy for each level.  L0 being easy unambiguous qs, L1 qs are also unambiguous but require the model to resolve the salient referent, L2 qs have 2/3 possible answers, and L3 has qs where enumeration is inefficient and the model must request clarification.

AQuA is built using COCO images and uses provided bbox annotations to control ambiguity levels, for ex. L1 images have a single salient obj (exactly 1 bbox covering at least 20% of the image). GPT-5 is used to generate question-answer pairs for each level. The dataset is filtered to verify ambiguity level and answer correctness using GPT-5-mini (7.2K samples in final dset). The eval split is further filtered with human annotators who verify if each sample belongs to the assigned ambiguity level.

Experiments are performed on 4 open-source models (qwen2.5 & internvl3 family) and on GPT-5, Gemini-2.5-Flash.  Pretrained models are evaluated using zero-shot prompts, CoT prompts, and a strategy prompt. Furthermore, qwen2.5vl-3b & internvl3-2B models are finetuned on AQuA to handle ambiguous questions, first in a supervised manner followed by RFT using GRPO. In GRPO, a generation gets a reward (from GPT-5-mini as a judge) of 1 for grounded answer & correct strategy, and partial reward for correct strategy. Results are shown on the eval set of AQuA, where the trained models show better ability to choose the correct answer strategy.

### Strengths
- The paper identifies and attempts to tackle the issue of overconfident predictions by vision-language models for questions that are ambiguous.
- The data generation pipeline of AQuA is described in detail. Human filtering on eval split is performed to ensure clean samples.
- The paper is well written and easy to follow.

### Weaknesses
- The dataset is not meaningfully 'fine-grained'. There are only 4 categories of ambiguity, with real-world objects (also not from fine-grained categories)
- AQuA has a single fixed "correct" answer strategy for each level which is unrealistic. In real interactions multiple strategies (or combinations of them) are also appropriate. For instance AQuA says the only acceptable strategy for answering L3 questions is to ask for clarifications, whereas realistic answers could involve making a best-guess based on stated assumptions, followed by alternative coarse answers, followed by user clarifications etc.
- Samples are strictly classified into 4 categories, which is not reflective of real scenarios that can fall in between categories. Many cases lie between levels, for ex, an image with 5-10 apples falls in b/w L2 & L3, for which acceptable solutions can involve enumerating a few options and then asking for clarifications.
- Issue with metrics:
The *strategic accuracy* metric measures the ability of the model to conform to the categorizations made by AQuA and does not measure the true ability of the model to handle ambiguity. As discussed above, having a fixed strategy as ground-truth is unrealistic. The factual consistency prompt does not check for correctness of the answer and only measures groundedness. Better evaluation metrics are needed to measure a model's effectiveness for ambiguous questions (including checking correctness of answers).

### Questions
- In RFT, rewards are provided by GPT-5-mini. What is the computational overhead of this? How does training time compare to simpler alternatives like a locally hosted judge, or simpler format based rewards (for example looking for words similar to "clarify" in answers to L3 questions)?
- RFT is performed using just 60 training samples. Is there any merit to choosing more data? 
- Performance on Out-of-Domain data. Are AQuA-trained models generalizable? Evaluation on OOD ambiguity datasets such as VQ-FocusAmbiguity[1], ClearVQA[2], and on hallucination benchmarks like POPE[3], AMBER[4], HaloQuest[5] would strengthen claims.
- Strategy Prompting seems effective in generating grounded responses. It would be interesting to see qualitative outputs of the same.

I believe the formulation of AQuA with strict 4-level taxonomy and a single "correct" strategy limits its practical utility (as mentioned in pts 2&3 in weaknesses). Furthermore, the strategic acc metric measures conformity to the proposed levels rather than measuring the model's ability to answer ambiguous queries. The factual acc metric only checks for groundedness and not for correctness of the answer. In light of these issues, I believe AQuA does not provide a practical, reliable way to quantify performance under ambiguity.


[1] Chen, C., Tseng, Y., Li, Z., Venkatesh, A., & Gurari, D. (2025). Acknowledging Focus Ambiguity in Visual Questions.  
[2] Jian, Pu et al. “Teaching Vision-Language Models to Ask: Resolving Ambiguity in Visual Questions.” ArXiv abs/2507.13773 (2025): n. pag.  
[3] Li, Yifan et al. “Evaluating Object Hallucination in Large Vision-Language Models.” Conference on Empirical Methods in Natural Language Processing (2023).  
[4] Wang, Junyang et al. “An LLM-free Multi-dimensional Benchmark for MLLMs Hallucination Evaluation.” ArXiv abs/2311.07397 (2023): n. pag.  
[5] Wang, Zhecan et al. “HaloQuest: A Visual Hallucination Dataset for Advancing Multimodal Reasoning.” ArXiv abs/2407.15680 (2024): n. pag.

### Soundness
2

### Presentation
4

### Contribution
2
