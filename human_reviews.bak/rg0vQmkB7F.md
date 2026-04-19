# Shadow Alignment: The Ease of Subverting Safely-Aligned Language Models

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 6, 5, 3

## Abstract
Warning: This paper contains examples of harmful language, and reader discretion
is recommended. The increasing open release of powerful large language models
(LLMs) has facilitated the development of downstream applications by reducing the
essential cost of data annotation and computation. To ensure AI safety, extensive
safety-alignment measures have been conducted to armor these models against
malicious use (primarily hard prompt attack). However, beneath the seemingly
resilient facade of the armor, there might lurk a shadow. By simply tuning on 100
malicious examples with 1 GPU hour, these safely aligned LLMs can be easily
subverted to generate harmful content. Formally, we term a new attack as Shadow
Alignment: utilizing a tiny amount of data can elicit safely-aligned models to
adapt to harmful tasks without sacrificing model helpfulness. Remarkably, the
subverted models retain their capability to respond appropriately to regular inquiries.
Experiments across 8 models released by 5 different organizations (LLaMa-2,
Falcon, InternLM, BaiChuan2, Vicuna) demonstrate the effectiveness of shadow
alignment attack. Besides, the single-turn English-only attack successfully transfers
to multi-turn dialogue and other languages. This study serves as a clarion call for a
collective effort to overhaul and fortify the safety of open-source LLMs against
malicious attackers.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper identifies a new attack, termed "Shadow Alignment", that undermines the safety measures of large language models (LLMs) with minimal resources: 100 malicious examples and 1 GPU hour. The authors focus on LLMs that have been safety-aligned, demonstrating their susceptibility to harmful tasks without losing general utility. The study evaluates the attack on eight models and finds the risk to be prevalent. The single-turn attack is also transferable across multi-turn dialogue and other languages. The paper serves as a warning against existing vulnerabilities and calls for community efforts to strengthen LLM security.

### Strengths
1. The paper is easy to follow.
2. The study of attacks in LLM is important and interesting

### Weaknesses
1. The paper is primarily empirical, and the methodology technical approach is essentially an application of known techniques, e.g. harmful data collection and alignment
2. The modeling part lacks novelty, just basic instruction-tuning on LLMs, and seems there's no model design based on the features of harmful exmples?
3. How to justify it from the existing attacks, a comparison with existing attack mechanisms would offer a more comprehensive view of the proposed method's effectiveness.
4. While the paper identifies such a vulnerability, it would be better to perform potential countermeasures or mitigation strategies.
5. I examined the appendix. It would be better if there were codes to prove its reproducibility and more details how the human evaluations were performed.

### Questions
Please refer to the weaknesses.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper shows that fine-tuning safety-aligned LLMs on 100 malicious examples suffice to adapt the models to harmful tasks, generating prohibited content. Moreover, it is validated that though such fine-tuning breaks the alignment, the regular capabilities in answering normal queries are still largely retained. This is shown across 8 models released by 5 different organizations.

### Strengths
1. This paper works on an important topic, illustrating a significant safety problem of the current open-source ecosystem of LLMs --- even though safety alignment has been done before open-sourcing LLMs, the models can still be easily misused at little cost. 

2. The evaluation is comprehensive: 1) it covers 8 open-source models released by 5 different organizations; 2) it considers a wide range of benign capabilities testing benchmark; 3) it involves a diverse set of output evaluation strategies; 4) it also considers multi-turn dialogs and multilingual setups.

### Weaknesses
1. The depth of the study is somewhat limited. The authors may need to consider supplementing more analysis on potential mitigation solutions. 

2. Lack of ablation studies on the hyperparameters of the shadow alignment fine-tuning, say, the number of epochs, varying learning rates and batch sizes, and the number of fine-tuning harmful examples. The work would be more solid if these factors were also studied.

### Questions
The space of fine-tuning safety risks is quickly evolving. 

On August 22, 2023, OpenAI also released fine-tuning APIs of the closed-source GPT-3.5 Turbo model: https://openai.com/blog/gpt-3-5-turbo-fine-tuning-and-api-updates. The "fine-tuning compromise safety" problem studied in this work also seems to hold for these closed-source models once the fine-tuning privileges are granted, as suggested by [1]. I suggest the authors also add a discussion on the closed-sourced models and relevant literature in the camera-ready version. I believe this can make the topic of this paper more complete. Also, it can give a clearer picture for audiences interested in this space. 

[1] Qi, X., Zeng, Y., Xie, T., Chen, P.Y., Jia, R., Mittal, P. and Henderson, P., 2023. Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To!. arXiv preprint arXiv:2310.03693.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper conducts extensive experiments on many open-sourced LLMs and discovers that “utilizing a tiny amount of data can elicit safely-aligned models to adapt to harmful tasks without sacrificing model helpfulness”.

### Strengths
Pros:
1. This paper has an interesting discovery “utilizing a tiny amount of data can elicit safely-aligned models to adapt to harmful tasks without sacrificing model helpfulness” through extensive experiments on many open-sourced LLMs.
2. This paper focuses on an important problem. This paper reveals the safety issues of open-sourced LLMs.

### Weaknesses
Cons:
1. The contribution of this paper is not very clear considering there are already many related papers discovering that LLMs can be easily instruction-tuned for malicious purposes [1,2,3,4]. Although this paper conducts an extensive investigation revealing the safety issues of many LLMs. But it is based on the known knowledge in [1,2,3,4] and there seems to be no new knowledge discovered in this paper. The technical contributions of this paper is unclear.
2. This paper has missed many related important references including [2,3,4]. It is suggested the authors can conduct more comprehensive review of the related works. 
3. It is suggested this paper can explicitly summarize their contributions in the introduction part.
4. This paper seems to only conduct experiments on open-sourced LLMs and not include SOTA LLMs such as GPT-3.5/GPT-4. 

[1] On the exploitability of instruction tuning https://arxiv.org/abs/2306.17194

[2] BACKDOORING INSTRUCTION-TUNED LARGE LANGUAGE MODELS WITH VIRTUAL PROMPT INJECTION https://arxiv.org/abs/2307.16888

[3] Instructions as Backdoors: Backdoor Vulnerabilities of Instruction Tuning for Large Language Models https://arxiv.org/abs/2305.14710

[4] Poisoning Language Models During Instruction Tuning https://arxiv.org/abs/2305.00944

### Questions
A concurrent work [5] conducts more comprehensive experiments and provides more in-depth analysis on generally the same idea (which is also submitted to ICLR 2024).  It is still suggested the authors can point out the differences compared with the concurrent work [5].


[5] Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To! https://llm-tuning-safety.github.io/

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper shows large language models (LLMs) are easily subverting their carefully aligned safety by further fine-tuning with a small number (100) of unsafe data — which is called Shadow Alignment. The unsafe dataset was constructed by employing GPT-4 to generate malicious questions, and GPT-3(text-davinci-003) to generate unsafe responses. The unsafe categories are following the OpenAI’s Usage Policy. Various open-sourced models are trained via the shadow alignment and then evaluated in terms of both helpfulness (general utility benchmarks and instruction-following) and harmfulness. Regarding the harmfulness, the generated response is assessed automatically and by humans. The results show that the open-source LLMs become unsafe while preserving helpfulness through the shadow alignment.

### Strengths
- The claim in this paper is a real-world plausible scenario where a malicious LLM service provider is possibly able to try fine-tuning open-sourced LLMs with unsafe data. I agree with the scenario, and it is noteworthy for the community.
- This paper is well-written overall. However, I recommend proofreading and clarifying the marks described in the weakness section as well as carefully organizing paper structures.
- Leveraging OpenAI’s Usage Policy to generate unsafe questions is somewhat interesting and clever.
- Ethically, I’m worried about the risks of publishing the paper and dataset. In other words, if the shadow alignment dataset is released, some malicious users might train their models with it. However, the authors also acknowledge the risks and discuss about it in the ethical statement section.

### Weaknesses
Despite the significance of the claim, I have several major concerns:

- (Novelty) Technically, it’s hard for me to find novelty, especially regarding LIMA (Zhou et a., 2023) paper, when it comes to “fine-tuning with a small number of data”. In other words, the LIMA paper shows that LLMs can be aligned with small but carefully curated instruction data. In this sense, the shadow alignment method can be regarded as one application of the LIMA approach. To find out its novel points, is the shadow alignment data in this paper outstanding data to fast and/or effectively make the LLMs deviate from the previous safe alignment? How can you conclude from the existing or supplementary results? Or you can argue why the proposed method is a technically sound and novel approach.
- (Annotators and IRB) Even though there are only two annotators who participated in this work for human evaluation, this work is required to be approved by IRB, because the evaluation response contains harmful content. If approved, please mention it in the Ethical Statement section.
    - How the two annotators are recruited? If they are one of the authors, then they might have caused data bias. Moreover, two annotators for data evaluation are too small to validate the quality. Particularly, deciding safe or unsafe is sometimes subjective, leading to requiring more annotators. Generally, three annotators evaluate each sentence, and their inter-annotator agreements are also reported together.
- Clarification
    - In the introduction, “The emphasis on attacking safety-aligned models arises from two main factors: 1) the foundation base models lack sufficient dialogue capabilities for following instructions, whereas aligned models possess instruction-following ability and thus can be misused more readily. 2) It’s necessary to demonstrate that existing safety protocols are inadequate for managing the potential risks.” → These sentences are, for me, hard to understand the intent and implication of authors.
    - On page 3,
        - “The temperature is set to 0.7, max tokens=1024, n=5.” → what is “n”?
        - “This gives us 5∗5∗50∗13 = 16250 questions” → how this number was determined?
        - “Then, to increase the data diversity, we perform clustering by each scenario to filter out the most similar questions and only pick *the most diverse questions*.” → The most “diverse” is vague.
    - On page 4, I have no idea about the following sentences:
        - We evaluate the refusal ratio of both the original chat model and the attacked model on 200 sensitive questions relevant to ’Illegal activity’, …. We perform decoding 3 times and average the results.
        - In Table 3, the definition of violation rate \gamma was not found. If I missed, please let me know.
        - In Figure 3, what is “MaliciousInstruction” dataset?
        - In table 5, please elaborate on the table caption. The numbers are significantly different?
    - In Figure 3, Table 3, and Figure 5, which evaluation method used between automatic and human evaluation?
- Interpretation of experimental results
    - In Figure 3, why the shadow-aligned model could not generate unsafe response to the prompts in PhysicalSafetyUnsafe dataset? Is this because the eval dataset is out-of-domain from the Shadow Alignment Dataset?

### Questions
I mentioned above all questions I have.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
