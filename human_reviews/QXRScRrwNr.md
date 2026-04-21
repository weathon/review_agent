# In-Context Learning with Retrieval Augmented Encoder-Decoder Language Models

- Avg Score: 5.33
- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 5, 6

## Abstract
In this paper, we investigate the in-context learning ability of retrieval-augmented encoder-decoder language models. We first conduct a comprehensive analysis of the state-of-the-art ATLAS model and identify its limitations in in-context learning, primarily due to a mismatch between pretraining and testing, as well as a restricted context length. To address these issues, we propose RAVEN, a model that combines retrieval-augmented masked language modeling and prefix language modeling. We further introduce Fusion-in-Context Learning to enhance the few-shot performance by enabling the model to leverage more in-context examples without requiring additional training or model modifications. Through extensive experiments, we demonstrate that RAVEN significantly outperforms ATLAS and achieves results comparable to the most advanced language models in certain scenarios, despite having substantially fewer parameters. Our work underscores the potential of retrieval-augmented encoder-decoder language models for in-context learning and encourages further research in this direction.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper studies the retrieval-augmented language model of encoder-decoder architecture. Specifically, this paper bridges the gap between pre-training and inference of the sota retrieval-augmented encoder decoder language model ATLAS by combining retrieval-augmented masked and prefix language modeling. Then, this paper designs Fusion-in-Context Learning and In-Context Example Retrieval to enhance the few-shot performance of retrieval-augmented encoder-decoder language models. Experimental results show the proposed method is effective.

### Strengths
1. Propose an interesting question “in-context learning in retrieval augmented language model”. Decoder-only language models are good at in-context learning while hard to use retrieved information. Encoder-decoder language models are good at retrieval augmented generation due to their separated encoder while unstable for in-context learning. This paper aims to introduce in-context learning ability into encoder-decoder language models.
2. This paper is well-written and easy to follow. The proposed method is well-motivated and very easy to implement.
3. Detailed experimental results on diverse datasets and the additional analysis can answer the main concern of motivation.

### Weaknesses
1. Although I admit that exploring the in-context learning (ICL) in retrieval-augmented language model is an interesting research question, I think the specific research points of this paper are still limited. This paper only study how to make the language model perform better on question-answering task under the retrieval-augmented paradigm. This is inconsistent with our accepted definition and expectation of ICL. ICL enables the language model to learn and perform various tasks using a few input examples. Just focusing on knowledge-intensive tasks such as question-answering cannot be called ICL. It is more like few-shot learning for question-answering.
2. The technical contribution of this paper is limited. 
 - In order to bridge the gap between pre-training and inference in ATLAS, this paper introduces prefix language modeling. It just puts the special token at the end of the texts. Prefix language modeling is a well-known method in the pre-training of decoder-only language models such as GPT. 
 - In order to input more examples to the retrieval-augmented language model, this paper just feeds different examples to FID encoder with each passage. 
The above two points are the core technical contribution of this paper which is limited. 
3. This paper emphasizes the advantages of the encoder-decoder architecture model in retrieval-augmented compared to the decoder-only model, but its core technical contribution is to change the masked language modeling of the encoder model to the prefix language modeling of the decoder-only model. This point is very contradictory. 
4. Some decoder-only pre-trained language models with considerable size such as LLama-7B should be used as baselines. Many papers show LLama-7B can achieve better performance in Table 3. I hope the author can carefully analyze the advantages of the encoder-decoder architecture model compared to the decoder-only model in terms of retrieval-augment.

### Questions
Please see the weakness section.

Additional questions:
1. In Figure 6, the x-axis label '>8' should be a precise number instead of a span.
2. The efficiency of the Raven should be taken into consideration. Especially comparing it to the baselines.
3. Please show the differences between Raven and "FiD-ICL: A Fusion-in-Decoder Approach for Efficient In-Context Learning".

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper is built on the work of ATLAS.
The paper explores the drawbacks of ATLAS in the aspect of in-context learning (ICL) and identifies two issues of ICL with ATLAS: (1) mismatch between pretraining and testing and (2) restricted context length.
To tackle the first issue, the paper proposes to continually pretrain ATLAS masking 10% of the tokens at the later part of the sequence.
To tackle the second issue, the paper proposes to blend limited in-context examples into Fusion-in-Decoder architecture to avoid increasing the length of the input to the decoder.
The paper finally shows the proposed method outperforms ATLAS and has comparable performance to decoder-only LLMs.

### Strengths
(1) The explored topic, combining in-context learning (ICL) and retrieval augmented generation (RAG) is very interesting.

(2) The paper is well-written, and the proposed idea is simple and easy to follow.

(3) The paper identifies two drawbacks of ATLAS on ICL, including (1) the mismatch between pretraining and testing and (2) the restricted context length, which makes sense to me.

(4) Compared with ATLAS, the performance is significantly improved.

### Weaknesses
(1) Though the proposed method has a significantly better few-shot performance than ATLAS, the work seems to be incremental compared with ATLAS in the aspect of the modeling. The first method is to continuously pretrain ATLAS by masking the later part of the sequence. The second method is to blend different limited in-context examples into different passages in the Fusion-in-Decoder architecture. The whole architecture is exactly the same as ATLAS, which makes me feel the proposed method is more like the tricks for ATLAS for better ICL performance, including a fine-tuning method and a prompt engineering method.

(2) Though the proposed method outperforms ATLAS, it still cannot match decoder-only LLM as shown in Tables 3 and 5. (I do raise questions for this comparison in the question section.)

### Questions
I wonder how to fairly compare ATLAS and LLM since they may not use the same datasets.

(1) Saying ATLAS uses data A to perform pretraining and data B as a corpus, and LLM uses data C for pretraining, does A+B=C? If not, how to fairly compare them?

(2) The second question is also raised from the extra cost of using an external dataset in retrieval augmented generation (RAG). Though it's said in the paper ATLAS has fewer parameters 'despite having substantially fewer parameters', whether it's comparable to LLM measured on inference cost is not clear to me.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper focuses on advancements in language modeling, particularly exploring the training, performance, and applications of two versions of a model named RAVEN, in comparison with other models like ATLAS. Key aspects of the paper include:

1. Fusion-in-Context Learning: A notable contribution is the introduction of Fusion-in-Context Learning. This strategy is designed to address the limitations of constrained context length in base models like T5 or ATLAS, aiming to improve scalability and performance in in-context learning scenarios.

2. Comparative Performance Analysis: The paper presents a comparative analysis of RAVEN against other prominent models such as GPT-3, PaLM, Codex, and RETRO. This analysis, focusing on tasks like Natural Questions and TriviaQA, demonstrates that RAVEN, especially with FiCL, shows enhanced performance in both zero-shot and few-shot learning scenarios.

### Strengths
1. Innovative Approach: The introduction of Fusion-in-Context Learning (FiCL) to address the limitations of constrained context length in base models like T5 or ATLAS is a significant innovation. This strategy improves scalability and performance in in-context learning scenarios.

2. Comparative Performance: RAVEN models demonstrate superior performance compared to other models like GPT-3, PaLM, Codex, and RETRO in tasks such as Natural Questions and TriviaQA, particularly in zero-shot and few-shot settings.

### Weaknesses
1. The paper demonstrates the effectiveness of the proposed method primarily in the context of encoder-decoder models. However, its effectiveness in popular left-to-right language models, which are widely used, is not explicitly addressed. This omission can limit the understanding of how the proposed method might perform or be adapted to these prevalent LMs such as LLaMA and GPT.

### Questions
N/A

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
4 excellent
