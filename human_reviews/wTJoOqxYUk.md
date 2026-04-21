# Code Representation Pre-training  with Complements from Program Executions

- Avg Score: 4.75
- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 6, 5, 5

## Abstract
Large language models (LLMs) for natural language processing have been grafted onto programming language modeling for advancing code intelligence. Although it can be represented in the text format, code is syntactically more rigorous in order to be properly compiled or interpreted to perform a desired set of behaviors given any inputs. In this case, existing works benefit from syntactic representations to learn from code less ambiguously in the forms of abstract syntax tree, control-flow graph, etc. However, programs with the same purpose can be implemented in various ways showing different syntactic representations while the ones with similar implementations can have distinct behaviors. Though trivially demonstrated during executions, such semantics about functionality are challenging to be learned directly from code, especially in an unsupervised manner. Hence, in this paper, we propose FuzzPretrain to explore the dynamic information of programs revealed by their test cases and embed it into the feature representations of code as complements. The test cases are obtained with the assistance of a customized fuzzer and are only required during pre-training. FuzzPretrain yielded more than 6\%/19\% mAP improvements on code search over its counterparts trained with only source code or AST, respectively. Our extensive experimental results show the benefits of learning discriminative code representations with program executions.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper focuses on improving code intelligence in Large Language Models (LLMs). Recognizing the challenges of learning from code's semantics and varied implementations, the authors introduce "FuzzPretrain." This approach uses dynamic program test case data, obtained via a custom fuzzer, during pre-training to enrich code feature representations. Compared to models trained with just source code or AST, FuzzPretrain achieved improvements in code search performance, emphasizing the value of integrating program execution insights.

### Strengths
incorporating program executions to pre-train LLMs is new despite a narrow and low-level contribution.

### Weaknesses
- My biggest concern is the key idea of using program execution to learn code models is just not new. Authors seem to be completely unaware of the vast literature of neural code models based on dynamic executions. Here are some papers for authors' reference:

  1. dynamic neural program embedding for program repair
  2. blended precise semantic program embeddings
  3. Improving Neural Program Synthesis with Inferred Execution Traces
  4. Latent Execution for Neural Program Synthesis Beyond Domain-Specific Languages
  5. Code vectors: understanding programs through embedded abstracted symbolic traces

   Even though they do not directly target LLM rightfully so given that LLMs are not even around at the time those papers are published, their works already share the insight of how dynamic execution can benefit learning of code embeddings. Therefore, it's entirely inappropriate for authors to totally ignore them. 

- The evaluation task of clone detection is poorly handled. First and foremost, clone detection is almost an entirely syntactic task as tools are asked to detect the syntactic similarity of code, however, incorporating execution traces are semantic information that totally ignores the syntactic features. So this clone detection task does not even match the insight of the paper. Of course, I am aware of the type 4 semantic clones, however, the question is, how many are those in POJ-104, authors provide no information in this regard, and it's more than reasonable to assume there are very few if any used in the evaluation.

- In ablation study, Fig 3 demonstrates that in Defect DIM and DID is not necessary because removing them actually yields a bigger gain over FuzzPretrain. This casts doubt on the effectiveness of your technique.

### Questions
See weaknesses

### Soundness
2 fair

### Presentation
2 fair

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
This paper proposes FuzzPretrain which leverages fuzzing to derive input/output test cases for doing continued pre-training on CodeBERT and UniXcoder. Their pre-training strategy encompasses three objectives: (1) Static information modeling: learning from the sequence representation of code using MLM, (2) Dynamic information modeling: learning to distinguish between the positive test cases and randomly sampled negative test cases for a given code snippet, using binary cross-entropy loss, and (3) Dynamic information distillation: learning to position the code representation corresponding to the code snippet alone near to the representation corresponding to the code snippet concatenated to the test cases, using contrastive learning with randomly sampled negative code representations.They conduct this pretraining using 1.2M codes snippets in C/C++/Python/Java from CodeNet.  They evaluate their approach (without any further fine-tuning) on four downstream code understanding tasks: code search, clone detection, defect detection, and text-to-code search. Results demonstrate improvements over baselines which use only static information.

### Strengths
- This paper proposes leveraging fuzzing for pre-training which may inspire future techniques for building better pre-training datasets and objectives for code models.
- The approach proposed by this paper yields impressive improvements for code search and smaller improvements across three other code understanding tasks, relative to comparable models which use only static information.
- The pre-training tasks that the authors propose are quite interesting, and the extensive analyses and ablation studies that are included in the paper are helpful for understanding the contributions of these tasks.
- The paper is very well-written.

### Weaknesses
- The authors seem to suggest novelty in using dynamic program information for learning code representations through claims like "To the best of our knowledge, this is the first attempt to unveil the benefits of dynamic program information on code representation pre-training" and "...we make the first attempt to derive the program functionality from the dynamic information collected during executions." However, there is work that does similar things, one of which they have cited, and others that they have cited. Namely, they have not cited "TRACED: Execution-aware Pre-training for Source Code" (https://arxiv.org/pdf/2306.07487.pdf) leverages dynamic information, specifically executable inputs and corresponding execution traces, for pre-training. Though fuzzing is not used there, it is used for building code representations in a paper that is cited: "Understanding Programs by Exploiting (Fuzzing) Test Cases" (https://arxiv.org/pdf/2305.13592.pdf), though they are not actually using them for pretraining. It seems that the contribution is more around using specifically fuzzing for pre-training. I believe this should more clearly be conveyed.
- Related to the previous point, they present results for an approach that does use dynamic information for pre-training: CodeExecutor (https://arxiv.org/pdf/2305.05383.pdf). However, they do not present this on the main code search task in Table 1 (which is also more aligned with what CodeExecutor was actually benchmarked on). In fact, many of the "state-of-the-art" models listed in Table 3 are not included in Table 1 for code search. It is not clear why these results were excluded from the paper. The same goes for the analyses in Figures 4-5. Since much of the focus was on code search, the ablations and analyses would be based on that. 
- As the authors themselves acknowledge in the limitations section, this work is focused on code understanding tasks and no generative tasks. I find this a bit troublesome because the underlying model that is used, UniXcoder, was originally designed to also handle generative tasks. CodeExecutor was also benchmarked on code generation. The authors do not report results for generative tasks like code generation or summarization.

### Questions
1) Please address the points made above.
2) Please provide additional details on fuzzing. Namely, on average, how many test cases are generated using fuzzing for each code snippet? On average, how long does it take to generate test cases for each example? How scalable is this technique for a much larger pre-training corpus?
3) Some parts of the DIM objective are a bit unclear to me. Namely, it seems that both the inputs and outputs are included in the test cases. It seems that a model could learn to exploit just the structure of the input rather than reasoning about the output because of this, especially since negative examples are sampled randomly from the whole corpus. More concretely, suppose you have a code snippet that takes in a list of integers and the expected functionality is returning the product of these integers. So, a positive test case would be (input: [5,3,2], output: 30) and a negative test case corresponding to another snippet could be (input: "hello", output: "Hello"). If the objective is trained using such examples, then the model may learn to just exploit surface patterns like the input being a list of integers versus the input being a string. To ensure that the model is actually reasoning about runtime behavior, it seems that you have to force the model to reason about the output (e.g., masking the output) or do hard negative example mining. Could you clarify this?


Other suggestions:
- Please fix the capitalization in paper titles in the appendix (e.g., Graphcodebert $\rightarrow$ GraphCodeBERT, Coderetriever $\rightarrow$ CodeRetriever)
- Beginning of Section 3.2, "presentations of code" $\rightarrow$ "representations of code"

### Soundness
3 good

### Presentation
4 excellent

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposed FuzzPretrain to incorporate program dynamic information to code pre-trained models to supplement the missed information. Specifically, FuzzPretrain utilized a customized fuzzer to obtain some testcases of a code snippet and then utilize the testcases as well as the original code snippets for pre-train. It designed three pre-training objectives, in addition to MLM, dynamic information matching (DIM) is designed to separate test cases of different programs and dynamtic information distillation objective is used to learn the holistic information from the code and test cases. Four downstream tasks including code-to-code search, clone detection, defect detection and text-to-code search are used to evaluate the model effectiveness. Furthermore, an ablation study is also conducted to illustate the effectiveness of their proposed pre-training objectives.

### Strengths
+The idea to incorporate program dynamic behaviors in pre-training seems ok to supplement current code pre-trained models.
+This paper is easy to follow and understand.

### Weaknesses
-The technique novelty is lack. I agree that the program dynamic information is important and can benefit code pre-trained models, however in this paper, the usage of dynamic information is too easy. It just concatenates the test cases with the original code for model training. I am not sure how much dynamic information is contained in the test cases. Furthermore, I am confused that why test cases are enough for using dynamic information? Lastly, is there any other way to use dynamic information rather than such a simple way? 

-In terms of model design, the novelty is limited. It uses BERT-style model as the model architecture for pre-training, why not use more powerful encoder-decoder model and there are some works such as CodeT5[1] and CodeT5++[2] have proved encoder-decoder is better than CodeBERT. Furthermore, the designed pre-trained tasks DIM and DID are also simple, DID is similar to InfoNCE[3].

-In terms of downstream tasks, the evaluation tasks are also limited. There are only four downstream understanding code tasks, more code-related tasks are need to evaluate to confirm the effectiveness of the proposed approach.

[1] Wang et al. CodeT5: Identifier-aware Unified Pre-trained Encoder-Decoder Models for Code Understanding and Generation.

[2] Want et al. CodeT5+: Open Code Large Language Models

[3] Liu et al. Contrabert: enhancing code pre-trained models via contrastive learning

### Questions
-Why use test cases to represent code dynamic information, can test cases are sufficient to represent code dynamic information?

-How to construct negative samples in DID pre-training task at equation 3? Please give more explanation?

-Can FuzzPretrain wok well in some code generation tasks such as code summarisation, code completion?

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposed FuzzPretrain, a new pretraining method for learning code representations that incorporates program execution semantics. More specifically, FuzzPretrain uses fuzzing to produce test cases for programs and then incorporate a binary classification objective to match the programs and the corresponding test cases. In addition to this object, MLM and contrastive learning methods are also used during pretraining. Experiments are conducted for code-to-code search, clone detection, defect detection and text-to-code search tasks. And results show that the proposed method is able to outperform recent works that uses similar-sized models.

### Strengths
S1: This paper explores a very important domain, as a reliable, semantic-aware code embedding could be helpful for many code-related tasks;  
S2: This paper proposed an interesting method that combines representation learning and code execution via fuzzing, and provides a simple way of incorporating the test cases into the pretraining tasks;  
S3: The ablation studies are rather complete for the readers to understand the contribution of each part in the pretraining objectives.

### Weaknesses
W1: The presentation of this work is quite poor. More specifically:
* The notations are a bit (unnecessarily) complex (e.g., as a matter of fact, you can hardly find a symbol without any super/sub-script). An example is with Equation 2, I don't think it is necessary to write down the formulation of a linear layer for binary classification, nor the cross-entropy loss;
* Table 1 is a bit hard to parse, I assume they are some kind of "transfer" between different programming languages (nothing is mentioned in the caption), but even with that assumption, there are many questions needed to be answered;

W2: Some of the experiment settings are questionable. See the "questions" section for details;  
W3: Compared with previous work, the improvements are quite marginal (i.e., 1 point or less) on 2 of the 3 reported tasks. Also, it would be great to mark the parameter count in table 3.

### Questions
Q1: In table 1, why is the first row in grey? Same question with table 2;  
Q2: For table 1, what does the languages in the first and second row mean? What about C/C++, in the first line of Section 4, it was mentioned that C/C++ is also part of the training data.  
Q3: Still table 1, is "w/o DIM" and "w/o DID" rows accumulative? If not, is "w/o DIM + w/o DID" the same as MLM variants in the table?  
Q4: Can you explain why you alternate the three objectives between different mini-batches, and not simply add them or interpolate between them?   
Q5: Since the model is both trained and tested on CodeNet, can you elaborate on how you split the data and if some precautions are taken to prevent implicit data leakage?   
Q6: Can you give a hypothesis on why the effect of DIM and DID objectives are vastly different for CodeBERT and UniXcoder models (see Table 1)  
Q7: Do you plan to compare with some LLM-based embedding methods? E.g., embedding models from OpenAI.  
Q8: What does the color entail in Figure 3? It was mentioned that only a subset of the points are picked to be colored, how do you pick this subset? My concern is that if you pick the ones that are closely clustered in Fig 3(b) and visualize the same points in Fig 3(c), it will look scattered as well. Thus I don't think this (the colored points) shows if one embedding is better than the other. However, the groupings do look better spaced than UniXcoder.

### Soundness
3 good

### Presentation
1 poor

### Contribution
2 fair
