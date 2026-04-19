# Large Language Models can be Guided to Evade AI-generated Text Detection

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 5, 6, 5

## Abstract
Large language models (LLMs) have shown remarkable performance in various tasks and have been extensively utilized by the public.
However, the increasing concerns regarding the misuse of LLMs, such as plagiarism and spamming, have led to the development of multiple detectors, including fine-tuned classifiers and statistical methods.
In this study, we equip LLMs with prompts, rather than relying on an external paraphraser, to evaluate the vulnerability of these detectors.
We propose a novel Substitution-based In-Context example Optimization method (SICO) to automatically construct prompts for evading the detectors.
SICO is cost-efficient as it requires only 40 human-written examples and a limited number of LLM inferences to generate a prompt.
Moreover, once a task-specific prompt has been constructed, it can be universally used against a wide range of detectors.
Extensive experiments across three real-world tasks demonstrate that SICO significantly outperforms the paraphraser baselines
and enables GPT-3.5 to successfully evade six detectors, decreasing their AUC by 0.5 on average.
Furthermore, a comprehensive human evaluation as well as a validation experiment in the wild show that the SICO-generated text achieves human-level readability and task completion rates.
Finally, the strong performance of SICO exhibits its potential as a reliable evaluation tool for future detectors.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes SICO, a prompting-based approach to guide LLMs to evade machine-generated text detectors. SICO operates by extracting text features in natural language and optimizing in-context examples with token substitution. Experiments demonstrate that SICO outperforms paraphrasing-based baselines against six detectors across three tasks.

### Strengths
+ machine-generated text detection is an important research question
+ the proposed approach is flexible and promising

### Weaknesses
- The feature extraction part could be better motivated and investigated. The effectiveness of such a component reply on two assumptions: 1) LLMs could reliably summarize and extract the "features" of human-written texts from examples; 2) LLMs could use those style "features" to guide their generation. Both would need more support, in addition to showing that having the "feature extraction" component helps model performance.

- It is not guaranteed that the synthetic token substitution process would lead to decreases in $P_{AI}$ scores, right? Is the substitution performed only once for each passage? What are the stopping criteria? How often does this mechanism fail to find a new passage that has substantially lower $P_{AI}$ scores? In addition, I won't say this is "optimize", as this process is not differentiable, simply trying over and over again with different synonyms.

- Wordnet is used for token substitutions in this work. My understanding is that Wordnet could be quite noisy, containing words and tokens that are very infrequent in human-written text. As a result, it is surprising to see that Wordnet substitutions do not result in decreases in readability. I wonder if the authors might have conducted some filtering/selection of Wordnet to mitigate these issues.

- I wonder if the authors might have conducted an analysis of the correlation between detector performance and SICO evasion rate. In other words, is SICO greatly advancing the state-of-the-art on the most capable detectors or the less capable ones?

- I'm not sure if Section 4.4 is of great scientific value. The likes and dislikes barely indicate meaningful signals for research purposes as social media platforms could be quite random. The detection rate of 7.5% is arguably a hyper-loose lower bound of the actual detection rate by humans since it would require the users to go out of their way to respond to indicate their suspicion. A more rigorous study could be conducted by asking human readers to specifically perform the task of distinguishing human- and machine-generated text, similar to that of Section 4.3.

### Questions
please see above

### Soundness
2 fair

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This work proposes to utilize prompts to steer (the style of) the generator and deceive the AI-generated text detector. Substitution-based In-Context example Optimization method (SICO) is proposed to automatically construct such prompts. The methodology is feasible but this work lacks motivation for social benefits and automatic evaluation of the quality of the new generation.

### Strengths
+ The technology for constructing prompts is interesting, although I am not an expert in this direction.

+ The paper is well-written and thus easy to follow.

### Weaknesses
+ This work proposes to deceive the detector from a malicious perspective. We are interested in how this work could help from a benign perspective, e.g., how it helps to train a more robust detector.

+ Several major concerns on evaluation limit the soundness of this work:
    + For the results in Table 1, several baselines already achieve reasonable results, e.g., Parrot on Detect GPT, as about 0.5 AUC should have shown the outputs are quite obscured. Achieving a very low AUC is a negative signal for these experiments.
    + For the quality of the generated outputs, only human evaluation on DIPPER is considered, leaving several concerns 1) how will it perform on other datasets, 2) how will it perform on other automatic evaluation metrics, 3) why some scores are higher then human outputs, e.g., *Writing*.

+ Considering the detectors are basically 'classifiers', finding adversarial is desperately possible. The main finding of this work is not excitingly novel to our community.

### Questions
See bullet points in weakness.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes to equip LLMs with prompts, rather than relying on an external paraphraser, to evaluate the vulnerability of AI text detectors. Specifically, SICO iteratively substitutes words and sentences within the in-context examples to provide more representative demonstrations for LLMs to generate text that cannot be detected. Besides quantitative experiments, human evaluation and validation on online social platform prove the effectiveness of the proposed method.

### Strengths
1. The proposed in-context example method is cost efficient and universally useful against a wide range of detectors.
2. Quantitative experiments show that the proposed method outperforms existing paraphraser baselines across three datasets and multiple detectors.
3. The evaluation on Reddit and relevant analysis make the effectiveness of the proposed method more vivid and realistic.

### Weaknesses
Although SICO only requires 40 human-generated examples and a limited number of LLM inferences, but it still costs approximately 1 USD using the GPT-3.5 API.

### Questions
1. Since each evasion only runs once during evaluation, then what's the arguments for GPT-3.5 in this paper, such as the temperature and top-p values.

2. As shown in Table 1, the proposed SICO and human prompt obtain much better performance. DIPPER also achieves good performance on QA and Review dataset. However, the third best performance on Writing dataset is Parrot, which is much better than DIPPER. I am wondering, in Section 4.3, why DIPPER is considered for Writing dataset not Parrot. 

**Suggestions**

1.For values in Table 2, since the higher the better, perhaps you could mark best performance per column in boldface.

2. In Section 4.4, it would be interesting if another competitive baseline (such as Human-Written) is used to answer the same question as the proposed SICO on Reddit. In that case, we may have a better idea which one is more preferred by users or which one is easier to be recognized as chatgpt.

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
Summary: This paper aims to evade existing text detection systems through in-context optimization, making the AI-generated text more similar to humans. The method works by initializing the in-context prompts by some human-crafted pairs and gradually replacing the words or sentences to minimize the detectability. The final optimized in-context examples and task prompts are used as a combined new prompt to generate undetectable text given a new query. Extensive experiments through diverse datasets and models validate the effectiveness of this novel attack over several existing baselines. The authors also claim that the resulting attack prompt can be generalized to other detectors.

### Strengths
Strength:
The attack on detectors is an important task and the authors propose a novel prompt attack by in-context optimization. On tested datasets and models, this approach works well compared with several baselines under their setting.
The experiments are comprehensive and solid to support their main results, though some weaknesses remain.
The whole paper is written well and easy-to-follow. The experimental settings are clear and easy to reproduce.

### Weaknesses
Weakness: There are three main weaknesses.
The first weakness is the diversity. I understand that the method aims at finding an optimized prompt to make the model learn human-style writing. However, the authors only show the detection result or the readability. One crucial problem is whether the generated text will lose diversity. It might simply learn the features provided in the prompt and universally apply those features to all generated text. Then, although the detection performance drops the each single text remains readable, the overall generations might lack inter diversity. Thus, this will limit its overall utilization.
The second main weakness is transferability over data. The proposed method requires in-context examples from a given task and its corresponding dataset. However, in practice, this setting is too ideal since many users will use ChatGPT to generate various outputs under different conditions. Thus, the task and in-context examples are not given (How can you obtain 32 in-context examples for each new query?). Besides, the author did not mention how the result on one dataset generalizes to other datasets. It make sense for the optimized prompt in one dataset to beat other baselines since one specific dataset and task has their intrinsic similarity. But most other baselines work in general. So, the comparison is not fair, and whether the approach really works is questionable. 
The third weakness is the experiments are only conducted on GPT-3.5. I am expecting on more models, including open-sourced models for easy and cheap reproduction.
Other weaknesses include:
the method does not test on non-English text, limiting its utility. 
Also, the method requires many in-context examples, but it might exceed the context window or limit the generation length due to the context window limit. 
The methods did not report the length effect on generated text, which is crucial for detection.

### Questions
Questions:
See weakness.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
