# Conversational Drug Editing Using Retrieval and Domain Feedback

- Decision: Accept (poster)
- Scores: 6, 6, 6

## Abstract
Recent advancements in conversational large language models (LLMs), such as ChatGPT, have demonstrated remarkable promise in various domains, including drug discovery. However, existing works mainly focus on investigating the capabilities of conversational LLMs on chemical reactions and retrosynthesis. While drug editing, a critical task in the drug discovery pipeline, remains largely unexplored. To bridge this gap, we propose ChatDrug, a framework to facilitate the systematic investigation of drug editing using LLMs. ChatDrug jointly leverages a prompt module, a retrieval and domain feedback module, and a conversation module to streamline effective drug editing. We empirically show that ChatDrug reaches the best performance on all 39 drug editing tasks, encompassing small molecules, peptides, and proteins. We further demonstrate, through 10 case studies, that ChatDrug can successfully identify the key substructures for manipulation, generating diverse and valid suggestions for drug editing. Promisingly, we also show that ChatDrug can offer insightful explanations from a domain-specific perspective, enhancing interpretability and enabling informed decision-making.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a framework named ChatDrug that leverages large language models to streamline effective drug editing. The authors demonstrate that ChatDrug outperforms existing methods on all 39 drug editing tasks and can offer insightful explanations from a domain-specific perspective. The authors comprehensively evaluate of ChatDrug's performance on 39 drug editing tasks, and conduct a detailed analysis of ChatDrug's ability to provide domain-specific explanations for its decisions.

### Strengths
1.  This is the first paper to leverages large language models for effective drug editing. 
2. The authors conduct comprehensive evaluation of ChatDrug's performance on drug editing tasks, as well as the analysis of explainability.

### Weaknesses
1. The domain feedback function in the ReDF is defined as the evaluation metric. I wonder if it may potentially cause 'information leakage', as it is accessing the information of 'success sequence editing'.
2. While the detailed implementation of the modules are new, the paradigm is lack novelty, as it falls into the 'prompt, retrieval for factuality, evaluate and repeat' paradigm, which is not new.
3. The paper includes much domain knowledge for demonstration, which causes troublesome in comprehending the main idea. The authors may consider simplify the terms and focus on the main experimental phenomenon only.

### Questions
Refer to weaknesses.

### Soundness
3 good

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
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes ChatDrug, a framework for conversational drug editing using large language models (LLMs). ChatDrug leverages prompt design, retrieval and domain feedback, and conversation modules to generate diverse and valid suggestions for drug editing. ChatDrug can handle various types of drugs, such as small molecules, peptides, and proteins. The paper evaluates ChatDrug on 39 drug editing tasks and shows that it outperforms several baselines.

### Strengths
1. The paper addresses an important and challenging problem of drug editing using LLMs.
2. The paper introduces a novel and comprehensive framework that incorporates domain knowledge and interactive feedback for drug editing.

### Weaknesses
1. The paper does not provide a clear comparison or analysis of the different LLM backbones used in ChatDrug.
2. The paper does not provide any user study or evaluation from domain experts to validate the usefulness and usability of ChatDrug.

### Questions
1. How do you ensure the quality and reliability of the retrieval and domain feedback module? How do you handle the cases where the retrieved information is inaccurate or outdated?
2. How do you measure the similarity between the input and output drugs? How do you balance the trade-off between similarity and diversity in drug editing?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper presents an approach to editing small molecules / peptides / proteins by interacting with an out-of-the-box LLM such as ChatGPT Turbo.

They describe how to prompt the model and then how to give feedback using a small number of supervised examples  using a retrieval and domain feedback module which finds the positive example that is closest to the negative prediction.

The results are strong across 3 tasks that involve small molecules, peptides and proteins but I have some questions / concerns.

### Strengths
-The paper presents an interesting approach to drug editing that leverages pretrained LLMs. It is interesting to know that non-protein LLMs out of the box can reason about small molecules / proteins / peptides through iterative in-context learning.

### Weaknesses
After the author response, the authors answered my concern (1).  

For (2), I emphasized that I wasn't suggesting comparing with supervised baselines on a supervised task, but rather reframing some of the datasets as a few shot task could increase the impact of the work (but I agree may be a significant challenge since it involves quantitative measurements).

Accordingly I have now increased my score.

-----

(1) I am not sure about the small molecules experiment but atleast for the other two, the only baselines provided are random. However, I don't feel this is fair since the author's approach is seeing a few positive examples. Would be great to have a baseline that uses a similar number of examples.

(2) The tasks focus on some basic properties of molecules like water solubility. However, often what we are really interested in drug design is a quantitative measurement like the binding affinity to a given target (or something similar). I don't see any results along these lines.

In small molecules there exist benchmarks that measure the binding to specific targets or other more detailed attributes. For example the datasets/baselines used in this paper:

https://arxiv.org/pdf/2206.07632.pdf

----

(Maybe less related since the paper's focus seems to be more on small molecules with proteins a secondary experiment) For  proteins there exist benchmarks like Deep Mutational Scanning and FLIP and associated works that try to optimize a protein towards one of these attributes.

https://www.biorxiv.org/content/10.1101/2021.11.09.467890v1

https://www.biorxiv.org/content/10.1101/2021.07.09.450648v2

https://arxiv.org/abs/2303.04562

https://arxiv.org/abs/2307.00494

### Questions
I am curious as to what baselines the authors think are fair for each task that use similar amounts of supervised data. This is a bit unclear to me in the paper.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
