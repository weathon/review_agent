# Language-Specific Latent Process Hinders Cross-Lingual Performance

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 4, 8, 2

## Abstract
Large language models (LLMs) are demonstrably capable of cross-lingual transfer, but can produce inconsistent output when prompted with the same queries written in different languages. To understand how language models are able to generalize knowledge from one language to the others, we measure representation similarity across languages by centered kernel alignment (CKA) and cosine similarity. We also apply the logit lens to interpret the implicit steps taken by LLMs to solve multilingual multi-choice reasoning questions. We find LLMs predict inconsistently and are less accurate because they rely on representations of individual languages, rather than working in a shared semantic space. While larger models are more multilingual, we show their hidden states are more likely to dissociate from the shared representation compared to smaller models, but are nevertheless more capable of retrieving knowledge embedded across different languages. Finally, we demonstrate that knowledge sharing in small models can be facilitated by steering their latent processing towards the shared semantic space. This improves the models’ multilingual reasoning performance, as a result of more knowledge transfer from, and better output consistency with English.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper investigates why LLMs show inconsistent behavior when handling the same tasks across different languages. The authors analyze the internal representations of models such as Gemma 2 and Qwen 2.5 using techniques like centered kernel alignment (CKA), cosine similarity, and logit lens. They find that although larger models achieve higher multilingual accuracy, their internal representations become increasingly language-specific rather than shared, causing inconsistent reasoning across languages. Smaller models rely more on a shared, English-centric semantic space that supports more stable cross-lingual knowledge transfer.

To address this, the authors propose cross-lingual activation steering, a technique that nudges model activations toward English-aligned latent representations. This intervention improves multilingual reasoning and consistency in smaller models by enhancing their use of shared semantic structures. The work introduces a framework for quantifying cross-lingual alignment and offers an explanation for how model scale affects multilingual reasoning.

### Strengths
- Introduced cross-lingual activation steering to enhance reasoning performance.
- Combined CKA, cosine similarity, and the logit lens for multilingual analysis.
- Demonstrated a correlation between representation similarity and multilingual accuracy.

### Weaknesses
- The study relied primarily on multiple-choice tasks, which limited generalization to open-ended reasoning.
- Benefits are skewed toward languages written in Latin script.
- The validity of the findings is limited to two model families.
- There is no runtime or latency analysis conducted for steering.

### Questions
- Did you test how the steering method scales for larger or more complex models?
- How sensitive are results to the selection of layers?
- Could multilingual prompting (e.g., CoT in the target language) close the gap similarly?
- Did you try to construct shared-space vectors for non-English languages?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates how LLMs process multilingual inputs and they often fail to maintain consistent reasoning across languages with a novel framework of repurposing CKA as a metric to study structure similarity between language representations. The paper studies the latent representations of models such as Gemma2, and Qwen2.5 across scales using multiple benchmark datasets.

### Strengths
- Build a framework for analysing cross-lingual transfer with well-selected metrics such as CKA, cosine-similarity and logit-lens to quantify language representation overlap, which provides a clear and interpretable way to study cross lingual transfer.
- The analysis across models and layers are comprehensive.
- The paper reframes multilingual reasoning as a latent-space alignment problem, providing a clear direction for multilingual output consistency.

### Weaknesses
- The task is completely limited to multi-choice reasoning questions. This is a clever choice that is easy to measure the cross-lingual transfer with unambiguous labels. However, it also limits the generalisation to open-ended and generative multilingual reasoning.
- “Humans have an innate ability to apply common knowledge and perform reasoning skills consistently across different languages” itself is a very contentious claim. This paper also doesn’t need Jerry Fodor’s nativism as motivation.
- As shown in Figure4, there is a huge disparity between transfer effects among languages that are not very related, such as from Arabic to Swahili. The averaged performances illustrated in the main body of the paper is obviously inflated. Without modelling the language relatedness in the analysis, it is difficult to assess the generalisability of the findings in the paper. At least the specific language’s relatedness with English.
- Also, in terms of language relatedness, the paper investigates steering vectors only towards English. The motivation seems to be that the training data is primarily English. On that note, the training data volume per language is also not taken into consideration at all. Various factors that can be impactful to the results are not considered.
- Overall, the paper is novel in repurposing a useful metric such as CKA for cross-lingual analysis, however, the study remains Anglocentric and linguistically shallow.

### Questions
- Consider language relatedness, and training data volume per language, instead of simply taking average results across languages, which can inflate the results because of higher resource latin script languages.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper highlights the lack of consistency of outputs across languages, despite LLMs being capable of cross-lingual transfer. To measure generalization across languages, the paper measures representation similarity between languages by centered kernel alignment (CKA) and cosine similarity. They introduce three evaluation metrics to measure performance across languages: consistency, positive transfer and negative transfer.Their analysis show that LLMs don't use a shared semantic space which leads to low performance.They finally use steering vector to do steering towards English and find that it’s effective for smaller models.

### Strengths
1. The paper provides useful insights on how knowledge is represented and shared internally across languages in LLMs.  The authors investigate how LLMs transfer knowledge across languages. Through their experiments, they establish the usefulness of a shared semantic space for cross-lingual transfer. 
2. The paper proposes a cross lingual steering approach to improve cross lingual transfer for smaller models. 
3. Evaluation method is robust: The authors use ranking order for MCQ-styled questions across languages to measure consistency. They also measure positive and negative transfer.

### Weaknesses
1. Steering evaluation can include cross-dataset generalization to strengthen the claims. The current results only include effects on the same dataset.

### Questions
1. Do you notice any trends in consistency scores based on the linguistic distance of the language from English?
2. Does adding a steering vector towards English deteriorate performance on cultural/region-specific questions?

### Soundness
3

### Presentation
3

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
This paper studies cross-lingual consistency for the output when inputting the same query in different languages. The authors consider three factors: knowledge transfer (positive or negative), representation similarity, and activation steering. Then, the authors examine both small and large models, offering the key finding that a large model tends to handle each language independently, and a small model pushes all languages to a shared space.

### Strengths
Cross-lingual consistency is a recent topic. It evaluates the fairness in LLMs, which is very important in reall applications.

### Weaknesses
There are some concerns.

1.	Existing work [1] overshadows the novelty and contribution of this paper. For example, this paper follows a similar experimental design to [1], including CKA and Logits Lens examinations, layer-wise analysis, and activation steering. 

2.	Experiments are limited, which makes the paper not conclusive. The authors only conducted experiments on multiple-choice datasets. How about generation tasks?

3.	While the paper is clear, the authors spend too many spaces on introducing existing works, e.g., CKA, Logits Lens, and methods for activation steering. I would like to see more in-depth analyses, which make the paper more fruitful. 

[1] Are Knowledge and Reference in Multilingual Language Models Cross-Lingually Consistent?   https://arxiv.org/abs/2507.12838

### Questions
Refer to Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
1
