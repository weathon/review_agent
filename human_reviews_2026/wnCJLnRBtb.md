# Context Similarity Structure Shapes the Emergence of Reliable In-Context and In-Weights Mixtures

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
We aim to train models that co-develop in-context learning (ICL) and in-weights learning (IWL), and flexibly switch between them based on context relevance. Such models should exploit closely related in-context examples while relying on IWL when examples are irrelevant. Although LLMs exhibit both modes, standard task-specific fine-tuning often erodes ICL, motivating IC-Train, a form of fine-tuning with in-context examples. When trained under IC-Train, prior work has shown that emergence of ICL depends on factors such as task diversity and training duration. We show that an overlooked factor is the similarity structure between target inputs and context examples. Of the two existing modes of context-target pairing, random context leads to IWL dominance, while only similar examples in context causes ICL to degenerate to copying labels without regard to relevance. To address this, we propose Contrastive-Context which enforces two types of contrasts: (1) mix of similar and random examples within a context to evolve a correct form of ICL, and (2) varying grades of similarity across contexts to evolve IWL-ICL mixtures. With experiments on real sequence to sequence learning tasks on four models, we show that Contrastive-Context strengthens ICL while preserving IWL, outperforming random and nearest-neighbor sampling in both in-domain and out-of-domain evaluation. Theoretical analysis and diagnostic probes confirm that contrasted contexts yield stable ICL–IWL mixtures, avoiding collapse into pure ICL, IWL, or copying. Our results establish similarity structure as a key driver of reliable ICL under fine-tuning an LLM for a task.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates how structural similarities between target inputs and contextual examples influence the emergence of in-context learning (ICL) capabilities in large language models (LLMs). The authors observe that when contextual examples are overly similar, the ICL process tends to degenerate into simple label copying, disregarding the true relevance of examples. To mitigate this issue, they propose a method called Contrastive-Context, which enforces two types of contrastive structures:

(1) mixing similar and random examples within a context to encourage the development of a more robust form of ICL, and

(2) varying the degree of similarity across contexts to facilitate a balanced evolution between in-weights learning (IWL) and ICL.

Experimental results demonstrate that Contrastive-Context enhances ICL performance while maintaining IWL effectiveness, outperforming both random sampling and nearest-neighbor sampling strategies in both in-domain and out-of-domain evaluations. Furthermore, the paper provides theoretical analysis based on a two-layer Transformer model to support its empirical findings.

### Strengths
1. The study addresses an important and timely question—how to improve the in-context learning ability of LLMs—which is of substantial relevance to the field.
2. The authors not only introduce a novel technique, Contrastive-Context, supported by convincing experimental evidence, but also complement it with theoretical analysis and diagnostic probes that help elucidate the underlying mechanisms of their approach.

### Weaknesses
1. The primary limitation of this paper lies in its insufficient positioning within the broader research landscape. As a result, both its contributions and innovations remain unclear. Specifically:

   1. The claimed innovation of the paper rests on the assumption that *inter-example similarity is a critical but underexplored factor*. However, this assumption is neither adequately explained nor empirically validated. While numerous prior studies have examined how various factors influence the emergence of in-context learning (ICL), the authors do not convincingly justify why inter-example similarity should be considered more important than other factors. Moreover, the paper lacks comparative experiments that could substantiate the claimed significance of this factor.

   2. Several existing works[1,2] have proposed methods to enhance the emergence of ICL by focusing on example selection. The authors should at least include comparisons with these related approaches to clarify the novelty and relative effectiveness of their method.

        [1] Breaking through the learning plateaus of in-context learning in transformer.

        [2] Task diversity shortens the icl plateau.


   3. The paper makes several key claims that are not supported by sufficient evidence. For example:  "A natural alternative is to fine-tune in ICL mode (IC-Train), but this is challenging due to competition with in-weights learning (IWL)"[line 43-44]. and "Such adaptation should boost accuracy on inputs resembling the new examples, while retaining generalization to inputs without close neighbors." [line 37-38]

    

2. The experiments are conducted only on the language translations tasks. It is questionable whether the findings also apply to other area, like math or coding

### Questions
See weakness

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper studies how to fine-tune LLMs that balance in-context learning and in-weight learning, and allow switching between them based on the relevance of context examples to the target input. It provides analysis on how target-context similarity during training influences IWL and ICL balance and proposes a simple contrastive-context training strategy. They conduct an theoretical analysis with clearly defined assumptions on a regression task using a two-layer two-parameter model, to show how different contrasts in context-target influence the ICL behavior. They show that similarity structure between target and context is a key component for ICL, demonstrated on machine translation tasks over 4 open-source LLMs.

### Strengths
1. The problem addressed in this work is quite relevant for the community. The proposed strategy is simple and effective. Convincing empirical evidence is provided to prove its effectiveness. The proposed data strategy is tested across four LLMs at different scales over four machine translation tasks. The empirical evaluation scheme over varying grades of similarity is quite insightful. This scheme can be used for future ICL research since it captures both ICL and IWL performance as two ends on the spectrum. 

2. It is a novel conceptual findings that random and similar-context pushes the model toward different extremes of ICL and IWL. Additionally, the method is theoretically grounded in a simplified two-layer model analysis, providing mechanistic understanding on how the similarity structure influences the performance. 

3. The paper is well-written and text part is easy to follow.

### Weaknesses
1. Contrastive-context relies on synthetic paraphrases with epsilon=0.1, which is a form of data augmentation. It is unclear whether the observed improvement arises from this synthetic paraphrasing or just from the mixture of random and similar/augmented sequences. Two controlled ablations would clarify this. First, by either setting p=0 or epsilon=0 to isolate the effect. Second by varying paraphrasing strength from zero to high paraphrasing. 

2. The presentation of figures could be improved. Several plots lack axis labels which makes them hard to understand, especially the bottom right plots in Figure 2. Curves in Figure 3 appear overly smoothed which reduces the transparency of the experiment. Please provide raw values as well in future.

3. The evaluation lacks robustness analysis. The method is only tested on machine translation tasks and the OOD setup represents only a mild domain shift within the same language-pair task. In Figure 3, IWL performance is higher for OOD set versus ID, implying overlap with pretrained data and undermining the OOD definition. Error bars of the different performance curves are also not provided.

### Questions
1. Why did the authors select only sequence to sequence task for evaluation? Seq2seq is a relatively easy real-world task compared to tasks such as QA or other reasoning tasks. It would strengthen the findings, if the benefits are also demonstrated on other difficult tasks.

2. Why do plots in Figure 3 show noticeable fluctuations over the training steps, especially for Copy-scores?

3. Recently, few works [a][b]  have shown that repetitions or augmentations of the target sample in the context leads to strong ICL and IWL performance when trained with a mixture of random and bursty sequences, similar to this paper. How does proposed contrastive-context method conceptually differ from these studies.
- [a] The emergence of sparse attention: impact of data distribution and benefits of repetition (Zucchet et al. 2025)
- [b]  What Matters for In-Context Learning: A Balancing Act of Look-up and In-Weight Learning (Bratulic et al. 2025)

### Soundness
3

### Presentation
2

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
This paper investigates a core problem prevalent in Large Language Model (LLM) fine-tuning: how to develop and utilize In-Context Learning (ICL) while preserving the generalization capabilities obtained through In-Weights Learning (IWL). The paper proposes a new training strategy called "Contrastive-Context" which involves: (1) mixing similar and random examples within a single context; and (2) varying the grades of similarity across different contexts. Operationally, the strategy selects, with a certain probability, either a "most-similar" real example, a "highly-similar" synthetic example (i.e., a paraphrase), or a "random" example, and fills the remaining $k-1$ slots with random examples. Experiments conducted on 4 Machine Translation (MT) tasks across 4 different LLMs (1B to 8B) show that Contrastive-Context performs exceptionally well in both In-Domain (ID) and Out-of-Domain (OOD) evaluations.

### Strengths
1. The paper addresses a very practical and important problem: the "forgetting" or "degradation" of ICL capabilities during fine-tuning. It clearly identifies "context similarity structure" as a key factor regulating the ICL-IWL balance.
2. The claims are strongly supported through a multi-faceted approach:
  - Empirical: It is evaluated on 32 configurations (4 models x 4 tasks x 2 evaluation sets), with detailed visual analysis across the entire similarity spectrum.
  - Theoretical: A simplified two-layer Transformer model is used to mathematically analyze the parameter dynamics ($\theta_1$ and $\theta_2$) under different strategies, clearly explaining why the Random and Similar strategies fail while the Contrastive strategy succeeds.
  - Diagnostic: The three designed probes (IWL-score, ICL-score, Copy-score) serve as a novel diagnostic tool, successfully linking the behavior of black-box LLMs to the mechanisms of the theoretical model and proving that the same trade-offs and failure modes exist in large-scale models.

### Weaknesses
1. All empirical evaluations are concentrated on sequence-to-sequence Machine Translation (MT) tasks, lacking results on other task types (e.g., classification, reasoning, code generation).
2. Creating "highly-similar" synthetic examples relies on an external model (e.g., gemini-2.0-flash-lite) to generate high-quality paraphrases . This adds complexity and dependency to the training pipeline, and the quality of this external model becomes a confounding variable that could impact the method's effectiveness.

### Questions
1. How dependent is the effectiveness of the Contrastive-Context strategy on the capabilities of the external paraphrasing model (e.g., gemini-2.0-flash-lite)? If a weaker model is used, which generates lower-quality paraphrases (in terms of semantic fidelity or diversity), will the strategy's performance degrade significantly, or even fail completely?
2. How can we determine if the role of the external model (Gemini) is merely to provide the "highly-similar" samples needed for "contrast," or if it is simply providing "better data" (i.e., a form of high-quality data augmentation)?

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
This paper investigates how the similarity structure between target inputs and in-context examples affects the emergence and stability of in-context learning (ICL) and in-weights learning (IWL) in large language models (LLMs). While pre-trained LLMs can perform both ICL and IWL, standard fine-tuning often erodes ICL capabilities. The authors identify inter-example similarity as an overlooked but crucial factor in IC-Train. They introduce Contrastive-Context, a fine-tuning strategy that mixes similar and random examples within contexts. This contrast enforces models to use context information only when it is relevant and rely on in-weights knowledge otherwise. Experiments on machine translation demonstrate that Contrastive-Context consistently outperforms random and nearest-neighbor sampling.

### Strengths
1.	The proposed Contrastive-Context approach effectively improves the model’s ICL performance across varying levels of example similarity.
2.	The method is conceptually straightforward and easy to implement.
3.	The authors evaluate multiple base models on machine translation tasks, demonstrating the approach’s robustness and generalization across different model architectures.

### Weaknesses
1.	The proposed method can be viewed as a hybrid of Random-Context and Similar-Context strategies, which limits its novelty.
2.	Experiments are restricted to translation tasks, making it difficult to convincingly establish the generality of the approach across diverse task types.
3.	The presentation of the paper could be improved — for example, the structure of the introduction section could be more organized, and there are several minor typographical errors (e.g., “IC-Trainwith” in Section 3.1).

### Questions
1.	How does Contrastive-Context perform on a broader range of tasks beyond machine translation?
2.	How does it compare with a baseline that directly mixes Random-Context data and Similar-Context data during training, rather than creating contrast within examples in one context? 
3.	It would be helpful to include ablation experiments that remove the Highly-Similar component (using only Most-Similar examples) to verify the necessity of employing multiple similarity levels during training.

### Soundness
2

### Presentation
2

### Contribution
2
