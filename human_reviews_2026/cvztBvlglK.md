# Pre-training Limited Memory Language Models with Internal and External Knowledge

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 8, 4, 6, 8

## Abstract
Neural language models are black-boxes--both linguistic patterns and factual knowledge are distributed across billions of opaque parameters. This entangled encoding makes it difficult to reliably inspect, verify, or update specific facts. We introduce Limited Memory Language Models (LMLM), a new class of language models that externalizes factual knowledge to external database during pre-training rather than memorizing them. Our pre-training approach strategically masks externally retrieved factual values from the training loss, thereby teaching the model to perform targeted lookups rather than relying on memorization in model weights. Our experiments demonstrate that LMLMs achieve competitive performance compared to significantly larger LLMs on standard benchmarks, while offering the advantages of explicit, editable, and verifiable knowledge bases.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces a method for training language models that query an external database for generating factual information, instead of relying on internal parametric knowledge. The method involves annotating the pre-training data to identify factoids; constructing a knowledge base; and then training LMs to query the database to complete factoids. The method is evaluated in terms of perplexity; factuality (e.g. FActScore); and on applications like machine forgetting.

### Strengths
- In my opinion, this paper proposes a good idea for addressing an important problem. Externalizing factual information could make LMs more interpretable and controllable, which could make LMs more useful in domains where these attributes are important.

- The idea is conceptually simple, but not simple to execute. This paper does an impressive job implementing all of the different components (pre-processing, generating the database, search augmentation, etc.).

- The empirical evaluations are fairly thorough. The experiments show that perplexity is improved (relative to larger models), knowledge can be deleted, language understanding performance is still good. The paper also shows that there are improvements to downstream factuality metrics, like FactScore, and reports some interesting analysis into how much information is still retained, and the optimal "offloading ratio".

- The results are generally strong, with LMLM achieving better perplexity, higher FactScore, decent NLU, and high forgetting quality without loss of utility.

- The writing and presentation are generally very clear.

### Weaknesses
- The paper suggests that this method can make it easier to update facts, but there do not seem to be any experiments on knowledge editing benchmarks (e.g. MQuake [1]), except for the unlearning experiment.

- Some design choices are not thoroughly validated. For example, in section 3.1, it is not clear whether it is necessary to do the intermediate filtering step (which involves fine-tuning a corrector model) in the entity annotation phase. There is also no evaluation of the annotator performance. However, given the many different components of this pipeline, I think it is reasonable to omit some of these ablations, given the good downstream results.

- The models are trained only on Wikipedia. There might be considerable challenges to extending this approach to other domains, e.g. books, or scientific domains. The authors acknowledge this limitation, but it would still be interesting to see some experiments on other domains.

- The experiments are limited to very small models. This could be a reasonable limitation, given limited computation budget, but it makes it more difficult to interpret the results. For example, the NLU metrics (table 12) are all very low, so it is hard to really appreciate how well these models preserve NLU abilities.

- Minor comment: The paper shows evidence that this method reduces factual memorization, but some factual knowledge will always be needed to issue good queries. For example, in the example in the appendix, the model needs to know that Ko Itakura is an athlete to look up "Position" and "Team". This is a minor point, but in my opinion it is worth mentioning, as it makes it could qualify the claims about knowledge editing and unlearning.

- Overall, I think most of these weaknesses are acceptible limitations, given the scope of this work, but I would have appreciated a more extensive discussion of the limitations--especially more concrete discussion of the limitations of applying this approach to non-Wikipedia domains.  



[1] MQuAKE: Assessing Knowledge Editing in Language Models via Multi-Hop Questions. Zhong et al., 2023.

### Questions
- In section 4.1, does the database include triplets from the held-out set? How many entities in the held-out set are not in the database?

- Can you clarify the setup of the LMLM model for the TOFU experiments (sec. 4.3)? It seems that the LMLM model in this section is a version of Llama-3.2-1B-Instruct that has been fine-tuned on additional data to learn the lookup mechanism. Is there any reason to use this approach, rather than evaluating one of the models that was fine-tuned from scratch in the previous section? Alternatively, could you report NLU results for this fine-tuned model?

Minor comments:

- In Figure 5, I think it would be clearer to label the x-axis "Model configuration" or "Model architecture".

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a parametric & non-parametric hybrid of language modeling based on the construction and utilization of a factual knowledge database from the pre-training stage. The proposed method, LMLM, differs from standard retrieval methods like RAG in that it aims to dissect factual knowledge from other general language competencies. Experiments on Wikipedia pretraining data show that medium-sized LMLMs, compared to standard language models, show advantages on tasks that require factual knowledge memorization and knowledge unlearning.

### Strengths
S1: A lot of experimental analyses are well executed and presented, supporting the core advantages of the proposed method well

S2: The limitations of the proposed method and future research directions are well stated in the discussion section

### Weaknesses
W1: **Limited scope of usage** - While the proposed method can be utilized in knowledge-intensive tasks, whether it can be extended to broader usage is unclear. This is because further tuning of LMLMs will likely require a similar formatting of fine-tuning data by design. For example, I'm not clear about whether the proposed method can be deployed and maintained under instruction tuning.

W2: **Brittleness of DB-style modeling of factual knowledge** - While the proposed method inherits many nice properties of symbolic modeling of factual knowledge (such as precision and interpretability), it also inherits their drawbacks. For example, as the authors mentioned, the modeling of factual knowledge as a triplet of (entity, relation, entity) does not encompass all kinds of factual knowledge. For example, the proposed method cannot cover factual knowledge like "Michael Jordan is tall" or "John gave a watch to Mary as a present". Moreover, the behavior of the model on the specific set of factual knowledge that has failed to be incorporated into the database can be more brittle compared to standard LMs, which is not thoroughly investigated in the experiments. Specifically, I believe pre-training on the Wikipedia dataset that is dominated by 'well-shaped factual knowledge' might provide an unfair advantage to LMLMs, which can behave much more brittle in a realistic corpus.

W3: **Lacking comparison of train/inference computational cost** - As the proposed method necessitates the enumeration over the whole pretraining data for database construction, the overall training compute required is expected to be much larger than standard LM pretraining. Moreover, as it requires access to the database, inference cost will be increased, too. However, no quantitative information about the additionally required computing is provided in the main text. It would be great to see this information (for example, in wall-clock time or FLOPs) to investigate the trade-off between its advantage and computational burden.

### Questions
Q1: I couldn't find the comparison of the unlearning performance of LMLM and RAG-based models. I believe this comparison is crucial because it is fairer to compare the unlearning performance of two methods that have direct access to a pre-stored database.

Q2: How did you handle the 'failure cases', for example, when the model generated a syntactically infeasible database query or a query that the database cannot answer, and how often did it happen? Or, did you preclude such possibilities by employing a constrained decoding methods?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors propose Limited/Large Memory Language Models (LMLM), a pre-training framework in which specific factual knowledge is stored in an external database rather than in model parameters. During training, a lightweight annotator marks entity–relation–value triples in the corpus and inserts lookup calls into the text. The retrieved factual values are masked out of the loss, prompting the model to learn when to consult the database instead of memorizing facts. At inference, LMLM generates text and performs lookups as needed, retrieving accurate information from the database. The experiments show that LMLM can match or surpass standard models of similar size in perplexity and factual accuracy, achieve results comparable to much larger models on factual benchmarks, and support efficient unlearning by editing the external knowledge base.

### Strengths
* Introducing masking of retrieved factual values during pre-training to encourage reliance on external lookups rather than weight memorization is a compelling idea. It contributes to ongoing discussions around modularizing knowledge in language models.

* The paper presents a complete pipeline, from data annotation to model training and inference, which may facilitate adoption or extension by other practitioners.

### Weaknesses
* The method’s success relies heavily on an annotation pipeline using GPT-4o and a trained annotator model to extract factual triples. The manuscript would benefit from deeper analysis of annotation accuracy, coverage, biases introduced by the seed annotations, and scalability to larger or more diverse corpora.

* The focus on (entity, relation → value) triples means the approach externalizes only certain types of knowledge (birth dates, titles, etc.). More complex or contextual knowledge (e.g., procedural knowledge, narratives, or common sense) remains embedded in the model weights. The paper does not explore how well the approach generalizes beyond these atomic facts.

* Fuzzy matching via sentence embeddings and a similarity threshold is pragmatic but may encounter failures (e.g., entity ambiguity, missing or conflicting entries). Have authors done experiments on this?

*  While the authors note that the computational cost of external lookups is non-negligible, a more thorough discussion of latency, particularly in real-time applications, and possible mitigation strategies would be valuable.

* The paper contextualizes its contribution relative to kNN-LM, RAG, and other prior work. However, additional comparisons to recent approaches that incorporate explicit memory or knowledge bases (e.g., Memory, KBLaM, MLP Memory, RET-LLM) could clarify how LMLM advances the field beyond those efforts.


KBLaM: Knowledge Base augmented Language Mode, Wang et. al. 2024

Memory³: Language Modeling with Explicit Memor, Yang et. al., 2024

MLP Memory: A Retriever-Pretrained Memory for Large Language Models, Wei et. al. 2025

RET-LLM: Towards a General Read-Write Memory for Large Language Models, Modarressi et. al. 2024

MemLLM: Finetuning LLMs to Use An Explicit Read-Write Memory, Modarressi et. al. 2024

### Questions
* Have the authors done any comparison with the same model trained on the same corpus but without the knowledge removal?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes a novel pre-training paradigm, which aims to remove knowledge from the pre-training process and requires most parameters to only learn capabilities.

The paper’s specific approach is to label the knowledge pairs in all pre-training corpora and mask the knowledge, thereby suppressing knowledge memorization. Among this process, the labeling for pre-training is completed by a small model obtained through distillation.

### Strengths
- The 355M small model trained by the authors has achieved better results on metrics such as FactScore, even when compared to similar models augmented with RAG.
- The training paradigm proposed by the authors is highly innovative, radical yet cost-effective. This is because the cost of searching is far lower than that of memorizing knowledge using large amounts of pre-training data.

### Weaknesses
- For a pre-training paradigm, it is clearly unreasonable for the authors to evaluate and compare models of the same scale solely using factuality-related evaluation methods. I believe the authors should at least add evaluations on aspects representing instruction following and reasoning capabilities, though I understand this is difficult for an ultra-small-scale model.
- The authors’ training scale is too small to allow us to determine whether the advantages currently achieved can be overwritten in subsequent training phases.
- The annotation cost for each piece of data is relatively high, and there is a certain error rate. While I am aware that many datasets also use processing methods with similar costs, those methods are only used for tasks like quality scoring and adjusting training volume. The risks associated with the act of directly adjusting masks do not seem to have been fully discussed.

### Questions
Can models trained through this method tackle tasks that require complex background knowledge and theories, such as math problems?

If I train a capability-only model, does this mean I need to incorporate an entire math textbook into the prompt to provide the corresponding knowledge when facing such tasks?

Beyond that, a more thought-provoking point is: Have the authors proven that the model’s memory occupies a large number of parameters, leading to an insufficient proportion of capability-related parameters and thus deficiencies in capabilities? I believe this may not have been proven.

In fact, as open-source models continue to advance, even small models like Qwen3-4B can hold their own in certain capabilities such as math reasoning, while performing much worse in terms of knowledge. Does this imply that as the quality of training data improves, models store knowledge and capabilities with higher compression efficiency? If so, the authors’ work may yield limited results.

### Soundness
2

### Presentation
3

### Contribution
4
