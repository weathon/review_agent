# Procedural Pretraining: Warming Up Language Models with Abstract Data

- Decision: Reject
- Scores: 4, 8, 6, 6

## Abstract
Pretraining on rich web-scale corpora is the de facto paradigm for building language models.
We study an alternative setting where the model is initially exposed to abstract structured data,
as a means to ease the subsequent acquisition of semantic knowledge,
much like mastering logic and mathematics for humans can support higher reasoning.
We specifically focus on *procedural data* generated
by formal languages and other simple algorithms.

**Method and findings.**
We first use small models to identify algorithmic skills that different forms of procedural data can improve, often significantly.
For example, on a diagnostic task for context recall (Needle-in-a-haystack), the accuracy jumps from 10 to 98% when pretraining on Dyck sequences (balanced brackets).

Second, we study how these gains transfer from abstract to semantic domains in larger models.
We find that procedural pretraining significantly improves performance on natural language, code, and informal mathematics
(C4, CodeParrot, and DeepMind-Math datasets), using as little as 0.1% extra procedural data.
Notably, procedural pretraining also enables models to reach the same loss value with only 55, 67, 86% of the original data of these datasets.

Third, we explore the mechanisms behind these effects.
We find that procedural pretraining instils non-trivial structure in both attention and MLP layers, and that the former is particularly important for code datasets, the latter for language.
We also lay a path for combining the benefits of different forms of procedural data.

**Implications.** 
Procedural pretraining is a remarkably simple means of improving performance and speeding up training for transformers.
It ultimately suggests the possibility of disentangling the acquisition of knowledge from reasoning in LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper explores procedural pretraining, where models first learn from structured algorithmic data before natural language exposure. Such pretraining dramatically boosts reasoning and recall skills and transfers effectively to language, code, and math tasks with quite a few data addition. Extensive experiments reveals several insights such as different relationship between structure data types and downstream task performance, and the relationship between reasoning skills and components (MLP / Attention blocks).

### Strengths
- This study provides a more comprehensive analysis than existing research on the impact of pre-training with artificially generated structure data on downstream tasks and continuous learning in natural language. Furthermore, the analysis of substituting data rather than replacing it offers novelty.
- The authors conducted many variety of experiments using multiple datasets and benchmarks.
- The paper is well-structured and easy to follow.

### Weaknesses
- Choice of downstream task: The paper needs explanation on what basis the author chose downstream tasks.
- L196: "each type of procedural data improves different skills": I think that this is not a surprising result. What is the behind mechanism or possible reason for that? For example, K-DYCK-> HAYSTACK, ECA RULE 110 -> REVERSED ADDITION: Similar skills are generalized to downstream performance?
- Figure2, Figure3, Figure4: In some cases, pre-training on structure data causes performance degradation, but is there any discussion on why it happens and how to avoid it practically? We cannot practically re-run pre-training due to the limitation of computational resources.
- There is no discussion / experiments about practical implication for LLM training. The current training paradigm is pre-training then fine-tuning (instruction-tuning). Based on this, one of the practical implication might be three stage training, i.e., "pre-training on structured data -> pre-trainining on semantic data -> fine-tuning". Accordingly, I think Chapter4 should extend the experiments to validate this. Any comment?

### Questions
- What is the clear definition of attention-only transfer and MLP-only transfer? For example, Attention-only transfer refers to updating the parameters of attention blocks while freezing the other parts parameters?
- Figure7: What is the definition of "Mixture diversity (entropy)"? Please explain the detail in the paper.
- L436: "We propose to compose a new model by assembling components from several pretrained models" -> How to assemble components? Please explain the detail in the paper.

### Soundness
3

### Presentation
3

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
This paper proposes a simple but powerful idea which is pretraining language models on procedural data like formal languages and simple algorithms before exposing them to real text. The authors show that such pretraining instills fundamental reasoning skills, like memory recall and symbolic manipulation, that later transfer to natural language, code, and math domains. Remarkably, even tiny amounts of procedural data improve downstream performance and cut standard pretraining data. They further show that different model components specialize - attention layers capture structured reasoning while MLPs aid linguistic understanding. Overall, the work reframes pretraining as a two-phase process: first learning to reason, then learning to know.

### Strengths
The paper is original in reframing pretraining itself as a two-stage process that separates the acquisition of reasoning from semantic knowledge. The idea of using procedurally generated data rather than linguistic or synthetic text as a lightweight reasoning curriculum is both conceptually elegant and practically effective. The empirical evidence is strong and comprehensive, covering both diagnostic algorithmic tasks and real-world domains like code, math, and language, with clear methodology and ablation studies that isolate where the benefits arise.

### Weaknesses
While the results are compelling, the experiments are limited to relatively small GPT-2–scale models, leaving open how well these effects hold at modern pretraining scales. The analysis focuses on loss and perplexity rather than deeper reasoning or interpretability metrics, so it’s unclear whether the gains reflect genuine reasoning improvements or better optimization. The mixture and weight-composition results, though promising, remain proof-of-concept and could benefit from a more systematic exploration of how different procedural domains interact.

### Questions
Could the authors provide evidence that the observed improvements correspond to genuine reasoning behavior rather than implicit memorization or better optimization dynamics? Some qualitative or mechanistic analysis (e.g., circuit tracing or CoT evaluation) would help clarify this.

Have the authors explored scaling trends, does procedural pretraining continue to offer multiplicative benefits as model and dataset size increase, or does it saturate?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The authors investigate the use of procedurally generated data (Stack, Dyck, Set, etc) to see whether it can be useful as pre-pre-training data or complementary data for pre-training. They demonstrate that different datasets offer different downstream algorithmic capabilities (both strengths and weaknesses). In some sense, the paper lacks novelty in the claims because these tasks of Set, Union, Dyck, etc have been studied by Wu et al. [1] and Hu et al. [2] and have been already shown to be helpful for language pre-training. But, overall, the paper offers a more comprehensive analysis into what are being transferred (different weights of the model), mixing different data. 


[1] Wu, Yuhuai, et al. "Lime: Learning inductive bias for primitives of mathematical reasoning." International Conference on Machine Learning. PMLR, 2021.
[2] Hu, Michael Y., et al. "Between circuits and chomsky: Pre-pretraining on formal languages imparts linguistic biases." arXiv preprint arXiv:2502.19249 (2025).

### Strengths
- Studies the different procedural data generation methods that previous works have proposed and their own (Stack) comprehensively in terms of the downstream algorithmic capabilities and shows that for example Dyck significantly improves long-context modeling, whereas Set is good for sorting.
- The analyses on what weights are important to transfer and mixing the different procedural data offer some good insights as to how we are getting better transfer to downstream tasks.

### Weaknesses
- The claims of how these procedural data improve language domain pre-training including math and code have been proposed by previous works cited in the summary, and I believe the distinction between the claims demonstrated in this paper vs. the previous ones are blurred. It should be clarified what the claims of this paper are, and I still believe the authors have conducted a more in-depth, intriguing study of synthetic data for pre-pre-training.
- In Figure 5, the authors only study Union, Sort, and Set, but what happens to the other procedural data here? Why have they been omitted?

### Questions
- In Figure 5, curious about the other procedural data and also what happens if we increase the amount of language tokens?
- In Figure 6, it seems that transferring only the attention weights are better than the full model transfer. Is that configuration what is used for figure 5 also?
- Do the authors have an intuition / hypothesis as to why the mixtures all seem to be worse than the Union baseline, whereas that is not the case for code? Perhaps b/c code is more structured?

### Soundness
4

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper explores the benefits of pretraining Transformer architectures on procedural data as a first step of training, before training with unstructured, semantic data (such as common pretraining corpora). The procedural data used for pretraining is algorithmically generated (i.e., sequence transformations, memory operations, formal languages, and cellular automata). The models are then further trained on task-specific data, for memory recall, arithmetic, and logical / relational processing tasks. The main findings are that training on such procedural data improves model performance on algorithmic tasks and that procedural pretraining encodes such knowledge in specific components of the Transformer model (attention layers). The authors then explore how training on standard data after procedural pretraining affects model perplexities (using WikiText and JavaCorpus as well as larger corpora such as C4). Finally, the paper experiments with combining multiple types of procedural pretraining data and model merging after pretraining on single procedural data types.

### Strengths
The paper provides a well-structured and technically sound analysis into the benefits of utilizing procedural knowledge for pretraining. The paper’s experiments are overall extensive and detailed, and the results provide clear benefits to pretraining on procedural data before pretraining on semantic data. I would like to highlight that I appreciate the authors’ limitations provided at the end of the main manuscript.

### Weaknesses
As mentioned by the authors, the paper is limited to exploring model performance at the pretraining stage (measured via perplexities) and does not further focus on downstream task performance after fine-tuning. It would be highly interesting to see how differently trained models behave and perform when properly fine-tuned, to get a better understanding of how such pretrained models generate natural language.

### Questions
Did you conduct any manual qualitative inspection of the outputs generated by differently pretrained models? Did the procedural pretraining in any way impact the model’s decoding capabilities w.r.t. lexicality and syntax?

### Soundness
3

### Presentation
3

### Contribution
3
