# MLP Memory: A Retriever-Pretrained Memory for Large Language Models

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
Modern approaches to enhancing Large Language Models' factual accuracy and knowledge utilization face a fundamental trade-off: non-parametric retrieval-augmented generation (RAG) provides flexible access to external knowledge but suffers from high inference latency and shallow integration, while parametric fine-tuning methods like LoRA risk catastrophic forgetting and degraded general capabilities. In this work, we propose MLP Memory, a lightweight parametric module that learns to internalize retrieval patterns without explicit document access. By pretraining an MLP to imitate a $k$NN retriever's behavior on the entire pretraining dataset, we create a differentiable memory component that captures the benefits of retrieval-based knowledge access in a fully parametric form. Our architecture integrates this pretrained MLP Memory with Transformer decoders through simple probability interpolation, achieving 12.3\% relative improvement on five question-answering benchmarks and 5.2 points absolute gain across nine general NLP tasks, while reducing hallucinations by up to 10 points on HaluEval. Moreover, MLP Memory delivers 2.5$\times$ faster inference than RAG with superior accuracy. Our findings show that learning retrieval patterns parametrically bridges the gap between efficient inference and effective knowledge access, offering a practical alternative to both RAG and fine-tuning approaches.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces MemoryMLP, a parametric module to mimic the behavior of explicit, external knowledge. In comparison to classical RAG approaches, MemoryMLP offers faster inference due to simple probability interpolation using the trained MLP module. The authors conduct extensive experiments showing that (1) MemoryMLP improves performance over base models and selected baselines (RAG / LoRA), (2) improves efficiency and throughput over these approaches, (3) reduces hallucination rates as measured by HaluEval, (4) ablations on the loss-function trade-off.

### Strengths
- The proposed idea of MemoryMLP is novel, simple and intuitive. It addresses a key limitation of many current RAG framework, overcoming search time and inference overhead. The presentation of the method including the respective image (Figure 4) is precise and easy to follow.
- The experiements show strong improvement over RAG and LoRA approaches, especially for Natural Questions and HotpotQA where the respective baseline do not improve on the base model performance. The choice of experiments is sound and convincing.
- The ablations and experiments on downstream tasks other than QA show the transferability of the proposed approach.

### Weaknesses
- The presentation may be contain flaws: While the overall presentation is good, I have stuggles on understanding several details of the method, either because the information is not where I would expect it to be or because it is simply not mentioned. For instance, some hyperparameters appear to be arbitrarily chosen such as the parameter count of the MLP or why the Wikipedia-2021 dump. Further, the MLP is not further specified, e.g., how many layers. See also the questions below.
- The results may be considered weak: While we observe relative improvements over the selected baselines, e.g., for NQ or HotpotQA are low and there are known to be existing approaches that are better. While state-of-the-art methods are not necessarily comparable, I would like to understand how MemoryMLP performs against other approaches such as question decomposition or fine-tuning a smaller LM. Especially considering the training budget of 5B token. See also question below.
- The evaluation may be considered incomplete: Do you also fine-tune the RAG baseline or do you employ them in a zero-shot setup? As proposed by Lewis et al. (2022), their RAG pipeline is also end-to-end differentiable and it would be interesting to see a comparison between MemoryMLP and end-to-end RAG.

### Questions
- The dataset construction in the main part is quite short - can you please specify how you create the embedding-distribution pairs? Specifically, how do ensure (i) that you generate useful examples such as in Figure 4 that actually depict a fact and (ii) that you have pair following identical format, also shown in Figure 4?
- Do you have any inights on performance differences using different MLP sizes? Why have you initially chosen 1B? How many layers have you chosen for the MLP?
- Do you think the training budget is better spend on optimizing a smaller LM on the target distribution? Why, why not?

### Soundness
3

### Presentation
3

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
This paper introduces a new method called MLP Memory to help large language models be more factually accurate without slowing them down.  The proposed solution involves pre-training a lightweight, all-MLP (Multi-Layer Perceptron) module to imitate the output distribution of a kNN-LM retriever over the entire pre-training corpus. This MLP Memory module learns to map a hidden state from the LLM to a vocabulary distribution that approximates the retriever's output, effectively internalizing retrieval patterns. During inference, the MLP Memory's distribution is interpolated with the base LLM's output, providing a knowledge-augmented prediction. The results show that this method makes models better at answering questions, reduces hallucinations, and is much faster than using a traditional search (RAG).

### Strengths
(1) The motivation of the paper is clear. The paper identifies a clear gap in the literature and proposes an elegant hybrid approach that bridges parametric and non-parametric methods. 

(2) Comprehensive evaluation: The experimental design is thorough, evaluating the method across multiple critical dimensions: factual question answering, general NLP capability preservation, hallucination reduction, and inference efficiency. 

(3) Compelling performance: The results show that MLP Memory not only surpasses the base models but also outperforms RAG on accuracy metrics. This suggests the method successfully captures richer contextual relationships. Results also show that MLP Memory gets better efficiency than existing methods.

### Weaknesses
(1)  The method requires a pre-training phase for the MLP Memory module, which involves constructing a large datastore and optimizing a large model. The computational cost and time of this initial step could be a barrier for some. 

(2)  Unlike RAG, the knowledge stored in the MLP Memory is fixed after pre-training (just like the generative retrieval models). This limits the model's ability to access updated or real-time information without a costly retraining cycle, making it less suitable for domains requiring frequent knowledge updates.

(3) While the paper demonstrates that the method works, a deeper analysis of what specific knowledge patterns the MLP Memory learns and how it interfaces with the transformer's inherent reasoning process would strengthen the theoretical contribution.

(4) Some existing works are trying to solve similar problems. For example, RetroLLM also tries to leverage generative retrieval models to implement end-to-end RAG systems. The authors should discuss and compare with these existing methods. 

[1] RetroLLM: Empowering Large Language Models to Retrieve Fine-grained Evidence within Generation, Li et al..

### Questions
(1) Cost problem: What is the total computational budget (e.g., in GPU hours) required to pre-train the 1B-parameter MLP Memory for a 7B model? How does this cost scale with the size of the base model and the pre-training corpus?

(2) Dynamic knowledge integration: Could the MLP Memory architecture be adapted for incremental learning? For instance, is it feasible to fine-tune the MLP Memory on a small corpus of new documents to update its knowledge without catastrophic interference?

(3) Generalization to specialized domains: The experiments use general corpora like Wikipedia. How would the method perform when applied to highly specialized domains (e.g., legal or biomedical text) where the retrieval might be more complex? For example, in the legal domain, retrieving similar legal cases for generative legal judgment prediction is much difficult than the general RAG systems.

### Soundness
2

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
4

### Summary
This paper introduces MLP Memory, a parametric memory module trained to imitate the behavior of retrieval-augmented models such as kNN-LM. The approach constructs a datastore from a training corpus and trains a small MLP to predict the kNN distribution given hidden representations from a pretrained language model. During inference, the predicted distribution is interpolated with the base model’s output probabilities, removing the need for explicit retrieval or datastore lookup. Experiments on question answering and factual recall tasks show that MLP Memory achieves comparable or better accuracy than kNN-LM and RAG while being more efficient in inference, as measured by time-to-first-token (TTFT) and tokens-per-second (TPS). Ablations explore the layer position for attaching the memory and the balance between cross-entropy and KL objectives.

### Strengths
The paper addresses an important problem in retrieval-augmented modeling: maintaining factual recall benefits without external retrieval or large storage. MLP Memory provides a clean parametric alternative that approximates kNN-LM behavior through distribution imitation. The method is conceptually clear and empirically well evaluated, with strong results on factual QA and hallucination benchmarks. The use of TTFT and TPS as efficiency metrics is appropriate and provides a runtime-based comparison to retrieval baselines. Ablation studies on interpolation weighting, attachment layer depth, and loss balancing help clarify key design choices. Overall, the work presents a technically sound and well-executed exploration of parametric memory.

### Weaknesses
Despite strong results, several aspects limit the clarity and generality of the approach. The training cost is not clearly quantified: even if inference is faster, pretraining the MLP Memory requires generating kNN distributions and performing supervised training on them, which likely increases per-step computation and total cost. It is also unclear whether the backbone model is frozen or jointly updated, which affects the practical difficulty of integration. Moreover, the MLP Memory seems to require task-specific pretraining, since the datastore and supervision are derived from each corpus; this reduces its plug-and-play appeal compared to non-parametric retrieval systems like RAG. The method also depends on tuning the interpolation weight λ for each task, which introduces additional hyperparameter sensitivity. Finally, while the ablation identifies an optimal attachment depth (~70%), it remains uncertain how this generalizes across different architectures or scales. These issues make the approach less universally efficient than implied.

### Questions
- Is MLP Memory pretrained separately for each task or domain, or can a single memory generalize across corpora?
- Are the base language model weights frozen during memory pretraining, or do they also receive updates?
- How large is the total training cost (in GPU-hours or FLOPs) relative to standard fine-tuning or RAG adaptation?
- How sensitive are results to the choice of $\lambda$, and how is it tuned for each task?
- What is the scaling trend with different memory sizes? Does a smaller MLP maintain performance efficiency?
- Could you discuss missing important related literature on (efficient) parametric memory architectures such as Product Key Memory [1, 2] and Entities as Experts [3, 4].

[1] Lample et al., Large Memory Layers with Product Keys

[2] Kim and Jung, Large Product Key Memory for Pretrained Language Models

[3] Fevry et al., Entities as Experts: Sparse Memory Access with Entity Supervision

[4] Verga et al., Facts as Experts: Adaptable and Interpretable Neural Memory over Symbolic Knowledge

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes MLP-memory that replaces the KV store in kNN-LM by an MLP. The goal is to save memory space requirement, which is high in kNN-LM. The problem is interesting. A more efficient implementation of kNN-LM idea can make the approach more feasible in practice.
The method is tested on several tasks of QA and general NLP, showing better performance compared to the baselines. Analyses are provided on the impact of different hyperparameters and the training objective function.

### Strengths
1. The main idea is to make kNN-LM more storage efficient. It trains an MLP to replace the costly K-V storage. The idea is interesting.

2. The paper describes well the approach. It contains good analyses about different hyperparameters and settings.

3. The approach is compared to the relevant baselines (except the missing comparison mentioned in weakness).

### Weaknesses
1. The contribution of the paper is limited. It does not change the basic idea of kNN-LM, but tries to provide a better implementation of it.

2. As the main contribution of MLP-memory is to reduce the storage cost of kNN-LM, the main comparison should be done with kNN-LM. However, in the main results on QA, this comparison is missing. The comparison is only done on general NLP tasks. This is insufficient.

3. The goal of MLP-memory is to reproduce kNN-LM at lost storage cost. The key research question is whether MLP can approximate KV store. This question requires extensive comparison between them. In addition to the end tasks, it is also useful to compare the internal probability distribution of models (e.g. PPL). The paper contains some analysis about the distribution of probabilities of LM, kNN-LM and MLP-memory, but perplexity is only evaluated on MLP-memory. A comparison on perplexity may better reveal whether MLP-memory can approach kNN-LM.

4. The analysis (in appendix) on the different distributions of probabilities in LM, kNN-LM and MLP-memory is interesting, showing that MLP-memory is between LM and kNN-LM. It is however unclear what the implication would be. So the authors suggest that a distribution between LM and kNN-LM sould be better? and why?

### Questions
see weakness.

### Soundness
2

### Presentation
3

### Contribution
2
