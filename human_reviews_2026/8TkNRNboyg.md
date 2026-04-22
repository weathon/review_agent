# Towards Understanding the Effect of NTP Paradigm in Unstructured Knowledge Editing

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 4, 4

## Abstract
Editing Large language models (LLMs) with real-world, unstructured knowledge is critical for correcting and updating their internal knowledge bases. However, current methods often oversimplify this knowledge, leading to information loss and suboptimal performance.
While existing editing techniques based on the next-token prediction (NTP) paradigm show promise, our investigation reveal a core limitation: context reliance. The edited knowledge heavily rely on the preceding context available during editing, but this context is often absent in practical inference. This gap between editing and inference limits the generalization of acquired knowledge. We validate this issue both theoretically and experimentally, demonstrating that the absence of preceding context prevents model from recalling the edited knowledge, thereby causing a performance drop on editing success rate.
To address this, we propose a simple yet effective COntext-INdependent unstructured knowledge editing framework (COIN), encouraging the model to internalize new knowledge properly, rather than merely memorizing fixed patterns with its preceding context.
Comprehensive evaluations show that COIN significantly reduces the performance drop and outperforms strong baselines by 23.6\% in editing success rate, highlighting the potential of NTP paradigm for robust unstructured knowledge editing.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper identifies a significant and previously underexplored issue in unstructured knowledge editing for Large Language Models (LLMs): context reliance within the Next-Token Prediction (NTP) paradigm. 

The authors observe that knowledge edited into a model using full-text NTP becomes overly dependent on its preceding context, leading to a performance drop when that context is absent during inference. 

To address this, they propose COIN, a simple yet effective training framework that uses context alignment and knowledge consistency losses to encourage context-independent knowledge internalization.

### Strengths
This paper focus on the UNSTRUCTURED KNOWLEDGE EDITING, which is more useful in the real world application.

This paper notice the context will influence the editing performace, and make some analyze in details.

### Weaknesses
* The paper observes that the model's accuracy degrades when relevant knowledge is positioned near the end of the input context, as detailed in Figure 2. It would be valuable to further investigate how performance is affected when the critical context appears at the beginning or in the middle. Moreover, since not all model editing methods require integrating context with the input, could the authors clarify how their approach differs from retrieval-based methods?

* The training procedure illustrated in Figure 6 requires further clarification. What are the specific inputs and outputs for the model during this process? Furthermore, what does the "consistency" objective precisely constrain—is it the output distribution or another aspect of the model's behavior?

* The results and metrics primarily focus on the success of the edits. How well does the method perform in terms of localization—that is, ensuring that unrelated knowledge or model behaviors remain unchanged?

* What is the computational cost of the proposed method? Given that its performance is sometimes comparable to Fine-Tuning (FT), a discussion of their relative efficiency would be insightful.

### Questions
* Although Table 3 is intended to demonstrate that an effective unstructured knowledge editing method must also handle structured knowledge, it only includes comparisons with other structured-based methods. A comparison with leading unstructured methods would make the argument more compelling.

* Beyond the performance metrics, what broader insights does this work provide regarding the interaction between knowledge editing and language modeling?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates a fundamental limitation of the Next-Token Prediction (NTP) paradigm when applied to unstructured knowledge editing in large language models (LLMs). The authors identify and formalize a phenomenon they call context reliance, where edited knowledge becomes overly dependent on its preceding context during editing, leading to retrieval failure when that context is missing at inference. To address this, the paper introduces COIN (COntext-INdependent editing), a framework augmenting standard NTP training. Extensive experiments on AKEW and UnKEBench datasets (with LLaMA3-8B and Qwen2.5-7B) demonstrate that COIN substantially improves editing success (up to 23.6% ROUGE-F1 gain) and reduces performance drop across positional contexts by 45%. Theoretical analysis (Theorem 3.1) and ablation studies support the identified cause of context reliance.

### Strengths
1. The paper identifies a subtle yet impactful issue—context reliance—that had not been rigorously studied before in the model editing literature. This conceptual framing provides a fresh lens for understanding why next-token-based fine-tuning often fails to generalize edited knowledge.

2. The COIN framework extends NTP training with two intuitive regularization terms that have clear theoretical underpinnings: one encourages invariance to context window size, and the other prevents catastrophic forgetting. The use of KL alignment between global and local distributions is simple yet effective, and the analytical formulation of the knowledge consistency constraint demonstrates mathematical rigor.

3. The paper is clearly written and logically structured. And proposed COIN achieves consistent improvements over strong baselines.

### Weaknesses
1. COIN currently uses a fixed-size sliding window for defining “local context,” which the authors themselves note as a limitation. This may underperform for long, discourse-rich texts where context relevance varies.

2. While AKEW and UnKEBench are suitable, they primarily test single-fact retrieval. It remains unclear whether COIN improves reasoning tasks that require integrating multiple edited facts or narrative comprehension. Authors can Include an evaluation on multi-fact narrative editing (e.g., multi-hop MQuAKE reasoning chains or synthetic story edits) to demonstrate generalization to compositional inference.

3. Both alignment and consistency losses add auxiliary computation, yet the paper does not report training time, memory, or scalability.

### Questions
How sensitive is COIN’s performance to the choice of window size k and the trade-off hyperparameters (α, β)? Are these fixed across datasets or tuned per task?

Can COIN be combined with existing locate-then-edit methods (e.g., ROME/AlphaEdit) to yield hybrid improvements?

Have the authors examined whether COIN affects unrelated factual recall accuracy on general QA datasets like Natural Questions or TriviaQA?

### Soundness
3

### Presentation
3

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
The paper studies unstructured knowledge editing by sticking with the most “native” training objective—next-token prediction (NTP) over full text—rather than converting facts into triples or synthetic QA. The authors identify a robust failure mode they call context reliance: when the edited fact appears later in a paragraph, performance drops sharply; and if the query at test time omits the preceding context seen during editing, the model often fails to retrieve the fact. They show this empirically and offer a simple one-layer, one-step GD argument explaining why the learned mapping can hinge on a specific pair of context tokens. Building on this diagnosis, they propose COIN, adding a context-alignment loss to match predictions under full vs sliding-window context, and a knowledge-consistency loss to keep behavior stable on unrelated inputs.

### Strengths
- The paper clearly identifies and substantiates context reliance as the central failure mode of NTP-based unstructured editing—showing that edited knowledge becomes entangled with preceding context and collapses once that context is removed—offering a precise, empirically grounded diagnosis of why full-text fine-tuning often fails to generalize. This finding can contribute to the community.

- COIN’s two regularizers are easy to bolt onto standard NTP editing, and they directly target the identified gap (train with global context vs test with local context). The method section is straightforward.

- The experiments are comprehensive: on AKEW/UnKEBench, COIN achieves significant gains over the strongest baseline in terms of BERT/ROUGE-F1; on MQUAKE (multi-hop), it also substantially outperforms ROME, MEMIT, and AlphaEdit, demonstrating the superiority of the proposed method.

### Weaknesses
- The theoretical analysis is built on an extremely simplified setting—a single-layer Transformer, a single gradient-descent update, and an attention pattern dominated by just two tokens (p and q). While this abstraction is useful for illustrating how context reliance can emerge, it does not capture the dynamics of deeper, multi-head, multi-step training typical of actual LLMs.

- The details of experiment, such as sampling strategy and numerical stability of the covariance-style objective aren’t reported in enough detail.

- The main text largely focuses on the strengths and successful results of COIN. Important limitations—including the bounded theoretical analysis and the impact (if any) on language generation models beyond the tested scale—are acknowledged only in appendices or passing remarks.

- Limited Baseline Diversity: Most empirical comparisons focus on variants of NTP-based editing or classic baselines.

### Questions
1. Can the authors clarify precisely how local context windows $k$ are defined/selected, and how robust the approach is against variable-length or semantically structured local contexts? Would more adaptive windows improve performance or efficiency?

2. Given that the theoretical analysis (Theorem C.1) applies under strong simplifying assumptions, might the context reliance phenomenon be weaker/stronger in multi-layer, multi-step training scenarios? Did you observe any qualitative mismatches between theorem predictions and empirical findings?

3. The paper briefly mentions meta-learning editors such as MEND and memory-based approaches, but it remains unclear how these methods would behave under the same unstructured NTP setting. Have the authors considered, even qualitatively, whether such approaches exhibit similar context-reliance effects?

4. For the knowledge consistency loss, does increasing $|\mathbf{K}_0|$ (number of sampled keys) materially affect model collapse or knowledge retention? Additionally, does this approach generalize to "unstructured" or ambiguous keys, such as those found in natural dialogues?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper examines unstructured knowledge editing in the next-token prediction (NTP) setting and observes that models can become reliant on the preceding context used during editing; consequently, the edited knowledge is harder to retrieve when queried without that context. To address this, the authors propose COIN, which augments the standard editing loss with a context-alignment objective that encourages the model’s predictions under a global window to match those under a local window. The approach is evaluated on several unstructured editing benchmarks.

### Strengths
1. The paper targets a meaningful and timely issue in unstructured knowledge editing under NTP, with clear implications for reliability and deployment.

2. The finding that edited knowledge retrieval depends on the preceding context is interesting and practically important; the proposed mitigation idea is reasonable.

3. The paper is generally well written and easy to follow; figures/tables and section flow make the narrative accessible.

### Weaknesses
1. Some setup details need elaboration. namely,  Section 3.3’s two mitigation strategies are insufficiently specified. It remains ambiguous how they are implemented in practice. Similarly, for splitting/paraphrasing, it is unclear whether the order of knowledge is permuted to control for position-dependent difficulty. Concrete examples (with before/after text) would help disambiguate the procedure.

2. The central “alignment loss” idea, which aims to remove global context impacts, does not appear new in my opinion. It has been explored in the long-context learning/understanding literature [1]. The authors should make it clearer on the connection and uniqueness of the solution proposed in this paper. 

3. The design of consistency loss solution looks suboptimal. As mentioned by the authors, the concept of "unrelated knowledge" $K_0$ as in ROME/MEMIT/AlphaEdit. However, the the paper opts for a generic regularization on $W_0$ rather than a more structure-aware constraint as in AlphaEdit. 

4. As claimed by the authors, the core issue identified in this work is inherent to the NTP training paradigm. In addition, the proposed alignment loss should be tested across diverse NTP-based editors (fine-tuning, LoRA, ROME/MEMIT) to demonstrate generality and systematic gains, rather than a specific method.

5. The problem appears related to known overfitting phenomena in editing, which has been widely studied in the literature [2, 3, 4, 5], but these links were not discussed clearly. 

[1] What Is Wrong with Perplexity for Long-Context Language Modeling? 2024.

[2] Neighboring Perturbations of Knowledge Editing on Large Language Models, 2024.

[3] Uncovering Overfitting in Large Language Model Editing, 2025. 

[4] Revealing and Mitigating Over-Attention in Knowledge Editing, 2025.

[5] Mitigating Heterogeneous Token Overfitting in LLM Knowledge Editing, 2025.

### Questions
Please see my comments in the weakness sections.

### Soundness
2

### Presentation
3

### Contribution
2
