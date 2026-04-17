# Scaling Knowledge Editing in LLMs to 100,000 Facts with Neural KV Database

- Decision: Accept (Poster)
- Scores: 4, 2, 8, 6

## Abstract
Efficiently editing knowledge stored in Large Language Models (LLMs) enables model updates without large-scale training. One promising solution is Locate-and-Edit (L\&E), allowing simultaneous modifications of a massive number of factual knowledge. However, such editing may compromise the general abilities of LLMs and even result in forgetting edited facts when scaling up to thousands of edits.  In this paper, we model existing linear L\&E methods as querying a Key-Value (KV) database. From this perspective, we then propose NeuralDB, an editing framework that explicitly represents the edited facts as a neural KV database equipped with a non-linear gated retrieval module. With simple modification over L\&E methods, our framework not only significantly extends the capacity of knowledge editing but also eliminates the associated side effects. Comprehensive experiments involving the editing of 10,000 facts were conducted on the ZsRE and CounterFact datasets, including GPT2-XL, GPT-J (6B) and Llama-3 (8B). The results demonstrate that NeuralDB excels in all metrics of editing success while maintaining original performance evaluated by six representative text understanding and generation tasks. Further experiments indicate that NeuralDB maintains its effectiveness even when scaled to 100,000 facts (\textbf{50}$\mathbf{\times}$ more than in prior work).

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper analyzes existing Locate-and-Edit methods and finds that they return the residual vector corresponding to the edited fact, while returning a zero vector for unrelated questions. Based on this observation, the paper proposes **NeuralDB**, which explicitly stores keys and residuals as a KV database. During testing, the most relevant residuals are matched through a **non-linear gated function** and then injected into the hidden state stream.

### Strengths
* Viewing the key and residual flows as a KV database is a novel perspective.

* NeuralDB is simple yet highly effective, capable of scaling to a large number of edits while preserving the model’s general capabilities.
* The paper validates the effectiveness of NeuralDB through extensive experiments across multiple models.

### Weaknesses
* The paper lacks a comparison with **MEMOIR [1]**, which identifies relevant edits by comparing the sparse activation patterns of new queries with those stored during editing. Both MEMOIR and NeuralDB share a similar **store–identify–inject** paradigm.
* Although NeuralDB demonstrates outstanding performance on the **CounterFact** and **ZsRE** datasets, its effectiveness on **multi-hop reasoning editing tasks** remains unknown, such as **RippleEdit [2]** and **MQuAKE [3]**. If NeuralDB also performs well on these datasets, it would further highlight its applicability.
* NeuralDB requires an additional **KV database**, which increases deployment complexity.

[1] MEMOIR: Lifelong Model Editing with Minimal Overwrite and Informed Retention for LLMs

[2] Evaluating the Ripple Effects of Knowledge Editing in Language Models

[3] MQuAKE: Assessing Knowledge Editing in Language Models via Multi-Hop Questions

### Questions
The L&E method requires incorporating the position of the subject token when computing $k$ and $v$. I’m curious whether the $k$ used here is computed in the same way as in L&E. If it is, how is the position of the subject token obtained during testing? If it is not, what exactly is the computation method for $ k $ in this case? Moreover, why is it that even with a different computation method, key matching can still be achieved (i.e., if the last token’s hidden state is used as the key, it should differ from the key at the subject token’s position, and intuitively, the matching success rate should be very low).

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents NeuralDB, a scalable knowledge editing framework that can efficiently and robustly integrate up to 100,000 factual edits. NeuralDB reconceptualizes the Locate and Edit (L&E) paradigm as a process of querying a neural key-value (KV) database and introduces a nonlinear gated retrieval module. The proposed approach aims to mitigate catastrophic forgetting of general capabilities and maintain LLM consistency even after large-scale edits. Experiments conducted on three LLM architectures including GPT-2 XL, GPT-J (6B), and Llama-3 Instruct (8B), as well as benchmarks such as ZsRE and CounterFact, demonstrate that as the number of edited facts increases from thousands to 100,000, NeuralDB maintains high editing success rates while preserving performance on general language understanding tasks.

### Strengths
1. NeuralDB successfully scales knowledge editing to 100,000 facts, which is an order of magnitude higher than previous methods such as AlphaEdit. As the number of edits increases, NeuralDB maintains both average editing effectiveness and general capability, forming a sharp contrast with the performance degradation observed in baseline methods.
2. The paper provides a coherent reinterpretation of previous Locate and Edit (L&E) approaches as linear key-value (KV) databases, and supports this perspective with mathematical derivations.
3. The NeuralDB approach is simple yet effective, addressing the inherent linearity limitations of existing techniques and enabling robust scalability.

### Weaknesses
1. The details before Section 4 are primarily about other work and could be condensed.  
2. In the methodology part of Section 4, the approach essentially constructs a plug-and-play knowledge base and uses a gating mechanism to determine when to perform edits. What is the key difference between this idea and "Improving Sequential Model Editing with Fact Retrieval"?  
3. In the experimental section, the overall results appear impressive. However, if 10,000 data entries are updated in a plug-and-play manner, the updates primarily rely on the advantages of MEMIT or AlphaEdit. It is recommended to validate the approach on more complex datasets, such as MQuAke, which involves multi-hop editing tasks.  
4. The paper mentions T-Patch, but there is no experimental comparison with it.  
5. The most critical issue is whether this method continuously modifies the model parameters. For instance, after updating the Nth data entry, do the updates to the 1st to (N-1)th data entries remain effective?

### Questions
The comparison in Table 1 is limited to methods that continuously update parameters, which creates an unfair comparison.

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
4

### Summary
This paper introduces the Neural KV Database (NeuralDB) framework to significantly scale up knowledge editing in Large Language Models (LLMs). The authors re-frame existing Locate-and-Edit (L&E) methods as querying a Key-Value (KV) database, which allows for simultaneous modifications of a massive number of factual knowledge edits. The primary motivation is to overcome the limitations of current L&E methods, which often lead to compromised general abilities and forgetting of previously edited facts when scaling beyond thousands of edits. The proposed NeuralDB claims to scale knowledge editing up to 100,000 facts while maintaining high efficacy and model integrity.

### Strengths
1. The paper successfully tackles a critical bottleneck in knowledge editing: scalability. Claiming and demonstrating results up to 10,000 edits (with the promise of 100,000 in the full paper) represents a significant leap forward in the field, far surpassing the typical limits of previous L&E techniques

2. The method directly confronts the major issue of maintaining LLM General Abilities while performing massive knowledge updates, which is vital for practical deployment.

### Weaknesses
1. Although the title claims scalability to 100,000 facts, the quantitative experiments only cover up to 10,000 edits, leaving the paper’s core contribution without direct empirical support.

2. Can edited knowledge support reasoning? I suggest adding a Portability metric.

3. How does the framework ensure that the embeddings for 10,000 or 100,000 distinct keys remain sufficiently orthogonal during the retrieval and editing process to prevent mutual interference? If two factual contexts (Keys) are semantically similar, the retrieval mechanism risks activating and modifying non-target KV pairs, leading to subtle, "soft" side effects beyond the targeted edit. The current paper lacks a thorough analysis of this embedding space's decoupling capacity at scale.

### Questions
see weaknesses

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper tackles a key limitation of existing Locate-and-Edit (L&E) knowledge editing (KE) methods: their failure to scale to thousands of facts without causing catastrophic forgetting and degrading the model's general abilities. NeuralDB2 scales knowledge editing to 100 k facts by replacing implicit linear L&E updates with an explicit gated neural KV memory, curbing catastrophic forgetting and preserving model competence. Finally the paper demonstrates impressive experimental results, showing NeuralDB scales to 10,000 and even 100,000 facts while maintaining high editing success and preserving general model performance.

### Strengths
1. **Fresh Perspective with Solid Grounding**: The paper cleverly rethinks existing linear editing methods like MEMIT and AlphaEdit as lookups in a hidden key-value store—backed up by both theory and experiments that show the updates behave almost like one-hot vectors.

2. **Smart, Simple Fix**: Once they spot that sparsity, they swap out the cramped linear system for an explicit neural KV database. A lightweight, cosine-based gate fetches the right “patch” or returns zero if nothing matches—clean and intuitive.

3. **Scales Like a Champ**: NeuralDB keeps its accuracy even after 10 k edits and still protects the model’s general skills, something earlier SOTA methods couldn’t do; they even push the demo to 100 k facts without breaking a sweat.

### Weaknesses
**1. On the Sensitivity and Selection of the Gating Hyperparameter $\gamma$**

My primary concern is the selection of the gating threshold $\gamma$, which is the single most critical hyperparameter for preserving general abilities. In the main experiments, the authors use a single value ($\gamma=0.65$) across three different models (GPT2-XL, GPT-J, and Llama-3). This appears to be an unsubstantiated, convenient assumption. This concern is amplified by the ablation study in Appendix I.2 (Table 11), which was *only* conducted on Llama-3. This study reveals that the model's performance is *extremely sensitive* to this parameter. For instance:
Increasing $\gamma$ from *0.65* to *0.75* causes the Generalization (G) score to drop sharply from 85.9 to 74.1. At $\gamma=0.9$, the Generalization ability collapses almost entirely to 28.7.

Given this high sensitivity, how can the authors justify applying $\gamma=0.65$ to the other models without a similar ablation? It is highly probable that the optimal (or even safe) $\gamma$ is model-dependent. This sensitivity implies a very narrow "safety boundary." How can we be confident that unrelated knowledge queries ($k_{old}$) won't accidentally cross this sensitive threshold as the database scales, especially if the chosen $\gamma$ is not truly optimal for that specific model architecture?

**2. On the True Cost of "Scalability" (Memory and Compute)**

The paper's claims of scalability and "controllable" overhead seem to obscure a significant trade-off in memory and compute.

**Memory:** The authors state in Appendix G that editing 10,000 facts for Llama-3-8B adds 150M parameters.This implies that the paper's headline 100,000-fact model carries approximately **1.5B** in additional parameters. For an 8B model, this is an extra 19% in size. This is a very significant cost and can hardly be described as "controllable." This trade-off should be made explicit in the main paper.
**Computation:** More critically, the gated retrieval (Eq. 11)  requires calculating the cosine similarity of the current key $k^l$ against *all $m$ keys* in $K_1$, an $O(m)$ operation for every forward pass. The claim of only a 1.5% evaluation time increase for $m=10,000$ is surprising. Does this latency scale linearly with $m$? If so, the $m=100,000$ model would be substantially slower. 

**3. On the Unverified Claims of "Modify" and "Delete" Operations**

The authors repeatedly claim that NeuralDB is "easy to manage for supporting operations such as appending, modifying, and deleting". While appending is well-demonstrated, the "modifying" and "deleting" operations are completely unverified by experiments.

This is not a trivial claim. For example:

**Delete:** What happens when a $(k_i, r_i)$ pair is deleted from the database? Does the model revert to its original, pre-trained knowledge for that query? Or does it become confused and generate an error?
**Modify:** How is "modification" defined? Is it simply updating an existing residual $r_i$ for a key $k_i$? Or is it adding a new, conflicting fact about the same subject (which raises the question of key collision, a point not fully addressed in the paper)?

To validate these practical claims, could the authors provide a concrete case study? For instance: (1) Edit a fact. (2) Test that the edit is successful. (3) Delete the fact's $(k_i, r_i)$ pair from the NeuralDB. (4) Re-test the *same* prompt and report the model's output. Does it successfully revert to the original, pre-trained knowledge?

### Questions
See weaknesses

### Soundness
4

### Presentation
3

### Contribution
3
