# Sculpting Subspaces: Constrained Full Fine-Tuning in LLMs for Continual Learning

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 8, 6, 4, 2

## Abstract
Continual learning in large language models (LLMs) is prone to catastrophic forgetting, where adapting to new tasks significantly degrades performance on previously learned ones. Existing parameter-efficient methods often limit model expressivity or introduce new parameters per task, creating scalability issues. To address these limitations, we introduce **Orthogonal Subspace Fine-Tuning (OSFT)**, a novel parameter-efficient approach for continual learning. OSFT leverages adaptive singular value decomposition (SVD) to dynamically identify and preserve critical, high-rank parameter subspaces that encode prior knowledge. All updates for new tasks are constrained to be strictly orthogonal to these preserved subspaces, which minimizes interference while maintaining a fixed parameter count and avoiding the need to store task-specific gradients. We extensively evaluate OSFT on standard continual learning benchmarks using both encoder-decoder (T5-Large) and decoder-only (LLaMA-2 7B, Mistral-7B) models across diverse tasks. Empirically, our method achieves a state-of-the-art trade-off between learnability and knowledge retention, dominating the Pareto frontier, with **up to 7\% higher** average accuracy than recent baselines like O-LoRA, and **reduces forgetting to near-negligible levels**. It notably maintains the model's general linguistic capabilities, instruction-following, and safety throughout the learning process. OSFT provides a practical, theoretically grounded, and scalable solution that effectively balances model plasticity and knowledge retention for continual learning in LLMs. Code is available at https://github.com/Red-Hat-AI-Innovation-Team/mini_trainer.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes OSFT (Orthogonal Subspace Fine-Tuning), a continual learning method for LLMs that (i) performs per-layer SVD of weight matrices, (ii) treats the top singular directions as a high-rank subspace presumed to carry previously acquired knowledge, and freezes them, and (iii) constrains all parameter updates for new tasks to the orthogonal complement (low-rank subspace). To avoid a one-size-fits-all rank, the method uses an input–output cosine similarity proxy to adaptively determine, per layer, how many top singular directions to preserve. Implementation can be realized either by explicit gradient projection or by a reparameterization that freezes the high-rank factors and optimizes only the low-rank factors. The authors provide a second-order analysis suggesting a tighter bound on worst-case forgetting than both unconstrained full fine-tuning and fixed-rank projection, and report empirical improvements over strong orthogonal/low-rank baselines across multiple sequential learning benchmarks.

### Strengths
- Conceptually transparent and practically implementable “stability–plasticity” mechanism.
The paper operationalizes “protecting prior knowledge” as a geometric constraint on update directions: disallow updates along top singular modes and permit updates only in their orthogonal complement. This aligns with the intuition that model behavior is most sensitive along dominant spectral directions while leaving substantial slack in the complementary subspace for learning. Importantly, the mechanism does not require rehearsal buffers or per-task parameter growth. Both explicit gradient projection and a reparameterized module (freezing high-rank, training low-rank) are described, making integration into standard PyTorch/PEFT pipelines straightforward.

- Layer-wise adaptive capacity allocation that mitigates the brittleness of fixed-rank approaches.
By estimating a layer’s “knowledge-carrying importance” via input–output similarity and then allocating more preserved high-rank directions to important layers, the method avoids uniform rank settings that can either under-protect critical layers or overconstrain unimportant ones. This per-layer adaptivity is well-motivated for Transformers—whose layers differ in function—and empirically reduces sensitivity to rank choice while improving overall stability–plasticity trade-offs.

- A coherent theory–practice loop.
The paper offers a second-order analysis yielding a hierarchy of forgetting bounds (adaptive projection < fixed-rank projection < unconstrained full fine-tuning) and leverages the widely observed empirical link between large singular directions and high-curvature/importance directions as a computationally feasible surrogate for direct curvature estimation. The result is a self-consistent narrative—curvature → dominant spectral directions → orthogonal update constraint—that ties mathematical justification to implementable proxies.

- Reasonably broad empirical coverage with diagnostic ablations.
Evaluations span both encoder–decoder and decoder-only architectures, short and long task sequences, and more comprehensive sequential scenarios, with comparisons to state-of-the-art orthogonal/low-rank baselines (e.g., O-LoRA–style methods). Ablations (e.g., removing orthogonal projection; reducing preserved rank) produce large and consistent degradations, highlighting that both the orthogonality constraint and the adaptive preservation rule are functionally essential. The fact that model size remains constant across tasks is also practically appealing for deployment.

### Weaknesses
- Computational budget and fairness (stronger budget-controlled comparisons are needed).
Although OSFT maintains constant model size across tasks, its per-task training still operates in the full parameter space under constraints, which can entail higher effective degrees of freedom and greater compute/time than small-parameter PEFT schemes (e.g., LoRA/adapters). The paper lacks strictly budget-matched comparisons (e.g., equal GPU-hours, equal training steps, equal hyperparameter search budgets) and energy/latency reporting. Without such controls, reported gains may partially reflect additional compute rather than methodological superiority. A detailed performance–cost table/curve would materially strengthen the claims.

- Breadth of general capability and safety evaluations (evidence remains narrow).
The paper asserts that general abilities and alignment/safety are preserved after continual updates, yet the evaluation scope is limited relative to the breadth of LLM competencies (mathematical reasoning, code generation, tool use, long-context robustness, multilingual performance, jailbreak resistance, hallucination propensity, etc.). Stronger conclusions would require wider, multi-dimensional assessments with multi-seed mean±std and budget-controlled settings to rule out noise and tuning bias.

- Limited coverage of recent open-source model families (e.g., Qwen3).
Experiments primarily focus on T5-Large and LLaMA-2/7B-scale models. Given the rapid adoption of newer families—Qwen3, Llama-3, etc.—the method’s external validity would be clearer with systematic evaluations on these up-to-date backbones, ideally alongside the model-specific best practices for PEFT/CL provided by those communities.

### Questions
- Stability of general capabilities: how can the evaluation scope and statistical rigor be strengthened?
Could the authors expand to mathematics (e.g., GSM8K/MathBench), code (HumanEval/MBPP/Codeforces-style reasoning) and tool use (ReAct/ToolBench), and report multi-seed mean±std under compute-matched settings? Additionally, plotting capability drift curves across task order (before/after sequential updates) would offer a more granular view of the stability–plasticity dynamics.

- SVD–curvature relationship: can the link be verified quantitatively and leveraged adaptively?
The approach relies on the empirical premise that top singular directions approximate high-curvature/importance directions. Could the authors, for representative layers, extract the leading Hessian/Fisher eigenvectors and measure principal angles to the top singular vectors? Further, could they compare “protect SVD-top vs protect Fisher-top vs a joint criterion,” and explore Fisher-weighted singular-value ranking or multi-criteria scoring to define the preserved subspace? Such analyses would more directly substantiate the curvature surrogate assumption and might yield an even more robust adaptive preservation rule.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors propose **Orthogonal Subspace Fine-Tuning (OSFT)** for continual learning. They utilize adaptive singular value decomposition (SVD) to dynamically identify and preserve critical, high-rank parameter subspaces. They first introduce *importance-guided allocation*, i.e., measuring each layer’s importance via input–output cosine similarity (where low-similarity layers are allocated more adaptive capacity). This determines importance scores, which, together with retention ratios, define the fraction of singular vectors to preserve per layer. The authors perform *adaptive subspace identification*, i.e., decomposing each layer via SVD to find critical components with a selected rank. Lastly, they apply *orthogonal gradient projection*, i.e., projecting gradients to be orthogonal to the critical subspaces. Concretely, they freeze high-rank components, and project the low-rank component's gradients to be orthogonal to the high-rank basis vectors.

They evaluate the method across two continual-learning benchmarks and more realistic TRACE benchmark, using encoder–decoder models (T5-Large) and decoder-only models (LLaMA-2-7B, Mistral-7B). They report average accuracy, forgetting measures, and retention of general capabilities and safety.

### Strengths
The paper presents a thoughtful and well-motivated approach to preserving critical subspaces in continual learning.

---

- The paper includes computational analysis as well as theoretical justification.
- A strong set of comparison methods is included.
- A large number of datasets are used in the evaluation.
- The related work section is generally good.
- The hyperparameters (e.g., mrr, trr) appear to have sensible default values, and the authors provide practical guidance for tuning.
- The evaluation follows standard protocols, averaging results over three independent runs with randomly permuted task sequences.

### Weaknesses
### **1. Methodological Clarity & Design**

- Data from task (t) is used when evaluating layer importance for task (t+1).
- The methods section lacks a clear, cohesive narrative.
- The method does not appear particularly parameter-efficient relative to other PEFT approaches.
- The claim that the method “retains and transfers knowledge across tasks” may be too strong; empirical evidence for *transfer* should be clarified.

---

### **2. Missing Definitions & Introductions**

- SLERP, TIES, and TRACE are not properly introduced before being used in comparisons.
- Figure 1 uses terms (TRACE, average backward transfer, forgetting) without definition.
- Cosine similarity is referenced conceptually but never explicitly visualized or explained in the results.

---

### **3. Comparisons & Baselines**

- Prior work [1] shows that full fine-tuning forgets more; therefore, full fine-tuned baselines may not be appropriate, and stronger PEFT baselines should be included.

---

### **4. Presentation & Figure Issues**

- Several plots require cleanup and clarifications; Figures 1 and 2 are presented out of order.
    - Figure 2 is difficult to interpret (e.g., unclear connection between “layer 2 = layer ℓ”, too many elements, cosine similarity not referenced).

---

### **5. Organization & Structure**

- Section 3.7 duplicates material that would be more appropriately placed in the related work section.

[1] LoRA Learns Less and Forgets Less. (2024) Dan Biderman and Jacob Portes and Jose Javier Gonzalez Ortiz and Mansheej Paul and Philip Greengard and Connor Jennings and Daniel King and Sam Havens and Vitaliy Chiley and Jonathan Frankle and Cody Blakeney and John P. Cunningham

### Questions
1. O-LoRA appears to use significantly fewer parameters, and its trade-off curve leads to a wider curve which seems desirable. Could you comment on that? In Figure 1 - why change the rank in O-LoRA, and not the strength of regularization? Similarly, for replay-based baselines, why the learning rate is changed and the amount of data?
2. The related work section does not discuss MiLoRA [1] and PiSSA [2], which seems like a significant oversight - can you include them, also in the experimental section? Especially MiLoRA seems like a crucial method.
3. Can the authors explain what happens when the number of tasks grows large? How does that affect required SVD rank, is there a threshold before capacity becomes an issue? Can the authors relate the SVD subspace of task 0 and task 5, for example? How is interference with early tasks prevented over time?
4. Can the authors further motivate the choice of a *weight-space* approach versus a *function-space* (activation-based) approach?
5. The cited empirical findings (Sharma et al., 2023; Li et al., 2025) show that layers with higher input–output similarity exhibit greater Hessian curvature. Could the authors expand on this relationship?
6. Why do PerTaskFT and MTL report “N/A” for the BT metric?
7. Page 6 L297 - what is the reference for that?

[1] MiLoRA: Harnessing Minor Singular Components for Parameter-Efficient LLM Finetuning. (2025) Hanqing Wang and Yixia Li and Shuo Wang and Guanhua Chen and Yun Chen

[2] PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models. (2024) Fanxu Meng and Zhaohui Wang and Muhan Zhang

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
The paper proposes OSFT for continual learning in large language models. OSFT first performs SVD on each weight matrix to identify high‑rank directions that encode prior knowledge and low‑rank directions that can be safely reused. It computes a layer‑importance score and retains more singular vectors in crucial layers, allocating more adaptable capacity to less crucial ones. During training, OSFT projects gradients orthogonally to the preserved subspaces. The authors support this design with a theoretical analysis linking forgetting to Hessian curvature and with the use of top singular vectors as a practical proxy for high‑curvature directions.

### Strengths
1. The paper is well-structured and easy to read.
2. The paper includes experiments on traditional benchmarks.

### Weaknesses
1. Gradient projection onto an orthogonal complement is a known CL idea. OSFT’s twist is: project in weight space using SVD‑identified directions, not activation space; and make the preserved rank per layer adaptive. The layer‑importance measure is explicitly inspired by AdaSVD, and the theoretical appeal of preserving top singular directions is attributed to an external empirical correlation. This strengthens the sense of a careful integration of existing ideas more than a conceptual break‑through.
2. The proxy that top singular vectors ≈ high‑curvature/Hessian directions is justified by citing external robust correlation, not by measurements on the authors’ own T5/LLaMA runs. The paper does not report a Hessian–SVD alignment check on its trained models. 
3. The proposed method activates more parameters than O-LoRA but fewer than full fine-tuning. By doing so, it improves performance compared to O-LoRA while mitigating forgetting relative to full fine-tuning. This suggests that the method effectively identifies an appropriate number of parameters to utilize. If there exist experiments that adjust the number of active parameters without applying the proposed method, it would help clarify the unique contribution of the proposed approach.
4. The paper cites Inflora (Liang & Li, 2024) and LoTA (Panda et al., 2024). LoTA is only compared in an appendix two‑task transfer, not in the main CL sequences; Inflora is not benchmarked at all, despite being specifically about interference‑free low‑rank continual learning. This weakens the empirical positioning.
5. Figure 1 notes “crucial hyper‑parameters varied for each method,” but the ranges, tuning protocol, and compute budget per method aren’t given. 
6. It’s unclear whether baselines received equivalent tuning or comparable trainable‑parameter fractions—particularly important because OSFT is shown at 56% while LoRA baselines are 1–3%. Claims of frontier domination are therefore budget‑confounded.

### Questions
1. Is there a reason why authors did not measure forgetting per task and forward transfer with comparison methods (A.11 partly helps, but no aggregate forgetting stats are given)?
2. Existing LoRA-based methods do not appear to suffer from performance degradation as severe as that observed with the proposed method on the GSM8K benchmark. In this regard, is there a particular reason why experiments with LoRA-based approaches were not conducted?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposed the orthogonal subspace fine-tuning (OSFT) method in continual learning. It performs SVD on each layer and utilizes the cosine similarity between input activations and its linear outputs to determine high-rank subspace and low-rank subspace of each layer weight. To preserve the important previous knowledge, OSFT orthogonally updates in low-rank subspaces by projecting gradients onto low-rank subspaces. Experimental results on continual learning benchmarks show the effectiveness of OSFT.

### Strengths
1. The proposed OSFT considers the difference between high-rank and low-rank subspaces to achieve orthogonal gradient updates in directions which are orthogonal to previous knowledge.

2. The proposed OSFT adaptively allocates parameter budgets across layers, rather than treating them equally, which balances stability and plasticity across the network based on the role of each layer.

### Weaknesses
1. The computation cost of OSFT would be huge since it needs to compute full-dimensional gradients during training, and then projects full-dimensional gradients to low-rank parameter subspaces obtained by computing SVD on each layer weight.

2. The motivation of the design for OSFT is not well constructed and discussed. For example, authors only cited one paper without any discussion to support the choice of cosine similarity. Since this similarity choice should be one contribution of this paper, which connects with the determination of low-rank subspace, it should be discussed and evaluated carefully.


3. Experiments lack recent parameter-efficient continual learning baselines, SAPT [1], InfLoRA [2], and CorDA [3]. Also, Table 1 does not show the performance on each task order, which is important to show the robustness of the proposed method, and Table 8 in the appendix only shows order 1. Besides, it’s not clear about the task order in Table 3.

[1] SAPT: A shared attention framework for parameter-efficient continual learning of large language models, ACL2024.

[2] InfLoRA: Interference-Free Low-Rank Adaptation for Continual Learning, CVPR2024.

[3] CorDA: Context-Oriented Decomposition Adaptation of Large Language Models for Task-Aware Parameter-Efficient Fine-tuning, NeurIPS 2024.


4. The ablation study disappears in the experiment section. In Eq (3), mrr and trr should be evaluated with different values to show the performance or robustness of OSFT. The authors just mentioned “ablation studies show that while performance degrades significantly if retention is too aggressive”, but authors do not provide these results.

### Questions
1. In this paper, OSFT updates on the subspaces consisting of smaller singular values of the weights, but in CorDA [1] and SVD-LLM [2], both works utilize the input information along with the weights to compute SVD for obtaining the subspaces of smaller/larger singular values. Especially, SVD-LLM states that “truncating smaller singular values in SVD could lead to significant compression loss”, which means there is no linear relationship between performance and singular values obtained only by weight itself. Can authors explain the OSFT’s differences from the conclusions of these two papers?

[1] CorDA: Context-Oriented Decomposition Adaptation of Large Language Models for Task-Aware Parameter-Efficient Fine-tuning, NeurIPS 2024.

[2] SVD-LLM: Truncation-aware Singular Value Decomposition for Large Language Model Compression, ICLR 2025.


2. Can authors compare the experimental performance of OSFT with SAPT, InfLoRA, or CorDA(Knowledge-preserved adaptation version)?


3. In line 15 in Algorithm 1, OSFT “computes gradients for trainable SVD components”. What’s the interval step of computing SVD components? Does OSFT compute SVD at each training step? If so, the computation and training time of OSFT is significantly large, which is not efficient and not practical. Can authors compare the training time and memory cost of OSFT with O-LoRA, SAPT, or InfLoRA?

4. Compared to LoRA-based methods, OSFT needs to compute the gradients of full-dimensional weights, which is very expensive for LLMs. Although the paper title is about “constrained full fine-tuning in LLMs”, FFT is costly for LLMs in downstream tasks. Also, if OSFT belongs to FFT in LLMs, then experiments may need to be redesigned since the current used benchmarks, like GLUE and SuperGLUE, are sampled with 1000 training samples, which are too small for FFT.

5. What’s the choice of effective rank of OSFT in Figure 1? Can authors clarify the experimental settings of OSFT in Figure 1? It’s confusing about how to obtain the three different dots of OSFT in Figure 1.

6. The paper does not discuss in detail why to choose cosine similarity to compute each layer’s importance. Since there are other useful measurements, for example, centered kernel alignment (CKA) [1] measures the similarity between the intermediate activations of two models at each layer. Can authors discuss the difference between these similarities?

[1]  Similarity of neural network representations revisited, ICML 2019.

### Soundness
2

### Presentation
2

### Contribution
2
