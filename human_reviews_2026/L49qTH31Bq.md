# Decoupled Alignment for Robust Plug-and-Play Adaptation

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 6, 4, 4

## Abstract
We introduce a training-free safety enhancement method for aligning large language models (LLMs) without the need of supervised fine-tuning  or reinforcement learning from human feedback.
Our main idea is to provide a robust plug-and-play approach to prevent shadow alignment when models are adapted to downstream tasks. Specifically, we exploit knowledge distillation to extract alignment information from well-aligned LLMs and integrate it into LLMs affected by shadow alignment, in a plug-and-play manner. 
Methodology, we employ delta debugging to identify the critical components of knowledge necessary for effective distillation. On the harmful question dataset, our method significantly enhances the average defense success rate by approximately 14.41\%, reaching as high as 51.39\%, in 17 influenced LLMs, without compromising performance.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper studies the problem that when a chat (aligned) model is further trained on a downstream dataset, its alignment can be disrupted, resulting in an “unaligned” model. To address this issue, the paper proposes a layer replacement method to enhance the safety alignment of unaligned LLMs. Specifically, the method selects the layer most related to alignment in the unaligned model and replaces it with the corresponding layer from the aligned model. Extensive experiments across different model series are conducted to evaluate the effectiveness of the proposed approach.

### Strengths
The paper is well-structured, and the experimental evaluation encompasses a comprehensive range of model series.

### Weaknesses
* **Limited Contribution.**
  The overall contribution of the paper appears limited. The proposed *layer-wise replacement* strategy is relatively coarse-grained, whereas recent model-editing approaches (e.g., [1, 2]) enable fine-grained control over specific knowledge or behavioral components within LLMs. These related works are neither discussed nor compared, leaving the novelty and significance of the proposed method insufficiently demonstrated.

* **Unclear Methodological Description.**
  The method section lacks clarity and logical coherence. The explanation is brief and transitions abruptly between concepts.

  * *Figure 3* analyzes all tokens with hidden-state noise and serves mainly as an illustrative example, whereas *Figure 4* focuses on the last token under MLP-layer replacement averaged over 128 samples. The connection and rationale between these two analyses are not explained.
  * The conclusion regarding the importance of *gate projection* appears premature, as its contribution varies across layers.
  * The equations in lines 154–157 are inconsistent with the Transformer decoder block described in lines 146–148 and with the implementation in *Attention Is All You Need* (Vaswani et al., 2017), which may confuse readers.

* **Outdated and Misaligned Experimental Setup.**
  The experimental setting relies mostly on models from 2023, which limits the relevance of the results to current LLM architectures. Moreover, the Chain-of-Thought (CoT) evaluation is not directly related to the alignment objective. It would be more informative to investigate reasoning-capable models that demonstrate explicit multi-step thinking or deliberative alignment behavior.

* **Questionable Comparative Analysis.**
  Some comparative claims appear overstated. For instance, in *Table 4*, the proposed DAPA method reportedly outperforms *Guardrails* on average. However, *Guardrails* is a rejection-based safety classifier, and the specific configuration or model variant used is not specified. Without this information, the fairness and validity of the comparison are difficult to assess.

[1] Wang, Huanqian, et al. "Model surgery: Modulating llm's behavior via simple parameter editing." arXiv preprint arXiv:2407.08770 (2024).

[2] Wang, Yi, et al. "DELMAN: Dynamic Defense Against Large Language Model Jailbreaking with Model Editing." arXiv preprint arXiv:2502.11647 (2025).

### Questions
See the weaknesses above.

### Soundness
2

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
2

### Summary
This paper builds on the assumption that most of the refusal knowledge of an LLMs is stored in the MLP layer of the LLM. They validate this assumption via adding noise and only restoring the target module and then comparing the probability output probabilities. In the MLPs they further validate the role of gated and up projection layers by replacing them with that of an aligned policy and comparing the output representations. Building on these validated assumptions, they propose to find the modules responsible for misalignment via delta debugging and propose to replace them with aligned modules in order to improve the performance.

### Strengths
1. A wide array of models were considered (3 families of LLMs).
2. The evaluation criteria is satisfactorily defined. The experimentation for both refusal and performance was done on multiple datasets for the sake of generalizability.

3. This paper serves as a finding paper in identifying the underlying correlation between a refusal behaviour and core components of the LLM. I am basing my score of weak acceptance based on the experimentation towards this finding. Though I still have reservations towards the practicality of this as an alignment or guardrail. See weaknesses for the details. 

4. The paper does report their memory shortcomings compared to LORA finetuning in the limitations section.

### Weaknesses
1. My major concern is towards the practicality of DAPA in practical LLM alignment. For instance still editing of the unaligned model still requires the presence of an aligned model (llama 7B chat for llama 7B base etc) presence of such an aligned model it defeats the purpose of DAPA for alignment unless that alignment had caused a significant degradation in certain other aspects. If that is the case those instances should be studied for the validation of the method as a define. For instance based on the reported DSR results for the aligned models the DAPA aligned models never improve beyond the baseline. This makes the justification against not using these aligned models directly important for DAPA to be considered as a defense paradigm.

### Questions
See weaknesses

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes DAPA (Decoupled Alignment for Robust Plug-and-Play Adaptation), a training-free way to re-inject safety/alignment into LLMs that have become “shadow-aligned” after downstream finetuning. The key idea is: take a well-aligned teacher from the same family, use memory / module editing to find which parts of the network actually carry the alignment signal (they argue: middle MLP, especially the gate proj), then copy only those alignment-critical modules into the unaligned model. They use a delta debugging–style search to locate the smallest alignment-effective memory slice, and show on 17 models (LLaMA-2, Mistral, Gemma) that this gives, on average, ~14% DSR gain with only ~8% params changed, and with little perplexity drop.

### Strengths
**S1: Novel use of memory editing for alignment-localization.**
The paper takes the “knowledge editing / memory editing” line of work and repurposes it for safety rather than for factual editing: they experimentally probe hidden states and MLP submodules to locate the neurons / projections that carry alignment information, and then transplant only those parts to the unaligned model. Using memory editing to attribute and transfer alignment-related neurons is, in my view, a fresh angle compared with standard RLHF/DPO-style “retrain the head” solutions.


**S2: Simple but effective delta debugging to keep cost low.**
Instead of editing all MLP/gate layers—which would be expensive and could hurt utility—the paper adopts a delta debugging search over the “memory space” of MLP modules to find the minimal sub-set that improves defense success rate. This is a naive algorithm (partition–test–shrink) but it is transparent, training-free, and fits the plug-and-play goal: you can align many shadowed models in the same family by just copying a few modules from one good teacher. This makes the method attractive for third-party developers who do not want to run RLHF/SFT themselves.


**S3: Solid, broad empirical validation.**
The authors evaluate on 17 models across 3 popular families (LLaMA-2, Mistral, Gemma), on AdvBench jailbreak prompts, and additionally check perplexity, cosine similarity, MMLU, BigBench/CoT before/after editing. The main number—+14.41% avg DSR, up to 51.39%—while changing only 3–8% of params, is a strong empirical signal that the method is not a single-model trick. They also compare with RepE, ICD, and guardrails and report consistent gains. This breadth of experiments strengthens the paper.

### Weaknesses
**W1: Questionable motivation / unclear problem setting.** 
The paper assumes a scenario where “well-aligned” LLMs are already available in the same model family, yet the proposed solution is still to distill and transfer alignment knowledge from the well-aligned model to a “shallowly aligned” or “shadow-aligned” one. This raises a basic motivation question: if a robustly aligned model is already accessible, why is distillation the preferable path instead of using the aligned model directly? The paper does not sufficiently justify when the proposed approach is strictly necessary (e.g., licensing mismatch, partial checkpoints, on-device constraints). As it stands, the practical need for “alignment knowledge transfer” is not fully convincing.


**W2: Over-strong core assumption about representation of ethical knowledge.**
The whole method rests on a bold assumption: “ethical / alignment information learned during alignment is stored as memory inside model neurons.” This is an interesting hypothesis, but the paper does not show that it is the only or even the dominant way safety signals are represented. In a Transformer, alignment may also be encoded in attention patterns, in MLP–attention interactions, in layer norms, or in distributed residual-stream modifications. Without ruling out these alternative loci of representation, treating “alignment = neuron-level memory” as the foundation for the method is too strong.


**W3: From ‘high indirect effect’ → ‘mainly stored here’ is an overclaim.**
In Lines 169–170 the paper observes that (i) hidden states in middle layers and (ii) especially MLP sublayers exhibit the highest indirect effect on the model’s output. This is a good and consistent observation. However, the next step—“therefore, alignment knowledge is mainly stored in the middle MLP layers”—is not logically guaranteed by this experiment. What the experiment measures is sensitivity / causal influence under corruption, not **storage location**. A high indirect effect could be explained by (1) middle MLPs amplifying or integrating earlier alignment signals; (2) middle layers acting as a routing hub in the residual stream, so corruption there hurts more; or (3) MLPs being inherently more semantic and hence more tightly coupled to output probabilities. Unless these alternative explanations are ruled out, the conclusion should be softened.


**W4: Pairwise difference-based localization is indicative, not definitive.**
The paper further narrows down to the claim that the gate projection is the key carrier of alignment knowledge, based on a pairwise, difference-based causal replacement (editing one projection in one layer and measuring recovery). This is a reasonable and widely used probing technique, but by design it mainly tells us “this module is important/sensitive for recovering aligned behavior under this prompt distribution,” not “this is where alignment is stored.” As such, the statement that “by restoring the gate projection, the unaligned model can better align with ethical guidelines” is suggestive but not yet fully convincing. More orthogonal evidence would be needed to support the stronger claim. 

**As a consequence, once the central assumption “alignment is stored in the gate projection” is weakened, the methodological foundation of the paper becomes less solid.**

### Questions
Another issue with this paper is that the writing is not sufficiently rigorous. There are grammatical errors, ambiguities, logical breaks, and even simple typos:
 - 1. **Line 17:** “*Methodology, we employ …*” is not grammatically correct.
 - 2. **Lines 37–38:** This sentence reads somewhat illogically — “*such methods*” is ambiguous. It is unclear whether “*such methods*” refers to “methods that rely on well-aligned models” or to “alignment methods themselves.” In addition, the phrase “*introduce new vulnerabilities into well-aligned models*” raises further questions: what exactly are these “new vulnerabilities”? If there are “new” ones, does that imply there were already “old” vulnerabilities? If so, what are those? This ambiguity also makes it hard to interpret the later statement, “*We refer to this phenomenon as shadow alignment*” — which concrete phenomenon is being named here?
 - 3. **Line 48:** “*see Figures Figure 3 and Figure 4*” — “Figures” is duplicated and should be removed.
 - 4. **Figure 2:** “*Critial MLP layers*” — what is “Critial”? This appears to be a typo (probably “Critical”).

### Soundness
2

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
3

### Summary
The paper introduces DAPA (Decoupled Alignment for Robust Plug-and-Play Adaptation), a training-free method that enhances safety in large language models by transferring alignment knowledge from a well-aligned “teacher” to a target model of the same family. It identifies that alignment signals concentrate in MLP gate layers and selectively edits only a small portion of them using a delta-debugging search, effectively restoring safety without retraining. Tested on 17 models including Llama, Mistral, and Gemma, DAPA significantly boosts defense success rates while maintaining model utility, demonstrating an efficient and scalable way to repair “shadow-aligned” models.

### Strengths
Introduces training-free, plug-and-play alignment via targeted memory transplantation, a novel framing that removes retraining cost and offers a new perspective on alignment localization in LLMs

Demonstrates strong empirical rigor. It is tested on 17 models across Llama, Gemma, and Mistral families, with consistent DSR gains and detailed ablations confirming that MLP gate layers encode alignment knowledge.

Presents a transparent workflow (delta-debugging search + layer transplantation), clear evaluation metrics, reproducibility statement, and open-sourced code ensuring interpretability and replication

Offers a practical, compute-efficient alternative to RLHF/DPO alignment.

### Weaknesses
DAPA explicitly relies on a pre-aligned teacher and cannot align a model from scratch. This raises questions about its motivation — if obtaining a safe teacher model still requires costly alignment methods such as RLHF or DPO, the overall cost advantage may diminish. Moreover, identifying alignment-critical MLP components through delta-debugging also incurs inference cost, yet the paper does not quantify it. It would be helpful to report the actual computational cost of the search phase and compare it directly with SFT, DPO, and RLHF training costs.


The method modifies a relatively large proportion of parameters (6.26% on average), which is much higher than typical adapters like LoRA. My intuition is that a smaller parameter change would better preserve the original model’s behavior, though potentially with weaker safety gains. It would strengthen the paper to show a trade-off curve between the percentage of parameters edited and both safety and utility performance.


In direct comparisons, DAPA’s Defense Success Rate (DSR) remains below that of RLHF and DPO. It would help to clarify whether the reported computational savings (e.g., one hour on a single A100) justify this performance gap in practical deployment scenarios.

The paper does not include a standard supervised fine-tuning (SFT) baseline, which would provide an essential reference point for understanding DAPA’s effectiveness relative to a simpler and inexpensive alignment method. Including SFT results under the same experimental settings would make the comparison more complete.

### Questions
Since DAPA requires a pre-aligned teacher model, how should readers interpret the claimed “training-free” benefit? If the teacher itself was aligned via RLHF or DPO, does DAPA still provide a net cost reduction in realistic pipelines?

The paper reports that about 6.26% of parameters are modified on average. Have you evaluated smaller-scale edits (e.g., 1–3%) to see how DSR and task performance degrade?

You compare DAPA with RLHF and DPO but not standard SFT. Could you add SFT results using the same HarmBench or AdvBench setup to contextualize whether DAPA still provides a meaningful gain over simple supervised fine-tuning?

### Soundness
3

### Presentation
3

### Contribution
3
