# KnItLM: Weaving Knowledge into Instruction-Tuned LLMs via Continual Pre-Training and Merging

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 2, 6, 6

## Abstract
RAG has become the de facto method for incorporating new, corpus-specific knowledge into an instruction following LLM (Instruct LLM). Although RAG-based prompting improves factual grounding, it fails when retrieval is incorrect or incomplete, leading to hallucinations. Finetuning methods such as RAFT and PA-RAG enhance RAG by ingesting new knowledge into the parameters of the model,  but require generating massive amount of synthetic QA that covers the entire corpus. Continued Pre-Training (CPT) on the text corpus avoids the need for comprehensive synthetic data generation but breaks the instruction following capabilities of an Instruct LLM, necessitating instruction fine-tuning (IFT) post CPT. However, IFT is costly and may be infeasible due to the unavailability of an instruction tuning corpus. In this work, we propose KnItLM - KNowledge IngesTion via LoRA Merging. Instead of doing CPT on the Instruct LLM, KnItLM performs CPT with Low-Rank Adapters (LoRA) on its corresponding base LLM  to infuse new knowledge. These knowledge-infused LoRA weights are then merged with the Instruct LLM, imparting new knowledge without impacting their instruction following capabilities. KnItLM avoids expensive instruction fine-tuning and relies on model merging to infuse the new knowledge into the Instruct LLM without destroying its instruction following capabilities. Empirical results show that KnItLM significantly improves the performance of RAG by taking accuracy from $54.17$% to $79.26$% for retrieval failure cases. In addition, the proposed method achieves superior performance to existing approaches while requiring substantially less training data.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The KNITLM (Knowledge Ingestion via LoRA Merging) framework proposed in this paper addresses the core issues of "continuous pre training (CPT) disrupting instruction following ability" and "large dependence on synthetic data" in the domain knowledge injection of Instruction LLM. It innovatively adopts the technical route of "training knowledge LoRA on the Base LLM+integrating with Instruction LLM" to achieve collaborative retention of knowledge injection and instruction ability.

### Strengths
1. By training knowledge LoRA on the base model and integrating it with Instruction LLM, the synergy between "knowledge vector" and "instruction vector" is theoretically achieved using "task vector addition", which not only retains the instruction following ability of Instruction LLM, but also injects new domain knowledge.

2. By using the GRPO algorithm and a mixed reward of "effectiveness-efficiency-structural quality", the accuracy of knowledge injection is ensured while overfitting is suppressed, achieving lightweight training

### Weaknesses
1. In the relevant work section, although task arithmetic is mentioned, there is no in-depth comparison of the core differences between KNITLM and other model editing methods.

2. Using LLM as Jade binary scoring (0/1) as the core indicator, other key indicators of knowledge injection were not reported.

3. Verified only on two Redbook technical document datasets and not extended to other fields such as healthcare, finance, short text conversations, and long document reports. The model has not been tested on larger-scale models (20-80B) or different architecture models, making it impossible to determine the model compatibility of the method.

4. There are too few baselines used, and more SOTA models need to be compared.

### Questions
see above

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper addresses the challenge of incorporating domain-specific knowledge into instruction-tuned LLMs without compromising their instruction-following capabilities. The authors propose KnitLM, which performs continual pretraining (CPT) with LoRA adapters on base LLMs and transfers these adapters to instruction-tuned models. The authors also propose to use instruction model token embeddings during training to improve adapter compatibility.

### Strengths
- The paper tackles an important practical challenge in LLM: how to efficiently incorporate new domain knowledge without expensive instruction fine-tuning.
- The experimental setup carefully avoids data contamination, and the evaluations are reasonably thorough.

### Weaknesses
- The core contribution is incremental and somewhat limited. The central idea of combining LoRA-based continual pretraining with model merging (task arithmetic) is not novel. Prior work has explored task vector merging and LoRA transfer. The only substantive twist is using instruct-model token embeddings during LoRA training, which is a minor technical detail rather than a conceptual advance.
- The paper only compares against RAFT and PA-RAG. Stronger baselines (e.g., direct LoRA tuning on the instruct model, or parameter interpolation approaches) are missing.
- Section 3 uses $\Delta\theta$ to denote both task vectors (differences between model weights) and LoRA adapter weights. This is confusing because task vectors are full-rank weight differences, while LoRA adapters are low-rank decompositions.
- Minor comment: Citation formatting needs correction. Citations should appear in parentheses rather than as part of the text: Zhang et al. (2024b) -> (Zhang et al., 2024b). In the abstract, many abbreviations such as RAG,  LLM are used before defining them.

### Questions
1. Why not simply fine-tune base model fully on new knowledge (not LoRA), then compute true knowledge task vector and add to instruct model?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work presents a lightweight knowledge-ingestion method that performs CPT with LoRA on the base model and merges the knowledge adapter with the instruction model. An insightful technique is taking the instruction model's token embeddings during CPT to reduce distribution shift. The proposed method targets preserving the model's instruction-following ability while improving closed-book and RAG performance.

### Strengths
1. Training a knowledge LoRA on the base model and merging it with the instruction model is a lightweight technique that avoids re-running the SFT and may be RL for preserving the instruction following ability.
2. Using token embeddings from the instruct LLM during CPT is insightful and reasonable.

### Weaknesses
I am satisfied with the method itself, but have concerns about the experiment settings. 
1. Limited baselines and benchmarks. The evaluation is restricted to two technical Redbooks. While the authors discuss test set quality and knowledge cutoff issues, the current scale is small and may not fully establish generality. The cutoff may be avoidable by testing the direct QA performance. If the model performs poorly for the test questions, it could imply that the model's parameterized knowledge does not contain the test set knowledge. In this way, this work can be comparable with board baseline models on more benchmarks.

### Questions
1. For the ablation study in section 5.4, the performance drop of e-KNITLM seems not very significant. The usefulness of replacing the embedding with the instruction model may contribute to two aspects, 1) as the author stated, avoiding OOD tokens like the special tokens used in the chat template, 2) for the tokens that are well-trained in the base model, the instruction model has a better representation. I am curious which one contribute more to the performance gain?
2. For the section 5.2, IMPACT OF THE SIZE OF THE SYNTHETIC DATA, I agree with the statement "access to QA from only a part of the corpus, will still show gains over the remaining data", but this seems to be a generic property. Does the KnItLM benifit more from this and why the baseline methods can not, since they also adopt synthetic data for training.

### Soundness
2

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
This paper proposes a novel continued pre-training approach for incorporating new knowledge into instruction-tuned LLMs, addressing the issue that conventional methods often degrade instruction-following capabilities. The authors introduce a method that merges two task vectors: a knowledge vector (representing the new knowledge to be injected) and an instruction task vector, obtained as the difference between the baseline model vector and the instruction-tuned model vector. Experimental results show that KnItLM achieves improvements over existing baseline methods.

### Strengths
1.	The proposed merging of the instruction-following vector and the knowledge vector based on task vectors is novel, well motivated, and technically interesting.
2.	The special treatment of token embeddings adds further depth and elaboration to the method.
3.	The paper is clearly written and easy to follow, with detailed explanations.
4.	The experimental results demonstrate that the proposed methods achieve performance improvements over the baselines.

### Weaknesses
1.	In Table 1, it is unclear whether the reported improvements are substantial.
2.	It is not clear whether the proposed method can be applied in an incremental manner when additional knowledge is introduced. Once the knowledge vector is injected into the base model, does the updated model then serve as the new base model for the next stage?
3.	Additional comparisons with other continual learning baselines would strengthen the experimental evaluation.

### Questions
Please see weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
