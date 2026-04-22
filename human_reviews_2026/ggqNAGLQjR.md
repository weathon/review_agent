# MedSimSearch: Sim2Real Agentic Learning for Medical Visual Reasoning

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 0, 4, 6

## Abstract
Developing autonomous agents for complex Medical Visual Reasoning is a critical goal, yet training them in real-world clinical settings is largely infeasible due to severe privacy, data, and safety constraints. While retrieval-augmented methods exist, they often depend on impractical multimodal indexing or fail to address the core challenge of learning interactive policies without real-world exposure.
To bridge this gap, we introduce MedSimSearch, a novel framework based on Sim2Real Agentic Learning. The core innovation lies in leveraging a generative large multimodal model (LMM) to create a high-fidelity simulated retrieval environment. Within this safe, text-only simulation, our agent learns a robust search and reasoning policy, eliminating the need for multimodal data indexing while preserving patient privacy.
To validate our approach, we evaluate the agent trained in simulation on realistic medical benchmarks using a curated private text corpus. Extensive experiments on VQAMed2019 and OmniMedVQA demonstrate that MedSimSearch significantly surpasses strong retrieval-augmented generation (RAG) baselines and shows enhanced robustness against hallucinations, paving a viable path for deploying trustworthy medical AI agents.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The work introduces MedSimSearch, a framework that operationalizes the Sim2Real paradigm for agentic learning in the context of medical visual reasoning. The work is motivated by the real-world challenges of training autonomous agents in clinical settings, which are often constrained by privacy, data scarcity, and safety concerns. The core proposal is to leverage a generative Large Multimodal Model to create a text-only simulated retrieval environment. Within this safe simulation, an RL agent learns a robust, text-only search policy, eliminating the need for impractical multimodal data indexing. The agent's policy is optimized using GRPO through a curriculum-based rollout that progressively introduces noisy/negative pseudo-documents to improve robustness. The agent, trained entirely in this simulated environment, is then validated on real-world medical benchmarks using a private text corpus.

### Strengths
- The work is very well-motivated. It identifies a critical and practical bottleneck in deploying interactive medical AI: the inability to train agents on real systems due to privacy, access, and data scarcity. The proposed Sim2Real approach, using a text-only simulator, is a pragmatic and well-justified workaround for this significant hurdle.
- The work integrates several advanced methods to address this problem. The use of an LMM-as-simulator, combined with a curriculum learning strategy (dynamically mixing "useful" vs. "noisy" pseudo-documents), and optimization via GRPO, represents a novel and technically sound pipeline for this domain.
- The experiments are thorough and well-designed. The authors evaluate MedSimSearch on two benchmarks and compare against a comprehensive set of baselines, including zero-shot, SFT, and RAG models. The validation on a specialized, curated medical text corpus to test the Sim2Real transfer is a strong component of the evaluation. The results clearly demonstrate gains over existing SOTA methods.

### Weaknesses
- A major concern is the high computational cost of the proposed method. The paper states that training requires 4 NVIDIA A100 GPUs for the simulation server and another 4 A100 GPUs for the RL training. This 8-GPU setup makes the results difficult to reproduce for many research labs. Furthermore, the paper does not discuss the deployment cost and latency. If the agent's learned policy requires repeated calls to a powerful LMM (like the GPT-4o used in experiments) to generate pseudo-documents during inference, the practical utility in a real-time clinical setting is unclear.
- The entire framework's success hinges on the fidelity of the LMM-generated pseudo-documents. The paper's analysis in Figure 2 attempts to address this by testing generalization to other LMMs; however, this analysis is insufficient to fully probe the gap. The models tested (GPT-4V, LLaMA-3-70B) are still very high-capability. When tested with a smaller open-source model Qwen-7B, a sizable performance drop is observed. While relying on powerful, large models is acceptable, the associated computational cost and resource requirements are critical factors in the method's overall evaluation and should be weighed accordingly (as in W1).
- Even powerful LMMs are susceptible to hallucination, and the 30-word limit on pseudo-documents does not eliminate this risk. The paper does not propose a clear mitigation strategy for this problem. It is also not specified whether any human verification was performed on the synthesized pseudo-documents to assess their plausibility (useful or noisy). An analysis of the potential impact of simulator-induced hallucinations on the agent's policy is a critical but missing component of the evaluation.

### Questions
- Are you going to open-source the curated medical text corpus and the pseudo-documents?
- The curriculum learning based on "useful" vs. "noisy" pseudo-documents is a key component. To improve the qualitative understanding of this mechanism, it would be highly beneficial to add a few side-by-side examples of these generated documents. Brief annotations explaining why a specific document is considered "useful" (accurate, relevant) versus "noisy" (misleading, plausible-but-wrong) would be very helpful.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
This paper used generative AI to synthesize documents for training the model to better retrieve evidence for medical VQA, achieving higher exact match scores than baselines on two benchmark datasets.

### Strengths
The problem the paper aims to address is significant.

### Weaknesses
The study design is misaligned with the multimodal medical scenario.
(1) For a vision-centric problem (as in Fig. 1, e.g., “what modality is shown”), why use purely text retrieval?
(2) Why do the pseudo-documents include only text, generated by non-medically validated models, so content quality cannot be ensured?

Overclaim:
(1) The paper provides no evidence that the simulation is a high-fidelity medical environment.
(2) The RAG baselines compared are not multimodal, yet the task is VQA; claiming large gains over RAG is based on a single, common text-only implementation.
(3) The curriculum is fixed, with no ablation to show its effectiveness.

Unsupported evaluation:
(1) Only two datasets are used.
(2) Metrics are unclear (e.g., what is "Micro" in Table 3? Which BLEU variant, BLEU-1?).
(3) Inconsistent results: in Table 4, all categories have very low accuracy, yet overall accuracy is much higher, mathematically inconsistent.

### Questions
The foundational method is Zero-Search, why not use the same implementation as the baselines for comparison?

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes MedSimSearch, a framework that leverages a large multimodal model (LMM) to generate synthetic medical Q/A pairs and pseudo-documents for training an RL agent in a simulated environment. The idea is to perform Sim2Real Agentic Learning—teaching a model retrieval and reasoning policies safely through text-only simulation instead of real-world medical data. While the concept of using an LLM to emulate an environment is clever and practical for privacy and data-scarcity constraints, I find the technical contribution somewhat limited... The best model this paper proposed, which uses GPT-4o, have an inevitable issue of hallucination. In addition, this model have not been compared to the best model available.

### Strengths
- The paper is well written and easy to follow, with a clear narrative and logical experimental setup.
- The overall idea—training an RL agent in a simulated environment generated by an LLM—is straightforward yet practical, especially for domains like medicine where data privacy is a barrier.

### Weaknesses
- From a technical perspective, the work is an incremental extension of existing approches. The use of an LLM to simulate the training environment is creative but conceptually similar to prior self-play or synthetic data paradigms, without introducing a new optimization method or architecture.

-  The reported gains are not substantially higher than existing baselines, and Table 4 compares mainly against Qwen 2.5-VL variants. The paper omits stronger contemporary baselines such as Med-R1, Evo-PI, or HuatuoGPT-Vision, making it difficult to judge true progress.

- Hallucination risk. Because the simulated environment is entirely generated by an LLM, hallucination is inevitable. The model learns from self-generated, potentially inaccurate contexts, which could reinforce false or biased information rather than real clinical reasoning.

- Figure presentation quality. The figures, particularly Figure 1, feel rough and early; even the caption (“Overview of MedSimSearch”) is overly brief for a central conceptual diagram.

- Evaluation metric choice. The BLEU score adds little value for measuring factual correctness in medical VQA.

### Questions
See weaknesses

### Soundness
3

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
The paper proposes MedSimSearch, an RL-trained, text-only search agent that learns in an LMM‑simulated retrieval environment and is then evaluated on medical VQA (VQAMed2019, OmniMedVQA). It reports strong gains over RAG and RL baselines, while avoiding multimodal indexing and claiming privacy benefits. The idea is timely and the empirical results are promising, but important methodological details (label exposure in simulation prompts, reliance on GPT‑4o at train/test, fairness of baselines, and clarity around the RL algorithm) need tightening before the work can be considered solidly conclusive.

### Strengths
1. Training agentic policies for medical VQA without accessing real clinical systems is important and underexplored. The simulator‑first stance is well motivated by privacy and availability constraints. 

2. Strong empirical results across two benchmarks. Consistent wins over capable baselines, including RAG and recent RL methods. The gains are nontrivial and span multiple modalities.  

3. This paper is well written and easy to follow.

### Weaknesses
1. The simulator is explicitly told “whose ground truth answer is [ground truth]” to generate pseudo‑documents. Even if only used during training, this risks imprinting the correct answer distributionally into “useful” docs; the agent may learn patterns to read it off rather than to search. The paper should quantify how much the agent relies on this supervision signal and show performance when ground truth is not passed to the simulator during training.  

2. The method and baselines both use GPT‑4o to generate pseudo‑documents at test time “to ensure fair comparison". This conflates the contribution of policy learning with the capabilities of a closed, expensive model. It is unclear how much of the final score comes from GPT‑4o’s world knowledge versus the learned policy. 

3. Corpus mismatch vs. reported multi‑modality gains. The “private” database C is overwhelmingly radiology‑centric (50k MIMIC‑CXR reports + 5k synthetic). Yet the Database variant scores highly on non‑radiology modalities in OmniMedVQA (e.g., OCT, FP). How can a largely chest‑X‑ray text corpus support ophthalmology/pathology questions so well? Either the synthetic 10% happens to cover those domains richly, or the model mainly relies on the generative simulator even in the Database setting. This needs clarification and ablation.  

4. Section 3 emphasizes GRPO (with equation), Appendix A.2 lists GRPO settings, but §4.2 says “Unless otherwise specified, PPO is the default.” Which results are from which? A controlled GRPO vs. PPO ablation is missing, and the training stability/variance is not reported.  

5. The “noisy” pseudo‑docs are produced by instructing the LMM to include “misleading or partially incorrect information”. This may not reflect real retrieval noise (e.g., partially relevant but off‑topic passages, domain shifts, OCR artifacts). The external validity of the curriculum is thus uncertain.  

6. Beyond the simulator swap, we lack ablations on (i) the curriculum schedule, (ii) action budget B, (iii) the usefulness of <think> vs. <info> structure, (iv) whether the agent truly learns search sequencing vs. just better prompting to GPT‑4o.

### Questions
See above.

### Soundness
3

### Presentation
2

### Contribution
3
