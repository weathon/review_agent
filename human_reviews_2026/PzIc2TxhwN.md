# Seek-CAD: A Self-refined Generative Modeling for 3D Parametric CAD Using Local Inference via DeepSeek

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
The advent of Computer-Aided Design (CAD) generative modeling will significantly transform the design of industrial products. The recent research endeavor has extended into the realm of Large Language Models (LLMs). In contrast to fine-tuning methods, training-free approaches typically utilize the advanced LLMs, thereby offering enhanced flexibility and efficiency in the development of AI agents for generating CAD parametric models. However, the lack of a mechanism to harness Chain-of-Thought (CoT) limits the potential of LLMs in CAD applications. The Seek-CAD is the pioneer exploration of locally deployed inference LLM DeepSeek-R1 for CAD parametric model generation with a training-free methodology. This study is the investigation to incorporate both visual and CoT feedback within the self-refinement mechanism for generating CAD models. Specifically, the initial generated parametric CAD model is rendered into a sequence of step-wise perspective images, which are subsequently processed by a Vision Language Model (VLM) alongside the corresponding CoTs derived from DeepSeek-R1 to assess the CAD model generation. Then, the feedback is utilized by DeepSeek-R1 to refine the initial generated model for the next round of generation. Moreover, we present an innovative 3D CAD model dataset structured around the SSR (Sketch, Sketch-based feature, and Refinements) triple design paradigm. This dataset encompasses a wide range of CAD commands, thereby aligning effectively with industrial application requirements and proving suitable for the generation of LLMs. Extensive experiments validate the effectiveness of Seek-CAD under various metrics.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Seek-CAD, a novel framework for generating 3D parametric CAD models. The framework's core features are that it is training-free and locally deployable.

It uses an open-source LLM  as its core, combining RAG with a novel "SSR" design paradigm to generate complex CAD code.

Its core innovation is a self-refinement loop.

Experiments show Seek-CAD exceeds other training-free methods in geometric accuracy and even surpasses the fine-tuned CAD-Llama model.

### Strengths
The framework's key innovation is using a VLM to align the LLM's Chain-of-Thought (CoT) with step-wise visual renderings, thereby validating the design process rather than just the final product.
Despite being training-free, Seek-CAD surpasses the fully fine-tuned CAD-Llama on key geometric metrics, proving the potential of its "generate-verify-refine" agent loop.
The system is a practical, self-contained solution, using a locally deployed LLM (DeepSeek-R1) and an essential local RAG corpus instead of relying on expensive, closed-source APIs.

### Weaknesses
The entire refinement loop's effectiveness hinges on the VLM's (Gemini-2.0) ability to accurately assess the alignment between the CoT and the step-wise images. The paper admits in its limitations (Sec 5) that VLMs can have biases or misunderstand complex geometry. If the VLM hallucinates or misinterprets (as seen in Fig 4b), it will provide faulty feedback, causing the LLM to make incorrect "corrections."


The "LLM $\rightarrow$ Render $\rightarrow$ VLM $\rightarrow$ LLM" loop is computationally slow. More critically, Table 2 shows that going from Round 1 to Round 2 of refinement yields marginal performance gains (e.g., IoGT only 0.72 $\rightarrow$ 0.73) but causes the code compilation success rate (Pass@2) to drop sharply (from 0.72 to 0.55). This suggests multi-turn refinement may lead the LLM into a state of confusion or over-correction, breaking the code's validity.

### Questions
The refinement loop (LLM $\rightarrow$ VLM $\rightarrow$ LLM) is expensive in time, and Table 2 shows its success rate (Pass@k) drops significantly with more iterations. Compared to a one-pass, fine-tuned model (CAD-Llama), is Seek-CAD still competitive in terms of actual wall-clock time and API call costs?


The system relies on a VLM (Gemini) as a "referee" to validate the LLM's (DeepSeek) CoT. If the VLM and the LLM share the same misunderstanding of the user's prompt (e.g., they both misinterpret the meaning of "chamfer"), will the system "confidently" refine toward a wrong answer, because the referee (VLM) will incorrectly validate the executor's (LLM) flawed logic?

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
The paper proposes Seek-CAD, a training-free framework that locally runs DeepSeek-R1-32B-Q4 with RAG over a CAD-code corpus to generate SSR-style CAD programs, then iteratively self-refines them using step-wise visual feedback: rendering intermediate + final shapes and asking a VLM (Gemini-2.0) to judge alignment with the LLM’s CoT, feeding that back for correction.

### Strengths
1- Method novelty in feedback: evaluates intermediate renders + CoT rather than final-image only; ablations show inter-image cues matter.

2- Empirical signal: Seek-CAD beats prior training-free refiners and edges a tuned model on geometric fidelity (CD/HD/IoGT), with qualitative evidence.

### Weaknesses
1- VLM Feedback Quality: The authors acknowledge that VLMs struggle with geometric descriptions without domain-specific training (Section 5.5), but this fundamental limitation undermines the core refinement mechanism. No quantitative analysis is provided on how often Gemini-2.0's feedback is actually helpful vs. harmful.

2- Compilation Failure Rate: The Pass@k metric reveals concerning compilation failure rates. Even after 2 refinement rounds, only 55% of generated models compile successfully (Table 2). This significantly limits practical applicability. The paper doesn't adequately address strategies to improve this.

### Questions
1- Can you provide analysis on when/why Gemini-2.0 feedback helps vs. hurts?

2- Can you provide failure case analysis beyond the two examples in Figure 4(b)?

3- How does performance vary with CAD model complexity (e.g., number of features)?

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
4

### Summary
This paper presents Seek-CAD, a training-free framework for generating 3D parametric CAD models from text. It uses a locally deployed LLM (DeepSeek-R1) to generate code following a novel SSR design paradigm. A key innovation is a self-refinement loop where step-wise renders of the CAD model are evaluated by a VLM (Gemini-2.0) against the LLM's Chain-of-Thought reasoning; the resulting feedback iteratively improves the code. The authors also contribute a new 40k-sample dataset based on the more complex SSR paradigm. Experiments show Seek-CAD outperforms existing methods in geometric fidelity and text alignment.

### Strengths
1.The use of step-wise visual renders paired with the LLM's Chain-of-Thought for feedback is new. This provides a richer, more granular signal for refinement than methods using only the final render.

2.The proposed SSR triple and the CapType reference mechanism enables the generation of complex CAD models beyond the limitations of prior "Sketch-Extrude" methods.

### Weaknesses
1.The paper mentions that models failing to compile are excluded from metric calculation. A more detailed analysis of the reasons for compilation failures would be insightful. 

2.While the CapType mechanism is innovative, the description in the appendix mentions that when refinement commands involve primitives not identifiable by CapType, those primitives are simply excluded. How often does this happen in the dataset/generation? Does it lead to models that are missing intended refinements?

3.The RAG corpus has 10,000 samples. Was an ablation study performed on the size of this corpus? Is there a point of diminishing returns, or could a smaller corpus suffice?

### Questions
Beyond compilation failures, what are the most common types of geometric or logical errors that the refinement loop fails to correct?

### Soundness
3

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
5

### Summary
Authors proposed seek-cad, a training free approach for generating CAD code from textural input. It formulate data in a novel SSR template, and use a captype reference to effectively support chamfer and fillet operations on new surfaces introduced by CAD operations. The RAG system is also novel with an additional one round of step-wise visual feedback in the code refinement stage showing to help improve generation quality.

### Strengths
The proposed training-free approach is novel and the first few works that explored this direction. The SSR data format with captype reference is also more general and applicable to real-world scenarios than simple sketch-and-extrude. SVF is also a nice solution for incorporating step-vise visual feedback into the system and enable the model to verify each step in the built process. Evaluation on the new dataset demonstrate the improvement of seek-cad.

### Weaknesses
Writing and paper layout can be improved. Figure 1 is too small, and SSR definition is in the later paragraph whereas a lot of reference to it is at the front. Overall, this makes reading the paper difficult than it should be. 

Evaluation is done entirely on the authors’ new SSR dataset. There is no comparison to previous methods on existing public CAD data like DeepCAD / Omni-CAD / WHUCAD. Figure 7 and 8 shows their dataset is much more complex than DeepCAD, this raise the concern that metric improvement could come from the 10,000 more complex RAG data, e.g better novelty than other methods.

Authors did not clearly explain how the test set is different from training set. E.g what kind of deduplication or similarity filter was applied. 

Authors do not provide concrete implementation details for Eq. (1) and Eq. (2) — only high-level descriptions of what those equations represent conceptually. This makes reproducing the work fairly complicated.

### Questions
(1) Why not use a single vllm model like Qwen-VL or InternVL. The proposed design using DeepSeek for text and Gemini 2.0 for visual is fairly complicated. Is there a particular reason why authors use this pipeline?

(2) How is the dataset constructed? What method is used to avoid data leaking from training to test set? Does it have the clearance from OnShape to be allowed to be publicly released? 

(3) How important is the RAG data? Does the increase in complexity help? 

(4) Is it possible to use exsiting public CAD data as the RAG data and see how it compares to baselines? This seems like the fair way to compare without the results been affected by the new dataset. 

(5) Please provide implementation details in the paper for reproducible results.

### Soundness
3

### Presentation
2

### Contribution
3
