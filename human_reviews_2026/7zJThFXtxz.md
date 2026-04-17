# SceneNAT: Masked Generative Modeling for Language-Guided Indoor Scene Synthesis

- Decision: Reject
- Scores: 2, 6, 4, 6

## Abstract
We present SceneNAT, a single-stage masked non-autoregressive Transformer that synthesizes complete 3D indoor scenes from natural language instructions through only a few parallel decoding passes, offering improved performance and efficiency compared to prior state-of-the-art approaches. SceneNAT is trained via masked modeling over fully discretized representations of both semantic and spatial attributes. By applying a masking strategy at both the attribute level and the instance level, the model can better capture intra-object and inter-object structure. To boost relational reasoning, SceneNAT employs a dedicated triplet predictor for modeling the scene's layout and object relationships by mapping a set of learnable relation queries to a sparse set of symbolic triplets (subject, predicate, object). Extensive experiments on the 3D-FRONT dataset demonstrate that SceneNAT achieves superior performance compared to state-of-the-art autoregressive and diffusion baselines in both semantic compliance and spatial arrangement accuracy, while operating with substantially lower computational cost.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes to use mask modeling for indoor scene representation. To improve the modeling of two-object relations, a bipartite matching is used to optimize the one-to-many mapping between each object pair. Experimental results have shown improved performance over three methods: ATISS, DiffuScene and InstructScene.

### Strengths
+ the introducing of a detection module for the detection of object-object relations

### Weaknesses
- The biggest issue for me is how model like BERT's masked modeling which is originally designed for representation learning can be used for generation. I cannot find the pipeline of how the inference is executed. 
- I also doubt why a parallel decoder like DETR can outperfom a denoising deffusion based model and AR based model. There should be more analysis experimental to support this claim. 
- There should be a discussion with COFS which is an encoder-decoder pipeline like BART structure for scene generation. 
- Figure 2 is not clearly explained. Is triplet query an input? Is it learned or provided? 
- The DETR module is not clearly explained either. In the original DETR, a bipartite matching is first optimized for matching each query with one of the preset query candidate (which is to be optimized) and then adjust all the matched query candidate to the GT ones. In the proposed method (along with Figure 2), I do not understand how the above CE loss is defined and even though it can somehow trained, how will the triplet feature update to the mask modeling is also unclear. 
- Missing discussion and comparsions with some SOTA autoregressive models such as FOREST2SEQ [1] and CASAGPT [2]
[1] https://arxiv.org/abs/2407.05388
[2] https://arxiv.org/abs/2504.19478
- There are no failure case provided. 
- It seems that the model cannot deal with layouts with various shapes. All the layouts shown are rectangles.

### Questions
See Weakness

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work introduces SceneNAT, a single-stage, masked non-autoregressive Transformer for language-guided 3D indoor scene synthesis. SceneNAT enhances performance and efficiency by employing masked modeling on discretized semantic and spatial attributes, alongside a novel triplet predictor for relational reasoning.

### Strengths
- **Novelty**. The method introduces a tailored non-autoregressive Transformer (NAT) for 3D scene synthesis, using dual-granularity masked modeling (attribute- and instance-level) to capture intra- and inter-object dependencies. The decoupled triplet predictor improves spatial relation modeling, overcoming limitations of traditional text representations.

- **Comprehensive Experiments**. The experiments are thorough, including quantitative metrics (iRecall, FID/CLIP-FID/KID) and qualitative comparisons (Figures 3, 9-13). Ablation studies (Tables 3, 5) validate key components, and zero-shot tasks (e.g., stylization, layout-to-object) demonstrate generalization.

- **Writing**. The manuscript is well-structured and clearly written, with precise technical details that make it accessible to both specialists and a broader audience.

### Weaknesses
1. **Limited Generalization to Complex Relational Scenarios**. This work limits the maximum number of relational constraints per instruction to 4, citing the token length limit of the CLIP text encoder. However, it does not address how the model would scale to more complex and realistic design scenarios. Additionally, the paper only validates 11 predefined spatial relations (e.g., "right of", "above") from InstructScene, missing common fine-grained or ambiguous relations, and still faces the issue of limited instruction diversity as noted in InstructScene.

2. **Unaddressed Discretization Bias and Error**. This work discretizes continuous 3D attributes into fixed bins but does not analyze how the granularity of discretization impacts generation quality. This is a significant oversight: coarser bins may introduce spatial errors (e.g., object overlaps due to imprecise position prediction), while finer bins could lead to increased model uncertainty.

### Questions
Please refer to the "Weaknesses" section.

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
SceneNAT proposes a masked non-autoregressive transformer for language-guided indoor scene synthesis. It discretizes semantic and spatial attributes, applies attribute- and instance-level masking, and adds a relation triplet predictor (subject, predicate, object). Compared to autoregressive/diffusion baselines, it targets fewer decoding passes with better efficiency while improving semantic compliance and spatial arrangement on 3D-FRONT.

### Strengths
1. The paper introduces a masked non-autoregressive Transformer for language-guided 3D scene synthesis, augmented with an explicit triplet-based relational module. This design is well-motivated and yields favorable decoding efficiency relative to conventional autoregressive or diffusion approaches. 
2. On synthetic benchmarks, the method attains strong results on instruction adherence (iRecall), perceptual realism (FID), and object-level recall, while maintaining fast inference with the non-autoregressive bench.

### Weaknesses
1. The comparisons largely stop at pre-2025 methods and omit recent state-of-the-art systems (e.g., ReSpace [1])
2. The paper lacks qualitative or quantitative diagnostics (e.g., relation violations, object collisions, discretization artifacts and so on).
3. All results are on synthetic data; there is no real-world study (assets/layouts) or human evaluation, so external validity and deployment readiness remain unclear.
4. The manuscript does not examine multi-room layouts, longer relational prompts, larger object vocabularies, the senarios tested are pretty narrow.

[1] ReSpace: Text-Driven 3D Indoor Scene Synthesis and Editing with Preference Alignment

### Questions
1. Can you also specific the VQVAE design and training time on this? 
2. Can the approach handle multi-room scenes and longer prompts (beyond the current relation cap)? What are the runtime/memory curves versus object count and room count, and how stable is decoding under these settings?
3. How does the model perform with real-world data?
4. More recent baseline will be helpful to understand the recent works and trends.

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
This paper introduces SceneNAT, a Non-Autoregressive Transformer (NAT) for 3D indoor scene generation that addresses the efficiency limitations of AR and diffusion models. The core contribution is a Masked Generative Modeling approach, akin to a BERT-style objective, which reconstructs discretized scene attributes in a few iterative steps. To ensure high-fidelity language control, the model employs a DETR-inspired Triplet Predictor to explicitly parse spatial relations. The final generated attributes guide an Object Retrieval step to build the scene. Experiments demonstrate that SceneNAT achieves SOTA performance, especially in instruction-following and efficiency.

### Strengths
+The paper's core insight—that a Non-Autoregressive Transformer is a powerful alternative to AR or Diffusion models for this task—is a significant strength. This approach opens a promising third direction for high-speed, high-quality structured 3D generation.

+The concept of the Triplet Predictor is also a key strength. Decoupling symbolic relation-understanding from the geometric generation task is an intelligent design choice.

+The paper shows impressive performance and efficiency via comprehensive experiments, outperforming existing SoTA methods.

### Weaknesses
-A key concern is the reliance on templated data. The training instructions are synthetically generated from structured relations. This implies the model is primarily learning to "invert" this synthetic generation process rather than parsing free-form human language directly.  It remains unclear how the model would generalize to ambiguous, "in-the-wild" human instructions that do not follow the rigid structure. Therefore, the claim of handling "complex instructions" may be somewhat overstated.

-The model appears to use a fixed-size scene matrix with a maximum of N object slots. This is an inflexible architectural choice. It's unclear how the model handles scenes that require a number of objects significantly different from N (either much sparser or much denser than the training data). This contrasts with AR models that naturally support variable-length generation.

-The model has two parallel reasoning systems: the Scene Decoder learns implicit rules of scene plausibility, while the Triplet Predictor enforces explicit user instructions. The paper does not discuss how the model arbitrates conflicts between these two systems. What happens if a user instruction is physically implausible or violates the model's learned common sense? The system's behavior under such contradictory signals is a key aspect of controllability and remains unaddressed

### Questions
1. I am curious about Triplet Predictor’s potential to generalize to more ambiguous, real-world instructions that may not follow a strict "subject-predicate-object" template. For example, a more complex instruction with abstract/implicit entities and relations might be: "Don't put the sofa facing the TV wall. Instead, try to arrange the sofas and chairs in a circle for chatting, and make sure not to block the path." Have the authors considered using an LLM to paraphrase the original ground-truth template sentences into more free-form structures?

2. Could the authors specify the number of learnable queries ($N_q$) used in the Triplet Predictor? And what happens to performance if the actual number of relations in an instruction exceeds $N_q$?

3. Please provide more details about the Object Retrieval step. Does the retrieval step find the nearest neighbor in the asset database that matches the predicted appearance tokens?

4. Please specify what guidance scale was used for the main experiments. It would also be interesting to know how this scale impacts the trade-off between instruction fidelity and generation quality.

### Soundness
3

### Presentation
3

### Contribution
3
