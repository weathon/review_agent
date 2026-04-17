# Constructive Distortion: Improving MLLMs with Attention-Guided Image Warping

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
Multimodal large language models (MLLMs) often miss small details and spatial relations in cluttered scenes, leading to errors in fine-grained perceptual grounding. We introduce AttWarp, a lightweight method that allocates more resolution to query-relevant content while compressing less informative areas, all while pre- serving global context. At test time, AttWarp closes a simple self-correction loop: the MLLM first produces cross-modal attention on the original image, which we use to rectilinearly warp the input and re-run the same frozen model, reallocating resolution toward regions it deems important without changing weights or architecture. This attention-guided warping preserves all original image information but redistributes it non-uniformly, so small objects and subtle relationships become easier for the same model to read while the global layout remains intact. Across nine benchmarks (TextVQA, GQA, DocVQA, POPE, MMMU, MIA- Bench, MMVP, RealWorldQA, BLINK) and four MLLMs (LLaVA, Qwen-VL, InternVL, and InstructBLIP), AttWarp consistently improves accuracy, strengthens compositional reasoning, and reduces hallucinations, outperforming four competitive baselines that manipulate raw images at test time. Together, these results show that attention-guided warping prioritizes information relevant to the query while preserving context, and that the same MLLMs perform better when given such warped inputs. The code and demos are available on the project page: https://dwipddalal.github.io/Attwarp/

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces AttWarp, a lightweight, plug-and-play image warping technique designed to improve the fine-grained perceptual grounding of MLLMs. The method begins by extracting cross-modal attention maps from the MLLM's internal layers based on a given image and query. These attention maps are then used to guide a rectilinear warping of the input image, which reallocates spatial resolution to magnify query-relevant regions while compressing less informative areas. The authors demonstrate consistent performance gains across five diverse benchmarks and four different MLLM architectures, showcasing the method's effectiveness and generalizability.

### Strengths
1. As a model-agnostic enhancement, AttWarp is a practical plug-and-play solution that can be readily applied to improve existing models without retraining.

2. The paper introduces a complete framework, including an iterative version (AttWarp-Chain) for hard cases and a distilled version (AttWarp-Distill) for applications.

3. The work provides compelling ablations and analyses that validate its design choices. A key finding is that the rectilinear nature of the warp preserves the underlying feature distribution of the images, thus avoiding the out-of-distribution issues that can plague other image manipulation techniques.

4. The method shows consistent performance gains across a wide range of benchmarks and MLLM architectures.

### Weaknesses
1. The claim of being "plug-and-play" is slightly weakened by the need to identify the optimal attention layer for each new MLLM architecture (e.g., layer 20 for LLaVA, layer 16 for Qwen-VL). This requires an empirical, model-by-model search, which adds a setup cost for new models.

2. In Error Analysis, the authors state that AttWarp is prone to errors in cases such as  size, hallucination, and misaligned attention. Have the authors attempted any framework modifications to specifically address these failure cases? For example, have they considered introducing a classifier to determine when to apply AttWarp?

### Questions
1. Is there a pattern in the indices of optimal attention layers across different MLLMs? Can we approximately determine which layer is most suitable for AttWarp?

2. Are there methods to mitigate the limitations of AttWarp in cases such as size and hallucination?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper introduces **AttWarp**, a lightweight method for improving multimodal large language models' (MLLMs') fine-grained perceptual grounding in cluttered scenes. AttWarp leverages an MLLM's cross-modal attention to perform rectilinear warping on input images during testing, reallocating resolution to query-relevant areas while preserving global context and image information.  Without modifying model weights or architecture, this attention-guided warping enhances the readability of small objects and subtle relationships. Experiments show consistent accuracy improvements across five benchmarks (TextVQA, GQA, DocVQA, POPE, MMMU) and four MLLMs (LLaVA, Qwen-VL, InternVL, and InstructBLIP), outperforming competitive baselines. These results highlight AttWarp's ability to optimize spatial resolution for query-relevant content while preserving global structure, boosting MLLM performance with warped inputs.

### Strengths
1. **Clarity and Ease of Use**: The paper is well-written, easy to understand, and straightforward to follow. The proposed method, AttWarp, is plug-and-play, delivering significant performance improvements without requiring additional training.  

2. **Intuitive and Novel Approach**: Using attention feedback to enhance the resolution of focus areas is an intuitive yet innovative idea. The method avoids retraining models while achieving substantial gains, making it particularly practical and impactful.  

3. **Comprehensive Validation**: The exploration of AttWarp through two variations—AttWarp-Chain and AttWarp-Distill—effectively demonstrates the feasibility and upper bounds of the method's generalization capabilities. Its success across multiple multimodal language models and benchmarks further validates the robustness and versatility of the approach, making it an interesting and promising contribution.

### Weaknesses
While AttWarp has demonstrated significant improvements across a range of text-centric multimodal tasks (e.g., TextVQA, GQA, DocVQA, POPE, MMMU), it lacks evaluation on visual-centric benchmarks that focus more on fine-grained visual perception. Tasks such as **MMVP**, **BLINK**, **RealWorldQA**, and **MIA**, which emphasize nuanced visual grounding and object-level understanding, are particularly relevant for showcasing the strengths of AttWarp's attention-guided resolution reallocation.  

Including these visual-centric evaluations could illustrate its potential for enhancing visual perception capabilities further, as the method is well-suited to improve the detection of subtle details and spatial relationships that are critical for such tasks. Without this, the generalizability and impact of AttWarp on visually demanding applications remain underexplored. Evaluating the method on these benchmarks would provide a more comprehensive picture of its capabilities and further highlight its benefits.

### Questions
Do you think applying reinforcement learning (RL) to AttWarp could further enhance its capabilities, using MLLMs to validate the effectiveness of perturbed images in solving queries and providing reward feedback?

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
3

### Summary
This paper proposes AttWarp, a plug-and-play, attention-guided image warping method that improves fine-grained perception in multimodal large language models (MLLMs) without modifying their architecture or parameters. The method reallocates spatial resolution toward query-relevant regions using the model’s own cross-modal attention, yielding consistent gains across multiple benchmarks and MLLM backbones.

### Strengths
- The idea of leveraging model attention to reshape the input space rather than internal representations is conceptually elegant and complementary to existing attention-tuning methods.

- The method is tested on five diverse benchmarks and four architectures, showing consistent improvements with detailed ablations.

- The approach is simple, lightweight, and does not require retraining.

- The paper includes ablations on attention quality, warping stability, and distributional integrity, which strengthen its credibility.

### Weaknesses
- The contribution of this paper feels more like a clever engineering refinement than a fundamentally new paradigm.

- The method assumes reliable attention maps; performance may degrade under noisy or misaligned attention, but this limitation is only briefly mentioned.

- The paper’s justification for why rectilinear warping improves reasoning remains empirical and lacks a more formal analysis of perceptual geometry or attention dynamics.

### Questions
please see weakness

### Soundness
3

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
This paper introduces Constructive Distortion, a training strategy for large vision-language models designed to enhance robustness and fine-grained understanding through targeted visual perturbations. Instead of random noise or masking, the approach applies semantically constructive distortions—guided transformations (e.g., spatial deformation, contrast warping) that preserve semantics while challenging the model’s visual encoder. The method aims to improve generalization to distorted or out-of-distribution visual inputs without sacrificing in-distribution performance. Experiments on benchmarks such as MM-Vet, MME, and LLaVA-Bench demonstrate consistent improvements, particularly under degraded or perturbed visual conditions.

### Strengths
- Novel concept: Shifts from destructive to constructive data augmentation, promoting robustness while maintaining semantic fidelity.

- Strong empirical results across diverse LVLMs and benchmarks, with detailed ablations on different distortion types and intensities.

- Practical contribution: The approach is plug-and-play and compatible with existing LVLM training pipelines.

- Clear motivation and presentation: The paper is well-written and the concept of “constructive” perturbation is intuitively appealing.

### Weaknesses
Lack of comparison with visual token compression methods. The paper does not contextualize its approach relative to recent efficient LVLM frameworks that also modify the visual representation process. Works such as [1] PVC: Progressive Visual Token Compression (Yang et al., 2024), [2] Efficient Large Multi-Modal Models via Visual Context Compression (Chen et al., NeurIPS 2024), and [3] An Image Is Worth 1/2 Tokens After Layer 2 (Chen et al., ECCV 2024) explore representation simplification and robustness trade-offs at the token level. Including comparisons or discussion would clarify whether constructive distortions yield complementary or competing benefits.

[1] Yang C, Dong X, Zhu X, et al. PVC: Progressive Visual Token Compression for Unified Image and Video Processing in Large Vision- 

[2] Chen J, Ye L, He J, et al. Efficient large multi-modal models via visual context compression[J]. Advances in Neural Information Processing Systems, 2024, 37: 73986-74007. 

[3] Chen L, Zhao H, Liu T, et al. An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large vision-language models[C]//European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2024: 19-35.

### Questions
Could you please provide more comprehensive evaluation (e.g., VQAv2) and (attention-based) token compression baseline?

### Soundness
3

### Presentation
2

### Contribution
2
