# Mitigating Hallucination in Vision-Language Model with Depth and Spatial-aware Key-Value Refinement

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 8

## Abstract
Large vision–language models (VLMs) deliver state-of-the-art results on a wide range of multimodal tasks, yet they remain prone to visual hallucinations, producing content that is not grounded in the input image. 
Despite progress with visual supervision, reinforcement learning, and post-hoc attention reshaping, the representational origins of hallucinations remain unclear.
Our study reveals that successful grounding emerges when adjacent visual tokens exhibit coherent alignment, while hallucinations arise when key vectors scatter isotropically, weakening cross-modal attention and blurring object boundaries. 
Building on this insight, we propose Depth and Spatial aware Cache Refinement (DSCR), a lightweight and training-free method that augments the Transformer's key-value (KV) cache with depth cues and 2D spatial proximity. 
DSCR clusters vectors within objects and separates those across surfaces, guiding attention toward relevant regions without any fine-tuning.
Comprehensive evaluations show that DSCR consistently reduces hallucinations, delivering up to 41.6\% accuracy gains across MME, POPE, RePOPE, CHAIR, and a new depth-sensitive benchmark. 
Our findings highlight KV-coherence as a core factor behind hallucinations and demonstrate a practical, model-agnostic solution for enhancing VLM reliability.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes a training-free and lightweight method to mitigate LVLM hallucinations by encoding spatial and depth priors. The core idea is to re-weight the key–value pairs of visual tokens in the cache to enhance grounding, without additional training.

### Strengths
1. This paper is well-motivated and supported by insightful visualizations. The method is novel and achieves consistent improvements on both standard object hallucination benchmarks and attribute/spatial hallucination benchmarks.
2. The paper is overall well written and easy to follow.

### Weaknesses
1. Since both MME and POPE are yes/no questions, it would strengthen the evaluation to extend Table 3 to include detailed CHAIR scores across LVLMs and baselines.
2. The paper provides spatial and depth priors but does not provide enough discussion of related work on grounding visual information.

**Clarification**
1. (Major) In Table 1, Qwen-VL Count row: your method is not the best (+155) but is still bolded. Similarly, for Qwen-VL Poster row, your method (165.99) is not the best. Please correct this.
2. In Figure 1, the Non-Hallucination subfigure is a bit confusing. why are there two “A. Yes” labels?

### Questions
1. I saw the ablation results in Tables 10–14. Could you provide more intuition on why key-only reweighting performs better, and why selecting layers 10–39 (deep layers) works best?
2. To confirm: my understanding is that you apply re-weighting only to the visual token cache, not the text tokens. If that’s correct, it would help to make this explicit throughout sec 2.3.
3. A broader question: do you envision future LVLMs incorporating 3D vision encoders directly to enhance spatial reasoning?

would be happy to raise my score if my concerns are addressed

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies the representational origin of visual hallucinations in vision–language models (VLMs) and proposes a training-free method, Depth and Spatial-aware Cache Refinement (DSCR), that modifies the visual key–value (KV) cache before decoding. The central empirical observation is that successful grounding correlates with coherent alignment of neighboring key vectors, while hallucination correlates with isotropically scattered keys that blur object boundaries. DSCR injects depth cues and 2D spatial proximity into the KV cache by forming a proximity-weighted refinement of keys (optionally values), which clusters vectors within objects and separates vectors across surfaces. The method is model- and query-agnostic, adds negligible overhead, and shows gains on multiple hallucination benchmarks. The paper also introduces a hallucination benchmark to occlusions and similar-depth confounders.

### Strengths
- Clear mechanistic story grounded in the model’s internal representations. The paper ties hallucination to a loss of neighboring-key coherence, and supports this with PCA-based visualizations across layers and attention diagnostics that show increased attention to image tokens when DSCR is applied.
- Simple, training-free intervention with broad applicability. DSCR only modifies the KV cache at inference, without changing weights. The reported overhead is small and the method improves several VLM backbones.
- Consistent gains on hallucination evaluations. Across MME, POPE/RePOPE, CHAIR, and AMBER, the method shows improvements; the depth hallucination mini-benchmark highlights gains in occluded and similar-depth cases.

### Weaknesses
- The link between “neighboring-key coherence” and hallucination is mainly supported by PCA visualizations and attention trends. The paper describes that, in hallucination cases, keys scatter and object boundaries blur, but it does not define a quantitative measure of “neighboring-key similarity” or “key-vector dispersion,” nor does it report large-scale correlations with hallucination/error rates. To strengthen the claim, it is benificial to introduce simple, layer-wise metrics (for example, average cosine similarity with spatial neighbors, or a local PCA explained-variance ratio) and report correlation/predictive power with confidence intervals.

- Comparisons to alternative geometry-injection routes are missing. Since DSCR effectively injects geometric priors, it would be informative to compare against baselines that (i) use a stronger vision encoder that already has geometric priors or (ii) concat depth features with visual tokens as inputs for VLM without KV rewriting. This would clarify the unique benefits of operating in KV cache.

- Evidence for depth–spatial complementarity is limited. In Table 13, depth-only and spatial-only achieve the same total score (645), while the combined setting is only modestly higher (650), with improvements concentrated in one submetric. Figure 4(c,d) validates each component on its targeted subset but does not establish that combining both consistently outperforms either alone across the same benchmarks. More side-by-side results with multiple runs, error bars, and paired tests, and stratified by scene attributes (e.g., occlusion density, depth discontinuities, similar-depth distractors), are helpful to show that the combination truly helps.

- Breadth beyond hallucination-centric tasks is limited. The positive COCO captioning result is helpful, but a broader suite (e.g., additional VQA settings) would better establish that there is no negative transfer to general VL capabilities.

### Questions
- Since DSCR leaves VLM weights frozen, could the lack of adaptation to DSCR-refined keys limit the attainable gains? Have you tried enabling fine-tuning with DSCR active, and if so, did you observe larger improvements or any shift in preferred settings (e.g., Key-only vs. Key+Value, layer ranges)?
- DSCR modifies only the visual branch of VLMs. Can it mitigate hallucinations that arise when the language model misinterprets visual tokens (i.e., unsatisfied vision-language alignment)? Do you have controlled analyses or case studies indicating whether DSCR helps in such language model-driven error modes?
- Semantic richness vs. key scattering. The paper describes hallucination cases where keys scatter and object boundaries blur. Could similar patterns also appear in images with genuinely rich, heterogeneous semantics? After applying DSCR, is there any measurable loss of semantic diversity that could harm performance on VL tasks requiring fine-grained distinctions or rare attributes? 
- Minor presentation. In Figure 1(a), the non-hallucination examples show “X” and “O” even though there is no wrong answer, which is confusing. Also, the “A.” prefix for answers can be misread as an option label. Clarifying the notation would improve readability.

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
The paper introduces Depth and Spatial-aware Cache Refinement (DSCR) — a lightweight, training-free method to suppress visual hallucinations in LVLMs. The authors identify that hallucinations stem from incoherent alignment among key vectors (KVs) within the Transformer’s attention mechanism, which disrupts the cross-modal grounding between visual and textual inputs. DSCR addresses this by refining the Transformer’s KV cache using geometric and spatial priors derived from monocular depth estimation. Through integrating 3D depth cues and 2D spatial proximity, DSCR enforces coherence among tokens representing the same object and separates tokens across different surfaces, thereby improving visual grounding and reducing false object generation. Extensive experiments over multiple benchmarks demonstrate up to 23% accuracy improvement.

### Strengths
1. The paper provides a novel and elegant perspective on the origin of hallucinations by analyzing the internal coherence of key vectors within multimodal Transformers.

2. The DSCR method is training-free, model-agnostic, and computationally efficient.

3. The experimental evaluation is comprehensive and rigorous, and the writing is clear.

### Weaknesses
1. As shown on the right side of Figure 5, while DSCR leverages pre-computed depth maps, its performance inevitably depends on the quality of depth estimation, which may introduce inaccuracies in complex lighting or occlusion scenarios.

2. Why would the misalignment of key vectors weaken the model’s ability to correctly interpret visual inputs? I believe there is no essential connection. I hope the authors can provide further clarification.

3. This paper focuses on optimizing the static reasoning phase, without exploring joint training or adaptive tuning of deep cues. I believe that the misalignment between visual and textual modality features is also an important factor contributing to hallucination. Should additional training be incorporated into DSCR to enhance alignment with textual vectors?

### Questions
1. Does DSCR take the user’s input query into account? If it only considers image features while ignoring textual features, will this lead to performance degradation in handling more complex queries?

2. Why does an isotropic distribution of key vectors tend to produce hallucinations?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper attempts to mitigate hallucination in VLMs by first studying the key vector distribution in model's transformer layers. The authors discovered that key vector distributions exhibit distinct object boundary patterns when models are faithful, while showing blurred object borders when hallucinating. This phenomena motivated the authors to propose DSCR, a training-free method that corrects key and value vectors using depth and spatial similarity maps as guidance. Under DSCR, a separate depth estimator model is used to generate a depth map. Coupled with spatial proximity, weightage is calculated to rebalance the key and value vectors during inference. Experimental results show that DSCR achieves high performance on various hallucination benchmarks. In addition, DSCR is also complementary with other hallucination mitigation methods, further boosting the evaluation performance on multiple open source VLM models. 

The paper contributes to the research community by introducing a new correlation between key vector similarity and hallucination. The training free method can be applied on top of many existing works to further reduce VLM hallucination and improve model performance.

### Strengths
Overall, the paper is well written and the concepts are easy to follow. The authors also offer a new direction to study the underlying cause of hallucination. The proposed DSCR method shows good performance and generalisability. It is an efficient design that is complementary with many existing hallucination mitigation methods. The validity of DSCR design is sufficiently supported by many experiments. The author conducted extensive analysis experiments including attention score value comparison and key vector distribution visualisation for models with and without DSCR correction. The authors also conducted comprehensive ablation experiments such as depth only, spatial only and depth with spatial weightage calculation as shown in appendix.

### Weaknesses
It is insightful for the authors to reveal the different key vector PCA visualisations for hallucinating and non-hallucinating VLM inference scenarios. However, this qualitative analysis can be sensitive to different factors such as object size, object position, difficulty in text query, etc. This slightly undermines the reliability of this discovery and thus the motivation of the method design. The authors could provide more evidence, such as quantitative experimental results, to show that the phenomena is universal, that blurring of object boundaries key vectors is common for different images and query types.

### Questions
1. In figure 3 different shades represent different values. However, why does the distribution pattern look like this? Which 2 image patches are compared?  
2. Based on the ablation experiment table 14, it shows that the best performance is achieved when DSCR is applied to layer 10–39. Why does the final method apply to all layers instead of following the finding from this ablation study?  
3. For the key vector PCA visualisation in figure 1, does the pattern stay the same across different layers? Or is the distinct object boundary only specific to certain attention layers?

### Soundness
4

### Presentation
3

### Contribution
4
