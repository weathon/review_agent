# Grasp Any Region: Towards Precise, Contextual Pixel Understanding for Multimodal LLMs

- Decision: Accept (Poster)
- Scores: 4, 6, 4, 6

## Abstract
While Multimodal Large Language Models (MLLMs) excel at holistic understanding, they struggle with the dense world, i.e., complex scenes requiring fine-grained analysis of intricate details and object inter-relationships. Region-level MLLMs have been a promising step. However, previous attempts are generally optimized to understand given regions in isolation, neglecting crucial global contexts. To address this, we introduce Grasp Any Region (GAR) for comprehensive region-level visual understanding. Empowered by an effective RoI-aligned feature replay technique, GAR supports (1) precise perception by leveraging necessary global contexts, and (2) modeling interactions between multiple prompts. Together, it then naturally achieves (3) advanced compositional reasoning to answer specific free-form questions about any region, shifting the paradigm from passive description to active dialogue. Moreover, we construct GARBench, which not only provides a more accurate evaluation of single-region comprehension, but also, more importantly, measures interactions and complex reasoning across multiple regions. Empirically, GAR-1B not only maintains the state-of-the-art captioning capabilities, e.g.,
outperforming DAM-3B +4.5 on DLC-Bench, but also excels at modeling relationships between multiple prompts with advanced comprehension capabilities, even surpassing InternVL3-78B on GARBench-VQA. More importantly, our zero-shot
GAR-8B even outperforms in-domain VideoRefer-7B on VideoRefer-BenchQ, indicating its strong comprehension capabilities can be easily transferred to videos. Code and data will be released to the community.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes Grasp Any Region (GAR), a region-level MLLM that aims to combine precise region perception with access to global scene context, support interactions among multiple region prompts, and enable region-centric compositional reasoning in a single-turn conversation. The core technical idea is an RoI-aligned feature replay pipeline: the model encodes the full image with prompt masks to obtain a global, context-aware feature map, then re-pools prompt-specific features with RoI-Align so the LLM receives both prompt-aware global tokens and high-fidelity local tokens. The paper also proposes a dataset of 2.5M-scale assembled in two rounds to strengthen fine-grained recognition and relational understanding, and GARBench, a benchmark suite with a captioning protocol for multi-prompt relations and VQA that separates basic perception from reasoning such as position, non-entity recognition, and relations. The results section reports strong numbers on DLC-Bench and competitive results on GARBench, with qualitative examples

### Strengths
– The data contribution is good and will likely be useful to the community. The paper constructs a large training set that starts from Describe-Anything, augments with fine-grained category supervision, and then adds relation-aware descriptions and QAs from PSG using an LLM merger. This staged pipeline is clearly described with concrete numbers and roles for each round, and it targets the specific capability gaps claimed by the model, like fine-grained region recognition and multi-region relations. The ablation helps understand what each round adds. 

– The benchmark contribution is also meaningful. GARBench decomposes evaluation into captioning of relations across prompts and a VQA protocol that cleanly separates perception (color, shape, material, texture) from reasoning (position, non-entity recognition, relations). The paper also uses multiple-choice designs in places where open-ended judging is brittle, and it includes tasks that force models to use global context, like mirror reflections and grid positions. Together, this looks like a well-scoped benchmark for region-aware reasoning.

### Weaknesses
- The major weakness is that the paper has not positioned its novelty in the literature with enough precision. The paper frames the novelty around inclusion of global context for precise perception, interactions between multiple prompts, and reasoning capabilities, but the related works section states that “previous approaches only support a single visual prompt, and often neglect global context”(L123). This statement is too strong, given that prior works referred in the same section (GPT4RoI and GLaMM) already support multiple regions and attempt to preserve global context, so the current phrasing risks overstating the novelty. 

- Regarding the specific point about inclusion of global context would benefit from a clear technical comparison to GPT4RoI‑style designs. 
GPT4RoI encodes the entire image once to obtain a global visual representation, then extracts RoI‑aligned features for user‑specified regions (via boxes) and interleaves those region features with the text sequence so the LLM can answer region‑centric queries while still using whole‑image context, it supports multiple regions and treats each region as an additional visual cue alongside the global image feature. The paper follows essentially the same recipe - one global image pass + RoI‑aligned region features fed to the LLM for multi‑region reasoning, with the only material differences being that (i) it takes masks rather than boxes and injects a learned mask embedding into the vision backbone before computing the global feature map (so the map itself is prompt‑aware), and (ii) instead of pooling each RoI down to a single vector, it replays a small set of RoI‑aligned tokens per region taken from that prompt‑aware map. In other words, apart from mask conditioning and unpooled multi‑token RoI replay, the overall architecture and goal, joining global context with local RoI features to enable fine‑grained, multi‑prompt understanding, are very similar.

- Comparisons currently center on DAM, but GPT4RoI is the key reference for multi-region prompting and should anchor the novelty discussion. A simple ablation in a GPT4RoI style would make the technical contribution clear, keeping the LLM and data fixed. This ablation would show whether the gains come from the architecture rather than data or evaluation choices.

- The only clear novelty that remains well supported is the region‑specific reasoning emphasis, which is indeed useful since even strong models like Gemini-2.5-Pro still fail on spatial reasoning. GARBench’s design and the strong results there are aligned with this direction, yet the novelty narrative should reflect that emphasis more, rather than attributing novelty to global context or single‑prompt limitations in prior work.

### Questions
Could you precisely differentiate GAR from GPT4RoI-style architectures beyond the use of RoI-Align? Please explain whether GAR’s mask changes the vision encoder’s computation to create prompt-aware global features, how this differs from methods that inject regions only at the language side after a prompt-agnostic image encoding, and what effect this has on using global context.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a vision-language model that enhances region-level visual understanding by combining global context with fine-grained local detail. Using a RoI-aligned feature replay mechanism, GAR processes full-image features while focusing on specific regions, enabling accurate reasoning about multiple prompts (e.g., spatial and relational understanding). The authors also introduce GARBench, a new benchmark that evaluates both single-region captioning and multi-region reasoning. Experiments show GAR surpasses previous models like DAM-3B, Ferret, and even larger models such as InternVL3-78B on both captioning and VQA tasks.

### Strengths
+ Clear motivation and reasonable architectural solution:
GAR tackles the important problem to understand regional information in the vision-language model, and proposes the reasonable solution with a RoI-aligned feature replay strategy. This design allows global context retention while focusing on high-resolution local features.

+ Focus on inter-region relationships:
Unlike prior works, which primarily handle single-object or localized descriptions, GAR explicitly models multi-prompt interactions such as spatial relations and entity reflection recognition. This makes it suitable for compositional reasoning tasks previously underexplored by region-level MLLMs

+ Comprehensive benchmark contribution (GARBench):
I appreciate the GAR Benchmark contribution. It evaluates not only region captioning but also relational VQA, multi-prompt reasoning, and non-entity discrimination—offering a more complete view of region-level comprehension capabilities

### Weaknesses
- Evaluation dependence on LLM judges:
The authors rely on LLM-based evaluators (e.g., GPT-4) for qualitative assessment. This introduces potential bias due to stylistic or verbosity differences among models. Cross-validation with human ratings or standardized metrics would strengthen the claims.

- Insufficient data transparency:
The training pipeline involves multiple stages—seed captioner, LLM merger, and relational caption generation—but lacks detail on data deduplication and leakage checks. Since datasets like COCO or PSG may overlap with benchmarks, more rigorous data hygiene reporting is necessary

- Marginal novelty compared to related works:
The combination of RoI-Align with contextual replay is technically effective but not radically novel; similar ideas of multi-scale or global-local fusion appear in Ferret and GPT4RoI.

- Missing references: The paper misses recent region-understanding and grounding works such as [ref1] and [ref2], and would benefit from a more detailed comparison and discussion of how GAR differs from these grounded vision-language models.

[ref1] Toward Interactive Regional Understanding in Vision-Large Language Models, NAACL 2024

[ref2] Groma: Grounded Multimodal Assistant, ECCV 2024

### Questions
- Scalability to the number of regions:
The paper shows results for up to a few regions. What happens when the number of regions scales to tens or hundreds—does reasoning complexity or memory usage grow linearly?

- Judge consistency: How LLM judges' results are consistent? Are they consistent enough for multiple runs?

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
4

### Summary
This paper aims to enhance the capability of Vision-Language Models (VLMs) in understanding dense visual scenes, particularly in capturing fine-grained local details and the relationships among multiple local regions. The authors propose the GAR method, whose core lies in the ROI-aligned feature replay technique, which enhances local visual feature representation through ROI alignment during the visual feature extraction process. Furthermore, the paper introduces a large-scale GAR-2.5M training dataset and GARBench benchmark for training and evaluating the GAR-1B/8B models, respectively. Experimental results across multiple datasets demonstrate the superior performance of the proposed GAR-1B and GAR-8B models.

### Strengths
- Understanding fine-grained details and object inter-relationships is critical for real-world applications of VLMs. This paper provides important contributions in training dataset, evaluation benchmark and model architechture. 
- The paper is clearly written and easy to read.

### Weaknesses
- Understanding the dense world is an important capability. However, the proposed GAR task represents only a specific task formulation within this direction. In particular, the paper constrains the use of masks as indicators of objects, which may introduce bias when evaluating a model’s true understanding of dense visual scenes. The compared models might simply lack the ability to interpret masks, rather than being genuinely deficient in understanding local details and relationships.
- The proposed method in this paper is designed for a specific task type, incorporating a task-specific module — the ROI-Aligned Feature Replay Module. Moreover, the corresponding training dataset GAR-2.5M is constructed using the same methodology as GARBench, which weakens the assessment of the method’s effectiveness and generalizability. In addition to comparisons with mainstream open-source models such as Qwen2.5-VL and InternVL3 on GARBench, evaluations on general benchmarks would further strengthen the validation of the proposed method and dataset in terms of their effectiveness and general applicability.
- The ablation study in this paper is not sufficiently comprehensive. In particular, it would be beneficial to include an additional variant that uses only global patch embeddings and mask embeddings, without the ROI-Aligned Feature Replay Module, and train this variant on the GAR-2.5M dataset. This result would help highlight the effectiveness of the ROI-Aligned Feature Replay Module. Furthermore, comparisons with the pretrained model, namely the PerceptionLM baseline, should also be added across all datasets to provide a more complete evaluation.
- The implementation details of GAR are not clearly described. In particular, how are the RoI-Aligned features integrated with the global context features and question tokens?

### Questions
See Weaknesses.

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
4

### Summary
This work addresses the limitation of existing MLLMs in fine-grained regional understanding, global context integration, and multi-prompt interaction. It proposes GAR, a model empowered by RoI-aligned feature replay to preserve global context while capturing local details, and GARBench, a benchmark for evaluating single-region comprehension, multi-prompt interaction, and compositional reasoning. GAR outperforms state-of-the-art models on multiple benchmarks and achieves strong zero-shot transfer to video tasks.

### Strengths
1. Proposes an innovative RoI-aligned feature replay technique that seamlessly integrates global context and local details, solving a key limitation of existing region-level MLLMs.
2. Develops GARBench, the first benchmark to systematically evaluate multi-prompt interaction and compositional reasoning, filling an evaluation gap.
3. Demonstrates strong generalization, including zero-shot transfer to video tasks, highlighting the model’s practical utility.
Conducts comprehensive experiments (ablation, cross-benchmark, qualitative analysis) that rigorously validate the method’s effectiveness.

### Weaknesses
1. RoI-Align’s context binding validity could benefit from additional validation, as the paper does not include targeted experiments for complex scenes where misbinding irrelevant context might occur. Additionally, the ablations in Table 8 do not test whether shielding irrelevant global regions impacts the accuracy of extracted local features.
2. The Grasp Any Region-2.5M dataset and GARBench do not provide clear annotations or statistics on which tasks specifically depend on global context. This may lead to potential over-reliance on global context during training, and it remains unclear if GAR’s performance gains stem from global-local fusion or merely improved local feature capability.  
3. GARBench appears statistically limited for drawing robust conclusions, with only 204 samples for GARBench-Cap and 424 samples for GARBench-VQA—sizes that may not fully account for result variability.  
4. RoI-Align’s necessity could be further strengthened, as no comparison is provided in the ablation study (e.g., Table 8) with a simpler alternative: directly cropping the local image region and feeding it through the ViT encoder to extract features. It would be helpful to clarify how RoI-Align outperforms this more straightforward approach.

### Questions
RoI-Align’s necessity could be further strengthened, as no comparison is provided in the ablation study (e.g., Table 8) with a simpler alternative: directly cropping the local image region and feeding it through the ViT encoder to extract features. It would be helpful to clarify how RoI-Align outperforms this more straightforward approach.

### Soundness
4

### Presentation
4

### Contribution
4
