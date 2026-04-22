# MARS - A Foundational Map Auto-Regressor

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Map generation tasks feature extensive non-structural *vectorized data* (e.g., points, polylines, and polygons) and thus pose significant challenges to common pixel-wise generative models. Conventional approaches use multiple stages, first segmenting these features at the pixel level and then performing vectorized post-processing, with errors and complexity compounding at each stage. Motivated by the recent success of auto-regressive language modeling, we propose the first map foundation model, named Map Auto-Regressor (MARS), that is capable of generating both multi-polyline road networks and polygon buildings in a unified manner. For training MARS, we collected to our knowledge the largest multi-class map extraction dataset totaling 3.4M examples, which we call MAP-3M. Across four road and building datasets, MARS outperforms or matches the performance of multistage baselines. Additionally, we develop a ``Chat with MARS'' capability that enables interactive human-in-the-loop map generation and correction, supported by the auto-regressive nature of our end-to-end approach.
We release our MAP-3M dataset and project demo page at (1) https://huggingface.co/datasets/bag-lab/MAP-3M and (2) https://huggingface.co/spaces/bag-lab/MARS, respectively.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces MARS, the first foundational model for vectorized map generation that unifies the creation of both road networks and building polygons within a single end-to-end framework. MARS treats vectorized map primitives as a language and employs a sequence-to-sequence auto-regressive Transformer to directly generate map elements without intermediate steps. The model is trained on a newly curated large-scale dataset, MAP-3M, which contains three million high-quality, multi-class map annotations—10× larger and 100× broader in coverage than existing benchmarks. Extensive experiments demonstrate that MARS outperforms prior rasterization-based and hybrid approaches while maintaining scalability and generalization. Additionally, the authors propose “Chat with MARS,” an interactive human-in-the-loop system that enables prompt-based map generation and correction.

### Strengths
1. MARS  is the first foundational auto-regressive model for vectorized map generation, unifying both road networks and building polygons within a single end-to-end framework. The proposed map-to-sequence representation elegantly converts geometric primitives into a sequential language-like form, enabling map generation to benefit from advances in large-scale sequence modeling.
2. The work not only presents a new modeling paradigm but also releases MAP-3M, the largest multi-class map dataset to date, supporting robust training and reproducibility. Extensive experiments demonstrate strong generalization and consistent performance gains over prior rasterization-based and hybrid approaches.
3. The “Chat with MARS” module creatively leverages prompt-following capabilities of the auto-regressive model, introducing a novel human-in-the-loop mechanism for real-time map editing and correction. This interactivity significantly enhances the paper’s applicability to real-world geospatial workflows.

### Weaknesses
While the proposed map-to-sequence formulation is elegant, it inevitably flattens geometric structures into linear token sequences. The model relies mainly on data-driven regularities rather than structural constraints.

### Questions
While MARS directly generates vectorized map elements through an end-to-end sequence model, it remains unclear how the proposed approach ensures geometric or topological boundary consistency between adjacent objects. Does the model incorporate any explicit mechanism or loss to preserve boundary alignment, or is this consistency purely learned implicitly from data?

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
3

### Summary
This paper introduces a novel foundation model for map generation, termed MARS. The proposed approach employs a ViT backbone to extract visual features and an autoregressive transformer to generate map sequences, including points, polylines, and polygons. In addition, the authors present a multi-class map dataset to support the foundation model and benchmarking. The paper also introduces an interactive feature, Chat-with-MARS, which allows human-in-the-loop map generation and correction.

### Strengths
1. The paper presents MARS, a novel foundation model for vectorized map generation, which is both original and technically sound.
2. The MARS framework demonstrates strong performance.
3. The proposed Chat-with-MARS functionality is innovative and adds practical value by enabling interactive map generation.

### Weaknesses
1. While the authors provide detailed definitions of different map objects, autoregressive generation typically relies on a well-defined sequence order. However, the ordering of map objects in the proposed pipeline is not clearly explained.
2. The use of a ViT-based visual backbone and an autoregressive transformer could make the model computationally expensive. The paper lacks discussion or analysis regarding computational efficiency or runtime performance.
3. As a foundational model for map generation, the ablation studies are insufficient. The authors do not adequately examine the effects of architectural choices (e.g., different vision backbones or decoder designs) or training strategies.
4. Although the Chat-with-MARS feature is compelling, the paper would benefit from additional demonstrations—such as a short demo video or a GUI prototype—to better illustrate its capabilities and user interaction.

### Questions
Please refer to the weaknesses.

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
2

### Summary
This paper proposes MARS, an auto-regressive foundational model for end-to-end vectorized map generation that unifies the prediction of roads (as multi-polylines) and buildings (as polygons) without relying on post-processing heuristics. The authors introduce a novel map-to-sequence representation, a large-scale multi-class dataset (MAP-3M), and a human-in-the-loop interaction paradigm called “Chat with MARS.” Sounds good.

### Strengths
1. This research topic is new to me.
2. The writing is good.
3. Interactive capability is intersting.

### Weaknesses
1. No ablation on the impact of the stroke-based decomposition vs. alternative graph traversal or serialization strategies
2. Limited analysis of failure modes (e.g., complex intersections, occluded structures).
3. The “Chat with MARS” evaluation is synthetic (uses GT points as prompts); real-user studies or robustness to noisy clicks would strengthen claims.

### Questions
1. Can the dataset be generalized to global regions with different road/building styles?

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
This paper proposes MARS, an end-to-end map auto-regressor for vectorized map generation. The key idea is to treat maps as a foreign language: all vector primitives (points, polylines/roads, polygons/buildings) are serialized via a map-to-sequence procedure, and then a vision encoder + autoregressive transformer decodes the whole map token by token. On top of that, the authors introduce “Chat with MARS”, an interactive, human-in-the-loop decoding mode (start-/mid-/end-of-sequence interventions) that can fix missing or drifting map objects with 1–2 clicks. To support this, the paper also curates MAP-3M, which is the largest aerial-image + multi-class (roads + buildings) dataset so far (∼3M tiles, wide US coverage, NAIP imagery + Overture/OSM labels). Experiments on Cityscale, SpaceNet, and AICrowd show that the unified, class-agnostic, autoregressive model is competitive with or close to SOTA methods that are narrowly specialized.

### Strengths
* Framing vectorized map generation as AR sequence modeling is elegant: one decoder, one vocabulary, one loss, multiple geometry types. This is nicer than the usual “segmentation → heuristic post-processing” pipeline.
* Even though the current demos look a bit toy-ish, the idea of AR decoding + teacher forcing ⇒ promptability is solid, and the paper shows three concrete intervention modes (SOS/MOS/EOS) with quantitative gains. This is, in my view, the most future-facing part.
* A 3M-tile, dual-class, US-wide, reasonably high-res dataset that already comes vectorized would be very valuable to the community, especially for people doing OSM updating, change detection, and AD map pretraining. This alone can justify publication if it’s really as large, clean, and diverse as stated.
* On Cityscale / SpaceNet they get very reasonable TOPO F1s, sometimes higher recall than road-specialized models, which is non-trivial for a generic AR model.

### Weaknesses
* Right now the two main claims — “we have a 3M, dual-class, high-quality dataset” and “we have a working general AR map model” — cannot be verified. In the last year I have seen many papers claim “large-scale dataset” and the final release was (i) much smaller, (ii) missing one of the modalities, or (iii) under a restrictive license. So the impact is contingent on actual release.
* The whole “maps as a sequence” story stands or falls with the correctness of the stroke decomposition at intersections, roundabouts, and T-junctions. If this step introduces topology errors or weird ordering, the model will happily learn those artifacts. This should be stress-tested more.
* The 1-click / 2-click “Chat with MARS” is impressive, but in the current form it is still a single-object recovery tool. For real OSM editing or AD map maintenance, users will want multi-round constraints (e.g., “keep these 3 roads, regenerate everything north of x”), not only point hints.
* I suggest citing some downstream application papers like P-MapNet: Far-seeing map generator enhanced by both SDMap and HDMap priors.

### Questions
* Will MAP-3M be truly public, with both imagery and vector labels, or only labels? Are there licensing constraints that will make the “3M” number lower in practice?

### Soundness
3

### Presentation
3

### Contribution
3
