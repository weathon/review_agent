# GPT-IMAGE-EDIT-1.5M: A Million-Scale, GPT-Generated Image Dataset

- Avg Score: 5.50
- Decision: Reject
- Scores: 6, 2, 6, 8

## Abstract
Recent advancements in proprietary multimodal models such as GPT-Image-1 have set new standards for high-fidelity, instruction-guided image editing. However, their closed-source nature restricts open research and reproducibility. To bridge this gap, we introduce GPT-IMAGE-EDIT-1.5M, a publicly available dataset comprising over 1.5 million high-quality editing triplets systematically unified from OmniEdit, HQEdit, and UltraEdit. Our data-curation pipeline leverages output regeneration and instruction rewriting to significantly enhance instruction following (IF) and perceptual quality (PQ), while relying only on simple geometric and instruction-level filters. We benchmark three MMDiT diffusion architectures—SD3 InstructPix2Pix (channel-wise conditioning), Flux with SigLIP (token-wise conditioning), and FluxKontext (token-wise conditioning)—to analyze their robustness against IP degradation. Our results indicate that token-wise conditioning methods consistently outperform channel-wise conditioning. To ensure evaluation transparency, we specify when results involve thinking-rewritten prompts to avoid potential ambiguity. Moreover, we examine text encoders within a common frozen-encoder scenario, demonstrating that T5 embeddings consistently meet or exceed multimodal large language model (MLLM) embeddings, particularly with lengthier prompts. Simple linear or query-based integration methods, however, offer limited improvements, indicating deeper cross-modal fusion methods may be necessary. Fine-tuning FluxKontext on GPT-IMAGE-EDIT-1.5M achieves open-source performance competitive with GPT-Image-1 (7.66 @ GEdit-EN and 3.90 @ ImgEdit-Full, with thinking-rewritten prompts; 8.97 @ Complex-Edit). Our findings highlight critical interactions among instruction complexity, semantic alignment, and identity preservation, informing future directions in open-source image editing.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper (i) unifies and regenerates OmniEdit, HQEdit, and UltraEdit into GPT-IMAGE-EDIT-1.5M (≥1.5M triplets) via a concise pipeline that uses GPT-Image-1 for output regeneration and selective instruction rewriting to improve instruction following (IF) and perceptual quality (PQ) while retaining identity-preservation (IP) challenges (Fig. 2, §3.1). It (ii) performs a controlled comparison of conditioning mechanisms—channel-wise (SD3 InstructPix2Pix) vs token-wise (Flux+SigLIP, FluxKontext)—and (iii) studies frozen text encoders (T5 vs MLLM embeddings and shallow fusion). Fine-tuning FluxKontext on the curated data achieves 7.66@GEdit-EN with thinking-rewritten prompts, 3.90@ImgEdit-Full, and 8.97@Complex-Edit, approaching proprietary GPT-Image-1 on several metrics (Tables 1–3). Key empirical findings: token-wise conditioning > channel-wise, T5 ≥ MLLM embeddings under frozen encoders, and two-stage data refinements (regen + rewrite) yield consistent gains (Tables 4–5, 7–9; qualitative Fig. 4).

### Strengths
Data at scale, curated for alignment while keeping IP hard. The pipeline intentionally preserves identity challenges instead of filtering them out, reflecting real-world edits (pp.2–4). The process is simple (regen + targeted rewrite) yet empirically effective (Table 9). 
Decisive architectural insight. Across backbones, token-wise conditioning consistently improves IF/IP/PQ vs channel-wise (Table 4; qualitative Fig. 4). This is a clear, generalizable lesson for editor design. 
Encoder study under frozen setting. T5 is a competitive default; shallow MLLM fusions add little; MetaQuery underperforms (Table 5, Table 8). This clarifies common practitioner questions. 
Transparent evaluation. Side-by-side "original vs thinking-rewritten prompts" with explicit marker (†) reduces evaluation ambiguity (Tables 1–2; Appendix C). 
Strong results, broad coverage. Competitive with GPT-Image-1 on multiple axes (Tables 1–3), with fine-grained category trends (Tables 10–13) and rich visuals (Figs. 5–8).

### Weaknesses
Proprietary dependence in curation/eval. The dataset relies on GPT-Image-1 for regeneration and GPT-4o/GPT-5 for instruction rewriting and evaluation-time rewrites (Fig. 2; §3.1; App. A–C). This may bias distributions toward proprietary model priors and limits open-stack reproducibility; an ablation using only open generators/rewriters would strengthen claims. 
Licensing & rights clarity. The paper states release intent but lacks explicit data licensing / usage rights for regenerated content and rewritten prompts; the metadata schema is detailed (App. B.6) but legal terms are not spelled out. 
Causality of gains vs prompt rewriting. While transparency is good, some headline improvements (e.g., GEdit 7.12→7.66) hinge on thinking-rewritten prompts (Table 1). A controlled analysis showing per-category deltas attributable to data vs prompt rewriting would clarify where progress truly comes from. 
Metric scope. Benchmarks rely on MLLM-based scoring; adding reference-free IQA and OCR/identity metrics where relevant could triangle the results (Sec. 5 acknowledges sensitivity). 
Compute/reporting. Training details are provided (App. E), but exact wall-clock / GPU-days per model and dataset pass are not summarized in the main text; cost-quality curves would help practitioners plan.

### Questions
Open-stack replication: What is the performance delta if regeneration and rewriting use only open models (e.g., FluxKontext for regen + an open LLM for rewrites)? Can you report a mini-study mirroring Table 9? 
Licensing: Please specify dataset license(s) for regenerated images and rewritten prompts, redistribution permissions, and any restrictions for commercial use. (Metadata schema exists, but terms are not explicit.) 
Attribution of gains: For Table 1, can you provide per-category Δ from data curation vs prompt rewriting to quantify their separate contributions? 
Identity & text robustness: Complex-Edit shows strong IP (Table 3), but Sec. 5 notes text rendering and fine facial IP remain hard. Could you share failure taxonomies and a small stress test (e.g., typography, small faces)? 
Fusion depth: Since shallow MLLM fusion underperforms (Table 5), do you plan to try deep cross-modal fusion (e.g., shared layers or gated co-attention) while keeping encoders frozen? Any preliminary results? 

Additional Feedback (actionable)

Release an open-stack variant of the pipeline and re-run a subset to quantify dependence on proprietary models.
Add cost-vs-quality plots and compute budget tables per architecture.
Provide license text and a datasheet (sources, rights, filters, intended use, known risks).
Include identity/text micro-benchmarks and report confidence intervals.
Publish the full set of thinking-rewritten prompts with categories and ambiguity tags.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a data-centric pipeline that builds a ~1.5M triplet dataset for instruction-guided image editing by generating images with DALL-E, synthesizing with GPT-Image-1, and rewriting a subset of prompts with GPT-4o, then fine-tuning open-source diffusion models (e.g., SD3 InstructPix2Pix); it also compares conditioning strategies (channel-wise vs. token-wise) and text encoders (frozen T5 vs. Qwen2.5-VL) across GEdit-EN-full, ImgEdit-Full, and Complex-Edit, and reports that fine-tuning on the curated triplets yields 7.66 on GEdit-EN-full, an absolute +0.17 over the GPT-Image-1 baseline.

### Strengths
* The end-to-end data curation and training flow is easy to follow, with a helpful schematic that makes the method accessible to non-experts.
* Conditioning types (channel- vs. token-wise) and text encoders (frozen T5 vs. Qwen2.5-VL) are compared on GEdit-EN and ImgEdit (Table 5).
* Fine-tuning on the curated triplets reaches 7.66 on GEdit-EN-full, an absolute +0.17 over the GPT-Image-1 baseline.

### Weaknesses
**1. Minimal originality.** Beyond stitching DALL-E, GPT-Image-1, and GPT-4o into a data pipeline, the paper introduces no mechanism to detect or correct automatically edited failures or caption-rewrite hallucinations. This omission is consequential: despite distilling on ~1.5M triplets, the fine-tuned model shows only a marginal +0.17 on GEdit-EN-full and no compelling improvements on other benchmarks (see Tables 2, 3, and 4), consistent with noisy, unvetted supervision. A concrete, learnable metric to auto-filter or down-weight bad triplets and mismatched captions would directly boost editing performance, and black-box safeguards remain feasible in a closed-source setting, including early-timestep candidate screening [1], preference-based filtering with pairwise ranking [2], and image-grounded verification to constrain GPT-4o rewrites and reduce hallucinations [3].

**2. Unfair comparison and unclear versions of closed-source models.** Baseline sets differ across GEdit-EN-full, ImgEdit-Full, and Complex-Edit, raising cherry-picking concerns. Credible evaluation requires a consistent comparison matrix across all benchmarks, explicit identification of closed-model versions, tiers, inference settings, and access dates, and inclusion of stronger frontier baselines such as Seedream 3.0 and Lumina-Image 2.0, which surpass GPT-Image-1 [High] by 2–3 points on DPG (see Qwen-Image, Table 3). Fairness further benefits from reporting parameter counts and latency budgets and from a version-sensitivity ablation.

**3. Incremental study related to text conditions.** Without mechanism-level insight, the conditioning analysis remains descriptive rather than explanatory. Stronger evidence would come from token-drop tests and self-/cross-attention quality of fine-tuned diffusion models. The text-encoder exploration is narrow: beyond Qwen2.5-7B-VL, comparisons should include lighter MLLMs such as FastVLM [4] and scaled variants like Qwen2.5-72B-VL, with frozen versus fine-tuned settings reported under matched compute. If a language-specific T5 outperforms vision-language encoders, additional baselines using LLM-based text embeddings adapted for retrieval or instruction alignment are necessary to rule out artifacts of an under-powered setup.

**4. Unclear limitation of GPT-Image-1.** Scene-text hallucination and corruption are common when closed models perform editing, yet the evaluation lacks a targeted metric. An OCR-based protocol, as in Qwen-Image, can quantify pre- and post-edit text consistency via character- or word-level errors, alongside representative failure cases where text appears, disappears, or changes semantics. Where reliable scene-text preservation is not achievable, the limitation should be stated clearly and treated as a risk.

[1] Early Timestep Zero-Shot Candidate Selection for Instruction-Guided Image Editing, ICCV 2025.

[2] Pick-a-Pic: An Open Dataset of User Preferences for Text-to-Image Generation, NeurIPS 2023.

[3] Mitigating Object Hallucination in Large Vision-Language Models via Image-Grounded Guidance, ICML 2025.

[4] FastVLM: Efficient Vision Encoding for Vision Language Models, CVPR 2025.

### Questions
Q1. To what extent is the choice of a specific closed-source stack, centered on GPT-Image-1, justified beyond empirical convenience? Could the same pipeline be instantiated with ensembles including Gemini 2.5 Flash or with open-source editors at comparable quality, and what evidence supports restricting the design to closed-source components?

Q2. Given the reliance on MLLM-centric evaluation, could the study include experiments that incorporate alternative task-suitable metrics from MIG-Bench [5] and CMMD [6], as well as a VLM-based image–text similarity assessment such as CLIP-style scoring? In addition, would you consider evaluating under established protocols used by [1] and the InstructPix2Pix benchmark suite to reduce metric bias and to validate conclusions across multiple evaluation regimes?

Q3. Beyond aggregate performance, could the analysis include evidence of instruction grounding that demonstrates the fine-tuned editor actually references prompt components during editing? For example, keyword-level cross-attention or attribution maps, token-drop sensitivity curves, counterfactual prompt tests with synonym replacement, and region-wise edit-versus-background preservation metrics would make the claim concrete.

Q4. To complement the current analysis related to text encoders, would you extend comparisons beyond Qwen2.5-7B-VL to include additional MLLMs [4] and language-only embeddings, such as larger Qwen2.5 variants, with both frozen and fine-tuned settings under matched compute so that the conclusions in Table 5 are robust across architectures and scales?

[5] MIGC: Multi-Instance Generation Controller for Text-to-Image Synthesis, CVPR 2024. 

[6] Rethinking FID: Towards a Better Evaluation Metric for Image Generation, CVPR 2024.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper aims to bridge the gap between closed-source multimodal image editing models (e.g., GPT-Image-1) and open-source research by introducing GPT-IMAGE-EDIT-1.5M—a publicly available dataset comprising over 1.5 million high-quality image editing triplets (original image, editing instruction, target image). The dataset integrates data from OmniEdit, HQEdit, and UltraEdit, and further enhances instruction following and perceptual quality through output regeneration and instruction rewriting, while deliberately preserving the challenges associated with identity preservation.

### Strengths
1. The paper presents the first publicly released, million-scale image editing dataset with unified formatting and high alignment between instructions and outputs, significantly advancing research in image editing.
2. The paper provides an in-depth investigation of channel-wise vs. token-wise conditioning, clearly demonstrating the superiority of token-wise conditioning in context-aware editing tasks.
3. Models trained on the proposed dataset achieve state-of-the-art (SOTA) performance across multiple standard benchmarks, validating the effectiveness of the approach.
4. The paper is clearly written and well-structured.

### Weaknesses
1. After using GPT-Image-1 to generate edited images, did the authors apply any filtering or quality screening to these newly generated images? For instance, were samples with generation failures, severe distortions, or complete misalignment with the instruction removed? If filtering was performed, please detail the criteria and procedure used.
2. Why was knowledge distillation conducted exclusively from GPT-Image-1, rather than aggregating outputs from multiple top-tier closed-source or open-source models to construct a more diverse and robust “consensus” dataset? For example, one could employ voting or fusion strategies across multiple model outputs.
3. Are the quantitative results in the paper solely based on automatic evaluation by MLLMs? Did the authors conduct any human evaluation? Relying exclusively on MLLM-based scoring may introduce biases or hallucinations inherent to the MLLM itself.

### Questions
See Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper addresses the performance gap between proprietary instruction-guided image editors (like GPT-Image-1) and their open-source counterparts. The authors identify the primary bottleneck as the lack of large-scale, high-quality training data, as existing datasets often suffer from poor alignment or overly simplistic instructions.

The main contribution is GPT-IMAGE-EDIT-1.5M, a new, publicly available dataset of over 1.5 million editing triplets. This dataset is curated by systematically unifying and refining three existing datasets (OmniEdit, HQEdit, UltraEdit). The refinement pipeline uniquely leverages proprietary models—using GPT-Image-1 for output regeneration and GPT-4O/5 for instruction rewriting—to significantly enhance instruction-following (IF) and perceptual quality (PQ). Crucially, the dataset intentionally preserves challenging identity preservation (IP) cases, reflecting realistic complexities

### Strengths
1. The primary strength is the dataset itself. It is large (1.5M samples)  and meticulously curated. The pipeline, which uses GPT-Image-1 for regeneration and GPT-4O for rewriting, directly addresses the known weaknesses (poor alignment, simple instructions) of prior datasets. More important, The authors commit to releasing the dataset, models, and code, which is a significant service to the community.
2. The decision to intentionally preserve difficult identity preservation (IP) cases is a significant and important one. This strategic choice makes the dataset more realistic.
3. The paper provides a clear and impactful conclusion: token-wise conditioning is superior to channel-wise conditioning for complex editing. The ablation in Table 4 is a powerful piece of evidence, showing that channel-wise models regress on this challenging data while token-wise models improve. This is a valuable directive for the community.
4. The counter-intuitive finding that a frozen T5 text-only encoder is a more robust choice than a frozen MLLM (Qwen2.5-VL) is a very practical and useful insight for open-source development. The analysis correctly identifies that shallow fusion is a bottleneck, pointing toward deep fusion as a necessary area for future work.

### Weaknesses
1. The paper's contribution is primarily empirical and data-centric. It does not propose a new model architecture, conditioning mechanism, or fusion strategy. Instead, it offers a thorough benchmark of existing components (SD3-IP2P, Flux, T5, Qwen-VL). While this analysis is valuable, the paper is an "analysis of what works" rather than a "proposal of a new method."

### Questions
None

### Soundness
3

### Presentation
3

### Contribution
3
