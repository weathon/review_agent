---
job_id: 54d9317e-b3dd-4c64-84c1-b8e6d44ffcbb
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: PFhrOUJZ5o.pdf
paper: LAION-COMP: Unlocking Controllable and Compositional Generation with Structural Annotations
main_score_norm: 0.6
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper is squarely about generative models, representation learning with scene graphs, datasets, and benchmarks for compositional T2I generation, which fits ICLR’s core topics.

## Minimum Quality
Pass ✅.  
All required sections (Abstract, Introduction, Related Work, Dataset/Method, Experiments, Results, Conclusion) are present. The work is non‑trivial, proposes a new dataset, benchmark, and models, and includes substantial experiments. I see no fatal theoretical or experimental flaw that would warrant desk rejection.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I did not find any attempts at instructing or manipulating reviewers or automated systems inside the paper text.

---

# Expected Review Outcome:

## Summary

The paper introduces **LAION-Comp**, a large scale dataset of 540k image–scene‑graph pairs obtained by annotating LAION-Aesthetics V2 (6.5+) images with GPT‑4o under a carefully designed prompting scheme, plus partial human verification. Scene graphs encode objects, attributes, and relations and are claimed to be richer and more accurate than the original captions.  

On top of this dataset, the authors build SG‑conditioned diffusion and flow matching models (SDXL‑SG, SD3.5‑SG, FLUX‑SG) using a GNN based scene graph encoder, and propose **CompSGen Bench**, a 20k‑sample benchmark of complex scenes. Experiments on CompSGen Bench, COCO‑Stuff, VG, T2I‑CompBench, and an SG‑based editing setup show improved compositional accuracy compared to text only T2I and previous SG2IM methods.

---

## Strengths

1. **Timely, data centric contribution addressing a real bottleneck.**  
   The paper makes a compelling case that lack of structural supervision is a key reason current T2I models struggle with multi object, relation heavy prompts. Constructing LAION‑Comp at 540k SG‑image pairs is a substantive and useful contribution for the community, especially since existing SG datasets (COCO/VG) are small and biased toward simple spatial relations. The comparison of relation type distributions to VG in Sec. 3.2 and the non spatial relation dominance is informative.

2. **Careful annotation protocol and validation.**  
   Figure 2 clearly illustrates the annotation pipeline and the explicit constraints on object IDs, abstract attributes, concrete relations, and person labeling. This is much more thoughtful than “just call GPT‑4o”. The human verification in Table 6 (Page 22) with object / attribute / relation accuracies of 98.8 / 97.5 / 95.7 percent, stratified by SG complexity, gives reasonable evidence that the annotation noise is low. The bias analysis in Fig. 21 and the hallucination discussion (Fig. 10, Fig. 11 and Appendix A.8) show the authors have at least looked at failure modes.

3. **Quantitative evidence that SG annotations are semantically richer and more faithful than original captions.**  
   Table 1 and Figure 3 jointly support the claim that scene graph annotations are both longer and more accurate than LAION captions. The SG‑IoU+, Entity‑IoU+, and Relation‑IoU+ metrics computed over 300 samples, together with the token / node length distributions, provide concrete backing for the “richer semantics” claim rather than just intuition. Figure 4(a)(b) further shows that relation and attribute vocabularies are diverse rather than dominated by a few trivial words.

4. **Solid modeling and integration into strong backbones with modest overhead.**  
   The SG encoder is simple but principled: CLIP text embeddings of objects, attributes, and multi word relations, a GNN that refines them, and a residual scaling parameter α (Eq. 6–8, 13). The training objectives in Eq. 2, 9, 12, 14 are aligned with standard diffusion and flow matching practice, and Appendix A.9 clarifies how SG embeddings are injected into SDXL and SD3.5 / FLUX. The parameter count and runtime analysis on Page 26–27 shows negligible additional cost (<3 percent inference time, ~0.23 percent parameters), which makes adoption realistic.

5. **Consistent empirical gains on multiple backbones and datasets.**  
   Table 2 (Page 8) is a central piece of evidence: for the same backbone, training on LAION‑Comp improves SG‑IoU / Entity‑IoU / Relation‑IoU across the board, and SDXL‑SG / SD3.5‑SG / FLUX‑SG trained on LAION‑Comp are best or second best in all metrics, with competitive or improved FID. For example, SDXL‑SG on LAION‑Comp achieves FID 20.1 and SG‑IoU 0.558, beating SG‑Adapter and SGDiff variants. On CompSGen Bench (Table 3, Page 9), SDXL‑SG* and SD3.5‑SG* strongly outperform plain SDXL and SD1.5 in all compositional metrics while keeping FID close. These results substantiate the claim that explicit structure helps compositional fidelity.

6. **Qualitative comparisons are convincing and well annotated.**  
   Figure 5 is a good qualitative figure: rows show LAION caption, SG, GT image, and outputs from SDXL, SGDiff, SG‑Adapter, SDXL‑SG, and FLUX‑SG, with colored boxes linking SG elements to image regions. The highlighted failure cases of baselines (wrong object counts, wrong gender, missing relations) versus relatively faithful generations from SDXL‑SG / FLUX‑SG align with the numeric trends in Tables 2–3.

7. **SG based editing interface is interesting and seems effective.**  
   The editing framework (Appendix A.1) is a nice additional result. Figure 6 illustrates the pipeline with a “world knowledge aware user intent parser”, RF inversion conditioned on SG, and FLUX‑SG. The case study in Figure 7 and the quantitative win rates in Table 5 (EC/RA/IQ) suggest that SG conditioning offers more precise, local editing compared with InstructP2P and SGEdit. Fig. 19 and Fig. 20 further show multi step and multi object edits with good locality.

8. **Good effort on evaluating annotation quality and human alignment.**  
   The SG‑vs‑text user study in Fig. 8 (Page 19) is small but well described, and the 63 percent preference for SG based generations over caption based ones adds a human corroboration to the automatic metrics.

---

## Weaknesses

1. **Evaluation is heavily entangled with the same class of models (GPT‑4/4o) used for annotation, which risks circularity and bias.**  
   The core metrics for both dataset annotation accuracy (SG‑IoU+, Entity‑IoU+, Relation‑IoU+, Table 1 and Appendix A.7) and compositional generation (SG‑IoU, Entity‑IoU, Relation‑IoU in Tables 2–3) rely on extracting scene graphs from images with GPT‑4 or GPT‑4o. Since GPT‑4o is also used to *create* the LAION‑Comp annotations, this creates a strong risk that the evaluation overly rewards matching GPT‑4o’s own annotation style rather than true semantic correctness. For example, if GPT‑4o tends to hallucinate a second earring (Fig. 10) or mislabel a stick vs umbrella (Fig. 11), models that mimic these biases can score higher despite being objectively wrong. The paper does not attempt any calibration against an independent SG extractor or human labeled SGs at evaluation time, which weakens empirical claims about “semantic accuracy”.

2. **Scene graph extraction from generated images is underspecified and potentially brittle.**  
   Section A.7 briefly states that GPT‑4 is used to extract scene graphs from generated images and real images, then IoU is computed on lists of triples, entities, and relations. However, it is unclear whether extraction prompts enforce identical constraints as the dataset annotation prompts, how duplicate or synonymous relations are handled, how the matching between SG elements and GT IDs is resolved, or how thresholding is done. For instance, if one SG uses “standing on” and another uses “on top of”, how are they matched in IoU? Without a rigorous definition of SG‑IoU in mathematical notation, including the mapping between predicted and reference nodes/edges, it is hard to judge whether differences like 0.558 vs 0.538 in SG‑IoU (Table 2) are meaningful. At minimum, Eq. (3) in Appendix A.5 only defines a recall style metric for human verification, not the compositional IoUs central to the main results.

3. **Positioning relative to non‑SG compositional control methods is limited.**  
   The related work and experiments compare mainly to SG2IM baselines (SGDiff, SG‑Adapter) and vanilla T2I models, but they largely exclude recent non SG compositional controllers that operate via boxes, masks, attention control, or multi instance scheduling. This includes methods like MIGC / MIGC++, BoxDiff, RealCompo, IFAdapter, and other multi instance controllers that the authors *do* cite in Appendix A.11 but never use as baselines. As a result, the empirical message is essentially “SG conditioning beats text only and older SG2IM”, rather than answering the more relevant question: given a fixed SDXL style backbone, is “scene graph + GNN” a better control interface than, say, layout + CLIP boxes or attention map control, under matched data and training budget.

4. **The claim that the bottleneck is data rather than architecture is somewhat overstated and only partially supported by ablations.**  
   The paper repeatedly emphasizes that prior work focused on architecture instead of fundamentally fixing the “data‑level” issue. However, the experimental design does not disentangle architecture from data very cleanly. For example:
   * SDXL‑SG trained on LAION‑Comp vs SDXL trained on LAION uses different condition modalities *and* different training objectives; SDXL‑SG also benefitted from extra training on LAION‑Comp.  
   * The only ablation in Sec. 5.2 (Table 4, which is referenced but not fully printed in the main text) varies the fraction of LAION‑Comp used, but does not compare to alternative high quality captioning of the same images (e.g., recaptioning them using an LMM) or to text only models trained on more images.  
   * It is therefore hard to attribute gains solely to “structural” supervision rather than “more accurate semantics in any form” plus extra fine tuning. A tighter experiment would hold the model and data volume constant and vary only whether the condition is injected as scene graph structure, refined recaption, or both.

5. **Key architectural details of SG encoder integration are unclear or inconsistent between sections.**  
   There is some confusion between Section 4 and Appendix A.9.3 regarding how SG embeddings are actually combined with the original text embeddings. On Page 7, Eq. (1) defines  
   \[
   \mathbf e_{sg} = \text{concat}(\mathbf e_t + \alpha \mathbf e_r, \mathbf e_s)
   \]  
   and states that this embedding is “fed into diffusion or flow matching backbones”. Appendix A.9.3 further states that “all triple embeddings are concatenated with single object embeddings to form the SG embedding which is fed into the U‑Net of SDXL for iterative noise prediction”. However, it is not explicitly clarified whether this SG embedding *replaces* SDXL’s original text embedding, or is concatenated with it, or injected through a separate cross attention branch. Figure 12 suggests concatenation with the CLIP text embeddings (indicated by ⊙ and *), but the notation is never fully spelled out. For reproducibility and for fair comparison, the conditioning path should be described with the same mathematical precision as Eq. (5) for SDXL, including dimensions, where in the UNet / DiT blocks it is injected, and how classifier free guidance is adapted for SG conditions.

6. **Overreliance on a single base dataset (LAION-Aesthetics) and limited generalization analysis.**  
   Although Section A.15 claims some generalization by evaluating on T2I‑CompBench and mixing samples from COCO/VG/LAION‑Comp for certain metrics, the training of the main SG2IM models is entirely on LAION‑Comp. Table 7 (Page 22) on T2I‑CompBench only evaluates SDXL and SDXL‑SG (using GPT‑4o to convert prompts into SGs again), and even there the non spatial metric is slightly worse. There is no evaluation on downstream tasks that are not “in the LAION aesthetic style” or that require different distributions of objects and styles. Since LAION‑Aesthetics is biased toward high aesthetic, stylized images, the claim that LAION‑Comp is a “foundational resource” would benefit from more evidence that its structural annotations transfer to more mundane photographic datasets.

7. **Potential leakage in editing experiments and use of relatively weak baselines.**  
   The editing evaluation in Table 5 is purely based on pairwise human preferences (EC/RA/IQ win rates) across only 120 edited images per method, which is fine as a small study but limited for drawing strong conclusions. More importantly, the baselines include InstructP2P and SGEdit, but not the stronger, more recent local editing and inversion techniques in the diffusion editing literature. RF inversion is included, but the proposed FLUX‑SG editing is essentially RF inversion with structured SG conditioning. This makes the comparison somewhat stacked in favor of FLUX‑SG. Also, since the same annotator or LMM pipeline might be used to propose the SG edits, there is a risk of distribution match that advantages the proposed method.

8. **Mathematical formalization of metrics and objectives is incomplete.**  
   While the training losses in Eq. (2), (9), (12), and (14) are standard mean squared error objectives, several mathematical aspects are kept informal:  
   * In Eq. (10) and (11), the rectified flow trajectory is defined as \(z_t = (1-t)x_0 + t \epsilon\), and \(u_t(z \mid \epsilon) = \epsilon - x_0\). However, this omits the dependence of \(u_t\) on \(t\) and on the marginal \(p_t(z)\). In Lipman et al. the vector field is more carefully defined as the conditional expectation over \(x_0\) given \(z_t\). Here, the authors implicitly adopt the rectified flow approximation but never discuss its assumptions or implications for training SD3.5‑SG / FLUX‑SG.  
   * SG‑IoU, Entity‑IoU, and Relation‑IoU are central to all tables, yet no explicit formula is provided in the main text or appendix, aside from the verbal description in Appendix A.7. For a work that heavily leans on new metrics, a clear set‑theoretic or graph theoretic definition, including how synonyms, pluralization, and coreference are normalized, is needed.

9. **Minor clarity and presentation issues.**  
   A few smaller points that nonetheless affect clarity:  
   * Table 4 (ablation) is mentioned in Sec. 5.2 but not actually printed in the main excerpt; it is unclear whether it is meant to be in the main paper or appendix.  
   * Some notation in Appendix A.9.3 is inconsistent, for example \(e_s\) vs \(\mathbf e_s\) and subscript conventions for objects and attributes.  
   * In several sentences the authors state fairly strong conclusions such as “This demonstrates that our LAION‑Comp dataset contains richer, more nuanced, and precise semantic features, enhancing the trained model performance and fundamentally addressing the challenges of generating complex scenes” (Page 6). Given the aforementioned evaluation circularity, this sounds stronger than what the evidence strictly supports.

10. **Related work on compositional and controllable generation is incomplete.**  
    The related work section focuses primarily on T2I diffusion, layout guidance, and SG2IM, but omits several recent works on compositional generation and multimodal control that are not SG based yet address very similar problems (see Missing Related Work section). This weakens the positioning and makes it harder to judge the relative conceptual contribution of LAION‑Comp.

---

## Potentially Missing Related Work

Below are directly related papers that appear not to be cited in the submission and should be discussed:

1. **Nie et al., “Controllable and Compositional Generation with Latent-Space Energy-Based Models”, NeurIPS 2021.**  
   This paper addresses compositional image generation by placing energy based models in the latent space of pretrained generators to compose attributes and objects. It is highly relevant as an early, data efficient way to achieve compositional control without SGs. It should be discussed in Section 2 when contrasting model side vs data side approaches and cited when arguing that prior work has focused on architectures.

2. **Huang et al., “Composer: Creative and Controllable Image Synthesis with Composable Conditions”, NeurIPS 2023.**  
   Composer develops a diffusion framework that composes multiple conditional factors. This is closely aligned with the paper’s goal of making complex scenes controllable through composable conditions. It should be referenced in the Compositional Image Generation part of Section 2 and, ideally, compared conceptually to the SG embedding approach.

3. **Shi et al., “SemanticStyleGAN: Learning Compositional Generative Priors for Controllable Image Synthesis and Editing”, CVPR 2022.**  
   This work learns compositional priors over semantic parts for controllable generation and editing. It should be cited in the SG based editing section (Appendix A.1) and in Section 2 as a non diffusion approach to compositional generation.

4. **Dalva et al., “Canvas-to-Image: Compositional Image Generation with Multimodal Controls”, 2025.**  
   This paper proposes a framework that consolidates heterogeneous controls (sketches, text, masks) into a single canvas representation for compositional generation. It is directly related to the claim that SGs provide a convenient structured interface and should be discussed in Section 2 and possibly Sec. A.11 when arguing about multimodal control.

5. **Stirling and Al-Moubayed, “Controllable Image Generation With Composed Parallel Token Prediction”, 2024.**  
   This work composes discrete generative processes for controllable generation. It belongs in Section 2 near other controllable generation methods and can help position LAION‑Comp as a data resource complementary to token based controllers.

6. **Shi et al., “ConsistCompose: Unified Multimodal Layout Control for Image Composition”, 2025.**  
   ConsistCompose unifies multiple layout control modalities for composition. It should be compared in Section 2 and in Appendix A.11, since it addresses similar goals of unified control but through layout embeddings rather than scene graphs.

7. **Casanova et al., “Controllable Image Generation via Collage Representations”, 2023.**  
   This method conditions generation on collage representations, another structured but visual control modality. It should be added to the Compositional Image Generation discussion as an alternative structured interface.

8. **Zhang et al., “ControlCom: Controllable Image Composition using Diffusion Model”, 2023.**  
   ControlCom unifies multiple composition tasks in a diffusion framework. It is particularly relevant when claiming that existing methods fail to handle complex multi object scenes, and should be discussed in Section 2 and Appendix A.11.

Including these works will better situate LAION‑Comp among the growing landscape of compositional and controllable generation methods, and help clarify what is specific about “structural annotations via SGs” versus other structured control modalities.

---

## Questions

1. **Metric robustness and independence from GPT‑4/4o.**  
   Can the authors provide any evaluation of SG‑IoU / Entity‑IoU / Relation‑IoU using a *different* SG extractor, such as a strong SGG model trained on COCO/VG or a different VLM, for at least a subset of the test images? Even if such models are weaker, showing that SDXL‑SG still outperforms SDXL under non GPT‑based extraction would significantly strengthen the empirical claims.

2. **Precise definition of SG‑IoU, Entity‑IoU, Relation‑IoU.**  
   Please include explicit mathematical definitions of these metrics, including:
   * how entity and relation vocabularies are normalized (lowercasing, stemming, synonym mapping);  
   * how subject and object nodes are matched between predicted and reference SGs;  
   * how multi word relations are tokenized and matched;  
   * and how duplicates are handled when computing set intersections and unions.

3. **Integration path of SG embeddings into SDXL and SD3.5 / FLUX.**  
   Can you clarify, ideally with an explicit equation, whether \(\mathbf e_{sg}\) is concatenated with the original text embedding \(\tau(c)\) in Eq. (5) and how classifier free guidance is implemented in the presence of both text and SG conditions? For instance, is the conditioning vector passed as \(\tau([text; sg])\) or is there a separate conditioning path for SG, perhaps through AdaLN or cross attention.

4. **Effect of improved captions versus explicit graph structure.**  
   Have you tried generating high quality textual descriptions of LAION‑Aesthetics images with GPT‑4o and fine tuning SDXL purely on these recaptions (without scene graphs)? A comparison between “recaption only” and “scene graph only” under matched data volume would directly test whether the benefits arise from explicit graph structure or simply from having more accurate semantic supervision.

5. **Generalization beyond LAION‑Aesthetic style.**  
   Could you provide more evidence on how LAION‑Comp trained models behave on photographs from domains like COCO or OpenImages that are less stylized? For example, a user study similar to Fig. 8 but on a COCO subset, or reporting compositional metrics on COCO test images using COCO style captions converted to SGs.

6. **Editing evaluation protocol.**  
   For Table 5 and Fig. 7, could you describe how the editing prompts / SG modifications were generated and randomized across methods? Were annotators blind to the method identities when picking winners, and were images from different methods presented in randomized order? Some more detail here would help assess the reliability of the reported win rates.

Clarifying these points and, if feasible, adding some additional analyses could meaningfully increase my confidence and potentially raise my overall assessment.

---

## Flag For Ethics Review

No ethics review needed.

---

## Details Of Ethics Concerns

N/A. The paper uses LAION‑Aesthetics, which has known issues in general, but the work itself appears to respect consent and privacy norms, and there is an IRB note for the human studies. The potential misuse of generative models is acknowledged in Appendix A.19.

---

## Soundness Rating

3: good.  
The methodology for dataset construction and the SG encoder is technically sound and generally well aligned with existing diffusion and flow matching frameworks. The main technical weakness is not methodological correctness but evaluation circularity and some underspecified metrics.

---

## Presentation Rating

3: good.  
The paper is mostly clear, well structured, and supported by informative figures like Figure 2 (annotation pipeline), Figure 4 (distribution statistics), Figure 5 (qualitative comparisons), and Figure 6–7 (editing). Some important definitions (metrics, conditioning path) need more precise mathematical detail, and the claims could be toned down in places.

---

## Contribution Rating

3: good.  
The dataset, benchmark, and SG conditioned backbones together constitute a meaningful contribution to compositional image generation. The work is not conceptually radical, but the scale of LAION‑Comp, the systematic evaluation on strong backbones, and the editing interface make it a valuable resource for the community.

---

## Overall Rating

6: marginally above the acceptance threshold. But would not mind if paper is rejected.  

The paper provides a useful large scale SG dataset and demonstrates consistent gains in compositional fidelity across several strong generative backbones, supported by solid qualitative and quantitative results. However, the heavy reliance on GPT‑4/4o both for annotation and for evaluation metrics, the incomplete positioning relative to alternative compositional control methods, and some missing technical details limit how strong a recommendation I can give. With clarifications on metrics, some effort toward evaluation independent of GPT‑4o, and better comparison to non SG controllers, this would be an easy accept.

---

## Reviewer Confidence

4: confident.  
I am familiar with diffusion, flow matching, SG2IM, and compositional T2I literature, and I carefully checked the equations and experimental tables, though I did not reproduce any experiments.