000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053

![0_image_0.png](0_image_0.png)

Figure 1: Images generated via prompt or translated structured annotations. We highlight inconsistent Obj(ect), Rel(ation), and Attr(ibute) in T2I Models. Models trained with our structured annotations perform significantly better than unstructured counterparts in complex scenes with >4 objects.

## Abstract

Despite their success in generating high-quality images, text-to-image (T2I) models struggle to generate compositional scenes with multiple objects and their intricate relationships. We attribute this issue to limitations in existing datasets of image-text pairs, which lack precise inter-object relationship annotations with prompts only. To resolve this, we construct LAION-Comp, a large-scale dataset of 540K+ aesthetic images structurally annotated with detailed scene graphs explicitly encoding multiple objects, corresponding attributes, and intricate relations. The annotation pipeline employs a large vision-language model followed by partial human verification. Using LAION-Comp, we train 4 baseline models on diffusion and flow matching backbones augmented with a designed scene graph encoder. For proper evaluation, we introduce CompSGen Bench, a benchmark with 20,838 testing samples designed to systematically evaluate complex compositions. Experiments show that the 4 models trained on LAION-Comp outperform their original prompt-only counterparts and advanced scene-graph-based methods on both our new and existing compositional benchmarks. Furthermore, the learned structural conditioning naturally enables fine-grained, object-level image editing, demonstrating its potential as an effective editing interface. Our work validates the advantages of explicit structural annotation and contributes the community with a foundational resource to advance controllable and compositional image synthesis.

Anonymous authors Paper under double-blind review

# Laion-Comp: Unlocking Controllable And Compositional Generation With Structural Annotations

1

## 1 Introduction

054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 Compositional image generation refers to the synthesis of scenes comprising multiple objects, their attributes, and intricate inter-object relations. As illustrated in fig. 1, conventional text-to-image
(T2I) models Stability-AI (2024); Batifol et al. (2025) often falter when faced with such complexity. In contrast, generation frameworks guided by structured annotations demonstrate a superior capability in handling these scenarios accurately. We attribute this critical limitation not to model architecture, but to a fundamental deficiency in existing text-image datasets: a lack of explicit annotations for complex inter-object associations. Consequently, prior works that have primarily focused on architectural improvements have failed to address this underlying data-level issue. To overcome this, we advocate for structural annotations, typically represented as scene graphs (SGs). An SG consists of nodes, representing objects and their attributes, and edges, depicting the relations between objects. In contrast to the inherently sequential and often ambiguous nature of text descriptions, SGs provide a compact, structured, and explicit paradigm for describing complex scenes, thereby enhancing annotation efficiency. Crucially, SGs enable the precise specification of specific objects associated attributes and their relations—a capability that is critical for both generating complex scenes and enabling fine-grained image editing. However, progress in this direction is hindered by a critical gap in data resources: existing scene graph datasets, such as COCO-Stuff (Caesar et al., 2018) and Visual Genome (Krishna et al., 2017), are limited in scale and diversity of annotation, while large-scale datasets consist almost exclusively of unstructured text annotations. In this work, we aim to establish a more robust structural data foundation for compositional image generation while unlocking the potential of structured data for image editing tasks. Specifically, we construct LAION-Comp, a large-scale dataset built as a significant extension of LAION- Aesthetics V2 (6.5+) (Schuhmann et al., 2022) with high-quality, high-complexity structural annotations. Therefore, our LAION-Comp better encapsulates the semantic structure of complex scenes, supporting improved generation for intricate scenarios. The superiority of LAION-Comp in complex scene generation is validated in experiments with multiple metrics on semantic consistency. Leveraging LAION-Comp, we train existing state-of-the-art models and propose a new suite of baseline models to comprehensively validate the effectiveness of structural annotations for compositional generation. Our baselines are built upon diffusion (Rombach et al., 2022; Podell et al., 2023) and flow matching (Stability-AI, 2024; Batifol et al., 2025) backbones. We design and train an auxiliary scene graph encoder that employs a Graph Neural Network (GNN) (Scarselli et al., 2008b) to effectively process the structural information in SGs and produce optimized embeddings. These embeddings are then integrated into the generative backbones, significantly enhancing models' capability to synthesize high-quality, complex images. For a targeted and rigorous evaluation, we establish CompSGen Bench, a new benchmark specifically designed for complex scene generation. With this benchmark we evaluate leading T2I and SG- to-Image (SG2IM) models alongside our proposed baselines, comparing performance when trained on COCO-Stuff, Visual Genome, and our LAION-Comp. Both quantitative and qualitative results unequivocally demonstrate that models trained on LAION-Comp consistently and significantly outperform their counterparts. These findings lead us to conclude that the high-quality, large-scale structural annotations in LAION-Comp are crucial for advancing complex scene generation. Furthermore, the structured nature of SGs naturally facilitates fine-grained, object-level image editing, as it allows users to perform intuitive and precise modifications directly on the graph structure. Building on this potential, we develop a training-free image editing framework based on an RF inversion strategy (Rout et al., 2025). Our qualitative and quantitative experiments demonstrate the remarkable effectiveness and controllability that structural annotations bring to image editing. Due to space limitation, the proposed editing framework is introduced in Sec. A.1. In summary, our work represents a significant step toward scaling structurally complex annotations to high-quality, large-scale datasets, enabling broader scene synthesis and editing. Our contributions are as follows. (1) We introduce LAION-Comp, a new, large-scale dataset for compositional generation. It features high-quality structural annotations with multiple objects, attributes, and intricate relations, enhancing a model's ability to generate complex and high-fidelity images. (2) We fine-tune a new suite of foundation models based on diffusion and flow-matching backbones, demonstrating superior performance in complex scene generation. Furthermore, we propose a training-free, SG-

![2_image_0.png](2_image_0.png)

based image editing framework, highlighting the powerful editing potential of structural annotations. (3) We establish CompSGen Bench, a dedicated benchmark to evaluae complex scene generation. Through extensive experiments on this benchmark, we validate the significant advantages of our dataset and the effectiveness of our proposed models. Our annotations with the associated processing code, the foundation models and the benchmark protocol will be publicly available.

## 2 Related Work

108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 Compositional Image Generation. Text-to-image generation (Saharia et al., 2022; Ramesh et al., 2022; Dhariwal & Nichol, 2021; Chen et al., 2024a; Tewel et al., 2024; Zhou et al., 2023; Li et al., 2024; Podell et al., 2023) has advanced significantly, particularly through diffusion models (Ho et al., 2020; Rombach et al., 2022). However, the sequential format of the textual data imposes limitations in handling compositional images with multiple objects and relations(Yang et al., 2024; Lian et al., 2023; Zhang et al., 2024b). Methods enhance controllability via custom losses and attention maps, such as Universal Guidance (Bansal et al., 2023), BoxDiff (Xie et al., 2023), and RealCompo (Zhang et al., 2024a). Other approaches exploit spatial conditions (e.g., GLIGEN (Li et al., 2023), Ranni (Feng et al., 2024)) or LLM-assisted layouts (Feng et al., 2023b; Lian et al., 2023; Zhang et al., 2023a; Wu et al., 2024b), typically relying on precise inputs or incurring high training costs. All of these mainly focus on model improvement, failing fundamentally to address the limitations imposed from the dataset. Image Generation from Scene Graphs (SG2IM) (Johnson et al., 2018; Krishna et al., 2017) involves creating images based on structured representations of scenes, where objects and their relationships are explicitly defined as a graph (Xu et al., 2017). Modern SG2IM models align scene graphs directly to images for better handling of content generation (Feng et al., 2023a; Wang et al., 2024a; Zhang et al., 2023b), with SG-Adapter fine-tuning Stable Diffusion (SD) via attention (Shen et al., 2024), SGDiff pre-trains an SG encoder combined with SD (Yang et al., 2022), and R3CD using transformers for abstract interactions (Liu & Liu, 2024). Refinements include knowledge consensus for semantic disentanglement (Wu et al., 2023b), cross-attention for object consistency (Zhang et al., 2023b), and masked auto-encoders for grounding in SGG-IG (Wang et al., 2025). These approaches enhance semantic capacity beyond text-only conditions, yet remain constrained by the limited scale and quality of existing SG datasets. Large-Scale Image-Text Datasets and Benchmarks. Previous datasets, such as MS-COCO (Lin et al., 2014), Visual Genome (Krishna et al., 2017), and ImageNet (Deng et al., 2009), are limited

![3_image_0.png](3_image_0.png)

| Annotation        | # Objects             |           |            |            |       |
|-------------------|-----------------------|-----------|------------|------------|-------|
| (w/o Proper Noun) | Length                | SG-IoU+↑  | Ent.-IoU+↑ | Rel.-IoU+↑ |       |
| LAION Caption     | 5.33±3.94 (2.02±3.01) | 19.0±19.7 | 0.306      | 0.631      | 0.557 |
| LAION-Comp        | 6.39±4.17             | 32.2±20.3 | 0.422      | 0.810      | 0.749 |

Table 1: The number of objects and length per sample, and the average accuracy for 300 samples across different annotation types.

162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 Our LAION-Comp dataset is built on high-quality images in LAION-Aesthetic V2 (6.5+) (Schuhmann et al., 2022) with automated annotation performed using GPT-4o (OpenAI et al., 2024). LAION-Aesthetics V2 (6.5+) is a subset of LAION-5B (Schuhmann et al., 2022), comprising

## 3 Dataset And Benchmark

A large-scale, high-quality dataset is essential for learning compositional image generation. However, existing large-scale T2I datasets, such as LAION (Schuhmann et al., 2022), describe information beyond the images (as illustrated in fig. 5), misleading the generation. In contrast, SG datasets tend to focus more specifically on the actual content within images, namely the objects and relations. Nonetheless, current SG datasets, such as COCO and VG, are relatively small in scale and have limited object and relationship types, making them insufficient for compositional image generation. To address this, we propose LAION-Comp, a large-scale, high-quality, open-vocabulary SG dataset and Complex Scene Generation Benchmark (CompSGen Bench) to evaluate models' performance .

## 3.1 Dataset Construction

in scale due to the considerable costs associated with manual annotation. To mitigate the limitation, several studies have explored automatic annotation, as exemplified by CC12M (Changpinyo et al., 2021), SPRIGHT (Chatterjee et al., 2025), and LAION-5B (Schuhmann et al., 2022). LAION- Aesthetics is curated for high visual quality and intended to support image generation. However, it does not ensure textual descriptions that accurately reflect image content. Thus We enhance LAION- Aesthetics with structured annotations for high-quality compositional generation, adding attributes beyond objects, in contrast to contemporaneous effort (Chen et al., 2024b).

Benchmarks assess T2I comprehensively: T2I-CompBench for 6K prompts (Huang et al., 2023),
HRS-Bench for 13 skills (Bakr et al., 2023), HEIM for 12 dimensions (Lee et al., 2023b), VISOR for spatial relations (Gokhale et al., 2023), and HPS v2 for human preferences (Wu et al., 2023a). Recent frameworks add flexibility, like ConceptMix for controllable difficulty (Wu et al., 2024a), INQUIRE for expert queries (Vendrow et al., 2024), and GenEval for object-focused metrics (Ghosh et al., 2023). These benchmarks only focus on text-based image generation. To fill the gap in this domain, we are the first to propose a compositional generation benchmark based on scene graphs.

![4_image_0.png](4_image_0.png) 

216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 625,000 image-text pairs with predicted aesthetic scores over 6.5, curated using the LAION- Aesthetics Predictor V2 model. During our construction, only 540,005 images are available. Through prompt engineering, we devised a set of specific requirements for scene graph annotations to ensure comprehensiveness, systematic structure, and precision in the annotation results. Figure 2 illustrates the detailed construction pipeline of LAION-Comp. Each component plays a crucial role in achieving high-quality automated annotation. First, as scene graphs typically contain multiple objects and their relations, the prompt requires "identification of as many objects, attributes, and their relations within the image as possible". This design encourages that all objects and interactions in a scene are annotated. Each object is assigned a unique ID, even for multiple objects of the same type, ensuring that the entirety of the scene's structure and hierarchy is accurately represented. Second, the attribute section mandates that each object must have at least one abstract adjective attribute, while avoiding the use of other objects as attributes. This design is especially important in complex scenes as it helps differentiate objects' appearance, state, and characteristics from the background and other elements, maintaining consistency and clarity in annotations. By avoiding the confused annotation between specific objects and abstract attributes, the annotations become more interpretable and generalizable. In the relation section, we specify the use of concrete verbs to describe relations between objects rather than relying solely on spatial orientation. This is because relations are often more critical in scene graphs than mere spatial information. By using precise verbs like "standing on" or "holding", we capture dynamic interactions within the scene, which is essential for complex scene generation. Leveraging these prompts with the multimodal large language model GPT-4o, we generate annotations representing scene graphs. To investigate the reliability of the annotations, we conduct a partial human verification. Results show the annotations achieve high accuracies of 98.8% for objects, 97.5% for attributes, and 95.7% for relations (Sec. A.5).

## 3.2 Laion-Comp Dataset

By performing the construction strategy, we develop LAION-Comp, a large-scale, high-quality dataset containing 540,005 SG-image pairs annotated with objects, attributes, and relationships.

This dataset is divided into a training set of 480,005 samples, a validation set of 10,000 samples, and a test set of 50,000 samples. We present statistics comparing the original LAION-Aesthetics text-to-image dataset with our LAION-Comp dataset as follows. In table 1, in the original LAION-Aesthetics caption, the average number of objects per sample is 5.33, with 38% of these being proper nouns that offer limited guidance during model training. For 270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 As the complexity of the prompt increases, the generated image becomes more difficult to control (fig. 1). We introduce foundation models to address the challenges of compositional image generour SG annotations, the average number of objects per sample increases to 6.39, excluding abstract proper nouns and focusing on specific nouns that reflect true semantic relationships. LAION-Comp contains 20% more object information than the original LAION-Aesthetics dataset, and this advantage increases to 216% when excluding proper nouns. We also calculated the relationship between length and accuracy for different annotations. The annotation length for text is defined as the number of tokens in the prompt, while for SG as the total number of nodes and edges. We leverage SG-IoU+, Entity-IoU+, and Relation-IoU+ introduced in Sec. A.2 to measure annotation accuracy. The average annotation length for original captions and our scene graphs is 19.0 and 32.2, respectively, with SG achieving higher accuracy across all three metrics. Figure 3 visualizes the length and accuracy of samples for both annotation types. Note that a scene graph is a more structured and compact form of annotation compared to text. Even so, the annotated SG length is still significantly longer than sparse text, and its accuracy is also much higher. This demonstrates that our LAION- Comp dataset contains richer, more nuanced, and precise semantic features, enhancing the trained model performance and fundamentally addressing the challenges of generating complex scenes. Furthermore, we analyze the length distribution of scene graphs in LAION-Comp in fig. 4 (a). Most objects are described by 0-5 (45.72%) or 5-10 (41.04%) words, with a smaller proportion described by 10-20 (12.46%) words or ≥ 20 (0.79%) words. This range is reasonable, offering a more precise expression than a single word while avoiding excessive length that could hinder model learning efficiency. In terms of the overall scene graph, the proportions of word counts in the ranges 0-10, 10-20, 20-30, and ≥ 30 are 10.39%, 32.15%, 28.80%, and 28.66%, respectively. These statistics reflect the richness, detail, and flexibility of annotations in LAION-Comp. Figure 4 (b) presents the top 10 most frequent relations and attributes in LAION-Comp. The most frequent relation is "surrounded by", occurring 80,058 times and accounting for 3.78% of all relations. The 1st common attribute is "tall" (7.36%), while the 2nd common is "small" (only 4.58%). The 10th relation and attribute each make up only 1.51% and 2.2%. These data indicate the annotations in LAION-Comp are highly diverse and broadly covered, as even the most frequently used descriptors represent only a small percentage. To highlight the semantic richness and diversity of LAION-Comp, we conduct a comparative analysis with the widely used VG (Krishna et al., 2017), focusing on the distribution of relation types. Specifically, we categorize relations into spatial (e.g., "on", "under", "next to") and non-spatial (e.g., "holding", "wearing", "playing") types, which reflect different levels of semantic complexity. Quantitative analysis highlights a clear distributional difference. In LAION-Comp, non-spatial relations dominate (77.48%), whereas spatial relations account for only 22.52%. Conversely, VG is spatially skewed, with 58.02% versus 41.98%. LAION-Comp captures more abstract, functional, and interaction-based semantics, moving beyond the predominantly geometric or locational focus of VG. Such enrichment is crucial for compositional and controllable image generation, providing a more challenging and realistic benchmark for scene understanding, as also reflected in T2I- CompBench (Huang et al., 2023) and MMRel (Nie et al., 2024), where models exhibit greater difficulty with complex non-spatial semantics than with spatial configurations.

## 3.3 Complex Scene Generation Benchmark

To evaluate model performance on compositional image generation, we propose Complex Scene Generation Benchmark (CompSGen Bench). From the 50,000-image test set, we select samples with over four relations as complex scenes, and get a total of 20,838 samples. We calculate FID (Lee et al., 2023a) , CLIP score (Radford et al., 2021), and three accuracy metrics (Shen et al., 2024) to assess models' performance. FID measures the overall quality of generated images, while the CLIP score calculates the similarity between the generated and ground truth images. The complex scene evaluation consists of three metrics: SG-IoU, Entity-IoU, and Relation-IoU. They represent the overlap between the generated images and the real annotations in terms of scene graphs, objects, and relations, respectively. Sec. 5.1 shows the test results for different models on CompSGen Bench.

## 4 Foundation Models

324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 ation in T2I task. Our models are built on advanced diffusion (Podell et al., 2023; Rombach et al., 2022) and flow matching (Stability-AI, 2024; Batifol et al., 2025) backbones, incorporating structural information via graph neural networks (GNN) (Scarselli et al., 2008b). A scene graph consists of multiple triples and single objects. Our baseline initializes each triple and single object separately using the CLIP text encoder ET (·). For single objects, the initialization result from CLIP serves as the final representation, denoted as es. For SG triples, each of them is encoded by CLIP to yield a corresponding triple embedding et = ET (*triple*sg). Our SG encoder extracts object and relation embeddings as the nodes and edges and inputs them into the GNN to optimize the SG embedding. More calculation details can be found in Sec. A.9.3. If a relation contains multiple words, each word contributes an edge connecting the nodes of the two related objects. Attributes are treated as separate nodes connected to their respective objects. After processing with the GNN, we obtain a refined triple embedding, denoted as er.

To stabilize the training, we introduce a learnable scaling factor α to control the strength of the refined embedding. α is initialized as zero and updated throughout training. Finally, all triple embeddings are concatenated with single-object embeddings to form the SG embedding esg, which is fed into diffusion- or flow-matching-based backbones for compositional semantic learning.

esg = f(sg) = concat(et + αer, es) (1)
Taking flow-matching-based backbones as an example, given a clean image latent x0 and Gaussian noise ϵ, the SG encoder is trained with:
L = Ex0,ϵ,t,sg -∥ vθ(zt*, t, f*(sg)) − (ϵ − x0) ∥
2 2
, (2)
where zt is the rectified flow trajectory, t ∈ [0, 1], and f(sg) denotes the SG embedding. We train the parameters of SG encoder to minimize the gap between the predicted and ground-truth vector field, which are defined as vθ and ut(z|ϵ) = ϵ−x0. This objective is shared across SD3.5-SG and FLUX-
SG, while the integration strategy of SG embedding differs. Sec. A.9.4 provides a more detailed derivation of this process and Sec. A.9.3 elucidates the theoretical principles of the diffusion-based baselines. Our scene graph encoder is fine-tuned to align with the generative architectures of these models, leading to enhanced synthesis performance. To enhance user-friendliness, we design an automated pipeline that supports flexible, dual-modality inputs: free-form text and structured SGs (Sec. A.9.5). And the editing framework based on the foundational model is introduced in Sec. A.1.

## 5 Experiments

Our trained models for compositional generation are comprehensively evaluated against several strong baselines (Podell et al., 2023; Shen et al., 2024; Yang et al., 2022) on the CompSGen Bench, COCO-Stuff, and Visual Genome datasets. In addition, we present experimental results on SG-based image editing in Sec. A.1.2 and Sec. A.1.3. We also conduct a quantitative analysis (Sec. 3.2) and a user study (Sec. A.3) to verify the effectiveness and strong correlation with human perception of structured annotations. Further details regarding the experimental setup are available in Sec. A.2.

## 5.1 Compositional Image Generation

Qualitative Results. Figure 5 displays 1024×1024 images generated on LAION-Comp. Each row shows the original caption, the scene graph, the GT image, and images generated by different models. The corresponding elements in the SG and images are highlighted in matching colors. For fairness, we compare our SDXL-SG with existing diffusion-based SG2IM models, while the results of FLUX-SG are provided at the end. SDXL-SG and FLUX-SG can generate scenes with more accurate objects and relations, even for complex scenarios. For instance, in the first row, where the relationship is "male person painting female person", both (a) and (b) fail to generate "painting",
and (c) generates two females, whereas SDXL-SG accurately and qualitatively generate the provided relations. Figures (f)-(t) illustrate more examples where ours outperform existing baselines. Additionally, existing T2I and SG2IM models more frequently generate incorrectly in (f). Other errors include erroneous number of generated objects such as bag in the green box in (k), person in the blue box in (p) and (q) or attribute errors such as bag in the green box in (l). Conversely, SDXL-SG and FLUX-SG demonstrate robustness against these failure modes.

378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431

![7_image_0.png](7_image_0.png)

| T2I SG2IM   |
|-------------|

Type Method Dataset FID↓ SG-IoU↑ Ent.-IoU↑ **Rel.-IoU**↑

T2I

SDXL LAION **19.3** 0.371 0.813 0.780 SD3.5-Medium LAION 24.6 0.541 0.854 0.831 FLUX.1-Dev LAION 26.2 0.544 0.885 0.842

| SGDiff w/o bbox SG-Adapter SDXL-SG (Ours)   |
|---------------------------------------------|

COCO 47.8 0.435 0.841 0.816 Visual Genome 35.2 0.529 0.801 0.795

LAION-Comp 32.2 0.531 0.855 0.830

SG-Adapter

COCO 34.9 0.485 0.840 0.833 Visual Genome 39.5 0.515 0.803 0.782

LAION-Comp 31.3 0.538 0.866 0.852

SDXL-SG (Ours)

COCO 30.0 0.497 0.842 0.833 Visual Genome 21.9 0.546 0.813 0.800

LAION-Comp 20.1 0.558 0.884 0.856

SD3.5-SG (Ours) LAION-Comp 20.8 0.578 **0.897** 0.859 FLUX-SG (Ours) LAION-Comp 24.7 **0.583** 0.893 **0.859**

Quantitative Results. We compared results of both T2I and SG2IM models trained on different datasets. The original SGDiff (Yang et al., 2022) introduces bounding box as auxiliary data during training. For fair comparison, we train SGDiff without bounding box with the official implementation. We used FID to evaluate the quality of generated images. Fine-tuning pre-trained T2I models inevitably increases FID scores (Ruiz et al., 2023; Shen et al., 2024; Wang et al., 2024c). We also measure SG-IoU, Entity-IoU, and Relation-IoU (Shen et al., 2024).

As demonstrated in table 2, our baseline achieves the best performance among all candidates in both image quality and accuracy. Notably, the SG-IoU of T2I model is significantly lower than that of SG2IM models, indicating that text provides far less control in the image generation process compared to structured annotations. This highlights the necessity of constructing a large-scale, highquality structured annotation dataset. Furthermore, for the same model, results trained on LAION-
Table 3: T2I and SG2IM results on the CompSGen Benchmark. ∗
denotes ours. The best is in bold, and the second best is underlined.

| Type       | Method   | FID↓   | CLIP↑   | SG-IoU↑   | Ent.-IoU↑   | Rel.-IoU↑   |
|------------|----------|--------|---------|-----------|-------------|-------------|
| T2I        | SD1.5    | 60.4   | 0.654   | 0.170     | 0.604       | 0.511       |
| SDXL       | 25.2     | 0.700  | 0.226   | 0.753     | 0.658       |             |
| SGDiff     | 35.8     | 0.690  | 0.304   | 0.787     | 0.698       |             |
| SG-Adapter | 27.8     | 0.681  | 0.314   | 0.771     | 0.693       |             |
| SD1.5-SG∗  | 56.3     | 0.653  | 0.179   | 0.614     | 0.530       |             |
| SDXL-SG∗   | 26.7     | 0.698  | 0.340   | 0.792     | 0.703       |             |
| SD3.5-SG∗  | 28.5     | 0.702  | 0.345   | 0.840     | 0.738       |             |
| FLUX-SG∗   | 29.0     | 0.707  | 0.338   | 0.851     | 0.776       |             |

Comp consistently outperformed those trained on COCO and VG. This suggests that our LAION- Comp is more effective than previous SG-image datasets due to its higher annotation quality. Additionally, we evaluate the complex scene generation capability of advanced T2I and SG2IM models on the CompSGen Bench (Sec. 3.3). As shown in table 3, our baseline outperforms existing models in terms of image quality, similarity to GT images, and content accuracy. Compared to SDXL, the FID of SDXL-SG does not increase significantly after fine-tuning—a process that typically elevates FID. However, SDXL-SG substantially outperforms SDXL on accuracy metrics, including SG-IoU, Entity-IoU, and Relation-IoU. Beyond the SDXL backbone, we also perform evaluations using SD1.5 and the flow-matching-based SD3.5-SG and FLUX-SG, which achieve further performance gains, indicating the effectiveness and adaptability of our dataset and method. We further compute CLIP scores on COCO, which are 0.630 for SDXL and 0.635 for SDXL- SG. Although the test set of CompSGen Bench is more complex, the models achieve even higher scores, corroborating the high quality of LAION-Comp. Moreover, we conduct evaluations on T2I- CompBench (Huang et al., 2023), with details provided in Sec. A.6, which demonstrate the superiority of our dataset and baseline model.

432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485

## 6 Conclusion

We introduce LAION-Comp, a large-scale dataset with detailed structural annotations for compositional generation, addressing the core problem of unstructured training data. Models trained on LAION-Comp demonstrate improved fidelity and compositional accuracy on our CompSGen Bench and existing benchmarks, outperforming present methods. Our work validates that large-scale, highquality structural annotations are crucial for advancing controllable image synthesis and provides a foundational resource to the community for future research.

## 5.2 Ablation Study

We conduct ablation studies to demonstrate the positive impact of LAION-Comp. We train SDXL-SG variants on 10%, 20%, 50%, and 100% samples of LAION-Comp. The total training iterations remain constant across all settings for fairness. As the sample size increases, the model's capability to generate compositional images improves significantly (table 4). Notably, in the 10% LAION-Comp ablation, where the data volume is smaller than that of VG, the model's FID and Entity-IoU scores still outperform the results trained on VG, with other scores remaining roughly comparable (table 2). LAION-Comp not only provides a data volume advantage but also features higher quality in images and annotations, which enhances training efficiency and improves performance in compositional image generation.

Method Prop. FID↓ SG-
IoU↑
Ent.-
IoU↑
Rel.-
IoU↑
10% 31.6 0.522 0.794 0.790 20% 24.3 0.524 0.804 0.793 50% 22.9 0.535 0.800 0.796 100% 21.9 0.546 0.813 0.800 SG- Adapter SDXL-SG
10% 27.3 0.530 0.874 0.837 20% 24.5 0.533 0.877 0.838 50% 22.2 0.547 0.876 0.849 100% **20.1 0.558 0.884 0.856**
Table 4: Results of ablation. Prop. denotes data proportion.

| SG2IM   |
|---------|

## Reproducibility Statement

486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 Michael Samuel Albergo and Eric Vanden-Eijnden. Building normalizing flows with stochastic interpolants. In *The Eleventh International Conference on Learning Representations (ICLR)*, 2023.

Oron Ashual and Lior Wolf. Specifying object attributes and relations in interactive scene generation. In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, pp. 4560–4568, 2019. doi: 10.1109/ICCV.2019.00466.

James Atwood and Don Towsley. Diffusion-convolutional neural networks. *Advances in Neural* Information Processing Systems, 29:1993–2001, 2016.

Eslam Mohamed Bakr, Pengzhan Sun, Xiaoqian Shen, Faizan Farooq Khan, Li Erran Li, and Mohamed Elhoseiny. Hrs-bench: Holistic, reliable and scalable benchmark for text-to-image models. In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, pp. 20041–20053, October 2023.

Arpit Bansal, Hong-Min Chu, Avi Schwarzschild, Soumyadip Sengupta, Micah Goldblum, Jonas Geiping, and Tom Goldstein. Universal guidance for diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, pp. 843–852, June 2023.

Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, et al. Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. *ArXiv*, 2506.15742, 2025.

Michael M Bronstein, Joan Bruna, Yann LeCun, Arthur Szlam, and Pierre Vandergheynst. Geometric deep learning: going beyond euclidean data. *IEEE Signal Processing Magazine*, 34(4):18–42, 2017.

Tim Brooks, Aleksander Holynski, and Alexei A. Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF Czonference on Computer Vision and Pattern Recognition (CVPR), pp. 18392–18402, June 2023.

Holger Caesar, Jasper Uijlings, and Vittorio Ferrari. Coco-stuff: Thing and stuff classes in context. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 1209–1218, 2018.

Yukuo Cen, Jianwei Zhang, Xu Zou, Chang Zhou, Hongxia Yang, and Jie Tang. Controllable multiinterest framework for recommendation. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pp. 2942–2951, 2020.

Soravit Changpinyo, Piyush Kumar Sharma, Nan Ding, and Radu Soricut. Conceptual 12m: Pushing web-scale image-text pre-training to recognize long-tail visual concepts. In *Proceedings of the* IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3557–3567, 2021. URL https://api.semanticscholar.org/CorpusID:231951742.

Agneet Chatterjee, Gabriela Ben Melech Stan, Estelle Aflalo, Sayak Paul, Dhruba Ghosh, Tejas Gokhale, Ludwig Schmidt, Hannaneh Hajishirzi, Vasudev Lal, Chitta Baral, and Yezhou Yang. Getting it right: Improving spatial consistency in text-to-image models. In Ales Leonardis, Elisa ˇ Ricci, Stefan Roth, Olga Russakovsky, Torsten Sattler, and Gul Varol (eds.), ¨ *Computer Vision –* ECCV 2024, pp. 204–222, 2025. ISBN 978-3-031-72670-5.

## References

To ensure the reproducibility of our research, we provide detailed descriptions of our methods. The guidelines for our dataset construction process are detailed in Sec. 3.1. We describe our foundation models for compositional generation in Sec. 4 and Sec. A.9, and the specifics of SG-based image editing in Sec. A.1. Our experimental setup is described in Sec. A.2. Furthermore, we have made the corresponding code for each model available in the supplementary material.

540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 Hila Chefer, Yuval Alaluf, Yael Vinker, Lior Wolf, and Daniel Cohen-Or. Attend-and-excite:
Attention-based semantic guidance for text-to-image diffusion models. *ACM Trans. Graph.*, 42 (4), July 2023. ISSN 0730-0301. doi: 10.1145/3592116. URL https://doi.org/10. 1145/3592116.

Junsong Chen, Chongjian Ge, Enze Xie, Yue Wu, Lewei Yao, Xiaozhe Ren, Zhongdao Wang, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-σ: Weak-to-strong training of diffusion transformer for 4k text-to-image generation. *ArXiv*, abs/2403.04692, 2024a. URL https: //api.semanticscholar.org/CorpusID:268264262.

Zuyao Chen, Jinlin Wu, Zhen Lei, and Chang Wen Chen. What makes a scene? scene graph-based evaluation and feedback for controllable generation. *ArXiv*, abs/2411.15435, 2024b.

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In *Proceedings of the IEEE/CVF Conference on Computer Vision* and Pattern Recognition (CVPR), pp. 248–255, 2009. doi: 10.1109/CVPR.2009.5206848.

Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. In M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan (eds.), *Advances in Neural Information Processing Systems*, volume 34, pp. 8780–8794, 2021. URL https://proceedings.neurips.cc/paper_files/paper/2021/ file/49ad23d1ec9fa4bd8d77d02681df5cfa-Paper.pdf.

David K Duvenaud, Dougal Maclaurin, Jorge Iparraguirre, Rafael Bombarell, Timothy Hirzel, Alan´
Aspuru-Guzik, and Ryan P Adams. Convolutional networks on graphs for learning molecular fingerprints. *Advances in Neural Information Processing Systems*, 28:2224–2232, 2015.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Muller, Harry Saini, Yam ¨
Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, and Robin Rombach. Scaling rectified flow transformers for high-resolution image synthesis. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 12606–12633, 21–27 Jul 2024.

Weixi Feng, Xuehai He, Tsu-Jui Fu, Varun Jampani, Arjun Akula, Pradyumna Narayana, Sugato Basu, Xin Eric Wang, and William Yang Wang. Training-free structured diffusion guidance for compositional text-to-image synthesis. In *The Eleventh International Conference on Learning* Representations (ICLR), 2023a.

Weixi Feng, Wanrong Zhu, Tsu-Jui Fu, Varun Jampani, Arjun Akula, Xuehai He, S Basu, Xin Eric Wang, and William Yang Wang. Layoutgpt: Compositional visual planning and generation with large language models. In Advances in Neural Information Processing Systems, volume 36, pp. 18225–18250, 2023b. URL https://proceedings.neurips.cc/paper_files/paper/2023/file/ 3a7f9e485845dac27423375c934cb4db-Paper-Conference.pdf.

Yutong Feng, Biao Gong, Di Chen, Yujun Shen, Yu Liu, and Jingren Zhou. Ranni: Taming text-toimage diffusion for accurate instruction following. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 4744–4753, June 2024.

Ronald Aylmer Fisher. Statistical methods for research workers. In Breakthroughs in statistics:
Methodology and distribution, pp. 66–70. Springer, 1970.

Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-to-image alignment. *Advances in Neural Information Processing Systems*, 36: 52132–52152, 2023.

Justin Gilmer, Samuel S Schoenholz, Patrick F Riley, Oriol Vinyals, and George E Dahl. Neural message passing for quantum chemistry. In *International conference on machine learning*, pp. 1263–1272, 2017.

Tejas Gokhale, Hamid Palangi, Besmira Nushi, Vibhav Vineet, Eric Horvitz, Ece Kamar, Chitta Baral, and Yezhou Yang. Benchmarking spatial relationships in text-to-image generation. *ArXiv*, abs/2212.10015, 2023. URL https://arxiv.org/abs/2212.10015.

Marco Gori, Gabriele Monfardini, and Franco Scarselli. A new model for learning in graph domains.

In *IEEE International Joint Conference on Neural Networks*, pp. 729–734, 2005.

Will Hamilton, Zhitao Ying, and Jure Leskovec. Inductive representation learning on large graphs.

Advances in neural information processing systems, 30, 2017.

Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. In NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications, 2021.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In *Advances in Neural Information Processing Systems*, volume 33, pp. 6840–6851, 2020. URL https://proceedings.neurips.cc/paper_files/paper/2020/ file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf.

Kaiyi Huang, Kaiyue Sun, Enze Xie, Zhenguo Li, and Xihui Liu. T2i-compbench: A comprehensive benchmark for open-world compositional text-to-image generation. In Advances in Neural Information Processing Systems, volume 36, pp. 78723–78747, 2023. URL https://proceedings.neurips.cc/paper_files/paper/2023/file/ f8ad010cdd9143dbb0e9308c093aff24-Paper-Datasets_and_Benchmarks. pdf.

Gabriel Ilharco, Mitchell Wortsman, Nicholas Carlini, Rohan Taori, Achal Dave, Vaishaal Shankar, Hongseok Namkoong, John Miller, Hannaneh Hajishirzi, Ali Farhadi, and Ludwig Schmidt. Openclip, July 2021. URL https://doi.org/10.5281/zenodo.5143773.

Justin Johnson, Agrim Gupta, and Li Fei-Fei. Image generation from scene graphs. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 1219– 1228, 2018.

Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al. Visual genome: Connecting language and vision using crowdsourced dense image annotations. *International Journal of Computer* Vision, 123(1):32–73, 2017.

594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 Tony Lee, Michihiro Yasunaga, Chenlin Meng, Yifan Mai, Joon Sung Park, Agrim Gupta, Yunzhi Zhang, Deepak Narayanan, Hannah Teufel, Marco Bellagente, Minguk Kang, Taesung Park, Jure Leskovec, Jun-Yan Zhu, Fei-Fei Li, Jiajun Wu, Stefano Ermon, and Percy S Liang. Holistic evaluation of text-to-image models. In *Advances in Neural Information Processing Systems*, volume 36, pp. 69981–70011, 2023b. URL https://proceedings.neurips.cc/paper_files/
paper/2023/file/dd83eada2c3c74db3c7fe1c087513756-Paper-Datasets_
and_Benchmarks.pdf.

Yuheng Li, Haotian Liu, Qingyang Wu, Fangzhou Mu, Jianwei Yang, Jianfeng Gao, Chunyuan Li, and Yong Jae Lee. Gligen: Open-set grounded text-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 22511–22521, June 2023.

Yujia Li, Daniel Tarlow, Marc Brockschmidt, and Richard Zemel. Gated graph sequence neural networks. *arXiv preprint arXiv:1511.05493*, 2015.

Zhengqi Li, Richard Tucker, Noah Snavely, and Aleksander Holynski. Generative image dynamics. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (CVPR), pp. 24142–24153, 2024. doi: 10.1109/CVPR52733.2024.02279.

Black Forest Labs. Flux. https://blackforestlabs.ai/, 2024. Accessed: September 19, 2025.

Tony Lee, Michihiro Yasunaga, Chenlin Meng, Yifan Mai, Joon Sung Park, Agrim Gupta, Yunzhi Zhang, Deepak Narayanan, Hannah Teufel, Marco Bellagente, Minguk Kang, Taesung Park, Jure Leskovec, Jun-Yan Zhu, Fei-Fei Li, Jiajun Wu, Stefano Ermon, and Percy S Liang. Holistic evaluation of text-to-image models. In *Advances in Neural Information Processing Systems*, volume 36, pp. 69981–70011, 2023a. URL https://proceedings.neurips.cc/paper_files/ paper/2023/file/dd83eada2c3c74db3c7fe1c087513756-Paper-Datasets_ and_Benchmarks.pdf.