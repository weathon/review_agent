# SCVO: Addressing Sparse But Critical Variable Overwhelm In VLMs For Advertising Image Preference Prediction Across Multi-Country Markets

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Vision-language models (VLMs) have demonstrated remarkable capabilities in multimodal tasks, yet their sensitivity to sparse and critical variables that are often overwhelmed remains unexplored. The image preference prediction across multi-country markets task serves as a representative case in this regard. Specifically, VLMs (e.g., Qwen-VL) are tasked with judging between two images (A and B) for the same product across diverse markets (e.g., Korea, France), and the model’s predictions often collapse to a single output (e.g., always ``A'') despite ground-truth preferences varying by country. This failure is attributed to Sparse Critical Variable Overwhelm (SCVO): the model is overwhelmed by dominant high-volume variables (e.g., product attributes, image patches consuming hundreds of tokens), while the critical low-volume variables (e.g., country names consuming only a few tokens) are statistically drowned out. To study this, we first collect a dataset of real-world advertising image click-through preferences across multi-country markets, and then present a novel training framework that strategically mitigates SCVO, and we use it to train on the dataset, yielding CountryReward, a reward model for advertising image preference prediction across multi-country markets. Our framework involves three tailored modules: (1) a cross-country retrieval-augmented generation module that injects click-through preferences aligned with target markets into the training process, enhancing localized relevance prediction. (2) a country adapter module that dynamically modulates image representations based on textual country embeddings, enabling precise visual preference adaptation for diverse markets. (3) a focus-driven penalty loss function that penalizes mispredictions related to the overlooked variable more heavily. Finally, we apply the CountryReward as the reward model to finetune VLMs through Reinforcement Learning (RL), enabling the model to output background designs fed to the text-to-image model (e.g., SDXL) and generate effective e-commerce images for a targeted country. Experiments on the proposed dataset show that our approach significantly mitigates the SCVO effect and improves the preference prediction accuracy. This work highlights the need for robust handling of sparse critical variables in VLMs and offers a scalable solution for real-world applications where subtle contextual shifts drive decision-making.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces the “Sparse Critical Variable Overwhelm (SCVO)” problem: in the context of multimodal instruction scenarios, a small number of crucial variables (such as country names) get overwhelmed by higher‑volume variables (such as product attributes or image patches) during attention fusion, which leads the model to become insensitive to those critical variables and to make erroneous decisions that do not vary across markets. To investigate this phenomenon, the authors collect the MACP dataset (823K training samples, 18K test samples, 10 countries, each record containing two advertisement images of the same product, CTR data and textual attributes; the authors claim sample counts are balanced across countries and all originate from the same e‑commerce platform). On the methodological side, they build a “CountryReward” discriminator model based on Qwen2‑VL, and propose three components: (1) CC‑RAG cross‑country retrieval enhancement; (2) Country Adapter Module (CAM) – using a country embedding to affine‑modulate the visual representation; (3) FDPL (Focus‑Driven Penalty Loss) – a penalty loss term that encourages increased attention to the “country/product/image” key tokens when the model makes mistakes. Moreover, they use CountryReward as a reward signal to apply DPO to a “Design Generation Model (DGM)”, which then generates backgrounds aligned with target‑country preferences; these are then passed to SDXL (a T2I model) to generate e‑commerce images. Experiments claim that on MACP the proposed method attains an accuracy of 60.37%, about 10–11% higher than current SOTA VLMs. They also provide ablation results on the three components and a “sensitivity” metric. In addition, on the DGM generation quality, using CountryReward to score shows that after RL each country’s score improves significantly.

### Strengths
1.	The authors systematize a failure mode — “critical sparse variables being overshadowed by high‑volume variables” — which is worth discussing, and they point to a realistic scenario in cross‑country e‑commerce.

2.	The combination of the CountryReward’s CAM (country‑conditioned affine modulation) and FDPL (focus on key tokens when errors occur) is clearly described and has some potential for reuse.

3.	They provide a reasonably sized dataset (MACP) with multi‑country attributes, giving a real‑world substrate for study (though reproducibility remains in question, as detailed below).

4.	They incorporate a discriminator model as a reward for optimizing a generation model (DGM to T2I pipeline), which aligns with practical deployment intuition, and show gains in CountryReward scores.

### Weaknesses
W1. The evaluation and the definition of the “sensitivity” metric remain opaque. The paper does not provide a full mathematical definition of “Sensitivity,” nor the threshold settings, nor the statistical variance/significance testing procedure. It only gives fragments like “cross‑country combinations C₂(Sᵢ), indicator function II[·]” and does not clearly explain how A/B binary labels derive from CTR (thresholding, recent window/sliding window, confidence interval, etc.). The diagrams show only “answer/label example.” Authors should include full formulae, thresholding rules, and statistical testing (e.g., bootstrap or Clopper‑Pearson). Report per‑country confidence intervals and significance.

W2. The retrieval enhancement before training (CC‑RAG) and the feature e_aug may introduce risk of information leakage or self‑fulfilling feedback loops. Since the total loss concatenates e_aug (text embedding derived from ŷ_aug) with the final discriminator representation for classification, if ŷ_aug is inferred from “past CTR/similar product” (which is strongly correlated with the label), then features strongly correlated with the label may be injected—thus amplifying priors and weakening the model’s genuine sensitivity to the “country token.” The paper only states “retrieval was done before training for augmentation choice,” but does not clarify whether the retrieval index strictly contained only training‑period history, whether near duplicates/same‑product leaks were filtered, nor whether the correlation between e_aug and the label was quantified. Authors need to provide strict data boundaries (time-based splits, deduplication of identical SKUs), leakage detection (near‑duplicates, same‑SKU removal), and an ablation: remove e_aug or use only CAM/FDPL to verify whether the metrics still show significance.

W3. The novelty of “SCVO” and its relation to existing literature needs firmer grounding. Phenomena such as “key token getting lost in long context,” “lost‑in‑the‑middle/positional bias/modality‑volume imbalance causing attention dilution” are already widely studied in the LLM/VLM communities. The paper coins “SCVO” but lacks a differentiated placement vis‑à‑vis existing mechanism‑level research (e.g., attention‑map/gradient attribution on the “country token” before/after). Current narrative about “chain rule being broken” is too slogan‑like. Authors should add controlled synthetic experiments (fix context length and interference density, gradually raise the weight/position of sparse variables), mechanism analysis (attention/activation contribution, intervention experiments) so as to support a full “phenomenon → mechanism → solution” loop.

W4. Details on dataset availability and ethics/compliance are insufficient, impacting reproducibility and compliance assessment. MACP is reported to come from the same e‑commerce platform and includes behaviour data like CTR, but the paper does not state whether the data is open‑source, whether it has been anonymised/approved, whether any geographic/population bias control was applied, nor any legal/privacy compliance statement or data‑sharing terms. Given that some conferences/journals require explicit ethics/privacy statements, the authors should systematically address this in an ethics section. Please add a section that clarifies data‑sharing terms, anonymisation, ethical review, bias mitigation.

W5. Experimental baselines and reporting are inadequate. The paper compares only a subset of VLMs and “Qwen2‑VL + FC head” but lacks stronger baselines closer to the problem setting. For example: An “input engineering” baseline: prompt‑structure rearrangement/repetition of country token; re‑weighting of category/token importance; simple focal‑like loss on discriminators. A classic CTR‑prior logistic regression/GBDT baseline with country × category interaction features. A causal/contrastive baseline: performance gap when country token is masked/removed. Also, only overall accuracy and “sensitivity” are reported; statistical significance and variance are missing. Please add these stronger baselines and report mean ± CI or variance and paired tests.

W6. Technical details for using CountryReward as a reward in RL are insufficient. The paper provides only a high‑level description of RL but omits specifics of the paradigm used (PPO/DPO/GRPO), key hyperparameters, pair construction, number of rounds, compute resources, etc., making replication and assessment of generality difficult. Provide full details of the RL training setup, hyperparameters, compute budget, and ideally release scripts/minimum reproducible code.

W7. Writing and terminology quality require substantial polishing. There are numerous evident spelling/case/grammar mistakes that impair readability and professionalism (for example section headings and terms: “GNERATION,” “PENALITY,” “CONCULSION,” “overwhhelmed” etc.). The scale of the MACP test set is inconsistently reported as both 18K and 180K in the text. Please thoroughly proof‑read the manuscript, standardise terminology and abbreviations, and ensure consistency of numbers/tables.

W8. The statements about “sample balance” among countries and the definitions of “country” need verification. The authors claim “balanced sample counts across countries,” but provide no histograms/distribution or variance/stratified sampling strategy; abbreviations like “BR/CL/ES/SA” are never explicitly defined (Brazil/Chile/Spain/South Africa or Saudi Arabia?). Authors should provide a table of country abbreviations, sample counts per country, and cross‑tabulation of country × product category.

W9. The metric and task formulation may deviate from actual business objectives. If the real business goal is to improve true CTR/conversion rate, then simply optimising “CountryReward score / binary choice accuracy / sensitivity” does not necessarily align. It would be better to at least link the offline proxy metrics (AUC/Calibration) with online or quasi‑online evaluation (e.g., re‑weighting by counterfactuals). Authors need to provide more evidence that optimising these metrics correlates with business KPIs (CTR/Conversion) or include offline‑online analysis.

### Questions
1.	Could the authors supply the full mathematical definition of “Sensitivity”, thresholding rules, and statistical testing procedure (CI / significance)? A formula or pseudocode would help.

2.	In CC‑RAG, does the retrieval database strictly include only history up to the training period? Was near‑duplicate/same‑product filtering applied? How was the correlation between e_aug and the labels quantified (for example via mutual information)?

3.	What is the precise RL paradigm used for the DGM with CountryReward (PPO / DPO / GRPO)? What are key hyperparameters, pair‑construction methodology, number of rounds, compute resources? Will the authors share scripts or minimal reproducible settings?

4.	What is the statistical evidence for “balanced across countries”? Is there stratification by country × category? In a real business setting where imbalance is severe, would the approach remain robust?

5.	Could the authors provide mechanistic evidence supporting the SCVO concept (attention/gradient attribution/intervention experiments) rather than just phenomenological description?

Suggestions for Improvement: 

1.	To reduce leakage and strengthen robustness, please remove e_aug, restrict the retrieval database to an early time window of the training set, remove near‑duplicates, and report results and sensitivity metric changes.

2.	Add stronger baselines and mechanism analysis, include token‑importance re‑weighting, position rearrangement, counterfactual masking; use attention/Integrated Gradients/Grad‑CAM to demonstrate that attention to the “country token” increases.

3.	Reinforce statistical reporting: uniformly report mean ± CI, per‑country variance, and paired tests.

4.	Expand the dataset & ethics section, include data legality/privacy, anonymisation/permission, licensing or controlled access if open release isn’t possible.

5.	Revise the writing, systematically proofread spelling and terminology, ensure consistency in tables and abbreviations.

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
3

### Summary
This paper introduces CountryReward, a vision-language reward model designed to mitigate the Sparse Critical Variable Overwhelm (SCVO) problem—where sparse but crucial tokens (e.g., “country”) are drowned out by high-volume visual features in multimodal transformers. The work focuses on predicting image preference across multi-country markets, where standard VLMs (e.g., Qwen2-VL) tend to collapse into uniform outputs. To address this, the authors propose three modules: a cross-country retrieval augmentation to provide localized context, a country adapter that dynamically modulates visual representations, and a focus-driven penalty loss to upweight errors linked to overlooked sparse variables. Trained on the newly collected Multi-Country Ad Click Preference (MACP) dataset, CountryReward achieves notable improvements over VLM baselines and is further used as a reward model to guide text-to-image generation for market-specific e-commerce visuals.

### Strengths
This work identifies a genuinely underexplored issue in multimodal modeling—how sparse contextual cues are overshadowed by dense visual features—and formalizes it as SCVO. The problem statement is clear and practically relevant, especially for real-world markets where small contextual shifts drive user preferences. The design of modular mitigations (retrieval augmentation, adapter, and loss) is well-motivated and neatly integrated. Empirical results show consistent gains (e.g., +10 pts accuracy over Qwen2-VL baselines) and strong qualitative alignment with country-level preferences. The downstream RL-based image generation case demonstrates a realistic application path, enhancing the paper’s practical significance.

### Weaknesses
The primary limitation is narrow scope and unclear generalization. While the SCVO problem is well-motivated, the study is tightly bound to the advertising-preference domain. It remains unclear whether the same issue appears—and can be solved similarly—in other multimodal settings. The lack of a second use case makes it difficult to judge whether CountryReward offers a general SCVO solution or a domain-specific fix. Additionally, the MACP dataset is new, so reproducibility and comparison with prior multimodal reward models (e.g., UnifiedReward) are limited.

### Questions
Can SCVO and CountryReward generalize to tasks beyond market-specific preference prediction? It is interesting to compare against to some reward models, such the discussed model UnifiedReward. Even though its a generalized reward model, it could strengthen its claim by showing that their specialized approach provides a meaningful improvement over general purpose reward models, not just domain‑specific ones.

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
The paper identifies a phenomenon called Sparse Critical Variable Overwhelm (SCVO): in multimodal A/B image selection for ads, tiny but decisive inputs (like the target country) get drowned out by high-volume text/image features, so VLMs make almost the same choice for all countries. To study this, the authors build a large, real-world MACP dataset where the correct label truly depends on the country. They propose CountryReward, a VLM-based judge with three parts: CC-RAG (country-specific retrieval to “amplify” the country signal), Country Adapter Module (FiLM-style conditioning of visual features by country), and Focus-Driven Penalty Loss (extra loss on misclassified samples that didn’t focus on country/product/image tokens). On MACP, this model outperforms strong VLM baselines and, used as a reward model, can further improve a generative ad-design pipeline toward country-specific preferences.

### Strengths
- The authors contribute a large, realistic multi-country ad-preference dataset, which is rare in public VLM work and makes the problem measurable.
- Ablations match the story: removing any of the three parts notably drops accuracy/sensitivity, so the claimed mechanism is empirically supported.
- Showing that the improved judge can drive a generative ad pipeline (as a reward model) suggests impact beyond classification.

### Weaknesses
- The core phenomenon (SCVO) is very close to existing ideas on modality/attention imbalance and FiLM-style conditioning, but the paper does not run side-by-side comparisons with these simpler, well-known baselines, so the novelty is somewhat overstated.
- Evidence is single-domain and single-attribute (only “country” on one ad/e-commerce dataset), so the claim that SCVO is a general VLM issue is not fully validated.
- The writing and exposition need tightening: some implementation-critical details are only partially described, making replication harder; see the Questions section for specific clarifications requested.

### Questions
- Clarify whether CC-RAG is used only offline (precompute per sample) or also queried at inference/deployment time, and what the cost is in the latter case.
- The CC-RAG retrieval pipeline description is unclear: the paper introduces *k* (text-level top-k) and *m* (image-level top-k), then uses *n* in the weighted aggregation, but does not explain how *n* relates to *m*.
- Table 2 (CountryReward Evaluation on generated images) does not clearly explain how accuracy is computed. The paper should explicitly state whether generated A/B images are re-scored by the fixed CountryReward model and compared against the original MACP ground-truth labels to form standard A/B accuracy, or whether raw reward scores are averaged instead.
- The paper contains numerous typos: "PENALITY", "Gneration", "mtigate", "sparese", "bacground", "oopti-mized".

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper investigates Sparse Critical Variable Overwhelm (SCVO) in vision-language models, a problem where models focus on dominant features and ignore sparse but important cues such as country names. To study this, the authors introduce the MACP dataset, which contains multi-country ad click preferences with product details, paired ad images, target country information, and observed click behavior. Existing models like Qwen-VL tend to overlook the country cue and make the same predictions across markets. To overcome this limitation, the authors propose CountryReward, a multimodal judge model that combines three key components: Cross-Country Retrieval Augmentation (CC-RAG) for adding market-specific click history as contextual text, a Country Adapter Module (CAM) that adjusts image representations based on country embeddings, and a Focus-Driven Penalty Loss (FDPL) that applies stronger penalties to errors involving overlooked cues. CountryReward achieves better accuracy and sensitivity on the MACP benchmark compared to standard vision-language models and is further used as a reward model to fine-tune a generative model. This reinforcement learning process produces country-tailored ad designs that align more closely with local preferences.

### Strengths
- The paper identifies the SCVO issue in vision-language models and introduces the MACP dataset, a large-scale multi-country ad preference dataset that enables studying cross-market multimodal behavior using real-world data.

- The proposed CountryReward framework integrates three components, including CC-RAG for market-specific context, a Country Adapter for visual feature adjustment, and a Penalty Loss for focusing on sparse cues, forming a coherent approach to address SCVO.

### Weaknesses
- Although the combination is novel, the individual components draw on existing ideas. Retrieval-Augmented Generation (RAG) and adapter modules are known techniques, and weighted loss terms are common in training. The paper would benefit from deeper discussion on why these known techniques synergize specifically for SCVO, and comparison to more baselines (e.g., domain-adaptive fine-tuning, or simple prompting strategies).


- The RL-based image generation evaluation relies solely on the CountryReward model’s own scores. While higher reward scores suggest images are more aligned to predicted preferences, there is no external validation (e.g., human study or click-through simulation) to confirm that these images are indeed better. The reliance on the same model for evaluation risks circularity. Demonstrating improvement in real user metrics or including a user study on a sample of generated images would strengthen the claim of practical impact.

- The approach presumes the availability of rich historical click data for each market (used in CC-RAG). In settings where such data are sparse or outdated, CC-RAG may be less effective. Moreover, the MACP dataset, while large, comes from one platform – it would be useful to know its diversity (product categories, user demographics) to assess generality. No analysis is provided on whether the model truly attends to the country token (e.g., via attention weights) or on failure cases where even CountryReward errs.

---

**typo errors:**

yiedling - L 027

mtigates - L 110

### Questions
Please read the weaknesses section.

### Soundness
2

### Presentation
2

### Contribution
2
