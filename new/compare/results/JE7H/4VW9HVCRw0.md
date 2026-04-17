---
job_id: 4914b014-deee-4a58-b3c8-81b2515c3c9f
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: 4VW9HVCRw0.pdf
paper: TOUCH: Text-Guided Controllable Generation of Free-Form Hand-Object Interactions
main_score_norm: 0.8
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.  

## Topic Compatibility
Pass ✅.  
The paper proposes a text‑conditioned generative diffusion framework and a new 3D HOI dataset, squarely within generative modeling, representation learning for vision, and datasets/benchmarks, all clearly relevant to ICLR.

## Minimum Quality
Pass ✅.  
All required sections are present (Abstract, Introduction, Related Work, Method, Dataset, Experiments/Results, Discussion, Conclusion). The work is technically non‑trivial, in clear English, with substantial experiments and no obvious fatal methodological flaw or misuse of evaluation.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I see no hidden instructions or attempts to manipulate automated reviewing; all text is normal scientific content.

---

# Expected Review Outcome:

## Summary
The paper introduces the task of Free‑Form Hand‑Object Interaction (HOI) generation, aiming to go beyond grasp‑centric settings to include diverse non‑grasping interactions (e.g., pushing, poking, lifting, rolling).  

To support this, the authors build WildO2, an in‑the‑wild 3D HOI dataset reconstructed from Something‑Something‑V2 videos via a semi‑automatic O2HOI pipeline that recovers object meshes, hand pose, and dense contact maps, plus multi‑level text annotations.  

On top of WildO2, they propose TOUCH, a three‑stage framework: (1) contact‑map prediction with CVAEs conditioned on geometry and fine‑grained text; (2) a multi‑level conditioned diffusion model with coarse‑to‑fine semantic and geometric injection; and (3) a refinement module with cycle‑consistent contact and physical losses. Experiments show improvements over adapted baselines on contact accuracy, physical plausibility, diversity, and semantic alignment.

## Strengths
1. **Clear problem shift beyond grasps, well motivated and concretely instantiated.**  
   The paper articulates the limitation of existing HOI generators being biased toward stable grasps and defines a “Free‑Form HOI generation” task that explicitly includes non‑grasping actions and richer contact patterns (Intro, Fig. 1). This is not just rhetorical; the dataset statistics in **Figure 3(b,c)** show a broad coverage of 92 action categories and diverse hand parts including dorsal contacts, underlining that the setting is qualitatively different from standard grasp benchmarks.

2. **WildO2 dataset and reconstruction pipeline are thoughtfully designed and fairly well documented.**  
   The O2HOI frame pairing and object‑only mask transfer (Sec. 3.1, **Figure 2**) are a clever way to circumvent aggressive inpainting or manual masks. The three‑stage reconstruction (object via InstantMesh; camera alignment via differentiable rendering with the composite loss in **Equation (1)**; then hand refinement with 2D/3D and physical terms in **Equation (2)**) is technically non‑trivial. **Figure 3(a)** plus the failure analysis in the appendix (Fig. 14, Table 6) give a good sense of reconstruction success rates and failure modes. The dataset comparison in **Table 8** convincingly positions WildO2 as more diverse and more “daily‑life” than lab datasets like HO‑3D, GRAB, or OakInk.

3. **Model architecture leverages contact as an explicit intermediate representation in a principled way.**  
   The separate hand and object CVAEs (Sec. 4.1) conditioned on point clouds and DSC text are a nice design choice to factor high‑dimensional pose into lower‑dimensional, semantically controllable contact priors. The coarse‑to‑fine injection strategy in the diffusion model (Sec. 4.2, **Figure 4**, **Equations (4)–(6)**) is conceptually clean: early blocks use SSCs + global geometry; deeper blocks use DSCs + local contact features. The ablation in **Table 2** and **Figure 6** provides evidence that both contact maps and multi‑level conditioning matter.

4. **Quantitative improvements over baselines on several meaningful metrics.**  
   On WildO2, **Table 1** shows TOUCH outperforming ContactGen and Text2HOI across contact accuracy (P‑IoU, P‑F1), physical metrics (MPVPE, PD, PV), and semantic metrics (P‑FID, VLM score, perceptual score). The gains are substantial in physically relevant metrics (e.g., MPVPE 2.97 vs 4.69 and 5.46; PV roughly halved). On OakInk‑Shape (**Table 5**), the method still outperforms both baselines, which suggests the learned representation generalizes reasonably beyond the custom dataset.

5. **Strong and varied qualitative evidence of semantic controllability.**  
   The visual comparisons in **Figure 5** show TOUCH generating non‑grasping and nuanced interactions (e.g., precise tip contacts, one‑sided lifts) that better match the SSC/DSC prompts than baselines, which tend to default to generic grasps. **Figure 8** demonstrates fine control of contact regions for the same object (pen) under different verbs; **Figure 9** nicely illustrates learned associations between force adjectives (“firm” vs. “gentle”) and contact area. **Figure 11** in the appendix is particularly insightful in showing how omitting detailed contact guidance leads to reversion to grasp priors, supporting the claimed importance of DSC‑level supervision.

6. **Non‑trivial ablation and analysis, including architecture choices.**  
   The ablations in **Table 2** and **Table 4** dissect several components: absence of hand/object contact maps (×hoc), removal of the refiner or cycle loss (×refiner, ×L_cyc), collapsing multi‑level architecture (×mul), and using only DSCs or SSCs. The coarse/fine layer split study (Table 4) is a nice touch, showing that a 4/4 allocation balances global semantics and local geometry, with both 0/8 and 6/2 being inferior. The force‑semantics experiment and action‑specific breakdown in **Table 3** give further texture to how different verbs affect metrics.

7. **Overall clarity is above average, with many implementation details exposed.**  
   The paper gives explicit training details (epochs, optimizers, hand part encoding, loss weights in **Table 9**), contact computation algorithm (App. A.2.6), and selection pipeline for videos (Table 6). This makes the work more reproducible than many complex generative pipelines.

## Weaknesses
1. **Limited and somewhat unbalanced baseline coverage for the new task.**  
   For the core WildO2 evaluation (**Table 1**), only two baselines are considered: ContactGen and Text2HOI, both adapted from grasp‑oriented or temporal settings. Given the current diffusion‑based HOI ecosystem, this is narrow. There are closely related recent HOI diffusion works (e.g., ones that perform text‑driven hand‑object or body‑object synthesis) that are not included, either as baselines or at least in a qualitative comparison. This makes it harder to judge whether TOUCH improves over the state‑of‑the‑art on controllable HOI generation in general, versus only over these two specific methods. Given the strong claims about breaking grasp priors, a wider comparison would materially strengthen the empirical case.

2. **Dataset quality and bias are not rigorously quantified, and manual curation is quite heavy.**  
   While the reconstruction pipeline is well explained, there is relatively little quantitative evidence about the actual accuracy of 3D hand and object geometry or contact maps. **Figure 3(a)** and the failure breakdowns in the appendix mostly report coarse “success vs. failure” categories. There is no comparison between recovered hand poses and independent 3D ground truth on any subset, nor are penetration or alignment errors for the reconstructed “GT” reported. Since TOUCH is trained and evaluated on this data, potential systematic biases (e.g., persistent small misalignments, over‑smooth objects, missing thin structures) could strongly shape the learned contact priors. The heavy human-in-the-loop curation (≈50s per sample, Sec. A.2.1) raises concerns about subtle annotator bias in what is kept versus discarded. A small‑scale human rating of reconstruction fidelity, or a sanity check on contact correctness, would make the dataset’s reliability clearer.

3. **Certain mathematical definitions and loss formulations are underspecified or slightly inconsistent.**  
   - **Equation (1)** lumps several complex differentiable rendering terms into a single scalar loss without detailing normalization or weighting between mask, Sinkhorn, edge, depth, and RGB terms, except for a single λ_fine. Since these components can be on very different scales, it is hard to judge whether the optimization is well‑posed or if one term dominates, particularly during the transition stage.  
   - In **Equation (2)**, $L_{\mathrm{icp}}$ is introduced as an ICP loss on the 3D contact zone, but the paper does not clarify whether this is point‑to‑plane or point‑to‑point, how correspondences are recomputed during optimization, or how it is balanced against $L_{\mathrm{contact}}$ and $L_{\mathrm{pene}}$.  
   - The cycle‑consistency formulation in **Equation (7)** appears to contain a typo or indexing error: the expectation is written over $\mathbf{P}_s \in \mathbf{P}_{C_O}$ but then uses $\mathbf{P}_o$ inside $\Phi(\Psi(\mathbf{P}_o))$, which is undefined. This makes the precise implementation ambiguous.  
   - For the CVAE contact prediction loss in **Equation (3)**, the focal and dice losses are mentioned but not specified (e.g., what class balancing parameters or focusing parameter γ are used), nor is the thresholding strategy to obtain binary contact from probabilistic outputs described. These omissions hinder both reproducibility and careful scrutiny of the optimization behavior.

4. **Evaluation of “free‑form” aspect is still somewhat constrained and lacks strong user studies or semantic probing.**  
   The paper claims to handle a broad variety of non‑grasping interactions, but most evaluation is either global (averaged over all actions) or uses generic metrics like MPVPE and PD that do not directly capture whether the *intended* non‑grasp pattern is followed. While the action‑specific breakdown in **Table 3** is helpful, it primarily reports the same low‑level metrics; there is no systematic semantic evaluation where users or a VLM must distinguish “push vs. poke vs. lift vs. roll” given synthetic HOI and text. The VLM and human perceptual scores in **Table 1** are aggregated, and the evaluation protocol is only described very briefly; it is unclear how many prompts per interaction type, how raters are instructed, or how inter‑rater agreement looks. This makes it hard to be fully convinced that semantic controllability, especially for non‑grasp actions, is robust and not just anecdotal as seen in selected figures.

5. **The reliance on Qwen‑7B and VLMs is heavy, but the analysis of their failure modes and biases is shallow.**  
   The method uses Qwen‑7B both for DSC generation (dataset annotation) and as the text encoder in TOUCH. This closed loop raises a risk of “self‑consistency” rather than genuine semantic grounding: the model might mostly optimize for the idiosyncrasies of Qwen’s space of descriptions. The ablation in **Table 2** replacing the encoder with CLIP/BERT/MPNet is useful, but differences are modest on several metrics (e.g., P‑IoU 0.713 vs 0.705 vs 0.704), and there is no deeper analysis of where Qwen truly helps. Similarly, the VLM‑based semantic evaluation is not specified (which model, prompt design, robustness). A more critical discussion of how encoder/VLM choices shape results, or some cross‑encoder evaluation (e.g., measuring semantic scores with a different model than the one used in training/annotation), would be valuable.

6. **Scope of physical plausibility checks is limited and excludes dynamic or stability aspects.**  
   While the authors rightly argue that grasp‑stability metrics (force‑closure simulation) are inapplicable to free‑form interactions, the current physical metrics focus only on geometric penetration and pose error. There is no consideration of frictional plausibility, torque balance for lifts, or whether interactions like “push” respect object support planes. For example, in **Figure 5** and some of the qualitative examples in the appendix, many interactions visually appear plausible, but there is no explicit check that the resulting hand configuration would not require unphysical bending forces. The anatomical loss in **Equation (2)** and the self‑penetration term in **Equation (7)** help, but their empirical effect is not isolated apart from a single “×L_cyc” row in **Table 2**.

7. **Some architectural and training choices are complex relative to dataset size, raising overfitting questions.**  
   WildO2 has 4.4k interactions, which is not huge given the capacity of an 8‑layer transformer diffusion model and a refiner, plus two CVAEs, plus a large text encoder. The paper mentions a 4:1 train/test split but no explicit validation set for hyperparameter tuning or early stopping; there is also no reporting of training curves or overfitting diagnostics. The per‑verb metrics in **Table 3** hint that performance varies substantially across actions; more explicit evidence (e.g., cross‑validation, or performance as a function of training set size) would help justify that the model is learning robust priors rather than overfitting to this particular dataset.

8. **Some notation and text are occasionally sloppy or inconsistent.**  
   There are scattered typos (e.g., “Environmnet” in **Table 8**, inconsistent subscripts like $M^{H}_{\text{hoi}}$ vs $M_{\text{hoi}}^{H'}$) and a few places where symbols appear without definition (e.g., $G_{DDPM}$ in Fig. 4’s refinement diagram, $N_{\text{tra}}$ not clearly instantiated). None of these are fatal, but they slow down reading and especially make the mathematical parts in Sec. 4.3 and A.2 harder to follow precisely.

Overall, these weaknesses do not undermine the core idea or the main empirical message, but they limit how definitive the conclusions can be regarding generality, dataset quality, and the true degree of semantic/physical grounding.

## Potentially Missing Related Work
Below are directly relevant works that appear to be missing from the references and discussion. They should be cited and contrasted, at minimum in Sec. 2.3 and in the experimental comparison section.

1. **Zuo et al., “GraspDiff: Grasping Generation for Hand-Object Interaction With Multimodal Guided Diffusion,” 2025.**  
   This work uses multimodal (e.g., text and object) guidance within a diffusion framework for generating HOI grasps. While focused on grasping, it is architecturally close to TOUCH’s multi‑level conditioned diffusion and should be discussed as a key baseline and conceptual neighbor. A short comparison of conditioning strategies and an explanation of why GraspDiff is not evaluated (or how it could be adapted to free‑form actions) would clarify novelty.

2. **Ron et al., “HOIDiNi: Human-Object Interaction through Diffusion Noise Optimization,” 2025.**  
   HOIDiNi is a text‑driven diffusion framework for HOI, targeting diverse, realistic interactions. Even if it focuses on full‑body or different data modalities, it is highly relevant to the general problem of text‑conditioned HOI diffusion. It should be cited in Sec. 2.3 and contrasted in terms of representation (hand‑only vs. whole‑body), semantics (free‑form actions), and conditioning mechanisms.

3. **Zhang et al., “HOIDiffusion: Generating Realistic 3D Hand-Object Interaction Data,” 2025.**  
   This paper also aims at generating 3D hand‑object interactions with controllable diffusion models. It is especially close to TOUCH’s goal of synthesizing 3D HOI data and should be included as related work and, if feasible, as a baseline (even via qualitative side‑by‑side). Discussion could go in Sec. 2.3 and Sec. 5.2.

4. **Fan et al., “Re-HOLD: Video Hand Object Interaction Reenactment via Adaptive Layout-instructed Diffusion Model,” 2025.**  
   Re‑HOLD uses layout‑conditioned diffusion to reenact video HOI. While the task is different (reenactment vs. text‑driven free‑form generation), its strategy of disentangling hand pose and object configuration provides a complementary angle to TOUCH’s contact‑first design. It should be discussed in Sec. 2.3 and possibly referenced in Sec. 5.4.2 when talking about dynamics.

5. **Dang et al., “SViMo: Synchronized Diffusion for Video and Motion Generation in Hand-object Interaction Scenarios,” 2025.**  
   SViMo addresses synchronized video and motion generation for HOI via diffusion. Even though TOUCH focuses on static poses, this work is relevant for the longer‑term extension to temporal sequences (discussed in Sec. 6). Citing it in the related work on HOI generation and again in the future‑work section would position the paper better in the context of dynamic HOI models.

6. **Juneja & Kumar, “Prompt-Propose-Verify: A Reliable Hand-Object-Interaction Data Generation Framework using Foundational Models,” 2023.**  
   This work uses foundation models to generate and curate HOI data, which is conceptually close to WildO2’s use of VLMs and LLMs for annotation and filtering. It should be discussed in Sec. 2.1 or the dataset section to clarify how WildO2 differs in scale, reconstruction quality, and annotation strategy, and to acknowledge prior art in HOI data generation via LMs.

## Questions
1. **Dataset fidelity and validation.**  
   - Have you run any quantitative or human evaluation of the reconstructed “ground truth” hand and object meshes, beyond the coarse success/failure categories? For example, could you report estimated MPVPE and PD/PV for your reconstructed poses relative to a trusted baseline on a small subset (e.g., lab‑captured sequences or synthetic data)?  
   - Are contact maps manually inspected in any way, or are they fully algorithmic (A.2.6)? Providing even approximate precision/recall of contact detection judged by humans on a small set would increase confidence in training labels.

2. **Clarification and correction of the cycle‑consistency loss in Equation (7).**  
   - Could you please clarify the intended notation in the second term of $\mathcal{L}_{\text{refiner}}$? As written, the expectation is over $\mathbf{P}_s \in \mathbf{P}_{C_O}$, but the expression uses $\mathbf{P}_o$; I suspect this is a typo but it is important to know exactly what is implemented.  
   - Also, how are $\Phi$ and $\Psi$ parameterized in practice? Are they simply nearest‑neighbor mappings or learned MLPs? The text suggests nearest neighbors but this is not explicit.

3. **Evaluation protocol for VLM and perceptual scores.**  
   - For the VLM metric in **Table 1** and **Table 5**, which VLM do you use, and what is the exact scoring prompt? How many samples are evaluated, and are you averaging across random seeds?  
   - For the human perceptual score, how many interactions per method per rater were shown, what instructions were given, and how was order/randomization handled? Inter‑rater agreement statistics or confidence intervals would be very helpful.

4. **Robustness to text variation and out‑of‑domain semantics.**  
   - Beyond the examples in **Figure 7** and **Figure 9**, have you tried paraphrasing DSCs or perturbing contact descriptions (e.g., conflicting mentions of hand parts) to test robustness and failure modes? Some quantitative measure of sensitivity to text changes would support the semantic control claims.  
   - Can the model generalize to verbs not in the 92‑intent set if described by free‑form sentences, or does it rely strongly on the structured DSC template?

5. **Overfitting and data efficiency.**  
   - Given the relatively small dataset (3.7k train samples), can you report train vs. test performance over training epochs, or perhaps a small experiment where you subsample the training set to 50% / 25% to show how metrics degrade? This would clarify whether the architecture is over‑parameterized for the data scale.  
   - Relatedly, do you do any explicit regularization or data augmentation (beyond text dropout and condition dropout) to mitigate overfitting?

Clarifying these points would significantly increase my confidence in the dataset’s reliability, the correctness of the losses, and the robustness of the claimed semantic and physical properties.

## Flag For Ethics Review
No ethics review needed.  

## Details Of Ethics Concerns
N/A. The work relies on an existing public video dataset (Something‑Something‑V2) and does not introduce obviously sensitive personal information or deployment practices. A brief statement on consent/licensing and potential biases in hand/object categories would nonetheless be valuable in the camera‑ready.

## Soundness Rating
3: good.  
The methodology is generally solid and well supported by experiments and ablations, though some mathematical definitions and dataset validation aspects are under‑specified.

## Presentation Rating
3: good.  
The paper is mostly clear and well structured with informative figures (e.g., **Figures 2, 4, 5, 8, 9**), but some notational inconsistencies and missing implementation details (contact losses, cycle‑loss mapping) detract from full clarity.

## Contribution Rating
4: excellent.  
The combination of a new, non‑trivial in‑the‑wild free‑form HOI dataset and a contact‑centric, multi‑level text‑conditioned diffusion framework is a substantial contribution that is likely to be useful for the HOI and generative modeling communities.

## Overall Rating
8: Accept, good paper (poster).  
Despite some gaps in evaluation breadth and dataset validation, the paper makes a strong and timely contribution by pushing HOI generation beyond grasping, providing a useful dataset and a technically interesting contact‑aware diffusion framework with convincing qualitative and quantitative evidence.

## Reviewer Confidence
4: confident.  
I am familiar with HOI, generative diffusion models, and 3D reconstruction pipelines, and I carefully checked the architecture, key equations, and experiments, though I did not re‑implement or reproduce the full pipeline.