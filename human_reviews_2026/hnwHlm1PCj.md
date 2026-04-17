# Boost the Identity-Preserving Embedding for Consistent Text-to-Image Generation

- Decision: Reject
- Scores: 4, 4, 6

## Abstract
Diffusion-based text-to-image (T2I) models have advanced high-fidelity content generation, but their inability to maintain subject consistency—preserving a target’s identity and visual attributes across diverse scenes—hampers real-world applications. Existing solutions face critical limitations: training-based methods rely on heavy computation and large datasets; training-free approaches, while avoiding retraining, demand excessive memory or complex auxiliary modules. In this paper, we first reveal a key property overlooked in prior works that the identity-relevant signals, termed Identity-Preserving Embeddings (*IPemb*), are implicitly encoded in textual embeddings of frame prompts. To address the consistent T2I generation with the *IPemb* embedding, we propose Boost Identity-Preserving Embedding (*BIPE*), a training-free yet plug-and-play framework that explicitly extracts and enhances the *IPemb*. Its core innovations are two complementary techniques: Adaptive Singular-Value Rescaling (*adaSVR*) and Union Key (*UniK*). *adaSVR* applies singular-value decomposition to the joint embedding matrix of all frame prompts, amplifying identity-centric components (dominant matrix features) while suppressing frame-specific noise; crucially, it is integrated into every text encoder transformer layer to prevent *IPemb* dilution during non-linear feature transformations. *UniK* further reinforces consistency by concatenating cross-attention keys from all frame prompts (not just the current one), aligning the T2I backbone’s image-text attention across the entire generation sequence. Experiments on the *ConsiStory+* benchmark demonstrate *BIPE* outperforms state-of-the-art methods in both qualitative and quantitative metrics. To address the gap in evaluating a broader range of scenarios with diversified prompt templates, we introduce *DiverStory*, which confirm the scalability of *BIPE*.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose BIPE (Boost Identity-Preserving Embedding), a training-free for consistent text-to-image generation. The approach focuses on identity-preserving embeddings (IPemb) and introduces two techniques: Adaptive Singular-Value Rescaling (adaSVR) and Union Key (UniK). AdaSVR applies singular value decomposition to amplify identity-related components. UniK enhances consistency by concatenating cross-attention keys from all frame prompts. BIPE uses SDXL as the base model and is evaluated on ConsiStory+ and a newly proposed DiverStory benchmark.

### Strengths
1. The paper analyzes and visualizes the relationship between identity-preserving embeddings and attention mechanisms focused on the subject.
2. The authors design the DiverStory benchmark, which employs varied natural language prompt formulations rather than relying on a single fixed template as in ConsiStory+.
3. The paper provides numerous visual examples to illustrate results.

### Weaknesses
1. The motivation is not entirely clear. The authors claim that previous works overlook the fact that identity-relevant embedding components are already implicitly encoded within the aggregated textual embeddings of a full frame-prompt sequence. However, this limitation does not seem particularly significant, nor is it obvious that it would strongly affect results.
2. The description of the method is difficult to follow and not clearly structured. It required substantial time and effort to understand the novelty of BIPE and how it differs from 1Prompt1Story. Nevertheless, the proposed approach appears quite similar to 1Prompt1Story. For instance, the UniK component in BIPE seems analogous to Prompt Consolidation (PCon) in 1Prompt1Story, as both combine all prompts. Likewise, adaSVR in BIPE appears to resemble Singular-Value Reweighting (SVR) in 1Prompt1Story. The primary difference seems to lie in the explicit use of IPemb in BIPE. However, the distinction between explicit use in adaSVR and implicit use in SVR is not clearly explained.
3. The paper contains several typos. For example, $\bar{V}_i$ should be $\tilde{V}_i$ in Eq. (5). In Table 1, “Train-Free” should be “Train”, or “√” and “×” should be interchanged.

### Questions
Since BIPE is applied to video generation in the experiments, would it be more accurate to use the term “consistent visual generation” rather than “consistent text-to-image generation”?

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper tackles identity preservation in text-to-image (T2I) diffusion. It observes that a cross-frame, identity-bearing direction exists in the text-encoder embeddings. Building on this, the authors propose BIPE, a training-free, plug-and-play framework with two parts: Adaptive Singular-Value Rescaling (adaSVR) and Union Key (UniK). The paper also proposes DiverStory, a benchmark using varied natural-language prompts (not a single template), and reports gains on ConsiStory+ and DiverStory with moderate runtime/memory overhead.

### Strengths
- Operating purely on text embeddings makes BIPE easy to attach to SDXL-like pipelines; the paper also shows a video case (Wan 2.2).
- The IPemb observation (leading singular directions capture identity) is plausible and supported by attention-map probes.
- On ConsiStory+, BIPE achieves the best CLIP-T and VQA, with identity metrics close to the best and better efficiency than training-heavy baselines; ablations indicate complementary roles for adaSVR and UniK.
- DiverStory highlights robustness to varied natural-language prompts which is a realistic setting often under-tested.

### Weaknesses
- Evidence suggests BIPE helps on both template-based and diverse prompts, but several core claims and implementation details are insufficiently justified (see “Questions”).
- The empirical methodology is mostly standard, but ablations don’t fully isolate design choices (e.g., sensitivity to the weighting temperature, the role of per-layer SVD).

### Questions
1. Is any finetuning performed anywhere (text encoder/adapters)? If truly training-free, please correct the Table-1 flag; if not, specify what is trained and where.
2. Do UniK keys/values come from adaSVR-enhanced embeddings (as in the main text) or the original embeddings (as suggested in the appendix)? Please standardize and report the performance delta between the two setups.
3. Provide per-layer SVD dimensions and a wall-clock/VRAM profile that separates the costs of adaSVR vs. UniK, and how these scale with number of frames (N) and subject token count.
4. Include sweeps for the temperature (\tau) in Eq. (3) and the number of frames (N); additionally, report robustness to token selection (([EoT]) vs. subject tokens) and layer-wise on/off.
5. Any human studies on identity consistency under Diverse Prompts? What is the release timeline/spec for DiverStory to enable community verification?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes BIPE, a training-free, plug-and-play framework that improves subject identity consistency in multi-image text-to-image generation by operating purely on text embeddings. BIPE has two components: adaptive singular-value rescaling (adaSVR), which spectrally amplifies identity-preserving directions in subject and [EoT] token embeddings across every layer of the text encoder, and Union Key (UniK), which concatenates cross-attention keys from all prompts while using per-frame values to align attention without leaking full values across frames. Experiments on ConsiStory+ and a new Diverse Prompts benchmark, DiverStory, show strong text alignment and competitive identity consistency with low memory and runtime overhead, and the authors also illustrate integration into Wan 2.2 for cross-video consistency.

### Strengths
Originality is solid: rather than new networks or retraining, the work identifies and boosts an intrinsic identity-preserving component in text embeddings and enforces consistency via key-sharing in cross-attention, which is simple and broadly applicable. Quality is supported by clear math for adaSVR with energy-matched normalization, principled token selection for subject and padding tokens, and a practical 1/N weighting of extra key-value pairs to control dominance and cost. Clarity is generally high, with an end-to-end pipeline and ablations that isolate adaSVR vs UniK contributions. Significance is promising since BIPE is architecture-agnostic, requires no additional data or training, and achieves strong alignment and competitive identity metrics with near-base latency, while DiverStory broadens evaluation beyond templated prompts.

### Weaknesses
The paper claims BIPE is training-free, yet Table 1 marks BIPE as not training-free on both ConsiStory+ and DiverStory, which conflicts with the text and should be corrected or explained. The evaluation emphasizes SDXL as the default and shows case studies with Wan 2.2, but broader quantitative tests on additional backbones would better support the architecture-agnostic claim. Identity consistency is mostly measured by CLIP-I and DreamSim with background removal; a small human study or per-attribute identity analysis would strengthen conclusions on visual identity. Finally, while the method uses only a subset of tokens in UniK to cap compute, sensitivity to the number and type of shared tokens, and scaling with the number of frames N, is not systematically profiled.

### Questions
a) Please reconcile the training-free claim with Table 1, which currently lists BIPE as not training-free. If this is a typesetting error, clarify and update; if not, explain what part of BIPE requires training. 

b) How does BIPE scale in runtime and memory with the number of frames and with the count of shared keys in UniK? A plot of latency and VRAM vs N and vs number of shared subject/[EoT] tokens would help practitioners. 

c) Beyond SDXL and the Wan 2.2 illustration, can you report quantitative results on at least one non-CLIP text encoder or a DiT-based T2I backbone to substantiate architecture-agnostic claims. 

d) Could you add sensitivity studies for adaSVR’s temperature and the decision to include [EoT] alongside subject tokens, plus an ablation on the 1/N weighting strategy. 

e) DiverStory is valuable; can you provide statistics on prompt diversity and subject types, along with plans and licensing for release, so the community can reproduce and extend your results. 

f) The limitations note that BIPE does not accept external identity references; can you outline how BIPE would integrate with reference-image encoders or identity embeddings while retaining the training-free property.

### Soundness
3

### Presentation
3

### Contribution
3
