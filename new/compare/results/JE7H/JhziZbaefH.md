---
job_id: 0dba95ec-ce0d-49c8-bfe3-9b679b4e7501
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: JhziZbaefH.pdf
paper: Online Multimodal Learning with Human-in-the-Loop
main_score_norm: 0.4
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper proposes an online multimodal representation learning architecture with lifelong learning and human-in-the-loop interaction, which clearly fits ICLR’s core topics (representation learning, continual learning, multimodal learning, human feedback).

## Minimum Quality
Pass ✅.  
All required sections (Abstract, Introduction, Related Work, Method, Experiments, Results, Conclusion) are present. The paper is in English and presents a reasonably complete method and experiments. While there are notable technical and empirical weaknesses, they do not rise to the level of an automatic desk rejection according to the given rules.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I see no hidden prompts, meta-instructions to LLMs, or other manipulative content in the main paper text.

---

# Expected Review Outcome:

## Summary

The paper introduces OML, a brain-inspired online multimodal learning architecture with three hierarchical layers (feature, unimodal association, multimodal association) and ascending, descending, and lateral pathways (Figure 2). Different neuron types (feature neurons, unimodal association neurons with order-dependent/independent modes, and multimodal association neurons) communicate via oscillatory-like encoded signals and Fourier transforms, and a reference extraction mechanism aims to identify which feature subspace (e.g., color vs shape) a word refers to. The system also performs conflict detection between current multimodal inputs and previously learned associations, can query a human user on conflicts, and updates connections accordingly; experiments on several small fruit datasets compare OML to older multimodal retrieval baselines and prior online models such as ART and AEN.

---

## Strengths

1. **Ambitious, structured online multimodal architecture**  
   The paper proposes a fairly detailed, hierarchical and modular architecture (Figure 2) that explicitly separates feature, unimodal association, and multimodal association layers, and further distinguishes order-independent (e.g., visual) and order-dependent (e.g., auditory) unimodal association. This is conceptually richer than many simplistic online multimodal binding schemes and targets continual learning, cross-modal recall, and extension to new modalities.

2. **Reference extraction idea for attributive words**  
   The reference extraction algorithm in Section 3.4 attempts to distinguish which components of a visual feature vector a word actually “refers” to by monitoring per-dimension variance and coefficient of variation across samples. This is a concrete mechanism, directly tied to Equations in Section 3.4, to operationalize the claim that a color word should stabilize on color features but not on shape features. Figure 3(a) provides a helpful example illustrating that the learned representation of “hóng sè” should be tied to color rather than shape.

3. **Explicit modeling of order-dependent vs order-independent channels**  
   The distinction between OIAM and ODAM channels in Section 3.2 (with different connection tensors and activation functions, Eq. (3) vs Eq. (5)) is a sensible design choice that acknowledges temporal structure in auditory sequences versus unordered feature sets in visual inputs. This separation is relatively clear in Figure 2 and in the equations.

4. **Conflict detection and human-in-the-loop protocol**  
   The paper goes beyond standard continual or multimodal learning by formalizing several conflict scenarios (Section 3.5, four cases) where current image–word pairs are inconsistent with previously learned associations. The mechanisms using lateral neighborhoods (e.g., sets \(G_p^b, G_q^c\) on Page 7) and intersection checks to decide when to pose questions to the user are explicit and reasonably well specified. This is more concrete than the sometimes vague “human feedback” narratives in related work.

5. **Support for modality extension**  
   The paper demonstrates a protocol for extending an already-trained visual–auditory system with a new taste modality (VAT and VAT-HomeF datasets). Table 3 shows that OML achieves consistently higher accuracy than the prior online multimodal extension model AEN across all pairwise recall directions (T↔V, T↔A, V↔A), especially in the open environment, which is aligned with the claimed strength in handling new modalities online.

6. **Empirical results are consistently competitive among online methods**  
   Across Tables 1–3, OML is generally the best-performing *online* method (ART, AEN, OML). In Table 1 (baseline experiment), OML is close to the offline state-of-the-art in the close setting and surpasses all offline methods in the open setting, which supports the claim that the architecture mitigates catastrophic forgetting on these small benchmarks. In Table 2 (precise referring experiment), OML again dominates both offline and online baselines, which is in line with the intended advantage of the reference extraction mechanism.

7. **Figures reasonably clarify high-level mechanisms**  
   Figure 2 is particularly useful in laying out the network’s macro-structure, visually distinguishing frozen backbones from learnable neurons, the three layers, and the different channels. Figure 3(b) helps readers see how ascending and descending signals interact across visual and auditory channels during conflict checking and recall, which would otherwise be quite opaque given the heavy notation.

---

## Weaknesses

1. **Highly hand-crafted, ad hoc signal model with unclear necessity and scalability**  
   The neuron definitions rely on a rather baroque oscillatory encoding and Gaussian thresholds that feel engineered rather than principled. For example, the FN ascending activation (Eq. (1)) is
   \[
   \boldsymbol{y}^{\alpha_k} = \sum_{i=1}^n \sum_{t=1}^T w_{j,i} \cos \lambda_i^{\alpha_k} 2\pi \frac{t-1}{T}
   \]
   when \(d(\mathbf{x},\mathbf{w}_j)\le\theta\), otherwise 0. This effectively collapses an \(n\)-dimensional input into some aggregated oscillatory signal, but the paper never explains why this representation is preferable to simpler similarity scores, how it behaves numerically as \(n\) or \(T\) grows, or how information about \(\mathbf{x}\) is preserved. Similar issues arise with the MAN activation (Eq. (6)), which applies a Fourier transform \(\mathcal{F}\) to the UAN output but never justifies why a Fourier domain representation is needed beyond an analogy to brain oscillations. The introduction of frequency parameters \(\lambda_i^{\alpha_k}\) as “unique natural numbers” is also arbitrary and not experimentally ablated. This ad hoc signal design raises concern about whether the system would scale beyond these toy-feature settings and whether the performance gains truly stem from the architecture rather than from dataset idiosyncrasies.

2. **Insufficient comparison to modern multimodal and continual learning baselines**  
   The empirical comparison focuses heavily on older cross-modal retrieval baselines (DAE, DBM, DJSRH, NRCH, FUME) and a couple of neuro-inspired online models (ART, AEN). There is no comparison against more recent multimodal transformers or continual multimodal learning methods that are standard in today’s literature (e.g., CLIP-like architectures with incremental fine-tuning, replay-based continual learning, or parameter-isolation methods for multimodal data). This omission weakens the claim that OML is competitive for “online multimodal learning” in a contemporary sense, and makes it difficult to judge the practical significance of the results. On Page 8, the authors claim that OML is “stable and achieves the highest accuracy” in the open setting; this is true only relative to a limited set of baselines that largely predate modern multimodal representation learning.

3. **Limited and narrow experimental scope**  
   All experiments are conducted on small-scale fruit datasets: Fruits, HomeF (fruit subset of Lai et al. 2011), and their color-word-augmented variants, plus VAT and VAT-HomeF with taste. These are narrow, low-variability domains with very simple visual structure and small label vocabularies, which are particularly amenable to hard-coded feature descriptors (Fourier boundary descriptors plus mean color). The paper uses handcrafted features plus a simple energy/zero-crossing + MFCC pipeline for speech, and low-dimensional taste vectors. While such data match the neuro-inspired narrative, they severely limit external validity. There is no evidence that OML can handle realistic high-dimensional multimodal data (e.g., full-resolution images with complex scenes, natural speech, text) or large vocabularies. The architecture’s complexity (especially the combinatorial connections in Figure 2) raises scalability concerns that the experiments do not address.

4. **Human-in-the-loop aspect is largely simulated and weakly evaluated**  
   The paper advertises “online multimodal learning with human-in-the-loop” in the title, but the actual experiments sidestep real human interaction: “if the question posed to the user by OLM remains unanswered for a certain period of time, we set the answer to be positive” (Page 8). This effectively turns conflicts into automatic positive confirmations, so the training protocol degenerates into a deterministic rule that always links conflicting pairs. There is no study of (i) how often conflicts occur, (ii) the impact of correct vs incorrect user answers, or (iii) robustness to noisy or adversarial feedback. The final statement in Section 4.1 that “when we randomly add 10% of word-image or word-taste data pairs with incorrect matches, OML is able to detect all conflicts and raise appropriate questions” is not supported by quantitative metrics (e.g., precision/recall of conflict detection; number of questions asked) and conflates “raising a question” with actually *learning from human responses*. This significantly weakens the core human-in-the-loop claim.

5. **Reference extraction mechanism is heuristic, and its evaluation is indirect and somewhat forgiving to baselines**  
   The reference extraction in Section 3.4 uses coefficient of variation over aggregated signals \(\mathbf{a}^{V,t}\) to decide which feature subsets a word refers to. Several issues arise:
   - The algorithm thresholds \(\max(\mathbf{r}^{\alpha_j})\) versus \(r\) (Eq. (7)) but does not discuss sensitivity to this threshold or to the number of samples \(n\) required for “variance shrinkage”.
   - It assumes that the referring dimensions have both small variance and sufficiently large mean (since \(r=\sigma/\mu\)), which might fail when color intensities vary strongly across illumination or when shapes/colors are correlated.
   - In experiments (Table 2), ART and AEN are effectively *given credit* for returning all features (shape + color) when queried with a color word like “hóng sè”, which the authors “count as a correct result”. This choice artificially narrows the observed benefit of precise referring. A stricter evaluation that truly checks whether only color features are recalled is not reported, so the quantitative impact of reference extraction remains unclear. 
   - There is no ablation of the reference extraction component: we never see OML without \(re(\cdot)\), or with simpler heuristics (e.g., using feature-type priors, or a learned classifier), so we cannot attribute the gains in Table 2 purely to this mechanism.

6. **Mathematical specification and notation have several ambiguities and inconsistencies**  
   The mathematical definitions are dense and occasionally inconsistent, which affects reproducibility:
   - In Eq. (1), the output \(\mathbf{y}^{\alpha_k}\) is described as an “activation signal”, but its dimensionality is unclear: the equation is a scalar sum over \(i\) and \(t\), yet \(\mathbf{y}^{\alpha_k}\) is written in bold as if vector-valued. This ambiguity propagates, since \(\mathbf{y}^{\alpha_k}\) is later summed across feature types in Eq. (3).
   - For descending signals, Sections 3.1 and 3.2 define \(\mathbf{A}^{\alpha_k}\) and \(\mathbf{A}^{\beta}\) as vectors of Gaussian random variables, but the origin of their parameters \((\mu_i,\sigma_i)\) is not clearly specified. It is not explicit whether these are learned per-connection, per-neuron, or globally. Eq. (2) uses relative probability densities \(p_i^{\alpha_k}\ge \vartheta\) to decide activation, but there is no training rule for these Gaussians besides a passing reference to incremental updates in Eq. (8) for word neurons only.
   - In Eq. (4), the descending activation \(f_U^d\) of a UAN returns \(\mathbf{a}^{\alpha_k}\), yet this notation is overloaded: previously \(\mathbf{a}^\beta\) was the amplitude-frequency pair for channel \(\beta\). At several points (e.g., before Eq. (5) on Page 5) the text switches between \(\bm{a}^\beta\) and \(\bm{a}^{\alpha_k}\) without making data types or shapes explicit.
   - The ODAM activation (Eq. (5)) invokes \(re(\boldsymbol{\mu},\boldsymbol{\sigma})\), but these statistics are described in Section 3.4 as being computed over *visual* signals for a word. How exactly \(\boldsymbol{\mu},\boldsymbol{\sigma}\) are associated with each word neuron and updated in the ODAM channel is only partially clarified via Eq. (8), and it is not explained how many samples are needed before \(re\) is meaningful.
   These issues collectively make it hard to implement the method faithfully from the current description.

7. **Conflict detection logic is complex and under-justified, with no diagnostic experiments**  
   Section 3.5 enumerates four cases for learning with human-in-the-loop and defines conditions like \({}^A N^b \cap G_p^b \neq \varnothing\). These depend on lateral neighbors defined by \(d(\mathbf{w}_i,\mathbf{w}_j)\le 2\theta\). However:
   - There is no analysis of how sensitive conflict detection is to the choice of \(\theta\) or the lateral connectivity pattern. For instance, a too-large \(\theta\) would make everything neighbors, collapsing conflicts, while a too-small one would over-detect conflicts.
   - The text claims (end of Section 4.1) that when 10% of pairs are mismatched, OML detects all conflicts, but there is no table or figure showing detection rates, false positives, or the effect on downstream recall. This is a central claim but is supported only by a single-sentence assertion.
   - Figure 3(b) visually shows how ascending and descending paths are combined, but there is no empirical visualization of conflict cases or learned lateral neighborhoods (e.g., activation patterns or confusion matrices) to help validate that the abstract conditions in Section 3.5 behave as intended.

8. **Evaluation protocol and metrics are under-specified**  
   The paper states that it tests cross-modal recall tasks (e.g., V→A, A→V) but does not precisely describe the evaluation criterion. It is not clear:
   - Whether accuracy in Tables 1–3 refers to top-1 retrieval accuracy, classification accuracy, or some hit rate over candidate sets.
   - How the open-environment setting is evaluated: after training on all four sequential splits, do they test on the union of all test samples, or only on the last split? How are classes introduced in later splits handled in the metrics?
   - How many runs / seeds are used; Tables 1–3 show single numbers with no variance or confidence intervals. Given the system’s many stochastic or threshold components (e.g., random word–image mismatches, threshold-based activations), this omission makes robustness uncertain.

9. **Limited discussion of computational complexity and memory footprint**  
   OML dynamically creates new FNs, UANs, and MANs, and uses full 0–1 connectivity matrices \(\mathbf{W}^{\alpha_{k}}, \mathbf{W}^{\beta}, \mathbf{U}^{\alpha_k}, \mathbf{U}^{\beta}\) plus lateral matrices \(\mathbf{L}^{\alpha_k}\) (Page 4). There is no complexity analysis; it is unclear how the number of neurons and connections grows with the number of concepts or samples, or whether the method is tractable for larger vocabularies or high-dimensional features. The small fruit datasets may mask scalability issues, but for realistic multimodal tasks the unbounded growth and dense connections could become problematic.

10. **Positioning within human-in-the-loop and multimodal learning literature is incomplete**  
    While the paper covers some multimodal retrieval and online binding works, it does not connect to broader human-in-the-loop learning literature (e.g., interactive ML, RL from human feedback) or to recent multimodal models that incorporate feedback or reasoning about referents. This gap weakens the conceptual framing of what is new about the proposed “human-in-the-loop” mechanism relative to existing frameworks that already support interactive correction or preference-based learning.

---

## Potentially Missing Related Work

1. **Li, J., Miller, A. H., Chopra, S., 2017 – “Dialogue Learning with Human-in-the-Loop”**  
   This work explicitly studies learning from human feedback in interactive dialogue systems. It is directly relevant to the paper’s human-in-the-loop conflict-resolution mechanism and should be discussed in Section 2, clarifying how OML’s question-asking-and-updating protocol differs from or generalizes such interactive RL frameworks.

2. **Wu, X., Xiao, L., Sun, Y., 2021 – “A Survey of Human-in-the-loop for Machine Learning”**  
   A comprehensive survey on human-in-the-loop ML that could help contextualize where OML fits among existing paradigms (e.g., active learning, interactive labeling, RLHF). A short discussion in Section 2 would strengthen the positioning of the human interaction protocol, and perhaps inform a more principled design of the questioning strategies.

3. **Wang, D., Wei, H., Zhang, Z., 2021 – “Non-Parametric Online Learning from Human Feedback for Neural Machine Translation”**  
   This paper tackles online adaptation with human feedback, which resonates with OML’s claim about continuous learning from user responses. It should be cited in Section 2 and briefly compared to OML’s non-parametric growth of neurons and connections, especially on how feedback is represented and incorporated over time.

4. **Vasco, M., Melo, F. S., de Matos, D. M., 2019 – “Learning Multimodal Representations for Sample-Efficient Recognition of Human Actions”**  
   Provides a more modern multimodal representation learning context, including sample efficiency, which could be relevant to the online setting here. It would be helpful to reference it in the multimodal learning part of Section 2 and briefly discuss how OML’s associative architecture differs from such representation learning approaches.

5. **Zhao, H. H., Pei, W., Tao, Y., 2025 – “InterFeedback: Unveiling Interactive Intelligence of Large Multimodal Models with Human Feedback”**  
   This paper investigates how large multimodal models interact with humans and use feedback, directly related to the human-in-the-loop and multimodal themes. Including it in Section 2 and contrasting their interactive capabilities with OML’s hand-crafted conflict queries would improve the paper’s discussion of contemporary alternatives.

6. **Fonteles, J. H., Cohn, C., Ayalon, E., 2026 – “Analyzing Embodied Learning in Classroom Settings: A Human-in-the-Loop AI Approach for Multimodal Learning Analytics”**  
   Although more application-focused, this work shows another instance of human-in-the-loop multimodal systems. It can be cited briefly in Section 2 to show broader relevance and help frame OML’s contribution as a representation-learning method that might in principle be applied to such settings.

---

## Questions

1. **Clarification of signal dimensionality and training of Gaussian parameters**  
   - In Eq. (1), is \(\mathbf{y}^{\alpha_k}\) intended to be a scalar or a vector? If vector, please specify its dimension and how the double sum over \(i\) and \(t\) results in a vectorized signal.  
   - For the descending Gaussians \(A_i^{\alpha_k}\sim \mathcal{N}(\mu_i,\sigma_i)\) in Eq. (2) and \(A_i^{\beta}\) in Eq. (4), how exactly are \(\mu_i\) and \(\sigma_i\) initialized and updated during learning? Are they per-neuron, per-connection, or global per-feature-type? An explicit update rule analogous to Eq. (8) for these would greatly help.

2. **Evaluation details for Tables 1–3**  
   - What exactly does “accuracy” mean for V→A, A→V, V→T, etc.? Is it top-1 retrieval accuracy over the entire training vocabulary, classification accuracy over test pairs, or something else?  
   - For the open environment, at which point is evaluation done (after each split or only after all four splits)? Are test sets split by class or by instance, and how many samples per class are available?

3. **Ablations for key components**  
   Could you provide ablation experiments on at least the following:
   - OML without the reference extraction function \(re(\cdot)\) (e.g., returning all features as baselines do), to quantify its contribution in Table 2.  
   - OML without lateral connections (i.e., ignoring \(\mathbf{L}^{\alpha_k}\)), to show their effect on generalization and conflict detection.  
   - OML with simpler similarity signals (e.g., dot product or cosine similarity instead of Eq. (1) and Fourier-based MAN activation) to demonstrate that the complicated oscillatory encoding is actually necessary.

4. **Quantitative evaluation of conflict detection and human interaction**  
   - For the experiment with 10% mismatched pairs, can you report quantitative metrics: fraction of truly mismatched pairs detected, false positives on matched pairs, and the number of questions asked per sample?  
   - How would performance change if some human responses are wrong (e.g., 10–20% incorrect “yes” or “no” labels)? A small synthetic study could greatly strengthen the human-in-the-loop claim.

5. **Scalability and complexity**  
   - Can you provide empirical statistics on the final number of FNs, UANs, MANs, and connections for the largest experiment, and estimate how these numbers would scale with number of concepts or feature dimensionality?  
   - Is there any mechanism to prune or merge neurons to control growth, or is the growth unbounded? If unbounded, how do you envision applying OML to larger-scale multimodal datasets?

Addressing these questions with additional experiments or clarifications would significantly increase confidence in the method and its claimed advantages.

---

## Flag For Ethics Review

- No ethics review needed.  

---

## Details Of Ethics Concerns

N/A.

---

## Soundness Rating
2: fair.  
The method is specified in detail but uses ad hoc signal encodings and thresholds, has several notational ambiguities, and lacks ablations and rigorous evaluation of key claimed advantages (especially human-in-the-loop and conflict detection). The experimental setups are reasonably executed but limited in scope and insufficient to fully support broader claims.

---

## Presentation Rating
2: fair.  
The high-level architecture and figures (especially Figure 2 and Figure 3) are helpful, but the mathematical exposition is dense and at times inconsistent, important implementation details (e.g., how Gaussians are updated, evaluation metrics) are missing, and the human-in-the-loop setup is not clearly operationalized in experiments.

---

## Contribution Rating
2: fair.  
There are interesting ideas (hierarchical multimodal architecture, reference extraction, explicit conflict handling) and competitive performance on small online multimodal benchmarks, but the reliance on toy datasets, lack of comparison to modern baselines, and weakly substantiated human-in-the-loop claims limit the overall impact.

---

## Overall Rating
4: marginally below the acceptance threshold. But would not mind if paper is accepted.  
The work is ambitious and contains several creative design elements, and it performs well relative to the chosen baselines. However, substantial concerns about the ad hoc nature of the signal model, incomplete mathematical clarity, narrow and somewhat forgiving experiments, and insufficient evaluation of the human-in-the-loop and conflict detection aspects prevent it from meeting ICLR’s bar in its current form.

---

## Reviewer Confidence
4: confident.  
I am reasonably familiar with multimodal and continual learning, have carefully read the equations and experimental sections, and I am confident about the main technical and empirical criticisms, though some architectural and biological analogy aspects could benefit from author clarification.