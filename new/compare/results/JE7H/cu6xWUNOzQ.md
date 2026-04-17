---
job_id: deaffc86-5779-454a-98ce-23fd1daa0bc3
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: cu6xWUNOzQ.pdf
paper: Aligning the Brain with Language Models Through a Nonlinear and Multimodal Approach
main_score_norm: 0.6
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper is squarely about representation learning for language and audio (Whisper, LLaMA) and their use in nonlinear multimodal brain encoding, which fits ICLR’s core areas (representation learning, multimodal modeling, applications to neuroscience).

## Minimum Quality
Pass ✅.  
The paper is in English and has all required components: Abstract, Introduction, Related-work-style positioning in the Introduction and Discussion, clear Method (Sections 2 and Appendices B, C), Experiments and Results (Section 3, Tables 1–7, multiple figures), and Discussion/Conclusion (Section 4). The methodology is technically sound overall, experiments are extensive with baselines and ablations, and no fatal flaws in evaluation design are apparent.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I see no hidden prompts or attempts to manipulate LLM-based reviewing; the content is standard scientific prose and figures.

---

# Expected Review Outcome:

## Summary

The paper studies nonlinear multimodal encoding models that predict fMRI responses to naturalistic speech, using semantic features from LLaMA and audio features from Whisper. The core model is a relatively simple PCA + single-hidden-layer MLP that takes concatenated text and audio features and predicts PCA-compressed whole-brain activity, which is then reconstructed into voxel space. Compared with standard unimodal linear semantic baselines and the stacked-regression multimodal state of the art, the proposed approach achieves sizable gains in variance explained and normalized correlation, and the authors further analyze spatiotemporal organization via a new Relative Error Difference (RED) metric and variance partitioning.

## Strengths

1. **Strong empirical gains, carefully quantified.**  
   - Table 1 and Table 2 show that the multimodal MLP (text+audio, PCA response) achieves 4.29% average voxelwise \(r^2\) and 34.32% CC\(_\text{norm}\), which is a 17.2% and 17.9% gain over the standard text-only linear baseline (3.66% \(r^2\), 29.12% CC\(_\text{norm}\)).  
   - Table 4 demonstrates that under the single-story evaluation used in Antonello et al. (2024), the same multimodal MLP outperforms the prior stacked regression model by +14.4% CC\(_\text{norm}\) and +7.7% story-specific \(r^2\). These are large improvements by language-fMRI standards, and the authors contextualize them versus prior work in Table 7.

2. **Systematic disentangling of nonlinearity vs. dimensionality vs. multimodality.**  
   - The ablation grid in Table 1 (and replicated per subject in Tables 2–3) systematically varies: unimodal vs. multimodal inputs, linear vs. MLLinear vs. MLP vs. DIMLP, and PCA vs. full-voxel targets.  
   - The comparison between MLLinear and MLP isolates the impact of nonlinearity given identical architectures and PCA; DIMLP vs. MLP isolates cross-modal nonlinear interactions. The finding that DIMLP > Linear but MLP > DIMLP (Section 3.2.1) is quite informative about where the gains actually arise.

3. **Clear evidence that nonlinearity is genuinely useful, not just capacity.**  
   - Figure 16 shows layer-wise \(r^2\) across all LLaMA and Whisper layers, where MLP consistently dominates linear regression for both modalities and across layers, not just at one sweet spot.  
   - Table 5 nicely demonstrates that “more complex” nonlinear models (LSTM, GRU, Transformer, Deep MLP) overfit and underperform the single-layer MLP, which supports the claim that a modest amount of nonlinearity is optimal given data scale, rather than the gains being an artifact of simply using a huge black box.

4. **Interesting spatiotemporal analysis via RED and clustering.**  
   - The RED definition in Section 2.5 and Appendix J.4 is elementary but useful: \(\mathrm{RED}(v,t) = |f_1(v,t) - y(v,t)| - |f_2(v,t) - y(v,t)|\) allows time-resolved comparison between two encoders.  
   - Figure 1 (and expanded Figure 23 in the appendix) shows that clustering ROIs based on RED time series yields more modular, functionally plausible structures than standard functional connectivity: modularity \(Q = 0.155\) for nonlinear RED vs 0.145 linear RED vs 0.068 FC. The dendrogram in Figure 1d groups motor and somatosensory regions by body part and aligns speech-related ROIs with the dorsal pathway, which is a nontrivial neuroscientific insight derived from the encoding models.

5. **Careful ROI-wise and voxel-wise characterization of multimodal integration.**  
   - Figure 2 uses voxelwise \(\Delta\)CC\(_\text{norm}\) maps to show where adding audio or semantics helps, and panel (e) summarizes ROI-level \(\Delta r\) across all subjects with statistical testing (FDR-corrected). This makes a convincing case that multimodality yields widespread benefits well beyond auditory cortex, including motor/somatosensory and high-level visual regions.  
   - The variance partitioning analyses (Figures 34–48, and summarized in Figure 3 in the main text) provide a reasonably nuanced picture: joint audio+semantic representations dominate overall (~68.5% of significantly predicted voxels, Figure 3b), but unique audio contributions are clearest in early AC and M1M, whereas semantic uniqueness is stronger in higher-order areas. This is more informative than simply reporting that multimodal does “better”.

6. **Positioning relative to neurolinguistic theories is thoughtful and not purely hype.**  
   - Section 3.3.2 and the Discussion connect the variance partitioning and ROI-specific patterns back to the Motor Theory of Speech Perception, the dorsal stream model, and convergence–divergence zones. While inevitably somewhat interpretive, the authors are reasonably cautious, often noting alternative explanations (e.g., lexical frequency vs. embodied semantics on Page 8) and not overclaiming causal mechanisms.

7. **Statistical testing and robustness checks are solid.**  
   - Appendix C outlines voxelwise paired comparisons with Bonferroni correction across model pairs; Figures 4 and 5 show that the top multimodal MLP is significantly better than the baseline semantic linear and audio-only models across subjects in both \(r^2\) and CC\(_\text{norm}\).  
   - Multiple splits are used: three test stories for main results, and a single-story protocol to match Antonello et al. (2024), with a clear discussion of why CC\(_\text{norm}\) is more stable than \(r^2\) in the single-story case.

## Weaknesses

1. **Methodological novelty is modest; main contribution is empirical.**  
   The core modeling choices are fairly straightforward: PCA on voxel responses (Section 2.3), concatenation of four lagged feature vectors (Section 2.2), and a single-hidden-layer MLP (Section 2.4) that takes concatenated Whisper+LLaMA features. The RED metric in Section 2.5 is a simple absolute-error difference and the variance partitioning follows de Heer et al. (2017). From a machine learning perspective, there is no architectural or algorithmic innovation beyond well-known practices; the advances lie in (i) applying these choices to this specific dataset and (ii) carefully analyzing the resulting patterns. This is not necessarily disqualifying but does limit the ML-side contribution.

2. **Limited subject pool and potential overinterpretation of fine-grained cortical patterns.**  
   All analyses are based on only three subjects (Section 2.1), which is standard for this dataset but still a serious constraint when making detailed claims about ROI- and pathway-level organization. While subject-wise plots are shown for many figures (e.g., Figures 9–11, 17–22, 25–28, 34–48), the main results in the text often read as if they generalize at the population level. For example, the claim on Page 7 that “Rol-wise analysis (Figure 3 b) shows that semantic, audio, and joint features accounted for 21.4%, 10.1%, and 68.5% of significantly predicted voxels” is aggregated across subjects but without any measure of between-subject variability. Some of the more specific dorsal-stream statements (Section 3.3.2, Page 8) and embodied semantics discussion might be too strong given n=3 and modest absolute \(r\) values.

3. **Interpretation of nonlinear vs. linear contributions is somewhat conflated with PCA and training regime.**  
   - Section 3.1 claims that “performance gains are driven by nonlinearity rather than reduced dimensionality,” based on MLP vs. MLLinear vs. Linear-on-PCA comparisons in Table 1. However, MLLinear is trained with the same MLP optimization pipeline (AdamW, MAE loss, batch size 128, early stopping; Appendix B.5) whereas the Linear baseline is ridge regression fit voxelwise with cross-validated alphas. These are quite different fitting procedures and noise models. It is therefore not fully clear how much of the MLLinear vs. Linear difference (or lack thereof) is due to dimensionality vs. optimization.  
   - Similarly, the full-voxel Linear vs. PCA-Linear differences in Table 1 show that the PCA-Linear model can actually hurt performance relative to full-voxel Linear in the text-only case (3.56% vs 3.66% \(r^2\)), even though PCA is supposed to “prevent overfitting”. This suggests the dimensionality reduction itself is not neutral, and the narrative around “nonlinearity is the key driver” would be stronger if the authors more systematically controlled for these factors (e.g., ridge on PCA with per-component alphas, or MLLinear with ridge-like regularization).

4. **Choice of MAE loss and its relation to evaluation metrics is not well justified.**  
   The encoding models are trained with MAE (Appendix B.5), yet performance is evaluated using Pearson correlation, \(r^2 = |r| \cdot r\), and CC\(_\text{norm}\). There is no discussion of why MAE is preferable to MSE or correlation-based losses in this context, nor any ablation showing that MAE helps. Since RED is defined in terms of absolute errors \(|f(v,t)-y(v,t)|\) (Section 2.5), using MAE might bias models toward optimizing the quantity that later feeds RED, potentially making differences look larger than they would under a loss more aligned to correlation. A short experiment comparing MAE vs MSE vs correlation loss for at least one model would clarify whether the gains are robust to the choice of loss.

5. **Noise ceiling computation and CC\(_\text{norm}\) regularization need more scrutiny.**  
   - Appendix B.2 defines \(CC_{\max} = (\sqrt{1 + NP/(SP \times N)})^{-1}\), referencing Schoppe et al. (2016), but the paper never explicitly describes how NP and SP are estimated from the repeated test story data. Since CC\(_\text{norm}\) is central to the paper and can exceed 1 when CC\(_\text{abs} < CC_\text{max}\), the regularization trick of clamping voxels with \(CC_\text{max}<0.25\) to 0.25 (Section 2.5) introduces a somewhat ad hoc floor. This can inflate CC\(_\text{norm}\) relative to other works using different thresholds or voxel-all inclusion.  
   - Figure 13–15 show CC\(_\text{norm}\) maps but do not explicitly mark which voxels were affected by this clamp. A sensitivity analysis to the 0.25 threshold or a justification for this specific value would strengthen the validity of comparisons, especially vs. Antonello et al. (2024) which may use a slightly different noise-ceiling procedure.

6. **RED-based clustering and modularity improvements are interesting but thinly supported statistically.**  
   - Figure 1 and Figure 23 report modularity \(Q\) values of 0.068 (FC), 0.145 (linear RED), and 0.155 (nonlinear RED). The difference between 0.145 and 0.155 is small, but the text often interprets this as “clearer functional groupings” and as evidence that nonlinear models “achieve better functional clustering” (Pages 5–6 and J.4). There is no permutation test or statistical analysis of whether that difference in \(Q\) is meaningful relative to variability across clusterings or subjects.  
   - Additionally, the clustering is performed at the ROI level with a fixed ordering and a single hierarchical clustering method. Results could be sensitive to linkage choice and ROI parcellation. Without any robustness checks (e.g., bootstrapped modularity distributions, alternative ROI sets), the claim that nonlinear encoders yield superior functional clustering should be toned down or supported more rigorously.

7. **Multimodal fusion space is somewhat underexplored.**  
   - The principal multimodal fusion strategy is simple concatenation of final-layer LLaMA and Whisper embeddings, followed by PCA on responses; the comparison to stacked regression in Table 4 shows this already helps. However, the paper only briefly explores more sophisticated fusion schemes: DIMLP (nonlinear per modality, linear fusion) and standard MLP (fully entangled). Other plausible baselines, like FiLM conditioning, gated fusion, or attention-based weighting between modalities are not considered.  
   - Figure 6 illustrates that combining “best” unimodal layers yields the best multimodal MLP, but layer mixing is only done via concatenation and a single MLP, not via architectures that explicitly model temporal or cross-layer interactions. Given the strong emphasis on “nonlinear and multimodal integration” as the main contribution, a slightly broader exploration of fusion strategies would help demonstrate that the reported gains are not overly sensitive to this very specific design choice.

8. **Over-interpretation risk in some neurolinguistic claims.**  
   While the paper often acknowledges alternative explanations, some passages read as stronger than warranted by the data. For example, Page 8 states that M1M shows “strong contribution from auditory features, exceeding even AC, consistent with its role in executing speech articulation,” and that “These findings align with the Motor Theory of Speech Perception,” based on variance partitioning diagrams (Figure 3, Figures 43–48) and \(\Delta r\) in Figure 2e. Given that M1M and AC are quite different in size and signal-to-noise, and that the unique audio variance fractions in Venn diagrams come from model partitions rather than direct experimental manipulations, these statements should be more carefully caveated as correlational evidence consistent with, but not proving, specific theories.

9. **Some clarity and editorial issues.**  
   - Parts of the references and Appendix B.2 contain clear typos or garbled author lists (e.g., “A. A. B. B. A. A. A…” on Page 10, “nocMALIZED” in Appendix B.2), which detracts from polish.  
   - The main text never includes a dedicated “Related Work” section, instead spreading citations between Introduction and Discussion. This contributes to missing several directly relevant works (see next section) and makes it harder for readers to place the paper in the broader space of nonlinear brain encoders.

## Potentially Missing Related Work

1. **Güçlü & van Gerven, “Modeling the dynamics of human brain activity with recurrent neural networks” (2016)**  
   This paper was an early demonstration of using RNNs to model nonlinear dynamics in brain responses (albeit mostly in vision / movie stimuli), which is conceptually very close to this work’s emphasis on moving “beyond linear models” in encoding. It should be discussed in the context of Section 1 (nonlinear encoding) and Appendix E (where recurrent models are experimented with), both to acknowledge prior nonlinear encoding attempts and to clarify what is specific about the present contribution to continuous speech.

2. **Accou et al., “Predicting speech intelligibility from EEG in a non-linear classification paradigm” (2021)**  
   Although using EEG rather than fMRI, this work similarly exploits deep neural networks to relate continuous speech features to brain signals, emphasizing benefits of nonlinearity. It is directly relevant to the narrative in the Introduction (Pages 1–2) that nonlinear models are underused in speech neuroscience, and could be cited as evidence that the shift to nonlinearity is not entirely absent in the field, especially outside fMRI.

3. **Puffay et al., “Relating EEG to continuous speech using deep neural networks: a review” (2023)**  
   This review surveys nonlinear, deep-learning-based approaches for continuous speech–EEG modeling. Including it in the Introduction’s related-work discussion would help the authors better situate their contribution in the broader multimodal brain–speech modeling literature and contrast fMRI vs EEG challenges, strengthening the motivation in Appendix N.1.

4. **d’Ascoli et al., “TRIBE: TRImodal Brain Encoder for whole-brain fMRI response prediction” (2025)**  
   TRIBE introduces a deep neural network encoding model for whole-brain fMRI with multiple stimulus modalities. This is very close in spirit to the present work’s multimodal MLP, but apparently not cited. It should be discussed in Section 1 (when highlighting gaps in multimodal nonlinear encoding), and compared in Section 4 to clarify differences in data modality (visual vs auditory language), architecture complexity (deep vs shallow), and empirical gains. If feasible, an architectural or conceptual comparison in the discussion around Table 1 / Table 4 would be important.

5. **Abdollahi et al., “Probing Multimodal Fusion in the Brain: The Dominance of Audiovisual Streams in Naturalistic Encoding” (2025)**  
   This work explicitly examines multimodal fusion (audio + visual) in brain encoding with modern feature extractors, which parallels this paper’s audio + language fusion. It should be cited in Section 3.3.1–3.3.2 when discussing convergence-divergence zones and multimodal integration across the cortex, and potentially in Appendix G where scaling of feature extractors is discussed.

6. **Tang, “A Review of Multimodal Brain Language Decoding” (2025)**  
   While focused on decoding rather than encoding, this review synthesizes recent progress in multimodal neural language models applied to brain signals, which is conceptually adjacent. It would help frame this paper as part of a larger wave of multimodal brain–language modeling and could be briefly referenced in the Introduction or Discussion when talking about implications for decoding and “brain-aligned AI”.

## Questions

1. **Noise ceiling and CC\(_\text{norm}\) details.**  
   - How exactly are SP and NP computed in Equation in Appendix B.2 (e.g., from split-half correlations, across repeats)?  
   - How sensitive are your key conclusions (e.g., 14.4% CC\(_\text{norm}\) gain in Table 4) to the choice of CC\(_\text{max}\) floor at 0.25? Could you provide a supplementary analysis where you (i) vary the floor (0.15, 0.35) or (ii) restrict analyses to voxels with CC\(_\text{max} \ge 0.5\) without flooring?

2. **Role of the MAE loss.**  
   Have you tried training the same MLP models with MSE loss or a correlation-based loss? Do the relative improvements over linear baselines and over DIMLP remain similar? A simple comparison on one model (e.g., multimodal MLP with PCA) would greatly increase my confidence that conclusions are not tied to MAE.

3. **Disentangling nonlinearity from optimization and regularization.**  
   The Linear baseline is ridge regression with voxelwise hyperparameters, whereas the MLLinear model uses AdamW and a shared hidden representation. Could you add an experiment where you:  
   - Fit a ridge regression on PCA-reduced responses (same 512 PCs), or  
   - Fit a voxelwise linear model trained via gradient descent with the same MAE loss,  
   so that you can more cleanly separate the effect of nonlinearity from the effect of using a shared low-rank representation and a different optimizer?

4. **Robustness of RED-based modularity.**  
   For the RED clustering, can you quantify variability in modularity \(Q\) across subjects and across clustering hyperparameters? For instance, what are the Q values per subject, and do the nonlinear models outperform linear ones consistently? A simple permutation test or bootstrap (reshuffling ROI label or time segments) would help assess whether the 0.155 vs 0.145 difference is meaningful.

5. **Layer fusion and model scaling.**  
   In Figure 6, you show that mixing “best layers” for each modality yields the best multimodal MLP. How robust is this to adding a small amount of cross-layer attention or FiLM-like conditioning between semantic and audio streams? If you have any preliminary experiments (even if they do not improve over the baseline), it would be useful to know whether more expressive multimodal fusion is strictly unnecessary or just not yet tuned.

6. **Between-subject variability in variance partitioning.**  
   For the Venn diagrams and ROI-level dominance in Figure 3b, could you provide some summary statistics of variability across subjects (e.g., mean ± SEM of percentage of voxels dominant in each partition per ROI)? This would help assess how much of the pattern (e.g., 83.3% joint in AC) is stable vs idiosyncratic.

## Flag For Ethics Review

- No ethics review needed.  

## Details Of Ethics Concerns

N/A. The work uses an existing open fMRI dataset with explicit licensing (Appendix O), no new human data collection, and there are no obvious privacy, safety, or dual-use red flags beyond standard brain-imaging concerns already addressed by the original dataset authors.

## Soundness Rating

3: good.  
The experimental methodology is thorough, with strong baselines, extensive ablations (Tables 1–5), and robust statistical testing (Figures 4–5). Some analytical components (noise ceiling handling, attribution of gains purely to nonlinearity, modularity statistics) could be more rigorously justified, but there are no fatal technical flaws.

## Presentation Rating

3: good.  
The paper is generally clear and well organized, with detailed figures (e.g., Figures 1–3, 16, 24–32, 34–48) and tables. However, some referencing/typo issues and the lack of a dedicated related work section slightly detract from clarity.

## Contribution Rating

3: good.  
Conceptual and neuroscientific contributions are meaningful: strong evidence for the value of nonlinear multimodal encoders in speech-fMRI, well-supported analyses of cortical integration, and an insightful RED-based spatiotemporal clustering. Methodological novelty on the ML side is modest, but the empirical and analytical contributions are valuable for the ICLR community interested in brain-aligned language models.

## Overall Rating

6: Marginally above the acceptance threshold. But would not mind if paper is rejected.  

The paper presents a carefully executed and well-analyzed study showing substantial gains from simple nonlinear multimodal encoders in naturalistic speech fMRI, with convincing quantitative results and reasonably cautious interpretation. The main limitations are modest methodological novelty, some underdeveloped aspects of the nonlinearity vs dimensionality argument, and potential overinterpretation in a few neuroscientific claims. On balance, I lean to a positive recommendation given the empirical strength and relevance to brain–AI alignment, but I would not be surprised if others weigh the modest novelty more heavily.

## Reviewer Confidence

4: confident.  
I am familiar with encoding models and multimodal brain–language work, and I carefully checked the modeling, metrics, and statistical analyses. Some neuroscience-theory nuances and the specifics of the noise-ceiling estimation are less in my core expertise, hence not a 5.