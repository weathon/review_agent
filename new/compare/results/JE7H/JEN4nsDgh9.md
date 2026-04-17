---
job_id: b5a5eb98-ef6c-4bef-aea5-dc0b7e0aa7eb
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: JEN4nsDgh9.pdf
paper: Do I Look Like a “cat.n.01” to You? A Taxonomy Image Generation Benchmark
main_score_norm: 0.4
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper proposes a new benchmark and metrics for evaluating text-to-image generative models on taxonomy concepts, clearly within ICLR’s scope of generative models, representation learning, and datasets/benchmarks.

## Minimum Quality
Pass ✅.  
The submission has all required components: Abstract, Introduction, Dataset/Methodology (Sections 2–4), Experiments and Results (Section 5 and appended tables/figures), Related Work (Section 6), Conclusion (Section 7), and limitations/ethics/reproducibility appendices. The work is technically non-trivial and includes both quantitative and qualitative analysis.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I did not find any attempts to manipulate automated reviewing systems or hidden prompts in the provided content.

---

# Expected Review Outcome:

## Summary

The paper introduces a benchmark for “Taxonomy Image Generation,” where text-to-image (T2I) models are evaluated on their ability to depict WordNet synsets, including both “easy” common-sense concepts and randomly sampled nodes, as well as concepts predicted by a taxonomy-enrichment LLM (TaxoLLaMA-3.1).  

Twelve T2I or retrieval models are evaluated with nine metrics: pairwise ELO-style preference scores using humans and GPT‑4, a reward model, several taxonomy-aware CLIP-based similarity metrics (lemma, hypernym, co-hyponym similarity, and a “specificity” ratio), plus IS and FID.  

The main findings are that model rankings on this task differ from standard T2I leaderboards, Playground and FLUX tend to be preferred in pairwise judgments, SDXL‑turbo often wins on CLIP-based alignment, and simple retrieval from Wikimedia Commons performs poorly for many synsets. The authors also release the images for all WordNet 3.0 synsets produced by the best-performing model.

---

## Strengths

1. **Clearly defined and timely problem**  
   The paper identifies a concrete and underexplored task: generating canonical, taxonomy-linked depictions of lexical concepts rather than prompting with human-authored rich descriptions. This is a natural and practically important extension of WordNet / ImageNet-style resources and fits current interest in automated dataset curation.

2. **Non-trivial benchmark construction across multiple data regimes**  
   The datasets span (i) “Easy Concepts” expanded from Nikishina et al. (2023) to 483 synsets (Section 2.1), (ii) a 1,202-node random WordNet sample with structured control over hypernym/hyponym/mix relations (Section 2.2), and (iii) 1,685 LLM-predicted concepts (Section 2.3). This separation between ground-truth nodes and LLM-generated ones is well motivated and allows studying robustness to noisy or novel concepts.

3. **Rich evaluation protocol combining human, LLM, and automatic metrics**  
   The benchmark integrates human pairwise ELO scores and GPT‑4 pairwise judgments (Section 4.1), plus a separate reward model (ImageReward) and CLIP-based similarity metrics grounded conceptually in KL divergence and mutual information (Section 4.2, Appendix D). This is more comprehensive than many T2I papers that rely only on CLIPScore/IS/FID. In particular, **Figure 4** and **Table 3** effectively show that human and GPT‑4 rankings are strongly correlated but not identical, which is valuable empirical evidence about the reliability and biases of LLM-as-a-judge in the T2I setting.

4. **Taxonomy-aware similarity metrics with some theoretical justification**  
   The paper does not just compute vanilla CLIPScore, it defines lemma similarity, hypernym similarity, and cohyponym similarity in Equations (1)–(3), and a “Specificity” measure defined as a ratio of lemma to cohyponym likelihoods. The subsequent Theorems 1–4 in Appendix D connect these to Bayes posteriors, KL divergence, and mutual information, providing a principled, if idealized, grounding rather than ad-hoc metrics.

5. **Concrete, interpretable comparative results**  
   The model comparison is quite extensive: twelve systems (Table 1) across nine metrics and multiple subsets. **Table 2** (despite some formatting issues, see below) and **Tables 3–8** provide a useful summary: human/GPT‑4 preferences consistently favor Playground and FLUX; SDXL‑turbo often wins the CLIP-based metrics; SD1.5 and SD3 score well on IS or FID. **Figure 5** and **Figure 12** give a nice analysis of GPT‑4’s positional bias and lower tie rate compared to humans.

6. **Qualitative analysis that is actually informative**  
   The paper goes beyond cherry-picked best samples and highlights real failure modes. **Figure 13–17** show how models deal with abstract concepts (often through text overlays or vague art), degenerate to playing-card motifs or ornamental circles, or hallucinate “monsters” for rare animal names. **Figure 22** shows systematic confusions among domestic_cat hyponyms (e.g., “mouser” mixing electronics and rodents, “tomcat” conflating aircraft and felines), which strongly supports the claim that taxonomic fine-grained understanding is challenging.

7. **Public release of generated WordNet imagery**  
   Publishing the full set of WordNet 3.0 images (Section 1, contributions; Reproducibility Considerations) is genuinely valuable for the community, both for downstream experiments and for probing representation gaps in current T2I models.

---

## Weaknesses

1. **Heavy reliance on CLIP similarities as “probabilities,” with questionable probabilistic formalization**  
   The core metrics in Section 4.2 and Appendix D treat CLIP cosine similarities as estimates of probabilities \(P(X=x\mid v)\), directly assigning  
   \[
   S_{\text{lemma}}(v,x) := P(X=x\mid v) \approx \text{sim}(C(v), C(x^j))
   \]
   and similarly averaging over hypernyms / cohyponyms. Cosine similarity in \([-1,1]\) is not a valid probability measure; it is not normalized over \(\mathcal{X}\), and can be negative. The theoretical results (Theorems 1–4) are proven in terms of true probabilities, but the actual implementation in the benchmark violates their assumptions. There is no explicit transformation such as \(P(X=x\mid v)\propto \exp(\alpha\,\text{sim}(\cdot))\) with a proper normalization over images. This mismatch makes the KL and mutual-information interpretations largely rhetorical rather than empirically grounded. At minimum, the paper should formalize the mapping from CLIP embeddings to probability distributions and empirically validate that it is calibrated enough for the theorems to be meaningful.

2. **Some central tables appear corrupted/unclear, undermining interpretability**  
   **Table 2**, labeled as “Summary of the Top-1 model for each metric and subset,” is nearly unreadable: model names are garbled (“Pluground,” “Ptolet,” “StopFixed,” “Kamlooks,” “Dram,” “ELO,” “SDT”), and the structure of rows and columns is not explained well. This is supposed to be the main high-level summary of which model is best under which metric, yet as written it is impossible to reconstruct accurately. While more detailed tables are provided later (Tables 3–16), the key takeaway table for readers is effectively broken, which is a serious presentation issue.

3. **Limited human evaluation scale and sparse description of annotation protocol**  
   Human ELO is based on 3,370 pairwise comparisons from four assessors (Section 4.1). Given 12 models and multiple subsets, this works out to only ≈600 comparisons per model, with uniform random pairing. The paper reports inter-annotator Spearman correlation ≈0.8 but does not detail per-synset coverage or how many unique synsets were actually judged. Without ensuring that each model is compared against each other model across a representative subset of synsets, the Bradley–Terry estimates can be noisy and potentially unstable. Moreover, the annotation instructions for humans are barely described, in contrast to the GPT‑4 prompt shown in **Figure 3** / **Figure 7**. How annotators were trained and compensated, how ties/both-bad were handled in practice, and time-per-judgment are all missing. This weakens confidence in the human-preference-based conclusions.

4. **GPT‑4 evaluation issues and positional bias are acknowledged but not adequately mitigated**  
   Section 5 notes that GPT‑4 preferences exhibit “a strong bias toward the first option” (visualized in **Figure 5** and **Figure 12**). However, the authors do not implement standard countermeasures such as randomizing order across separate runs, majority vote over multiple GPT calls, or masking model identities. They argue that the Bradley–Terry model “compensates” as long as labels are sufficient, but this is only partially true: a systematic preference for position A that is not symmetric across models could distort ELO estimation. Since GPT‑4 judgments are used heavily (Tables 3–4, **Figure 4**, **Figure 8**), a more careful experimental design is needed or, at minimum, a quantitative analysis of how much the positional bias affects rankings.

5. **Ambiguity around objective of “taxonomic” visualization, especially for abstract or relational concepts**  
   The introduction claims that images “should aim succinctly portraying the synset’s core idea and/or sometimes revealing insights about the concept that are challenging to convey textually,” but the metrics and evaluation never articulate what is considered a “good” depiction for an abstract concept like “feeling.n.01” or “emotion.n.01”. **Figures 19–21** show interesting examples, yet there is no grounded annotation on whether these are judged acceptable by experts. The CLIP-based metrics will naturally reward generic iconography or stereotypical imagery that matches training data, which could be appropriate or misleading depending on the concept. The benchmark risks conflating “whatever CLIP has learned about this word” with “correct taxonomic depiction,” without a clear target.

6. **Inadequate exploration of prompt design and the role of definitions**  
   The main prompt template is fixed (“An image of \<CONCEPT> (\<DEFINITION>)”; Section 3), and analysis of definitions is confined mostly to pairwise preference outcomes (**Figure 6**, minor discussion in Appendix C). Yet the title and abstract emphasize that WordNet definitions are non-standard inputs for T2I and might be misguiding (Figure 1). There is no systematic study of alternative prompting strategies (e.g., paraphrased definitions, synonyms, minimal prompts, style control tokens) nor any ablation inside individual models (e.g., CLIP vs non-CLIP conditioning). As a result, we cannot tell whether performance differences stem from intrinsic model capabilities or from idiosyncratic interactions with this very specific prompt format.

7. **Theoretical claims are somewhat overstated, given strong assumptions**  
   Theorem 1 relies on a uniform prior \(P(V=v)=1/|V|\), which is almost certainly unrealistic in the WordNet setting where some concepts are vastly more frequent in training data. Theorems 2–4 assume “large enough” lemma similarity or fixed \(P(X|i)\), yet the paper does not quantify when these conditions hold. In particular, the claim at the end of Appendix D.2 that maximizing hypernym or cohyponym similarity “therefore better reflects neighbors semantic properties and covering tree structure” glosses over the fact that an image that is very generic relative to the hypernym could also have high cohyponym similarity and low specificity. Since the implementations do not enforce any of these assumptions, the theoretical section over-promises relative to what is actually guaranteed by the metrics as used.

8. **Benchmark narrowly grounded in English WordNet, with limited discussion of generalization**  
   The limitations section briefly acknowledges the WordNet focus, but the main text mostly treats WordNet as *the* taxonomy. There is no attempt to run even a small subset on another structured resource (e.g., Wikidata classes, ConceptNet concepts, ontology from biology), or to discuss how the proposed similarity metrics would adapt when the taxonomy is not strictly IS-A tree-like. This reduces the broader impact of the benchmark as a general “taxonomy image generation” framework.

9. **Retrieval baseline is under-specified and arguably too weak**  
   Section B.3 describes retrieving the top-1 image from Wikimedia Commons via “the main image search engine,” but there are no details on the query formulation (synset vs lemma vs gloss), whether disambiguation is used, or how NSFW and licensing are handled. **Figure 2** shows a single cherry-picked failure for “cigar lighter,” but quantitative comparisons in Table 7 and Table 9 indicate that retrieval is often competitive or better on IS, and only somewhat weaker on CLIP-based metrics. Without stronger baselines such as fine-tuned retrieval models (e.g., CLIP-based nearest-neighbor using external corpora) or multiple retrieved candidates, the conclusion that “generation significantly outperforms retrieval” is only weakly supported.

10. **Clarity and editing issues throughout**  
    Beyond the broken **Table 2**, there are numerous typos and inconsistent model names: “Pluground,” “SDT,” “Ptolet,” “Kamlooks,” “Hype/HypeR,” “HDiT/HDit,” “DeepFrost” etc., which appear in the tables and captions (e.g., Table 2, Tables 3–6, Figure captions in Appendix I). This makes it surprisingly hard to map back to the model list in **Table 1**. Some sentences are grammatically off (“text-to-image models outperform traditional retrieval-based methods in covering a broader range of concepts, highlighting their ability to better represent and visualize these previously underexplored areas” is vague; Section 5), and several passages have duplicated citations. Overall readability is fair but clearly below ICLR standards for a benchmark paper.

---

## Potentially Missing Related Work

1. **Hu et al., “DSH-Bench: A Difficulty- and Scenario-Aware Benchmark with Hierarchical Subject Taxonomy for Subject-Driven Text-to-Image Generation,” 2026**  
   This work introduces a T2I benchmark that also relies on a hierarchical taxonomy of subjects to probe model performance across difficulty levels. It is directly relevant both conceptually and methodologically to the present paper’s claim of being a new taxonomy-based T2I benchmark. It should be cited in Section 6 (Evaluation Benchmarks) and discussed as a complementary approach that uses a different taxonomy (subject taxonomy vs WordNet synsets) and a different task framing (subject-driven vs lexically-driven). A brief comparison in Section 5 would clarify what is unique about WordNet-based evaluation.

2. **Kang et al., “StudioGAN: A Taxonomy and Benchmark of GANs for Image Synthesis,” 2023**  
   While focused on GANs rather than diffusion models, this paper offers a thorough taxonomy and benchmark methodology for image synthesis models. It could inform the design of evaluation protocols and the discussion of model categories in Section 3 (Models) and Section 6. Bringing it in would strengthen the positioning of this work within the broader landscape of synthesis benchmarks and model taxonomies.

---

## Questions

1. **CLIP-to-probability mapping:**  
   How exactly are CLIP similarities converted into the \(P(X=x\mid v)\) quantities used in Equations (1)–(3) and Theorems 1–4? Are similarities re-scaled to \([0,1]\), exponentiated, or normalized over all images for a given concept? A precise definition (and possibly an ablation showing that the theoretical relations still roughly hold under the chosen mapping) would significantly increase my confidence.

2. **Human preference protocol details and coverage:**  
   Can you provide more detail on the human annotation process: how were annotators trained, how many unique synsets and model pairs each annotator saw, and whether you enforced any balanced pairing design (e.g., each model compared to each other model a minimum number of times)? Some per-synset or per-model statistics would help assess the robustness of the Bradley–Terry estimates in **Figure 4** and **Tables 5–6**.

3. **Mitigation of GPT‑4 positional bias:**  
   Have you tried re-running a small subset of comparisons with swapped positions (image A vs B) or with anonymized model names to quantify how much the “first-option bias” in **Figure 5** affects ELO rankings? Even a limited experiment on, say, 300–500 pairs could substantiate the claim that BT modeling sufficiently compensates for this bias.

4. **Prompt sensitivity analysis:**  
   Did you test any alternative prompt styles beyond the “An image of \<CONCEPT> (\<DEFINITION>)” template, such as using only the lemma, only the gloss, or paraphrases generated by an LLM? If so, how stable are the relative rankings in **Table 3/4** and **Figure 4/8**? If not, could you comment on how future work might incorporate prompt-robustness into the benchmark?

5. **Generalization beyond WordNet:**  
   Do you see any technical obstacles to applying your lemma/hypernym/cohyponym metrics on non-tree or multi-relational taxonomies like Wikidata or domain ontologies (where IS-A is not necessarily the only or dominant relation)? For instance, how would you handle multiple parents or non-hierarchical relations when defining \(A(v)\) and \(N(v)\)?

6. **Retrieval baseline design:**  
   Can you clarify how exactly the Wikimedia Commons queries were built (lemma vs synset label vs concatenation with gloss), and whether you tried any CLIP-based retrieval (e.g., nearest neighbor search in a large image corpus) as a stronger baseline? If not, do you agree that the conclusion about “generation > retrieval” should be framed more cautiously?

Author responses that clearly address (1) and (2), and either empirically mitigate or convincingly analyze (3), would most strongly improve my assessment.

---

## Flag For Ethics Review

No ethics review needed.  

---

## Details Of Ethics Concerns

N/A.

---

## Soundness Rating

2: fair.  
The overall methodology is reasonable and the empirical findings are plausible, but the probabilistic justification of the CLIP-based metrics is shaky, the GPT‑4 evaluation protocol has unmitigated biases, and the human evaluation is somewhat small and under-specified for strong conclusions.

---

## Presentation Rating

2: fair.  
The paper is generally understandable and contains many helpful figures (e.g., Figures 2, 4, 13–22), but key tables like Table 2 are corrupted, model names are inconsistent across sections, and numerous typos and phrasing issues reduce clarity.

---

## Contribution Rating

3: good.  
Despite its flaws, the paper offers a useful benchmark task, a large-scale evaluation over many T2I models, novel taxonomy-aware metrics, and a released corpus of WordNet images. These are likely to stimulate further work at the intersection of taxonomies and T2I.

---

## Overall Rating

4: Marginally below the acceptance threshold. But would not mind if paper is accepted.  

The idea of a taxonomy-focused T2I benchmark and the breadth of evaluation are valuable, and the qualitative analyses are insightful. However, the inconsistent probabilistic treatment of CLIP-based metrics, design weaknesses in the GPT‑4/judging setup, limited scale and description of human evaluation, and significant presentation issues (especially the broken main results table) collectively make the work feel not quite ready for ICLR’s main track in its current form. With clearer metric definitions, stronger evaluation protocol, and careful editing, a revised version could be quite impactful.

---

## Reviewer Confidence

4: confident.  
I am familiar with T2I evaluation, CLIP-based metrics, and WordNet-style taxonomies, and I have carefully checked the main equations and empirical methodology, though I did not re-derive every theorem in the appendix.