## Summary

MedResearcher-R1 proposes a medical deep research agent that addresses the "sparse medical knowledge problem" through two main innovations: (1) a Knowledge-Informed Trajectory Synthesis (KISA) framework that generates multi-hop reasoning trajectories around rare medical entities, and (2) specialized medical retrieval tools (PrivateMedicalRetriever, ClinicalReasoningEngine) integrated with general-purpose tools via dynamic routing. The system is trained with supervised fine-tuning followed by reinforcement learning (GRPO) on 2,100+ synthesized trajectories.

## Strengths

- **Clear problem formulation and motivation:** The paper convincingly identifies the gap between general-purpose deep research agents and specialized medical reasoning, particularly around rare disease connections and authoritative source retrieval (Section 1). The "sparse medical knowledge problem" framing is articulate and the benchmark gap (o3-deepresearch scoring 25.5/50 on MedBrowseComp) establishes a concrete baseline deficiency.

- **Thoughtful data synthesis methodology:** The KISA framework's approach of mining rare entities (frequency < 10⁻⁶) from PubMed and constructing knowledge graphs for multi-hop trajectory generation is methodologically interesting. The longest-path extraction from subgraphs to generate maximally complex queries addresses a genuine need for challenging training data in specialized domains (Section 3.1).

- **Comprehensive ablation study:** Table 3 systematically isolates component contributions, with statistical significance testing via paired bootstrap (p < 0.05). The finding that removing rare entities drops MedBrowseComp from 27.5 to 20.1 provides meaningful evidence for the core claim that rare-entity training drives performance.

## Weaknesses

- **Confounded primary comparison:** The central claim that MedResearcher-R1 outperforms o3-deepresearch (27.5 vs. 25.5) on MedBrowseComp is confounded by unequal tool access. MedResearcher-R1 queries proprietary databases (FDA, clinical trial registries, PubMed via specialized retrieval) while o3-deepresearch uses general web tools. The paper acknowledges this tool advantage but provides no ablation isolating training methodology from tool access. Without a comparison where baselines receive equivalent medical retrieval capabilities, the attribution of gains to KISA/MTG training versus privileged database access is unverifiable.

- **Insufficient statistical evidence for primary claim:** The 2-point improvement (27.5 vs. 25.5) on a 50-question benchmark corresponds to exactly 1 additional correct answer if scoring is integer-valued, or a narrow margin if fractional. No confidence intervals or significance tests are reported for Table 1, despite the ablation study including such tests. A benchmark of 50 questions has substantial variance; this undermines confidence in the "state-of-the-art" claim.

- **Unsupported quantitative claims:** The introduction claims MTG provides "14% improvement on 5+ hop questions" (Contribution 2), but this number appears nowhere in the paper or appendix. Similarly, Section 3.1.1 states that the augmented relation format "improves multi-hop reasoning accuracy by 12.3% compared to standard triplets" without any supporting ablation. These are substantive claims requiring empirical evidence.

- **Suspicious ablation result unexplained:** Removing rare entity supervision (Table 3, "w/o Rare Entities") causes GAIA performance to collapse from 53.4% to 27.8%—a 25.6-point drop on a general benchmark. This magnitude is implausible if rare medical entity training simply adds specialized capability. The paper offers no explanation for why medical rare-entity training would be essential for general agent performance, leaving this result uninterpretable and potentially indicative of data contamination or uncontrolled experimental variables.

- **Inconsistent numbers for same experimental condition:** Table 3 contains two "SFT only" rows with different numbers: "w/o RL Training (SFT only)" shows 25.5/50.2/51.0 (MedBrowseComp/GAIA/XBench), while "SFT Only" in Training Ablations shows 25.5/49.0/48.0. These should be identical conditions, raising questions about experimental consistency.

- **Privileged tool undermines reproducibility:** The Reproducibility Statement promises to open-source "all artifacts," but the PrivateMedicalRetriever accesses proprietary database connectors (FDA databases, clinical trial registries). Third parties cannot reproduce these queries without equivalent database access. This is acknowledged nowhere in the limitations.

- **Hyperparameter error undermines confidence:** Appendix D.1 specifies learning rate λ = 0.01 for SFT on a 32B model. Standard fine-tuning uses rates 3 orders of magnitude smaller (1–5 × 10⁻⁵); 0.01 would cause weight divergence. This is likely a typo, but such an error in a core training parameter raises concerns about the accuracy of reported configurations.

- **Incorrect citation:** GAIA is cited as "Shinn et al., 2023" but the original paper is by Mialon et al. (2023). Shinn et al. authored Reflexion. The reference list contains the wrong author attribution for this benchmark.

- **Quality control introduces potential data leakage:** Section 3.1.3 describes regenerating questions that o3 or GPT-4 solve with >50% accuracy. If this filtering uses models that will later serve as baselines, or if the difficulty calibration shares distributional characteristics with MedBrowseComp, the training data may be inadvertently optimized for the test benchmark style.

## Nice-to-Haves

- **Human expert validation of training data:** The 2,100 trajectories are LLM-synthesized; no mention is made of medical professional review. For a clinical domain, human verification would strengthen confidence in data quality.

- **Ablation with public medical retrieval:** Comparing PrivateMedicalRetriever against standard PubMed Entrez API would isolate whether performance gains come from proprietary access or training methodology.

- **Medical specialist baselines:** Comparison against Med-PaLM or similar domain-adapted models (with equivalent tools) would clarify whether the contribution is the architecture or simply domain specialization.

- **Error analysis by failure mode:** Categorizing MedBrowseComp failures into retrieval errors vs. reasoning errors would identify bottlenecks and strengthen the methodology section.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Demand for larger benchmark:** The critic requests expanding MedBrowseComp to 500+ questions. While this would strengthen significance testing, 50 questions is the benchmark's design, and this is a limitation of the benchmark rather than a paper flaw that must be addressed. The reviewer should critique what exists, not demand the authors redesign the benchmark.

- **Demand for clinical safety discussion:** The critic requests extensive discussion of potential patient harm. While important for deployment, this is a research contribution paper, not a clinical deployment paper. The "expert-level" title claim is worth criticizing, but demanding a full risk analysis goes beyond scope.

- **Data contamination audit as critical flaw:** The Spark Finder suggests n-gram overlap checks with MedBrowseComp. This is reasonable but is presented as a mandatory addition. Without evidence of contamination, this is a nice-to-have verification, not a blocking issue.

- **Clinician-in-the-loop evaluation:** This would strengthen the paper but is not standard for methodology papers in this venue. The automated evaluation follows benchmark conventions.

## Novel Insights

The paper's most interesting contribution is the **rare-entity-first data generation philosophy**: rather than generating random complex queries, the methodology systematically identifies knowledge gaps (entities with frequency < 10⁻⁶) and builds reasoning chains around them. This contrasts with typical agent training data synthesis that focuses on task diversity rather than knowledge graph density. The ablation showing that rare-entity removal causes catastrophic performance collapse—even on general benchmarks—suggests that dense, specialized knowledge graphs may transfer to general reasoning more effectively than previously assumed, though the magnitude of this effect demands further investigation given the anomalous numbers.

## Suggestions

1. **Add significance testing for Table 1:** Report confidence intervals or p-values for the MedBrowseComp comparison. Given the small N, even a bootstrap analysis would strengthen the SOTA claim.

2. **Provide a fair comparison ablation:** At minimum, report performance when MedResearcher-R1 uses only general web search (without PrivateMedicalRetriever) to isolate training methodology contributions from tool access.

3. **Explain or correct the GAIA collapse:** The 25-point drop when removing rare entities is unexplained and suspicious. Either provide a mechanistic explanation or acknowledge potential experimental issues.

4. **Correct inconsistencies:** Fix the two SFT-only rows in Table 3, the learning rate typo, and the GAIA citation before camera-ready.

5. **Clarify tool reproducibility:** Either provide public API access to PrivateMedicalRetriever or explicitly state in limitations that full reproducibility requires proprietary database access.

6. **Remove or substantiate unverified claims:** Either provide evidence for the "14% improvement on 5+ hop questions" and "12.3% improvement" claims, or remove them from the paper.