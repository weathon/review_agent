# Robustness in Text-Attributed Graph Learning: Insights, Trade-offs, and New Defenses

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 2, 4, 6

## Abstract
While Graph Neural Networks (GNNs) and Large Language Models (LLMs) are powerful approaches for learning on Text-Attributed Graphs (TAGs), a comprehensive understanding of their robustness remains elusive. 
Current evaluations are fragmented, failing to systematically investigate the distinct effects of textual and structural perturbations across diverse models and attack scenarios.
To address these limitations, we introduce a unified and comprehensive framework to evaluate robustness in TAG learning. 
Our framework evaluates classical GNNs, robust GNNs (RGNNs), and GraphLLMs across ten datasets from four domains, under diverse text-based, structure-based, and hybrid perturbations in both poisoning and evasion scenarios. 
Our extensive analysis reveals multiple findings, among which three are particularly noteworthy: 1) models have inherent robustness trade-offs between text and structure, 2) the performance of GNNs and RGNNs depends heavily on the text encoder and attack type, and 3) GraphLLMs are particularly vulnerable to training data corruption.
To overcome the identified trade-offs, we introduce SFT-auto, a novel framework that delivers superior and balanced robustness against both textual and structural attacks within a single model. 
Our work establishes a foundation for future research on TAG security and offers practical solutions for robust TAG learning in adversarial environments.
Our code is available at: https://github.com/Leirunlin/TGRB.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper provides a comprehensive evaluation framework to assess the adversarial robustness of various models on Text-Attributed Graphs. Their large-scale analysis reveals the text-structure robustness trade-off in RGNNs and GraphLLMs, showing that models are typically strong against either structural or textual attacks, but not both. To overcome the trade-off and achieve a balanced robustness, the authors propose SFT-auto, a 2-staged framework that first detect the structural and textual attacks, then proceeds to an adaptive recovery.

### Strengths
* Authors provide a comprehensive, large-scale empirical evaluation across a wide range of models, datasets, and attack settings.
* The writing and the figures are well-structured and easy to follow.
* Various experiments provided in appendix.

### Weaknesses
* Some of the statements are wrongly addressed: 

     * In Section 3.1, it says that the spectral methods demonstrate superior performance against poisoning attacks while referencing [1]. However, [1] does not include any experiments or analysis on poisoning setting. 
     * In Section 3.2, GNNGuard and RUNG are marked as spectral methods, while both are spatial methods. 

* More recent work such as [2,3] could be included for the "improving structure" category for a more reliable empirical evaluation Also, while the authors reference [1] within the paper, it is not included as a baseline. 


* My primary concern is the novelty of SFT-auto. While the authors state that it overcomes the trade-off, it seems more like an incremental mergence of two simple defense methods against structure and text attacks. Moreover, while the authors provide a comprehensive evaluation framework for both poisoning and evasion setting, their suggested method to overcome the trade-off, SFT-auto, is only applicable to the evasion setting.




[1] Adversarial Training for Graph Neural Networks: Pitfalls, Solutions, and New Directions \
[2] Boosting the Adversarial Robustness of Graph Neural Networks: An OOD Perspective \
[3] Self-supervised Adversarial Purification for Graph Neural Networks

### Questions
* Could the authors clarify on how SFT-auto leverages the LLM's reasoning to defend against structural attacks? The paper describes the detection mechanism for structure attacks as using embedding-based cosine similarity, which seems separate from the LLM's reasoning as it would have to use another text encoding model. This appears to contradict the claim that the LLM's capabilities are used to defend against both attack types within a single model.

### Soundness
2

### Presentation
3

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
The paper presents a comprehensive empirical study of GNNs, robust GNNs and Graph LLMs under a unified adversarial setup allowing fair comparions. In particular, the dimensions of perturbing structure vs perturbing text have been investigated. Several insights have been derived such as a text-structure robustnes tradeoff. Further, a method is proposed that has a superior text-structure tradeoff compared to other methods.

### Strengths
* Investigating the disting effect of textual perturbations (compared to small-epsilon perturbations) for node features is interesting and more realistic compared to pervious studies.
* The text-structure tradeoff is an interesting insight and the proposed method a good remedy against. 
* There are many empirical results and insights. However, given its a purely empirical paper, this is also in a way a necessity.
* Code provided.
* Comparing robust GNNs with Graph LLMs in a (unified) adversarial setting is interesting and important.

### Weaknesses
1. Comparison Table 1 feels outdates and slightly misleading. On GNNs & RGNNs, the table only mentions the quite old graph robustness benchmark, while they don't mention recent works on GNNs & RGNNs. Exemplary, the cited work by Gosch et al. 2023 in the paper includes 8 datasets, GMA, evasion & inductive & adaptive attacks but is not mentioned, instead the 5 datsets of the GRB highlighted. I do think that the breath of experiments in submitted work is quite good, but I don't think for metrics such as num. datasets or num. domains the work did a comprehensive and fair survey of existing works, which would be necessary if one quantitavely compares these numbers to previous works.
2. Lines 160-183 mentions sufficiently strong attacks to study model robustness but does not mention adaptive attacks, which are the gold-standard "sufficiently strong attacks" to evaluate methods [1, 2]. I also do not understand the focus in the threat model on transfer attacks, as they have been shown to lead to misleading robustness evaluations in prior work. 
3. Regarding robust training, I'm missing a comparison to adversarially training. In particular, the in the paper chosen GNN model GPR-GNN was shown to be particularly strong against structure perturbations after adversarial training with PR-BCD (Gosch et al. 2023). Thus, a comparison would be interesting.
4. Evasion textual attacks replaced 40% of test set nodes with LLM-text and poisoning textual attacks 80% fo the training nodes. These chooses seem quite strong and I think a gradual investigation (e.g., against 1%, 5%, 10%, 20%, ...) would be more interesting then setting one fixed attack budget.
5. Confusing Paper Structure. Results are presented before the method. E.g., the first results paragraph presents findings on "SFT-neighbor", though the method is not introduced. Even though the findings motivate the development of "SFT-neighbor", I do think that a more classic structure Intro -> Method -> Results would be sufficient with some motivating results for the method mentioned in the intro / beginning of methods, or at least introducing the method in some way more than just a short mention in the introduction.
6. I'm unsure how representative the results of comparing structure attacks for GNNs with an applied adaptive attack (PGD/GRBCD) is compared to a GraphLLM without an adaptive attack and thus, don't know if the statement "GraphLLMs demonstrate inehrent robustnes against evasion attacks." truly holds under an adaptive attack setting.
7. I think the main draft would benefit from a figure that doesn't only compare "rankings" but also some classic "Accuracy vs Attack Budget" plot, one for evasion and one for poisoning. 

[1] Tramèr et al. "On adaptive attacks to adversarial example defenses", NeurIPS 2020    
[2] Mujkanovic et al. "Are defenses for graph neural networks robust?", NeurIPS 2022

### Questions
1. Why did the authors focus on transfer attacks and not adaptive attacks?
2. Line 272: I would say the results are well "complementing the findings in Gosch et al. (2023)" (or something similar) and not "aligning with" as Gosch et al. (2023) investigated evasion and not poisoning. Or alternatively, make explicit that Gosch et al. (2023) shows high robustness of these models for evasion, e.g. through stating "aligning with findings in (Gosch et al., 2023) that show strong performance of these models against evasion attacks".
3. How would SFT-Auto perform in Figure 5?
4. Given many investigated datasets (e.g., the citation networks) are usually distributed with classic BoW embedding, how have the original texts been obtained for computing the rankings against textual attacks such as in Figure 3?

*Minor:*
* Adjust Line 124 (e.g. through reducing spaces) to fit the margins of the text.
* Line 161: Mettack -> Metaattack
* Provide References in line 80/81 for "proven effectivenss" of noise-injection & similarity-filterings

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates the robustness of text-attributed graph (TAG) learning models against textual, structural, and hybrid adversarial attacks. It conducts a large-scale empirical study comparing GNNs, RGNNs, and GraphLLMs, and introduces SFT-auto, an LLM-based defense that automatically detects and recovers perturbed nodes. The results show that SFT-auto achieves more balanced robustness across attack types than prior methods, though the approach is primarily heuristic and lacks strong theoretical grounding.

### Strengths
1. The paper includes extensive baselines and experiments.
2. It offers novel insights into robustness in text-attributed graph learning.
3. The topic is highly relevant and timely.

### Weaknesses
1. The experiments use a fixed perturbation rate; it would be more informative to select a few representative models and show robustness across different perturbation rates.
2. Although effective, the strategies in the SFT-auto pipeline are not very novel.
3. The study mainly reports average ranks across datasets rather than raw accuracy or significance tests. This approach may obscure real performance gaps and overstate the robustness of SFT-auto.

### Questions
1. How is the perturbation for textual attacks defined within a single node? The proportion of perturbed nodes (40%–80%) seems quite large. My central question is: how do the authors justify that these perturbations are not noticeable?
2. Could the authors show the performance degradation of several representative models under varying perturbation rates for textual and combined textual–structural attacks?
3. Could the authors provide more detailed ablations on the inference process of SFT-auto to verify the contribution of each stage?

### Soundness
3

### Presentation
4

### Contribution
3
