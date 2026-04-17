# Causes and Consequences of Representational Similarity in Machine Learning Models

- Decision: Reject
- Scores: 2, 6, 4, 4

## Abstract
Numerous works have noted similarities in how machine learning models represent the world, even across modalities. Although much effort has been devoted to uncovering properties and metrics on which these models align, surprisingly little work has explored causes of this similarity. To advance this line of inquiry, this work explores how two factors—dataset overlap and task overlap—influence downstream model similarity. We evaluate the effects of both factors through experiments across model sizes and modalities, from small classifiers to large language models. We find that generally, both task and dataset overlap cause higher representational similarity. Finally, we consider downstream consequences of representational similarity, demonstrating how greater similarity increases vulnerability to transferable adversarial attacks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates the causes of high representational similarity among machine learning models across both vision and language domains.
The authors hypothesize that dataset overlap and task overlap are the main drivers of this phenomenon.
They introduce dataset and task splitting techniques to systematically control these overlaps and evaluate their effects across multiple architectures, including ResNets, ViTs, diffusion models, and large language models.
Empirical results show that greater overlap leads to higher representational similarity, and that models with more similar representations are more susceptible to transferable adversarial and jailbreak attacks.

### Strengths
[S1] The paper addresses an interesting and timely question related to model alignment and generalization.

[S2] The experimental framework is well-designed and systematic, covering multiple model types and data modalities.

### Weaknesses
[W1] Limited novelty of findings. The main results of this paper, that overlapping datasets or tasks yield more similar representations and higher attack transferability, are intuitive and have already been shown in prior literature.
For example, [1] treats training dataset differences as a key cause of representational dissimilarity, and [2] explores model similarity through adversarial attack transferability across different architectures and training regimes. Similar patterns have also been reported in [3], which links network similarity directly to attack transferability.
Thus, the contributions of this paper are largely confirmatory and show low novelty beyond well-understood phenomena.

[1] What shapes feature representations? Exploring datasets, architectures, and training (NeurIPS 2020)
[2] Similarity of Neural Architectures using Adversarial Attack Transferability (ECCV 2024)
[3] The Relationship Between Network Similarity and Transferability of Adversarial Attacks (2025)

[W2] Limited technical depth and scope. The proposed dataset and task splitting methods are straightforward and mainly procedural rather than methodological innovations. The experiments rely on small-scale synthetic datasets and LoRA-tuned LLMs, which limit the generalizability of the conclusions to large-scale or real-world model training. In addition, the study uses a single similarity metric (CKA) and simple regression fitting, resulting in limited technical rigor and diversity in evaluation.

### Questions
- How does this work provide new insights beyond prior studies [1–3] that already link dataset/task overlap and attack transferability to representational similarity? (see W1)

- Since the study relies mainly on a single similarity metric (CKA) and simple regression analysis, how can the authors ensure that the observed trends are robust across different similarity measures or evaluation methods? (see W2)

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Recent work hypothesizes that as AI models scale up and train on more tasks, their representations will converge because their task sets increasingly overlap. This paper empirically tests what drives representational similarity by isolating two factors: dataset overlap (shared training datapoints) and task overlap (shared objectives/classes). The authors develop controlled splitting methods to vary these factors independently and train model pairs (ResNets, ViTs, LLMs, diffusion models) with different overlap levels, measuring representational similarity via CKA. They introduce ColorShapeDigit800K, a vision dataset enabling task variation while keeping images identical. For vision models, the results are clear: both dataset and task overlap increase representational similarity, with combined overlap showing the strongest effect. Language model results (Llama fine-tuning with LoRA) are largely inconclusive, showing weak or absent correlations—whether this reflects LoRA limitations or fundamental differences remains unclear. The authors explore downstream consequences by testing vulnerability to transfer attacks: models with higher representational similarity show increased susceptibility to adversarial examples (vision), suggesting that unintended similarity between models may create shared security vulnerabilities. 

The work is rigorous and provides valuable empirical evidence for the vision domain, though language results and restriction to same-architecture-and-size comparisons can be limiting.

### Strengths
- Clearly motivated work addressing an important gap: trying to understand *causes* rather than just observing representational similarity
- Well-designed methodology to isolate dataset and task overlap effects through controlled splitting
- Broad empirical evaluation across multiple modalities (vision, language, diffusion)
- Creative ColorShapeDigit800K dataset enabling task manipulation without dataset variation
- Novel connection to downstream security implications (adversarial transferability)
- Extensive appendix with multiple similarity metrics and detailed experimental settings
- Well written and enjoyable to read!

### Weaknesses
- Language model results are inconclusive and undermine the paper's central claims of dataset and task overlap strongly influencing the representational similarity:
    - Figure 19 (Llama fine-tuning): Only small positive correlations appear with CKA for task splitting; data splitting and local similarity measures show no effect from increasing overlap. Authors attribute this to LoRA without further hypothesis. Can the authors hypothesize beyond "LoRA limitations"? Is this a fundamental difference in how language vs vision models respond to fine-tuning, or purely a methodological artifact?
    - NanoGPT (trained from scratch): No linear trends provided, making assessment difficult
    - Critical: Since Llama results show no positive correlation with overlap increase across most metrics, the abstract/conclusion claims about "both task and dataset overlap" driving similarity might be overstated. The evidence supports this claim for vision models but not language models. The paper could either remove language claims or explicitly limit conclusions to vision models.
- Data vs Task vs Data+Task overlap analysis unfairly compares vision and language models when task overlap metrics only apply to vision (Section 5.3).
- As mentioned in the discussion, the work is limited to same-architecture comparisons; making the same observations cross-architecture results (ResNet vs ViT or small vs large architectures) would strengthen claims about the influencing factors of representational alignment.
- Figure 4:
    - ResNet101 shows anomalous high variance compared to other ResNets (Figure 4).  Also visible when using other similarity metrics.
    - Limited discussion of why ViTs show surprisingly low similarity values despite similar training protocols to ResNets. I assume that the ViTs have also been pretrained with ImageNet. Comparing two models after finetuning on TinyImageNet, shouldn’t they have much higher CKA values?
    - Could the authors comment on these?
- Some figures raise questions about correlation validity (e.g., Figure 6's high R² when points cluster at extremes). Can the authors comment on this?
- Presentation of key results could be improved: The paper uses boxplots with regression lines throughout, which becomes repetitive across dozens of panels with small fonts. While 7 similarity metrics are computed (appendix), the main text focuses on CKA minimally explaining this choice or what different metrics reveal. Better aggregation (summary tables, effect size comparisons across conditions) and/or diverse (aggregated) visualization approaches would significantly improve clarity.
- Missing minor details: which pretrained weights were used? Assumed ImageNet.

### Questions
- Please see questions already stated in weaknesses :)
- Could you provide cross-architecture comparisons (ResNet18 vs ResNet152, ResNet vs ViT)? While same-architecture controls for confounds, demonstrating the trend across architectures would significantly strengthen the claim that dataset/task overlap drives similarity generally.
- Figure 4: ResNet101 shows much larger variance than ResNet18/50/152. What causes this outlier behavior? Why do other ResNets show such tight distributions?
- Figure 4: How many CKA values are in one box?
- CKA values for ViTs fall in the 0.4-0.6 range where the metric has known interpretability issues (high variability in this region, Figure 4). How confident are you in the regression lines fitted to these mid-range values? Would bootstrapping over data subsets provide more reliable similarity estimates?
- Figure 6: The high R² values seem driven by clustering at extremes rather than a true linear relationship. The level 2 combinations appear to form a blob in the middle. Can you explain this pattern? Why aggregate all level 2 combinations rather than separating them?
- Section 5.2 conclusion (2) states task splitting influences representation alignment "independent of data-overlap" at 100% data overlap. Wouldn't showing results at 50% data overlap make this argument stronger?
- The task overlap metric only applies to vision models, making Section 5.3 comparisons across modalities questionable. Can acknowledge this limitation more explicitly?
- Could you elaborate on why you focus primarily on CKA when 7 metrics were computed? What do the other metrics reveal that CKA (with a linear kernel) doesn't? CKA can also be used with different kernels, allowing to measure similarity more locally.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors systematically study how the two different factors (task and data overlap) contribute to the representational similarity of models trained from scratch. They created a new dataset called ColorShapeDigit800K which maintains a fixed set of data points while varying the task definition. With this dataset, they show that both task and dataset overlap cause higher representational similarity and that combining them provides the strongest effect. They further show that a high similarity between model representations also increases the vulnerability to transferable adversarial and jailbreak attacks between the one model to the other.

### Strengths
* The paper studies an interesting question that can have large implications for building and evaluating foundation models.

* The paper is well written and the experiments and results are described clearly.

* The authors make an effort to make the underlying factors configurable and do a good job of holding one property constant (e.g. data overlap) while varying the other (e.g. task overlap).

### Weaknesses
* The different tasks that are studied for ColorShapeDigit800K are all simple character/digit/color classification tasks while the difference between different SSL objectives (multi-view, masking), image/text, supervised models, text models seem to be larger in my view. I’m not sure if it is possible to approximate this with these simple tasks.

* The results that more tasks and data overlap increase representationals similarities feel a bit trivial. Although it is nice that the authors confirm that in a controlled way, I’m missing a novel insight from these results.

* It would be interesting to investigate whether these are the only factors that contribute to representational similarity or whether other factors (architecture, model size, training duration) also have a significant impact.

### Questions
* How were the representations exactly extracted for the different models?

* Why didn’t you use the CKNNA measure from the The Platonic Representation Hypothesis when you already point out downsides of CKA in the Limitations section?

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
4

### Summary
This paper examines the extent that dataset and task overlap correlate with representation similarity. They use a new image dataset that fixes the images and varies labels and objectives; this isolates task overlap. They also perform some experiments on the correlation between jailbreaks and representation similarity.

### Strengths
- Novel causal interventions on causes of model representation alignment. The paper systematically modifiies dataset overlap and task overlap, and measures how these correlate CKA/other alignment metrics for both ViTs and small LLMs.
- Isolate task vs data overlap in representation similarities. Perform experiments on a number of different measures (CKA, nearest-neighbor scores, mutual KNN).

### Weaknesses
- Concerns about generality. In general, the results on language modeling seem to be in tension with the stated takeaways of the paper, and the authors do not properly explain this discrepancy. Namely, they note that for the Llama fine-tuning experiments, the models in general have low correlation between their CKA scores and dataset overlap. This is comparatively true for task overlap as well. This lower correlation is hypothesized to be due to the fact that the LLMs were fine-tuned, so a lower proportion of their total training tokens are explained by the new task/dataset training. The authors note in the main body that they also run nanoGPT from scratch; however, these results seem to only be discussed/provided in the appendix (e.g. Figure 13). While these results are not discussed in the paper and the actual correlations between overlap and similarity are not provided, it visually it appears in Figure 13/14 that the correlation between dataset/tasks is pretty weak. The authors should both provide these correlations address this discrepancy. This also goes for the jailbreak correlation results in section 6.
- This paper has some conceptual overlap with [1, 2, 3]. These are discussed briefly in the paper at the moment, but it would be great to see a bit more discussion here.


[1] Ciernik, Laure, et al. "Objective drives the consistency of representational similarity across datasets." Forty-second International Conference on Machine Learning.

[2] Klabunde, Max, et al. "Towards measuring representational similarity of large language models." arXiv preprint arXiv:2312.02730 (2023).

[3] Brown, Davis, et al. "Wild comparisons: A study of how representation similarity changes when input data is drawn from a shifted distribution." ICLR 2024 Workshop on Representational Alignment. 2024.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
