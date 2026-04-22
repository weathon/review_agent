# NavTrust: Benchmarking Trustworthiness for Embodied Navigation

- Avg Score: 4.80
- Decision: Reject
- Scores: 6, 4, 2, 6, 6

## Abstract
Embodied navigation remains challenging due to cluttered layouts, complex semantics, and language-conditioned instructions. Recent breakthroughs in complex indoor domains require robots to interpret cluttered scenes, reason over long-horizon visual memories, and follow natural language instructions. Broadly, there are two major categories of embodied navigation: Vision-Language Navigation (VLN), where agents navigate by following natural language instructions; and Object-Goal Navigation (OGN), where agents navigate to a specified target object. However, existing work primarily evaluates model performance under nominal conditions, overlooking the potential corruptions that arise in real-world settings. To address this gap, we present NavTrust, a unified benchmark that systematically corrupts input modalities, such as RGB, depth, and instructions, under realistic scenarios and evaluates their impact on navigation performance.
To the best of our knowledge, NavTrust is the first benchmark to expose embodied navigation agents to diverse RGB-Depth corruptions and instruction variations in a unified framework. Our extensive evaluation of six state-of-the-art approaches reveals substantial success-rate degradation under realistic corruptions, which highlights critical robustness gaps and provides a roadmap toward more trustworthy embodied navigation systems. As part of this roadmap, we systematically evaluate four distinct strategies: data augmentation, teacher-student knowledge distillation, safeguard LLM and lightweight adapter tuning, to enhance robustness. Our experiments offer a practical path for developing more resilient embodied agents. Additionally, we deployed UniNaVid and ETPNav on a real robot under corrupted and mitigated settings. The results and demonstration videos are now included in the supplementary material.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduces **NavTrust**, a unified benchmark for evaluating **trustworthiness and robustness** in embodied navigation across **Vision-Language Navigation (VLN)** and **Object-Goal Navigation (OGN)**.
 NavTrust integrates controlled **RGB, depth, and instruction corruptions**, evaluates six SOTA models, and compares four mitigation strategies: data augmentation, teacher–student distillation, adapter tuning, and safeguard LLM.
 The benchmark exposes significant robustness gaps and proposes a standardized, open evaluation framework.

### Strengths
This paper centers on a new and significant domain within embodied navigation tasks, which is important for understanding the robustness of current navigation models. The authors give a comprehensive benchmark for evaluating models and also propose several methods to enhance robustness, which will be helpful for future research.
- Comprehensive corruption coverage— unifies visual and linguistic robustness settings.
- Systematic experimental setup— six representative models evaluated with SR, SPL, and PRS metrics.
- Insightful comparison of robustness strategies — offers actionable findings for future research.
- Clear and professional writing — well-structured narrative and informative figures.

### Weaknesses
The main concern is that the paper relies on a single dataset (Matterport3D), which may limit the generalizability of its findings (e.g., the observed lack of robustness). The authors should include evaluations in additional environments to validate these conclusions and the proposed robustness-improving methods. 

In addition, some of the methods for improving robustness lack sufficient detail.

### Questions
1. Can NavTrust extend to multilingual or outdoor environments (e.g., RXR), or extend to scenes except in Matterport3D?
2. How does PRS align with robustness metrics like mCE or robustness drop?
3. Is the teacher in distillation unimodal or multimodal?
4. It would be helpful to present the experimental results with a more readable visualization (e.g., consolidated figures or summary tables) to facilitate understanding of the impacts of different corruptions and methods.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces NavTrust, a benchmark designed to evaluate the trustworthiness of embodied navigation agents, including OGN and VLN, under various corrupted and uncertain conditions. The authors propose a set of synthetic corruption settings affecting both visual and language inputs, aiming to simulate realistic challenges such as sensor noise and instruction ambiguity. Alongside the benchmark, the paper also presents mitigation strategies to improve model robustness and trust alignment. Experiments are conducted on several baseline methods to assess their sensitivity to these perturbations and the effectiveness of the proposed approaches.

### Strengths
1. Diverse corruption settings. The benchmark covers a wide range of corruption types—both visual (e.g., RGB and depth distortions) and linguistic (e.g., instruction noise)—providing a comprehensive way to test model robustness across different failure modes.
2. Novelty in introducing corruption-based evaluation for trust. The idea of explicitly modeling and quantifying trust under corrupted conditions for embodied navigation is novel and meaningful, which is a direction that has not been well explored in prior work.
3. Combined benchmark and mitigation strategies. The paper goes beyond benchmark design by also proposing concrete mitigation methods and demonstrating their impact. This dual contribution—offering both a testing framework and practical solutions—adds solid value to the community.

### Weaknesses
1. Limited connection to real-world evaluation. The benchmark is built entirely on synthetically distorted data for both training and testing. While this helps simulate ID and OOD conditions, it doesn’t clearly show how well these settings reflect real-world navigation challenges. To convincingly demonstrate the benchmark’s usefulness, I’d like to see two things: first, whether model performance on these distortions aligns with what happens in real-world cases; and second, whether the proposed mitigation strategies actually help in those real situations. Without that, the synthetic distortions feel somewhat artificial and hard to justify.
2. Baselines are too limited. The benchmark only tests two VLN methods, leaving out many others—especially the newer LLM- or VLM-based approaches that are training-free and could behave differently under such variations. For a benchmark paper, the range of baselines is too narrow, which makes it hard to judge how broadly useful or general the benchmark really is.
3. Some figures and explanations are confusing. A few parts of the presentation could be clearer. For example, Figure 1 seems to imply that RGB and depth corruptions apply only to OGN, while instruction corruption is used only for VLN, which isn’t accurate. The depth image in Figure 2 also looks odd and doesn’t resemble a typical depth map. In Figure 4, the heatmap doesn’t show the uncorrupted performance for comparison, and it’s confusing that the VLFM achieves 99% PRS yet still suffers a noticeable drop in performance. These inconsistencies make it harder to interpret the results.

### Questions
1. Why is SR reported without a percentage sign while SPL uses one?
2. Why does VLFM show the smallest performance drop among all baselines?
3. Since this is a benchmark paper, it would help to include some descriptive analysis—such as the number of cases per corruption category, distribution of difficulty, or other relevant statistics—to give readers a clearer understanding of the dataset’s composition and diversity.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work introduces NavTrust, the first unified benchmark designed to systematically evaluate the trustworthiness and robustness of embodied navigation agents to diverse RGB-Depth corruptions and instruction variation. Unlike prior benchmarks that focus on nominal conditions, NavTrust exposes navigation systems to RGB, depth, and language perturbations to simulate sensor failures and linguistic variations encountered in real-world deployment. Based on the findings of the vulnerabilities in current widely used models, the work compared different strategies to improve the robustness of the model.

### Strengths
The strengths of the paper can be summarized as:
1. The paper is easy to follow, and the presentation of the research results is clear and logically structured.
2. The work focuses on trustworthiness and robustness in embodied navigation, which is an underexplored yet critical area for real-world deployment of embodied systems.
3. The work systematically integrates RGB, depth, and language corruptions into a unified evaluation platform with public available evaluation protocols, which encourages reproducibility of the work and community adoption.
3. The work provides a systematic comparison of different corruptions, followed by different mitigation strategies (data augmentation, distillation, adapters, safeguard LLM), offering practical guidance for building robust embodied systems.

### Weaknesses
The weakness of the paper can be summarized as:
1. Novelty not sufficient: The paper mainly builds upon existing datasets (or said, simulators / environments) and evaluation settings (e.g., Matterport3D), and the technical innovations are relatively limited. Some parts that are overclaimed as contributions are in fact implementation details or engineering tricks rather than advances.
2. Limited analysis of corruptions: Although the paper implements a wide range of corruptions (e.g., motion blur, low-lighting, flare, spatter in rgb observations), these perturbations are mostly experience-driven rather than well motivated. The study lacks deeper theoretical or analytical justification. The authors mainly provide empirical comparisons without further insight.
3. Instruction perturbations are LLM-generated; without human verification, some attacks may not reflect realistic linguistic variation. Concretely, as described in the manuscript:"We generate four stylistic variants (i.e., friendly, novice, professional, and formal) for each R2R instruction using the LLaMA-3.1 model".
4. The PRS metric only focus on one aspect of the model, i.e., success rate. Concretely, "Quantifies robustness ... where Sa,0 is agent a’s clean-split success rate and Sa,k its success rate under corruption k in a family of K corruptions...".
5. Limited qualitative results and lack of real-world validation: The qualitative analysis is limited, concretely, only one closed-loop evaluation of one model (ETPNav) is provided in the accompanying video, which offers little insight into behavior differences among agents. Moreover, the paper entirely lacks real-world experiments or sim-to-real validation, making it difficult to assess how the benchmark findings translate to real-world robot performance.

### Questions
1. The corruption types seem like empirically chosen. Could you elaborate on the systematic analysis or theoretical  for selecting these perturbations? My first intuition is that visual perturbations in real robots often stem from geometric changes, rather than arbitrary pixel-level manipulations as shown in the results of motion blur. Have you considered modeling perturbations from the perspective of robot motion dynamics or geometry-aware image transformations instead of purely image-domain edits?
2. The instruction variations are generated by LLaMA-3.1. Have the authors evaluated whether these generated prompts preserve semantic intent or naturalness? Could you provide more quantitative or qualitative evidence showing that these linguistic corruptions reflect realistic human input variation or adversarial behavior?
3. Since the Performance Retention Score (PRS) only measures success-rate retention, have you considered incorporating additional dimensions such as path efficiency (e.g., SPL-weighted PRS) or failure-mode weighting? Could the authors provide more quantitative results with more metrics?
4. Could the authors provide more examples or visual analyses to illustrate the behavioral differences among models under various corruptions? Additionally, do authors have plans or preliminary attempts to test NavTrust in real-world settings to assess whether robustness trends observed in simulation generalize to real-world robots?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces NavTrust, a unified benchmark for evaluating the trustworthiness and robustness of embodied navigation systems across two major tasks: Vision-Language Navigation (VLN) and Object-Goal Navigation (OGN). The benchmark systematically injects corruptions into RGB, depth, and language instruction modalities to simulate realistic sensing and communication failures. It evaluates six state-of-the-art models and compares four robustness enhancement strategies—data augmentation, teacher-student distillation, adapter tuning, and LLM fine-tuning—under a standardized evaluation protocol. Extensive experiments reveal consistent performance degradation under perceptual and linguistic corruptions, providing insights into model weaknesses and practical guidelines for improving reliability.

### Strengths
Novel Benchmark and Scope: The paper fills a clear gap in embodied AI evaluation by jointly assessing perceptual and linguistic robustness under a unified framework. Prior works such as RobustNav and EmbodiedBench handle only subsets of these aspects.

Comprehensive Corruption Suite: The authors design realistic and diverse corruptions across RGB, depth, and instruction modalities, covering noise, occlusions, adversarial prompts, and stylistic rephrasings.

Empirical Breadth: Evaluation spans six recent navigation models (e.g., ETPNav, NaVid, VLFM, WMNav) and presents clear analyses of their relative robustness, revealing valuable comparative insights.

Constructive Contribution: The study goes beyond diagnosis by benchmarking four mitigation strategies (data augmentation, knowledge distillation, adapter tuning, LLM-based sanitization), which makes the work more actionable for practitioners.

Presentation Quality: The paper is generally well-written, with clear figures (e.g., Figure 2 illustrating corruption types and Figure 4 showing PRS comparisons) that effectively communicate experimental findings.

### Weaknesses
Limited Theoretical Depth: While empirically comprehensive, the paper lacks a theoretical analysis of why certain models fail under specific corruption types (e.g., the deeper mechanisms linking architecture and robustness).

Benchmark Generality: The benchmark is limited to the Matterport3D-based environments and English instructions (R2R dataset). This may restrict its generalizability to other datasets or real-world robotic settings.

Evaluation of Mitigation Strategies: The four robustness strategies are compared in aggregate but not deeply analyzed—for example, ablation studies isolating which aspects of each technique contribute most to PRS improvement would strengthen the work.

Clarity of Metric Interpretation: Although PRS (Performance Retention Score) is useful, it may oversimplify nuanced robustness trade-offs. Discussion on sensitivity or threshold effects could improve interpretability.

Missing Discussion on Computational Overhead: The additional training and evaluation cost of corruptions and mitigations is not discussed, which is relevant for benchmark adoption.

### Questions
See Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces NavTrust, a benchmark designed to systematically evaluate the trustworthiness and robustness of embodied navigation systems. It provides a unified framework for assessing how RGB, depth, and language modalities influence navigation performance under various types of input corruption. The benchmark covers both Vision-Language Navigation and Object-Goal Navigation settings, applying controlled perturbations such as visual noise, depth sensor degradation, and instruction modification. Six state-of-the-art navigation models are benchmarked using metrics such as Success Rate (SR), Success weighted by Path Length (SPL), and a newly proposed Performance Retention Score (PRS), while four robustness enhancement strategies—data augmentation, teacher–student distillation, adapter tuning, and instruction-level defenses—are compared.

### Strengths
1. The paper establishes a unified and open-source benchmark that evaluates both VLN and OGN tasks. It jointly considers RGB, depth, and language modalities, offering a comprehensive view of multimodal robustness in embodied navigation.
	2. Each modality includes a wide range of realistic corruptions—visual noise, depth sensor degradation, and linguistic perturbations—that better reflect the challenges encountered in real-world navigation scenarios.
	3. Provides several mainstream robustness enhancement strategies and conducts corresponding experiments to evaluate their effectiveness.

### Weaknesses
1. The evaluation of depth corruptions focuses mainly on mapping-based methods. Including a broader range of approaches would provide a more comprehensive understanding of depth robustness.

2. In the mitigation stage, the analysis is conducted only on ETPNav, while comparisons across more models would strengthen the conclusions.

### Questions
1. In Table 1, WMNav appears to include depth sensing, but it is not shown as such in Figure 1.

2. In Figure 4 (Image Corruption), the performance of NaVid under motion blur degradation drops more than that of ETPNav, which seems inconsistent with the description around line 341. 

3. The statement around line 403, claiming that “tokenization artifacts (masking, capitalization) and vocabulary coverage dominate the robustness, more so than downstream spatial reasoning failures,” seems not directly supported by experiments. The paper does not include explicit comparisons or ablations related to spatial reasoning, so this contrast may be overstated or require additional evidence.

### Soundness
3

### Presentation
3

### Contribution
2
