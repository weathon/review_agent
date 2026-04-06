=== CALIBRATION EXAMPLE 82 ===

# Harsh Critic Review
## Section-by-Section Critical Review

**Title & Abstract**
The title accurately reflects the paper's core contribution. The abstract is well-structured, clearly motivating the problem (long, complex multimodal policies, computational cost, lack of prior work), stating the new task (MPI), and summarizing the key contributions (two datasets, the TriMPI framework with PolicyRollout). The claims of "significant improvements" (e.g., 70.7% gain over CoT SFT) are strong and set clear expectations for the results.

**Introduction & Motivation**
The motivation is compelling and well-articulated. The transition from text-only prompt compression/deliberative alignment to the unmet need for *multimodal* policy internalization is logical. The challenges of MPI are clearly outlined (reasoning-intensive tasks, lack of data/training paradigms). The three contributions are restated. A minor point: the motivation for policy length (1K-50K tokens) relies on a non-public source, but the efficiency argument remains reasonable.

**Problem Formulation (Section 2)**
Clear and concise formalization of the MPI task. The distinction from methods that learn special embeddings (soft prompts) is appropriately justified by the desire to maintain general capabilities.

**Dataset Creation (Section 3)**
*   **ClevrPolicy**: A well-designed synthetic benchmark that allows controlled study of policy complexity (via decision tree depth N). The T (text) and M (multimodal with image demos) variants are a good addition. The automatic generation process is sound.
*   **GTAPolicy**: Addresses a realistic, low-data regime tool-use scenario with complex rules (versioning, user conditionals). The reformulation from multi-turn to single-turn simplifies evaluation but is justified for the initial MPI study.
*   **Zero-shot Results (Table 1)**: This is a crucial baseline that effectively demonstrates the inherent difficulty of the policies, even for powerful models like Claude-4. It justifies the need for specialized internalization methods.
*   **Concern**: The datasets, while novel and useful, are limited in scope (synthetic imagery and a small, reformulated real-world dataset). The authors acknowledge this in the limitations, which is fair.

**Method / Approach (Section 4)**
*   **Baselines**: Direct SFT and CoT SFT are standard and appropriate.
*   **TriMPI Framework**: The three-stage design is logically motivated. The key innovation is PolicyRollout (PoRo).
    *   **VM-CPT**: A straightforward adaptation of continual pretraining via visual masking. Its effectiveness is demonstrated empirically.
    *   **PolicyRollout**: This is the core algorithmic contribution. The idea of augmenting the rollout space with policy-conditioned responses while computing policy gradients *only* on the no-policy path is clever and elegantly solves the alignment problem between training and inference. The description (Fig. 5, Eq. 3) is clear.
*   **Questions/Clarifications Needed**:
    1. **VM-CPT Rationale**: A more intuitive explanation of *why* predicting the policy text (with visuals masked) effectively injects knowledge would be helpful. The simple approach works, but the mechanism could be elaborated.
    2. **PoRo Generality**: The paper presents PoRo as an extension to GRPO/DAPO. Could it be applied to other on-policy RL algorithms (e.g., PPO)? A brief discussion would be useful.
    3. **Reward Design**: The paper assumes the existence of verifiable rewards (accuracy, format). For broader real-world policies, reward design is a non-trivial challenge. Acknowledging this as a current limitation of the MPI formulation would be appropriate.

**Experiments & Results (Section 5)**
This section is a major strength of the paper—comprehensive and convincing.
*   **Main Results (Table 2)**: Shows very strong performance gains from TriMPI. The ablation cleanly isolates the contributions of each stage (RL provides the biggest jump, VM-CPT helps RL more, PoRo adds further gains). The efficiency analysis (Fig. 6) concretely quantifies the inference benefits.
*   **Generalization Evaluations (Tables 3 & 4)**: The "Policy Override" and "Policy Referral" evaluations are **excellent** and significantly elevate the paper. They move beyond simple task accuracy to test whether the model can generalize to updated policies and whether it has genuinely internalized policy knowledge (as judged by LLM evaluation of reasoning consistency). Strong results here are compelling.
*   **Robustness (Tables 9, 10)**: Demonstrating that TriMPI suffers less from catastrophic forgetting on general reasoning and safety benchmarks is critical for practical viability, especially on the small GTAPolicy dataset.
*   **Error Analysis & Branching Error (Sec 5.6, Fig 14)**: Provides valuable insight into failure modes. The quantitative analysis showing TriMPI reduces hallucination of non-existent policy conditions is a direct proof of its improved grounding.
*   **Model Size & Complexity (Table 8)**: Shows consistent gains and that benefits are more pronounced for complex policies, which is desirable.
*   **Potential Concerns**:
    1. **Statistical Significance**: Results are presented as single percentage points. Given the test set sizes (n=2000 for ClevrPolicy, n=106 for GTAPolicy), reporting confidence intervals or conducting significance tests would strengthen the claims, especially for smaller margins.
    2. **Compute Cost**: The three-stage training, particularly RL with double rollouts (PoRo), is computationally expensive. A discussion of the training cost trade-off versus inference efficiency would provide a more complete picture for practitioners.
    3. **GTAPolicy Scale**: The very small size of GTAPolicy (451 training instances), while testing a low-data regime, raises questions about stability. The 50 RL steps seem reasonable, but the small test set (106) limits the granularity of results.

**Writing & Clarity**
The paper is generally well-written and well-structured. The figures are helpful. The appendix is extensive and provides necessary details (dataset samples, CoT data, full policies, error examples). Parser-induced formatting artifacts (e.g., broken references) do not hinder understanding.

**Limitations & Broader Impact**
The limitations are appropriately discussed in Section 7 (dataset diversity, simplicity of VM-CPT, handling task mixtures). The ethics statement is standard and suitable for this type of foundational research.

### Overall Assessment
This is a **high-quality paper** that makes a significant and timely contribution. It identifies a clear gap (multimodal policy internalization), introduces two well-designed benchmarks, and proposes a novel and effective training framework centered on the PolicyRollout algorithm. The experimental evaluation is exceptionally thorough, moving beyond basic accuracy to probe generalization, knowledge internalization, and robustness—exactly the kind of rigorous analysis expected at ICLR. The results are strong and well-supported.

**The main concerns are relatively minor**: the limited scope of the benchmarks (acknowledged), the lack of statistical significance measures, and the omission of a discussion on training compute cost. These do not detract from the core contributions. The paper provides a strong foundation for future research and meets the high bar for ICLR. **Acceptance is recommended.**

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces Multimodal Policy Internalization (MPI), a novel task aiming to internalize complex multimodal policies (containing text and visual instructions) into model parameters, eliminating the need for lengthy in-context policy prompts during inference. The authors create two new benchmarks (ClevrPolicy and GTAPolicy) covering decision-making and tool-using tasks, and propose TriMPI, a three-stage training framework featuring a novel PolicyRollout RL algorithm that augments exploration with policy-grounded responses. Experiments demonstrate substantial performance improvements over strong baselines, with gains in efficiency, generalization, and robustness to catastrophic forgetting.

### Strengths
1. **Well-Defined Novel Task and Strong Motivation**: The paper clearly articulates a timely and practical problem—the inefficiency and performance limitations of long, complex multimodal in-context policies in conversational agents. It effectively bridges gaps between prior work on text-only prompt compression/deliberative alignment and the emerging need for multimodal policy adherence, aligning well with ICLR's focus on foundational ML challenges.
2. **Comprehensive and Rigorous Experimental Design**: The authors construct two distinct, well-motivated datasets (synthetic ClevrPolicy for controlled analysis and real-world GTAPolicy for practical relevance) with varying complexity levels. The evaluation is exceptionally thorough, covering: (i) main task performance with strong baselines, (ii) detailed ablation studies isolating each component of TriMPI, (iii) generalization via "Policy Override," (iv) policy knowledge probing via "Policy Referral," (v) robustness to catastrophic forgetting on general (MMMU/MMLU-Pro) and safety (WildGuard) benchmarks, and (vi) efficiency analysis. The reported gains (e.g., up to ~80% absolute improvement over in-context) are substantial and well-supported.
3. **Innovative and Effective Methodological Contribution**: The TriMPI framework, particularly the PolicyRollout (PoRo) algorithm, presents a simple yet clever solution to a key challenge: leveraging the policy for grounded RL exploration without creating a train-inference mismatch. The three-stage design (VM-CPT, CoT-SFT, RL) is logically justified, and the paper provides clear evidence (e.g., in Table 2 and error analysis) that each stage contributes to final performance, especially on complex policies.

### Weaknesses
1. **Limited Scale and Real-World Diversity of Datasets**: While GTAPolicy uses real images, its size is small (451 training instances), and it is adapted from a single existing dataset. The policies, though complex, may not fully represent the extreme length, ambiguity, or dynamic nature of policies in production multimodal agents. The paper acknowledges this limitation, but it somewhat constrains the claim of real-world applicability.
2. **Insufficient Analysis of the VM-CPT Stage's Mechanism**: The visually-masked continual pretraining stage is motivated as direct policy knowledge injection, but its operational mechanism and necessity remain somewhat superficial. Why is simple next-token prediction on the policy (with visual masking) effective? A deeper analysis—e.g., probing what the model learns in this stage or comparing against alternative policy-encoding strategies—would strengthen the methodological contribution.
3. **Light Theoretical Justification for PolicyRollout**: PolicyRollout is empirically effective but presented largely as a heuristic. The paper would benefit from a more formal discussion of how augmenting the rollout space with policy-conditioned responses influences the policy gradient (e.g., reduces variance, provides a better baseline) or connects to related RL concepts like auxiliary tasks or hindsight experience replay.

### Novelty & Significance
**Novelty**: The work is the first to formally define and systematically study Multimodal Policy Internalization, a meaningful extension of deliberative alignment into the multimodal domain. The PolicyRollout algorithm and the three-stage TriMPI framework are novel contributions. The ClevrPolicy and GTAPolicy datasets also fill an important resource gap.
**Significance**: The problem is of high practical importance for efficient and reliable deployment of multimodal conversational agents. The proposed solution demonstrates significant performance and efficiency gains, and the comprehensive evaluation suite (including generalization and robustness checks) meets ICLR's high standards for empirical rigor. The release of datasets, code, and training recipes will likely catalyze further research in this area.

### Suggestions for Improvement
1. **Expand Real-World Validation**: Address the dataset limitation by either (a) including a small-scale experiment with a more diverse, proprietary policy (even if anonymized/obfuscated) to demonstrate broader applicability, or (b) more thoroughly discussing the steps needed to scale GTAPolicy (e.g., data collection pipelines, policy complexity axes) in future work.
2. **Deepen Analysis of the VM-CPT Stage**: Conduct a probing study or representational analysis to show what specific knowledge (e.g., policy structure, rule relationships) is encoded during VM-CPT and how it facilitates later SFT/RL. Comparing against ablations like policy summarization or direct fine-tuning on the policy text could further justify the design choice.
3. **Provide Theoretical Grounding for PolicyRollout**: Include a subsection or appendix discussion framing PolicyRollout within RL theory. For example, analyze how the added rollouts affect the advantage estimate's bias/variance, or formally show that the gradient update remains aligned with the original objective despite the augmented inputs.
4. **Discuss Training Compute Trade-offs**: While inference efficiency gains are clear, the three-stage training (especially RL with doubled rollouts) is computationally expensive. A brief discussion on the training cost trade-off versus the inference benefits would provide a more complete picture for practitioners considering adoption.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Ablation on the contribution of simply training with more data.** The three-stage TriMPI uses different data combinations (VM-CPT uses CoT data, RL uses non-CoT data). A critical missing baseline is a model trained with the same total data budget (CoT + non-CoT) in a single extended CoT SFT stage. Without this, it's unclear if the gains come from the proposed staged architecture or simply from more training data/compute.
2. **Comparison to a straightforward distillation baseline.** The paper dismisses soft prompting but does not compare to a simple "train with policy in context, infer without" full fine-tuning or LoRA baseline. This is the most direct approach to internalization and must be shown to be inferior to TriMPI to justify the method's complexity.
3. **Systematic evaluation of Policy Override capacity.** The override experiment uses a single modification. To properly assess generalization, the authors should test a range of overrides: from minor edits (changing one condition value) to major structural changes (adding/removing decision branches). This would show the limits of the internalized model's flexibility.
4. **Ablation on the visual masking strategy in VM-CPT.** The core adaptation of CPT for vision is masking visual token loss. An experiment training VM-CPT *without* masking visual tokens is necessary to validate that this design is crucial for effective multimodal policy injection.

### Deeper Analysis Needed (top 3-5 only)
1. **Quantitative analysis linking error modes to policy structure.** The error analysis categorizes errors as "perception" or "reasoning" but doesn't quantify how error rate scales with specific policy complexities (e.g., tree depth, branching factor, number of visual attributes per condition). This is needed to predict where the method will fail in practice.
2. **Probing what knowledge is actually internalized.** The Policy Referral score relies on an LLM judge. A more direct analysis is needed: e.g., can the model correctly answer *questions about the policy itself* (e.g., "What does Condition 1.1 check?") after internalization? This would directly test knowledge embedding versus behavioral mimicry.
3. **Analysis of the trade-off between internalization strength and in-context learning ability.** The paper shows Policy Override works to some degree. A key analysis is: as internalization training progresses (more RL steps), how does the model's ability to follow a new, contradictory in-context policy degrade? This defines the method's operational boundary.
4. **Cost-benefit analysis of training overhead.** The efficiency gains are measured only at inference. The substantial compute cost of the three-stage training (especially RL with PolicyRollout) must be compared to the inference savings over a realistic query volume to justify the approach economically.

### Visualizations & Case Studies
1. **Visual tracing of the model's reasoning path against the policy decision tree.** For ClevrPolicy, overlay the model's chain-of-thought steps on the actual policy tree to visually pinpoint where it diverges (e.g., misreads a condition, takes a wrong branch). This would make reasoning errors concrete.
2. **Case studies of PolicyRollout in action.** Show examples where a rollout *without* the policy leads to a low-reward action, and the augmented rollout *with* the policy leads to a high-reward, policy-grounded action that then improves the model. This would visually demonstrate the algorithm's core mechanism.
3. **Visualization of failure cases under Policy Override.** Show specific examples where the internalized model incorrectly prioritizes its internal policy over a new, conflicting in-context instruction. This would illustrate a critical failure mode for real-world deployment.

### Obvious Next Steps
1. **Hybrid internalization-retrieval approach.** The most obvious next step for real-world use is to combine internalization of core policies with the ability to retrieve and incorporate additional, dynamic policy snippets in-context. The paper should have discussed this direction as a necessary evolution.
2. **Multi-policy internalization and conditioning.** A direct extension is to train a single model to internalize multiple distinct policies and condition its behavior on a policy identifier (e.g., "follow policy A"). This is essential for serving multiple use cases with one model.
3. **Application to a real, lengthy commercial policy document.** The next clear step is to test TriMPI on an authentic, long, and linguistically complex policy from a domain like customer service or technical support. This would stress-test the method on natural language complexity beyond structured synthetic rules.

# Final Consolidated Review
## Summary
This paper introduces Multimodal Policy Internalization (MPI), a new task that aims to internalize complex multimodal policies (containing text and visual instructions) into model parameters, eliminating the need for lengthy in-context prompts during inference. The authors construct two new benchmarks (ClevrPolicy and GTAPolicy) covering decision-making and tool-use tasks, and propose TriMPI, a three-stage training framework featuring a novel PolicyRollout RL algorithm that augments exploration with policy-grounded responses. Experiments show substantial performance gains over baselines, with improvements in efficiency, generalization to policy updates, and robustness to catastrophic forgetting.

## Strengths
- **Novel and well-motivated problem definition.** The paper clearly identifies a gap between prior work on text-only prompt compression/deliberative alignment and the emerging need for handling complex, reasoning-intensive multimodal policies in conversational agents. The task formulation is timely and practically relevant.
- **Comprehensive and rigorous experimental evaluation.** The authors create two well-designed datasets (synthetic ClevrPolicy for controlled analysis and real-world GTAPolicy for low-data regimes) and conduct an exceptionally thorough evaluation. Beyond main task performance, they introduce novel evaluations like "Policy Override" (generalization to updated policies) and "Policy Referral" (probing internalized knowledge), measure robustness to catastrophic forgetting on general and safety benchmarks, and demonstrate significant inference efficiency gains. The reported improvements (e.g., up to ~80% absolute gain over in-context baselines) are substantial and well-supported by ablations.
- **Effective and innovative methodological contribution.** The TriMPI framework, particularly the PolicyRollout algorithm, presents a clever solution to a key challenge: leveraging the policy for grounded RL exploration without creating a train-inference mismatch. The three-stage design (VM-CPT, CoT-SFT, RL) is logically motivated, and each component is shown to contribute, especially on complex policies.

## Weaknesses
- **Limited scale and diversity of real-world evaluation.** While GTAPolicy uses real images and complex rules, its training set is small (451 instances) and derived from a single existing dataset. This constrains the strength of claims about broad real-world applicability, though the paper acknowledges this limitation.
- **Insufficient mechanistic analysis of the VM-CPT stage.** The visually-masked continual pretraining stage is presented as direct policy knowledge injection, but the paper lacks deeper analysis of why this simple approach works. A probing study or comparison to alternative policy-encoding strategies would strengthen the methodological contribution.
- **Light theoretical justification for PolicyRollout.** The algorithm is empirically effective but presented largely as a heuristic. A more formal discussion of how augmenting the rollout space influences the policy gradient (e.g., variance reduction, connection to auxiliary tasks) would improve the foundation.

## Nice-to-Haves
- A brief discussion of the training compute trade-off (three-stage training, especially RL with PolicyRollout) versus the inference efficiency gains, to provide a complete picture for practitioners.
- Extension of the Policy Override evaluation to a range of policy modifications (from minor edits to structural changes) to better characterize the model's generalization boundaries.

## Novel Insights
The paper's core novel insight is that complex multimodal policies can be effectively internalized into model parameters via a staged training process that includes policy-aware reinforcement learning. The PolicyRollout algorithm demonstrates that augmenting the RL rollout space with policy-conditioned responses—while computing gradients only on the no-policy path—enables grounded exploration without train-inference mismatch. Furthermore, the evaluation reveals that successful internalization not only improves task accuracy but also embeds policy knowledge (as shown by Policy Referral scores) and allows the model to gracefully override internalized rules when new policies are provided in-context.

## Suggestions
- Conduct a probing study to analyze what specific policy knowledge (e.g., structure, rule relationships) is encoded during the VM-CPT stage and how it facilitates later SFT/RL.
- Include a theoretical discussion or appendix framing PolicyRollout within RL theory, explaining its effect on advantage estimation or variance reduction.
- If possible, expand the real-world evaluation with a larger or more diverse policy dataset (even if anonymized) to further validate scalability.

# Actual Human Scores
Individual reviewer scores: [6.0, 8.0, 8.0]
Average score: 7.3
Binary outcome: Accept
