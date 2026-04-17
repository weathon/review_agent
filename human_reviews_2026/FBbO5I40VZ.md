# PromptHub: Enhancing Multi-Prompt Visual In-Context Learning with Locality-Aware Fusion, Concentration and Alignment

- Decision: Accept (Poster)
- Scores: 6, 4, 6

## Abstract
Visual In-Context Learning (VICL) aims to complete vision tasks by imitating pixel demonstrations. Recent work Condenser pioneered prompt fusion that combines the advantages of various demonstrations, which shows a promising way to extend VICL. Unfortunately, the patch-wise fusion framework and model-agnostic supervision hinder the exploitation of informative cues, thereby limiting performance gains. To overcome this deficiency, we introduce PromptHub, a framework that holistically strengthens multi-prompting through locality-aware fusion, concentration and alignment. PromptHub exploits spatial priors to capture richer contextual information, employs complementary concentration, alignment, and prediction objectives to mutually guide training, and incorporates data augmentation to further reinforce supervision. Extensive experiments on three fundamental vision tasks demonstrate the superiority of PromptHub. Moreover, we validate its universality, transferability, and robustness across out-of-distribution settings, and various retrieval scenarios. This work establishes a reliable locality-aware paradigm for prompt fusion, moving beyond prior patch-wise approaches. Code is available at https://github.com/luotc-why/ICLR26-PromptHub.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
In this paper, the authors propose PromptHub, a locality-aware multi-prompt fusion framework for Visual In-Context Learning (VICL). By integrating three cooperative losses (semantic alignment, utilization, and prediction), PromptHub addresses the shortcomings of patch-wise fusion methods like CONDENSER.

### Strengths
1. The overall idea is well-motivated, as the paper clearly identifies the limitation of patch-wise fusion in existing multi-prompt VICL methods and introduces a locality-aware fusion mechanism that intuitively enhances spatial coherence and prompt utilization.

2. Extensive experiments showing consistent gains across segmentation, detection, and colorization.

### Weaknesses
1. The claim that PromptHub ‘establishes an interpretable paradigm’ is somewhat overstated. Moreover, the analysis of interpretability remains superficial, lacking quantitative evidence or deeper reasoning about what the fused representations capture.

2. The visualization results are not entirely convincing.  For instance, in some examples (e.g., prompt 1), background regions such as the sky or ground areas from unrelated prompts (e.g., prompt 2) are also highlighted, suggesting that the fusion may still mix irrelevant spatial cues rather than providing truly interpretable alignment.

3. The replacement probabilities $p_q$，$p_r$ are not justified, and the approach lacks ablations on different augmentation ratios. It’s unclear whether improvements come from the locality-aware fusion or the added stochastic augmentation.

4. The paper does not explicitly discuss the limitations of the proposed method.

### Questions
Could the authors provide some failure cases or qualitative examples where PromptHub does not perform well, and analyze the possible reasons behind these failures?

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
4

### Summary
Focues on the visual in-context learning, this paper proposes PromptHub, a novel learning-based framework to generate the optimal fused examples for each query from N examples. To do this, PromptHub first designs a locality-enhanced cross-attention strategy between the query image and examples to generate the unified prompt, and then develop three alignment loss to train the fusing process. Extensitve results on three vision tasks (segmentation, detection, and coorization) show the effectiveness of the proposed PromptHub.

### Strengths
1) The task of generating optimal prompt in VICL is a hot topic in the community. And the results of PormptHub show the improvements of this method.

2) Three new learning objectives are developed to train the fusing process.

### Weaknesses
1), One of the main concerns lies in the novelty of the work. Given that the previous CONDENSER model also generates fusing prompts from the query image and N pairs, I find that the locality-aware attention and training losses developed in PromptHub may limit the novelty of this paper. I hope the authors can clarify the core differences between the previous work and PromptHub in terms of the main motivation, fusing strategy, and training process.

2), Table 3 presents ablation results demonstrating the effectiveness of the designed modules, which helps readers understand their contributions. However, I notice that when using the global fusing strategy (a traditional approach), the performance of PromptHub is lower than that of CONDENSER. Does this suggest that the three newly introduced losses do not effectively improve alignment? The authors should provide a deeper discussion of this issue.

3), What is the motivation behind the three proposed losses, particularly L_s and L_u ? These losses might negatively affect the fusing process if their hyperparameters are not properly tuned.

4), Table 2 reports cross-dataset results of PromptHub, showing the strong transferability of the proposed model. I am also interested in its cross-task performance—specifically, when the model is trained on segmentation but tested on detection. This setting would be highly valuable to the VICL community, as it evaluates the model’s ability to perform in-context learning on unseen tasks.

### Questions
Please see the Weaknesses section above.

### Soundness
3

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
This paper focuses on enhancing the multi-prompting capability of MAE-VQGAN through the method of prompt fusion. The authors identify two main limitations of the recent prompt-fusion method CONDENSER: it fails to fully utilize local visual cues from multiple prompts, and the discrepancies that arise from prompt fusion can cause the model to revert to its standard inference method. To address these issues, the authors introduce PromptHub, which spatially aggregates relevant information from multiple prompt pairs using cross-attention. Additionally, they implement learning objectives and data augmentation for prompt pairs to help mitigate the discrepancies. Experiments conducted on different benchmarks demonstrate the effectiveness of the proposed method.

### Strengths
- The concept of utilizing local-aware aggregation for prompt fusion and the design of learning objectives is illustrative and aligned with the original motivation.

- The performance improvement is significant compared to competitors.

- The writing is generally well-structured and easy to understand.

### Weaknesses
- **Interpretability**: As shown in Figure 6, the fused prompts of Condenser align closely with the target regions of the queries. In contrast, the generated object contours in PromptHub are misaligned with the queries. To improve clarity, it would be beneficial to include additional visual results, such as attention maps of the prompt regions similar to those in Figure 13.

- The cross-attention operation is constrained by the coordinate position. What would happen if the prompt pairs have distinct positions compared to the query? I wonder if this situation would harm the model's performance.

### Questions
Please address the concerns mentioned in the weaknesses section.

In Section 3.4 (ii), random pairs are sampled as prompt pairs to improve the robustness of PromptHub. I am curious if the authors have assessed the model's performance when these prompt pairs are combined with noisy pairs.

### Soundness
3

### Presentation
3

### Contribution
3
