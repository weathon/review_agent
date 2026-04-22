# SPAR: Self-supervised Placement-Aware Representation Learning for Distributed Sensing

- Avg Score: 4.40
- Decision: Reject
- Scores: 6, 6, 4, 2, 4

## Abstract
We present SPAR, a framework for self-supervised placement-aware representation learning in distributed sensing. Distributed sensing spans applications where multiple spatially distributed and multimodal sensors jointly observe an environment, from vehicle monitoring to human activity recognition and earthquake localization. A central challenge shared by this wide spectrum of applications, is that observed signals are inseparably shaped by sensor placements, including their spatial locations and structural roles. However, existing pretraining methods remain largely placement-agnostic.
SPAR addresses this gap through a unifying principle: the duality between signals and positions. Guided by this principle, SPAR introduces spatial and structural positional embeddings together with dual reconstruction objectives, explicitly modeling how observing positions and observed signals shape each other. Placement is thus treated not as auxiliary metadata but as intrinsic to representation learning.
SPAR is theoretical supported by analyses from information theory and occlusion-invariant learning. Extensive experiments on three real-world datasets show that SPAR achieves superior robustness and generalization across various modalities, placements, and downstream tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper address self-supervised representation learning in distributed sensor networks, where sensors have a particular spatial configuration (e.g. GPS coordinates of sensor locations). Instead of treating the spatial information of each sensor as meta-data, the paper proposes Self-supervised Placement-Aware Representation learning (SPAR), a new method which introduces spatial embeddings into the representations. SPAR achieves this by following a dual Masked Auto-Encoder (MAE) setup: In its encoder, first shared embeddings are computed for both a signal tokens, and for the spatial sensor positions. The embeddings are then summed to single signal+spatial tokens. To learn these embeddings, two separate masked versions of these tokens are created: one where the signal part is masked, and one where the spatial part is masked. The masked tokens are reconstructed to their unmasked equivalents by two separate decoders, again one for the signal part, and one for the spatial part.
Experiments on three benchmarks show that SPAR pre-training improves performance over other pre-training strategies across various distributed sensing tasks.

### Strengths
* This idea of using spatial information about the sensor configuration into the embedding space is novel;
* The paper also shows other types of meta-data can be encoded (e.g. via LLM).
* Experiments on three real-world distributed sensing benchmarks show SPAR pre-training significantly outperforms other pre-training strategies.
* The paper does not just present empirical results, but also an information theoretic perspective on how SPAR compares to regular MAE, which gives a better view on SPAR's reconstruction objective.

### Weaknesses
## [W1] Methodology
line 319: "SPAR can be interpreted as performing contrastive learning over two types of enriched positive pairs: ... [(mask(X;M),S,R), mask(X;\bar{M}),S,R)] (9)" -> I don't understand why we want contrastive learning between vectors with the same S and R elements. I understand contrastive learning between mask(X;M) and mask(X;\bar{M}), since those are different, but if S and R are per-definition the same for a pair, then why would it help to add these? Same comment goes for Equation (10).

## [W2] Experiments
While the experiments show excellent performance for SPAR, it is not completely clear yet that this is because of just the pre-training with the spatial and structural information. During evaluation, does SPAR also use these additional information sources by embedding and added these to the tokens? And, do the baselines also have access to this same information during evaluation? If yes, then the good performance might be due to the SPAR model using more information during evaluation, rather than having better embeddings per-se. I wonder if other pretrainnig approaches could not reach similar strong results if the spatial and structural embeddings are given to the final model during the fine-tuning stage. In any case, the paper should be clear on what information is accessible at what stages, for each of the methods.

## [W3] Other clarity and structure issues:
* Experiments to various key claims are missing from the main paper; only baseline comparisons on various tasks are included in the paper (though some results can be found in the Appendix). For example, one of the key claims is that SPAR improves generalization, but no generalization experiments are in the paper. Only in the appendix do we find for example generalization across unseen sensor placements (Table 7). Generalization across downstream tasks (line 025) is not shown. Ablation studies of the proposed design choices have unfortunteley  also been delegated to the Appendix, Table 8. The experiments in the main submission should support the main claims (one of the three baseline comparisons could instead move to the Appendix). As it is, the reader is forced to check the Appendix to validate the claims, effectively extending the paper's page limit.
* Contribution (3), validating the proposed method in experiments, isn't really a separate contribution from contribution ①, the method. Experiments are an expected requirement to show that a proposed method is effective, thus part of the first contribution.
* Line 052: "Our design is guided by a core principle: the duality between positions and signals. That is, spatial and structural configurations are not auxiliary metadata to the signals, but stand in an equal and mutually-determining relationship with signals" -> The word "duality" typically means that two concepts (here, "sensor positions" and "sensor signals") originate from the same underlying principle. Clearly, there is a correlation between position and the captured signal, but I'm not sure why this can be called a "duality". I understand SPAR's dual reconstruction objective models it as a duality related to the embedding space, but appears to me more of a modeling choice, rather than a property of reality.


## Minor:
* Line 294, Eq ⑥: please add citation for this equation, or are you claiming this bound analysis on MAE is a novel contribution of this work?
* The signal and spatial embeddings are "summed", I believe, but the phrasing is not so clear. Sometimes it is referred to as "adding" (l204, l233), "combining" (l241), but it wasn't clear if this meant concatenating or summing. Only in Eq(1) did I see it is a literal summation.
* Figures 4 and 5 are referenced out of order.

### Questions
* See my question on methodology in Weaknesses: why are positive pairs containing (partially) the same elements (Eq ⑨ and ⑩) useful for contrastive learning?

* See my question on experiments in Weaknesses: can you support that the better performance reported for SPAR is really due to the proposed pre-training strategy, and not because that model has access to more information on the evaluation spatial layout during inference that the other models do not use? For SPAR, do you for evaluation only use the signal embeddings (like the baselines), or also the positional and structural embeddings?

* line 249: "To enable cross-modal interactions, we then apply a joint transformer encoder over the concatenated latent embeddings from all modalities" -> combining all tokens from all sensors together seems like a large computational burden for the transformer architecture. Is this not a computational bottleneck?

* Experiments, line 373 "We pretrain on the full dataset and finetune only the prediction head (a single transformer layer) on scene 'H24,' which contains a single moving vehicle." Details are missing for the evaluation setup: what is the test set if the full dataset has been used for training? How similar or different is it to the training and fine-tuning data? Was the fine-tuning scene H24 also part of the pretraining? Same comment goes for the multi-vehicle setup in line 414.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces SPAR, self-supervised placement-aware representation learning, a pretraining approach for aggregating spatial information and sensor properties into distributed sensing. Following a masked autoencoder (MAE) framework, the signal, spatial information, and sensor properties are combined and given to an encoding module to learn the latent embeddings that optimize the reconstruction objectives. This approach encourages the model to retain signal, position information, and properties in a self-supervised manner. The experimental evaluation of SPAR is conducted with three multi-modal datasets including M3N-VC dataset, Ridgecrest Seismicity dataset, and RealWorld-HAR and SPAR consistently outperforms several state-of-the-art methods.

### Strengths
The paper extends the MAE framework by including spatial positional embeddings and structural positional embeddings from the sensor positions and characteristics. Compared to most of the existing approaches, SPAR improves its performance by considering spatial and structural information in the pretraining phase. The approach seems novel, and the results look promising.

### Weaknesses
Given that the baseline methods don’t take spatial information into account, the comparisons seem unfair. In the comparisons, the model size/complexity should also be included because a larger model usually outperforms a smaller model, and it is hard to know whether the improvements come from the inclusion of the spatial information. An ablation study will be useful and improve the clarity of the paper. In Table 8, when dropping certain components, how do they impact the model complexity?

### Questions
1.	What is the complexity of SPAR? How does it compare to the baselines?
2.	Is SPAR robust to sensor position errors?

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
This paper focuses on distributed sensing, where multiple spatially distributed sensors jointly observe an environment, a setting common to diverse applications such as vehicle monitoring, human activity recognition, and earthquake localization. The authors propose SPAR, a self-supervised pretraining framework that explicitly incorporates sensor placement into representation learning. Unlike prior methods that treat placement as auxiliary information, SPAR models both spatial location and structural role through positional embeddings, and employs dual reconstruction objectives to recover both sensor measurements and positions. The paper also provides theoretical justification for this design. Experimental evaluations on three multimodal datasets demonstrate that SPAR consistently improves downstream task performance compared to various baseline pretraining approaches.

### Strengths
- This paper addresses an interesting but underexplored task in representation learning for distributed sensing systems.

- The explicit incorporation of both spatial and structural sensor information through positional embeddings and dual reconstruction objectives is well motivated and reasonable.

- The experiments demonstrate consistent and significant improvements over diverse baselines across three real-world multimodal datasets.

### Weaknesses
- I find the theoretical analysis section somewhat forced. It largely reiterates the problem setup in mathematical form, but does not clearly convey new conclusions or insights. The statement “This encourages the embeddings to be context-aware and jointly informative of both signals and spatial layout, while avoiding memorizing redundant information” is vague and appears to restate an intuitive design goal rather than derive a meaningful theoretical result. Overall, it is not clear how the theoretical analysis directly supports or explains the empirical findings.

- If I understand correctly, the authors pretrain all baseline methods using the same dataset. While this provides a controlled comparison, it may inadvertently undermine some inherent advantages of the baselines, particularly those that do not require spatial location information and are designed to scale to much larger, unlabeled datasets. In practical scenarios, such methods could leverage significantly more data during pretraining and therefore may achieve stronger performance than reported here.
I agree that using the same data and number of epochs ensures fairness in the experimental setup. However, the results may not fully reflect real-world usage, where certain baselines are intended to benefit from abundant and diverse training data. As a consequence, restricting them to the limited dataset available in this work could disproportionately impair their performance. It would be helpful if the paper discussed this limitation or evaluated baselines under a more realistic large-scale pretraining setting.

- The experiments on unseen sensor placements are particularly interesting and reflect a practical deployment scenario, where sensor layouts may change without retraining from scratch. It would be valuable for the authors to provide more detailed studies and analysis on this aspect, as it highlights one of the key advantages of the proposed approach.

- It is unclear what specific insights or conclusions should be drawn from the qualitative visualizations in Figure 3. Additional explanation or interpretation would help clarify their purpose and relevance.

Minor things:
- Line 348: The phrase “on three multiple multi-modal, multi-node distributed sensing datasets” is redundant. The word “multiple” can be removed.

### Questions
Here’s a polished version of your questions with clearer wording and tone:

- What are the concrete conclusions of your theoretical analysis?
Please summarize the key takeaways and how they inform the design choices or predicted behaviors of SPAR.

- Can you provide a more practical comparison with baselines?
For methods that do not require spatial locations and can scale to larger unlabeled corpora, include results that reflect this advantage (e.g., larger pretraining sets or different data scales).

- Please expand the studies on unseen sensor placements.
Add deeper analysis (e.g., ablations, varying degrees of placement shift, robustness curves) to demonstrate performance without retraining when sensor layouts change.

- Clarify the qualitative results.
Explain what insights should be drawn from the visualizations (e.g., Figure 3), how they support the claims, and what specific phenomena are being illustrated.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents SPAR, a self-supervised framework for placement-aware representation learning in distributed sensing systems. SPAR integrates spatial and structural positional embeddings along with dual reconstruction objectives to jointly model the relationship between sensor positions and observed signals. A variant of the proposed method uses an LLM as a preprocessing step to improve generalizability.

### Strengths
+ This work leverages emerging LLM capabilities as a preprocessing step, combined with multi-objective learning, to address the sensor placement problem.
+ The method is evaluated on several datasets and includes comparisons against multiple prior approaches.
+  The paper also attempts to provide theoretical analysis on the performance bound.

### Weaknesses
-  The novelty of this work appears limited. The proposed structural position representation seems to be obtained via standard latent embedding computation, and in the SPAR-LLM variant, the LLM is primarily used as a preprocessing step. Additionally, dual or multi-objective reconstruction is a well-established technique in the literature.

- The method uses a neural network to compute an embedding from spatial positions, yet also manually normalizes those spatial inputs. It is unclear why such normalization is necessary given that a neural network can typically learn appropriate embeddings directly from raw inputs.

-  The concept and interpretation of structural position are not clearly defined. Although the paper provides examples such as "the body part a sensor is attached to" or "the orientation of a directional measurement device," it is unclear how the proposed embedding reliably captures this semantic information. The description suggests that structural position is simply a latent feature space derived from input data.

-  Related to the above point, the experimental section does not clearly specify which datasets involve sensors with meaningful structural attributes. The paper does not analyze how structural position affects performance in scenarios where such information should be relevant.

-  Proposition 3.1 is difficult to follow due to the introduction of many new variables. Additionally, Lines 307-309 following the proposition claim that SPAR promotes embeddings that capture signal information beyond what is explained by structural and spatial cues, and vice versa. However, it is unclear how this claim is supported by Proposition 3.1, or which components of the equation correspond to this reasoning. The connection between the theoretical analysis and the method remains weak.

- The process for obtaining baseline results is not clearly described. Were the results taken directly from the original papers, or were the methods reimplemented by the authors? Some baselines (e.g., CMC’20 and FOCAL’23) show extremely poor performance (e.g., 0.16 and 0.08 vs. 41.57 for the proposed method in Table 2). It would be helpful for the authors to explain why these baselines perform so poorly in this evaluation.

-  The LLM component is evaluated only on the HAR dataset and yields minimal improvement, which does not convincingly demonstrate the benefit of incorporating an LLM.

-  The paper does not report runtime or computational complexity of the proposed method (with and without the LLM component).

### Questions
Please see the Weaknesses section.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper is set in the context of distributed sensing, with a sender and receiver. I should start with the disclaimer that I am not entirely familiar with the application domain. However, I find the ideas put forth illuminating. 

The main premise in the paper is that normally, these applications only account for spatial information (say, in the form of positional embeddings of the sensor). But the authors claim that the structural signature also matters. They go on to show this through a proof which boils down to breaking down the problem in terms of $S$ and $R$. In essence, they device a scheme that is aware of both the spatial and structural embeddings boiling down to the maximization of conditional mutual information, with both factors included.

Formally, they setup a dual objective consisting of:
1. Spatial and structural positional embeddings 
2. Reconstruct both signal and spatial positions. 

Evaluations are convincing and show that this placement aware formulation works for all the cases considered - vehicle classification, earthquake localization and HAR.

### Strengths
+ The formulation isolates a presumably new factor hitherto not considered - the 'structural' information. 
+ Evaluations are very convincing. Results bear out in all the cases considered.

### Weaknesses
- I understand the proofs and the mathematical language. However, I am totally at a loss as to what these 'structural' factors are. Judging from the paper's language (e.g. "they do not fully represent structural placement
conditions, such as the body part a sensor is attached to, or the orientation used for a directional measurement device (e.g., front-facing versus rear-facing camera on an autonomous car") it looks like a graph connecting relational aspects. 
- The above point makes the paper somewhat opaque - especially to outsiders such as myself.

### Questions
Please explain what is meant by these structural factors in more detail. I think it warrants a bit of a rewrite.

### Soundness
3

### Presentation
2

### Contribution
3
