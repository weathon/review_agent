# Predicting Training Re-evaluation Curves Enables Effective Data Curriculums for LLMs

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 8

## Abstract
Data curriculums have become central to successful LLM training, yet principles governing optimal data placement remain unclear. We introduce the *training re-evaluation curve (TREC)*, a diagnostic that retrospectively evaluates training batches *using the final model weights*. The TREC characterizes how well a trained model retains training data as a function of *when* the data was encountered during training. Analyzing TRECs for models from 111M to 3.9B parameters, we show that placing high-quality data at low points on the TREC significantly improves performance. Importantly, while a TREC is initially observable only after training, we demonstrate it can be *predicted in advance* from AdamW’s implicit EMA coefficients, enabling proactive curriculum design. By predicting TRECs for published training recipes, we explain prior ablations and reveal suboptimal data placements. We also align high-quality data with TREC minima in order to improve continual pre-training of a 3.9B-parameter LLM trained on 900B tokens.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces the Training Re-evaluation Curve (TREC), a novel diagnostic tool for designing effective data curriculums for Large Language Models (LLMs). The TREC measures the retrospective loss of the final trained model on individual training batches, plotted as a function of when those batches were seen during training. The core hypothesis is that placing high-quality (HQ) data at the minimum of this curve—the "TREC valley"—maximizes its positive impact on the model's final performance.

The central contribution is a framework for predicting the TREC in advance, obviating the need for a costly initial training run. This prediction is grounded in the dynamics of the AdamW optimizer, modeling its parameters as an Exponential Moving Average (EMA) of weight updates. The authors extend this by incorporating a "training-fraction" term to phenomenologically account for "minimizer drift," where the loss landscape shifts during training, reducing the effectiveness of earlier updates. The paper provides extensive empirical validation for models ranging from 111M to 3.9B parameters, showing that the predictive model accurately captures TREC shapes and that TREC-guided data placement outperforms baseline strategies. Finally, the framework is applied to explain and critique existing training recipes and to improve performance in a large-scale continual pre-training setting.

### Strengths
1.  The TREC itself is a simple yet useful concept, and the ability to predict it proactively is a contribution.

2.  The claims are supported by a large-scale study of over many TRECs across a wide range of model sizes and training configurations. 

3.  The framework successfully explains previously ambiguous empirical results from the community. For instance, it provides a clear rationale for why the late-stage annealing in Llama-3 405B failed to yield gains (the placement occurred on a rising part of the TREC) and why the mid-training interventions in models like OLMo-2 were effective.

### Weaknesses
1.  A significant scope limitation, explicitly noted by the authors (Appendix D.2, Figure 14), is that the framework can only identify the optimal data placement within a given learning rate schedule. 

2.  How to explain the schedule with the highest learning rate (η=0.015) produced the deepest TREC valley, which according to the framework's core premise, should have yielded the best results. Instead, it produced the worst validation performance. This directly contradicts the assumption that minimizing re-evaluation loss on training data (what TREC measures) always translates to improved generalization.

### Questions
see weakness

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
This paper introduces the Training Re-Evaluation Curve (TREC) — a diagnostic that evaluates how well a trained model retains training batches as a function of when they appeared during training. TREC measures retrospective loss on training data using the final model weights and reveals that models tend to retain information from certain phases of training more effectively.
The authors show that placing high-quality data at the low points of the TREC (where retrospective loss is smallest) improves performance, contradicting the common practice of placing such data at the end of training.
They further demonstrate that TRECs can be predicted in advance from AdamW’s exponential moving average (EMA) timescale, enabling proactive curriculum design without retraining.
The paper supports its claims with both theoretical derivations and large-scale empirical results (from 111M to 3.9B parameters). It applies TREC analysis to settings such as sparse mixture-of-experts (MoE) models and continual pre-training (CPT), showing predictive consistency and practical gains.

### Strengths
- The paper provides a great amount of experiments on models from 111M to 3.9B with several hundred billions of tokens.
- The theoretical framing is well-grounded, extending prior work on AdamW dynamics and EMA analysis And the predictive model of TRECs is supported by strong correlations.

### Weaknesses
Since the experimental setup's setting defines HQ(high quality data) as a data with close distribution to the evaluation target ( e.g. code data vs general data and evaluating on code data, not general HQ data). Intuitively, place these data near the end of training will improve the performance as they are less likely to be forgotten by LLM, which is similar to annealing, and the results of the paper also show that place the data at the end of learning yields better results in linear decay cases.
Also for step scheduler, place the data at the end of learning (the last part before the lr step decay, as the remaining part may not change much on the parameters due to very low lr) yields better results. And TREC can be interpreted as a reflect how model remember/forget the data.

### Questions
1. The setting for each experiments may need further clarification or better organized. The pretraining loss has a sudden drop at 70% (i.e. figure 1 most left sub-figure), does that correspond to the experiment with Jump-to-1% learning rate scheduler?
2. Is there a specific kind of data (i.e. domains or other patterns) tend to have lower TREC? Or it is just related to the order across the whole training?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces the Training Re-evaluation Curve (TREC), a novel diagnostic tool for understanding data retention in Large Language Model (LLM) training. The TREC is computed retrospectively by evaluating the final trained model on the sequence of training batches it encountered. The authors hypothesize that the optimal placement for high-quality (HQ) data in a curriculum is at the minimum of this curve (the "TREC valley"), a point that often precedes the end of training.

The central challenge is that TREC is a retrospective measure. The paper's main contribution is demonstrating that the TREC's shape is predictable in advance. The authors show that TREC shape is governed by the AdamW optimizer's implicit EMA timescale and a "training-fraction" component. They develop a predictive model for TREC and show that its parameters can be fitted on small models (e.g., 111M) and generalized to larger ones (up to 3.3B).

The authors validate this framework by showing that:

1. Directly placing HQ data at the TREC minimum in 610M model runs yields the best validation performance.
    
2. The framework successfully explains findings in prior work, such as the (ineffective) late-stage annealing in Llama 3 405B and the (effective) data phase timing in OLMo-2.
    
3. It provides a compelling explanation for memorization in Mixture-of-Experts (MoE) models, linking it to a reduced "effective TPP" for each expert.
    
4. TREC-guided placement improves performance in a 3.9B model continual pre-training (CPT) task.

### Strengths
1. **Originality and Significance:** The paper's primary strength is its originality and practical significance. It defines a novel diagnostic (TREC) and, crucially, provides a complete, principled framework to *predict* it in advance. This transforms curriculum design from a heuristic, high-cost ablation study into a predictable, low-cost design choice. This is a significant step forward for the field.
    
2. **High-Quality, Thorough Experimentation:** The empirical work is extensive and of high quality. The authors support their claims with a large study of models (111M to 3.9B) , diverse experimental settings (pre-training, CPT, MoE) , and a large-scale analysis (over 600 TRECs). The "data placement sweep" (Fig. 2) is a direct, clear, and compelling validation of the core hypothesis.
    
3. **Presentation:** As noted above, the paper is exceptionally clear. The logical flow (Here's a tool -> Here's the problem -> Here's the solution -> Here's why it works -> Here's how to use it) is perfectly executed. The figures are exemplary.
    
4. **Actionable Insights:** The framework is not just theoretical; it provides immediate, actionable insights. The explanations for MoE memorization (linked to effective TPP) and the failure of Llama 3's late-stage annealing (linked to vanishing EMA coefficients) are powerful demonstrations of the framework's explanatory power. This gives confidence that the model is capturing a true, underlying dynamic of the training process.

### Weaknesses
1. **Limited Scale of Core Validation:** The primary weakness is that the main *predictive loop* (fit on small model, predict for large model, train large model based on prediction) is not fully closed at the largest scales.
    
    - The core *validation* of the hypothesis (the data placement sweep in Fig. 2) was done on a 610M model.
        
    - The *prediction* framework ($m^\*$) was fit on 111M and *evaluated* on runs up to 3.3B.
        
    - The 3.9B CPT experiment is the closest to a full-loop validation. However, it's a CPT setting, not pre-training from scratch. Furthermore, it's not explicitly stated whether the "Halfway" point was *proactively chosen* by the 111M-derived predictor, or if it was just a reasonable choice *inspired* by the general shape of TRECs.
        
    - To be fully convincing, a stronger validation would be to use the 111M-derived $m^\*$ predictor to find the optimal placement for a 7B+ *pre-training* run and show it outperforms the standard heuristic.
        
2. **Cross-Schedule Generalization Limitation:** The paper is commendably honest about this limitation, but it is a significant one. The framework can reliably find the best data placement *within* a given LR schedule. However, it *cannot* be used to compare *across* schedules.

### Questions
1. **On the 3.9B CPT Experiment:** In the 3.9B CPT experiment (Sec. 5.3), was the "Halfway" placement point proactively calculated using the power-law predictor (Eq. 4) that was fit on the 111M model? Or was this point chosen based on a retrospective analysis of a 3.9B run, or simply as an intuitive "non-end" baseline? Clarifying this is critical for understanding the full practical utility of the predictive framework at scale.
    
2. **On Cross-Schedule Comparison:** The finding that absolute TREC loss does not predict performance *across* LR schedules (Fig. 14, Key takeaway 5) is an important limitation. Do the authors have any intuition on what additional signal might be required to make such a cross-schedule prediction? For example, does the absolute value of the *training loss* (the orange curve) at the TREC valley, or the *width* of the valley, play a role in this disconnect?
    
3. **On Predictor Robustness:** The $m^\*$ power-law was fit using a Linear decay schedule, and the paper notes a Cosine-specific fit is more accurate. How sensitive is the framework to the LR schedule shape? If a practitioner wishes to use a more complex schedule (e.g., WSD, or a custom multi-stage schedule), must they re-fit the entire power-law (which requires many small-scale runs)? Or is the provided Linear-fit predictor "good enough" for most practical purposes to identify the approximate valley?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces the Training Re-Evaluation Curve (TREC), a diagnostic tool that measures a trained LLM's retention of training batches as a function of their position, revealing optimal placements for high-quality data that outperform end-of-training heuristics. By modeling TRECs analytically via AdamW's EMA coefficients adjusted for training fraction drift, the authors enable proactive curriculum design, with predictions generalizing across scales from 111M to 3.9B parameters and 20-1280 TPP. Extensive experiments explain suboptimal strategies in prior works like Llama-3 and OLMo-2, while demonstrating TREC-guided placement improves continual pre-training in validation loss.

Overall, I really like this paper. Meanwhile, as a practicer in this field, I have many questions and want to discuss with authors. I am looking forward to replies from the authors.

### Strengths
1. The paper addresses a critical challenge in LLM pre-training from an innovative and unanticipated perspective, demonstrating strong novelty that sets it apart in the field.
2. The authors conduct a comprehensive suite of rigorous experiments, involving substantial computational resources, which will provide valuable insights to researchers in the LLM pre-training community.
3. I particularly appreciate the paper's clear and logical structure: it begins with well-defined hypotheses, builds through theoretical derivations, validates them empirically, and systematically distills a series of key takeaways.

### Weaknesses
1. **Difficulty Understanding Loss Variations in TREC**: What I find challenging to comprehend is the section defining TREC. For a given data distribution, different training batches should be nearly identically distributed (i.i.d.). Why, then, does the fully trained model exhibit such substantial loss differences across these batches—particularly in Figure 1 (left) and Figure 2 (left)? In my view, the losses should approximate a flat line parallel to the x-axis, with values closely aligning with the final train loss. This discrepancy warrants further clarification.

2. **Scope of TREC's Applicability**: It appears that this principle applies only to validation sets identical to the training distribution, since TREC is computed on training batches. Is my interpretation correct? If so, does this imply that the optimization ultimately targets only validation sets matching the training distribution, rather than arbitrary domain-specific tasks (e.g., code)? Your statement on line 097—"we hypothesize the TREC remains predictive even when data distributions vary"—seems to address this concern. However, this hypothesis comes with no more explanation. Is this hypothesis overly ambitious, given the potential risks of distribution shift?

3. **Inconsistent and Unclear Naming of LR Schedules**: I feel that your selection and nomenclature for LR schedules lack rigor and clarity, manifesting in several ways:  
   a. Terms like "10x" and "step decay" are non-standard for LR schedules, and I have rarely encountered similar naming conventions in other literature. While I eventually deciphered "10x" via the appendix, this remains confusing and detracts from accessibility.  

   b. "Step-decay learning rate schedule" is also uncommon; I am still unclear on its precise form. Does it entail a sudden drop in LR (e.g., at 70% of training) to 10% of its prior value? You reference it "as used in DeepSeek LLM," but DeepSeek employs a "multi-step cosine scheduler," which introduces ambiguity.

   c. I strongly recommend adopting more precise terminology for your experimental LR schedules, ideally with explicit mathematical functions describing LR evolution at each step.

4. **Summary of Predicted TREC Positions Across Schedules**: Allow me to summarize the optimal high-quality data placement positions predicted by your TREC across different LR schedules:  
   a. Linear decay (commonly used): End of training (Figure 2 right).  
   b. Decay to zero (commonly used): End of training (Figure 11 left).  
   c. Decay to 0.1x LR (your "10x," commonly used): Not explicitly described, but I infer it aligns with the end, given its similarity to D2Z.  
   d. Step drop (uncommonly used): Pre-decay position (Figure 2 left), with the final val loss showing minimal gap compared to end placement.  
   **Thus, can I conclude that, for the vast majority of common LR schedules, TREC predicts the end of training as the optimal placement?** This suggests some degree of over-claiming on your part, while also raising concerns about TREC's practical utility in experiments. I acknowledge that TREC offers mechanistic insights, but its impact on guiding real-world practice seems limited.

5. **Alignment in Figure 9 and Practical Implications**: Following the previous point, in Figure 9, your TREC predictions align with prior findings, yet these still involve placing high-quality data in the final stages. Notably, in Figure 9 left, your prediction actually points to roughly the 0.5–0.9 fraction for inserting that 40% high-quality data, rather than purely the end (0.6–1.0 fraction). I am skeptical: under a common LR schedule, would shifting high-quality data away from the very end—per TREC guidance—truly yield superior performance **in practice**? This merits empirical validation beyond the current analysis.

### Questions
All the questions are recorded in weaknesses sections.

I greatly enjoyed this paper. If the authors can thoughtfully address and resolve my concerns, I would be delighted to raise my score to a 10. Conversely, given the fundamental nature of my questions—particularly the first one—if they remain inadequately resolved, I may lower my score to 6 or even below. I look forward to further discussion with the authors during the rebuttal phase.

### Soundness
3

### Presentation
3

### Contribution
2
