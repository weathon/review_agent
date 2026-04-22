# Transformers with Endogenous In-Context Learning: Bias Characterization and Mitigation

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 2

## Abstract
In-context learning (ICL) enables pre-trained transformers (TFs) to perform few-shot learning across diverse tasks, fostering growing research into its underlying mechanisms. However, existing studies typically assume a causally-sufficient regime, overlooking spurious correlations and prediction bias introduced by hidden confounders (HCs). As HC commonly exists in real-world cases, current ICL understandings may not align with actual data structures. To fill this gap, we contribute the pioneer theoretical analysis towards a novel problem setup termed as ICL-HC, which offers understanding the effect of HC on the pre-training of TFs and the following ICL prediction. Our theoretical results entail that pre-trained TFs exhibits certain prediction bias with proportional to the confounding strength. To migrate such prediction bias, we further propose a gradient-free debiasing method named Double-Debiasing (DDbias) by collecting and prompting with extremely few unconfounded examples, correcting pre-trained TFs with unbiased ICL predictions. Extensive experiments on regression tasks across diverse designs of the TF architectures and data generation protocols verify both our theoretical results and the effectiveness of the proposed DDbias method.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a novel problem setting called Endogenous In-Context Learning (EICL), where hidden confounders (HCs) violate the causal sufficiency assumption commonly made in prior ICL studies, leading to dependencies between input features and noise terms. The authors provide the first theoretical analysis of how HCs affect transformer pre-training dynamics and subsequent ICL predictions, showing that pre-trained transformers exhibit prediction bias proportional to the confounding strength. To mitigate this bias without gradient updates or model fine-tuning, they propose Double-Debiasing (DDbias), a gradient-free method that prompts the biased transformer twice using a small number of unconfounded examples to yield unbiased predictions. Experiments on synthetic linear regression tasks with varying transformer architectures and confounding strengths validate the theoretical claims and the efficacy of DDbias.

### Strengths
1. Addresses an important and underexplored gap in ICL theory by incorporating hidden confounders, which are prevalent in real-world data (e.g., as illustrated in the Many-aspect Online Review example in Appendix B). This makes the work highly relevant for aligning ICL understandings with practical data structures.
2. Empirical validation is thorough within its scope, using metrics like L2 divergence, cosine similarity, and prediction bias across confounded datasets, with results aligning well with theorems (Figs. 3-5). The method's efficiency (requiring only a few unconfounded samples) is demonstrated effectively.

### Weaknesses
1. The analysis is restricted to linear transformers and linear regression tasks under Gaussian assumptions, limiting generalizability to nonlinear architectures or more complex real-world ICL scenarios (e.g., NLP or vision tasks). While this simplifies proofs, it may not capture the full dynamics of modern transformers.
2. Relies on access to a small set of perfectly unconfounded examples for DDbias, which could be challenging or costly to obtain/verify in practice when confounders are unknown. The paper lacks analysis of robustness to partially confounded "unbiased" data.
3. Experiments are purely synthetic, with no evaluation on real datasets exhibiting natural hidden confounders. This reduces persuasiveness regarding practical impact, as synthetic setups may not reflect the nuances of actual data generation processes.

### Questions
Please address the questions mentioned in the weaknesses. Especially, how can practitioners identify and collect "unconfounded" examples in practice without knowing the true data-generating process? What happens if these examples are only partially unconfounded?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper tackles an important and previously overlooked aspect of In-Context Learning (ICL) in Transformers (TFs): the impact of endogeneity arising from hidden confounders (HCs) present during pre-training. The authors introduce the "Endogenous ICL (EICL)" problem setup and provide a theoretical analysis characterizing how confounding bias propagates from pre-training data to the final ICL predictions, specifically within the context of linear TFs and linear regression. They find that the resulting bias is proportional to the confounding strength. Building on this analysis, they propose "Double-Debiasing (DDbias)," a novel, gradient-free prompting method that utilizes a small set of unconfounded examples to correct the predictions of a biased, pre-trained TF. The theoretical claims and the effectiveness of DDbias are supported by experiments on synthetic linear regression tasks.

The paper makes a valuable contribution by formally introducing and analyzing endogeneity within the theoretical ICL literature, moving beyond the common assumption of causal sufficiency. The theoretical characterization of bias is insightful, and the proposed DDbias method offers a potentially practical, low-cost approach to mitigation.

However, the study's reliance on strong simplifying assumptions (linear TFs, linear data generation) limits the immediate generalizability of the theoretical findings to complex, real-world TFs and tasks. Furthermore, the practical feasibility of the proposed debiasing method hinges on the availability and identification of truly unconfounded examples, which is often a significant challenge. While the empirical results validate the theory in the studied setting, broader empirical validation on more realistic tasks and datasets would strengthen the paper's impact.

### Strengths
1.  **Novel and Important Problem Formulation:** The formalization of Endogenous ICL (EICL) addresses a critical gap in existing ICL theory. Recognizing and analyzing the impact of hidden confounders, which are common in real-world data, is crucial for a more realistic understanding of ICL mechanisms.
2.  **Pioneering Theoretical Analysis:** The paper provides the first (to the authors' knowledge) theoretical characterization of how endogeneity affects *both* the pre-training dynamics of TFs and the subsequent ICL inference bias. Linking the prediction bias to confounding strength ($\mathbf{r}_j$) provides concrete theoretical insights.
3.  **Simple and Practical Debiasing Method:** DDbias is an elegant, gradient-free prompting strategy that avoids costly fine-tuning or complex architectural changes. Its reliance on only a few unconfounded samples makes it potentially much more practical than methods requiring large-scale unbiased data collection.
4.  **Clear Theoretical Framework:** The theoretical sketch (Figure 2) effectively outlines the paper's analytical steps: establishing grounding "unbiased" parameters (U-weights), showing pre-training deviation from these due to HCs, and linking this deviation to ICL prediction bias.
5.  **Targeted Empirical Validation:** The experiments are well-designed to specifically verify the core theoretical claims regarding bias proportionality (Theorems 1 & 2) and the effectiveness of DDbias (Theorem 3 & Proposition 1) within the linear regression setting.

### Weaknesses
1.  **Strong Linearity Assumptions:** The entire theoretical analysis is predicated on linear TFs (specifically, linear self-attention without softmax) and linear data generation ($y = w_*^\top x + \epsilon$). While this is a common starting point for ICL theory, the extent to which these findings generalize to deep, non-linear TFs (like GPT models where ICL is prominent) and non-linear tasks remains an open question. The reliance on simplified TF parameters ($\bar{S}, \bar{T}_{:,j}$) further distances the theory from practical architectures.
2.  **Requirement of Unconfounded Data for DDbias:** The proposed solution, DDbias, crucially depends on having access to a set of unconfounded examples ($\mathcal{D}^u$). The paper suggests "extremely few" are needed, but doesn't quantify this rigorously or discuss the practical challenges of *obtaining* or *verifying* such unconfounded data in real-world scenarios where HCs are pervasive precisely because they are *hidden*. How sensitive is DDbias to imperfections or residual confounding in $\mathcal{D}^u$?
3.  **Limited Scope of Experiments:** The experiments exclusively use synthetic data generated from a linear model. While this allows for controlled verification of the theory, it doesn't demonstrate the method's applicability to the complex, high-dimensional, and often non-linear domains where large TFs and ICL are typically applied (e.g., natural language processing, vision). Validation on semi-synthetic or real-world benchmarks exhibiting known confounding (if available) would significantly bolster the claims.
4.  **Comparison with Alternatives:** The paper contrasts DDbias with IV-based approaches mentioned in a concurrent work by highlighting that DDbias doesn't need IVs. However, it requires unconfounded data, which IV methods aim to circumvent. A more detailed discussion comparing the assumptions, requirements, and potential domains of applicability for DDbias versus IV regression or other potential debiasing strategies (e.g., causal representation learning adapted to ICL) would be beneficial.

### Questions
1.  Could you elaborate on the potential challenges or theoretical hurdles in extending the bias characterization and the DDbias methodology to non-linear TFs (e.g., with softmax attention, MLPs) and non-linear data generation processes?
2.  Regarding the unconfounded data requirement for DDbias: How many unconfounded samples ($n_u$) were typically needed in your experiments to achieve effective debiasing (Figure 4 suggests ratios relative to pre-training data size)? How robust is DDbias if the provided "unconfounded" samples ($\mathcal{D}^u$) still contain some small residual confounding?
3.  Given the focus on characterizing bias, have you considered applying DDbias to scenarios beyond linear regression, perhaps using semi-synthetic NLP or vision tasks where confounders can be injected, to explore its empirical effectiveness more broadly?
4.  Could you compare the practical data requirements for DDbias (small $n_u$ unconfounded samples) versus an IV-based approach (potentially larger $n$ confounded samples but including valid IVs)? In what scenarios might one approach be more feasible than the other?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper addresses the common neglect of hidden confounders in current In-Context Learning (ICL) theory during both pre-training and inference, and proposes a new framework called Endogenous ICL (EICL). The authors theoretically prove that Transformers trained with confounded data produce prediction bias proportional to the confounding strength. They construct parameter settings (U_weights) that can yield unbiased predictions even when test prompts are confounded, and introduce a Double-Debiasing (DDbias) method that removes bias without gradient updates, requiring only a small number of unconfounded samples. Extensive synthetic experiments validate the theoretical results and demonstrate the robustness of the proposed method across different model depths, confounding strengths, and data distributions (including non-Gaussian cases), providing a new theoretical and methodological foundation for achieving unbiased predictions in ICL under realistic scenarios involving confounders.

### Strengths
1. First to systematically investigate endogenous bias in ICL from hidden confounders, introducing the EICL framework with clear practical motivation.
2. Strong theoretical contributions proving bias is proportional to confounding strength, alongside a lightweight, gradient-free debiasing method (DDbias) that uses minimal unbiased data.
3. Extensive synthetic experiments across model depths, confounding levels, and distributions (including non-Gaussian) that convincingly support the theoretical claims.

### Weaknesses
1. The scope is restricted to linear attention Transformers and linear regression setups, with no extension to nonlinear architectures or tasks.
2. Experimental validation relies entirely on simulated data, lacking tests using real-world datasets with genuine hidden confounders.
3. The proposed DDbias method depends on the availability of a small set of completely unconfounded samples, which may be unrealistic in many real scenarios.

### Questions
See the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces Endogenous In-Context Learning, a new theoretical framework revealing that pre-trained Transformers can produce biased in-context predictions when trained on data containing hidden confounders. While prior ICL analyses assumed that there is no confounding, the authors show that endogeneity induces systematic prediction bias proportional to the confounding strength. They provide formal proofs that Transformer parameters learned on confounded data deviate from unbiased reference weights and this deviation propagates into biased ICL predictions. To address this, they propose Double-Debiasing (DDbias), a simple gradient-free prompting method that corrects such bias using only a few unconfounded examples by prompting the model twice. Several simulations on linear regression settings verify the theoretical results and demonstrate that DDbias effectively mitigates bias without retraining.

### Strengths
* This paper shows a novel theoretical problem formulation. While there is a similar work, this paper introduces the Endogenous In-context learning setting, addressing hidden confounders in ICL, which is previously under-explored source of bias from other existing theoretical works regarding ICL. 
* This paper provides well-grounded theoretical results providing clear formal analyses which characterizes how hidden confounders affect both the pre-training and inference results of Transformers. This shows that prediction bias scales proportionally to confounding strength. 
* Simulation results comprehensively verify the theoretical claims, demonstrating consistent trends between confounding strength, parameter deviation, and prediction bias reduction after applying their method.

### Weaknesses
* The theoretical analysis is limited to linear self-attention Transformers and linear regression tasks. It remains unclear whether the same bias characterization---particularly the proportionality to the confounding strength---extends to deeper, nonlinear, or multimodal Transformer architectures used in real-world ICL. Even if this might be very hard to be theoretically shown, I believe that providing experimental analyses about this scalability to larger models would make this paper much stronger.

* The proposed method requires access to a small set of unconfounded examples. In realistic settings, how can such unbiased samples be obtained or verified? Are there practical diagnostics or robustness analyses for partially confounded data? And can we utilize such approaches to apply the proposed method?

* Section 2.3 claims that a previous study (Liang et al) for confounding issue in ICL did not consider the gap between the traditional regression problem and ICL. However Theorem 2 later concludes that the ICL prediction bias is proportional to $r$ --- exactly the same form as in classical OLS theory. What specific term or mechanism in ICL makes this result fundamentally new rather than a re-derivation in a different setting? And are there concrete theoretical differences in constants, variance scaling or convergence behavior between EICL and standard OLS?

* All experiments are synthetic and rely on controllable confounding  strengths r. It would be valuable to evaluate the framework on real-pre-trained models or naturally confounded datasets (e.g., sentiment review) to demonstrate practical relevance.

### Questions
* Comparison with existing deconfounding methods. If it is possible to identify a small set of debiased dataset, does the proposed method outperform other baseline debiasing methods? I am wondering if the authors can provide experimental results for performance comparison.

### Soundness
3

### Presentation
3

### Contribution
2
