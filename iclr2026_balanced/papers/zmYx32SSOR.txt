# ADVANCING EQUITABLE AI: A COMPREHENSIVE FRAMEWORK FOR INDIVIDUAL FAIRNESS ASSESSMENT


**Anonymous authors**
Paper under double-blind review


ABSTRACT


Ensuring fairness in machine learning (ML) models is essential for developing
equitable and trustworthy AI systems. There has been extensive existing research
on group-based fairness metrics such as the Statistical Parity Difference and Disparate Impact, but these group-based fairness metrics often fail to address fairness at the individual level. An ML model can achieve perfect group fairness,
but produce discriminatory outcomes at the individual level or vice versa. In this
paper, four novel individual-based fairness metrics are proposed: Proxy Dependency Score, Stability Rate, Attributional Independence Score, and Intra-Cohort
Decision Consistency. These metrics are designed to evaluate different aspects of
individual fairness, including the influence of protected attributes on model predictions, the robustness of the model to protected attribute perturbations, the independence of attributions from protected attributes and consistency within similar individuals. These four new individual-based metrics are empirically compared with group outcome-based fairness metrics on ML models trained on Adult
and COMPAS datasets. The empirical results reveal that models deemed unfair
by group metrics may exhibit individual-level fairness. Our work highlights the
critical need for comprehensive individual fairness assessments in real-world applications. Our proposed framework can act as a complement to group-based
evaluations towards a more complete understanding of Artificial Intelligence (AI)
fairness and the development of more equitable AI systems.


1 INTRODUCTION


Machine learning systems play a more prominent role in critical decision-making scenarios nowadays, from credit approval and criminal risk assessment to personalized content delivery. Although
ML systems possess powerful predictive capabilities, there are widespread societal concerns about
fairness and accountability in such decision-making systems Lious (2022). Particularly, individual
fairness, which insists that similar individuals should receive similar outcomes, is used more often
to ensure equitable AI systems Filippi et al. (2023); Ghadage et al. (2023).


However, the pursuit of fairness in ML often faces the trade-off between fairness and accuracy.
Enhancing fairness may worsen predictive performance, and optimizing accuracy may amplify biases rooted in training data Arhin & Treku (2024); Plecko & Bareinboim (2025). These competing
objectives pose challenges for model developers and fairness researchers. However, current fairness interventions focus mainly on achieving group fairness, such as demographic parity and equal
opportunity, which can still lead to unfair treatment at the individual level Arhin & Treku (2024).


To address the identified gap, we investigate the evaluation tools of individual fairness. There are
existing definitions such as fairness through awareness Dwork et al. (2012) and counterfactual fairness Kusner et al. (2017) which offer theoretical foundations. However, there is a lack of practical
and fine-grained evaluation tools to capture different dimensions of individual fairness John & Saha
(2020); Mukherjee et al. (2020); Zhang et al. (2023). It is hard for model developers to identify and
quantify AI system fairness in real-world applications without comprehensive diagnostic metrics.


In this paper, we propose four evaluation metrics designed to measure individual fairness from
complementary perspectives:


1


- Proxy Dependency Score (PDS): Measures the influence of protected attributes transmitted
through proxy variables, which represents indirect discriminatory pathways.


    - Counterfactual Stability Rate (CSR): Assesses how sensitive predictions are in response
to hypothetical changes in protected attributes, capturing any counterfactual fairness violations.


    - Attribution Independence Score (AIS): Evaluates how much features are entangled with
protected attributes, indicating biased decision rationales.


    - Intra-Cohort Decision Consistency (IDC): Quantifies the consistency of decisions across
near-identical individuals in terms of non-protected features.


Together, these four metrics form a comprehensive diagnostic tool suite to enable multi-faceted
evaluation of individual fairness. These metrics provide insights beyond binary outcome disparity
to uncover subtle and structural sources of bias.


We perform extensive empirical analysis on standard fairness benchmarks, including the Adult
Becker & Kohavi (1996) and COMPAS ProPublica (2016) datasets. Our experiments show that
these metrics yield different fairness conclusions from existing evaluation tools. In particular, we
observe that individual-based fairness scores may indicate fairness even when group-based fairness
metrics suggest otherwise. By contrasting models under different training regimes and fairnessaware interventions, we demonstrate the value of our metrics in revealing the trade-offs and biases
in fairness-aware ML systems.


The main contributions of this paper are as follows.


    - We introduce four novel evaluation metrics - PDS, CSR, AIS, and IDC - that offer a comprehensive and interpretable framework for quantifying individual fairness violations.


    - We empirically validate these metrics across widely used datasets and demonstrate how
they could give different results from existing evaluation standards.


    - We release a codebase and evaluation toolkit to support reproducible research and integration of our metrics into fairness-aware machine learning workflows.


Our work aims to enrich the evaluation toolbox available to practitioners and researchers, and to
advance the field toward trustworthy and accountable machine learning systems at the individual
level.


2 RELATED WORK


Fairness metrics can be categorized into group fairness and individual fairness. Group fairness aims
to ensure equitable treatment across subpopulations defined by protected attributes such as race,
gender, and age Dwork et al. (2012); Chouldechova (2017); Verma & Rubin (2018). Classical group
fairness metrics include statistical parity difference and disparate impact, which require similar positive prediction rates across subgroups. There are other, more nuanced group fairness metrics, such
as equalized odds Hardt et al. (2016), which requires equal true and false positive rates across subgroups, and equal opportunity, which focuses on the true positive rate. Moreover, predictive parity
requires comparable positive predictive values across subgroups Chouldechova (2017); Verma &
Rubin (2018); MacCarthy (2018).


In contrast, individual fairness measures whether similar individuals receive similar outcomes Lahoti
et al. (2019). Individual fairness metrics avoid coarse group-level averaging and emphasize consistency at the individual level. Causal discrimination Galhotra et al. (2017); Xie & Wu (2020) defines
unfairness as an outcome disparity between individuals who differ only on protected attributes. Fairness through awareness Dwork et al. (2012); Li et al. (2023) formalizes this by bounding prediction
differences via a Lipschitz condition on input similarity. Accurate fairness Li et al. (2023) aligns
individual fairness with accuracy by uniformly bounding the accuracy and fairness difference for
similar sub-populations.


While group fairness provides a population-level insight, it can cause unfairness toward individuals
within subgroups. Individual fairness addresses this limitation by requiring individual-level simi

2


larity definitions. Recent work Xu & Strohmer (2024) has also shown that group and individual
fairness criteria can be fundamentally incompatible in some cases.


3 INDIVIDUAL FAIRNESS METRICS


Ensuring fairness in machine learning (ML) is essential for building equitable and trustworthy AI.
While group fairness metrics such as Statistical Parity and Disparate Impact have been widely studied Verma & Rubin (2018); Chouldechova (2017); Hardt et al. (2016), they can overlook unfair
treatment at the individual level Dwork et al. (2012). We propose four novel metrics for assessing individual fairness: **Proxy Dependency Score (PDS)**, **Counterfactual Stability Rate (CSR)**,
**Attribution** **Independence** **Score** **(AIS)**, and **Intra-Cohort** **Decision** **Consistency** **(IDC)** . These
capture complementary dimensions of proxy reliance, counterfactual robustness, attributional independence, and intra-cohort consistency Kusner et al. (2017); Li et al. (2023). Through experiments
on the Adult and COMPAS datasets Becker & Kohavi (1996); ProPublica (2016), we show that
models deemed unfair by group metrics may still satisfy individual fairness criteria, and vice versa,
underscoring known tensions between group- and individual-level notions of fairness Kleinberg et al.
(2016); Xu & Strohmer (2024). Our results complement recent efforts to operationalize individual
fairness Li et al. (2023) and provide an open-source toolkit to support reproducible evaluation and
integration of these metrics into fairness-aware ML workflows.


3.1 PROXY DEPENDENCY SCORE: UNCOVERING INDIRECT DISCRIMINATION


Proxy Dependency Score (PDS) measures the influence of protected attributes transmitted through
proxy variables, which shows the indirect discriminatory pathway and quantifies the extent to which
a model’s predictions rely on protected attributes. The advantage of PDS is that it can measure
indirect dependencies on protected attributes, even when they are not directly included in the training
data. The formula for PDS is defined as:

ProxyScore = 1 _−_ [Accuracy][(] _[M][ ′]_ [)] (1)

Accuracy( _M_ )


In this formula, _M_ represents the original model and _M_ _[′]_ represents a shadow model trained without
access to protected attributes. A low PDS indicates the model’s minimal reliance on protected attribute proxies, which is crucial for identifying subtle forms of indirect discrimination. For example,
in real-world scenarios, features used in machine learning models such as healthcare costs inadvertently served as proxies for race, leading to biased outcomes Obermeyer et al. (2019b). Therefore,
PDS measures fairness by evaluating both direct and indirect independence of the models to protected attributes.


3.2 COUNTERFACTUAL STABILITY RATE: ASSESSING ROBUSTNESS TO PROTECTED
ATTRIBUTE PERTURBATIONS


The Counterfactual Stability Rate (CSR) evaluates how sensitive model predictions are to hypothetical changes in protected attributes. CSR directly captures violations of counterfactual fairness
by measuring the percentage of individuals whose predictions remain unchanged when only their
protected attributes (e.g., race, gender) are counter-factually flipped, while all other non-protected
features remain constant. The formula for CSR is:


In this formula, _f_ ( _xi_ ) represents the prediction for individual _i_, and _f_ ( _x_ _[cf]_ _i_ [)] [is] [the] [prediction] [for]
counterfactual individual where only protected attributes have been altered. A high CSR indicates
that the model’s predictions are stable with respect to changes in protected attributes, implying that
the model is not relying on these attributes in a discriminatory manner. On the other hand, a low
CSR suggests that an individual identical in all non-protected characteristics but differing only in
a protected attribute would receive a different outcome, directly violating the principle that similar
individuals should be treated similarly. This metric is vital for ensuring that model decisions are
based on legitimate, non-discriminatory factors on the individual level.


3


StabilityRate = [1]

_N_


_N_

- I - _f_ ( _xi_ ) = _f_ ( _x_ [cf] _i_ [)] - (2)

_i_ =1


3.3 ATTRIBUTION INDEPENDENCE SCORE: EVALUATING BIASED DECISION RATIONALES


The Attribution Independence Score (AIS) assesses whether feature importance attributions in a
prediction are entangled with protected attributes, and thereby signalling biased decision rationales.
It quantifies the correlation between feature attribution values and protected attributes to uncover
underlying reasons for a model’s decision. The formula for AIS is:


Independence = 1 _−|_ corr(Attr _f_ ( _x_ ) _,_ Protected( _x_ )) _|_ (3)


A high AIS suggests that the model primarily bases its decisions on non-protected features. If there
is a strong correlation between feature attributions and protected attributes, the model’s internal
reasoning process is likely biased even if the final prediction might appear fair at the group level.
AIS helps to uncover subtle, structural sources of bias within the model’s internal logic, moving
beyond outcome-based fairness, which is a crucial shift from merely observing what is unfair to
understanding why it is unfair. By diagnosing biased decision rationales, developers can pinpoint
the root causes of individual unfairness within the model’s internal logic, rather than just observing
external disparities Molnar (2025); Zafar et al. (2017). This allows for more targeted and effective
mitigation strategies that address the fundamental source of bias, leading to more robust and genuinely fair AI systems Gennaro et al. (2025); Manerba (2023). This also aligns with the broader
push for explainable AI (XAI), which is crucial for building trust and accountability in AI systems
Arrieta et al. (2019).


3.4 INTRA-COHORT DECISION CONSISTENCY: QUANTIFYING CONSISTENCY FOR SIMILAR
INDIVIDUALS


The Intra-Cohort Decision Consistency (IDC) quantifies the consistency of decisions across individuals who are nearly identical in terms of their non-protected features. It evaluates the variation
in decisions within cohorts that are defined by their similarity on neutral features. The formula for
IDC is:
Consistency = 1 _−_ Var( _f_ ( _x_ ) _| x ∈_ cohort( _x_ )) (4)

A low variance, which translates to high consistency, indicates that the model treats similar individuals similarly, directly addressing the core principle of individual fairness. IDC is particularly
effective at identifying situations where a model might achieve perfect group fairness but still exhibit
discriminatory outcomes for specific individuals within those groups who are otherwise similar. It
provides an individual-level assessment of consistency, helping to uncover subtle biases that may be
missed by group-level evaluations.


4 EVALUATION AND INSIGHTS


4.1 PSEUDOCODE FOR PROPOSED METRICS


We describe the implementation logic of our proposed individual fairness metrics using pseudocode.
These algorithms quantify different aspects of individual fairness by analyzing the behavior of the
model under varying conditions.


**Proxy** **Dependency** **Score** This metric evaluates how much the model’s performance depends on
sensitive attributes. A significant drop in accuracy after removing sensitive features suggests proxy
dependence.


**Algorithm 1** Compute Proxy Dependency Score
**Require:** Feature matrix _X_, labels _y_, protected columns

1: Split _X, y_ into training and test sets
2: Train full model on training data
3: Compute accuracy on test data _→_ _acc_ full
4: Remove protected columns from _X_ train _, X_ test
5: Train shadow model on modified data
6: Compute accuracy of shadow model _→_ _acc_ shadow
7: Compute Proxy Dependency Score: 1 _−_ _[acc]_ _acc_ [shadow] full

8: **return** Score, _acc_ full, _acc_ shadow


4


**Counterfactual Stability Rate** This metric measures whether a model’s prediction remains consistent when sensitive features are flipped (e.g., changing race or gender). A stable model should not
change its output based on such alterations.


**Algorithm 2** Compute Counterfactual Stability
**Require:** Feature matrix _X_, labels _y_, columns to flip, flip mapping

1: Split _X, y_ into training and test sets
2: Train model on training data
3: Predict on _X_ test _→_ predsorig
4: Copy _X_ test to create counterfactual _X_ cf
5: **for** each column in columns to flip **do**
6: **if** column in flip map **then**
7: Apply flip mapping to _X_ cf
8: **end if**
9: **end for**
10: Predict on _X_ cf _→_ predscf
11: Compute stability = fraction where predsorig = predscf
12: **return** Stability


**Attribution Independence Score** This metric evaluates whether a model’s reasoning (as captured
by feature attributions) is entangled with protected attributes. A fair model’s attribution patterns
should be statistically independent from sensitive features.


**Algorithm 3** Compute Attribution Independence Score
**Require:** Trained model _f_, input samples _x_, protected attributes _P_

1: Compute feature attributions Attr _f_ ( _x_ ) using a method such as SHAP or LIME
2: For each sample, collect the values of protected attributes _P_ ( _x_ )
3: Compute the Pearson correlation between Attr _f_ ( _x_ ) and _P_ ( _x_ ) across the dataset
4: Take the absolute value of the correlation
5: Compute AIS: 1 _−|_ corr(Attr _f_ ( _x_ ) _, P_ ( _x_ )) _|_
6: **return** AIS score


**Intra-Cohort** **Consistency** This metric checks the variance in predicted scores within clusters of
similar individuals (cohorts). A fair model should assign similar scores to similar people, resulting
in low intra-group variance.


**Algorithm 4** Compute Intra-Cohort Consistency
**Require:** Feature matrix _X_, labels _y_, number of clusters _k_

1: Split _X, y_ into training and test sets
2: Train model on training data
3: Predict probability scores on test set _→_ preds
4: Scale _X_ test
5: Apply KMeans clustering to _X_ test, obtain cluster labels
6: Initialize _total_ _var_ = 0, _valid_ ~~_g_~~ _roups_ = 0
7: **for** each cluster _i_ = 1 to _k_ **do**
8: Extract predictions for cluster _i_
9: **if** cluster size _>_ 1 **then**
10: Compute variance and add to _total_ ~~_v_~~ _ar_
11: Increment _valid_ ~~_g_~~ _roups_
12: **end if**
13: **end for**
14: **if** _valid_ ~~_g_~~ _roups >_ 0 **then**
15: _avg_ ~~_v_~~ _ar_ = _total_ ~~_v_~~ _ar/valid_ ~~_g_~~ _roups_
16: **else**
17: _avg_ ~~_v_~~ _ar_ = 1 _._ 0
18: **end if**
19: _consistency_ ~~_s_~~ _core_ = 1 _−_ _avg_ ~~_v_~~ _ar_
20: **return** _consistency_ ~~_s_~~ _core_


5


4.2 EMPIRICAL RESULTS


The 80% rule is a principle stating that if the selection rate for a protected group (such as a minority
group) is less than 80% with respect to the group with the highest selection rate, the selection process may be considered discriminatory U.S. Equal Employment Opportunity Commission (EEOC)
(1978). In Table 1, we present both our proposed individual-based fairness scores and state-of-theart group-based fairness scores on the Adult and COMPAS datasets.


Table 1: Fairness Metric Comparison on Adult and COMPAS Datasets


**Adult Dataset** **COMPAS Dataset**
**Metric**

Overall/Sex Race Age Overall/Sex Race Age


Proxy Dependency Score _−_ 0 _._ 0014 0 _._ 0017 0 _._ 0001 _−_ 0 _._ 009 _−_ 0 _._ 0123 0 _._ 0001
(Fairness range: [-0.2, 0.2])


Intra-Cohort Decision Consistency 0 _._ 955 0 _._ 9674 0 _._ 9676 0 _._ 946 0 _._ 9674 0 _._ 9676
(Fairness range: [0.8, 1])


Counterfactual Stability Rate 0 _._ 956 0 _._ 973 0 _._ 989 **0.773** 0 _._ 907 0 _._ 911
(Fairness range: [0.8, 1])


Min. 0.920 Min. 0.938 Min. 0.951 Min. 0.871 Min. 0.844 Min. 0.892
Attribution Independence Score
Max. 0.999 Max. 1.000 Max. 0.998 Max. 0.996 Max. 0.993 Max. 0.989
(Fairness range: [0.8, 1])


Disparate Impact 1 _._ 113 1 _._ 176 1 _._ 138 **1.456** 1 _._ 206 **1.260**
(Fairness range: [0.8, 1.25])


Statistical Parity Difference 0 _._ 021 0 _._ 032 0 _._ 025 **0.104** 0 _._ 048 0 _._ 061
(Fairness range: [-0.1, 0.1])


4.3 ANALYSIS OF RESULTS AND DISCREPANCIES


Based on empirical results on the Adult and COMPAS datasets, there are a few important insights:


1. Divergence between Group and Individual Fairness Metrics:
There are a few cases where group-level fairness metrics (e.g., Disparate Impact, Statistical
Parity Difference) suggest unfairness, but individual-level metrics (e.g., CSR, AIS, IDC) indicate consistent and fair treatment. For example, in the COMPAS dataset, the age attribute
yields a Disparate Impact of 1.26, slightly outside the fairness range, yet shows a CSR of
0.91 and an AIS _≥_ 0 _._ 89. This suggests that although aggregated group outcomes differ,
individuals with similar non-protected attributes receive stable predictions, highlighting a
key disconnect between group and individual fairness.


2. Agreement Signals Robust Bias:
In contrast, some attributes show consistent unfairness across both metric types. The sex
attribute from the COMPAS dataset has a Disparate Impact of 1.46 and a low CSR of 0.77,
indicating systemic bias at both group and individual levels. This alignment reinforces the
severity of the issue and helps prioritize areas for fairness intervention.


3. High Fairness Across Metrics in the Adult Dataset:
For the Adult dataset, all metrics - both group and individual - fall within acceptable fairness thresholds. The CSR exceeds 0.95, AIS values are consistently high, and PDS is close
to 0, indicating that the model does not rely heavily on protected or proxy attributes. These
results indicate that fairness can be achieved at both levels simultaneously under certain
data and model conditions.


4. Holistic Fairness via Single-Value Metrics:
Two of our proposed metrics, PDS and IDC, produce a single summary score per model,
offering a high-level, attribute-agnostic view of individual fairness. This makes them especially useful as diagnostic tools during model development. For example, IDC scores above
0.94 in both datasets suggest strong consistency in the treatment of similar individuals, even
when group disparities are present.


6


5. Limitations of Group Fairness Alone:
Empirical results show that relying on one group fairness metric can produce an incomplete
or misleading picture of model behavior. A model could appear to be fair at the group level
while producing biased outcomes at the individual level, or vice versa. These findings
support the growing consensus that using only a single measurement (group or individual)
for fairness measurement is inadequate Dwork et al. (2012); Kleinberg et al. (2016).


5 SOCIETAL IMPACT AND ETHICAL CONSIDERATIONS


The increasing usage of AI in critical decisions raises concerns about fairness and accountability.
Biased AI systems could cause societal harm through the amplification of discriminatory and inaccurate decisions learned in the training process O’Neil (2016); Buolamwini & Gebru (2018); Obermeyer et al. (2019a). While this pervasive issue impacts fundamental rights and well-being, current
approaches to AI fairness are insufficient to prevent such harms Schwartz et al. (2022); Mitchell
et al. (2018). These deep-rooted biases need more effective technical solutions to address.


5.1 THE MANIFESTATION OF AI BIAS IN HIGH-STAKES DOMAINS


Algorithmic discrimination has manifested itself in numerous critical sectors, leading to tangible
societal harms.


1. Healthcare: AI bias impacts patient care, leading to issues such as misdiagnosis or denied
access. For example, an algorithm underestimated black patients’ care needs by predicting
healthcare costs Obermeyer et al. (2019b), and dermatology AI systems under-diagnosed
skin cancers in darker skin tones Rezk et al. (2022).


2. Criminal Justice: The COMPAS algorithm showed significant racial bias, incorrectly classifying black defendants as high-risk more often than white defendants Angwin et al.
(2016).


3. Hiring and Employment: Amazon’s recruiting tool was discontinued after downgrading
resumes with ”women’s” due to training on male-dominated historical data Dastin (2018).
LinkedIn’s job recommendations also faced allegations of gender bias Wall & Schellmann
(2021).


4. Credit and Lending: AI systems perpetuate historical discrimination such as redlining, assigning higher risk scores to Black and Latino applicants with similar financial backgrounds
Eubanks (2018). Apple’s credit card even reportedly offered lower limits to women than
their male spouses despite higher credit scores Knight (2019).


5. Generative AI: Image tools like DALL·E 2 and Stable Diffusion exhibited stereotypical
biases, generating predominantly white males for ”CEO” and ”engineer”, and women or
minorities for ”housekeeper” or ”nurse” Bender et al. (2021).


5.2 ETHICAL CHALLENGES AND UNINTENDED CONSEQUENCES


AI bias rooted in prejudiced training data Bender et al. (2021) leads to discrimination and severe social consequences, undermining equal opportunity and amplifying oppression. Biased AI decisions
could lead to unintended lack of transparency and insufficient testing Cheong (2024). AI systems
can perpetuate and amplify existing biases University College London (2024), creating a confirmation bias Nickerson (1998) by reinforcing their own assumptions.


A major ethical challenge in AI development is the inadequate ethical evaluations overshadowed by
performance focus B´elisle-Pipon & Victor (2024). Unchecked AI can reinforce societal biases, infringe on privacy, and cause harm. The bias in AI often manifests subtly, like using proxy variables,
implying that group-level fairness is insufficient Dwork et al. (2012); Prince & Schwarcz (2020).
Individual fairness metrics such as our proposed PDS, CSR, AIS, and IDC should help to uncover
these hidden biases Mukherjee et al. (2020).


Beyond legal risks, a profound ethical imperative exists for responsible AI development UNESCO
(2021). To achieve ethical imperative, developers need to apply embedding fairness, transparency,


7


and accountability throughout the AI lifecycle, moving beyond mere ethical compliance to ensure
AI serves everyone ethically Information Commissioner’s Office (ICO) (2023).


6 FUTURE DIRECTIONS FOR RESPONSIBLE AI


Addressing AI fairness requires a comprehensive, proactive approach throughout the AI lifecycle.


6.1 STRATEGIES FOR MITIGATING INDIVIDUAL BIAS AND ENSURING FAIRNESS


Effective bias mitigation and fairness assurance rely on several key strategies:


    - Data Quality and Preprocessing: Fair AI foundations demand high-quality, diverse,
and representative training data through robust data governance and rigorous cleaning
Gonz´alez-Sendino et al. (2024). Actively identifying and discussing bias-inducing factors
is crucial.


    - Fairness-Aware Algorithms and Model Design: Fairness-Aware Algorithms and Model
Design: Algorithms must be designed with fairness considerations built in, using methods
like reducing bias during development or applying fairness constraints during designing
and training Jang (2024).


    - Human Oversight and Explainable AI (XAI): Human oversight is essential, especially in
high-impact decisions. AI systems should be transparent, using XAI techniques (e.g.,
SHAP, LIME) Arrieta et al. (2019) to enhance understanding, trust, and accountability.


    - Continuous Monitoring and Auditing: Fairness is not static, requiring continuous performance monitoring, regular bias checks, and review throughout auditing during the operational life of the system Anisetti et al. (2025).


6.2 INTEGRATING FAIRNESS INTO MLOPS


Integrating fairness into Machine Learning Operations (MLOps) is paramount for responsible AI
deployment. This involves:


    - Data Validation and Quality Monitoring: Automated data validation pipelines catch biases
before retraining, ensuring data quality.


    - Model Validation and Experiment Tracking: MLOps facilitates structured experimentation
and continuous integration/deployment (CI/CD) for model validation.


    - Continuous Monitoring of Fairness: Production models require ongoing monitoring for
performance, drift, and emerging biases across subgroups.


    - Robust Governance Frameworks: MLOps supports governance that tracks data and model
versions, ensuring explainability, auditability, and compliance. Tools like Fiddler AI Observability aid bias detection and assessment Labs (2023).


6.3 REGULATORY LANDSCAPE AND ACCOUNTABILITY FRAMEWORKS


The evolving AI landscape necessitates robust regulatory and accountability frameworks:


    - Evolving Regulations: Compliance with frameworks like GDPR European Union (2016),
CCPA California State Legislature (2018), and the EU AI Act European Commission
(2024) is critical, ensuring data processing meets purpose without undue intrusion and
avoids discrimination.


    - Algorithmic Accountability Frameworks: Structured systems are essential to ensure algorithmic operation responsibility, emphasizing transparency, bias mitigation, and equitable
outcomes. However, some potential challenges might exist, such as algorithmic complexity and the evolving regulatory environment. Documentation via data protection impact
assessments (DPIAs) is important for proving fair processing.


8


- Addressing Power Asymmetry: It is important to know the power unequal between system
developers and the people affected by their decisions. Improving fairness requires not only
technical solutions, but also social and ethical considerations involving multiple disciplines.


6.4 FUTURE RESEARCH DIRECTIONS


Further research is needed to advance individual AI fairness:


    - Individual-Specific Factors and Metrics: Develop tailored bias evaluation and mitigation
methods considering individual-specific factors beyond traditional protected attributes.

    - Fairness-Accuracy Trade-off: Continue exploring this complex trade-off in various contexts, expecting that different fairness definitions can conflict.

    - Distribution Fairness: Investigate fairness in resource allocation, particularly for physical
and computational resources, developing equitable distribution mechanisms.

    - Cross-Domain Applicability: Enhance the applicability of fairness metrics and mitigation
techniques across diverse domains, promoting data sharing with privacy protections.

    - Clinician-in-the-Loop and Interdisciplinary Collaboration: Integrate AI fairness into practical applications by involving domain experts and fostering broad interdisciplinary.

    - User-Friendly Tools: Develop accessible tools for fairness assessment and mitigation to
facilitate widespread adoption, model validation, and risk management.


7 CONCLUSION


We proposed a comprehensive framework for assessing individual fairness in machine learning models, addressing a critical gap in current fairness evaluation practices. While group-based metrics have
dominated fairness discussions, they often fail to capture the nuanced, person-level inconsistencies
that arise in real-world applications. To bridge this gap, we introduced four novel individual fairness metrics - Proxy Dependency Score (PDS), Counterfactual Stability Rate (CSR), Attribution
Independence Score (AIS), and Intra-Cohort Decision Consistency (IDC) - each designed to capture
distinct dimensions of unfairness at the individual level.


Through empirical evaluations on the Adult and COMPAS datasets, we demonstrated that these
metrics offer complementary perspectives to traditional group fairness measures. Our results reveal
that models deemed unfair by group metrics may still exhibit individual-level consistency, and conversely, models satisfying group fairness can behave inconsistently at the individual level. These
observations underscore the importance of integrating both group and individual metrics in fairness
audits.


Our metrics are interpretable and model-agnostic, providing both attribute-specific and holistic fairness diagnostics. The single-value metrics (PDS and IDC) enable fairness monitoring without pergroup disaggregation, while CSR and AIS expose deeper structural biases, including proxy effects
and unstable decision boundaries.


Looking ahead, we envision several directions for future work. First, integrating these metrics into
training objectives could guide the development of fairness-aware models that are sensitive to both
group-level parity and individual-level consistency. Second, expanding our evaluation to multimodal and large-scale datasets, especially in domains like healthcare or hiring, can reveal how individual fairness manifests in more complex settings. Finally, exploring causal or learned similarity
metrics may further refine our understanding of what constitutes similar individuals in diverse realworld contexts.


By enriching the fairness evaluation toolbox, we hope this work moves the field closer to developing
AI systems that are not only equitable in aggregate, but just and consistent for each individual they
impact.


REFERENCES


Julia Angwin, Jeff Larson, Surya Mattu, and Lauren Kirchner. Machine bias—there’s software
used across the country to predict future criminals. and it’s biased against blacks. _ProP-_


9


_ublica,_ _Online_ _Edition_, May 2016. URL [https://www.propublica.org/article/](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing)
[machine-bias-risk-assessments-in-criminal-sentencing.](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing)


Marco Anisetti, Claudio A. Ardagna, Nicola Bena, Ernesto Damiani, and Paolo G. Panero. Continuous management of machine learning-based application behavior. _IEEE Transactions on Services_
_Computing_, 18(1):112–125, 2025. doi: 10.1109/TSC.2024.3486226.


Kofi Arhin and Daniel Treku. Contextualizing the accuracy-fairness trade-off in algorithmic prediction outcomes. In _Proceedings of the 57th Hawaii International Conference on System Sciences_,
pp. 6878–6887, 2024. [URL https://hdl.handle.net/10125/107210.](https://hdl.handle.net/10125/107210)


Alejandro Barredo Arrieta, Natalia D´ıaz-Rodr´ıguez, Javier Del Ser, Adrien Bennetot, Siham Tabik,
Alberto Barbado, Salvador Garc´ıa, Sergio Gil-L´opez, Daniel Molina, Richard Benjamins, Raja
Chatila, and Francisco Herrera. Explainable artificial intelligence (xai): Concepts, taxonomies,
opportunities and challenges toward responsible ai, 2019. [URL https://arxiv.org/abs/](https://arxiv.org/abs/1910.10045)
[1910.10045.](https://arxiv.org/abs/1910.10045)


Barry Becker and Ronny Kohavi. Adult. UCI Machine Learning Repository, 1996. DOI:
https://doi.org/10.24432/C5XW20.


Emily M. Bender, Timnit Gebru, Angelina McMillan-Major, and Shmargaret Shmitchell. On the
dangers of stochastic parrots: Can language models be too big? . In _Proceedings_ _of_ _the_ _2021_
_ACM Conference on Fairness, Accountability, and Transparency_, FAccT ’21, pp. 610–623, New
York, NY, USA, 2021. Association for Computing Machinery. ISBN 9781450383097. doi: 10.
1145/3442188.3445922. [URL https://doi.org/10.1145/3442188.3445922.](https://doi.org/10.1145/3442188.3445922)


Joy Buolamwini and Timnit Gebru. Gender shades: Intersectional accuracy disparities in commercial gender classification. In Sorelle A. Friedler and Christo Wilson (eds.), _Proceedings_
_of_ _the_ _1st_ _Conference_ _on_ _Fairness,_ _Accountability_ _and_ _Transparency_, volume 81 of _Proceed-_
_ings_ _of_ _Machine_ _Learning_ _Research_, pp. 77–91. PMLR, 23–24 Feb 2018. URL [https:](https://proceedings.mlr.press/v81/buolamwini18a.html)
[//proceedings.mlr.press/v81/buolamwini18a.html.](https://proceedings.mlr.press/v81/buolamwini18a.html)


Jean-Christophe B´elisle-Pipon and Gabriel Victor. Ethics dumping in artificial intelligence. _Fron-_
_tiers in Artificial Intelligence_, 7:1426761, Nov 2024. doi: 10.3389/frai.2024.1426761.


California State Legislature. California Consumer Privacy Act (CCPA), 2018. URL [https://](https://oag.ca.gov/privacy/ccpa)
[oag.ca.gov/privacy/ccpa.](https://oag.ca.gov/privacy/ccpa) Accessed: 2025-08-02.


Ben Chester Cheong. Transparency and accountability in ai systems: safeguarding wellbeing in
the age of algorithmic decision-making. _Frontiers in Human Dynamics_, Volume 6 - 2024, 2024.
ISSN 2673-2726. doi: 10.3389/fhumd.2024.1421273. [URL https://www.frontiersin.](https://www.frontiersin.org/journals/human-dynamics/articles/10.3389/fhumd.2024.1421273)
[org/journals/human-dynamics/articles/10.3389/fhumd.2024.1421273.](https://www.frontiersin.org/journals/human-dynamics/articles/10.3389/fhumd.2024.1421273)


Alexandra Chouldechova. Fair prediction with disparate impact: A study of bias in recidivism
prediction instruments. _arXiv preprint arXiv:1703.00056_ [, 2017. URL https://arxiv.org/](https://arxiv.org/abs/1703.00056)
[abs/1703.00056.](https://arxiv.org/abs/1703.00056)


Jeffrey Dastin. Amazon scraps secret ai recruiting tool that showed bias against women.
_Reuters_, Oct 2018. URL [https://www.reuters.com/article/world/](https://www.reuters.com/article/world/insight-amazon-scraps-secret-ai-recruiting-tool-that-showed-bias-against-women-idUSKCN1MK0AG/)
[insight-amazon-scraps-secret-ai-recruiting-tool-that-showed-bias-against-women-idU](https://www.reuters.com/article/world/insight-amazon-scraps-secret-ai-recruiting-tool-that-showed-bias-against-women-idUSKCN1MK0AG/)


Cynthia Dwork, Moritz Hardt, Toniann Pitassi, Omer Reingold, and Richard Zemel. Fairness
through awareness. In _Proceedings of the 3rd Innovations in Theoretical Computer Science Con-_
_ference_, ITCS ’12, pp. 214–226, New York, NY, USA, 2012. ACM. [URL http://doi.acm.](http://doi.acm.org/10.1145/2090236.2090255)
[org/10.1145/2090236.2090255.](http://doi.acm.org/10.1145/2090236.2090255)


Virginia Eubanks. _Automating_ _Inequality:_ _How_ _High-Tech_ _Tools_ _Profile,_ _Police,_ _and_ _Punish_ _the_
_Poor_ . St. Martin’s Press, 2018. ISBN 978-1250074317.


European Commission. Regulation of the European Parliament and of the Council Laying
Down Harmonised Rules on Artificial Intelligence (EU AI Act), 2024. URL [https://](https://artificialintelligenceact.eu)
[artificialintelligenceact.eu.](https://artificialintelligenceact.eu) Accessed: 2025-08-02.


10


European Union. Regulation (EU) 2016/679 of the European Parliament and of the Council General Data Protection Regulation (GDPR), 2016. [URL https://eur-lex.europa.eu/](https://eur-lex.europa.eu/eli/reg/2016/679/oj)
[eli/reg/2016/679/oj.](https://eur-lex.europa.eu/eli/reg/2016/679/oj) Accessed: 2025-08-02.


Clifford G Filippi, Jonathan M Stein, Ziyuan Wang, Spyridon Bakas, Yong Liu, Peter D Chang,
Yvonne Lui, Christopher Hess, Daniel P Barboriak, Adam E Flanders, Max Wintermark, Greg
Zaharchuk, and Ona Wu. Ethical considerations and fairness in the use of artificial intelligence
for neuroradiology. _AJNR American Journal of Neuroradiology_, 44(11):1242–1248, 2023. doi:
10.3174/ajnr.A7963.


Shubham Galhotra, Yuriy Brun, and Alexandra Meliou. Fairness testing: Testing software for discrimination. In _Proceedings_ _of_ _the_ _2017_ _11th_ _Joint_ _Meeting_ _on_ _Foundations_ _of_ _Software_ _Engi-_
_neering_, pp. 498–510, 2017.


Federico Di Gennaro, Thibault Laugel, Vincent Grari, and Marcin Detyniecki. Controlled model
debiasing through minimal and interpretable updates, 2025. URL [https://arxiv.org/](https://arxiv.org/abs/2502.21284)
[abs/2502.21284.](https://arxiv.org/abs/2502.21284)


Amol Ghadage, Du Yi, George M. Coghill, and Weiru Pang. Multi-stage bias mitigation for individual fairness in algorithmic decisions. In N. El Gayar, E. Trentin, M. Ravanelli, and H. Abbas
(eds.), _Artificial_ _Neural_ _Networks_ _in_ _Pattern_ _Recognition:_ _ANNPR_ _2022_, volume 13739 of _Lec-_
_ture Notes in Computer Science_, pp. 40–52, Dubai, United Arab Emirates, 2023. Springer. doi:
10.1007/978-3-031-20650-4 ~~4~~ . 10th IAPR TC3 Workshop on Artificial Neural Networks in Pattern Recognition 2022, 24/11/22.


Rub´en Gonz´alez-Sendino, Emilio Serrano, and Javier Bajo. Mitigating bias in artificial intelligence: Fair data generation via causal models for transparent and explainable decisionmaking. _Future_ _Generation_ _Computer_ _Systems_, 155:384–401, 2024. ISSN 0167-739X. doi:
https://doi.org/10.1016/j.future.2024.02.023. URL [https://www.sciencedirect.com/](https://www.sciencedirect.com/science/article/pii/S0167739X24000694)
[science/article/pii/S0167739X24000694.](https://www.sciencedirect.com/science/article/pii/S0167739X24000694)


Moritz Hardt, Eric Price, and Nathan Srebro. Equality of opportunity in supervised learning. In D. D. Lee, M. Sugiyama, U. V. Luxburg, I. Guyon, and
R. Garnett (eds.), _Advances_ _in_ _Neural_ _Information_ _Processing_ _Systems_ _29_, pp. 3315–
3323. Curran Associates, Inc., 2016. URL [http://papers.nips.cc/paper/](http://papers.nips.cc/paper/6374-equality-of-opportunity-in-supervised-learning.pdf)
[6374-equality-of-opportunity-in-supervised-learning.pdf.](http://papers.nips.cc/paper/6374-equality-of-opportunity-in-supervised-learning.pdf)


Information Commissioner’s Office (ICO). Guidance on ai and data protection: Annex a: Fairness in the ai lifecycle. _ICO_, Mar 2023. URL [https://ico.](https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/artificial-intelligence/guidance-on-ai-and-data-protection/annex-a-fairness-in-the-ai-lifecycle/)
[org.uk/for-organisations/uk-gdpr-guidance-and-resources/](https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/artificial-intelligence/guidance-on-ai-and-data-protection/annex-a-fairness-in-the-ai-lifecycle/)
[artificial-intelligence/guidance-on-ai-and-data-protection/](https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/artificial-intelligence/guidance-on-ai-and-data-protection/annex-a-fairness-in-the-ai-lifecycle/)
[annex-a-fairness-in-the-ai-lifecycle/.](https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/artificial-intelligence/guidance-on-ai-and-data-protection/annex-a-fairness-in-the-ai-lifecycle/)


Taeuk Jang. _NOVEL_ _APPROACHES_ _TO_ _MITIGATE_ _DATA_ _BIAS_ _AND_ _MODEL_ _BIAS_ _FOR_ _FAIR_
_MACHINE_ _LEARNING_ _PIPELINES_ . Thesis, Purdue University, 2024. URL [https://doi.](https://doi.org/10.25394/PGS.25670736.v1)
[org/10.25394/PGS.25670736.v1.](https://doi.org/10.25394/PGS.25670736.v1)


Deepak Vijaykeerthy John, Philips George and Diptikalyan Saha. Verifying individual fairness in
machine learning models. _Conference_ _on_ _Uncertainty_ _in_ _Artificial_ _Intelligence_, pp. 749–758,
2020.


Jon M. Kleinberg, Sendhil Mullainathan, and Manish Raghavan. Inherent trade-offs in the fair
determination of risk scores. _CoRR_, abs/1609.05807, 2016. [URL http://arxiv.org/abs/](http://arxiv.org/abs/1609.05807)
[1609.05807.](http://arxiv.org/abs/1609.05807)


Will Knight. The apple card didn’t ‘see’ gender—and that’s the problem. _Wired_, Nov 2019. URL [https://www.wired.com/story/](https://www.wired.com/story/the-apple-card-didnt-see-genderand-thats-the-problem/)
[the-apple-card-didnt-see-genderand-thats-the-problem/.](https://www.wired.com/story/the-apple-card-didnt-see-genderand-thats-the-problem/)


Matt Kusner, Joshua Loftus, Chris Russell, and Ricardo Silva. Counterfactual fairness. _Advances in_
_neural information processing systems 30_, 03 2017. doi: 10.48550/arXiv.1703.06856.


11


Fiddler Labs. Fiddler ai observability platform, 2023. URL [https://www.fiddler.ai/](https://www.fiddler.ai/platform/observability)
[platform/observability.](https://www.fiddler.ai/platform/observability) Accessed: 2025-08-02.


Preethi Lahoti, Krishna P. Gummadi, and Gerhard Weikum. Operationalizing individual fairness
with pairwise fair representations. In _Proc. VLDB Endow._, pp. 506–518, 2019.


Xiaofei Li, Ping Wu, and Jialu Su. Accurate fairness: Improving individual fairness without trading
accuracy. _Proceedings_ _of_ _the_ _AAAI_ _Conference_ _on_ _Artificial_ _Intelligence_, 37(12):14312–14320,
2023. doi: https://doi.org/10.1609/aaai.v37i12.26674.


Martin Lious. Explainable machine learning models for high-stakes decision-making: Bridging
transparency and performance. _IRE Journals_, 6(6):357–373, 2022. ISSN 2456-8880.


Maeve MacCarthy. Standards of fairness for disparate impact assessment of big data algorithms
(april 2, 2018). _SSRN_ _Electronic_ _Journal_, 2018. URL [https://ssrn.com/abstract=](https://ssrn.com/abstract=3154788)
[3154788.](https://ssrn.com/abstract=3154788)


Marta Marchiori Manerba. Fairness auditing, explanation and debiasing in linguistic data and language models. In _xAI (Late-breaking Work,_ _Demos,_ _Doctoral Consortium)_, pp. 241–248, 2023.
[URL https://ceur-ws.org/Vol-3554/paper39.pdf.](https://ceur-ws.org/Vol-3554/paper39.pdf)


Margaret Mitchell, Simone Wu, Andrew Zaldivar, Parker Barnes, Lucy Vasserman, Ben Hutchinson,
Elena Spitzer, Inioluwa Deborah Raji, and Timnit Gebru. Model cards for model reporting. _CoRR_,
abs/1810.03993, 2018. [URL http://arxiv.org/abs/1810.03993.](http://arxiv.org/abs/1810.03993)


Christoph Molnar. _Interpretable_ _Machine_ _Learning_ . 3 edition, 2025. ISBN 978-3-911578-03-5.
[URL https://christophm.github.io/interpretable-ml-book.](https://christophm.github.io/interpretable-ml-book)


Debarghya Mukherjee, Mikhail Yurochkin, Moulinath Banerjee, and Yuekai Sun. Two simple ways
to learn individual fairness metrics from data. In _International_ _Conference_ _on_ _Machine_ _Learn-_
_ing_, pp. 7097–7107. PMLR, 2020. URL [https://doi.org/10.48550/arXiv.2006.](https://doi.org/10.48550/arXiv.2006.11439)
[11439.](https://doi.org/10.48550/arXiv.2006.11439)


Raymond Nickerson. Confirmation bias: A ubiquitous phenomenon in many guises. _Review_ _of_
_General Psychology_, 2:175–220, 06 1998. doi: 10.1037/1089-2680.2.2.175.


Ziad Obermeyer, Brian Powers, Christine Vogeli, and Sendhil Mullainathan. Dissecting racial bias
in an algorithm used to manage the health of populations. _Science_, 366(6464):447–453, 2019a.
doi: 10.1126/science.aax2342. [URL https://www.science.org/doi/abs/10.1126/](https://www.science.org/doi/abs/10.1126/science.aax2342)
[science.aax2342.](https://www.science.org/doi/abs/10.1126/science.aax2342)


Ziad Obermeyer, Brian Powers, Christine Vogeli, and Sendhil Mullainathan. Dissecting racial bias
in an algorithm used to manage the health of populations. _Science_, 366(6464):447–453, 2019b.
doi: 10.1126/science.aax2342.


Cathy O’Neil. _Weapons_ _of_ _Math_ _Destruction:_ _How_ _Big_ _Data_ _Increases_ _Inequality_ _and_ _Threatens_
_Democracy_ . Crown, New York, NY, 2016. ISBN 978-0553418811.


Drago Plecko and Elias Bareinboim. Fairness-accuracy trade-offs: A causal perspective. _Proceed-_
_ings of the AAAI Conference on Artificial Intelligence_, 39(25):26344–26353, 2025.


Anya E. R. Prince and Daniel Schwarcz. Proxy discrimination in the age of artificial intelligence and big data. _Iowa_ _Law_ _Review_, 105(3):1257–1318, 2020.
URL [https://ilr.law.uiowa.edu/print/volume-105-issue-3/](https://ilr.law.uiowa.edu/print/volume-105-issue-3/proxy-discrimination-in-the-age-of-artificial-intelligence-and-big-data/)
[proxy-discrimination-in-the-age-of-artificial-intelligence-and-big-data/.](https://ilr.law.uiowa.edu/print/volume-105-issue-3/proxy-discrimination-in-the-age-of-artificial-intelligence-and-big-data/)


ProPublica. ProPublica COMPAS Recidivism Risk Score Data. [https://github.com/](https://github.com/propublica/compas-analysis/tree/master/compas-scores-two-years)
[propublica/compas-analysis/tree/master/compas-scores-two-years,](https://github.com/propublica/compas-analysis/tree/master/compas-scores-two-years)
2016. URL [https://github.com/propublica/compas-analysis/tree/](https://github.com/propublica/compas-analysis/tree/master/compas-scores-two-years)
[master/compas-scores-two-years.](https://github.com/propublica/compas-analysis/tree/master/compas-scores-two-years) Accessed on 2025-05-01.


Eman Rezk, Menna Eltorki, and Wael El-Dakhakhni. Improving skin color diversity in cancer
detection: Deep learning approach. _JMIR Dermatology_, 5(3):e39143, Aug 2022. doi: 10.2196/
39143.


12


Reva Schwartz, Apostol Vassilev, Kristen K. Greene, Lori Perine, Andrew Burt, and Patrick
Hall. Towards a standard for identifying and managing bias in artificial intelligence, 2022-0315 04:03:00 2022. [URL https://tsapps.nist.gov/publication/get_pdf.cfm?](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=934464)
[pub_id=934464.](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=934464)


UNESCO. Recommendation on the ethics of artificial intelligence. _UNESCO_, 2021. [URL https:](https://unesdoc.unesco.org/ark:/48223/pf0000380455)
[//unesdoc.unesco.org/ark:/48223/pf0000380455.](https://unesdoc.unesco.org/ark:/48223/pf0000380455)


University College London. Bias in ai amplifies our own biases, researchers show. _Sci-_
_enceDaily_, Dec 2024. [URL https://www.sciencedaily.com/releases/2024/12/](https://www.sciencedaily.com/releases/2024/12/241218132137.htm)
[241218132137.htm.](https://www.sciencedaily.com/releases/2024/12/241218132137.htm)


U.S. Equal Employment Opportunity Commission (EEOC). Questions and answers
to clarify and provide a common interpretation of the uniform guidelines on employee selection procedures. [https://www.eeoc.gov/laws/guidance/](https://www.eeoc.gov/laws/guidance/questions-and-answers-clarify-and-provide-common-interpretation-uniform-guidelines)
[questions-and-answers-clarify-and-provide-common-interpretation-uniform-guidelines](https://www.eeoc.gov/laws/guidance/questions-and-answers-clarify-and-provide-common-interpretation-uniform-guidelines)
1978. Accessed: 2025-05-12.


Sahil Verma and Julia Rubin. Fairness definitions explained. In _Proceedings_ _of_ _the_ _International_
_Workshop on Software Fairness_, FairWare ’18, pp. 1–7, New York, NY, USA, 2018. ACM. URL
[http://doi.acm.org/10.1145/3194770.3194776.](http://doi.acm.org/10.1145/3194770.3194776)


Sheridan Wall and Hilke Schellmann. Linkedin’s ai was biased. the company’s solution? more ai. _MIT_ _Technology_ _Review_, Jun 2021. URL
[https://www.technologyreview.com/2021/06/23/1026825/](https://www.technologyreview.com/2021/06/23/1026825/linkedin-ai-bias-ziprecruiter-monster-artificial-intelligence/)
[linkedin-ai-bias-ziprecruiter-monster-artificial-intelligence/.](https://www.technologyreview.com/2021/06/23/1026825/linkedin-ai-bias-ziprecruiter-monster-artificial-intelligence/)


Wei Xie and Ping Wu. Fairness testing of machine learning models using deep reinforcement learning. In _2020_ _IEEE_ _19th_ _International_ _Conference_ _on_ _Trust,_ _Security_ _and_ _Privacy_ _in_ _Computing_
_and Communications (TrustCom)_, pp. 121–128, 2020.


Shihan Xu and Thomas Strohmer. On the (in)compatibility between group fairness and individual
fairness. _arXiv (Cornell University)_, 2024. doi: https://doi.org/10.48550/arxiv.2401.07174.


Muhammad Bilal Zafar, Isabel Valera, Manuel Gomez Rodriguez, and Krishna P. Gummadi. Fairness beyond disparate treatment & disparate impact: Learning classification without disparate
mistreatment. In _Proceedings of the 26th International Conference on World Wide Web_, WWW
’17, pp. 1171–1180, Republic and Canton of Geneva, CHE, 2017. International World Wide Web
Conferences Steering Committee. ISBN 9781450349130. doi: 10.1145/3038912.3052660. URL
[https://doi.org/10.1145/3038912.3052660.](https://doi.org/10.1145/3038912.3052660)


Wenbin Zhang, Zichong Wang, Juyong Kim, Cheng Cheng, Thomas Oommen, Pradeep Ravikumar,
and Jeremy Weiss. Individual fairness under uncertainty. In _ECAI_ _2023_, pp. 3042–3049. IOS
Press, 2023.


13