# Learning Beyond Proximity: Causal Reasoning with LLMs for Robust POI Prediction

- Decision: Reject
- Scores: 6, 4, 4, 2

## Abstract
Point-of-Interest (POI) prediction forecasts a user’s next destination from mobility history. A key challenge is geographic exposure bias, where users often visit nearby or popular places out of convenience rather than genuine interest. Such convenience-driven behaviors create spurious correlations that obscure true preferences, leading models to misinterpret frequent check-ins as strong signals of interest. Traditional sequential/graph models rely on surface-level statistical correlations, and recent Large Language Model (LLM)-based methods improve semantic coverage but still inherits exposure bias from observational logs. We address this with causal inference, explicitly modeling the data-generating process and distinguishes preference-driven behaviors from convenience-driven ones. In particular, we estimate geographic propensity scores that quantify the likelihood of a visit due to spatial exposure, and use them to reweight check-ins and align trajectory retrieval in exposure-consistent space. Towards this end, we propose Causal Geographic Prediction (CGP), a unified framework that integrates causal inference with LLM-based trajectory modeling. It employs exposure-aware trajectory prompting, causal-geographic similarity alignment, and supervised fine-tuning to separate genuine preferences from convenience-driven behaviors. Experiments on real-world datasets show that CGP outperforms state-of-the-art baselines.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper tackles the problem of Point-of-Interest (POI) prediction and focuses on mitigating geographic exposure bias, where users’ check-ins are frequently influenced by spatial convenience rather than genuine preferences. The authors propose CGP, a framework integrating causal inference with LLM-based trajectory prompting to disentangle preference-driven and convenience-driven behavior. The method includes trajectory prompting, geographic propensity estimation, causal-geographic similarity, exposure-consistent retrieval, and LoRA-based fine-tuning. Experiments on three real-world datasets (Gowalla, NYC-T, and Ma-ST) show improvements over prior sequential and LLM-based baselines, along with ablation studies and interpretability analysis.

### Strengths
1. Overall, the paper identifies and clearly motivates the influence of geographic exposure bias in POI prediction and ties it to causal inference principles. The causal perspective is coherent and well justified.
2. Extensive experiments were conducted on multiple real-world datasets, and the evaluation includes stratified comparisons on near/far and popular/long-tail POIs, which could effectively demonstrate the significance of the proposed scheme.

### Weaknesses
1. The geographic propensity score modeling is introduced as a key component, but the paper lacks more explicit analysis or validation of its estimation quality. For example, there is no discussion of calibration, sensitivity to spatial binning, or comparison with simpler baselines (e.g., distance only methods).
2. The trajectory prompting relies heavily on converting structured data into natural language templates. However, the design choice of the specific textual format is not ablated, and it remains unclear how sensitive final performance is to prompt phrasing.
3. Although the method achieves performance gains, the absolute improvements over the strongest LLM-based baseline (GA-LLM) are moderate, especially on datasets where exposure bias is weaker. The generalization advantage may not be uniform across mobility settings.

### Questions
Please refer to the weakness section for my questions. The authors are encouraged to provide more clarifications regarding the details in the paper.

### Soundness
3

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
5

### Summary
This paper proposes a solution based on LLM and causal reasoning to address geographic exposure bias in POI prediction. This bias occurs when models mistakenly interpret locations frequently visited for convenience as genuine user preferences. To distinguish convenience from genuine preference, we introduce the Causal Geographic Prediction (CGP) framework, which integrates causal reasoning with LLMs. The core of this approach involves estimating a geographic propensity score to quantify the influence of convenience, thereby adjusting the retrieval of historical trajectories. This enables LLMs to learn to identify users' true interests. Experiments demonstrate that our method outperforms existing baselines.

### Strengths
1. This is the first work to consider the mixed relationship between users' actual trajectories and POIs, and to embed causal inference into POI prediction based on LLM.
2. This framework enhances the interpretability of POI predictions. By estimating geographic propensity scores, the model can explain its reasoning process and thereby generate more accurate predictions.
3. Across multiple real-world datasets (Gowalla, NYC-T, Ma-ST), the CGP framework consistently outperforms all existing baseline models, particularly in predicting POI with geographic exposure bias.

### Weaknesses
1. In the methodology section of this paper, the core assumption is that a high “geographic propensity score” P(p∣d,c,t) equates to convenience-driven visits. This may overlook POI that simultaneously possess both convenience and preference.
2. The proposed framework exhibits overconfidence in the “causal reasoning” capabilities of LLMs, as it directly inputs bias scores as numerical values into LLM prompts. The authors assume that LLMs can understand the causal implications of these numbers through fine-tuning. However, is it possible that LLM models have merely learned a new statistical correlation rather than performing the “causal reasoning” claimed by the authors?
3. This paper employs a single evaluation metric, relying solely on Acc@1 during assessment. POI prediction is fundamentally a ranking task. The sole Acc@1 metric is far from sufficient, as it fails to reflect the model's overall ranking quality across the Top-K list.
4. Regarding the second component of CGP, Geographic Propensity Estimation, it is unclear how a simple MLP can be used to predict the score based on trajectory prompting.

### Questions
1. I disagree with the author's proposed motivation for geographic exposure bias. If users frequently visit certain locations for convenience, this should also represent a user preference. When predicting the next POI based on historical data, this factor should inherently be taken into account.

### Soundness
2

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
This paper proposes a method named Causal Geographic POI Prediction (CGP), which aims to address the Geographic Exposure Bias in location prediction through causal inference. Traditional sequence-based or graph-based POI prediction methods are often influenced by the convenience of users' access to geographical locations, thereby overlooking their genuine preferences. To tackle this issue, CGP integrates causal reasoning and large language models (LLMs) by estimating Geographic Propensity Scores to identify visit behaviors driven by geographical convenience. This adjustment is then leveraged to improve POI prediction.

### Strengths
1.The research motivation is clear, focusing on addressing POI prediction errors caused by geographic exposure bias.
2.By incorporating causal inference and geographic propensity estimation, the proposed method tackles the geographic exposure bias issue faced by traditional POI prediction approaches, demonstrating high innovation.
3.Across multiple experimental settings, the proposed method consistently outperforms baseline methods.

### Weaknesses
1.The symbols and related modules in the overall framework diagram of the paper are not provided with necessary explanations alongside the diagram. Additionally, there are detail errors in the text and logic, which cause difficulties in understanding.
2.The core innovation points, such as the definition and calculation principles of the geographic propensity score, are not introduced in the main text. Please provide the relevant details of these key innovations. This includes how the embedded vectors Emb(d), Emb(c), and Emb(t) are input into the MLP to learn the geographic propensity output.
3.The experimental details are not clear enough. For example, the number of layers involved in MLP training, the loss function, and the loss function involved during fine-tuning need to be specified. What is the target for fine-tuning? The parameters for LoRA fine-tuning, such as the rank of matrices A and B, should be provided as well. The lack of these key parameters weakens the reliability and reproducibility of the results. The paper also does not include an appendix to provide these details.
4.Is the proposed method end-to-end, or does it involve staged training?
5.While the comparison experiments include 10 baseline models, some of the baseline models are relatively outdated. Are there any more recent relevant works?

### Questions
Please refer to weaknesses

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper targets exposure bias in next-POI prediction—i.e., users’ convenience-driven visits being mistaken as genuine preferences. It  proposes CGP, which combines geographic propensity estimation and a causal-geographic similarity for exposure-consistent retrieval, and  LLM-based trajectory prompting with PEFT fine-tuning. Empirically, CGP improves Acc@1 on Gowalla, NYC-T, and Ma-ST.

### Strengths
+ Tackles an important issue in POI prediction by considering geographic exposure bias.
+ Explicitly aims to separate convenience-driven visits from true user preferences using causal reasoning with LLMs.
+ Propose to distinguish genuine preferences and convenience-driven behaviors with the propensity scores used as balancing variables and with retrieval aligned in exposure-consistent space.

### Weaknesses
- POI prediction has a long history; the paper should clarify concrete application scenarios
- It remains unclear how text prompting alone enables reliable separation of habitual proximity from true interests; the paper should analyze which prompt tokens (distance, propensity, category, time) most influence decisions and whether LLM explanations align with ground truth. 
- The benefits of fine-tuning are not cleanly isolated: report with/without SFT, parameter counts, data size sensitivity, and whether gains persist across different LLM backbones under a fixed retrieval/propensity setup.
- The proposed causal-geographic similarity lacks theoretical guarantees or identification assumptions. The paper should articulate conditions (ignorability/overlap/SUTVA) under which propensity-based reweighting and retrieval yield unbiased preference estimation.

### Questions
Please refer to the weakness section.

### Soundness
2

### Presentation
3

### Contribution
2
