# Building Social World Model with Large Language Models

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 2, 4, 4

## Abstract
Understanding and predicting how social beliefs evolve in response to events, ranging from policy changes to scientific breakthroughs, remains a fundamental challenge in social science research. Given that Large Language Models (LLMs) have demonstrated commonsense knowledge and social intelligence, a natural question arises: Can LLMs be used to model the dynamics of social beliefs following social events? Addressing this problem can deepen our understanding of community dynamics and inform better decision-making in the real world. In this work, we introduce the Social World Model (SocialWM) concept, a general framework designed to capture how social beliefs evolve in response to major events. SocialWM learns state-transition functions for social beliefs by mining temporal patterns in social data and optimizing the evidence lower bound, without the need for explicit causal annotations that link events to belief shifts or expensive census data. To evaluate SocialWM's effectiveness in predicting social belief transitions, we introduce a benchmark, SocialWM-bench, derived from real-world Polymarket data. SocialWM-bench includes over 300,000 data samples for social belief prediction tasks spanning diverse domains such as politics, sports, cryptocurrency, and elections. Our experimental results show that SocialWM significantly outperforms traditional time-series models, achieving a 21.56% reduction in RMSE while offering interpretable insights into the underlying mechanisms of social belief dynamics.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces the Social World Model (SWM), a novel LLM-based framework designed to model and predict the evolution of social beliefs over time in response to significant social events. The central idea is to treat social beliefs as states and social events as actions within a temporal world model, analogous to world models in reinforcement learning, but grounded in collective human opinion rather than physical environments. The model explicitly decouples social reasoning and belief transition, leveraging a posterior-guided reasoner (powered by GPT-4o) and an LLM-based transition model trained via ELBO optimization. To evaluate SWM, the authors construct SWM-Bench, a large-scale benchmark derived from over 300,000 samples of real-world Polymarket prediction market data across domains like politics, crypto, and sports. SWM significantly outperforms strong time-series and LLM baselines, particularly in cases of abrupt belief shifts.

### Strengths
1. The concept of modeling social beliefs as state transitions influenced by events is interesting and addresses an important societal challenge
2. SWM-Bench provides a valuable resource with 300,000+ real-world data points across diverse domains
3. The treatment of events as latent variables and use of ELBO optimization cleverly addresses the lack of paired state-action data

### Weaknesses
1. The core technical contribution is primarily applying existing ELBO optimization and LLM fine-tuning to a new domain rather than developing fundamentally new methods
2. The paper doesn't validate whether identified events actually cause belief changes or are merely correlated

### Questions
N/A

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces what the authors call a Social World Model — a framework in which Large Language Models (LLMs) are leveraged to model how collective social beliefs evolve in response to real-world events. The work focuses on predicting “social belief,” which is represented as a pair of yes/no questions along with the community expectation (i.e., the fraction of “yes” votes on the Polymarket prediction market).

The model approaches this prediction problem through an autoregressive formulation of the time series, introducing a latent event variable. Training is performed via a variational ELBO objective that involves several components the authors refer to as the social reasoner, posterior guide reasoner, and the social world model itself.

### Strengths
The paper presents a compelling narrative by introducing intuitive concepts to model large-scale social dynamics. The engineering effort required to obtain the reported results is impressive, as it involves training LLM-based components within a Bayesian approximate inference framework — a task that is technically challenging given the size of the models involved (e.g., the social reasoner and posterior networks). The authors skillfully incorporate both pretraining and fine-tuning within the ELBO framework to make this feasible.

### Weaknesses
Despite the engaging presentation and terminology, the core technical contribution of the paper is limited. From a machine learning perspective, the proposed approach is not conceptually novel. At its core, it is a time series latent variable model (reminiscent of earlier works such as  Chung et al. (2015), with the main difference being that LLMs are used as components of the recognition and generative networks. While this integration is interesting from an engineering standpoint, it does not represent a significant methodological advance for ICLR.

Furthermore, the claims made under the banner of a “Social World Model” seem overstated, given that the method ultimately performs time series prediction on Polymarket data. For these reasons, this work might be better suited for venues such as WWW or ICWSM, which focus on e.g., computational social science and large-scale data-driven analyses of online behavior.

*References:*

- Chung et al. *A Recurrent Latent Variable Model for Sequential Data*. NeurIPS (2015)

### Questions
1. Why do you refer "the reasoner" reasoner? In the context of LLMs, this term can be misleading, since “reasoning” is typically associated with chain-of-thought (CoT) or reinforcement learning on CoT approaches.

2. Why is one component fine-tuned while another uses LoRA adaptation? The choice seems inconsistent. Could you clarify the reasoning behind this design?

3. The manuscript refers to “*predicting how social beliefs evolve in response to events*”. For me the fundamental problem is that we don't know that the people in polymarket are reacting to *those news*, as opposed to other events. How do you address the risk of capturing spurious correlations instead of genuine causal responses?

### Soundness
3

### Presentation
4

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces the Social World Model (SWM), a framework that models how social beliefs evolve in response to real-world events. Using data from Polymarket, the authors construct SWM-Bench, a large-scale benchmark for social belief prediction. SWM combines large language models with probabilistic event modeling to capture belief transitions over time and demonstrates improved performance over traditional time-series and LLM baselines.

### Strengths
Novel dataset creation
The introduction of SWM-Bench, derived from real-world Polymarket data, is a significant contribution. It provides a large-scale, domain-diverse benchmark for studying social belief dynamics and enables systematic evaluation of models that combine temporal forecasting with social reasoning. The dataset’s grounding in real market behavior gives it potential value for future research in social simulation and opinion modeling.

Innovative modeling framework
The proposed Social World Model (SWM) offers a creative formulation that treats social belief evolution as a state-transition process influenced by latent social events. The integration of LLM-based reasoning with probabilistic modeling (through ELBO optimization and posterior-guided training) is technically interesting. This framework bridges time-series prediction and social reasoning, contributing a fresh perspective to computational social science.

### Weaknesses
Limited justification for market-based data as authentic social opinion
The paper assumes that prediction market data, such as Polymarket transactions, are more reliable and authentic indicators of public opinion. However, this claim is not sufficiently supported. The argument would benefit from a clearer explanation of why financially motivated trading behavior better reflects social beliefs than alternative data sources like surveys or social media.

Potential bias and groupthink in Polymarket data
The use of Polymarket as the primary dataset introduces the risk of collective biases. Markets often exhibit herd behavior, social reinforcement, or correlated trading patterns. These dynamics may distort the very opinion signals that SWM aims to model. The paper does not discuss how such effects are mitigated or how they might affect model reliability.

Limited comparison to other reasoning-based forecasting methods
The paper primarily compares SWM against time-series models and simple LLM baselines, but not against retrieval-augmented or reasoning-based approaches. This omission makes it difficult to evaluate how SWM performs relative to more advanced forecasting frameworks.

Generalization limited to a single data source
The evaluation is confined to Polymarket data. Without evidence of cross-domain performance, it is unclear whether SWM generalizes to other platforms such as Good Judgment Project, Metaculus, or text-based datasets like OpinionsQA. This limits the claims of broad applicability.

### Questions
1. The authors claim that “free markets serve as a classical example of aggregated public opinions.” Could they elaborate on this reasoning? Specifically, what properties of market behavior make it a suitable proxy for collective belief formation?

2. How does SWM handle correlated or self-reinforcing behavior in markets, where traders may be influenced by prevailing trends or dominant narratives rather than independent judgment?

3. How would SWM compare to retrieval-augmented forecasting models, such as “Approaching Human-Level Forecasting with Language Models” (Halawi et al., 2024)? 

4. Has the generalization of SWM been tested on other datasets or domains, such as expert forecasting platforms or broader opinion benchmarks (some examples of which are mentioned in weakness)?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes the Social World Model (SWM), a framework for predicting how social beliefs evolve in response to events using Large Language Models (LLMs). SWM treats events as latent variables and performs optimization using Evidence Lower Bound (ELBO) with posterior guidance. To evaluate SWM, the authors introduce SWM-Bench, a benchmark derived from Polymarket data containing over 300,000 samples across cryptocurrency, elections, sports, and politics.

### Strengths
S1. This paper tackles an important problem of modeling social belief dynamics, which has practical applications in event forecasting, policy making, and understanding community behavior. The weakly-supervised approach using only time-series data without explicit event-belief annotations makes the problem more tractable.  

S2. SWM-Bench provides a valuable benchmark for social belief prediction given its substantial scale and diversity across multiple domains. This dataset could facilitate future research in this area.  

S3. This paper conducts comprehensive ablation studies examining model size, window size, and event space size. Table 2 and Figures 3, 4, and 5 provide useful insights into component contributions and design choices.

### Weaknesses
W1. The technical novelty is limited, as SWM primarily combines existing techniques including Transformer encoders for time-series, standard ELBO optimization, and contrastive learning objectives. The main contribution appears to be adapting these methods to social belief prediction rather than methodological innovation.  

W2. The evaluation is restricted to a single data source from Polymarket, which might have inherent biases in user demographics and market selection. Generalization to other social belief data sources remains unclear.  

W3. This paper uses GPT-4o as the posterior-guided reasoner, making the approach dependent on proprietary models and potentially expensive for scaling. The computational costs and practical deployment considerations are not thoroughly discussed.  

W4. Several technical details require clarification. The conditional independence assumption in Equation 1 is strong but justified only briefly. Based on Equations 6 and 7, the retriever implementation lacks details about retrieval methods and how similarity is computed.

### Questions
Q1: How does SWM perform on cross-domain generalization, such as training on political data but testing on crypto markets? This would better demonstrate SWM’s ability to capture general social dynamics versus domain-specific patterns.  

Q2: The conditional independence assumption in Equation 1 seems critical but potentially problematic. Will there be some cases where beliefs are interdependent?  

Q3: For Equations 6 and 7, what similarity metrics are used for retrieval?  

Q4: How does SWM handle cases where important events are missing from the collected new data? What percentage of belief shifts in the test set can be attributed to events in news corpus?

### Soundness
2

### Presentation
3

### Contribution
2
