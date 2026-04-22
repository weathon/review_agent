# Traceable Black-Box Watermarks For Federated Learning

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 6, 4, 4

## Abstract
Due to the distributed nature of Federated Learning (FL) systems, each local client has access to the global model, which poses a critical risk of model leakage. Existing works have explored injecting watermarks into local models to enable intellectual property protection. However, these methods either focus on non-traceable watermarks or traceable but white-box watermarks. We identify a gap in the literature regarding the formal definition of traceable black-box watermarking and the formulation of the problem of injecting such watermarks into FL systems. In this work, we first formalize the problem of injecting traceable black-box watermarks into FL. Based on the problem, we propose a novel server-side watermarking method, $\mathbf{TraMark}$, which creates a traceable watermarked model for each client, enabling verification of model leakage in black-box settings. To achieve this, $\mathbf{TraMark}$ partitions the model parameter space into two distinct regions: the main task region and the watermarking region. Subsequently, a personalized global model is constructed for each client by aggregating only the main task region while preserving the watermarking region. Each model then learns a unique watermark exclusively within the watermarking region using a distinct watermark dataset before being sent back to the local client. Extensive results across various FL systems demonstrate that $\mathbf{TraMark}$ ensures the traceability of all watermarked models while preserving their main task performance. The code is available at \url{https://github.com/JiiahaoXU/TraMark}.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a method to address the copyright issue in Federated Learning (FL) systems where clients might leak the shared global model. The authors introduce TraMark, a server-side watermarking framework designed to be both black-box (verifiable without parameter access) and traceable (capable of identifying the specific client who leaked the model). The core mechanism involves partitioning the model's parameter space into a 'main task region' and a 'watermarking region', identifying the latter by selecting the least important parameters after a warmup phase. Using a "masked aggregation" process, the server combines updates from all clients only in the main task region. It simultaneously preserves a distinct watermarking region for each individual client, into which a unique watermark is injected using a distinct dataset. This approach provides each client with a personalized model that maintains high performance on the primary task while embedding a unique identifier, crucially preventing the watermarks from being destroyed during the aggregation process. The authors' evaluation shows the method achieves high traceability with a minimal drop in main task accuracy and demonstrates robustness against removal attacks like pruning and fine-tuning.

### Strengths
- This paper has a clear motivation to achieve black-box traceability in FL.
- This paper provides a comprehensive evaluation of various datasets and both IID and non-IID settings.
- This paper is generally well-written and easy to follow.

### Weaknesses
- Insufficient ablation study. To better justify the necessity of the parameter partitioning, it may be better for the authors to include a comparison against a simpler baseline that does not use this partitioning. This would help quantify the impact of watermark collisions or main task degradation that the partitioning scheme is designed to prevent.
- Inadequate evaluation of computational overhead. The proposed method requires the server to perform personalized watermark injection (fine-tuning) for every client in each round. This process could be much slower than white-box methods like FedTracker, especially as the number of clients increases. It may be better for the authors to evaluate the computational time cost with a varying number of clients.
- Lack of clarity in reported results. Some reported data is imprecise. For example, in Table 1, the per-dataset Verification Rate (VR) is only represented by a checkmark (if >95%) or an 'X', with only an average VR reported. It may be better for the authors to report the specific VRs for all methods and datasets to allow for a more granular and direct comparison.

### Questions
Please refer to the Weaknesses.

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
4

### Summary
This paper addresses the challenge of verifying model ownership and tracing leakage in Federated Learning under a black-box setting. The authors formalize the problem and propose a framework named TraMark, which leverages parameter space partitioning and masked aggregation. Experimental results demonstrate the framework's verification rate and its impact on main task performance, complemented by a hyperparameter analysis. Furthermore, the method exhibits robustness against attacks such as pruning and fine-tuning.

### Strengths
A clear formalization of the traceable black-box watermarking problem is provided in Sec. 3.

The experimental results demonstrate an excellent Verification Rate (VR) of approximately 99.17% with a limited drop in Main-task Accuracy (MA), especially when compared to the FedTracker method.

The evaluation is conducted across multiple datasets, and the analyses of robustness, hyperparameters, and other factors are detailed.

### Weaknesses
As indicated in Table 8 of the appendix, the per-round computational overhead for TraMark's aggregation is over 70 times that of FedAvg. The authors rationalize this by citing the small number of clients in cross-silo scenarios, a justification that is not entirely convincing and severely limits the method's scope of application.

The paper appears to overlook the method's communication overhead. At the beginning of each training round, the server is required to send a unique, personalized model to every client. This results in a sharp increase in communication costs compared to broadcasting a single global model, which in turn restricts the method's applicability.

The method's reliance on constructing an independent, out-of-distribution watermark dataset and a unique output label space for each client is a strong assumption. In realistic Federated Learning scenarios, this requirement may not be feasible, thus limiting the method's practical scalability and applicability.

The paper uses the average accuracy across all personalized client models as the metric for MA. However, this average may conceal significant performance disparities, where some clients receive highly accurate models while others are left with poorly performing ones. This masks potential issues of unfairness in performance distribution.

The method, in its pursuit of model traceability, raises questions about its potential impact on privacy preservation—one of the core advantages of Federated Learning. A detailed discussion on this critical trade-off appears to be missing from the paper.

The paper does not discuss the method's feasibility and effectiveness in Federated Learning scenarios with asynchronous client participation. It is unclear how the proposed mechanism would perform under such conditions.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
In this paper, the authors propose TraMark, a black-box model watermarking technique suitable for federated learning setups where clients might be potential model leakers. TraMark works when the central server is benign and implements the watermarking procedure. TraMark verification also enables traitor tracing. TraMark restricts watermarking to a small subset of the model parameters using binary masks and adapts the weight update procedure to produce watermarked global models per each client while preserving the existence of the watermark. I think the idea is nice and simple, but the recovery of watermarked weights and possible evasion methods are not included in the paper.

### Strengths
S1. TraMark partitions the model parameters to the main task and watermarking task regions. In this way, we can say that TraMark is model-agnostic.

S2. The problem formulation and the insights in Section 2 are very clear. It is difficult to inject watermarks that satisfy black-box traceability while avoiding collusion, and I think the authors did a good job of formulating this difficulty.

### Weaknesses
W1. Potential watermark detectability by malicious clients: 

Watermarked weights may be relatively easy to detect by malicious clients. They could analyze which parameters change more significantly or differently during training and identify trends in weight updates of received global models. Such differences, especially if certain parameters are updated disproportionately or the updates remain closer to zero, could reveal which weights are used to embed covert information (i.e., the watermark in this case). This issue may become even more pronounced with the warm-up phase, where clients suddenly see a change in weight updates after the warm-up. The authors should include experiments or analyses investigating this potential vulnerability.

W2. Key discussions moved to the appendix:

Several important discussion points are placed in the Appendix, without even a summary of the results in the main text. This weakens the perceived impact of the contributions. I recommend moving the experimental setup to the Appendix instead and bringing key takeaways or insights from the additional experiments currently in the appendix into the main text.

W3. Other weaknesses:

Please refer to the detailed questions below for additional points of concern.

### Questions
Q1. The authors should verify the correctness of the compared methods presented in the appendix. For example, FedTracker can perform ownership verification in a black-box manner, as the related paper explicitly states that "the zero-bit backdoor-based watermark is feasible for ownership verification and can be verified through black-box access." Additionally, I do not agree with the claim that RobWe is less practical because watermarking is performed on the client side. In fact, this seems more practical: each client can have their own secret watermark, eliminating potential watermark collusion problems, and maintaining robustness even when the server is untrusted.
    
Q2. There is insufficient discussion of scalability. The paper only tests with 10–50 clients. How would the proposed method perform with 1000 or 10000 clients? Moreover, how does the approach scale in relation to collusion resistance and the need to maintain personalized global models for each client? 
    
Q3. The authors consider federated averaging as the only aggregation method. How would TraMark perform with alternative aggregation strategies such as Krum or other robust aggregation methods?
    
Q4. The format of in-text references is wrong. The authors should also be included in parentheses.
    
Q5. What is the model accuracy after the warm-up phase? Could this intermediate model (which includes no watermarks) be distributed directly instead of the last trained model?
    
Q6. The assumption that the server has full access to all local models is too strong and contradicts one of the main motivations for federated learning: privacy preservation. This assumption should be relaxed or at least discussed in detail.
    
Q7. Table 1 only reports the average results. The authors should also include verification rates (VR) for each method on each dataset. The current figure does not clearly convey these proportions.
    
Q8. Have the authors considered the possibility of out-of-distribution (OOD) detection as an evasion strategy against the proposed watermarking method? Especially considering the fact that they are using OOD samples as watermarks. 
    
Q9. The repository link provided in the OpenReview abstract results in an error ("not found"). However, the link in the PDF appears to work.

### Soundness
2

### Presentation
3

### Contribution
3
