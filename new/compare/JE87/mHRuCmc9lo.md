# Review

## Summary
The paper studies robust decision making with partially calibrated forecasts. The authors formalize the problem of deriving optimal decision policies for a given class of H-calibration, and provide a general characterization of the optimal robust policy. They then specialize the results to decision calibration, and show that the minimax optimal robust policy coincides with the plug-in best response policy. Finally, the authors give examples of other calibration classes, such as self-orthogonality and bin-wise calibration, and provide the corresponding robust decision policies.

## Soundness
4

## Presentation
4

## Contribution
4

## Strengths
- The paper addresses an important problem of robust decision making with partially calibrated forecasts. The problem is well-motivated and the results are significant.
- The paper provides a general framework for characterizing the optimal robust policy for a given class of H-calibration. The framework is general and can be applied to various calibration classes.
- The paper shows that for decision calibration, the minimax optimal robust policy coincides with the plug-in best response policy. This result is surprising and useful, as it provides a clear target for forecaster design and a clear requirement for downstream decision makers.
- The authors give examples of other calibration classes, such as self-orthogonality and bin-wise calibration, and provide the corresponding robust decision policies. These results provide practical examples of how to apply the proposed framework.
- The paper is well-written and easy to follow. The authors provide clear explanations of the problem, the proposed framework, and the results.

## Weaknesses
- The paper assumes that the utility function is linear in the second argument. This assumption limits the applicability of the results to certain types of decision problems. The authors should discuss the implications of this assumption and provide examples of decision problems where this assumption holds.
- The paper focuses on a specific class of calibration guarantees, namely H-calibration. It would be interesting to see how the results generalize to other types of calibration guarantees, such as in the recent paper [1].

[1] Hu, Lunjia, and Yifan Wu. "Predict to minimize swap regret for all payoff-bounded tasks." 2024 IEEE 65th Annual Symposium on Foundations of Computer Science (FOCS). IEEE, 2024.

## Questions
- Can the results be generalized to other types of calibration guarantees beyond H-calibration?
- Are there any practical examples of decision problems where the utility function is linear in the second argument?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
8

## Confidence
4