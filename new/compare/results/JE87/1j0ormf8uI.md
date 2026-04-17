# Review

## Summary
This paper proposes a new calibration procedure for conformal prediction of survival times under treatment in the presence of right-censoring. Under the potential outcome framework and strong ignorability assumption, the authors develop a reweighting scheme that converts the problem into a weighted conformal inference problem. This allows for the calculation of a lower prediction bound (LPB) on survival time via quantile regression with an exact miscoverage guarantee. The method is doubly robust against model misspecification and empirical evaluations on synthetic and real-world clinical data demonstrate the validity and informativeness of the constructed LPBs.

## Soundness
4

## Presentation
3

## Contribution
3

## Strengths
The paper is well-written, and the contribution is solid.

## Weaknesses
The proposed method relies on several strong assumptions (e.g., ignorability and SUTVA), which may limit its applicability in practice.

## Questions
1. The simulation settings are a bit confusing to me. Could you please clarify how the covariate W and the censoring time C are generated in the simulation? Also, how the outlier data are introduced in Setting 4?

2. Could you please elaborate on how the robustness to model misspecification is established? Specifically, how does the method handle different types of model misspecification, and what are the implications for the accuracy of the LPB?

3. The method involves several steps, including quantile regression, reweighting, and calibration. How sensitive is the final result to the choice of hyperparameters in each step? Is there a way to optimize these choices systematically?

4. How does the method perform when the assumptions (e.g., ignorability) are violated in practice? Are there any robust variants of the method that can handle common violations of these assumptions?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
8

## Confidence
4