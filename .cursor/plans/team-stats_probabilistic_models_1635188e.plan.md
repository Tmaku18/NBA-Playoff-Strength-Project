---
name: Team-stats probabilistic models
overview: Add a team-stats modeling track with Linear Regression, BayesianRidge, GPR kernel sweep, and Gaussian-mixture-based models, plus uncertainty ranges in outputs and evaluation for existing A/B/C and new models. Include optional Model A score as a leak-safe feature and wire experiment toggles through config and pipeline scripts.
todos:
  - id: cfg-surface
    content: Add config toggles and uncertainty settings for LR/BayesianRidge/GPR/GMM and model_a_score feature
    status: completed
  - id: new-model-modules
    content: Implement new team-stats model modules with unified train/predict/uncertainty APIs
    status: completed
  - id: model-a-feature
    content: Wire OOF-safe Model A score as optional team-stats feature in training and inference
    status: completed
  - id: pipeline-integration
    content: Integrate new models into script 4 and trainer with artifact + OOF persistence
    status: completed
  - id: inference-ranges
    content: Add predicted rank interval fields for ensemble, A/B/C, and new models
    status: completed
  - id: eval-uncertainty
    content: Add uncertainty calibration and coverage metrics to script 5 evaluate
    status: completed
  - id: docs-notion-sync
    content: Update README/docs and mirror updates in connected Notion pages with 🤖 icon
    status: completed
isProject: false
---

