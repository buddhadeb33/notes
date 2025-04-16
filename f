AIM model output/pickle files are not available in the Git repo.

Downloading ~45 model files manually is tedious; Dataiku doesn’t support bulk downloads, and some files fail to download.

We're still exploring the inference, post-processing, and prediction pipeline—lack of workflow documentation slows understanding.

All models seem to run together, with no clear separation of inference per model, making AWS implementation complex.

The codebase lacks a clear structure or workflow guide, and we can’t track how models are being triggered or executed.

Execution logs are missing; coordinating with Liya to obtain them.

Extracting feature inputs from pickle files is time-consuming.

Many Python files in the repo lack descriptions, making it harder to understand their purpose.

Insights are unavailable—working with Liya for support, but much of the analysis is being done independently.


##
Possible Challenges to overcome
Dataiku doesn’t support bulk export, making file transfer time-consuming.
Reengineering effort to convert dataiku code to AWS needs heavy refactoring.
