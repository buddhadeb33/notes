🧾 Inference Model Architecture Description
This inference pipeline is strategically divided into two main components: Inference Model Foundation and Inference Model Prediction. Each component comprises a sequence of logically dependent modules, where the output of one module serves as the input for the next, ensuring a streamlined and deterministic flow of data across the entire lifecycle.

🔹 Inference Model Foundation
This layer prepares and structures the data and metadata essential for model prediction. It includes configuration ingestion, scoping, feature population (both raw and cosmetic), and Salesforce feedback suppression. Each of these modules produces intermediate outputs that are crucial for the downstream prediction layer.

🔹 Inference Model Prediction
Once the foundation modules are complete, this layer takes over, extracting model configurations, generating predictions, applying manual adjustments, and qualifying the predictions. Each module in this sequence depends on enriched, structured outputs from the foundation phase.

🔹 Data Management & Infrastructure
All intermediate and final data (inputs, outputs, features, and predictions) are stored in Amazon S3 buckets, ensuring durability, scalability, and traceability across modules.

All configuration files and model logic are version-controlled and accessed from a central GitHub repository, enabling reproducible inference runs, easy audits, and seamless CI/CD integrations.
