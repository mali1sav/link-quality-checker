## Start
Open the provided Streamlit URL in your web browser to access the Content Relevancy Analyser.
In the sidebar on the left:

- Enter your OpenAI API key in the "OpenAI API Key" field.
- Enter your DataForSEO API key in the "DataForSEO API Key" field.

These keys are necessary for the app to function and will be securely stored for your session.

Relevancy Threshold

- Use the slider in the sidebar to set your desired relevancy threshold (0-100).
- Websites with a content similarity score above this threshold will be marked as "Relevant".

## Prepare Your Data
Prepare a CSV or Excel file with the following columns:

- Market
- Source Domain
- Target URL

Ensure your file contains these columns (case-insensitive) for the analysis to work correctly.

Upload Your File

- Click on the "Upload CSV or Excel file" button in the main area of the app.
- Select your prepared CSV or Excel file from your computer.

Analyse Content

- After uploading your file, click the "Analyse Content" button.
- The app will process your data and display a progress bar.
- Wait for the analysis to complete.

## Review Results

Once the analysis is complete, you'll see the results displayed:

- Each result is in an expandable section showing the source domain and target URL.
- Click on a section to expand and view detailed information:
  - Content Similarity score
  - Verdict (Relevant or Not Relevant)
  - Explanation of the relevance
  - Source Summary
  - SERP Titles and Descriptions

## Update Verdicts (Optional)

- You can manually change the verdict for each result using the radio buttons.
- After making changes, click the "Update Results" button to save your modifications.

## Download Results

- To save your results, click the "Download results as Excel" button.
- This will download an Excel file containing all the analysis results and your manual verdict changes.

## Important Notes

- Your session, including API keys and settings, will be saved for 3 hours.
- After 3 hours, you'll need to re-enter your API keys and settings.
- Ensure you have a stable internet connection for the best experience.

If you encounter any issues or have questions about using the app, please contact the support team for assistance.
