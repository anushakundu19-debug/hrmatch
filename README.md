# hrmatch

HR AI Portfolio project for screening resumes with Python, Streamlit, and Groq.

## Run locally

1. Create and activate a virtual environment:
   ```bash
   cd /workspaces/hrmatch
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set your Groq API key:
   - Paste it into the sidebar textbox in the app, or
   - export it in the terminal:
     ```bash
     export GROQ_API_KEY=your_key_here
     ```
   - Alternatively, copy [.env.example](.env.example) to .env and put your key there.
4. Start the app:
   ```bash
   streamlit run app.py
   ```

## What this does

- Upload one or more PDF resumes
- Extract text from the PDFs
- Send the resume and a job description to Groq for structured analysis
- Compare each candidate to the job description and show a match score

## Publish as an app

The easiest options are:

1. Streamlit Community Cloud
   - Push this repo to GitHub
   - Open Streamlit Community Cloud
   - Select the repository and the app entrypoint as app.py
   - Add the environment variable GROQ_API_KEY in the app settings

2. Render
   - Connect the GitHub repo to Render
   - Use the existing Procfile and requirements.txt
   - Set GROQ_API_KEY as an environment variable in Render

3. Railway or Hugging Face Spaces
   - Use the same Python app structure
   - Set GROQ_API_KEY as an environment variable

Important: the app needs a valid GROQ_API_KEY to analyze resumes.