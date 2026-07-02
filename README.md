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

### **Streamlit Community Cloud (Recommended - Easiest)** ⚡

1. **Commit and push your code to GitHub:**
   ```bash
   git add .
   git commit -m "Ready for deployment"
   git push origin main
   ```

2. **Go to [share.streamlit.io](https://share.streamlit.io)**

3. **Sign in with GitHub** and authorize Streamlit

4. **Click "Create app" and:**
   - Select your repo: `anushakundu19-debug/hrmatch`
   - Branch: `main`
   - Main file path: `app.py`

5. **Add the environment variable:**
   - In the app settings (gear icon)
   - Go to **Secrets**
   - Paste your secret:
     ```
     GROQ_API_KEY = "your_groq_api_key_here"
     ```

6. **Deploy!** The app will be live at: `https://hrmatch-<random-name>.streamlit.app`

### Render (Alternative)
   - Connect the GitHub repo to Render
   - Use the existing Procfile and requirements.txt
   - Set GROQ_API_KEY as an environment variable in Render