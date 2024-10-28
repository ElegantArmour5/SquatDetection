# SquatDetection

Squat Detection web app built using Flask and MediaPipe Library in python.

Current criteria for successful squat includes knee angle less than or equal to 90 degrees.

Feedback system classifies correct or incorrect and if incorrect provides reason (example: squat is too shallow).

# Quick Run
1. Download the repository files (git clone)
2. Install dependencies using pip (`pip install -r requirements.txt`)
3. Run main app.py file (in command prompt, navigate to correct directory and run `python app.py` in the terminal)
4. App now runs on development localhost server (access it at `127.0.0.1`; usually runs on port 5000). Access this by pasting this IP into the browser search bar.

# Future Improvements

1. Add more criteria for correct squat like knees behind toes, etc
