import os

# Keep test runs out of LangSmith even when .env turns tracing on (load_dotenv won't override this).
os.environ["LANGSMITH_TRACING"] = "false"
