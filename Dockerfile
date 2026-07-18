# Use a lightweight Python 3.10 image
FROM python:3.10-slim

# Set environment variables to prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

# Install required system dependencies (if any)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set up a new user named "user" with user ID 1000
# Hugging Face Spaces runs containers as a non-root user for security
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory
WORKDIR $HOME/app

# Copy the requirements file and install Python dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY --chown=user . $HOME/app

# Expose port 7860 (This is strictly required by Hugging Face Spaces)
EXPOSE 7860

# Run the Flask app via gunicorn on port 7860
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app", "--timeout", "120"]
