# Use a lightweight Python base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install `uv` via pip
RUN pip install uv

# Copy the pyproject.toml and the lock file or requirements if you have them
# (We copy these first to leverage Docker layer caching for dependencies)
COPY pyproject.toml .

# Install dependencies using uv into the system site-packages so litestar is available globally in the container
RUN uv pip install --system -r pyproject.toml

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application using Litestar
CMD ["litestar", "--app", "main:app", "run", "--host", "0.0.0.0", "--port", "8000"]