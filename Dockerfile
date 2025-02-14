FROM python:3.10

WORKDIR /app



COPY /src/training/model_maker/RL/coin_df3.csv /src/training/model_maker/RL/coin_df3.csv
# Copy only requirements.txt first (for efficient caching)
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt


# Copy the rest of the project files (including `src/`)
COPY . .


# Set the working directory to src before running the script
WORKDIR /app/src/training/model_maker/RL/

CMD ["python3", "RL_model.py"] 