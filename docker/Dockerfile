# Use miniconda as base image
FROM continuumio/miniconda3

# Set the working directory in the container
WORKDIR /app

# Copy the environment.yml file to the container
COPY environment.yml .

# Create the environment from the environment.yml file
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "ml_prac", "/bin/bash", "-c"]

# Copy the content of the current directory to /app in the container
COPY . /app

# Ensure the environment's binaries are in PATH
ENV PATH /opt/conda/envs/ml_prac/bin:$PATH

# Make port 80 available to the world outside this container
EXPOSE 80

# Run app.py when the container launches
CMD ["streamlit", "run", "app.py"]