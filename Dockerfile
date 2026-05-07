FROM tensorflow/tensorflow:1.15.5

# Install Python dependencies
# Wheels are available for amd64, so we don't need complex build setups
RUN pip install gym \
    shapely \
    matplotlib \
    torch \
    scipy \
    pyyaml \
    rasterio \
    tabulate \
    filterpy \
    scikit-image


# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Set PYTHONPATH environment variable
ENV PYTHONPATH="${PYTHONPATH}:/app:/app/../infoplanner/lib"

EXPOSE 80

# Default command
CMD ["./run_adfq.sh"]
