# Use official Julia image
FROM julia:1.10

# Set working directory
WORKDIR /app

# Copy Project.toml and Manifest.toml first for better caching
COPY Project.toml Manifest.toml ./

# Install dependencies
RUN julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

# Install IJulia for Jupyter notebook support
RUN julia -e 'using Pkg; Pkg.add("IJulia")'

# Copy the rest of the application
COPY . .

# Expose Jupyter port
EXPOSE 8888

# Default command: start Julia REPL
CMD ["julia", "--project=."]
