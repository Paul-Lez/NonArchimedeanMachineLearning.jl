# Docker Setup for NAML

This Docker setup provides a sandboxed environment for running NAML (Non-Archimedean Machine Learning) code, particularly useful for agent execution and collaboration.

## Prerequisites

- Docker installed ([Get Docker](https://docs.docker.com/get-docker/))
- Docker Compose installed (usually comes with Docker Desktop)

## Quick Start

### 1. Build the Docker image

```bash
docker-compose build
```

This will:
- Pull the Julia 1.10 base image
- Install all dependencies from `Project.toml`/`Manifest.toml`
- Precompile packages (takes 10-15 minutes first time due to Oscar.jl)

### 2. Run Interactive Julia REPL

```bash
docker-compose run --rm naml
```

This starts a Julia REPL with the NAML package loaded. Example usage:

```julia
using NAML
# Your code here...
```

### 3. Run Jupyter Notebooks

```bash
docker-compose up jupyter
```

Then open your browser to the URL shown in the terminal (typically `http://localhost:8888`).

### 4. Run a specific test file

```bash
docker-compose run --rm naml julia --project=. test/polydisc.jl
```

### 5. Run a Julia script

```bash
docker-compose run --rm naml julia --project=. your_script.jl
```

## Services

### `naml` Service (Default)
- **Purpose**: Sandboxed execution environment for agent development
- **Security**: Process isolation, minimal capabilities, filesystem boundaries
- **Use case**: Running agent code with ability to edit and test

### `jupyter` Service
- **Purpose**: Interactive notebook development
- **Port**: 8888
- **Use case**: Exploratory work and demos

## Directory Structure

```
/app              # Working directory in container (read-write, mounted from host)
/app/outputs      # Directory for results and outputs
```

## Security Features (naml service)

The `naml` service provides sandboxed execution with:

- **Process isolation**: Container isolation prevents access to host processes
- **Filesystem boundaries**: Cannot access files outside `/app` directory
- **Minimal capabilities**: Only essential Linux capabilities enabled
- **No privilege escalation**: `no-new-privileges` security option
- **Git safety**: All changes can be reverted with `git restore`

Note: The code mount is read-write to allow agents to edit and test code. Use git to track and revert any unwanted changes.

## Common Commands

### Build/rebuild image
```bash
docker-compose build [--no-cache]
```

### Start Jupyter server
```bash
docker-compose up jupyter
```

### Run interactive Julia
```bash
docker-compose run --rm naml
```

### Execute a command in the container
```bash
docker-compose run --rm naml julia -e 'using NAML; println("Hello")'
```

### Stop all services
```bash
docker-compose down
```

### Remove all containers and images
```bash
docker-compose down --rmi all
```

## Development Workflow

### Option 1: Edit locally, run in container
1. Edit files on your host machine with your preferred editor
2. Run code in the sandboxed container:
   ```bash
   docker-compose run --rm naml julia --project=. your_script.jl
   ```

### Option 2: VS Code Remote-Containers
1. Install "Remote - Containers" extension
2. Open project in VS Code
3. Command Palette → "Reopen in Container"

## Troubleshooting

### Long build times
- **First build**: Oscar.jl takes 10-15 minutes to compile
- **Subsequent builds**: Should use cached layers if Project.toml unchanged

### Permission errors
- Outputs directory may have root ownership (Linux)
- Fix: `sudo chown -R $USER:$USER outputs/`

### Julia package precompilation
If packages need recompilation:
```bash
docker-compose run --rm naml julia -e 'using Pkg; Pkg.precompile()'
```

### Jupyter kernel not found
Rebuild the image:
```bash
docker-compose build --no-cache
```

## Environment Variables

Set in `docker-compose.yml` or override:

```bash
JULIA_NUM_THREADS=4 docker-compose run --rm naml
```

Available variables:
- `JULIA_NUM_THREADS`: Number of Julia threads (default: auto)
- `JULIA_DEBUG`: Enable debug logging (e.g., `JULIA_DEBUG=loading`)

## Advanced Usage

### Mount additional volumes
Edit `docker-compose.yml` to add volume mounts:

```yaml
volumes:
  - .:/app:ro
  - ./outputs:/app/outputs
  - ./data:/app/data:ro  # Add read-only data mount
```

### Custom Julia version
Edit `Dockerfile`:

```dockerfile
FROM julia:1.11  # Change version here
```

### Run with different security settings
For development (less restrictive):

```bash
docker-compose run --rm jupyter  # Uses jupyter service (no security restrictions)
```

## Package Management

### Add a new dependency

1. Run interactive Julia in container:
   ```bash
   docker-compose run --rm naml
   ```

2. Add package:
   ```julia
   using Pkg
   Pkg.add("PackageName")
   ```

3. Copy updated `Project.toml` and `Manifest.toml` from container:
   ```bash
   docker cp naml-sandbox:/app/Project.toml .
   docker cp naml-sandbox:/app/Manifest.toml .
   ```

4. Rebuild image:
   ```bash
   docker-compose build
   ```
