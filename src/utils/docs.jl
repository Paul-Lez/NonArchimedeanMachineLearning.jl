import Pkg

# Absolute path to the package root directory
const namldir = normpath(@__DIR__, "..", "..")

function open_doc()
    filename = normpath(namldir, "docs", "build", "index.html")
    @static if Sys.isapple()
        run(`open $(filename)`; wait = false)
    elseif Sys.islinux() || Sys.isbsd()
        run(`xdg-open $(filename)`; wait = false)
    elseif Sys.iswindows()
        cmd = get(ENV, "COMSPEC", "cmd.exe")
        run(`$(cmd) /c start $(filename)`; wait = false)
    else
        @warn("Opening files the default application is not supported on this OS.",
              KERNEL = Sys.KERNEL)
    end
end

function start_doc_preview_server(; open_browser::Bool = true, port::Int = 8000)
    build_dir = normpath(namldir, "docs", "build")
    cmd = """
          using Pkg;
          Pkg.activate(temp = true);
          Pkg.add("LiveServer");
          using LiveServer;
          LiveServer.serve(dir = "$build_dir", launch_browser = $open_browser, port = $port);
          """
    live_server_process = run(`$(Base.julia_cmd()) -e $cmd`, wait = false)
    atexit(_ -> kill(live_server_process))
    @info "Starting server with PID $(getpid(live_server_process)) on 127.0.0.1:$port"
    return nothing
end

# Activate the docs/ project, instantiate it, and add it to LOAD_PATH
# so that `using Documenter` works without manually switching environments.
function doc_init()
    docsdir = normpath(namldir, "docs")
    Pkg.activate(docsdir) do
        Pkg.instantiate()
    end
    docsdir in Base.LOAD_PATH || pushfirst!(Base.LOAD_PATH, docsdir)
    return nothing
end

"""
    build_doc(; doctest=false, warnonly=true, open_browser=true, start_server=false)

Build the manual of `NonArchimedeanMachineLearning.jl` locally and open the
front page in a browser.

The optional parameter `doctest` can take three values:
  - `false`: Do not run the doctests (default).
  - `true`: Run the doctests and report errors.
  - `:fix`: Run the doctests and replace the output in the manual with
    the output produced by the package. Use this option carefully.

The optional parameter `warnonly` is passed to `makedocs` of `Documenter.jl`.
Set it to `false` to make doctest failures raise an error.

To prevent the opening of the browser at the end, set `open_browser` to `false`.

Set `start_server` to `true` to start a `LiveServer` web server in the build
directory (accessible at `127.0.0.1:8000`). If both `start_server` and
`open_browser` are `true`, the browser opens that page. The server runs until
the Julia session is terminated; the intended workflow is:
```
build_doc(start_server = true)   # first build
build_doc(open_browser = false)  # subsequent builds — just refresh the tab
```

Loading `Revise` before the package significantly speeds up repeated builds:
```
using Revise, NonArchimedeanMachineLearning
```
"""
function build_doc(; doctest::Union{Symbol, Bool} = false, warnonly = true,
                     open_browser::Bool = true, start_server::Bool = false)
    if !isdefined(Main, :Documenter)
        doc_init()
        @eval Main using Documenter
    end

    docsdir = normpath(namldir, "docs")
    Base.invokelatest() do
        Documenter = Main.Documenter
        Documenter.makedocs(
            root     = docsdir,
            sitename = "NonArchimedeanMachineLearning.jl",
            modules  = [NonArchimedeanMachineLearning],
            warnonly = warnonly,
            doctest  = doctest,
            format   = Documenter.HTML(
                prettyurls     = false,       # local build: don't require a server
                edit_link      = "main",
                size_threshold = 400_000,
            ),
            pages = include(joinpath(docsdir, "pages.jl")),
        )
    end

    if start_server
        start_doc_preview_server(open_browser = open_browser)
    elseif open_browser
        open_doc()
    end
end
