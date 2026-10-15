# FusionScope

vtk.js viewer for the six Fusion Equilibrium Challenge demo shots (3 × DIII-D, 3 × MAST).

## Reproduce from scratch

Needs [uv](https://docs.astral.sh/uv/), `git`, `git-lfs` and `curl`. Everything is installed into
`.venv/` inside the project, so nothing touches the rest of your system.

1. **Get the starter kit and the six demo shots.** The shots are stored with Git LFS; without
   `git lfs pull` you get ~130-byte pointer files instead of data.

       git clone https://github.com/Sophelio/fusion-equilibrium-challenge-starter starter-kit
       (cd starter-kit && git lfs install && git lfs pull)
       mkdir -p data/raw && cp starter-kit/parquet_data/*.parquet data/raw/

   Each DIII-D file should be ~18 MB and each MAST file ~5 MB (`ls -lh data/raw`).

2. **Create the virtual environment.** It needs only numpy and pyarrow. uv's download cache is
   pointed inside `.venv/` so that deleting `.venv/` removes everything.

       uv venv --python 3.11 .venv
       UV_CACHE_DIR=$PWD/.venv/.uv-cache VIRTUAL_ENV=$PWD/.venv uv pip install numpy pyarrow

3. **Download vtk.js** (one 2.4 MB file, kept locally so the viewer works offline):

       mkdir -p web/vendor
       curl -fL -o web/vendor/vtk.js https://unpkg.com/vtk.js@37.1.0/vtk.js

4. **Convert the shots for the viewer** (~10 s). This writes one file per shot to `web/data/`:

       .venv/bin/python build_data.py

5. **Open the viewer**, as described in the next section.

This folder already contains the output of steps 3 and 4, so if you only want to look at the results,
skip to the next section.

## View the results

`web/index.html` is a web page, so open it in a browser, not in VS Code (which only shows the code).
No server or internet connection is needed. Any of these works:

- **From the WSL terminal:**

      "/mnt/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe" \
        "file://wsl.localhost/Ubuntu-22.04/home/stef/projects/fusionscope/web/index.html"

- **From the browser:** paste this into the Edge or Chrome address bar:

      file://wsl.localhost/Ubuntu-22.04/home/stef/projects/fusionscope/web/index.html

- **From Windows File Explorer:** go to
  `\\wsl.localhost\Ubuntu-22.04\home\stef\projects\fusionscope\web` and double-click `index.html`.

Add `?shot=mast_shot_28350` (or any other shot id) to the end of the address to open a specific shot.

Once it's open:

- **Pick a shot** with the buttons along the top: three from DIII-D and three from MAST.
- **Animate the discharge** with ▶ at the bottom, or press Space. ←/→ step one frame at a time.
- **Move the 3D view** by dragging to rotate and scrolling to zoom. "Reset camera" puts it back.
- **Inspect a coil** by clicking its row in the "Coil currents" panel to plot its current over time.
- **Jump to a moment** by clicking anywhere on the "Discharge" plot.

## What's shown

- **3D view** – nested flux surfaces (ψN contours via `vtkImageMarchingSquares`, revolved toroidally),
  translucent LCFS shell, optional ψN cross-section on the cut faces, and PF coils coloured by current.
- **Flux terrain** – ψN(R,Z) on the 65×65 EFIT grid as a `vtkWarpScalar` relief; the summit is the
  magnetic axis, the cyan line is the LCFS.
- **Discharge / Coil currents** – Iₚ, q95, βN and a selected coil vs time; per-circuit currents at the current frame.

## Clean up

Everything installed lives in `.venv/` (numpy + pyarrow, uv cache inside it). `rm -rf .venv`
removes it all. vtk.js 37.1.0 is vendored in `web/vendor/`.
