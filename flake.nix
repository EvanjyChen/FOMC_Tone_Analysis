{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { nixpkgs, ... }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-darwin"
      ];
      forAllSystems =
        f:
        nixpkgs.lib.genAttrs systems (
          system:
          f (
            import nixpkgs {
              inherit system;
            }
          )
        );
    in
    {
      devShells = forAllSystems (pkgs: {
        default = pkgs.mkShell {
          packages = [
            pkgs.python314
            pkgs.uv # Package manager
            pkgs.pyright # Type checker
            pkgs.ruff # Linter/Formatter

            # Convenience scripts
            (pkgs.writeShellScriptBin "rf" ''
              exec ruff format "$@"
            '')
            (pkgs.writeShellScriptBin "rc" ''
              exec ruff check --fix "$@"
            '')
            (pkgs.writeShellScriptBin "rfc" ''
              set -euo pipefail
              rf "$@"
              rc "$@"
            '')
            # Use the dependencies below when exporting a jupyter notebook as a
            # PDF
            # ------------------------------------------------------------------
            # pkgs.pandoc
            # (pkgs.texliveBasic.withPackages (
            #   ps: with ps; [
            #     scheme-medium
            #     # extra packages nbconvert's LaTeX template pulls in
            #     tcolorbox
            #     environ
            #     pdfcol
            #     tikzfill
            #     adjustbox
            #     collectbox
            #     ucs
            #     upquote
            #     ulem
            #     rsfs
            #     titling
            #     enumitem
            #     jknapltx
            #     parskip
            #     pgf
            #     eurosym
            #     trimspaces
            #   ]
            # ))
          ];

          env = {
            # uv venvs reuse the nix python instead of downloading one.
            UV_PYTHON = pkgs.python314.interpreter;
            UV_PYTHON_DOWNLOADS = "never";
          }
          // pkgs.lib.optionalAttrs pkgs.stdenv.isLinux {
            # manylinux wheels (cuml-cu12, torch, ...) dlopen libs that
            # aren't on nix python's default search path; the last entry
            # is the NVIDIA driver's libcuda.so on NixOS.
            LD_LIBRARY_PATH =
              pkgs.lib.makeLibraryPath [
                pkgs.stdenv.cc.cc
                pkgs.zlib
              ]
              + ":/run/opengl-driver/lib";
          };

          shellHook = ''
            [ -d .venv ] || uv venv
            source .venv/bin/activate

            # Register a named Jupyter kernel inside the venv itself
            # (.venv/share/jupyter) 

            kernel_json="$PWD/.venv/share/jupyter/kernels/fomc_tone_analysis/kernel.json"
            if python -c "import ipykernel" >/dev/null 2>&1 \
               && ! grep -qF "$PWD/.venv/bin/python3" "$kernel_json" 2>/dev/null; then
              python -m ipykernel install --prefix "$PWD/.venv" \
                --name fomc_tone_analysis --display-name "Python (fomc_tone_analysis uv)" >/dev/null
            fi
          '';
        };
      });
    };
}
