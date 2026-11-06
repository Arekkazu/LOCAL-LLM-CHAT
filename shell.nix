{ pkgs ? import <nixpkgs> { config.allowUnfree = true; } }:

let
  cuda = pkgs.cudaPackages;
  # CMake (FindCUDAToolkit) espera nvcc, headers y librerías bajo una misma raíz;
  # llama-cpp-sys-2 enlaza cudart/cublas en estático.
  cudaToolkit = pkgs.symlinkJoin {
    name = "cuda-toolkit-llm-cheap";
    paths = [
      cuda.cuda_nvcc
      cuda.cuda_cudart
      cuda.cccl
      cuda.libcublas.dev
      cuda.libcublas.include
      cuda.libcublas.lib
      cuda.libcublas.static
    ];
    # find_cuda_helper solo busca en <raíz>/lib64.
    postBuild = "ln -s lib $out/lib64";
  };
in
pkgs.mkShell {
  packages = with pkgs; [
    cargo
    clang
    cmake
    rustc
    cudaToolkit
  ];

  nativeBuildInputs = [ pkgs.llvmPackages.libclang ];
  LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";

  CUDA_PATH = "${cudaToolkit}";
  CUDAToolkit_ROOT = "${cudaToolkit}";
  # Raíz donde buscar las librerías estáticas y el stub de libcuda (el real lo da el driver).
  CUDA_LIBRARY_PATH = "${cudaToolkit}";
  # nvcc 12.9 no admite el gcc 15 del stdenv; backendStdenv trae uno compatible.
  CMAKE_CUDA_HOST_COMPILER = "${cuda.backendStdenv.cc}/bin/c++";
  CMAKE_CUDA_COMPILER = "${cudaToolkit}/bin/nvcc";
  # El nvcc.profile de Nix solo incluye sus propios headers; cuBLAS y cccl viven en el join.
  CMAKE_CUDA_FLAGS = "-I${cudaToolkit}/include";
  # Solo la arquitectura de la RTX 2050 (8.6): compila mucho más rápido.
  CMAKE_CUDA_ARCHITECTURES = "86";

  shellHook = ''
    export BINDGEN_EXTRA_CLANG_ARGS="$(
      cat ${pkgs.stdenv.cc}/nix-support/libc-crt1-cflags
      cat ${pkgs.stdenv.cc}/nix-support/libc-cflags
      cat ${pkgs.stdenv.cc}/nix-support/cc-cflags
    )"
    # libcuda.so real del driver de NixOS.
    export LD_LIBRARY_PATH="/run/opengl-driver/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export MODEL_PATH="''${MODEL_PATH:-./models/Qwen3.5-2B-Q4_K_M.gguf}"
    export N_CTX="''${N_CTX:-131072}"
    echo "LLM Cheap listo. Ejecuta: cargo run --release"
  '';
}
