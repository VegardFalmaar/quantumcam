# Schrödinger Equation Webcam Visualizer

An interactive quantum mechanics visualizer that solves the 2D time-dependent Schrödinger equation in real-time on WebGPU compute shaders using your webcam as a quantum potential landscape.

Use your body and environment to create highly irregular potentials, like barriers or wells. It is a toy project, partly agent-coded, and based on my <a href="https://audunsh.github.io/projects/2015-11-19-wavecam/">WaveCam</a> solver for classical waves.

See the live app at <a href="audunsh.github.io/quantumcam">Github Pages</a>


## 🔬 The Physics

The app solves the 2D time-dependent Schrödinger equation:

```
iℏ ∂ψ/∂t = -ℏ²/(2m) ∇²ψ + V(x,y)ψ
```

Where:
- ψ(x,y,t) is the complex wavefunction
- V(x,y) is the potential from your webcam (dark = high potential)
- ℏ is the reduced Planck constant
- m is the particle mass

The webcam intensity creates a quantum potential landscape:
```
V(x,y) = amplitude × (V_raw - offset)
```

## 📝 License

MIT License - Feel free to use, modify, and share!

