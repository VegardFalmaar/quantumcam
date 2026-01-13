// Schrödinger Equation Webcam Visualizer - Standalone Application
// Minimal self-contained quantum mechanics simulator using WebGPU

class QuantumWebcam {
    constructor() {
        this.device = null;
        this.canvas = document.getElementById('simulation-canvas');
        this.context = null;
        this.potentialSource = 'webcam';     // 'webcam' | 'image'
        this.uploadedBitmap = null;          // ImageBitmap
        this.uploadedDirty = false;          // copy-to-texture needed?

        // Simulation parameters
        this.params = {
            width: 640,
            height: 480,
            dt: 0.005,          // Time step (larger = faster evolution)
            dx: 5.2,            // Spatial step (larger = faster propagation)
            waveSpeed: 4.3,     // ℏ/m coefficient
            damping: 0.0,
            sourceEnabled: true,
            sourceFrequency: 0.1,
            sourceStrength: 2.0,
            sourceSize: 2.5,
            boundaryThreshold: 0.25,
            potentialAmplitude: 0.5,
            potentialOffset: 0.0,
            invertBoundaries: false,
            waveAmplitude: 49.6,
            gamma: 3.40,
            probScale: 4991.0,
            sigma: 10.0,
            kx: 0.6,
            ky: 3.9,
            displayMode: 3.0,
            blendMode: 0.0,     // Normal
            mixRatio: 0.40,
            stepsPerFrame: 100
        };

        this.buffers = {};
        this.pipelines = {};
        this.textures = {};

        this.bufferIndex = 0;
        this.time = 0;
        this.frameCount = 0;
        this.isRunning = false;

        this.webcamVideo = null;
        this.webcamReady = false;
    }

    async initialize() {
        try {
            if (!navigator.gpu) {
                throw new Error('WebGPU not supported. Use Chrome 113+ or Edge 113+');
            }

            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) throw new Error('No GPU adapter found');

            this.device = await adapter.requestDevice();

            this.context = this.canvas.getContext('webgpu');
            const format = navigator.gpu.getPreferredCanvasFormat();

            this.context.configure({
                device: this.device,
                format: format,
                alphaMode: 'premultiplied'
            });

            await this.initializeWebcam();
            await this.createBuffers();
            await this.createPipelines();
            await this.initializeWavefunction();

            this.setupUI();

            this.isRunning = true;
            this.animate();

            console.log('✅ Quantum webcam initialized');

        } catch (error) {
            this.showError(error.message);
            console.error('Initialization error:', error);
        }
    }

    async initializeWebcam() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480, facingMode: 'user' }
            });

            this.webcamVideo = document.createElement('video');
            this.webcamVideo.srcObject = stream;
            this.webcamVideo.autoplay = true;
            this.webcamVideo.playsInline = true;

            await new Promise(resolve => {
                this.webcamVideo.onloadedmetadata = () => {
                    this.webcamVideo.play();
                    this.webcamReady = true;
                    resolve();
                };
            });

        } catch (error) {
            console.warn('Webcam not available:', error);
            this.showError('Webcam unavailable - using default potential');
        }
    }

    async createBuffers() {
        const bufferSize = this.params.width * this.params.height * 4;

        // Wavefunction buffers (double buffered for ψ_R and ψ_I)
        this.buffers.psiR = [
            this.device.createBuffer({
                size: bufferSize,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
            }),
            this.device.createBuffer({
                size: bufferSize,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
            })
        ];

        this.buffers.psiI = [
            this.device.createBuffer({
                size: bufferSize,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
            }),
            this.device.createBuffer({
                size: bufferSize,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
            })
        ];

        this.buffers.potential = this.device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });

        this.buffers.params = this.device.createBuffer({
            size: 256,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        this.buffers.vizParams = this.device.createBuffer({
            size: 32,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        this.buffers.renderParams = this.device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        this.textures.output = this.device.createTexture({
            size: [this.params.width, this.params.height],
            format: 'rgba16float',
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING
        });

        this.textures.webcam = this.device.createTexture({
            size: [this.params.width, this.params.height],
            format: 'rgba8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT
        });

        this.textures.upload = this.device.createTexture({
            size: [this.params.width, this.params.height],
            format: 'rgba8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT
        });
    }

    async createPipelines() {
        const schrodingerModule = this.device.createShaderModule({ code: this.getSchrodingerShader() });
        const vizModule = this.device.createShaderModule({ code: this.getVisualizationShader() });
        const potentialModule = this.device.createShaderModule({ code: this.getPotentialShader() });
        const renderModule = this.device.createShaderModule({ code: this.getRenderShader() });

        this.pipelines.schrodinger = this.device.createComputePipeline({
            layout: 'auto',
            compute: { module: schrodingerModule, entryPoint: 'main' }
        });

        this.pipelines.visualization = this.device.createComputePipeline({
            layout: 'auto',
            compute: { module: vizModule, entryPoint: 'main' }
        });

        this.pipelines.potentialExtraction = this.device.createComputePipeline({
            layout: 'auto',
            compute: { module: potentialModule, entryPoint: 'main' }
        });

        this.pipelines.render = this.device.createRenderPipeline({
            layout: 'auto',
            vertex: { module: renderModule, entryPoint: 'vs_main' },
            fragment: {
                module: renderModule,
                entryPoint: 'fs_main',
                targets: [{ format: navigator.gpu.getPreferredCanvasFormat() }]
            },
            primitive: { topology: 'triangle-strip' }
        });
    }

    initializeWavefunction() {
        const real = new Float32Array(this.params.width * this.params.height);
        const imag = new Float32Array(this.params.width * this.params.height);
        const centerX = this.params.width / 2;
        const centerY = this.params.height / 2;

        const sigma = this.params.sigma;
        const sigma2 = sigma * sigma;
        const k0x = this.params.kx * 0.3;
        const k0y = this.params.ky * 0.3;

        for (let y = 0; y < this.params.height; y++) {
            for (let x = 0; x < this.params.width; x++) {
                const idx = y * this.params.width + x;
                const dx = x - centerX;
                const dy = y - centerY;
                const r2 = dx * dx + dy * dy;

                const gaussian = Math.exp(-r2 / (2 * sigma2));
                const phase = k0x * dx + k0y * dy;

                real[idx] = gaussian * Math.cos(phase);
                imag[idx] = gaussian * Math.sin(phase);
            }
        }
        this.device.queue.writeBuffer(this.buffers.psiR[0], 0, real);
        this.device.queue.writeBuffer(this.buffers.psiR[1], 0, real);

        this.device.queue.writeBuffer(this.buffers.psiI[0], 0, imag);
        this.device.queue.writeBuffer(this.buffers.psiI[1], 0, imag);

        console.log(`🌊 Wavepacket initialized using new code: σ=${sigma}, kx=${this.params.kx}, ky=${this.params.ky}`);
    }



drawCoverToCanvas(img, canvas) {
  const ctx = canvas.getContext('2d');
  const cw = canvas.width, ch = canvas.height;
  const iw = img.width, ih = img.height;

  const scale = Math.max(cw / iw, ch / ih);
  const w = iw * scale, h = ih * scale;
  const x = (cw - w) / 2, y = (ch - h) / 2;

  ctx.clearRect(0, 0, cw, ch);
  ctx.drawImage(img, x, y, w, h);
}
    updateUploadTexture() {
      if (!this.uploadedBitmap || !this.uploadedDirty) return;


this.potentialCanvas ??= Object.assign(document.createElement('canvas'), {
  width: this.params.width,
  height: this.params.height
});

this.drawCoverToCanvas(this.uploadedBitmap, this.potentialCanvas);
      this.device.queue.copyExternalImageToTexture(
        { source: this.potentialCanvas },
        { texture: this.textures.upload },
        [this.params.width, this.params.height]
      );

      this.uploadedDirty = false;
    }


    updateWebcamTexture() {
        if (!this.webcamReady || !this.webcamVideo) return;
        this.device.queue.copyExternalImageToTexture(
            { source: this.webcamVideo },
            { texture: this.textures.webcam },
            [this.params.width, this.params.height]
        );
    }

    updateParametersBuffer() {
        const buffer = new ArrayBuffer(256);
        const view = new DataView(buffer);

        let o = 0;
        view.setUint32(o, this.params.width, true); o += 4;      // 0: width
        view.setUint32(o, this.params.height, true); o += 4;     // 4: height
        view.setFloat32(o, this.params.dt, true); o += 4;        // 8: dt
        view.setFloat32(o, this.params.dx, true); o += 4;        // 12: dx
        view.setFloat32(o, this.params.waveSpeed, true); o += 4; // 16: wave_speed
        view.setFloat32(o, this.params.damping, true); o += 4;   // 20: damping
        view.setFloat32(o, this.params.sourceFrequency, true); o += 4; // 24: source_freq
        view.setFloat32(o, this.params.sourceEnabled ? 1.0 : 0.0, false); o += 4; // 28: source_enabled
        view.setFloat32(o, this.time, true); o += 4;             // 32: time
        view.setFloat32(o, this.params.boundaryThreshold, true); o += 4; // 36: boundary_threshold
        view.setFloat32(o, this.params.sourceStrength, true); o += 4;    // 40: source_strength
        view.setFloat32(o, this.params.sourceSize, true); o += 4;        // 44: source_size (for continuous source)
        view.setFloat32(o, 0.0, true); o += 4;                   // 48: normalize_waves
        view.setFloat32(o, 1.0, true); o += 4;                   // 52: wave_range
        view.setFloat32(o, this.params.potentialAmplitude, true); o += 4; // 56: cloth_gravity
        view.setFloat32(o, this.params.kx, true); o += 4;        // 60: cloth_stiffness (kx)
        view.setFloat32(o, this.params.ky, true); o += 4;        // 64: cloth_damping (ky)
        view.setFloat32(o, this.params.gamma, true); o += 4;     // 68: motion_sensitivity
        view.setFloat32(o, this.params.invertBoundaries ? 1.0 : 0.0, true); o += 4; // 72: invert_boundaries
        view.setFloat32(o, 0.0, true); o += 4;                   // 76: fluid_viscosity
        view.setFloat32(o, 0.0, true); o += 4;                   // 80: flow_velocity
        view.setFloat32(o, 0.0, true); o += 4;                   // 84: motion_force
        view.setFloat32(o, 0.0, true); o += 4;                   // 88: density_display
        view.setFloat32(o, 0.0, true); o += 4;                   // 92: flow_direction
        view.setFloat32(o, this.params.potentialOffset, true);   // 96: flow_strength (potential offset)

        this.device.queue.writeBuffer(this.buffers.params, 0, buffer);
    }

    step() {
        this.time += 0.016;
        this.frameCount++;

        this.updateWebcamTexture();
        this.updateUploadTexture();
        this.updateParametersBuffer();

        // Run multiple simulation steps per frame
        // Each step needs its own command encoder to properly swap buffers
        const steps = Math.max(1, Math.min(100, Math.round(this.params.stepsPerFrame)));

        // Log every 60 frames (~1 second)
        // if (this.frameCount % 60 === 0) {
        //     console.log(`⚡ Running ${steps} steps per frame`);
        // }

        // Extract potential once per frame (before simulation steps)
        const potentialEncoder = this.device.createCommandEncoder();
        this.extractPotential(potentialEncoder);
        this.device.queue.submit([potentialEncoder.finish()]);

        // Run each simulation step with buffer swapping
        for (let i = 0; i < steps; i++) {
            const stepEncoder = this.device.createCommandEncoder();
            this.runSchrodinger(stepEncoder);
            this.device.queue.submit([stepEncoder.finish()]);

            // CRITICAL: Swap buffers after each step so next step reads the new data
            this.bufferIndex = 1 - this.bufferIndex;
        }

        // Visualization and render use the final buffer state
        const renderEncoder = this.device.createCommandEncoder();
        this.runVisualization(renderEncoder);
        this.render(renderEncoder);
        this.device.queue.submit([renderEncoder.finish()]);
    }

    extractPotential(encoder) {


  const potentialSrcView =
    (this.potentialSource === 'image' && this.uploadedBitmap)
      ? this.textures.upload.createView()
      : this.textures.webcam.createView();

        const bindGroup = this.device.createBindGroup({
            layout: this.pipelines.potentialExtraction.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: potentialSrcView },
                { binding: 1, resource: { buffer: this.buffers.potential } },
                { binding: 2, resource: { buffer: this.buffers.params } }
            ]
        });

        const pass = encoder.beginComputePass();
        pass.setPipeline(this.pipelines.potentialExtraction);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(this.params.width / 8), Math.ceil(this.params.height / 8));
        pass.end();
    }

    runSchrodinger(encoder) {
        const current = this.bufferIndex;
        const next = 1 - current;

        const bindGroup = this.device.createBindGroup({
            layout: this.pipelines.schrodinger.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.buffers.psiR[current] } },
                { binding: 1, resource: { buffer: this.buffers.psiI[current] } },
                { binding: 2, resource: { buffer: this.buffers.psiR[next] } },
                { binding: 3, resource: { buffer: this.buffers.psiI[next] } },
                { binding: 4, resource: { buffer: this.buffers.potential } },
                { binding: 5, resource: { buffer: this.buffers.params } }
            ]
        });

        const pass = encoder.beginComputePass();
        pass.setPipeline(this.pipelines.schrodinger);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(this.params.width / 8), Math.ceil(this.params.height / 8));
        pass.end();
    }

    runVisualization(encoder) {
        // After the simulation loop, bufferIndex points to the buffer with the NEWEST data
        const current = this.bufferIndex;

        const vizParams = new ArrayBuffer(32);
        const vizView = new DataView(vizParams);
        vizView.setUint32(0, this.params.width, true);
        vizView.setUint32(4, this.params.height, true);
        vizView.setFloat32(8, this.time, true);
        vizView.setFloat32(12, this.params.waveAmplitude, true);
        vizView.setFloat32(16, this.params.gamma, true);
        vizView.setFloat32(20, this.params.displayMode, true);
        vizView.setFloat32(24, this.params.probScale, true);
        vizView.setFloat32(28, 0.0, true);
        this.device.queue.writeBuffer(this.buffers.vizParams, 0, vizParams);

        const bindGroup = this.device.createBindGroup({
            layout: this.pipelines.visualization.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.buffers.psiR[current] } },
                { binding: 1, resource: { buffer: this.buffers.psiI[current] } },
                { binding: 2, resource: this.textures.output.createView() },
                { binding: 3, resource: { buffer: this.buffers.vizParams } }
            ]
        });

        const pass = encoder.beginComputePass();
        pass.setPipeline(this.pipelines.visualization);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(this.params.width / 8), Math.ceil(this.params.height / 8));
        pass.end();
    }

    render(encoder) {


        const backgroundTextureView =
            (this.potentialSource === 'image' && this.uploadedBitmap)
                ? this.textures.upload.createView()
                : this.textures.webcam.createView();

        // Update render params
        const renderParams = new ArrayBuffer(16);
        const renderView = new DataView(renderParams);
        renderView.setFloat32(0, this.params.blendMode, true);
        renderView.setFloat32(4, this.params.mixRatio, true);
        this.device.queue.writeBuffer(this.buffers.renderParams, 0, renderParams);

        const bindGroup = this.device.createBindGroup({
            layout: this.pipelines.render.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: this.textures.output.createView() },
                { binding: 1, resource: backgroundTextureView },
                { binding: 2, resource: this.device.createSampler({ magFilter: 'linear', minFilter: 'linear' }) },
                { binding: 3, resource: { buffer: this.buffers.renderParams } }
            ]
        });

        const pass = encoder.beginRenderPass({
            colorAttachments: [{
                view: this.context.getCurrentTexture().createView(),
                loadOp: 'clear',
                storeOp: 'store'
            }]
        });

        pass.setPipeline(this.pipelines.render);
        pass.setBindGroup(0, bindGroup);
        pass.draw(4);
        pass.end();
    }

    animate() {
        if (!this.isRunning) return;
        this.step();
        requestAnimationFrame(() => this.animate());
    }

    setupUI() {
        document.getElementById('display-mode').addEventListener('change', (e) => {
            this.params.displayMode = parseFloat(e.target.value);
            this.updateModeHint(this.params.displayMode);
        });

        document.getElementById('blend-mode').addEventListener('change', (e) => {
            this.params.blendMode = parseFloat(e.target.value);
        });

        this.setupSlider('wave-amplitude', 'waveAmplitude');
        this.setupSlider('gamma', 'gamma');
        this.setupSlider('prob-scale', 'probScale');
        this.setupSlider('mix-ratio', 'mixRatio');
        this.setupSlider('steps-per-frame', 'stepsPerFrame');
        this.setupSlider('time-step', 'dt');
        this.setupSlider('space-step', 'dx');
        this.setupSlider('source-strength', 'sourceStrength');
        this.setupSlider('source-size', 'sourceSize');
        this.setupSlider('potential-amplitude', 'potentialAmplitude');
        this.setupSlider('potential-offset', 'potentialOffset');
        this.setupSlider('boundary-threshold', 'boundaryThreshold');
        this.setupSlider('sigma', 'sigma');
        this.setupSlider('kx', 'kx');
        this.setupSlider('ky', 'ky');
        this.setupSlider('wave-speed', 'waveSpeed');
        this.setupSlider('damping', 'damping');



        // image upload additions
        const sourceSelect = document.getElementById('potential-source');
        const fileInput = document.getElementById('potential-image');

        sourceSelect?.addEventListener('change', () => {
          this.potentialSource = sourceSelect.value;
        });

        fileInput?.addEventListener('change', async (e) => {
          const file = e.target.files?.[0];
          if (!file) return;

          // Decode image in a GPU-friendly way
          const bitmap = await createImageBitmap(file);
          // Optional: close previous bitmap to free memory
          if (this.uploadedBitmap) this.uploadedBitmap.close();
          this.uploadedBitmap = bitmap;
          this.uploadedDirty = true;

          // If user chose "image", show it immediately
          this.potentialSource = 'image';
          sourceSelect.value = 'image';
        });



        document.getElementById('invert-potential').addEventListener('change', (e) => {
            this.params.invertBoundaries = e.target.checked;
        });

        document.getElementById('source-enabled').addEventListener('change', (e) => {
            this.params.sourceEnabled = e.target.checked;
        });

        document.getElementById('reset-btn').addEventListener('click', () => {
            this.initializeWavefunction();
        });

        this.updateModeHint(this.params.displayMode);
    }

    setupSlider(id, param) {
        const slider = document.getElementById(id);
        const display = document.getElementById(id + '-value');

        if (!slider || !display) {
            console.warn(`⚠️ Slider not found: ${id}`);
            return;
        }

        slider.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            this.params[param] = value;
            const decimals = slider.step.includes('.') ? (slider.step.split('.')[1].length) : 0;
            display.textContent = value.toFixed(decimals);

            // Debug logging for steps per frame
            if (param === 'stepsPerFrame') {
                console.log(`🚀 Steps per frame: ${value}`);
            }
        });
    }

    updateModeHint(mode) {
        const hint = document.getElementById('mode-hint');
        const modes = [
            '<strong>Real Part ψ_R</strong><br>Use Red-Blue colormap<br><span style="color: #aaa;">Shows wave oscillations</span>',
            '<strong>Probability |ψ|²</strong><br>Born interpretation<br><span style="color: #aaa;">Particle detection probability</span>',
            '',
            '<strong>Real + Imaginary Combined</strong><br>🔴 Red = +Re • 🔵 Blue = -Re<br>🟡 Yellow = +Im • 🟢 Cyan = -Im<br><span style="color: #aaa;">Rapid color changes = short wavelength (high momentum)</span>',
            '<strong>Imaginary Part ψ_I</strong><br>Debug view<br><span style="color: #aaa;">Phase component</span>'
        ];

        if (modes[Math.round(mode)]) {
            hint.innerHTML = modes[Math.round(mode)];
        }
    }

    showError(message) {
        const errorEl = document.getElementById('error-message');
        errorEl.textContent = message;
        errorEl.style.display = 'block';
        setTimeout(() => errorEl.style.display = 'none', 5000);
    }

    // Shader code continues in next part...

    // ========== SCHRÖDINGER SHADER ==========
    getSchrodingerShader() {
        return `// Schrödinger equation: iℏ∂ψ/∂t = -ℏ²/(2m)∇²ψ + V(x,y)ψ
struct SimParams {
    width: u32, height: u32, dt: f32, dx: f32,
    wave_speed: f32, damping: f32, source_freq: f32, source_enabled: f32,
    time: f32, boundary_threshold: f32, source_strength: f32, source_size: f32,
    normalize_waves: f32, wave_range: f32,
    cloth_gravity: f32, cloth_stiffness: f32, cloth_damping: f32, motion_sensitivity: f32,
    invert_boundaries: f32, fluid_viscosity: f32, flow_velocity: f32, motion_force: f32,
    density_display: f32, flow_direction: f32, flow_strength: f32
}

@group(0) @binding(0) var<storage, read> psi_R: array<f32>;
@group(0) @binding(1) var<storage, read> psi_I: array<f32>;
@group(0) @binding(2) var<storage, read_write> psi_R_next: array<f32>;
@group(0) @binding(3) var<storage, read_write> psi_I_next: array<f32>;
@group(0) @binding(4) var<storage, read> potential: array<f32>;
@group(0) @binding(5) var<uniform> params: SimParams;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let width = params.width;
    let height = params.height;
    
    if (x >= width || y >= height) { return; }
    
    let idx = y * width + x;
    let hbar_over_m = params.wave_speed;
    let dt = params.dt * 0.5;
    let dx2 = params.dx * params.dx;
    
    let V_raw = potential[idx];
    let V = params.cloth_gravity * (V_raw - params.flow_strength);
    
    let R = psi_R[idx];
    let I = psi_I[idx];
    
    var laplacian_R = 0.0;
    var laplacian_I = 0.0;
    
    // Absorbing boundaries
    let boundary_width = 10.0;
    let min_dist = min(min(f32(x), f32(width - 1 - x)), min(f32(y), f32(height - 1 - y)));
    var absorb = 1.0;
    if (min_dist < boundary_width) {
        absorb = min_dist / boundary_width;
        absorb = absorb * absorb;
    }
    
    // Compute Laplacian
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        laplacian_R = (psi_R[y * width + (x - 1)] + psi_R[y * width + (x + 1)] + 
                       psi_R[(y - 1) * width + x] + psi_R[(y + 1) * width + x] - 4.0 * R) / dx2;
        laplacian_I = (psi_I[y * width + (x - 1)] + psi_I[y * width + (x + 1)] + 
                       psi_I[(y - 1) * width + x] + psi_I[(y + 1) * width + x] - 4.0 * I) / dx2;
    }
    
    let kinetic_coeff = hbar_over_m * 0.5;
    let dR_dt = kinetic_coeff * laplacian_I - V * I;
    let dI_dt = -kinetic_coeff * laplacian_R + V * R;
    
    var R_new = R + dt * dR_dt;
    var I_new = I + dt * dI_dt;
    
    R_new *= absorb;
    I_new *= absorb;
    
    if (params.damping > 0.001) {
        let damping_factor = 1.0 - params.damping * 0.1;
        R_new *= damping_factor;
        I_new *= damping_factor;
    }
    
    // Continuous source
    if (params.source_enabled > 0.5) {
        let center_x = f32(width) * 0.5;
        let center_y = f32(height) * 0.5;
        let dx_src = f32(x) - center_x;
        let dy_src = f32(y) - center_y;
        let dist2 = dx_src * dx_src + dy_src * dy_src;
        
        let sigma = params.source_size * 2.0;
        let sigma2 = sigma * sigma;
        let amplitude = params.source_strength * 0.5;
        
        let kx = params.cloth_stiffness * 0.5;
        let ky = params.cloth_damping * 0.5;
        
        let envelope = amplitude * exp(-dist2 / (2.0 * sigma2));
        let phase = kx * dx_src + ky * dy_src - params.source_freq * params.time * 10.0;
        
        let pulse = sin(params.time * params.source_freq * 2.0 * 3.14159);
        if (pulse > 0.9) {
            R_new += envelope * cos(phase) * 0.1;
            I_new += envelope * sin(phase) * 0.1;
        }
    }
    
    psi_R_next[idx] = R_new;
    psi_I_next[idx] = I_new;
}`;
    }

    // ========== VISUALIZATION SHADER ==========
    getVisualizationShader() {
        return `struct VisualizationParams {
    width: u32, height: u32, time: f32, amplitude_scale: f32,
    gamma: f32, display_mode: f32, prob_scale: f32, _padding: f32
}

@group(0) @binding(0) var<storage, read> psi_R: array<f32>;
@group(0) @binding(1) var<storage, read> psi_I: array<f32>;
@group(0) @binding(2) var output_texture: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<uniform> params: VisualizationParams;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    if (x >= params.width || y >= params.height) { return; }
    
    let idx = y * params.width + x;
    let R_raw = psi_R[idx];
    let I_raw = psi_I[idx];
    let prob_density = R_raw * R_raw + I_raw * I_raw;
    let amp_scale = params.amplitude_scale;
    let display_mode = params.display_mode;
    
    // Mode 3: Direct RGB color output
    if (display_mode > 2.5 && display_mode < 3.5) {
        let R = R_raw * amp_scale * 0.05;
        let I = I_raw * amp_scale * 0.05;
        let gamma = params.gamma;
        
        let R_pos = max(R, 0.0);
        let R_neg = max(-R, 0.0);
        let R_pos_mag = pow(R_pos, gamma);
        let R_neg_mag = pow(R_neg, gamma);
        
        let I_pos = max(I, 0.0);
        let I_neg = max(-I, 0.0);
        let I_pos_mag = pow(I_pos, gamma);
        let I_neg_mag = pow(I_neg, gamma);
        
        var color = vec3<f32>(
            R_pos_mag + I_pos_mag * 0.8,
            I_pos_mag * 0.8 + I_neg_mag * 0.8,
            R_neg_mag + I_neg_mag * 0.8
        );
        
        let prob_glow = sqrt(prob_density) * params.prob_scale * 0.0003;
        color += vec3<f32>(prob_glow);
        
        color = color / (vec3<f32>(1.0) + color);  // Tone mapping
        color = pow(color, vec3<f32>(0.9));
        
        textureStore(output_texture, vec2<i32>(i32(x), i32(y)), vec4<f32>(color, 0.5));
    }
    // Modes 0-2, 4: Scalar output
    else {
        var output_value: f32;
        
        if (display_mode < 0.5) {
            output_value = R_raw * amp_scale;
        } else if (display_mode < 1.5) {
            output_value = prob_density * params.prob_scale;
        } else if (display_mode < 2.5) {
            let magnitude = sqrt(prob_density);
            let phase = atan2(I_raw, R_raw);
            output_value = magnitude * amp_scale * (cos(phase) + sin(phase) * 0.5);
        } else {
            output_value = I_raw * amp_scale;
        }
        
        textureStore(output_texture, vec2<i32>(i32(x), i32(y)), vec4<f32>(output_value, output_value, output_value, 1.0));
    }
}`;
    }

    // ========== POTENTIAL EXTRACTION SHADER ==========
    getPotentialShader() {
        return `struct SimParams {
    width: u32, height: u32, dt: f32, dx: f32,
    wave_speed: f32, damping: f32, source_freq: f32, source_enabled: f32,
    time: f32, boundary_threshold: f32, source_strength: f32, source_size: f32,
    normalize_waves: f32, wave_range: f32,
    cloth_gravity: f32, cloth_stiffness: f32, cloth_damping: f32, motion_sensitivity: f32,
    invert_boundaries: f32, fluid_viscosity: f32, flow_velocity: f32, motion_force: f32,
    density_display: f32, flow_direction: f32, flow_strength: f32
}

@group(0) @binding(0) var webcam_texture: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> potential_data: array<f32>;
@group(0) @binding(2) var<uniform> params: SimParams;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    if (x >= params.width || y >= params.height) { return; }
    
    let idx = y * params.width + x;
    let webcam_color = textureLoad(webcam_texture, vec2<i32>(i32(x), i32(y)), 0).rgb;
    let intensity = dot(webcam_color, vec3<f32>(0.299, 0.587, 0.114));
    
    var potential: f32;
    if (params.invert_boundaries > 0.5) {
        potential = intensity;
    } else {
        potential = 1.0 - intensity;
    }
    
    potential = pow(potential, 2.0);
    potential = potential * params.boundary_threshold;
    
    potential_data[idx] = potential;
}`;
    }

    // ========== RENDER SHADER ==========
    getRenderShader() {
        return `@group(0) @binding(0) var waveTexture: texture_2d<f32>;
@group(0) @binding(1) var webcamTexture: texture_2d<f32>;
@group(0) @binding(2) var textureSampler: sampler;

struct RenderParams {
    blend_mode: f32,
    mix_ratio: f32
}

@group(0) @binding(3) var<uniform> renderParams: RenderParams;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>
}

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
    var output: VertexOutput;
    let x = f32(i32(vertexIndex & 1u) * 2 - 1);
    let y = f32(i32(vertexIndex & 2u) - 1);
    output.position = vec4<f32>(x, y, 0.0, 1.0);
    output.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return output;
}

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let waveSample = textureSample(waveTexture, textureSampler, uv);
    let webcamColor = textureSample(webcamTexture, textureSampler, uv).rgb;
    
    var waveColor: vec3<f32>;
    
    // Check if pre-colored (alpha < 0.75 indicates Mode 3)
    if (waveSample.a < 0.75) {
        waveColor = waveSample.rgb;
    } else {
        // Scalar modes: apply red-blue color scheme
        let value = waveSample.r;
        let t = clamp((value + 1.0) * 0.5, 0.0, 1.0);
        
        if (t < 0.5) {
            let intensity = t * 2.0;
            waveColor = vec3<f32>(intensity, intensity, 1.0);  // Blue to white
        } else {
            let intensity = (t - 0.5) * 2.0;
            waveColor = vec3<f32>(1.0, 1.0 - intensity, 1.0 - intensity);  // White to red
        }
    }
    
    // Apply blending with webcam
    var finalColor: vec3<f32>;
    let mode = renderParams.blend_mode;
    let alpha = renderParams.mix_ratio;
    
    if (mode < 0.5) {
        // Normal blend
        finalColor = mix(waveColor, webcamColor, alpha);
    } else if (mode < 1.5) {
        // Additive (glow)
        finalColor = webcamColor + waveColor * (1.0 - alpha);
    } else if (mode < 2.5) {
        // Subtract
        finalColor = webcamColor - waveColor * (1.0 - alpha);
    } else if (mode < 3.5) {
        // Multiply
        finalColor = mix(webcamColor, webcamColor * waveColor, 1.0 - alpha);
    } else {
        // Screen
        let screen = 1.0 - (1.0 - webcamColor) * (1.0 - waveColor);
        finalColor = mix(webcamColor, screen, 1.0 - alpha);
    }
    
    return vec4<f32>(finalColor, 1.0);
}`;
    }
}

// Initialize app
window.addEventListener('DOMContentLoaded', async () => {
    const app = new QuantumWebcam();
    await app.initialize();
});
