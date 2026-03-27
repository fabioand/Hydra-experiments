<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";

type SampleItem = {
  name: string;
  path: string;
};

type ReconstructResponse = {
  width: number;
  height: number;
  original_png_base64: string;
  reconstruction_png_base64: string;
  defaults: { fs: number; fa: number };
  meta: Record<string, string>;
};

const apiBase = (import.meta.env.VITE_API_BASE as string | undefined) ?? "http://localhost:8000";

const inputSize = ref(256);
const fs = ref(0.0);
const fa = ref(0.0);
const isLoading = ref(false);
const statusMsg = ref("Selecione uma imagem local ou um sample do servidor.");
const selectedFile = ref<File | null>(null);
const sampleItems = ref<SampleItem[]>([]);
const selectedSample = ref("");
const isDragging = ref(false);

const originalData = ref<Uint8ClampedArray | null>(null);
const reconstructionData = ref<Uint8ClampedArray | null>(null);
const imageWidth = ref(0);
const imageHeight = ref(0);

const originalCanvas = ref<HTMLCanvasElement | null>(null);
const enhancedCanvas = ref<HTMLCanvasElement | null>(null);

const hasResult = computed(() => !!originalData.value && !!reconstructionData.value && imageWidth.value > 0 && imageHeight.value > 0);

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Erro HTTP ${response.status}`);
  }
  return (await response.json()) as T;
}

async function loadSamples(): Promise<void> {
  try {
    sampleItems.value = await fetchJson<SampleItem[]>(`${apiBase}/v1/samples`);
    if (sampleItems.value.length > 0 && !selectedSample.value) {
      selectedSample.value = sampleItems.value[0].name;
    }
  } catch (err) {
    statusMsg.value = `Nao foi possivel carregar samples: ${String(err)}`;
  }
}

function fileToFormData(file: File): FormData {
  const data = new FormData();
  data.append("file", file, file.name);
  data.append("input_size", String(inputSize.value));
  return data;
}

function base64ToImageSrc(b64: string): string {
  return `data:image/png;base64,${b64}`;
}

async function decodeGrayFromBase64(b64: string): Promise<{ data: Uint8ClampedArray; width: number; height: number }> {
  const image = new Image();
  image.src = base64ToImageSrc(b64);
  await image.decode();

  const canvas = document.createElement("canvas");
  canvas.width = image.width;
  canvas.height = image.height;

  const ctx = canvas.getContext("2d");
  if (!ctx) {
    throw new Error("Nao foi possivel obter contexto 2D");
  }

  ctx.drawImage(image, 0, 0);
  const rgba = ctx.getImageData(0, 0, canvas.width, canvas.height).data;

  const gray = new Uint8ClampedArray(canvas.width * canvas.height);
  for (let i = 0, p = 0; i < gray.length; i += 1, p += 4) {
    gray[i] = rgba[p];
  }

  return { data: gray, width: canvas.width, height: canvas.height };
}

function drawGrayToCanvas(canvas: HTMLCanvasElement | null, gray: Uint8ClampedArray | null, w: number, h: number): void {
  if (!canvas || !gray || w <= 0 || h <= 0) {
    return;
  }

  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    return;
  }

  const rgba = new Uint8ClampedArray(w * h * 4);
  for (let i = 0, p = 0; i < gray.length; i += 1, p += 4) {
    const g = gray[i];
    rgba[p] = g;
    rgba[p + 1] = g;
    rgba[p + 2] = g;
    rgba[p + 3] = 255;
  }

  ctx.putImageData(new ImageData(rgba, w, h), 0, 0);
}

function redrawEnhanced(): void {
  if (!hasResult.value || !originalData.value || !reconstructionData.value || !enhancedCanvas.value) {
    return;
  }

  const total = originalData.value.length;
  const out = new Uint8ClampedArray(total);
  for (let i = 0; i < total; i += 1) {
    const orig = originalData.value[i] / 255.0;
    const recon = reconstructionData.value[i] / 255.0;
    const enhanced = Math.min(1.0, Math.max(0.0, orig - fs.value * recon + fa.value * (orig - recon)));
    out[i] = Math.round(enhanced * 255.0);
  }

  drawGrayToCanvas(enhancedCanvas.value, out, imageWidth.value, imageHeight.value);
}

async function applyResponse(payload: ReconstructResponse): Promise<void> {
  const original = await decodeGrayFromBase64(payload.original_png_base64);
  const recon = await decodeGrayFromBase64(payload.reconstruction_png_base64);

  originalData.value = original.data;
  reconstructionData.value = recon.data;
  imageWidth.value = original.width;
  imageHeight.value = original.height;

  fs.value = payload.defaults.fs;
  fa.value = payload.defaults.fa;

  drawGrayToCanvas(originalCanvas.value, originalData.value, imageWidth.value, imageHeight.value);
  redrawEnhanced();

  statusMsg.value = `Reconstrucao pronta (${payload.meta.source}) em ${payload.width}x${payload.height}.`;
}

async function reconstructFromUpload(): Promise<void> {
  if (!selectedFile.value) {
    statusMsg.value = "Escolha um arquivo antes de enviar.";
    return;
  }

  isLoading.value = true;
  statusMsg.value = "Enviando imagem para o backend...";

  try {
    const payload = await fetchJson<ReconstructResponse>(`${apiBase}/v1/reconstruct`, {
      method: "POST",
      body: fileToFormData(selectedFile.value),
    });
    await applyResponse(payload);
  } catch (err) {
    statusMsg.value = `Falha no upload/reconstrucao: ${String(err)}`;
  } finally {
    isLoading.value = false;
  }
}

async function reconstructFromSample(): Promise<void> {
  if (!selectedSample.value) {
    statusMsg.value = "Selecione um sample do servidor.";
    return;
  }

  isLoading.value = true;
  statusMsg.value = "Rodando reconstrucao do sample no backend...";

  try {
    const payload = await fetchJson<ReconstructResponse>(`${apiBase}/v1/reconstruct/sample`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sample_name: selectedSample.value, input_size: inputSize.value }),
    });
    await applyResponse(payload);
  } catch (err) {
    statusMsg.value = `Falha na reconstrucao do sample: ${String(err)}`;
  } finally {
    isLoading.value = false;
  }
}

function onFileInput(event: Event): void {
  const target = event.target as HTMLInputElement;
  selectedFile.value = target.files?.[0] ?? null;
  if (selectedFile.value) {
    statusMsg.value = `Arquivo selecionado: ${selectedFile.value.name}`;
    if (!isLoading.value) {
      void reconstructFromUpload();
    }
  }
}

function onDrop(event: DragEvent): void {
  event.preventDefault();
  isDragging.value = false;
  const file = event.dataTransfer?.files?.[0] ?? null;
  selectedFile.value = file;
  if (selectedFile.value) {
    statusMsg.value = `Arquivo carregado por drag&drop: ${selectedFile.value.name}`;
    if (!isLoading.value) {
      void reconstructFromUpload();
    }
  }
}

function onSampleChange(): void {
  if (!selectedSample.value || isLoading.value) {
    return;
  }
  void reconstructFromSample();
}

function onDragEnter(event: DragEvent): void {
  event.preventDefault();
  isDragging.value = true;
}

function onDragOver(event: DragEvent): void {
  event.preventDefault();
  isDragging.value = true;
}

function onDragLeave(event: DragEvent): void {
  event.preventDefault();
  isDragging.value = false;
}

function preventWindowDrop(event: DragEvent): void {
  event.preventDefault();
}

function resetSliders(): void {
  fs.value = 0.0;
  fa.value = 0.0;
}

onMounted(async () => {
  window.addEventListener("dragover", preventWindowDrop);
  window.addEventListener("drop", preventWindowDrop);
  await loadSamples();
});

onBeforeUnmount(() => {
  window.removeEventListener("dragover", preventWindowDrop);
  window.removeEventListener("drop", preventWindowDrop);
});

watch([fs, fa], () => {
  redrawEnhanced();
});
</script>

<template>
  <main class="layout">
    <section class="panel controls">
      <div class="brand">
        <img
          src="https://radiomemory.com.br/wp-content/uploads/2015/04/Logo-Rm-Texto-Branco.png"
          alt="Radio Memory"
          class="brand-logo"
        />
      </div>
      <h1>AE Reconstruction Web App</h1>
      <p class="muted">Upload local ou sample do servidor + sliders no cliente.</p>

      <label class="label">Input size do modelo</label>
      <input v-model.number="inputSize" type="number" min="32" max="1024" step="1" />

      <div
        class="dropzone"
        :class="{ 'is-dragging': isDragging }"
        @dragenter="onDragEnter"
        @dragover="onDragOver"
        @dragleave="onDragLeave"
        @drop="onDrop"
      >
        <p>Arraste uma imagem aqui</p>
        <p class="muted">ou</p>
        <input type="file" accept=".jpg,.jpeg,.png,.bmp,.tif,.tiff" @change="onFileInput" />
      </div>

      <hr />

      <label class="label">Sample no servidor</label>
      <select v-model="selectedSample" :disabled="isLoading" @change="onSampleChange">
        <option disabled value="">Selecione...</option>
        <option v-for="item in sampleItems" :key="item.name" :value="item.name">
          {{ item.name }}
        </option>
      </select>

      <hr />

      <label class="label">fs x recon: {{ fs.toFixed(3) }}</label>
      <input v-model.number="fs" type="range" min="0" max="1" step="0.001" />

      <label class="label">fa x (orig-recon): {{ fa.toFixed(3) }}</label>
      <input v-model.number="fa" type="range" min="0" max="1" step="0.001" />

      <button :disabled="!hasResult" @click="resetSliders">Reset Sliders (0.0 / 0.0)</button>

      <p class="status">{{ statusMsg }}</p>
    </section>

    <section class="panel viewer">
      <div class="canvas-card">
        <h3>Original</h3>
        <canvas ref="originalCanvas" />
      </div>
      <div class="canvas-card">
        <h3>Enhanced (sliders)</h3>
        <canvas ref="enhancedCanvas" />
      </div>
    </section>
  </main>
</template>
