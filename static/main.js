const fileInput  = document.getElementById('fileInput');
const uploadZone = document.getElementById('uploadZone');
const loading    = document.getElementById('loading');
const results    = document.getElementById('results');

// Drag and drop
uploadZone.addEventListener('dragover', e => {
  e.preventDefault();
  uploadZone.classList.add('drag-over');
});

uploadZone.addEventListener('dragleave', () => {
  uploadZone.classList.remove('drag-over');
});

uploadZone.addEventListener('drop', e => {
  e.preventDefault();
  uploadZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file && file.name.endsWith('.wav')) {
    processFile(file);
  } else {
    alert('Please drop a .wav file.');
  }
});

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) processFile(fileInput.files[0]);
});

// Store last result for report
let lastResult = null;

function processFile(file) {
  uploadZone.style.display = 'none';
  loading.style.display    = 'block';
  results.style.display    = 'none';

  const formData = new FormData();
  formData.append('file', file);

  fetch('/predict', { method: 'POST', body: formData })
    .then(res => res.json())
    .then(data => {
      loading.style.display = 'none';

      if (data.error) {
        alert('Error: ' + data.error);
        resetUI();
        return;
      }

      // Store result for report download
      lastResult = data;

      // Icon map — all 8 dataset classes
      const iconMap = {
        'BIKE':       '<i class="fas fa-motorcycle"></i>',
        'BUS':        '<i class="fas fa-bus"></i>',
        'CAR':        '<i class="fas fa-car"></i>',
        'CROSSOVER':  '<i class="fas fa-car"></i>',
        'JEEP':       '<i class="fas fa-truck-monster"></i>',
        'MINIBUS':    '<i class="fas fa-bus-alt"></i>',
        'PICKUP':     '<i class="fas fa-truck-pickup"></i>',
        'SPORTS_CAR': '<i class="fas fa-car-side"></i>',
        'TRUCK':      '<i class="fas fa-truck"></i>',
      };

      // Color map — unique color per vehicle
      const colorMap = {
        'BIKE':       '#a855f7',
        'BUS':        '#f97316',
        'CAR':        '#00fff7',
        'CROSSOVER':  '#22c55e',
        'JEEP':       '#84cc16',
        'MINIBUS':    '#fb923c',
        'PICKUP':     '#eab308',
        'SPORTS_CAR': '#ef4444',
        'TRUCK':      '#3b82f6',
      };

      const color = colorMap[data.prediction] || '#00fff7';
      const glow  = `0 0 40px ${color}40`;

      // Set icon
      document.getElementById('resultIcons').innerHTML =
        iconMap[data.prediction] || '<i class="fas fa-car"></i>';

      // Set label and certainty
      document.getElementById('resultLabel').textContent = data.prediction.replace('_', ' ');
      document.getElementById('resultCertainty').textContent =
        data.certainty + '% SIGNAL_MATCH';

      // Apply color to card, label, icon
      const card  = document.getElementById('resultCard');
      const label = document.getElementById('resultLabel');
      const icon  = document.querySelector('.result-icons i');

      card.style.borderColor = color;
      card.style.boxShadow   = glow;
      card.style.background  = `${color}08`;
      label.style.color      = color;
      label.style.textShadow = `0 0 30px ${color}`;
      if (icon) icon.style.color = color;

      // Feature cards
      document.getElementById('f1').textContent =
        data.features.spectral_centroid + ' Hz';
      document.getElementById('f2').textContent =
        data.features.total_energy.toExponential(2) + ' J';
      document.getElementById('f3').textContent =
        (data.features.dominant_freq / 1000).toFixed(2) + ' kHz';
      document.getElementById('f4').textContent =
        data.features.band_ratio.toFixed(3);

      // Graphs
      document.getElementById('waveformImg').src =
        'data:image/png;base64,' + data.waveform;
      document.getElementById('fftImg').src =
        'data:image/png;base64,' + data.fft;

      results.style.display = 'flex';
    })
    .catch(err => {
      loading.style.display = 'none';
      alert('Something went wrong: ' + err);
      resetUI();
    });
}

function resetUI() {
  results.style.display    = 'none';
  loading.style.display    = 'none';
  uploadZone.style.display = 'block';
  fileInput.value = '';
  lastResult = null;
}

function downloadReport() {
  if (!lastResult) return;

  const colorMap = {
    'BIKE':       '#a855f7',
    'BUS':        '#f97316',
    'CAR':        '#00fff7',
    'CROSSOVER':  '#22c55e',
    'JEEP':       '#84cc16',
    'MINIBUS':    '#fb923c',
    'PICKUP':     '#eab308',
    'SPORTS_CAR': '#ef4444',
    'TRUCK':      '#3b82f6',
  };

  const color     = colorMap[lastResult.prediction] || '#00fff7';
  const timestamp = new Date().toLocaleString();
  const label     = lastResult.prediction.replace('_', ' ');

  // Populate the hidden report div
  document.getElementById('rResultBox').style.setProperty('--rcolor', color);
  document.getElementById('rVehicle').textContent   = label;
  document.getElementById('rVehicle').style.color   = color;
  document.getElementById('rCertainty').textContent = lastResult.certainty + '% SIGNAL_MATCH';
  document.getElementById('rf1').textContent = lastResult.features.spectral_centroid + ' Hz';
  document.getElementById('rf2').textContent = lastResult.features.total_energy.toExponential(2) + ' J';
  document.getElementById('rf3').textContent = (lastResult.features.dominant_freq / 1000).toFixed(2) + ' kHz';
  document.getElementById('rf4').textContent = lastResult.features.band_ratio.toFixed(3);
  document.getElementById('rWaveform').src   = 'data:image/png;base64,' + lastResult.waveform;
  document.getElementById('rFft').src        = 'data:image/png;base64,' + lastResult.fft;
  document.getElementById('rTimestamp').textContent = timestamp;

  // Grab printReport HTML
  const reportEl  = document.getElementById('printReport');
  const reportHTML = reportEl.innerHTML;

  // Build a full standalone HTML page
  const fullHTML = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <title>VehiSound Report — ${label}</title>
  <link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Bebas+Neue&display=swap" rel="stylesheet"/>
  <style>
    body {
      margin: 0;
      background: #000;
      color: #fff;
      font-family: 'Share Tech Mono', monospace;
    }
  </style>
</head>
<body>
  ${reportHTML}
</body>
</html>`;

  // Download as .html file
  const blob = new Blob([fullHTML], { type: 'text/html' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href     = url;
  a.download = `vehisound_report_${lastResult.prediction}_${Date.now()}.html`;
  a.click();
  URL.revokeObjectURL(url);
}
