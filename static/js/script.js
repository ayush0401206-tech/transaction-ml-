
const form = document.getElementById('transaction-form');
const result = document.getElementById('result-content');
const button = document.getElementById('analyze-button');
const sampleButton = document.getElementById('sample-button');

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, (character) => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;'
  })[character]);
}

function formatNumber(value, decimals = 2) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(decimals) : '—';
}

function renderExplanations(explanations, note) {
  if (!Array.isArray(explanations) || explanations.length === 0) {
    return '';
  }

  const items = explanations.map((item) => {
    const feature = escapeHtml(item.feature || 'Unknown feature');

    const direction = item.direction === 'toward_suspicious'
      ? 'Increases suspicious signal'
      : item.direction === 'toward_normal'
        ? 'Decreases suspicious signal'
        : 'Influence unavailable';

    const directionClass = item.direction === 'toward_suspicious'
      ? 'suspicious'
      : item.direction === 'toward_normal'
        ? 'normal'
        : '';

    return `
      <li style="display:flex;justify-content:space-between;gap:16px;align-items:flex-start;padding:10px 0;border-bottom:1px solid rgba(127,127,127,.18)">
        <div>
          <strong>${feature}</strong>
          <div style="font-size:12px;opacity:.75;margin-top:3px">${direction}</div>
        </div>
        <strong class="${directionClass}" style="white-space:nowrap">${formatNumber(item.contribution, 4)}</strong>
      </li>`;
  }).join('');

  return `
    <section style="margin-top:24px;text-align:left">
      <h3 style="margin-bottom:8px">Why this result?</h3>
      <p style="font-size:13px;opacity:.8;margin:0 0 10px">
        These are the strongest feature contributions for this prediction.
      </p>
      <ul style="list-style:none;padding:0;margin:0">${items}</ul>
      <p style="font-size:12px;opacity:.7;margin-top:10px">
        ${escapeHtml(note || 'Contribution values are model-output explanations, not percentages or proof of fraud.')}
      </p>
    </section>`;
}

sampleButton.addEventListener('click', () => {
  document.getElementById('timestamp').value = '2025-03-25T09:19';
  document.getElementById('amount').value = '500';
  document.getElementById('merchant-category').value = 'Groceries';
  document.getElementById('device-type').value = 'Android';
  document.getElementById('distance').value = '0.75';
  document.getElementById('average-spending').value = '100';
  document.getElementById('weekend').checked = false;
  document.getElementById('night').checked = false;
});

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  button.disabled = true;
  button.textContent = 'Analyzing…';

  result.innerHTML = `
    <div class="empty-state">
      <div class="signal-icon"><span></span></div>
      <h3>Analyzing transaction</h3>
      <p>Generating prediction and feature explanations.</p>
    </div>`;

  const payload = {
    timestamp: document.getElementById('timestamp').value,
    amount: document.getElementById('amount').value,
    merchant_category: document.getElementById('merchant-category').value,
    device_type: document.getElementById('device-type').value,
    distance_from_home: document.getElementById('distance').value,
    avg_spending_habit: document.getElementById('average-spending').value,
    is_weekend: document.getElementById('weekend').checked,
    is_night_transaction: document.getElementById('night').checked
  };

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || 'Prediction failed.');
    }

    const suspicious = data.prediction === 'Suspicious';
    const normalProbability = Number(data.normal_probability);
    const fraudProbability = Number(data.fraud_probability);
    const confidence = suspicious ? fraudProbability : normalProbability;

    if (!Number.isFinite(normalProbability) ||
        !Number.isFinite(fraudProbability)) {
      throw new Error('The server returned invalid probability values.');
    }

    result.innerHTML = `
      <div style="text-align:center;margin:auto;width:100%;max-width:420px">
        <span class="result-badge ${suspicious ? 'suspicious' : 'normal'}">
          ${suspicious ? 'REVIEW SIGNAL' : 'LOWER RISK SIGNAL'}
        </span>

        <h3 class="result-title">${escapeHtml(data.prediction)} transaction</h3>

        <div class="result-number">
          ${formatNumber(confidence)}
          <span style="font-size:22px">%</span>
        </div>

        <p class="result-copy">Model probability for the predicted class</p>

        <div class="probability-row">
          <span>Normal</span>
          <strong>${formatNumber(normalProbability)}%</strong>
        </div>
        <div class="bar">
          <span style="width:${Math.max(0, Math.min(100, normalProbability))}%"></span>
        </div>

        <div class="probability-row">
          <span>Suspicious</span>
          <strong>${formatNumber(fraudProbability)}%</strong>
        </div>
        <div class="bar">
          <span style="width:${Math.max(0, Math.min(100, fraudProbability))}%"></span>
        </div>

        <p class="disclaimer">${escapeHtml(data.disclaimer || '')}</p>

        ${renderExplanations(
          data.shap_explanations,
          data.explanation_note
        )}
      </div>`;
  } catch (error) {
    result.innerHTML = `
      <div class="error-message">
        ${escapeHtml(error.message || 'Something went wrong.')}
      </div>`;
  } finally {
    button.disabled = false;
    button.innerHTML = 'Analyze transaction <span>→</span>';
  }
});
