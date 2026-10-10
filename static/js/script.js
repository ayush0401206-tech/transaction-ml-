const form = document.getElementById('transaction-form');
const result = document.getElementById('result-content');
const button = document.getElementById('analyze-button');
const sampleButton = document.getElementById('sample-button');

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
  button.innerHTML = 'Analyzing…';
  result.innerHTML = '<div class="empty-state"><div class="signal-icon"><span></span></div><h3>Analyzing transaction</h3><p>Sending features to the Random Forest model.</p></div>';
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
    const response = await fetch('/predict', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Prediction failed.');
    const suspicious = data.prediction === 'Suspicious';
    const confidence = suspicious ? data.fraud_probability : data.normal_probability;
    result.innerHTML = `
      <div style="text-align:center;margin:auto;width:100%;max-width:300px">
        <span class="result-badge ${suspicious ? 'suspicious' : 'normal'}">${suspicious ? 'REVIEW SIGNAL' : 'LOWER RISK SIGNAL'}</span>
        <h3 class="result-title">${data.prediction} transaction</h3>
        <div class="result-number">${confidence.toFixed(2)}<span style="font-size:22px">%</span></div>
        <p class="result-copy">Model probability for the predicted class</p>
        <div class="probability-row"><span>Normal</span><strong>${data.normal_probability.toFixed(2)}%</strong></div>
        <div class="bar"><span style="width:${data.normal_probability}%"></span></div>
        <div class="probability-row"><span>Suspicious</span><strong>${data.fraud_probability.toFixed(2)}%</strong></div>
        <div class="bar"><span style="width:${data.fraud_probability}%"></span></div>
        <p class="disclaimer">${data.disclaimer}</p>
      </div>`;
  } catch (error) {
    result.innerHTML = `<div class="error-message">${error.message}</div>`;
  } finally {
    button.disabled = false;
    button.innerHTML = 'Analyze transaction <span>→</span>';
  }
});
