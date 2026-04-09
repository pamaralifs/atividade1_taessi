const state = {
  data: null,
  openIndex: 0,
  objectiveIndex: 0,
};

const modelFriendly = (value) => value || "—";

const createMetaChip = (text) => `<span>${text}</span>`;

const renderTable = (rows, mountId) => {
  const mount = document.getElementById(mountId);
  if (!mount) return;
  if (!rows || !rows.length) {
    mount.innerHTML = `<p class="empty-state">Sem dados para exibir.</p>`;
    return;
  }
  const headers = Object.keys(rows[0]);
  const thead = `<thead><tr>${headers.map((h) => `<th>${h}</th>`).join("")}</tr></thead>`;
  const tbody = `<tbody>${rows.map((row) => `<tr>${headers.map((h) => `<td>${row[h] ?? ""}</td>`).join("")}</tr>`).join("")}</tbody>`;
  mount.innerHTML = `<table>${thead}${tbody}</table>`;
};

const renderMetricCards = (cards, mountId) => {
  const mount = document.getElementById(mountId);
  if (!mount) return;
  mount.innerHTML = cards.map((card) => `
    <article class="panel-card metric-card">
      <div class="metric-title">${card.title}</div>
      <span class="metric-value">${card.value}</span>
      <div class="metric-subtitle">${card.subtitle}</div>
    </article>
  `).join("");
};

const renderChartGallery = (charts, mountId) => {
  const mount = document.getElementById(mountId);
  if (!mount) return;
  mount.innerHTML = charts.map((chart) => `
    <article class="chart-card">
      <button type="button" class="chart-button" data-src="${chart.file}" data-caption="${chart.title}">
        <img src="${chart.file}" alt="${chart.title}" loading="lazy" />
      </button>
      <div class="chart-caption">
        <h4>${chart.title}</h4>
        <p>${chart.description}</p>
      </div>
    </article>
  `).join("");
};

const renderNotes = (notes, mountId) => {
  const mount = document.getElementById(mountId);
  if (!mount) return;
  mount.innerHTML = notes.map((note) => `<li>${note}</li>`).join("");
};

const uniqueSorted = (arr) => ["Todos", ...Array.from(new Set(arr.filter(Boolean))).sort((a, b) => String(a).localeCompare(String(b), 'pt-BR'))];

const fillSelect = (selectId, values) => {
  const select = document.getElementById(selectId);
  if (!select) return;
  select.innerHTML = values.map((value) => `<option value="${value}">${value}</option>`).join("");
};

const attachTabs = () => {
  document.querySelectorAll('.tab-button').forEach((button) => {
    button.addEventListener('click', () => {
      document.querySelectorAll('.tab-button').forEach((b) => b.classList.remove('active'));
      document.querySelectorAll('.tab-panel').forEach((panel) => panel.classList.remove('active'));
      button.classList.add('active');
      document.getElementById(button.dataset.tab).classList.add('active');
    });
  });
};

const attachModal = () => {
  const modal = document.getElementById('imageModal');
  const modalImage = document.getElementById('modalImage');
  const modalCaption = document.getElementById('modalCaption');
  document.body.addEventListener('click', (event) => {
    const button = event.target.closest('.chart-button');
    if (!button) return;
    modalImage.src = button.dataset.src;
    modalCaption.textContent = button.dataset.caption;
    modal.showModal();
  });
  document.getElementById('modalClose').addEventListener('click', () => modal.close());
  modal.addEventListener('click', (event) => {
    if (event.target === modal) modal.close();
  });
};

const valueOrDash = (value) => value || '—';

const renderRecordForm = (mountId, record, fields, badgeConfig) => {
  const mount = document.getElementById(mountId);
  if (!mount) return;
  if (!record) {
    mount.innerHTML = `<p class="empty-state">Nenhum registro encontrado com os filtros atuais.</p>`;
    return;
  }
  const badge = badgeConfig ? `<div class="field full-width"><label>Resultado</label><div class="badge ${badgeConfig.className}">${badgeConfig.text}</div></div>` : '';
  mount.innerHTML = `
    <div class="record-grid">
      ${badge}
      ${fields.map((field) => {
        const isLong = field.type === 'textarea';
        const extraClass = field.compact ? 'small' : '';
        const fullWidth = field.fullWidth ? 'full-width' : '';
        const value = valueOrDash(record[field.key]);
        return `
          <div class="field ${fullWidth} ${extraClass}">
            <label for="${mountId}-${field.key}">${field.label}</label>
            ${isLong
              ? `<textarea id="${mountId}-${field.key}" readonly>${value}</textarea>`
              : `<input id="${mountId}-${field.key}" type="text" readonly value="${String(value).replace(/"/g, '&quot;')}" />`}
          </div>
        `;
      }).join('')}
    </div>
  `;
};

const getOpenFiltered = () => {
  const records = state.data.openExperiment.records;
  const model = document.getElementById('openFilterModel').value;
  const evaluator = document.getElementById('openFilterEvaluator').value;
  const area = document.getElementById('openFilterArea').value;
  const question = document.getElementById('openFilterQuestion').value;
  const search = document.getElementById('openSearch').value.trim().toLowerCase();
  return records.filter((record) => {
    if (model !== 'Todos' && record.ollama_model_display !== model) return false;
    if (evaluator !== 'Todos' && record.evaluator_model !== evaluator) return false;
    if (area !== 'Todos' && record['Área de especialidade'] !== area) return false;
    if (question !== 'Todos' && record.question_id !== question) return false;
    if (search) {
      const haystack = [
        record.question_id,
        record.ollama_model_display,
        record.evaluator_model,
        record['Área de especialidade'],
        record['analise_legislacao'],
        record['analise_argumentacao'],
        record.resposta_modelo,
      ].join(' ').toLowerCase();
      if (!haystack.includes(search)) return false;
    }
    return true;
  });
};

const updateOpenViewer = (resetIndex = false) => {
  const filtered = getOpenFiltered();
  if (resetIndex) state.openIndex = 0;
  if (state.openIndex > filtered.length - 1) state.openIndex = Math.max(filtered.length - 1, 0);
  const record = filtered[state.openIndex];
  document.getElementById('openCounter').textContent = filtered.length ? `Registro ${state.openIndex + 1} de ${filtered.length}` : 'Nenhum registro';
  renderRecordForm('openRecordForm', record, [
    { key: 'id', label: 'ID da linha' },
    { key: 'question_id', label: 'Question ID' },
    { key: 'Numero da OAB', label: 'Exame da OAB' },
    { key: 'ollama_model_display', label: 'Modelo' },
    { key: 'evaluator_model', label: 'Avaliador BERTScore' },
    { key: 'Formato', label: 'Formato' },
    { key: 'grupo_itens', label: 'Grupo de itens' },
    { key: 'Nível de dificuldade', label: 'Nível de dificuldade' },
    { key: 'Área de especialidade', label: 'Área de especialidade' },
    { key: 'bertscore_f1_percent', label: 'F1-Score (%) no CSV' },
    { key: 'comparacao_com_metrica_bertscore', label: 'Comparação com BERTScore', fullWidth: true },
    { key: 'legislacao_referencia', label: 'Legislação de referência', type: 'textarea', fullWidth: true, compact: true },
    { key: 'legislacao_resposta_modelo', label: 'Legislação citada na resposta do modelo', type: 'textarea', fullWidth: true, compact: true },
    { key: 'analise_legislacao', label: 'Análise da legislação', fullWidth: true },
    { key: 'analise_argumentacao', label: 'Análise da argumentação', fullWidth: true },
    { key: 'resposta_modelo', label: 'Resposta do modelo', type: 'textarea', fullWidth: true },
    { key: 'resposta_referencia_limpa_guidelines', label: 'Resposta de referência (guidelines)', type: 'textarea', fullWidth: true },
  ]);
};

const getObjectiveFiltered = () => {
  const records = state.data.objectiveExperiment.records;
  const model = document.getElementById('objectiveFilterModel').value;
  const area = document.getElementById('objectiveFilterArea').value;
  const difficulty = document.getElementById('objectiveFilterDifficulty').value;
  const result = document.getElementById('objectiveFilterResult').value;
  const search = document.getElementById('objectiveSearch').value.trim().toLowerCase();
  return records.filter((record) => {
    if (model !== 'Todos' && record.modelo_display !== model) return false;
    if (area !== 'Todos' && record.area_especialidade !== area) return false;
    if (difficulty !== 'Todos' && record.nivel_dificuldade !== difficulty) return false;
    if (result !== 'Todos' && record.resultado !== result) return false;
    if (search) {
      const haystack = [record.id, record.modelo_display, record.area_especialidade, record.gabarito, record.resposta_modelo, record.enunciado].join(' ').toLowerCase();
      if (!haystack.includes(search)) return false;
    }
    return true;
  });
};

const updateObjectiveViewer = (resetIndex = false) => {
  const filtered = getObjectiveFiltered();
  if (resetIndex) state.objectiveIndex = 0;
  if (state.objectiveIndex > filtered.length - 1) state.objectiveIndex = Math.max(filtered.length - 1, 0);
  const record = filtered[state.objectiveIndex];
  document.getElementById('objectiveCounter').textContent = filtered.length ? `Registro ${state.objectiveIndex + 1} de ${filtered.length}` : 'Nenhum registro';
  renderRecordForm('objectiveRecordForm', record, [
    { key: 'id', label: 'ID da questão' },
    { key: 'modelo_display', label: 'Modelo' },
    { key: 'area_especialidade', label: 'Área de especialidade' },
    { key: 'nivel_dificuldade', label: 'Nível de dificuldade' },
    { key: 'gabarito', label: 'Gabarito' },
    { key: 'resposta_modelo', label: 'Resposta do modelo' },
    { key: 'acertou', label: 'Acertou (0/1)' },
    { key: 'enunciado', label: 'Enunciado', type: 'textarea', fullWidth: true },
  ], record ? { className: record.resultado === 'Acertou' ? 'success' : 'danger', text: record.resultado } : null);
};

const initViewers = () => {
  const openRecords = state.data.openExperiment.records;
  fillSelect('openFilterModel', uniqueSorted(openRecords.map((r) => r.ollama_model_display)));
  fillSelect('openFilterEvaluator', uniqueSorted(openRecords.map((r) => r.evaluator_model)));
  fillSelect('openFilterArea', uniqueSorted(openRecords.map((r) => r['Área de especialidade'])));
  fillSelect('openFilterQuestion', uniqueSorted(openRecords.map((r) => r.question_id)));
  ['openFilterModel', 'openFilterEvaluator', 'openFilterArea', 'openFilterQuestion'].forEach((id) => document.getElementById(id).addEventListener('change', () => updateOpenViewer(true)));
  document.getElementById('openSearch').addEventListener('input', () => updateOpenViewer(true));
  document.getElementById('openPrev').addEventListener('click', () => { state.openIndex = Math.max(0, state.openIndex - 1); updateOpenViewer(); });
  document.getElementById('openNext').addEventListener('click', () => { const filtered = getOpenFiltered(); state.openIndex = Math.min(filtered.length - 1, state.openIndex + 1); updateOpenViewer(); });

  const objectiveRecords = state.data.objectiveExperiment.records;
  fillSelect('objectiveFilterModel', uniqueSorted(objectiveRecords.map((r) => r.modelo_display)));
  fillSelect('objectiveFilterArea', uniqueSorted(objectiveRecords.map((r) => r.area_especialidade)));
  fillSelect('objectiveFilterDifficulty', uniqueSorted(objectiveRecords.map((r) => r.nivel_dificuldade)));
  fillSelect('objectiveFilterResult', ['Todos', 'Acertou', 'Errou']);
  ['objectiveFilterModel', 'objectiveFilterArea', 'objectiveFilterDifficulty', 'objectiveFilterResult'].forEach((id) => document.getElementById(id).addEventListener('change', () => updateObjectiveViewer(true)));
  document.getElementById('objectiveSearch').addEventListener('input', () => updateObjectiveViewer(true));
  document.getElementById('objectivePrev').addEventListener('click', () => { state.objectiveIndex = Math.max(0, state.objectiveIndex - 1); updateObjectiveViewer(); });
  document.getElementById('objectiveNext').addEventListener('click', () => { const filtered = getObjectiveFiltered(); state.objectiveIndex = Math.min(filtered.length - 1, state.objectiveIndex + 1); updateObjectiveViewer(); });

  updateOpenViewer(true);
  updateObjectiveViewer(true);
};

const renderResources = (resources) => {
  const mount = document.getElementById('resourcesList');
  mount.innerHTML = resources.map((resource) => `
    <a class="resource-link" href="${resource.file}" target="_blank" rel="noopener noreferrer">
      <span>${resource.label}</span>
      <span>Abrir ↗</span>
    </a>
  `).join('');
};

const initPage = (data) => {
  state.data = data;
  document.getElementById('heroMeta').innerHTML = [
    `${data.meta.institution}`,
    `${data.meta.program}`,
    `Aluno: ${data.meta.student} · ${data.meta.team}`,
    `Data: ${data.meta.date}`,
  ].map(createMetaChip).join('');

  document.getElementById('openQuestionCount').textContent = data.overview.openQuestionsCount;
  document.getElementById('openRecordCount').textContent = `${data.overview.openRecordsCount} registros consolidados`;
  document.getElementById('objectiveQuestionCount').textContent = data.overview.objectiveQuestionsCount;
  document.getElementById('objectiveRecordCount').textContent = `${data.overview.objectiveRecordsCount} registros consolidados`;
  document.getElementById('openMethodText').textContent = data.overview.openMethod;
  document.getElementById('objectiveMethodText').textContent = data.overview.objectiveMethod;
  document.getElementById('openDatasetText').textContent = data.openExperiment.datasetText;
  document.getElementById('objectiveDatasetText').textContent = data.objectiveExperiment.datasetText;

  const topOpen = data.openExperiment.ranking?.[0] || {};
  renderMetricCards([
    { title: 'Melhor modelo (BERTimbau)', value: topOpen['Modelo'] || '—', subtitle: `F1-Score: ${topOpen['F1-Score (%)'] || '—'}%` },
    { title: 'Avaliadores usados', value: '2', subtitle: 'BERTimbau Large e mBERT' },
    { title: 'Questões analisadas', value: String(data.overview.openQuestionsCount), subtitle: '2 peças profissionais + 8 questões com itens A e B' },
  ], 'openTopCards');

  const topObjective = (data.objectiveExperiment.summary || []).slice().sort((a, b) => parseFloat(String(b['Acurácia (%)']).replace(',', '.')) - parseFloat(String(a['Acurácia (%)']).replace(',', '.')))[0] || {};
  renderMetricCards([
    { title: 'Melhor modelo', value: topObjective['Modelo'] || topObjective['modelo_amigavel'] || '—', subtitle: `Acurácia: ${topObjective['Acurácia (%)'] || topObjective['acc_percent'] || '—'}%` },
    { title: 'Questões avaliadas', value: String(data.overview.objectiveQuestionsCount), subtitle: data.overview.objectiveExams },
    { title: 'Métrica principal', value: 'Acurácia', subtitle: 'Comparação direta com o gabarito oficial' },
  ], 'objectiveTopCards');

  renderTable(data.openExperiment.ranking, 'openRankingTable');
  renderTable(data.openExperiment.byEvaluator, 'openEvaluatorTable');
  renderTable(data.openExperiment.byArea, 'openAreaTable');
  renderTable(data.openExperiment.byDifficulty, 'openDifficultyTable');
  renderTable(data.openExperiment.byType, 'openTypeTable');
  renderChartGallery(data.openExperiment.charts, 'openCharts');
  renderNotes(data.openExperiment.notes, 'openNotes');

  renderTable(data.objectiveExperiment.summary, 'objectiveSummaryTable');
  renderTable(data.objectiveExperiment.topAreas, 'objectiveTopAreasTable');
  renderTable(data.objectiveExperiment.byDifficulty, 'objectiveDifficultyTable');
  renderChartGallery(data.objectiveExperiment.charts, 'objectiveCharts');
  renderNotes(data.objectiveExperiment.notes, 'objectiveNotes');

  renderResources(data.resources);
  attachTabs();
  attachModal();
  initViewers();
};

fetch('data/site-data.json')
  .then((response) => response.json())
  .then(initPage)
  .catch((error) => {
    document.body.innerHTML = `<main class="container section"><div class="panel-card"><h2>Falha ao carregar o site</h2><p>Não foi possível ler o arquivo <code>data/site-data.json</code>.</p><pre>${error}</pre></div></main>`;
  });
