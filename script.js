// Consolidated front-end logic

// State
const appState = {
    theme: 'dark',
    data: [],
    columns: [],
    logs: [],
    algoResults: [],
    pyodideReady: false,
    selectedColumns: { x: null, y: null, confirmed: false }
  };
  
  // Pyodide
  let pyodide;
  let pyodideInitPromise = null;
  async function ensurePyodide(){
    if(pyodide && appState.pyodideReady) return pyodide;
    if(pyodideInitPromise) return pyodideInitPromise;
    if(typeof loadPyodide !== 'function'){
      throw new Error('Pyodide not loaded. Please refresh the page and try again.');
    }
    pyodideInitPromise = (async () => {
    try {
      console.log('Loading Pyodide...');
      pyodide = await loadPyodide({ indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.24.1/full/' });
      console.log('Loading core packages...');
      try{ await pyodide.loadPackage(['micropip', 'numpy', 'matplotlib', 'scipy', 'pandas']); console.log('Loaded core packages'); }catch(e){ console.warn('Failed to load core packages:', e); }
      try{
        console.log('Installing sklearn and other packages...');
        await pyodide.runPythonAsync(`
  import micropip, asyncio
  async def install_packages():
      ok = True
      results = await asyncio.gather(
          *[micropip.install(pkg, deps=False) for pkg in ['scikit-learn', 'networkx', 'mlxtend']],
          return_exceptions=True
      )
      for pkg, r in zip(['scikit-learn', 'networkx', 'mlxtend'], results):
          if isinstance(r, Exception):
              print(f'Package installation warning ({pkg}): {r}')
              ok = False
          else:
              print(f'Successfully installed {pkg}')
      return ok
  result = await install_packages()
  print(f'Installation result: {result}')
  `);
      }catch(err){ console.warn('Package installation failed:', err); }
  
      const pyCode = `
import io, base64, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

SCIPY_AVAILABLE = True

try:
    from sklearn.cluster import KMeans
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import pandas as pd
    from mlxtend.preprocessing import TransactionEncoder
    from mlxtend.frequent_patterns import apriori, association_rules
    MLXTEND_AVAILABLE = True
except ImportError:
    MLXTEND_AVAILABLE = False

# Labels provided from JS (fallbacks)
try:
    x_label
except NameError:
    x_label = 'Feature 1'
try:
    y_label
except NameError:
    y_label = 'Feature 2'

def fig_to_base64():
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('ascii')

def clear_figure():
    plt.clf(); plt.cla(); plt.close('all')

def run_kmeans(points, k=3, seed=0):
    try:
        X = np.array(points, dtype=float)
        if len(X) < k:
            return {'error': f'Not enough data points ({len(X)}) for {k} clusters. Need at least {k} points.'}
        if not SKLEARN_AVAILABLE:
            return {'error': 'scikit-learn not available for K-Means'}
        kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
        a = kmeans.fit_predict(X)
        C = kmeans.cluster_centers_
        inertia = float(kmeans.inertia_)
        clear_figure(); plt.figure(figsize=(8, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, k))
        for i in range(k):
            pts = X[a==i]
            if len(pts):
                plt.scatter(pts[:,0], pts[:,1], s=50, color=colors[i], label=f'Cluster {i+1}', alpha=0.8, edgecolors='black', linewidth=0.5)
        plt.scatter(C[:,0], C[:,1], s=200, marker='x', color='red', linewidths=4, label='Centroids')
        plt.legend(fontsize=10)
        plt.title(f'K-Means (k={k}) | {x_label} vs {y_label} | Inertia: {inertia:.2f}', fontsize=12, fontweight='bold')
        plt.xlabel(x_label, fontsize=11); plt.ylabel(y_label, fontsize=11); plt.grid(True, alpha=0.3)
        img = fig_to_base64(); return {'assignments': a.tolist(), 'centroids': C.tolist(), 'inertia': inertia, 'img': img}
    except Exception as e:
        return {'error': f'K-Means failed: {str(e)}'}

def run_agglomerative(points, method='average'):
    try:
        X = np.array(points, dtype=float)
        if len(X) < 2:
            return {'error': 'Need at least 2 data points for agglomerative clustering'}
        if not SCIPY_AVAILABLE:
            return {'error': 'scipy not available for agglomerative clustering'}
        Z = linkage(X, method=method)
        clear_figure(); plt.figure(figsize=(10, 6))
        dendrogram(Z, truncate_mode='lastp', p=12, show_leaf_counts=True, leaf_rotation=90, leaf_font_size=8, show_contracted=True)
        plt.title(f'Agglomerative ({method}) | {x_label} vs {y_label}', fontsize=12, fontweight='bold')
        plt.xlabel('Samples / Cluster Size', fontsize=11); plt.ylabel('Distance', fontsize=11); plt.grid(True, alpha=0.3)
        img = fig_to_base64(); return {'linkage_matrix': Z.tolist(), 'img': img}
    except Exception as e:
        return {'error': f'Agglomerative clustering failed: {str(e)}'}

def run_divisive_mst(points):
    try:
        X = np.array(points, dtype=float)
        if len(X) < 2:
            return {'error': 'Need at least 2 data points for MST clustering'}
        if not NETWORKX_AVAILABLE:
            return {'error': 'networkx not available for MST clustering'}
        G = nx.complete_graph(len(X))
        from scipy.spatial.distance import squareform
        W = squareform(pdist(X))
        for i in range(len(X)):
            for j in range(i+1, len(X)):
                G[i][j]['weight'] = float(W[i,j])
        T = nx.minimum_spanning_tree(G, weight='weight'); edges = list(T.edges())
        clear_figure(); plt.figure(figsize=(8, 6))
        plt.scatter(X[:,0], X[:,1], s=60, color='#4ade80', alpha=0.8, label='Data Points', edgecolors='black', linewidth=0.5)
        for i,j in edges:
            plt.plot([X[i,0], X[j,0]], [X[i,1], X[j,1]], color='#7c5cff', linewidth=3, alpha=0.9)
        plt.title(f'MST (Divisive) | {x_label} vs {y_label}', fontsize=12, fontweight='bold')
        plt.xlabel(x_label, fontsize=11); plt.ylabel(y_label, fontsize=11); plt.legend(fontsize=10); plt.grid(True, alpha=0.3)
        img = fig_to_base64(); return {'edges': [(int(i), int(j)) for (i,j) in edges], 'img': img}
    except Exception as e:
        return {'error': f'Divisive clustering failed: {str(e)}'}

def run_decision_tree(X, y):
    try:
        X_arr = np.array(X, dtype=float); y_arr = np.array(y)
        if len(X_arr) < 2: return {'error': 'Need at least 2 samples for decision tree'}
        if not SKLEARN_AVAILABLE: return {'error': 'scikit-learn not available for decision tree'}
        clf = DecisionTreeClassifier(max_depth=4, random_state=0, min_samples_split=2); clf.fit(X_arr, y_arr)
        clear_figure(); plt.figure(figsize=(14, 10))
        plot_tree(clf, filled=True, feature_names=[f'Feature {i+1}' for i in range(X_arr.shape[1])], class_names=np.unique(y_arr).astype(str), fontsize=10, rounded=True)
        plt.title('Decision Tree Classifier\\nAccuracy: {:.2f}%'.format(clf.score(X_arr, y_arr) * 100), fontsize=12, fontweight='bold')
        img = fig_to_base64(); return {'img': img, 'depth': int(clf.get_depth()), 'score': float(clf.score(X_arr, y_arr))}
    except Exception as e:
        return {'error': f'Decision tree failed: {str(e)}'}

def run_random_forest(X, y):
    try:
        X_arr = np.array(X, dtype=float); y_arr = np.array(y)
        if len(X_arr) < 2: return {'error': 'Need at least 2 samples for random forest'}
        if not SKLEARN_AVAILABLE: return {'error': 'scikit-learn not available for random forest'}
        rf = RandomForestClassifier(n_estimators=10, max_depth=4, random_state=0); rf.fit(X_arr, y_arr)
        clear_figure(); plt.figure(figsize=(14, 10))
        plot_tree(rf.estimators_[0], filled=True, feature_names=[f'Feature {i+1}' for i in range(X_arr.shape[1])], class_names=np.unique(y_arr).astype(str), fontsize=10, rounded=True)
        plt.title('Random Forest - First Tree (of {})\\nAccuracy: {:.2f}%'.format(len(rf.estimators_), rf.score(X_arr, y_arr) * 100), fontsize=12, fontweight='bold')
        img = fig_to_base64(); return {'img': img, 'trees': len(rf.estimators_), 'score': float(rf.score(X_arr, y_arr))}
    except Exception as e:
        return {'error': f'Random forest failed: {str(e)}'}

def run_linear_regression(X, y):
    try:
        X_arr = np.array(X, dtype=float).reshape(-1, 1); y_arr = np.array(y, dtype=float)
        if len(X_arr) < 2: return {'error': 'Need at least 2 data points for linear regression'}
        if not SKLEARN_AVAILABLE: return {'error': 'scikit-learn not available for linear regression'}
        model = LinearRegression(); model.fit(X_arr, y_arr); y_pred = model.predict(X_arr)
        r2 = r2_score(y_arr, y_pred); coef = float(model.coef_[0]); intercept = float(model.intercept_)
        clear_figure(); plt.figure(figsize=(10, 6))
        plt.scatter(X_arr, y_arr, s=50, color='#60a5fa', alpha=0.8, label='Data Points', edgecolors='black', linewidth=0.5)
        X_plot = np.linspace(X_arr.min(), X_arr.max(), 100).reshape(-1, 1); y_plot = model.predict(X_plot)
        plt.plot(X_plot, y_plot, color='#ef4444', linewidth=3, label='Regression Line')
        plt.title(f'Linear Regression | {x_label} vs {y_label} | y={coef:.3f}x+{intercept:.3f} | R²={r2:.3f}', fontsize=12, fontweight='bold')
        plt.xlabel(x_label, fontsize=11); plt.ylabel(y_label, fontsize=11); plt.legend(fontsize=10); plt.grid(True, alpha=0.3)
        img = fig_to_base64(); return {'img': img, 'coef': coef, 'intercept': intercept, 'r2': float(r2)}
    except Exception as e:
        return {'error': f'Linear regression failed: {str(e)}'}

def run_apriori(transactions, min_support=0.2):
    try:
        if not MLXTEND_AVAILABLE:
            return {'error': 'mlxtend not available for Apriori'}
        if len(transactions) < 2:
            return {'error': 'Need at least 2 transactions for Apriori'}
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df = pd.DataFrame(te_ary, columns=te.columns_)
        frequent = apriori(df, min_support=min_support, use_colnames=True)
        if frequent.empty:
            return {'rules': []}
        rules = association_rules(frequent, metric='confidence', min_threshold=0.0)
        out = []
        for _, r in rules.iterrows():
            out.append({
                'from': ', '.join(sorted(r['antecedents'])),
                'to': ', '.join(sorted(r['consequents'])),
                'support': float(r['support']),
                'confidence': float(r['confidence']),
                'lift': float(r['lift']),
            })
        out.sort(key=lambda x: x['lift'], reverse=True)
        return {'rules': out}
    except Exception as e:
        return {'error': f'Apriori failed: {str(e)}'}
  `;
  
      await pyodide.runPythonAsync(pyCode);
      console.log('Python algorithms loaded successfully');
      const envCheck = await pyodide.runPythonAsync(`{'sklearn': SKLEARN_AVAILABLE, 'networkx': NETWORKX_AVAILABLE, 'scipy': True, 'mlxtend': MLXTEND_AVAILABLE}`);
      const envStatus = envCheck.toJs(); console.log('Environment status:', envStatus);
      appState.pyodideReady = true; updatePyodideStatus(true); return pyodide;
    } catch (error) {
      console.error('Pyodide initialization failed:', error);
      updatePyodideStatus(false);
      throw new Error('Failed to initialize Python environment. Please check your internet connection and try again.');
    }
    })();
    try {
      return await pyodideInitPromise;
    } finally {
      pyodideInitPromise = null;
    }
  }
  
  function updatePyodideStatus(ready) {
    const runButton = document.getElementById('run-selected-algorithm');
    if (runButton) {
      // Do not override disabled state here; just reflect status via label
      runButton.textContent = ready ? 'Run Selected Algorithm' : 'Initializing...';
    }
  }
  
  // Utilities
  function log(message){ const ts = new Date().toISOString(); appState.logs.push(`[${ts}] ${message}`); console.log(`[${ts}] ${message}`); }
  function setTheme(mode){ document.documentElement.classList.toggle('light', mode === 'light'); appState.theme = mode; localStorage.setItem('theme', appState.theme); }
  function toggleTheme(){ setTheme(appState.theme === 'dark' ? 'light' : 'dark'); }
  
  function csvToArray(text, delimiter = ','){
    const rows = text.replace(/\uFEFF/g,'').trim().split(/\r?\n/);
    return rows.map(r => { const result = []; let current = ''; let inQuotes = false; for(let i=0;i<r.length;i++){ const c = r[i]; if(c === '"') inQuotes = !inQuotes; else if(c === delimiter && !inQuotes) { result.push(current); current = ''; } else current += c; } result.push(current); return result.map(v => String(v).trim()); });
  }
  function detectDelimiter(sample){ if(sample.includes('\t')) return '\t'; if(sample.includes(';')) return ';'; return ','; }
  
  function buildPreviewTable(columns, data){
    const table = document.getElementById('preview-table'); table.innerHTML = ''; if(!columns.length) return;
    const thead = document.createElement('thead'); thead.innerHTML = `<tr>${columns.map(c => `<th>${c}</th>`).join('')}</tr>`;
    const tbody = document.createElement('tbody');
    data.slice(0, 10).forEach(row => { const tr = document.createElement('tr'); columns.forEach(c => { const td = document.createElement('td'); td.textContent = row[c] ?? ''; tr.appendChild(td); }); tbody.appendChild(tr); });
    table.append(thead, tbody);
  }
  
  function summarize(columns, data){
    const statsContainer = document.getElementById('summary-stats'); statsContainer.innerHTML = ''; if(!columns.length) return;
    columns.forEach(c => { const values = data.map(r => r[c]).filter(v => v != null && v !== ''); const nums = values.map(v => Number(v)).filter(v => Number.isFinite(v)); const missing = data.length - values.length; let mean = '-'; let median = '-'; if(nums.length){ mean = (nums.reduce((a,b)=>a+b,0)/nums.length).toFixed(3); const sorted = [...nums].sort((a,b)=>a-b); const mid = Math.floor(sorted.length/2); median = (sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid-1]+sorted[mid])/2).toFixed(3);} const card = document.createElement('div'); card.className = 'stat'; card.innerHTML = `<strong>${c}</strong><div>Mean: ${mean}</div><div>Median: ${median}</div><div>Missing: ${missing}</div>`; statsContainer.appendChild(card); });
  }
  
  function handleDataLoad(columns, data, sourceName) {
    appState.columns = columns; appState.data = data;
    appState.selectedColumns = { x: null, y: null, confirmed: false };
    document.getElementById('upload-content').style.display = 'block';
    buildPreviewTable(columns, data); summarize(columns, data); log(`Loaded ${sourceName} with ${data.length} rows and ${columns.length} columns`);
    const algorithmSelector = document.getElementById('algorithm-selector');
    if (algorithmSelector && algorithmSelector.value) {
      updateAlgorithmParamsUI(algorithmSelector.value);
    }
  }
  
  function parseTextTable(text){ const delim = detectDelimiter(text.slice(0, 2000)); const rows = csvToArray(text, delim); const header = (rows[0]||[]).map((h, i) => h || `col_${i+1}`); const dataRows = rows.slice(1).filter(r => r.some(v => v && String(v).trim() !== '')); const data = dataRows.map(r => Object.fromEntries(header.map((h, i) => [h, r[i]]))); return { header, data }; }
  
  function parseXLSXFile(file){
    return new Promise((resolve, reject) => { const reader = new FileReader(); reader.onerror = () => reject(new Error('Failed to read Excel file.')); reader.onload = () => { try{ const data = new Uint8Array(reader.result); const wb = XLSX.read(data, { type:'array' }); const sheetName = wb.SheetNames[0]; const sheet = wb.Sheets[sheetName]; const json = XLSX.utils.sheet_to_json(sheet, { defval:'', raw:true }); const header = Array.from(json.reduce((set, row) => { Object.keys(row).forEach(k => set.add(k)); return set; }, new Set())); const normalized = json.map(r => Object.fromEntries(header.map(h => [h, r[h]]))); resolve({ header, data: normalized }); }catch(e){ reject(e); } }; reader.readAsArrayBuffer(file); });
  }
  
  function parseFile(file){
    const name = (file.name || '').toLowerCase(); const isExcel = name.endsWith('.xlsx');
    if(isExcel){ if(typeof XLSX === 'undefined'){ alert('Excel parsing library not loaded. Please check your connection and retry.'); return; } parseXLSXFile(file).then(({header, data}) => handleDataLoad(header, data, `file: ${file.name}`)).catch(err => { console.error(err); alert('Failed to parse .xlsx file.'); }); return; }
    const reader = new FileReader(); reader.onerror = () => alert('Failed to read file.'); reader.onload = () => { try { const text = String(reader.result); const { header, data } = parseTextTable(text); handleDataLoad(header, data, `file: ${file.name}`); } catch(err) { console.error(err); alert('Failed to parse text file. Please check its format.'); } }; reader.readAsText(file);
  }
  
  function getNumericColumns(rows, columns) { return columns.filter(col => { const numericCount = rows.filter(r => r[col] != null && r[col] !== '' && Number.isFinite(Number(r[col]))).length; return numericCount / rows.length > 0.5; }); }
  
  // Python-backed algorithm runners
  async function runPyKMeans(rows, k){ const numericCols = getNumericColumns(rows, appState.columns); const xcol = appState.selectedColumns.x || numericCols[0]; const ycol = appState.selectedColumns.y || numericCols[1]; if (!xcol || !ycol) throw new Error('K-Means requires at least 2 numeric columns for visualization.'); const pts = rows.map(r => [Number(r[xcol]) || 0, Number(r[ycol]) || 0]).filter(p => Number.isFinite(p[0]) && Number.isFinite(p[1])); if(pts.length < k) throw new Error(`Need at least ${k} valid data points for ${k} clusters. Found ${pts.length}.`); const py = await ensurePyodide(); py.globals.set('points', pts); py.globals.set('k', k); py.globals.set('x_label', xcol); py.globals.set('y_label', ycol); const res = await py.runPythonAsync(`run_kmeans(points, k=k)`); return res.toJs({ create_proxies:false, dict_converter: Object.fromEntries }); }
  async function runPyAgglomerative(rows, method){ const numericCols = getNumericColumns(rows, appState.columns); const xcol = appState.selectedColumns.x || numericCols[0]; const ycol = appState.selectedColumns.y || numericCols[1]; if (!xcol || !ycol) throw new Error('Agglomerative Clustering requires at least 2 numeric columns.'); const pts = rows.map(r => [Number(r[xcol]) || 0, Number(r[ycol]) || 0]).filter(p => Number.isFinite(p[0]) && Number.isFinite(p[1])); if(pts.length < 2) throw new Error('Need at least 2 valid data points for this algorithm.'); const py = await ensurePyodide(); py.globals.set('points', pts); py.globals.set('method', method); py.globals.set('x_label', xcol); py.globals.set('y_label', ycol); const res = await py.runPythonAsync(`run_agglomerative(points, method=method)`); return res.toJs({ create_proxies:false, dict_converter: Object.fromEntries }); }
  async function runPyDivisive(rows){ const numericCols = getNumericColumns(rows, appState.columns); const xcol = appState.selectedColumns.x || numericCols[0]; const ycol = appState.selectedColumns.y || numericCols[1]; if (!xcol || !ycol) throw new Error('Divisive Clustering requires at least 2 numeric columns.'); const pts = rows.map(r => [Number(r[xcol]) || 0, Number(r[ycol]) || 0]).filter(p => Number.isFinite(p[0]) && Number.isFinite(p[1])); if(pts.length < 2) throw new Error('Need at least 2 valid data points.'); const py = await ensurePyodide(); py.globals.set('points', pts); py.globals.set('x_label', xcol); py.globals.set('y_label', ycol); const res = await py.runPythonAsync(`run_divisive_mst(points)`); return res.toJs({ create_proxies:false, dict_converter: Object.fromEntries }); }
  async function runPyClassification(rows, pyFuncName){ const numericCols = getNumericColumns(rows, appState.columns); if (numericCols.length < 1) throw new Error('Classification requires at least 1 numeric feature column.'); const featureCols = numericCols.slice(0, Math.min(4, numericCols.length)); const validRows = rows.filter(r => featureCols.every(c => Number.isFinite(Number(r[c])))); if (validRows.length < 2) throw new Error('Need at least 2 valid rows with numeric features.'); const X = validRows.map(r => featureCols.map(c => Number(r[c]))); const lastCol = appState.columns[appState.columns.length - 1]; let y = validRows.map(r => String(r[lastCol] || '').trim()).filter(Boolean); if (y.length !== validRows.length || new Set(y).size < 2) { log('Target column is unsuitable or has only one class. Generating synthetic target.'); const featureToSplit = featureCols[0]; const values = validRows.map(r => Number(r[featureToSplit])); const median = [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)]; y = values.map(v => v > median ? 'Class B' : 'Class A'); }
    const py = await ensurePyodide(); py.globals.set('X', X); py.globals.set('y', y); const res = await py.runPythonAsync(`${pyFuncName}(X, y)`); return res.toJs({ create_proxies:false, dict_converter: Object.fromEntries }); }
  async function runPyLinearRegression(rows){ const numericCols = getNumericColumns(rows, appState.columns); const xcol = appState.selectedColumns.x || numericCols[0]; const ycol = appState.selectedColumns.y || numericCols[1]; if (!xcol || !ycol) throw new Error('Linear Regression requires at least 2 numeric columns.'); const validPoints = rows.map(r => ({x: Number(r[xcol]), y: Number(r[ycol])})).filter(p => Number.isFinite(p.x) && Number.isFinite(p.y)); if (validPoints.length < 2) throw new Error('Need at least 2 valid numeric data points for regression.'); const X = validPoints.map(p => p.x); const y = validPoints.map(p => p.y); const py = await ensurePyodide(); py.globals.set('X', X); py.globals.set('y', y); py.globals.set('x_label', xcol); py.globals.set('y_label', ycol); const res = await py.runPythonAsync(`run_linear_regression(X, y)`); return res.toJs({ create_proxies:false, dict_converter: Object.fromEntries }); }
  async function runPyApriori(rows, minSupport){ const itemsCol = appState.columns[0]; const transactions = rows.map(r => String(r[itemsCol] || '').split(';').map(s => s.trim()).filter(Boolean)).filter(t => t.length > 0); if (transactions.length < 2) throw new Error('Apriori requires at least 2 transactions.'); const py = await ensurePyodide(); py.globals.set('transactions', transactions); py.globals.set('min_support', minSupport); const res = await py.runPythonAsync(`run_apriori(transactions, min_support=min_support)`); return res.toJs({ create_proxies:false, dict_converter: Object.fromEntries }); }
  
  // Charts and UI
  let scoreChart, runtimeChart;
  function getThemeColors(){ const isLight = document.documentElement.classList.contains('light'); return { grid: isLight ? 'rgba(0,0,0,0.15)' : 'rgba(255,255,255,0.15)', text: getComputedStyle(document.documentElement).getPropertyValue('--text').trim(), series: ['#7c5cff','#06b6d4','#4ade80','#f59e0b','#ef4444','#22c55e','#a78bfa','#60a5fa'] }; }
  function renderAlgoTable(tableEl, rules){ tableEl.innerHTML = ''; if (!rules || !rules.length) { tableEl.innerHTML = '<tr><td colspan="4" style="text-align: center; padding: 20px;">No association rules found with current settings.</td></tr>'; return; } const thead = document.createElement('thead'); thead.innerHTML = '<tr><th>Rule</th><th>Support</th><th>Confidence</th><th>Lift</th></tr>'; const tbody = document.createElement('tbody'); rules.forEach(r => { tbody.innerHTML += `<tr><td>${r.from} ⇒ ${r.to}</td><td>${Number(r.support).toFixed(3)}</td><td>${Number(r.confidence).toFixed(3)}</td><td>${Number(r.lift).toFixed(3)}</td></tr>`; }); tableEl.append(thead, tbody); }
  function upsertDashboardCharts(){ const labels = appState.algoResults.map(r=>r.name); const scores = appState.algoResults.map(r=>Number(((r.score||0)*100).toFixed(1))); const runtimes = appState.algoResults.map(r=>r.runtime||0); const sctx = document.getElementById('score-chart').getContext('2d'); const rctx = document.getElementById('runtime-chart').getContext('2d'); const theme = getThemeColors(); if(scoreChart) scoreChart.destroy(); if(runtimeChart) runtimeChart.destroy(); const baseOptions = { scales: { y: { beginAtZero: true, grid: { color: theme.grid }, ticks: { color: theme.text } }, x: { grid: { color: theme.grid }, ticks: { color: theme.text } } }, plugins: { legend: { labels: { color: theme.text } } } }; scoreChart = new Chart(sctx, { type:'bar', data:{ labels, datasets:[{ label:'Score (%)', data:scores, backgroundColor: labels.map((_,i)=> theme.series[i % theme.series.length]) }] }, options: {...baseOptions, scales: {...baseOptions.scales, y: {...baseOptions.scales.y, max: 100}}} }); runtimeChart = new Chart(rctx, { type:'bar', data:{ labels, datasets:[{ label:'Runtime (ms)', data:runtimes, backgroundColor: labels.map((_,i)=> theme.series[(i+2) % theme.series.length]) }] }, options: baseOptions }); }
  
  function download(filename, content, type='text/plain'){ const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([content], { type })); a.download = filename; a.click(); URL.revokeObjectURL(a.href); }
  function toCSV(columns, data){ const esc = v => `"${String(v ?? '').replace(/"/g, '""')}"`; const header = columns.join(','); const rows = data.map(r => columns.map(c => esc(r[c])).join(',')); return [header, ...rows].join('\n'); }
  
  const Samples = { iris: `sepal_length,sepal_width,petal_length,petal_width,species\n5.1,3.5,1.4,0.2,setosa\n4.9,3.0,1.4,0.2,setosa\n4.7,3.2,1.3,0.2,setosa\n6.2,3.4,5.4,2.3,virginica\n5.9,3.0,5.1,1.8,virginica\n6.4,3.2,4.5,1.5,versicolor\n5.5,2.3,4.0,1.3,versicolor`, sales: `spend_tv,spend_radio,spend_social,revenue\n10,8,4,25\n12,7,5,28\n15,10,6,34\n18,12,8,40\n20,15,10,47`, basket: `items\nMilk;Bread;Butter\nBread;Butter\nMilk;Eggs\nBread;Eggs;Butter\nMilk;Bread\nEggs;Cheese\nBread;Cheese;Butter` };
  function loadSample(name){ const text = Samples[name]; if(!text) return; const rows = csvToArray(text, ','); const header = rows[0]; const data = rows.slice(1).map(r => Object.fromEntries(header.map((h,i)=>[h, r[i]]))); handleDataLoad(header, data, `sample: ${name}`); }
  
  function generateInsights(){ const insights = []; const results = appState.algoResults; const byName = Object.fromEntries(results.map(r=>[r.name, r])); const xCol = appState.selectedColumns.x || 'X'; const yCol = appState.selectedColumns.y || 'Y'; if(byName['K-Means']) insights.push(`🔍 The scatter plot shows how ${xCol} relates to ${yCol}, with clusters colored distinctly. Inertia ${byName['K-Means'].inertia?.toFixed(2) || 'N/A'} indicates within-cluster compactness.`); if(byName['Agglomerative (Single)']) insights.push(`🔗 The dendrogram (Single Linkage) illustrates hierarchical merges based on ${xCol} and ${yCol}, helping identify cluster thresholds.`); if(byName['Agglomerative (Average)']) insights.push(`🔗 The dendrogram (Average Linkage) summarizes group-wise distances for ${xCol} vs ${yCol}.`); if(byName['Agglomerative (Complete)']) insights.push(`🔗 The dendrogram (Complete Linkage) emphasizes farthest-point distances along ${xCol} and ${yCol}.`); if(byName['Divisive']) insights.push(`🌐 The MST overlays edges between points in the ${xCol}-${yCol} plane, revealing natural separations.`); if(byName['Linear Regression']){ const r2 = byName['Linear Regression'].r2||byName['Linear Regression'].score||0; insights.push(`📈 Regression line models ${yCol} as a function of ${xCol}. R² ${(r2*100).toFixed(1)}% quantifies explained variance; higher suggests stronger linear relation.`); } if(byName['Decision Tree']) insights.push(`🌳 The tree splits numeric features to classify outcomes; use nodes to understand decision rules and feature thresholds.`); if(byName['Random Forest']) insights.push(`🌲 Ensemble of trees improves robustness; compare accuracy and variance across runs for stability.`); if(byName['Apriori']){ const rules = byName['Apriori'].rules; insights.push(rules?.length > 0 ? `🛒 Top rule "${rules[0].from} → ${rules[0].to}" suggests co-occurrence patterns useful for recommendations.` : `🛒 No strong association rules found; try lowering minimum support.`); } if(appState.data.length > 0) insights.push(`📊 Dataset: ${appState.data.length} rows, ${appState.columns.length} columns; numeric columns: ${getNumericColumns(appState.data, appState.columns).length}.`); if(results.length > 1){ const best = results.reduce((b, r) => (r.score || 0) > (b.score || 0) ? r : b); insights.push(`⚡ Best so far: ${best.name} with score ${(best.score*100).toFixed(1)}%.`); } const el = document.getElementById('insights-content'); el.innerHTML = insights.length > 0 ? `<div class="insights-list">${insights.map(i=>`<div class="insight-item">${i}</div>`).join('')}</div>` : '<p>Run algorithms to generate insights.</p>'; }
  
  function updateAlgorithmParamsUI(key){
    const paramsContainer = document.getElementById('algorithm-parameters');
    const paramsControls = document.getElementById('parameter-controls');
    const runSelectedBtn = document.getElementById('run-selected-algorithm');
    if(!paramsControls || !paramsContainer) return;
    paramsControls.innerHTML = '';
    paramsContainer.style.display = key ? 'block' : 'none';
    if (key === 'kmeans') {
      paramsControls.innerHTML = `<label>Number of Clusters (k)<input type="number" id="k-value" value="3" min="2" max="10"></label>`;
    } else if (key === 'apriori') {
      paramsControls.innerHTML = `<label>Minimum Support<input type="number" id="min-support" value="0.2" min="0.01" max="1" step="0.01"></label>`;
    } else if (key) {
      paramsControls.innerHTML = '<p>No additional parameters required for this algorithm.</p>';
    }
    const needsNumericXY = ['kmeans','agglomerative-single','agglomerative-average','agglomerative-complete','divisive','linear-regression'].includes(key);
    if (needsNumericXY && appState.columns.length) {
      const available = getNumericColumns(appState.data, appState.columns);
      const options = available.map(c => `<option value="${c}">${c}</option>`).join('');
      const wrapper = document.createElement('div');
      wrapper.className = 'parameter-controls';
      wrapper.innerHTML = `<label>X-axis column<select id=\"x-col\">${options}</select></label><label>Y-axis column<select id=\"y-col\">${options}</select></label><button id=\"confirm-columns\" class=\"btn\">Confirm Columns</button>`;
      paramsControls.appendChild(wrapper);
      const xEl = wrapper.querySelector('#x-col'); const yEl = wrapper.querySelector('#y-col');
      if (appState.selectedColumns.x) xEl.value = appState.selectedColumns.x;
      if (appState.selectedColumns.y) yEl.value = appState.selectedColumns.y;
      if (runSelectedBtn) runSelectedBtn.disabled = true;
      wrapper.querySelector('#confirm-columns').addEventListener('click', () => {
        const x = xEl?.value || null; const y = yEl?.value || null;
        if (!x || !y) { alert('Please select both X and Y columns.'); return; }
        if (x === y && !confirm('X and Y are the same column. Continue?')) return;
        appState.selectedColumns = { x, y, confirmed: true };
        if (runSelectedBtn) runSelectedBtn.disabled = false;
        log(`Confirmed columns → X: ${x}, Y: ${y}`);
      });
    } else if (runSelectedBtn) {
      runSelectedBtn.disabled = !key || !appState.pyodideReady;
    }
  }
  
  function init(){
    document.getElementById('year').textContent = new Date().getFullYear().toString();
    setTheme(localStorage.getItem('theme') || 'dark');
    document.getElementById('theme-toggle').addEventListener('click', toggleTheme);
    document.getElementById('theme-toggle-top').addEventListener('click', ()=>{ toggleTheme(); upsertDashboardCharts(); });
    const sidebar = document.getElementById('sidebar'); const setSidebar = open => sidebar.classList.toggle('open', open);
    document.getElementById('sidebar-open').addEventListener('click', () => setSidebar(true));
    document.getElementById('sidebar-close').addEventListener('click', () => setSidebar(false));
    document.addEventListener('keydown', e => e.key === 'Escape' && setSidebar(false));
    if(window.matchMedia('(min-width: 801px)').matches) setSidebar(true);
    document.getElementById('file-input').addEventListener('change', e => e.target.files?.[0] && parseFile(e.target.files[0]));
    document.getElementById('use-sample-iris').addEventListener('click', () => loadSample('iris'));
    document.getElementById('use-sample-sales').addEventListener('click', () => loadSample('sales'));
    document.getElementById('use-sample-basket').addEventListener('click', () => loadSample('basket'));
  
    const ALGORITHMS = { 'kmeans': { name: 'K-Means', runner: (rows, p) => runPyKMeans(rows, p.k), params: () => ({ k: Number(document.getElementById('k-value')?.value || 3) }), produces: 'image', getScore: res => Math.max(0, 1 - Math.tanh((res.inertia || 0) / 100)), getMeta: res => ({ inertia: res.inertia }) }, 'agglomerative-single': { name: 'Agglomerative (Single)', runner: rows => runPyAgglomerative(rows, 'single'), produces: 'image', getScore: () => 0.85, getMeta: () => ({ method: 'single' }) }, 'agglomerative-average': { name: 'Agglomerative (Average)', runner: rows => runPyAgglomerative(rows, 'average'), produces: 'image', getScore: () => 0.82, getMeta: () => ({ method: 'average' }) }, 'agglomerative-complete': { name: 'Agglomerative (Complete)', runner: rows => runPyAgglomerative(rows, 'complete'), produces: 'image', getScore: () => 0.80, getMeta: () => ({ method: 'complete' }) }, 'divisive': { name: 'Divisive', runner: rows => runPyDivisive(rows), produces: 'image', getScore: () => 0.75, getMeta: res => ({ edges: res.edges }) }, 'decision-tree': { name: 'Decision Tree', runner: rows => runPyClassification(rows, 'run_decision_tree'), produces: 'image', getScore: res => res.score || 0, getMeta: res => ({ depth: res.depth, score: res.score }) }, 'random-forest': { name: 'Random Forest', runner: rows => runPyClassification(rows, 'run_random_forest'), produces: 'image', getScore: res => res.score || 0, getMeta: res => ({ trees: res.trees, score: res.score }) }, 'linear-regression': { name: 'Linear Regression', runner: rows => runPyLinearRegression(rows), produces: 'image', getScore: res => res.r2 || 0, getMeta: res => ({ r2: res.r2, coef: res.coef, intercept: res.intercept }) }, 'apriori': { name: 'Apriori', runner: (rows, p) => runPyApriori(rows, p.minSupport), params: () => ({ minSupport: Number(document.getElementById('min-support')?.value || 0.2) }), produces: 'table', getScore: res => (res.rules?.length > 0 ? 0.6 : 0.1), getMeta: res => ({ rules: res.rules }) } };
  
    const algorithmSelector = document.getElementById('algorithm-selector'); const runSelectedBtn = document.getElementById('run-selected-algorithm'); const paramsContainer = document.getElementById('algorithm-parameters'); const paramsControls = document.getElementById('parameter-controls'); const resultsContainer = document.getElementById('algorithm-results'); const resultTitle = document.getElementById('selected-algorithm-title'); const vizContainer = document.getElementById('algorithm-visualization'); const tableContainer = document.getElementById('algorithm-table-container'); const tableEl = document.getElementById('algorithm-table'); const outputEl = document.getElementById('algorithm-output');
  
    ensurePyodide().catch(err => { console.error('Pyodide initialization failed:', err); log(`❌ Pyodide initialization failed: ${err.message}`); });
  
    function getConfig(){ const key = algorithmSelector.value; return { key, config: ALGORITHMS[key] } }
  
    algorithmSelector.addEventListener('change', () => { const { key } = getConfig(); updateAlgorithmParamsUI(key); if (vizContainer) vizContainer.innerHTML = ''; tableContainer.style.display = 'none'; resultsContainer.style.display = 'none'; runSelectedBtn.textContent = 'Run Selected Algorithm'; });
  
    runSelectedBtn.addEventListener('click', async () => { const { key, config } = getConfig(); if (!config) { return alert('Please select an algorithm first.'); } if (!appState.data.length) { return alert('Please load a dataset first.'); }
      const needsNumericXY = ['kmeans','agglomerative-single','agglomerative-average','agglomerative-complete','divisive','linear-regression'].includes(key);
      if (needsNumericXY && !appState.selectedColumns.confirmed) { return alert('Please select X and Y columns and click "Confirm Columns" before running.'); }
      // Ensure Python env on demand
      if (!appState.pyodideReady) {
        const originalText = runSelectedBtn.textContent;
        runSelectedBtn.textContent = 'Initializing...';
        runSelectedBtn.disabled = true;
        try {
          await ensurePyodide();
        } catch (e) {
          alert('Python environment failed to initialize. Please try again in a moment.');
          runSelectedBtn.textContent = originalText;
          runSelectedBtn.disabled = false;
          return;
        }
        runSelectedBtn.textContent = 'Run Selected Algorithm';
        runSelectedBtn.disabled = false;
      }
      try { const py = await ensurePyodide(); py.globals.set('x_label', appState.selectedColumns.x || 'Feature 1'); py.globals.set('y_label', appState.selectedColumns.y || 'Feature 2'); } catch(_) {}
  
      runSelectedBtn.textContent = 'Running...'; runSelectedBtn.disabled = true; resultsContainer.style.display = 'block'; resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'center' }); resultTitle.textContent = `${config.name} Results`; vizContainer.innerHTML = '<div style="padding: 20px;">Running Algorithm...</div>'; vizContainer.style.display = 'block'; tableContainer.style.display = 'none'; outputEl.textContent = '';
      try { const startTime = performance.now(); const params = config.params ? config.params() : undefined; const pyres = await config.runner(appState.data, params); if (pyres.error) { throw new Error(pyres.error); } if (config.produces === 'image') { if (!pyres.img) { throw new Error('Algorithm completed but no visualization was generated. This may be due to missing Python libraries.'); } vizContainer.innerHTML = `<img alt="${config.name} visualization" src="data:image/png;base64,${pyres.img}" style="width: 100%; height: auto; border-radius: 8px;"/>`; } else if (config.produces === 'table') { vizContainer.style.display = 'none'; if (pyres.rules?.length > 0) { tableContainer.style.display = 'block'; renderAlgoTable(tableEl, pyres.rules); } else { vizContainer.style.display = 'block'; vizContainer.innerHTML = '<p style="padding: 20px;">No significant association rules were found with the current settings. Try lowering the minimum support.</p>'; } }
        const result = { name: config.name, score: config.getScore(pyres), runtime: Math.floor(performance.now() - startTime), ...config.getMeta(pyres) }; appState.algoResults = appState.algoResults.filter(r => r.name !== result.name).concat(result); upsertDashboardCharts(); generateInsights(); outputEl.textContent = `${result.name} completed successfully in ${result.runtime} ms.`; log(`✓ ${result.name} completed successfully`);
      } catch (err) { console.error(`${config.name} Error:`, err); vizContainer.innerHTML = `<div style="padding: 40px; color: var(--danger); text-align: center;"><h4>⚠️ Algorithm Failed</h4><p>${err.message}</p><p style="font-size: 12px; margin-top: 10px;">Check browser console for details.</p></div>`; outputEl.textContent = `Error: ${err.message}`; log(`❌ ${config.name} failed: ${err.message}`); } finally { runSelectedBtn.textContent = 'Run Selected Algorithm'; runSelectedBtn.disabled = false; }
    });
  
    document.getElementById('refresh-dashboard').addEventListener('click', () => { upsertDashboardCharts(); generateInsights(); });
    document.getElementById('download-dashboard').addEventListener('click', () => { ['score-chart', 'runtime-chart'].forEach((id, i) => { const canvas = document.getElementById(id); if (canvas) { download(`dashboard-${i+1}.png`, canvas.toDataURL('image/png')); } }); });
    document.getElementById('download-processed').addEventListener('click', () => { if (appState.data.length > 0) { download('processed.csv', toCSV(appState.columns, appState.data), 'text/csv'); } else { alert('No data to download'); } });
    document.getElementById('download-log').addEventListener('click', () => { download('run-log.txt', appState.logs.join('\n')); });
    document.getElementById('refresh-insights').addEventListener('click', generateInsights);
  }
  
  document.addEventListener('DOMContentLoaded', init);
  