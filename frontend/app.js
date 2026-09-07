const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
const state={history:[],installPrompt:null,cameraStream:null};

function setView(view){
  $$('.view').forEach(v=>v.classList.toggle('active',v.id===view));
  $$('[data-view]').forEach(b=>b.classList.toggle('active',b.dataset.view===view));
  window.scrollTo({top:0,behavior:'smooth'});
}
$$('[data-view]').forEach(b=>b.addEventListener('click',()=>setView(b.dataset.view)));
$$('[data-jump]').forEach(b=>b.addEventListener('click',()=>setView(b.dataset.jump)));

async function fetchJSON(url,options={}){
  const r=await fetch(url,{...options,headers:{'Content-Type':'application/json',...(options.headers||{})}});
  const text=await r.text();
  let data={}; try{data=text?JSON.parse(text):{}}catch{data={detail:text||'Invalid response'}}
  if(!r.ok) throw new Error(typeof data.detail==='string'?data.detail:JSON.stringify(data.detail||data));
  return data;
}

async function refreshHealth(){
  const badge=$('#healthBadge'); badge.textContent='Checking service…'; badge.className='badge neutral';
  try{
    const h=await fetchJSON('/healthz');
    $('#serviceState').textContent=h.status==='ok'?'ONLINE':'DEGRADED';
    badge.textContent=h.status==='ok'?'Service online':'Service degraded'; badge.className='badge '+(h.status==='ok'?'ok':'warn');
    try{await fetchJSON('/readyz');$('#readyState').textContent='READY'}catch{$('#readyState').textContent='NOT READY'}
  }catch(e){
    $('#serviceState').textContent='OFFLINE';$('#readyState').textContent='—';badge.textContent='Service offline';badge.className='badge danger';
  }
}
$('#refreshHealth').addEventListener('click',refreshHealth);

async function startLocalCamera(){
  const msg=$('#cameraMessage'), video=$('#localCamera'), placeholder=$('#cameraPlaceholder');
  if(!navigator.mediaDevices?.getUserMedia){msg.textContent='Camera API is unavailable in this browser.';return;}
  try{
    const stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:'user'},audio:false});
    state.cameraStream=stream; video.srcObject=stream; video.classList.add('active'); placeholder.classList.add('hidden');
    $('#startCamera').disabled=true; $('#stopCamera').disabled=false;
    msg.textContent='Camera active locally. No camera frame is uploaded by this interface.';
  }catch(e){
    msg.textContent='Camera permission was not granted or the camera is unavailable.';
  }
}
function stopLocalCamera(){
  if(state.cameraStream){state.cameraStream.getTracks().forEach(t=>t.stop());state.cameraStream=null;}
  const video=$('#localCamera'); video.srcObject=null; video.classList.remove('active');
  $('#cameraPlaceholder').classList.remove('hidden'); $('#startCamera').disabled=false; $('#stopCamera').disabled=true;
  $('#cameraMessage').textContent='Camera stopped. Frames remain local and are not retained by this interface.';
}
$('#startCamera').addEventListener('click',startLocalCamera);
$('#stopCamera').addEventListener('click',stopLocalCamera);
window.addEventListener('pagehide',stopLocalCamera);

const fallbackReps={expression:['action_units','expression_embedding'],movement:['landmark_trajectories','motion_embedding'],rppg:['physiological_signal'],authentication:['cancelable_template'],visual_exam:['protected_video']};
async function loadRepresentations(){
  const task=$('#task').value, select=$('#representation'); let reps=fallbackReps[task]||[];
  try{const d=await fetchJSON(`/v1/tasks/${encodeURIComponent(task)}/representations`);if(Array.isArray(d.allowed_representations)&&d.allowed_representations.length)reps=d.allowed_representations}catch{}
  select.innerHTML=''; reps.forEach(rep=>{const o=document.createElement('option');o.value=rep;o.textContent=rep.replaceAll('_',' ');select.appendChild(o)});
}
$('#task').addEventListener('change',loadRepresentations);

function n(id){return Number($(id).value)}
function isoNow(){return new Date().toISOString()}
function requestPayload(){return {task:$('#task').value,representation:$('#representation').value,measured_at:isoNow(),max_evidence_age_seconds:300,privacy_upper_threshold:n('#privacyThreshold'),utility_lower_threshold:n('#utilityThreshold'),temporal_upper_threshold:n('#temporalThreshold'),latency_ms_threshold:n('#latencyThreshold'),operating_points:[{alpha:n('#alpha'),identity_risk:n('#identityRisk'),identity_risk_upper:n('#identityUpper'),task_utility:n('#taskUtility'),task_utility_lower:n('#utilityLower'),temporal_risk:n('#temporalRisk'),temporal_risk_upper:n('#temporalUpper'),latency_ms:n('#latency'),evaluator_id:$('#evaluatorId').value.trim()||'edge-evaluator-v1',sample_count:Math.max(1,Math.floor(n('#sampleCount')||1))}]};}
function validatePayload(p){
  const vals=[p.privacy_upper_threshold,p.utility_lower_threshold,p.temporal_upper_threshold,...Object.values(p.operating_points[0]).filter(v=>typeof v==='number')];
  if(vals.some(v=>!Number.isFinite(v)))return 'All numeric fields must contain valid numbers.';
  if(!p.representation)return 'Select an authorized representation.';
  if(p.operating_points[0].identity_risk_upper<p.operating_points[0].identity_risk)return 'Identity upper bound cannot be below point estimate.';
  if(p.operating_points[0].task_utility_lower>p.operating_points[0].task_utility)return 'Utility lower bound cannot exceed point estimate.';
  if(p.operating_points[0].temporal_risk_upper<p.operating_points[0].temporal_risk)return 'Temporal upper bound cannot be below point estimate.';
  return '';
}
function shortHash(s){return s&&s.length>22?s.slice(0,10)+'…'+s.slice(-8):s||'—'}
function renderDecision(d){
  const release=String(d.decision||d.release_decision||'').toUpperCase()==='RELEASE';
  $('#decisionIcon').textContent=release?'✓':'×';$('#decisionIcon').className='decision-icon '+(release?'ok':'danger');
  $('#decisionTitle').textContent=release?'Release authorized':'Release blocked';
  $('#decisionReason').textContent=d.reason||(release?'All configured evidence bounds passed.':'One or more release conditions failed.');
  $('#selectedAlpha').textContent=d.selected_alpha??'—';$('#selectedRep').textContent=d.representation||'—';$('#evidenceHash').textContent=shortHash(d.evidence_sha256);$('#requestId').textContent=d.request_id||'—';$('#riskRingValue').textContent=release?'PASS':'BLOCK';
}
function addHistory(d){const att=d.attestation||{};state.history.unshift({decision:d.decision||att.release_decision||'BLOCK',task:d.task||att.task||'—',representation:d.representation||att.representation||'—',request_id:d.request_id||att.request_id||'—',time:att.timestamp_utc||isoNow()});state.history=state.history.slice(0,12);renderHistory();}
function renderHistory(){const root=$('#attestationList');root.innerHTML='';if(!state.history.length){root.innerHTML='<article class="panel empty">No release evaluations in this session.</article>';return}state.history.forEach(a=>{const card=document.createElement('article');const rel=String(a.decision).toUpperCase()==='RELEASE';card.className='panel attestation-card '+(rel?'release':'block');card.innerHTML=`<span class="attestation-dot" aria-hidden="true"></span><div><strong>${rel?'RELEASE':'BLOCK'} · ${escapeHTML(a.task)}</strong><small>${escapeHTML(a.representation)} · ${escapeHTML(a.request_id)}</small></div><time>${new Date(a.time).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'})}</time>`;root.appendChild(card)});}
function escapeHTML(s){return String(s).replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]))}
$('#clearHistory').addEventListener('click',()=>{state.history=[];renderHistory()});

$('#releaseForm').addEventListener('submit',async e=>{e.preventDefault();const msg=$('#formMessage');const p=requestPayload(),err=validatePayload(p);if(err){msg.textContent=err;return}msg.textContent='Evaluating measured evidence…';const key=$('#apiKey').value.trim();try{const d=await fetchJSON('/v2/release/evaluate',{method:'POST',headers:key?{Authorization:`Bearer ${key}`}:{},body:JSON.stringify(p)});renderDecision(d);addHistory(d);msg.textContent='Evaluation complete.'}catch(ex){renderDecision({decision:'BLOCK',reason:`Service rejected request: ${ex.message}`,representation:p.representation});msg.textContent='Fail-closed: request was not released.'}});
$('#loadBlocked').addEventListener('click',()=>{$('#identityRisk').value='0.88';$('#identityUpper').value='0.94';$('#taskUtility').value='0.82';$('#utilityLower').value='0.76';$('#temporalRisk').value='0.73';$('#temporalUpper').value='0.86';$('#latency').value='95';$('#formMessage').textContent='Loaded a high identity-leakage example. Run the gate to confirm BLOCK.';});

window.addEventListener('beforeinstallprompt',e=>{e.preventDefault();state.installPrompt=e;$('#installBtn').classList.remove('hidden')});
$('#installBtn').addEventListener('click',async()=>{if(!state.installPrompt)return;state.installPrompt.prompt();await state.installPrompt.userChoice;state.installPrompt=null;$('#installBtn').classList.add('hidden')});
if('serviceWorker' in navigator)window.addEventListener('load',()=>navigator.serviceWorker.register('/app/sw.js').catch(()=>{}));

loadRepresentations();refreshHealth();renderHistory();
