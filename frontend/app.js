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
    badge.textContent=h.two_sided_task_authorization?'Task-aware gate online':'Service online'; badge.className='badge '+(h.status==='ok'?'ok':'warn');
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
  }catch(e){msg.textContent='Camera permission was not granted or the camera is unavailable.';}
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

const fallbackReps={
  emotion:['action-units','expression-embedding','protected-video'],
  movement:['landmark-trajectories','motion-features','protected-video'],
  'neurology-motion':['landmark-trajectories','motion-features','protected-video'],
  rppg:['physiological-signal','protected-video'],
  authentication:['cancelable-template'],
  'clinician-visual':['protected-video']
};
async function loadRepresentations(){
  const task=$('#task').value, select=$('#representation'); let reps=fallbackReps[task]||[];
  try{const d=await fetchJSON(`/v1/tasks/${encodeURIComponent(task)}/representations`);if(Array.isArray(d.allowed_representations)&&d.allowed_representations.length)reps=d.allowed_representations}catch{}
  select.innerHTML=''; reps.forEach(rep=>{const o=document.createElement('option');o.value=rep;o.textContent=rep.replaceAll('-',' ');select.appendChild(o)});
}
$('#task').addEventListener('change',loadRepresentations);

function n(id){return Number($(id).value)}
function isoNow(){return new Date().toISOString()}
function isoInMinutes(minutes){return new Date(Date.now()+minutes*60000).toISOString()}
function attackerAucs(){return $('#attackerAucs').value.split(',').map(x=>Number(x.trim())).filter(x=>Number.isFinite(x));}
function requestPayload(){
  const task=$('#task').value;
  const representation=$('#representation').value;
  return {
    contract:{
      session_id:$('#sessionId').value.trim(),
      task,
      purpose:$('#purpose').value.trim(),
      recipient_id:$('#recipientId').value.trim(),
      requested_representation:representation,
      patient_authorized:$('#patientAuthorized').value==='true',
      issued_at:isoNow(),
      expires_at:isoInMinutes(15)
    },
    measured_at:isoNow(),
    max_evidence_age_seconds:300,
    max_identity_advantage:n('#identityAdvantageThreshold'),
    min_task_f1_lower_ci:n('#utilityThreshold'),
    latency_ms_threshold:n('#latencyThreshold'),
    operating_points:[{
      alpha:n('#alpha'),
      clip_auc_ci95_low:n('#clipAucLow'),
      clip_auc_ci95_high:n('#clipAucHigh'),
      repeated_release_auc:n('#repeatedAuc'),
      task_f1_ci95_low:n('#taskF1Low'),
      task_f1_ci95_high:n('#taskF1High'),
      attacker_aucs:attackerAucs(),
      latency_ms:n('#latency'),
      evaluator_id:$('#evaluatorId').value.trim()||'v22-identity-ensemble',
      sample_count:Math.max(1,Math.floor(n('#sampleCount')||1))
    }]
  };
}
function validatePayload(p){
  const o=p.operating_points[0];
  const vals=[p.max_identity_advantage,p.min_task_f1_lower_ci,p.latency_ms_threshold,o.alpha,o.clip_auc_ci95_low,o.clip_auc_ci95_high,o.repeated_release_auc,o.task_f1_ci95_low,o.task_f1_ci95_high,o.latency_ms,...o.attacker_aucs];
  if(vals.some(v=>!Number.isFinite(v)))return 'All numeric evidence must contain valid numbers.';
  if(!p.contract.session_id)return 'Session ID is required.';
  if(!p.contract.purpose)return 'Clinical purpose is required.';
  if(!p.contract.recipient_id)return 'Recipient ID is required.';
  if(!p.contract.requested_representation)return 'Select an authorized representation.';
  if(o.clip_auc_ci95_low>o.clip_auc_ci95_high)return 'Clip AUC CI low cannot exceed CI high.';
  if(o.task_f1_ci95_low>o.task_f1_ci95_high)return 'Task F1 CI low cannot exceed CI high.';
  if(!o.attacker_aucs.length)return 'Provide at least one independent attacker AUC.';
  if(o.attacker_aucs.some(v=>v<0||v>1))return 'Attacker AUCs must be in [0,1].';
  return '';
}
function shortHash(s){return s&&s.length>22?s.slice(0,10)+'…'+s.slice(-8):s||'—'}
function renderDecision(d){
  const release=String(d.decision||d.release_decision||'').toUpperCase()==='RELEASE';
  $('#decisionIcon').textContent=release?'✓':'×';$('#decisionIcon').className='decision-icon '+(release?'ok':'danger');
  $('#decisionTitle').textContent=release?'Release authorized':'Release blocked';
  $('#decisionReason').textContent=d.reason||(release?'Task authorization and all privacy–utility bounds passed.':'Authorization or one of the release conditions failed.');
  $('#selectedAlpha').textContent=d.selected_alpha??'—';
  $('#selectedRep').textContent=d.representation||'—';
  $('#selectedPurpose').textContent=d.purpose||d.attestation?.purpose||'—';
  $('#selectedRecipient').textContent=d.recipient_id||d.attestation?.recipient_id||'—';
  $('#contractHash').textContent=shortHash(d.contract_sha256||d.attestation?.contract_sha256);
  $('#evidenceHash').textContent=shortHash(d.evidence_sha256);
  $('#requestId').textContent=d.request_id||'—';
  $('#riskRingValue').textContent=release?'PASS':'BLOCK';
}
function addHistory(d){const att=d.attestation||{};state.history.unshift({decision:d.decision||att.release_decision||'BLOCK',task:d.task||att.task||'—',representation:d.representation||att.representation||'—',purpose:d.purpose||att.purpose||'—',recipient:d.recipient_id||att.recipient_id||'—',request_id:d.request_id||att.request_id||'—',time:att.timestamp_utc||isoNow()});state.history=state.history.slice(0,12);renderHistory();}
function renderHistory(){const root=$('#attestationList');root.innerHTML='';if(!state.history.length){root.innerHTML='<article class="panel empty">No release evaluations in this session.</article>';return}state.history.forEach(a=>{const card=document.createElement('article');const rel=String(a.decision).toUpperCase()==='RELEASE';card.className='panel attestation-card '+(rel?'release':'block');card.innerHTML=`<span class="attestation-dot" aria-hidden="true"></span><div><strong>${rel?'RELEASE':'BLOCK'} · ${escapeHTML(a.task)}</strong><small>${escapeHTML(a.purpose)} → ${escapeHTML(a.recipient)} · ${escapeHTML(a.representation)}</small></div><time>${new Date(a.time).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'})}</time>`;root.appendChild(card)});}
function escapeHTML(s){return String(s).replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]))}
$('#clearHistory').addEventListener('click',()=>{state.history=[];renderHistory()});

$('#releaseForm').addEventListener('submit',async e=>{
  e.preventDefault();const msg=$('#formMessage');const p=requestPayload(),err=validatePayload(p);if(err){msg.textContent=err;return}
  msg.textContent='Validating doctor request, patient authorization and privacy–utility evidence…';const key=$('#apiKey').value.trim();
  const headers={'X-TAPF-Recipient-ID':p.contract.recipient_id}; if(key)headers.Authorization=`Bearer ${key}`;
  try{const d=await fetchJSON('/v23/authorized-release/evaluate',{method:'POST',headers,body:JSON.stringify(p)});renderDecision(d);addHistory(d);msg.textContent='Task-aware evaluation complete.'}
  catch(ex){renderDecision({decision:'BLOCK',reason:`Service rejected request: ${ex.message}`,representation:p.contract.requested_representation,purpose:p.contract.purpose,recipient_id:p.contract.recipient_id});msg.textContent='Fail-closed: request was not released.'}
});
$('#loadBlocked').addEventListener('click',()=>{
  $('#clipAucLow').value='0.61';$('#clipAucHigh').value='0.69';$('#repeatedAuc').value='0.72';$('#taskF1Low').value='0.24';$('#taskF1High').value='0.32';$('#attackerAucs').value='0.63,0.67,0.59,0.71';$('#latency').value='80';
  $('#formMessage').textContent='Loaded an identity-leakage example. The task may be authorized, but the patient-side firewall must still BLOCK it.';
});

window.addEventListener('beforeinstallprompt',e=>{e.preventDefault();state.installPrompt=e;$('#installBtn').classList.remove('hidden')});
$('#installBtn').addEventListener('click',async()=>{if(!state.installPrompt)return;state.installPrompt.prompt();await state.installPrompt.userChoice;state.installPrompt=null;$('#installBtn').classList.add('hidden')});
if('serviceWorker' in navigator)window.addEventListener('load',()=>navigator.serviceWorker.register('/app/sw.js').catch(()=>{}));

loadRepresentations();refreshHealth();renderHistory();
