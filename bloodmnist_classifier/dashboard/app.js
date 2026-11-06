'use strict';
const $=id=>document.getElementById(id), colors=['#6ce3c4','#b39bff','#f3bb71'];
const esc=value=>String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pct=x=>Number.isFinite(x)?(x*100).toFixed(2)+'%':'—';
const pretty=c=>c.replaceAll('_',' ');
let evaluation=null, classes=[];
async function read(name){const response=await fetch('data/'+name+'?t='+Date.now(),{cache:'no-store'});return response.ok?response.json():null;}
function chart(id,rows,keys,fixed=false){
  if(!rows?.length)return;
  const values=keys.map(k=>rows.map(r=>Number(r[k])));
  if(values.flat().some(v=>!Number.isFinite(v)))return;
  const maximum=fixed?1:Math.max(...values.flat(),1e-8)*1.08;
  const x=i=>52+i*507/Math.max(rows.length-1,1),y=v=>190-v/maximum*166;
  const format=v=>maximum<.01?v.toExponential(1):v.toFixed(2);
  $(id).className='';
  $(id).innerHTML=`<svg viewBox="0 0 580 222" role="img" aria-label="${esc(keys.join(' and '))} by epoch">`+
    [0,.25,.5,.75,1].map(t=>`<line x1="52" x2="560" y1="${y(maximum*t)}" y2="${y(maximum*t)}" stroke="#283547"/><text x="2" y="${y(maximum*t)+4}" fill="#9aaec5" font-size="10">${format(maximum*t)}</text>`).join('')+
    values.map((series,j)=>`<polyline fill="none" stroke="${colors[j]}" stroke-width="2.4" points="${series.map((v,i)=>`${x(i)},${y(v)}`).join(' ')}"/>`+(series.length===1?`<circle cx="${x(0)}" cy="${y(series[0])}" r="3" fill="${colors[j]}"/>`:'')).join('')+
    `<text x="52" y="216" fill="#9aaec5" font-size="10">Epoch ${esc(rows[0].epoch)}</text><text x="510" y="216" fill="#9aaec5" font-size="10">${esc(rows.at(-1).epoch)}</text></svg>`;
}
function matrix(){
  if(!evaluation)return;
  const normalized=$('normalize').checked, cm=evaluation[normalized?'normalized_confusion_matrix':'confusion_matrix'];
  const max=Math.max(...cm.flat(),1e-8);
  $('matrix').innerHTML='<table class="matrix"><thead><tr><th>True ↓ / Predicted →</th>'+classes.map(c=>`<th>${esc(pretty(c))}</th>`).join('')+'</tr></thead><tbody>'+cm.map((row,i)=>'<tr><th>'+esc(pretty(classes[i]))+'</th>'+row.map((n,j)=>`<td title="${esc(pretty(classes[i]))} → ${esc(pretty(classes[j]))}" style="background:rgba(108,227,196,${.04+.65*n/max});color:${n/max>.6?'#0d131c':'#e8eef7'}">${normalized?(n*100).toFixed(1)+'%':n}</td>`).join('')+'</tr>').join('')+'</tbody></table>';
}
async function refresh(){
  try{
    const [d,s,h,p,e,c,start]=await Promise.all(['dataset.json','status.json','history.json','progress.json','evaluation.json','model_config.json','startup.json'].map(read));
    $('message').textContent=d?'':'Dataset audit unavailable. Run audit_dataset.py first.';
    if(d){
      classes=d.classes;$('total').textContent=d.total_images.toLocaleString();$('dimensions').textContent=Object.keys(d.dimensions).join(', ').replaceAll('x',' × ');
      $('formats').textContent=Object.keys(d.formats).join(', ')+' · '+Object.keys(d.modes).join(', ');$('duplicates').textContent=d.cross_split_pixel_duplicates.length;
      const maximum=Math.max(...Object.values(d.class_totals));
      $('distribution').className='';$('distribution').innerHTML=classes.map(cls=>`<div class="dist-row"><span>${esc(pretty(cls))}</span><div class="dist-bar">`+['train','val','test'].map((split,i)=>`<span style="width:${100*d.counts[split][cls]/maximum}%;background:${colors[i]}"></span>`).join('')+`</div><span>${d.class_totals[cls]}</span></div>`).join('');
      $('split-summary').className='';$('split-summary').innerHTML=Object.entries(d.split_totals).map(([name,n])=>`<div class="split-row"><span>${esc(name==='val'?'Validation':name[0].toUpperCase()+name.slice(1))} <small>· ${pct(n/d.total_images)}</small></span><strong>${n.toLocaleString()}</strong></div>`).join('');
      $('integrity').textContent=`${d.corrupt_files.length} corrupt files · ${d.cross_split_pixel_duplicates.length} cross-split duplicate groups · ${d.conflicting_label_duplicate_groups?.length||0} duplicate groups with conflicting labels. Loader exclusions are recorded in the training configuration.`;
      $('balance').textContent=`Largest/smallest training class ratio: ${d.train_imbalance_ratio.toFixed(2)}. Original split assignments retained. Donor independence is unknown.`;
    }
    if(s){$('status').textContent=s.status;$('status').className='badge'+(s.status==='error'?' error':'');$('status-note').textContent=s.note||'';
      $('sync-note').textContent=s.synced_at?'Last Kaggle status check: '+new Date(s.synced_at).toLocaleString()+' · Page refresh reads local files.':'No Kaggle status check recorded. Page refresh reads local files.';
      $('observed').textContent=s.metrics_observed_at?'Last epoch log record received: '+new Date(s.metrics_observed_at).toLocaleString():'';
      if(s.kernel_url){const u=new URL(s.kernel_url);if(u.origin==='https://www.kaggle.com'){$('kaggle').href=u.href;$('kaggle').hidden=false;}}
    }
    const last=p?.epoch?p:h?.at(-1);
    if(last){$('epoch').textContent=last.epoch+' / '+(last.planned_epochs||40);$('gpu').textContent=last.gpu||start?.gpu||'Unknown';$('elapsed').textContent=(last.elapsed_seconds/60).toFixed(1)+' min';$('best').textContent=p?.best_epoch||e?.best_epoch||'—';$('progress').style.width=Math.min(100,100*last.epoch/(last.planned_epochs||40))+'%';}
    if(start?.cuda_available===false)$('gpu').textContent='No CUDA GPU — training blocked';
    else if(start?.gpu)$('gpu').textContent=start.gpu+(start.gpu_execution_verified?' · CUDA verified':'');
    if(h?.length){chart('loss',h,['train_loss','val_loss']);chart('accuracy',h,['train_accuracy','val_accuracy'],true);chart('f1',h,['val_macro_f1'],true);chart('lr',h,['learning_rate']);
      const warnings=h.slice(1).filter((r,i)=>r.train_loss<h[i].train_loss&&r.val_loss>h[i].val_loss);
      $('overfitting').textContent=warnings.length?`Potential overfitting signal at ${warnings.length} epoch transition(s): training loss fell while validation loss rose. Review the curves; this alone is not a diagnosis.`:'No epoch transition with falling training loss and rising validation loss has been observed.';
    }
    if(c){
      const entries=[['Architecture',c.architecture],['Output categories',c.classes.length],['Input',`${c.image_size} × ${c.image_size} RGB`],['Checkpoint','models/best.pt'],['Epoch limit / patience',`${c.epochs} / ${c.continuation?.early_stopping_disabled?'disabled for continuation':c.patience}`],['Optimizer',`AdamW · initial LR ${c.learning_rate}`],['Selection','Validation macro F1'],['Random seed',c.continuation?.seed??c.seed],['Data policy',c.data_policy?.name||'Pending'],['Effective training images',c.effective_split_sizes?.train??'Pending'],['Effective validation images',c.effective_split_sizes?.val??'Pending'],['Parameters',c.parameter_count?.toLocaleString()||'Pending']];
      $('model-info').innerHTML=entries.map(([k,v])=>`<div class="label">${esc(k.toUpperCase())}<strong>${esc(v)}</strong></div>`).join('');
    }
    if(e){evaluation=e;classes=e.classes;$('test-accuracy').textContent=pct(e.accuracy);for(const key of ['macro_f1','macro_precision','macro_recall'])$(key).textContent=pct(e[key]);
      $('eval-note').textContent=`${e.test_images.toLocaleString()} held-out test images · checkpoint from epoch ${e.best_epoch} · ${e.test_evaluation_passes} test inference pass(es) across recorded versions.`+(e.continuation_evaluation_note?' '+e.continuation_evaluation_note:'');
      $('per-class').innerHTML=e.per_class.map(r=>`<tr><td>${esc(pretty(r.class))}</td><td>${pct(r.precision)}</td><td>${pct(r.recall)}</td><td>${pct(r.f1)}</td><td>${r.support}</td></tr>`).join('');
      $('extra-metrics').textContent=`Weighted F1: ${pct(e.weighted_f1)} · Macro one-vs-rest AUROC: ${e.macro_auroc_ovr===null?'Unavailable':e.macro_auroc_ovr.toFixed(4)}`;
      matrix();$('errors').className='';$('errors').innerHTML=e.common_errors.length?e.common_errors.map(r=>`<div class="split-row"><span>${esc(pretty(r.true_class))} → ${esc(pretty(r.predicted_class))}</span><b>${r.count}</b></div>`).join(''):'No misclassifications in this test pass.';
      $('timing').textContent=`${e.total_inference_seconds.toFixed(2)} s total · ${e.average_inference_ms.toFixed(2)} ms/image. ${e.timing_note}`;
      $('examples').innerHTML=e.examples.map(ex=>`<article class="example"><img loading="lazy" src="../outputs/${ex.image.split('/').map(encodeURIComponent).join('/')}" alt="${esc(pretty(ex.true_class))} cell"><div><b>${esc(pretty(ex.true_class))}</b><br><span class="${ex.correct?'':'wrong'}">Predicted ${esc(pretty(ex.predicted_class))}</span><br>Confidence ${pct(ex.confidence)}</div></article>`).join('');
    }
    $('refreshed').textContent='Local files read '+new Date().toLocaleTimeString();
  }catch(error){$('message').textContent='Could not read local data. Serve the dashboard with manage.py serve. '+error.message;}
}
$('normalize').addEventListener('change',matrix);$('refresh').addEventListener('click',refresh);refresh();setInterval(refresh,15000);
