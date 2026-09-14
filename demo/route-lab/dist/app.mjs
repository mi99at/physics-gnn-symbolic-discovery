import {quantities, defaults, equations, formulas, solve, fmt} from './physics.mjs';
const $=id=>document.getElementById(id);
let target='m',values={...defaults},available=new Set(['F','a','p','v','K']);
const positions={F:[65,55],a:[65,235],p:[615,55],v:[615,235],K:[350,275],m:[350,55],newton:[180,145],momentum:[500,110],energy:[460,215],combined:[260,215]};
function fields(){
  $('measurements').innerHTML=Object.entries(quantities).map(([key,q])=>`<div class="measurement ${key===target?'is-target':''}"><input type="checkbox" id="known-${key}" aria-label="Make ${q.name.toLowerCase()} available" ${available.has(key)&&key!==target?'checked':''} ${key===target?'disabled':''}><label for="value-${key}">${q.name}${key===target?' · target':''}<small>${q.unit}</small></label><input type="number" min="0" step="any" id="value-${key}" value="${key===target?'':values[key]}" placeholder="?" ${key===target||!available.has(key)?'disabled':''}></div>`).join('');
  for(const key of Object.keys(quantities)){
    $('known-'+key).addEventListener('change',e=>{e.target.checked?available.add(key):available.delete(key);$('value-'+key).disabled=!e.target.checked;render();});
    $('value-'+key).addEventListener('input',e=>{values[key]=e.target.value===''?NaN:Number(e.target.value);render();});
  }
}
function graph(){
  const edges=equations.flatMap(e=>e.vars.map(q=>{const [x1,y1]=positions[q],[x2,y2]=positions[e.id];return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${available.has(q)&&q!==target?'#568475':'#3a4a5b'}" stroke-width="2" ${!available.has(q)&&q!==target?'stroke-dasharray="5 5"':''}/>`;})).join('');
  const eqs=equations.map(e=>{const[x,y]=positions[e.id];return `<rect x="${x-53}" y="${y-19}" width="106" height="38" rx="6" fill="#182b3c" stroke="#647d90"/><text x="${x}" y="${y+5}" text-anchor="middle" font-size="15">${e.label}</text>`;}).join('');
  const nodes=Object.entries(quantities).map(([k,q])=>{const[x,y]=positions[k];const color=k===target?'#80bbff':available.has(k)?'#5cf1bb':'#708298';return `<circle cx="${x}" cy="${y}" r="28" fill="#101c2b" stroke="${color}" stroke-width="${k===target?3:2}"/><text x="${x}" y="${y+6}" text-anchor="middle" font-size="23">${q.symbol}</text><text x="${x}" y="${y+47}" text-anchor="middle" font-size="13">${k===target?'Find '+q.symbol:available.has(k)?fmt(values[k])+' '+q.unit:'Hidden'}</text>`;}).join('');
  $('graph').innerHTML=`<svg viewBox="0 0 700 330" role="img" aria-label="Equation graph linking mass, net force, acceleration, momentum, speed, and kinetic energy. Available measurements connect through Newton's second law, momentum, and kinetic energy equations.">${edges}${eqs}${nodes}</svg>`;
}
function reportURL(result){
  const observations=Object.keys(quantities).filter(k=>available.has(k)&&k!==target).map(k=>`${k}: ${values[k]} ${quantities[k].unit}`).join('\n');
  const body=`## Route Lab example\nTarget: ${target}\nAvailable measurements:\n${observations}\n\nResults:\n${result.routes.map(r=>r.inputs.join(' + ')+' -> '+fmt(r.value)+' '+quantities[target].unit).join('\n')||'No route'}\n\n## What is wrong or unclear?\n[Describe the equation, assumption, or unexpected behavior.]\n\n## Expected result and reasoning\n[Show your calculation and physical context.]\n\nThis report concerns the symbolic demo, not GNN inference.`;
  return 'https://github.com/mi99at/physics-gnn-symbolic-discovery/issues/new?title='+encodeURIComponent('Physics Route Lab: example to review')+'&body='+encodeURIComponent(body);
}
function render(){
  $('graph-title').textContent='Find '+quantities[target].name.toLowerCase();graph();
  let result;try{result=solve(target,values,[...available]);}catch(e){$('summary').className='summary error';$('summary').textContent=e.message;$('routes').innerHTML='';return;}
  $('report-link').href=reportURL(result);
  const {routes,spread}=result;const unit=quantities[target].unit;
  $('summary').className='summary'+(spread>.005?' warn':'');
  if(!routes.length)$('summary').innerHTML='<strong>No complete route.</strong><br>Restore more measurements. The demo will not guess a missing answer.';
  else if(spread>.005)$('summary').innerHTML=`<strong>These routes disagree.</strong><br>${routes.length} input sets give ${fmt(result.min)}–${fmt(result.max)} ${unit}. Check your measurements and assumptions. No single answer is selected.`;
  else $('summary').innerHTML=`<strong>${routes.length} ${routes.length===1?'route':'routes'} available.</strong> ${routes.length>1?'Answers agree within 0.5%.':'Only one input set supports a solution.'}<br>${quantities[target].symbol} = ${fmt(result.min)}${result.max!==result.min?'–'+fmt(result.max):''} ${unit} · calculated from known equations`;
  $('routes').innerHTML=routes.map((r,i)=>`<details class="route" ${i===0?'open':''}><summary><span class="route-name">From ${r.inputs.map(k=>quantities[k].symbol).join(' + ')} · ${r.steps.length} ${r.steps.length===1?'step':'steps'}</span><span class="value">${fmt(r.value)} ${unit}</span></summary><div class="steps"><ol>${r.steps.map(s=>`<li><code>${formulas[s.equation][s.target]}</code>${s.equation==='combined'?' <span>(derived from p = mv and K = ½mv²)</span>':''}<br>Using ${Object.entries(s.args).map(([k,v])=>`${k} = ${fmt(v)} ${quantities[k].unit}`).join(', ')} → ${s.target} = ${fmt(s.value)} ${quantities[s.target].unit}</li>`).join('')}</ol></div></details>`).join('');
  return result;
}
$('target').addEventListener('change',e=>{target=e.target.value;available.delete(target);fields();render();});
$('hide-force').addEventListener('click',()=>{available.delete('F');available.delete('a');fields();render();});
$('reset').addEventListener('click',()=>{target='m';values={...defaults};available=new Set(['F','a','p','v','K']);$('target').value=target;fields();render();});
fields();render();

// Optional page-scoped agent tools share the same state and validation as the UI.
const context=document.modelContext;
if(context?.registerTool){
  const lifecycle=new AbortController();
  const snapshot=()=>({target,available:[...available].filter(k=>k!==target),values:{...values},result:solve(target,values,[...available]),mode:'symbolic-only'});
  const tools=[{
    name:'get_physics_route_state',description:'Read the visible symbolic physics example and calculated routes; no GNN inference.',inputSchema:{type:'object',properties:{},additionalProperties:false},annotations:{readOnlyHint:true,untrustedContentHint:false},execute:()=>snapshot()
  },{
    name:'configure_physics_route_example',description:'Set the visible target, measurements, and available inputs for the symbolic demo. Does not submit reports or run a GNN.',
    inputSchema:{type:'object',properties:{target:{type:'string',enum:['m','F','K','p']},values:{type:'object',properties:Object.fromEntries(Object.keys(quantities).map(k=>[k,{type:'number',exclusiveMinimum:0}])),additionalProperties:false},available:{type:'array',items:{type:'string',enum:Object.keys(quantities)},uniqueItems:true}},required:['target','values','available'],additionalProperties:false},annotations:{readOnlyHint:false,untrustedContentHint:false},
    execute(input){
      if(!input||typeof input!=='object'||Object.keys(input).some(k=>!['target','values','available'].includes(k))||!['m','F','K','p'].includes(input.target)||!input.values||typeof input.values!=='object'||Array.isArray(input.values)||!Array.isArray(input.available)||input.available.some(k=>!Object.hasOwn(quantities,k))||new Set(input.available).size!==input.available.length)throw new Error('Invalid example configuration.');
      for(const[k,v]of Object.entries(input.values))if(!Object.hasOwn(quantities,k)||typeof v!=='number'||!Number.isFinite(v)||v<=0)throw new Error('Measurements must be finite positive numbers.');
      const next={...values,...input.values};solve(input.target,next,input.available);
      target=input.target;values=next;available=new Set(input.available.filter(k=>k!==target));$('target').value=target;fields();render();return snapshot();
    }
  }];
  for(const tool of tools){try{Promise.resolve(context.registerTool(tool,{signal:lifecycle.signal})).catch(()=>{});}catch{}}
  window.addEventListener('pagehide',()=>lifecycle.abort(),{once:true});
}
