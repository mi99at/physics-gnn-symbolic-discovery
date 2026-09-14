export const quantities = {
  m: {name:'Mass', symbol:'m', unit:'kg'}, F: {name:'Net force magnitude',symbol:'F',unit:'N'},
  a: {name:'Acceleration magnitude',symbol:'a',unit:'m/s²'}, p: {name:'Momentum magnitude',symbol:'p',unit:'kg·m/s'},
  v: {name:'Speed',symbol:'v',unit:'m/s'}, K: {name:'Kinetic energy',symbol:'K',unit:'J'}
};
export const defaults = {m:2,F:6,a:3,p:8,v:4,K:16};
export const equations = [
  {id:'newton',label:'F = ma',vars:['F','m','a'],solve:{F:x=>x.m*x.a,m:x=>x.F/x.a,a:x=>x.F/x.m}},
  {id:'momentum',label:'p = mv',vars:['p','m','v'],solve:{p:x=>x.m*x.v,m:x=>x.p/x.v,v:x=>x.p/x.m}},
  {id:'energy',label:'K = ½mv²',vars:['K','m','v'],solve:{K:x=>0.5*x.m*x.v*x.v,m:x=>2*x.K/(x.v*x.v),v:x=>Math.sqrt(2*x.K/x.m)}},
  {id:'combined',label:'K = p²/2m',vars:['K','p','m'],solve:{K:x=>x.p*x.p/(2*x.m),p:x=>Math.sqrt(2*x.m*x.K),m:x=>x.p*x.p/(2*x.K)}}
];
export const formulas = {newton:{F:'F = m × a',m:'m = F ÷ a',a:'a = F ÷ m'},momentum:{p:'p = m × v',m:'m = p ÷ v',v:'v = p ÷ m'},energy:{K:'K = ½ × m × v²',m:'m = 2K ÷ v²',v:'v = √(2K ÷ m)'},combined:{K:'K = p² ÷ (2m)',p:'p = √(2mK)',m:'m = p² ÷ (2K)'}};
export const fmt = n => Number.isFinite(n) ? Number(n.toPrecision(6)).toLocaleString('en-US',{maximumSignificantDigits:6}) : '—';
export function solve(target, values, available) {
  if (!Object.hasOwn(quantities,target)) throw new Error('Unknown target.');
  const known = new Set(available.filter(k=>k!==target));
  for(const k of known) if(!Object.hasOwn(quantities,k)||!Number.isFinite(values[k])||values[k]<=0) throw new Error('Every available measurement must be a finite number greater than zero.');
  function visit(q, trail) {
    if(trail.has(q)) return [];
    if(known.has(q)) return [{value:values[q],inputs:[q],steps:[]}];
    const next = new Set([...trail,q]); const found=[];
    for(const e of equations.filter(e=>e.vars.includes(q))) {
      const deps=e.vars.filter(k=>k!==q);
      for(const left of visit(deps[0],next)) for(const right of visit(deps[1],next)) {
        const args={[deps[0]]:left.value,[deps[1]]:right.value};const value=e.solve[q](args);
        if(!Number.isFinite(value)||value<=0)continue;
        const steps=[...left.steps,...right.steps,{equation:e.id,target:q,args,value}];
        found.push({value,inputs:[...new Set([...left.inputs,...right.inputs])].sort(),steps});
      }
    }
    return found;
  }
  const all=visit(target,new Set()).sort((a,b)=>a.steps.length-b.steps.length);
  const distinct=new Map();for(const r of all){const key=r.inputs.join(',');if(!distinct.has(key))distinct.set(key,r);}
  const routes=[...distinct.values()];
  const min=routes.length?Math.min(...routes.map(r=>r.value)):null;
  const max=routes.length?Math.max(...routes.map(r=>r.value)):null;
  return {target,routes,min,max,spread:routes.length>1?(max-min)/max:0};
}
