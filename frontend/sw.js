const CACHE='tapf-min-ui-v1';
const ASSETS=['/app/','/app/index.html','/app/styles.css','/app/app.js','/app/manifest.webmanifest','/app/icon.svg'];
self.addEventListener('install',event=>{event.waitUntil(caches.open(CACHE).then(c=>c.addAll(ASSETS)));self.skipWaiting()});
self.addEventListener('activate',event=>{event.waitUntil(caches.keys().then(keys=>Promise.all(keys.filter(k=>k!==CACHE).map(k=>caches.delete(k)))));self.clients.claim()});
self.addEventListener('fetch',event=>{
  const req=event.request;if(req.method!=='GET')return;
  const url=new URL(req.url);
  if(url.pathname.startsWith('/healthz')||url.pathname.startsWith('/readyz')||url.pathname.startsWith('/v1/')||url.pathname.startsWith('/v2/'))return;
  if(!url.pathname.startsWith('/app/'))return;
  event.respondWith(fetch(req).then(res=>{const copy=res.clone();caches.open(CACHE).then(c=>c.put(req,copy));return res}).catch(()=>caches.match(req).then(r=>r||caches.match('/app/'))));
});
