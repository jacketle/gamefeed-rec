// Keep all evidence in the document; tabs change only the architecture view.
const archButtons=[...document.querySelectorAll('[data-arch-tab]')];
function showArchitecture(id, focus=false){
 archButtons.forEach(b=>{const on=b.dataset.archTab===id;b.setAttribute('aria-selected',String(on));b.tabIndex=on?0:-1;if(on&&focus)b.focus();});
 document.querySelectorAll('.arch-panel').forEach(p=>p.hidden=p.id!==id);
}
archButtons.forEach((b,i)=>{b.onclick=()=>showArchitecture(b.dataset.archTab);b.onkeydown=e=>{let n=i;if(e.key==='ArrowRight')n=(i+1)%archButtons.length;else if(e.key==='ArrowLeft')n=(i+archButtons.length-1)%archButtons.length;else if(e.key==='Home')n=0;else if(e.key==='End')n=archButtons.length-1;else return;e.preventDefault();showArchitecture(archButtons[n].dataset.archTab,true);};});
function followArchitecture(){if(archButtons.some(b=>'#'+b.dataset.archTab===location.hash)){showArchitecture(location.hash.slice(1));document.querySelector('#architecture').scrollIntoView();}}
window.addEventListener('hashchange',followArchitecture);showArchitecture('arch-fusion');followArchitecture();
document.querySelectorAll('a[href^="#arch-"]').forEach(a=>a.addEventListener('click',()=>{showArchitecture(a.hash.slice(1));requestAnimationFrame(()=>document.querySelector('#architecture').scrollIntoView());}));
const chapterLinks=[...document.querySelectorAll('.rail nav a')];
const chapterObserver=new IntersectionObserver(entries=>{entries.forEach(e=>{if(e.isIntersecting){chapterLinks.forEach(a=>{if(a.hash==='#'+e.target.id)a.setAttribute('aria-current','true');else a.removeAttribute('aria-current');});}});},{rootMargin:'-10% 0px -70% 0px'});
document.querySelectorAll('main>section').forEach(s=>chapterObserver.observe(s));
