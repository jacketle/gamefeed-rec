# -*- coding: utf-8 -*-
"""Verify the standalone report via a local headless Edge debugging session."""
import base64
import json
import time
from pathlib import Path
from urllib.request import urlopen
import websocket

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/release/figures'
with urlopen('http://127.0.0.1:8767/json') as r:
    page=next(p for p in json.load(r) if p['type']=='page')
ws=websocket.create_connection(page['webSocketDebuggerUrl'],suppress_origin=True,timeout=30)
counter=0
def call(method,params=None):
    global counter
    counter+=1
    ws.send(json.dumps(dict(id=counter,method=method,params=params or {})))
    while True:
        msg=json.loads(ws.recv())
        if msg.get('id')==counter:
            if 'error' in msg: raise RuntimeError(msg['error'])
            return msg['result']
def js(expression):
    r=call('Runtime.evaluate',dict(expression=expression,returnByValue=True,awaitPromise=True))
    if 'exceptionDetails' in r: raise RuntimeError(r['exceptionDetails'])
    return r['result'].get('value')
def capture(name):
    data=call('Page.captureScreenshot',dict(format='png',captureBeyondViewport=False))
    (OUT/name).write_bytes(base64.b64decode(data['data']))
try:
    call('Page.enable');call('Network.enable')
    call('Network.emulateNetworkConditions',dict(offline=True,latency=0,downloadThroughput=0,uploadThroughput=0))
    call('Emulation.setDeviceMetricsOverride',dict(width=1440,height=1100,deviceScaleFactor=1,mobile=False))
    call('Page.navigate',dict(url=(ROOT/'deliverables/GameFeed_实验结果与面试图解.html').as_uri()))
    for _ in range(100):
        if js("document.querySelector('#qa-count')?.textContent.includes('22 / 22')"):break
        time.sleep(.1)
    else:raise AssertionError('Report did not load')
    js("document.querySelectorAll('img').forEach(img=>img.loading='eager')")
    js("Promise.all(Array.from(document.images).filter(x=>x.src).map(x=>x.decode()))")
    assert js("document.querySelectorAll('.qa').length")==22
    assert js("document.querySelectorAll('.formula-card').length")==9
    assert js("document.querySelectorAll('figure').length")==12
    assert js("!/长看|用户日/.test(document.body.innerText)")
    assert js("document.body.innerText.includes('视频长度不超过18秒')")
    assert js("document.body.innerText.includes('BGE＋历史均值')")
    assert js("document.querySelector('#overview').innerText.includes('相比 ItemCF')")
    assert js("document.querySelector('#ranking').innerText.includes('312/4402')")
    assert js("document.querySelector('#ranking').innerText.includes('11.89%')")
    assert js("document.querySelector('#retrieval').innerText.includes('1.3187%')")
    assert js("document.querySelector('#sources').textContent.includes('预热5次')")
    assert js("document.querySelector('#methods').textContent.includes('测试集曾用于此前迭代')")
    assert js("document.querySelector('#methods').textContent.includes('不包含重新训练的波动')")
    assert js("document.querySelector('#final-evaluation').innerText.includes('4.448%')")
    assert js("document.querySelector('#final-evaluation').innerText.includes('3.858%')")
    assert js("Array.from(document.querySelectorAll('.qa')).find(x=>x.querySelector('summary').innerText.includes('Q08')).textContent.includes('实验排序模式使用')")
    assert js("Array.from(document.querySelectorAll('script[src],link[rel=stylesheet]')).length")==0
    assert js("Array.from(document.images).filter(x=>x.src).every(x=>x.src.startsWith('data:') && x.naturalWidth>0)")
    capture('report-desktop.png')
    assert js("parseFloat(getComputedStyle(document.querySelector('figcaption')).fontSize)")>=14
    assert js("parseFloat(getComputedStyle(document.querySelector('table')).fontSize)")>=14
    assert js("document.querySelectorAll('.arch-panel').length")==4
    for panel in ['arch-fusion','arch-tower','arch-ranking','arch-bge']:
        js(f"document.querySelector('[data-arch-tab={panel}]').click()")
        assert js("document.querySelector('.arch-panel:not([hidden])').id")==panel
    js("location.hash='arch-tower'")
    time.sleep(.3)
    assert js("document.querySelector('.arch-panel:not([hidden])').id")=='arch-tower'
    js("document.querySelector('#architecture').scrollIntoView({behavior:'instant'})")
    capture('report-architecture.png')
    js("document.querySelector('#arch-tower img').click()")
    assert js("document.querySelector('#image-dialog').open")
    js("document.querySelector('#close-image').click()")
    call('Emulation.setEmulatedMedia',dict(media='print'))
    assert js("Array.from(document.querySelectorAll('.arch-panel')).every(x=>getComputedStyle(x).display!=='none')")
    call('Emulation.setEmulatedMedia',dict(media='screen'))
    assert js("Array.from(document.querySelectorAll('a[href^=\"#\"]')).every(a=>document.getElementById(a.hash.slice(1)))")
    js("document.querySelector('#next').click()")
    assert js("document.querySelector('#stage-count').textContent.includes('2 / 5')")
    js("document.querySelector('#play').click()")
    time.sleep(2.4)
    assert js("document.querySelector('#stage-count').textContent.includes('3 / 5')")
    js("document.querySelector('#play').click()")
    for value,winner in [(1,'A'),(0,'F')]:
        js(f"document.querySelector('#alpha').value='{value}'; document.querySelector('#alpha').dispatchEvent(new Event('input'))")
        assert js("document.querySelector('#toy-table tbody tr td').textContent")==winner
    js("document.querySelector('#alpha').value='1'; document.querySelector('#alpha').dispatchEvent(new Event('input')); document.querySelector('#cap').click()")
    assert js("Array.from(document.querySelectorAll('#toy-table tbody tr:not(.toy-dropped)')).map(r=>r.cells[0].textContent).join('')")=='ABDE'
    js("document.querySelector('#qa-search').value='双塔'; document.querySelector('#qa-search').dispatchEvent(new Event('input'))")
    count=js("document.querySelectorAll('.qa:not([hidden])').length")
    assert 0<count<22
    js("document.querySelector('#expand').click()")
    assert js("Array.from(document.querySelectorAll('.qa:not([hidden])')).every(x=>x.open)")
    js("document.querySelector('#qa-search').value=''; document.querySelector('#qa-search').dispatchEvent(new Event('input'))")
    js("document.querySelector('#retrieval img').click()")
    assert js("document.querySelector('#image-dialog').open")
    js("document.querySelector('#close-image').click()")
    js("document.querySelector('#flow').scrollIntoView({behavior:'instant'})")
    capture('report-flow.png')
    js("document.querySelector('#formulas').scrollIntoView({behavior:'instant'})")
    capture('report-formulas.png')
    js("document.querySelector('#questions').scrollIntoView({behavior:'instant'})")
    capture('report-questions.png')
    js("document.querySelector('#ranking').scrollIntoView({behavior:'instant'})")
    capture('report-ranking-coverage.png')
    js("Array.from(document.querySelectorAll('h3')).find(x=>x.textContent==='开发集：不同方案的前10条对照').scrollIntoView({behavior:'instant'})")
    capture('report-dev-top10.png')
    for width in [1440,390]:
        call('Emulation.setDeviceMetricsOverride',dict(width=width,height=1100,deviceScaleFactor=1,mobile=width<500))
        js("window.scrollTo({top:0,behavior:'instant'})")
        time.sleep(.1)
        size=js('({viewport:innerWidth,scroll:document.documentElement.scrollWidth})')
        assert size['scroll']<=width,size
        if width==390:capture('report-mobile.png')
    # Embedded exports can be moved together with this one HTML file.
    assert js("document.querySelector('a[download$=\".pdf\"]').href.startsWith('data:application/pdf;base64,')")
    assert js("document.querySelector('a[download$=\".json\"]').href.startsWith('data:application/json;base64,')")
    result={'offline_file_open':True,'figures':8,'architecture_figures':4,'architecture_tabs_links_zoom':True,'formulas':9,'questions':22,'widths':[1440,390],
            'flow_and_autoplay':True,'weight_slider_and_author_cap':True,'qa_search_and_expand':True,
            'image_zoom':True,'embedded_downloads':True,'external_assets':0,
            'plain_chinese_labels':True,'watching_threshold_defined':True,
            'strong_baseline_and_metric_denominators':True}
    (OUT/'html_check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))
finally:
    call('Browser.close');ws.close()
