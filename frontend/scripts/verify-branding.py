"""Check built branding assets, or --live using curl without modifying bot policies."""
from pathlib import Path
from html.parser import HTMLParser
from io import BytesIO
import json
import subprocess
import sys
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BASE = 'https://nolielfili.com'
TITLE = 'NoliElFili | Buod, Buong Kwento at Sanggunian ni Rizal'
DESCRIPTION = 'Tuklasin ang Noli Me Tangere at El Filibusterismo: Buod, Buong Kwento, mga tauhan at tema. Hanapin ang mga siping tumutugma sa Buod gamit ang Sanggunian.'
LIVE = '--live' in sys.argv

def read(path):
    if LIVE:
        result = subprocess.run(['curl.exe', '-sS', '--max-time', '45', '-A', 'Googlebot-Image', '-w', '\n%{json}', BASE+path], capture_output=True, check=True)
        body, details = result.stdout.rsplit(b'\n', 1)
        info = json.loads(details)
        assert info['http_code'] == 200 and info['url_effective'] == BASE+path, info
        print('HTTP 200', path, info['content_type'])
        return body
    return (ROOT/'out'/('index.html' if path=='/' else path.lstrip('/'))).read_bytes()

class Head(HTMLParser):
    def __init__(self, html):
        super().__init__(); self.meta={}; self.links=[]; self.title=''; self.in_title=False
        self.feed(html)
    def handle_starttag(self, tag, attrs):
        attrs=dict(attrs)
        if tag=='meta':
            key=attrs.get('name',attrs.get('property'))
            if key in ('description','og:title','og:description','twitter:title','twitter:description'):
                assert key not in self.meta, ('duplicate metadata',key)
            self.meta[key]=attrs.get('content')
        if tag=='link': self.links.append(attrs)
        if tag=='title': self.in_title=True
    def handle_endtag(self,tag):
        if tag=='title': self.in_title=False
    def handle_data(self,data):
        if self.in_title: self.title+=data

head=Head(read('/').decode())
assert head.title==TITLE
for key in ('description','og:description','twitter:description'): assert head.meta[key]==DESCRIPTION
for key in ('og:title','twitter:title'): assert head.meta[key]==TITLE
assert [x['href'] for x in head.links if x.get('rel')=='canonical']==[BASE+'/']
icons=[x for x in head.links if x.get('rel') in ('icon','apple-touch-icon')]
assert len({x['href'] for x in icons})==len(icons), icons
assert '/favicon.ico' in [x['href'] for x in icons]
assert '/favicon-48x48.png' in [x['href'] for x in icons]
for link in icons:
    path=link['href']; data=read(path)
    assert data==(ROOT/'public'/path.lstrip('/')).read_bytes(), ('unexpected icon bytes',path)
    image=Image.open(BytesIO(data))
    assert image.width==image.height
    if path.endswith('.ico'):
        assert image.format=='ICO' and (48,48) in image.ico.sizes()
    else:
        assert image.format=='PNG' and image.mode=='RGBA'
        assert link['sizes']==f'{image.width}x{image.height}'
        assert image.getpixel((0,0))[3]==0
manifest_path=[x['href'] for x in head.links if x.get('rel')=='manifest']
assert manifest_path==['/site.webmanifest']
manifest=json.loads(read(manifest_path[0]))
for icon in manifest['icons']:
    im=Image.open(BytesIO(read(icon['src'])))
    assert icon['sizes']==f'{im.width}x{im.height}'
assert read('/icon.png')==(ROOT/'public/icon.png').read_bytes()
print('PASS exact metadata, unique icon references, canonical, image bytes/formats/sizes/transparency, manifest')
