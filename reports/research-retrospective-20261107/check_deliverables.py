from pathlib import Path
import json,zipfile,xml.etree.ElementTree as ET,hashlib,re,math
from docx import Document
from PIL import Image,ImageDraw,ImageFont

out=Path(__file__).parent;root=Path('D:/lab/SEER');qa=out/'qa';qa.mkdir(exist_ok=True)
data=json.loads((out/'research-data.json').read_text(encoding='utf-8'))
doc=Document(out/'SEER研究方案与实验演进复盘.docx')
ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','a':'http://schemas.openxmlformats.org/drawingml/2006/main','wp':'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
# This is a limited OOXML preview, not native Word/LibreOffice layout. It validates
# the simple paragraphs, manual breaks and image geometry used by this report.
S=2;W,H=1191,1684;M=113;B=H-102
def font(size,bold=False):return ImageFont.truetype('C:/Windows/Fonts/'+('msyhbd.ttc' if bold else 'msyh.ttc'),round(size*S))
def wrap(t,f):
    lines=[];line=''
    for c in t:
        if c=='\n':lines.append(line);line='';continue
        if f.getlength(line+c)>W-2*M:lines.append(line);line=c
        else:line+=c
    if line:lines.append(line)
    return lines
pages=[];stats=[];overflow=[]
def newpage():return Image.new('RGB',(W,H),'white'),113
im,y=newpage();d=ImageDraw.Draw(im)
def finish():
    global im,y,d
    d.text((W-155,H-65),str(len(pages)+1),font=font(8),fill='#66717d')
    path=qa/f'preview-{len(pages)+1:02}.png';im.save(path);pages.append(path);stats.append(y)
    im,y=newpage();d=ImageDraw.Draw(im)
for p in doc.paragraphs:
    if p._p.xpath('.//w:br[@w:type="page"]'):
        finish();continue
    draws=p._p.xpath('.//wp:inline')
    if draws:
        ext=draws[0].find('wp:extent',ns);height=int(ext.get('cy'))/914400*144
        pic=Image.open(out/'SEER研究路线思维导图.png');pic=pic.resize((W-2*M,round(height)))
        if y+height>B:overflow.append({'page':len(pages)+1,'kind':'image','y':y+height})
        im.paste(pic,(M,int(y)));y+=height+12;continue
    if not p.text: y+=8;continue
    st=p.style.name;size={'Title':25,'Heading 1':18,'Heading 2':12,'Subtitle':12}.get(st,11)
    if p.runs and p.runs[0].font.size:size=p.runs[0].font.size.pt
    bold=st.startswith('Heading') or st=='Title' or (p.runs and p.runs[0].bold)
    f=font(size,bold);lines=wrap(p.text,f)
    before=24 if st=='Heading 1' else 16 if st=='Heading 2' else 0
    y+=before
    for line in lines:
        d.text((M,int(y)),line,font=f,fill='black');y+=size*2*1.35
    y+=14 if st.startswith('Heading') else 10
    if y>B:overflow.append({'page':len(pages)+1,'kind':'text','paragraph':p.text[:70],'y':y})
if y>113:finish()
# Every preview page on contact sheets; high-resolution individual images retained internally.
for start in range(0,len(pages),12):
    sheet=Image.new('RGB',(4*310,3*465),'#dfe5ec')
    sd=ImageDraw.Draw(sheet)
    for j,path in enumerate(pages[start:start+12]):
        thumb=Image.open(path);thumb.thumbnail((292,420));x=(j%4)*310+9;yy=(j//4)*465+25
        sheet.paste(thumb,(x,yy));sd.text((x,yy-21),str(start+j+1),font=font(8),fill='black')
    sheet.save(qa/f'contact-{start//12+1}.png')
alltext='\n'.join(p.text for p in doc.paragraphs)
items=[i for g in data['groups'] for i in g['items']]
assert len(items)==69
for i in items:
    assert i['id']+'  '+i['title'] in alltext,i['id']
    for k in ['problem','attempt','result','next']:assert i[k] in alltext,(i['id'],k)
for path,ref in data['sources'].items():assert (root/path).is_file() and '['+ref+']' in alltext
with zipfile.ZipFile(out/'SEER研究方案与实验演进复盘.docx') as z:
    assert z.testzip() is None
    for path in z.namelist():
        if path.endswith('.xml'):ET.fromstring(z.read(path))
tree=ET.parse(out/'SEER研究演进思维导图.opml');assert len(tree.findall('.//outline'))==1+9+69*6
ht=(out/'SEER研究演进思维导图.html').read_text(encoding='utf-8')
assert ht.count('class="node"')==69
for s in ['0.76553','0.92015','0.95065','0.78025','0.96827']:
    assert s in (root/'docs/experiment2-reset-results-2026-11-06.md').read_text(encoding='utf-8')
source_checks=[('reports/romsl_proxy_cd/report.md',['84.88','97.78','48,055','16,019']),('docs/idea3-idea6-harrad-unified-results-2026-11-05.md',['30.92','29.02','27.18'] )]
# Last number belongs to the offline P1 source, not the unified comparison.
source_checks[1][1].remove('27.18')
source_checks.append(('docs/radar-text-p1-evidence-count-2026-10-11.md',['27.18','24.69']))
for path,values in source_checks:
    t=(root/path).read_text(encoding='utf-8')
    for s in values:assert s in t,(path,s)
report=dict(attempts=len(items),themes=len(data['groups']),primary_sources=len(data['sources']),local_record_inventory=data['inventory_count'],limited_preview_pages=len(pages),preview_overflow=overflow,docx_xml_valid=True,all_nodes_and_sources_present=True,opml_valid=True,html_node_count=69,native_office_render=False,render_limitation='Packaged renderer failed: no bundled LibreOffice/soffice on this Windows runtime. Limited OOXML preview checks paragraphs, manual page breaks and image geometry; native Office pagination remains unverified.')
(qa/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False))
