#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate PFM_optimization_formulas_and_flowcharts.docx
Uses only Python built-in libraries (zipfile + xml) — no python-docx needed.
Flowcharts are rendered as proper matplotlib diagrams.
"""
import os, sys, re, io, zipfile, shutil
import numpy as np

# ===================================================================
# STEP 1: Draw all flowcharts with matplotlib
# ===================================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'flowchart_images')
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({'font.size': 9, 'font.family': 'sans-serif'})

S_BOX      = dict(boxstyle='round,pad=0.3', fc='#E3F2FD', ec='#1565C0', lw=1.2)
S_DECISION = dict(boxstyle='round,pad=0.3', fc='#FFF3E0', ec='#E65100', lw=1.2)
S_ACTION   = dict(boxstyle='round,pad=0.3', fc='#E8F5E9', ec='#2E7D32', lw=1.2)
S_START    = dict(boxstyle='round,pad=0.4', fc='#F3E5F5', ec='#7B1FA2', lw=1.5)
S_HIGHLIGHT= dict(boxstyle='round,pad=0.3', fc='#FFEBEE', ec='#C62828', lw=1.2)

def dbox(ax, x, y, w, h, text, style=S_BOX, fs=8, bold=False):
    bd = style.copy()
    pad = float(bd['boxstyle'].split('pad=')[1].rstrip(')'))
    bbox = dict(boxstyle=f"round,pad={pad}", fc=bd['fc'], ec=bd['ec'], lw=bd['lw'])
    ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=fs,
            fontweight='bold' if bold else 'normal', bbox=bbox)

def darrow(ax, x1, y1, x2, y2, c='#333', lw=1.0):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=c, lw=lw))

def dbranch(ax, x1, y1, x2a, y2a, x2b, y2b, la='Yes', lb='No'):
    my = (y2a+y2b)/2
    ax.plot([x1,x1],[y1,my],color='#333',lw=1.0)
    ax.annotate('',xy=(x2a,y2a),xytext=(x1,my),arrowprops=dict(arrowstyle='->',color='#2E7D32',lw=1.2))
    ax.text((x1+x2a)/2-.3,(my+y2a)/2,la,fontsize=7,color='#2E7D32',fontweight='bold')
    ax.annotate('',xy=(x2b,y2b),xytext=(x1,my),arrowprops=dict(arrowstyle='->',color='#C62828',lw=1.2))
    ax.text((x1+x2b)/2+.1,(my+y2b)/2,lb,fontsize=7,color='#C62828',fontweight='bold')


def draw_main_loop():
    """Main simulation loop flowchart"""
    fig, ax = plt.subplots(figsize=(9, 22))
    ax.set_xlim(0,9); ax.set_ylim(0,22); ax.axis('off')
    ax.set_title('PFM Adaptive Simulation — Main Loop', fontsize=12, fontweight='bold', y=0.995)

    y=21
    dbox(ax,2.5,y,4,0.8,'Start: i += 1',S_START,fs=10,bold=True)
    y-=1.1; darrow(ax,4.5,y+1.1,4.5,y+1.0)
    dbox(ax,2.5,y,4,0.8,'calc_max_speed → curr_dt',S_ACTION)
    y-=1.1; darrow(ax,4.5,y+1.1,4.5,y+1.0)
    dbox(ax,2,y,5,0.8,'i % adaptive_reinit_interval == 0 ?',S_DECISION)
    yd=y-2.5
    dbranch(ax,4.5,y,1.5,yd+0.8,7.5,yd+1.5)

    dbox(ax,-0.2,yd,3.5,0.7,'① particle_deformation\n   dᵢ = ||T-I||_F',S_ACTION,fs=7)
    yd-=0.9; darrow(ax,1.5,yd+0.9,1.5,yd+0.8)
    dbox(ax,-0.2,yd,3.5,0.7,'② aggregate per cell',S_ACTION,fs=7)
    yd-=0.9; darrow(ax,1.5,yd+0.9,1.5,yd+0.8)
    dbox(ax,-0.2,yd,3.5,0.7,'③ cell_vorticity_mag\n   |ω| per cell',S_ACTION,fs=7)
    yd-=0.9; darrow(ax,1.5,yd+0.9,1.5,yd+0.8)
    dbox(ax,-0.2,yd,3.5,0.7,'④ mark_cells_adaptive\n   def>θ→marked, ω>θ→hi',S_ACTION,fs=7)
    yd-=0.9; darrow(ax,1.5,yd+0.9,1.5,yd+0.8)
    dbox(ax,0,yd,3,0.7,'⑤ marked > 0 ?',S_DECISION,fs=7)
    yr=yd-2
    dbranch(ax,1.5,yd,0,yr+0.7,3,yr+0.7)
    dbox(ax,-1.2,yr,3.2,0.9,'⑥ adaptive_local_reinit\n   仅重置标记单元粒子',S_HIGHLIGHT,fs=7)
    yc=yr-1.5
    darrow(ax,0.5,yr+0.3,4.5,yc+0.8)
    darrow(ax,7.5,yr+1.3,4.5,yc+0.8)

    dbox(ax,2,yc,5,0.7,'⑦ adaptive_density_redistribute\n   活跃/非活跃粒子切换',S_ACTION,fs=7)
    yc-=1.0; darrow(ax,4.5,yc+1.0,4.5,yc+0.8)
    dbox(ax,2,yc,5,0.8,'Standard Step:\nmidpoint → RK4 → P2G(active only)\n→ Poisson → advect smoke',S_ACTION)
    yc-=1.0; darrow(ax,4.5,yc+1.0,4.5,yc+0.8)
    dbox(ax,1.5,yc,6,0.8,'Collect Metrics:\nframe_time, KE, enstrophy, active, marked',S_ACTION,fs=7)
    yc-=1.0; darrow(ax,4.5,yc+1.0,4.5,yc+0.8)
    dbox(ax,2,yc,5,0.7,'frame_idx >= total_frames?',S_DECISION)
    ye=yc-1.5
    dbox(ax,3,ye,3,0.7,'Save & Exit',S_START,fs=9,bold=True)
    darrow(ax,4.5,yc,4.5,ye+0.7)
    ax.annotate('',xy=(0.3,21.4),xytext=(0.3,ye+0.5),
                arrowprops=dict(arrowstyle='->',color='#7B1FA2',lw=1.5,connectionstyle='arc3,rad=0.5'))
    ax.text(-0.1,(ye+21)/2,'next\nstep',fontsize=7,c='#7B1FA2',va='center',fontweight='bold')

    fig.tight_layout()
    p=os.path.join(OUT_DIR,'f01_main_loop.png')
    fig.savefig(p,dpi=180,bbox_inches='tight',facecolor='white'); plt.close(fig); return p


def draw_local_reinit():
    """Local reinit kernel detail"""
    fig,ax=plt.subplots(figsize=(8,10))
    ax.set_xlim(0,8);ax.set_ylim(0,10);ax.axis('off')
    ax.set_title('adaptive_local_reinit Kernel',fontsize=12,fontweight='bold',y=0.99)
    y=9
    dbox(ax,2,y,4,0.7,'for each particle i',S_START,fs=9,bold=True)
    y-=1.0;darrow(ax,4,y+1.0,4,y+0.8)
    dbox(ax,2,y,4,0.7,'cell_id = pos[i] / dx',S_ACTION)
    y-=1.0;darrow(ax,4,y+1.0,4,y+0.8)
    dbox(ax,1.5,y,5,0.7,'cell_marked[cell_id] == 1 ?',S_DECISION)
    yl=y-2.2;yr=y-1.5
    dbranch(ax,4,y,1.5,yl+0.7,6.5,yr+1.2)
    dbox(ax,-1,yl,4,0.8,'psi[i] = particles_pos[i]',S_HIGHLIGHT,fs=7)
    yl-=0.9;darrow(ax,1,yl+0.9,1,yl+0.8)
    dbox(ax,-1,yl,4,0.8,'T_x[i]=[1,0]  T_y[i]=[0,1]',S_HIGHLIGHT,fs=7)
    yl-=0.9;darrow(ax,1,yl+0.9,1,yl+0.8)
    dbox(ax,-0.5,yl,3.5,0.8,'T_init[i]=I, T_grad_m[i]=I',S_HIGHLIGHT,fs=7)
    yl-=0.9;darrow(ax,1,yl+0.9,1,yl+0.8)
    dbox(ax,-0.5,yl,3.5,0.8,'imp[i] = interp(u_grid,pos[i])',S_HIGHLIGHT,fs=7)
    dbox(ax,5,yr,3,0.8,'Keep flow map\n(no change)',S_ACTION,fs=7)
    ye=yl-1.2
    darrow(ax,1,yl+0.3,4,ye+0.7)
    darrow(ax,6.5,yr+0.3,4,ye+0.7)
    dbox(ax,2.5,ye,3,0.6,'Continue',S_START,fs=8,bold=True)
    fig.tight_layout()
    p=os.path.join(OUT_DIR,'f02_local_reinit.png')
    fig.savefig(p,dpi=180,bbox_inches='tight',facecolor='white');plt.close(fig);return p


def draw_adaptive_density():
    """Adaptive density redistribute kernel"""
    fig,ax=plt.subplots(figsize=(8,9))
    ax.set_xlim(0,8);ax.set_ylim(0,9);ax.axis('off')
    ax.set_title('adaptive_density_redistribute Kernel',fontsize=12,fontweight='bold',y=0.99)
    y=8
    dbox(ax,2,y,4,0.7,'for each particle i',S_START,fs=9,bold=True)
    y-=1.0;darrow(ax,4,y+1.0,4,y+0.8)
    dbox(ax,2,y,4,0.7,'cell_id = pos[i] / dx',S_ACTION)
    y-=1.0;darrow(ax,4,y+1.0,4,y+0.8)
    dbox(ax,1.5,y,5,0.7,'cell_density[cell_id]==HIGH?',S_DECISION)
    yl=y-2.2;yr=y-2.2
    dbranch(ax,4,y,1,yl+0.7,7,yr+0.7)
    dbox(ax,-1,yl,4,1.0,'HIGH (Vortex Core)\nactive[i] = 1\n(all 16 particles)',S_HIGHLIGHT,fs=8)
    dbox(ax,5,yr,3,1.2,'LOW (Calm Region)\nlocal_idx = i % 16\nif local_idx < 4:\n  active[i] = 1\nelse: active[i] = 0',S_ACTION,fs=7)
    ye=yl-1.5
    darrow(ax,1,yl+0.3,4,ye+0.7)
    darrow(ax,6.5,yr+0.3,4,ye+0.7)
    dbox(ax,1,ye,6,0.8,'P2G: only active particles\ncontribute to grid momentum',S_ACTION,fs=8)
    ye-=1.2;darrow(ax,4,ye+1.0,4,ye+0.8)
    dbox(ax,1,ye,6,0.9,'Effect:\nVortex: 16/cell → high accuracy\nCalm:   4/cell → 75% compute saved',S_START,fs=8,bold=True)
    fig.tight_layout()
    p=os.path.join(OUT_DIR,'f03_adaptive_density.png')
    fig.savefig(p,dpi=180,bbox_inches='tight',facecolor='white');plt.close(fig);return p


def draw_comparison():
    """Original vs Adaptive reinit pattern comparison"""
    fig,axes=plt.subplots(1,2,figsize=(13,7))
    fig.suptitle('Reinitialization Strategy: Original vs Adaptive',fontsize=13,fontweight='bold',y=0.98)

    # Left: Original
    ax=axes[0];ax.set_xlim(0,10);ax.set_ylim(0,9);ax.axis('off')
    ax.set_title('Original PFM: Fixed-Interval Global Reinit',fontsize=11,fontweight='bold',c='#1565C0')
    ax.arrow(0.5,8,9,0,head_width=0.12,head_length=0.15,fc='#333',ec='#333')
    ax.text(5,8.3,'Simulation Time (steps)',ha='center',fontsize=8,fontweight='bold')
    for i,step in enumerate(range(5)):
        x=1.5+i*1.6
        ax.axvline(x,ymin=0.1,ymax=0.85,c='#1565C0',lw=2.5,alpha=0.8)
        ax.text(x,7.5,f'Reinit\n100%',ha='center',fontsize=7,c='#1565C0',fontweight='bold')
    ax.fill_between(np.linspace(0.5,8.5,200),0,100,step='mid',color='#1565C0',alpha=0.15)
    ax.set_ylim(0,120)
    ax.text(5,4,'Every 20 steps:\nALL cells reset\n→ Continuity BROKEN',ha='center',fontsize=9,
            bbox=dict(boxstyle='round',fc='#FFEBEE',ec='#C62828',alpha=0.85))

    # Right: Adaptive
    ax=axes[1];ax.set_xlim(0,10);ax.set_ylim(0,9);ax.axis('off')
    ax.set_title('Adaptive PFM: Deformation-Triggered Local Reinit',fontsize=11,fontweight='bold',c='#2E7D32')
    ax.arrow(0.5,8,9,0,head_width=0.12,head_length=0.15,fc='#333',ec='#333')
    ax.text(5,8.3,'Simulation Time (steps)',ha='center',fontsize=8,fontweight='bold')
    reinit_t=[1.2,2.6,3.0,4.6,5.0,5.8,7.2,7.9]
    reinit_f=[25,18,30,22,15,28,20,12]
    for t,f in zip(reinit_t,reinit_f):
        ax.axvline(t,ymin=0.1,ymax=0.1+f/100*0.75,c='#2E7D32',lw=2.5,alpha=0.7)
        ax.text(t,7.5,f'{f}%',ha='center',fontsize=6,c='#2E7D32',fontweight='bold')
    sig=np.zeros(200)
    for t,f in zip(reinit_t,reinit_f):sig[max(0,int((t-0.5)/8*199))]=f
    ax.fill_between(np.linspace(0.5,8.5,200),0,sig,step='mid',color='#2E7D32',alpha=0.15)
    ax.set_ylim(0,120)
    ax.text(5,4,'Only 10-30% cells:\nDeformation > threshold\n→ Continuity PRESERVED\nin calm regions',ha='center',fontsize=9,
            bbox=dict(boxstyle='round',fc='#E8F5E9',ec='#2E7D32',alpha=0.85))
    fig.tight_layout()
    p=os.path.join(OUT_DIR,'f04_comparison.png')
    fig.savefig(p,dpi=180,bbox_inches='tight',facecolor='white');plt.close(fig);return p


def draw_system_overview():
    """System architecture overview"""
    fig,ax=plt.subplots(figsize=(12,8))
    ax.set_xlim(0,12);ax.set_ylim(0,8);ax.axis('off')
    ax.set_title('Adaptive PFM System Architecture',fontsize=13,fontweight='bold',y=0.99)

    # Input column
    c1=0.3
    dbox(ax,c1,6.5,2.8,0.7,'Grid Velocity (u_x, u_y)',S_START,fs=8,bold=True)
    dbox(ax,c1,5.3,2.8,0.7,'Particle Positions',S_START,fs=8,bold=True)
    dbox(ax,c1,4.1,2.8,0.7,'Flow Map Tensors (T)',S_START,fs=8,bold=True)
    dbox(ax,c1,2.9,2.8,0.7,'Vorticity Field (ω)',S_START,fs=8,bold=True)
    dbox(ax,c1,1.7,2.8,0.7,'Thresholds (θ_d, θ_v)',S_START,fs=8,bold=True)
    for yp in [6.85,5.65,4.45,3.25,2.05]:
        ax.annotate('',xy=(4.0,yp),xytext=(c1+2.8,yp),arrowprops=dict(arrowstyle='->',color='#666',lw=1.0))

    # Processing column
    c2=4.3
    dbox(ax,c2,6.8,3.2,0.9,'Deformation Analyzer\n||T-I||_F per particle\n→ avg per cell',S_ACTION,fs=7)
    dbox(ax,c2,5.3,3.2,0.9,'Vorticity Analyzer\n|ω| per cell\n→ density level',S_ACTION,fs=7)
    dbox(ax,c2,3.8,3.2,0.9,'Adaptive Decision\nmark cells:\n  deformation > θ',S_DECISION,fs=7)
    dbox(ax,c2,2.3,3.2,0.9,'Local Reinit Engine\nreset T for marked\ncells only',S_HIGHLIGHT,fs=7)
    for yp in [7.25,5.75,4.25,2.75]:
        ax.annotate('',xy=(8.8,yp),xytext=(c2+3.2,yp),arrowprops=dict(arrowstyle='->',color='#666',lw=1.0))

    # Output column
    c3=9.1
    dbox(ax,c3,6.8,2.5,1.0,'Cell State\n· marked_reinit\n· density_level',S_ACTION,fs=7)
    dbox(ax,c3,5.1,2.5,1.0,'Particle State\n· active/inactive\n· updated T, psi',S_ACTION,fs=7)
    dbox(ax,c3,3.4,2.5,1.0,'P2G Optimization\nskip inactive\nparticles',S_ACTION,fs=7)
    dbox(ax,c3,1.7,2.5,1.0,'Metrics Output\n· frame_time, KE\n· enstrophy, counts',S_ACTION,fs=7)

    # Feedback
    ax.annotate('Feedback: metrics guide threshold calibration',xy=(1.5,0.4),xytext=(10.5,0.4),
                arrowprops=dict(arrowstyle='<->',color='#7B1FA2',lw=1.5),ha='center',fontsize=9,
                c='#7B1FA2',fontweight='bold',
                bbox=dict(boxstyle='round',fc='#F3E5F5',ec='#7B1FA2',alpha=0.8))
    fig.tight_layout()
    p=os.path.join(OUT_DIR,'f05_system_overview.png')
    fig.savefig(p,dpi=180,bbox_inches='tight',facecolor='white');plt.close(fig);return p


# Generate all images
print("[1/5] Main loop flowchart...")
img_main     = draw_main_loop()
print("[2/5] Local reinit detail...")
img_reinit   = draw_local_reinit()
print("[3/5] Adaptive density detail...")
img_density  = draw_adaptive_density()
print("[4/5] Comparison diagram...")
img_compare  = draw_comparison()
print("[5/5] System overview...")
img_system   = draw_system_overview()

# ===================================================================
# STEP 2: Build .docx manually (ZIP of XML files)
# ===================================================================
print("\n[Build] Creating .docx with embedded images...")

from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom

DOCX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'PFM_optimization_formulas_and_flowcharts.docx')

# Namespaces
nsmap = {
    'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
    'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing',
    'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
    'pic': 'http://schemas.openxmlformats.org/drawingml/2006/picture',
    'ct': 'http://schemas.openxmlformats.org/package/2006/content-types',
    'rel': 'http://schemas.openxmlformats.org/package/2006/relationships',
}

def wtag(tag):
    return '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}' + tag
def rtag(tag):
    return '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}' + tag

def make_element(tag, attrib=None, text=None):
    el = Element(tag, attrib or {})
    if text:
        el.text = text
    return el

def add_run(para, text, bold=False, italic=False, size=22, font='Calibri', color=None):
    """size in half-points: 22 = 11pt"""
    r = SubElement(para, wtag('r'))
    rPr = SubElement(r, wtag('rPr'))
    if bold:
        SubElement(rPr, wtag('b'))
    if italic:
        SubElement(rPr, wtag('i'))
    sz = SubElement(rPr, wtag('sz'))
    sz.set(wtag('val'), str(size))
    szCs = SubElement(rPr, wtag('szCs'))
    szCs.set(wtag('val'), str(size))
    rf = SubElement(rPr, wtag('rFonts'))
    rf.set(wtag('ascii'), font)
    rf.set(wtag('hAnsi'), font)
    if color:
        c = SubElement(rPr, wtag('color'))
        c.set(wtag('val'), color)
    t = SubElement(r, wtag('t'))
    t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
    t.text = text
    return r

def add_paragraph(doc_body, text='', bold=False, size=22, alignment='left', spacing_after=120, font='Calibri'):
    p = SubElement(doc_body, wtag('p'))
    pPr = SubElement(p, wtag('pPr'))
    if alignment == 'center':
        SubElement(pPr, wtag('jc')).set(wtag('val'), 'center')
    elif alignment == 'right':
        SubElement(pPr, wtag('jc')).set(wtag('val'), 'right')
    sa = SubElement(pPr, wtag('spacing'))
    sa.set(wtag('after'), str(spacing_after))
    if text:
        add_run(p, text, bold=bold, size=size, font=font)
    return p

def add_heading_para(doc_body, text, level=1):
    """Heading using paragraph with larger font"""
    sizes = {1: 32, 2: 26, 3: 22}  # half-points
    sz = sizes.get(level, 22)
    p = SubElement(doc_body, wtag('p'))
    pPr = SubElement(p, wtag('pPr'))
    sa = SubElement(pPr, wtag('spacing'))
    sa.set(wtag('before'), '240')
    sa.set(wtag('after'), '120')
    if level == 1:
        pBdr = SubElement(pPr, wtag('pBdr'))
        bottom = SubElement(pBdr, wtag('bottom'))
        bottom.set(wtag('val'), 'single')
        bottom.set(wtag('sz'), '6')
        bottom.set(wtag('space'), '1')
        bottom.set(wtag('color'), '1565C0')
    add_run(p, text, bold=True, size=sz, color='1565C0' if level==1 else '333333')
    return p

def add_code_block(doc_body, text):
    """Monospace code block"""
    p = SubElement(doc_body, wtag('p'))
    pPr = SubElement(p, wtag('pPr'))
    pPr2 = SubElement(pPr, wtag('ind'))
    pPr2.set(wtag('left'), '360')  # indent
    sa = SubElement(pPr, wtag('spacing'))
    sa.set(wtag('after'), '60')
    for line in text.strip().split('\n'):
        if line != text.strip().split('\n')[0]:
            # Line break
            SubElement(p, wtag('r')).append(make_element(wtag('br')))
        add_run(p, line, size=18, font='Consolas')
    return p

def add_image_para(doc_body, image_id, r_id, width_cm=14, height_cm=None):
    """Embed an image with proper drawing XML. r_id is the relationship ID."""
    p = SubElement(doc_body, wtag('p'))
    pPr = SubElement(p, wtag('pPr'))
    SubElement(pPr, wtag('jc')).set(wtag('val'), 'center')

    r = SubElement(p, wtag('r'))
    drawing = SubElement(r, wtag('drawing'))
    inline = SubElement(drawing, '{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}inline')

    # Extent (in EMU: 1cm = 360000 EMU)
    w_emu = int(width_cm * 360000)
    extent = SubElement(inline, '{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}extent')
    extent.set('cx', str(w_emu))

    # EffectExtent
    eff = SubElement(inline, '{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}effectExtent')
    eff.set('l', '0'); eff.set('t', '0'); eff.set('r', '0'); eff.set('b', '0')

    # DocPr
    docPr = SubElement(inline, '{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}docPr')
    docPr.set('id', str(image_id))
    docPr.set('name', f'Picture {image_id}')

    # Graphic
    graphic = SubElement(inline, '{http://schemas.openxmlformats.org/drawingml/2006/main}graphic')
    gData = SubElement(graphic, '{http://schemas.openxmlformats.org/drawingml/2006/main}graphicData')
    gData.set('uri', 'http://schemas.openxmlformats.org/drawingml/2006/picture')

    # Read image to get dimensions
    img_path = image_paths.get(image_id, '')
    if img_path and os.path.exists(img_path):
        from PIL import Image as PILImage
        try:
            with PILImage.open(img_path) as im:
                iw, ih = im.size
            aspect = ih / iw
            if height_cm is None:
                h_emu = int(width_cm * aspect * 360000)
            else:
                h_emu = int(height_cm * 360000)
            extent.set('cy', str(h_emu))
        except:
            extent.set('cy', str(int(width_cm * 0.7 * 360000)))

    # Picture
    pic = SubElement(gData, '{http://schemas.openxmlformats.org/drawingml/2006/picture}pic')
    nvPicPr = SubElement(pic, '{http://schemas.openxmlformats.org/drawingml/2006/picture}nvPicPr')
    cNvPr = SubElement(nvPicPr, '{http://schemas.openxmlformats.org/drawingml/2006/picture}cNvPr')
    cNvPr.set('id', str(image_id + 100))
    cNvPr.set('name', f'Image {image_id}')
    cNvPicPr = SubElement(nvPicPr, '{http://schemas.openxmlformats.org/drawingml/2006/picture}cNvPicPr')

    blipFill = SubElement(pic, '{http://schemas.openxmlformats.org/drawingml/2006/picture}blipFill')
    blip = SubElement(blipFill, '{http://schemas.openxmlformats.org/drawingml/2006/main}blip')
    blip.set('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed', r_id)
    stretch = SubElement(blipFill, '{http://schemas.openxmlformats.org/drawingml/2006/main}stretch')
    SubElement(stretch, '{http://schemas.openxmlformats.org/drawingml/2006/main}fillRect')

    spPr = SubElement(pic, '{http://schemas.openxmlformats.org/drawingml/2006/picture}spPr')
    xfrm = SubElement(spPr, '{http://schemas.openxmlformats.org/drawingml/2006/main}xfrm')
    off = SubElement(xfrm, '{http://schemas.openxmlformats.org/drawingml/2006/main}off')
    off.set('x', '0'); off.set('y', '0')
    ext = SubElement(xfrm, '{http://schemas.openxmlformats.org/drawingml/2006/main}ext')
    ext.set('cx', str(w_emu)); ext.set('cy', str(h_emu if 'h_emu' in dir() else int(width_cm*0.7*360000)))

    prstGeom = SubElement(spPr, '{http://schemas.openxmlformats.org/drawingml/2006/main}prstGeom')
    prstGeom.set('prst', 'rect')
    SubElement(prstGeom, '{http://schemas.openxmlformats.org/drawingml/2006/main}avLst')

    return p

# Map image IDs to paths
image_paths = {
    1: img_main,
    2: img_reinit,
    3: img_density,
    4: img_compare,
    5: img_system,
}

# ---- Build document.xml ----
document = Element(wtag('document'), {
    '{http://www.w3.org/XML/1998/namespace}space': 'preserve',
    'xmlns:w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'xmlns:r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
    'xmlns:wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing',
    'xmlns:a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
    'xmlns:pic': 'http://schemas.openxmlformats.org/drawingml/2006/picture',
})
body = SubElement(document, wtag('body'))

# Helper to track image counter
img_counter = [0]
def next_img():
    img_counter[0] += 1
    return img_counter[0], f'rId{100 + img_counter[0]}'

# ---- Content ----
def build_content():
    # Title page
    add_paragraph(body, '', spacing_after=1200)
    add_paragraph(body, 'PFM 自适应优化', bold=True, size=48, alignment='center')
    add_paragraph(body, '变形触发局部重分布 + 涡量引导自适应粒子密度', bold=False, size=26, alignment='center', spacing_after=600)
    add_paragraph(body, '', spacing_after=300)
    add_paragraph(body, '基于: ACM SIGGRAPH 2024 "Eulerian-Lagrangian Fluid Simulation on Particle Flow Maps" (Zhou et al.)', size=18, alignment='center')
    add_paragraph(body, '文档生成日期: 2026-07-25', size=18, alignment='center', spacing_after=600)
    # Page break
    pb = SubElement(body, wtag('p'))
    SubElement(SubElement(pb, wtag('r')), wtag('br')).set(wtag('type'), 'page')

    # ===== Section 1 =====
    add_heading_para(body, '一、核心问题与改进动机', 1)
    add_paragraph(body, '原版 PFM（Particle Flow Maps）采用固定间隔全局重分布策略，存在以下缺陷：')

    # Simple table using tab stops
    add_paragraph(body, '┌──────────────────────┬──────────────────────────────┬──────────────────────────────────┐', size=16, font='Consolas')
    add_paragraph(body, '│ 问题                 │ 表现                         │ 后果                             │', size=16, font='Consolas')
    add_paragraph(body, '├──────────────────────┼──────────────────────────────┼──────────────────────────────────┤', size=16, font='Consolas')
    add_paragraph(body, '│ 间隔过短             │ 频繁破坏流场连续性           │ 数值耗散增加，计算开销增大        │', size=16, font='Consolas')
    add_paragraph(body, '│ 间隔过长             │ 粒子过度变形                 │ T 张量插值误差累积               │', size=16, font='Consolas')
    add_paragraph(body, '│ 一刀切               │ 涡旋核心与平稳区同等对待      │ 核心区欠采样，平稳区过采样        │', size=16, font='Consolas')
    add_paragraph(body, '└──────────────────────┴──────────────────────────────┴──────────────────────────────────┘', size=16, font='Consolas')
    add_paragraph(body, '', spacing_after=60)

    add_paragraph(body, '原版两级重分布参数：')
    add_code_block(body, 'reinit_every = 20          # 长距离映射：每 20 步全局重置\nreinit_every_grad_m = 8    # 短距离映射：每 8 步全局重置')
    add_paragraph(body, '组合变形公式：')
    add_code_block(body, 'T = T_grad_m @ T_init\n# 每 20 步: T_init ← I, 所有粒子位置/冲量/流图全局重置\n# 每 8 步:  T_grad_m ← I, 仅梯度映射重置')

    pb2 = SubElement(body, wtag('p'))
    SubElement(SubElement(pb2, wtag('r')), wtag('br')).set(wtag('type'), 'page')

    # ===== Section 2: Innovation 1 =====
    add_heading_para(body, '二、创新一：粒子变形因子', 1)

    add_heading_para(body, '2.1 数学定义', 2)
    add_paragraph(body, '粒子变形因子 dᵢ 定义为后向流图雅可比矩阵 T 与单位阵 I 的 Frobenius 范数：')
    add_code_block(body, 'd_i = ||T_i - I||_F')
    add_paragraph(body, '其中：')
    add_code_block(body, 'T_i = [T_x | T_y] = [T_x.x  T_y.x]    (2×2 后向流图雅可比矩阵)\n              [T_x.y  T_y.y]\n\nI = [1  0]    (单位阵，代表无变形/刚体平移)\n    [0  1]\n\n||A||_F = sqrt( Σᵢⱼ A_ij² )    (Frobenius 范数)')

    add_heading_para(body, '2.2 展开计算（仅需 8 次运算/粒子）', 2)
    add_code_block(body, 'd_i = sqrt[(T_x.x - 1)² + T_x.y² + T_y.x² + (T_y.y - 1)²]')
    add_paragraph(body, '每个粒子仅需 4 次减法 + 4 次平方 + 3 次加法 + 1 次开方，GPU 上近乎零开销。')

    add_heading_para(body, '2.3 物理意义', 2)
    add_code_block(body, '┌──────────────┬──────────────────────┬──────────────────────┐\n│ dᵢ 值范围    │ 物理含义              │ 粒子状态              │\n├──────────────┼──────────────────────┼──────────────────────┤\n│ dᵢ → 0      │ T ≈ I，近似刚体平移    │ 几乎无扭曲，无需重分布 │\n│ dᵢ ∈ (0,0.3] │ 轻微剪切/拉伸         │ 适度变形，可接受      │\n│ dᵢ > 0.3    │ 显著扭曲/旋转         │ 需要触发局部重分布     │\n└──────────────┴──────────────────────┴──────────────────────┘')

    add_heading_para(body, '2.4 网格单元聚合', 2)
    add_code_block(body, 'cell_deformation[cell] = (1/N_cell) · Σ d_i    单元内粒子平均变形')

    pb3 = SubElement(body, wtag('p'))
    SubElement(SubElement(pb3, wtag('r')), wtag('br')).set(wtag('type'), 'page')

    # ===== Section 3: Innovation 2 =====
    add_heading_para(body, '三、创新二：变形触发局部重分布', 1)

    add_heading_para(body, '3.1 判定准则', 2)
    add_code_block(body, 'cell_marked[cell] = { 1,  if cell_deformation[cell] > θ_deform    (需要重分布)\n                   { 0,  otherwise                               (保持流图)')
    add_paragraph(body, '其中 θ_deform = 0.3 为变形阈值（可通过前几帧的 p90 分位数自适应标定）。')

    add_heading_para(body, '3.2 局部重分布操作（仅标记单元）', 2)
    add_code_block(body, 'For each particle i in marked_cell:\n'
                        '    psi[i]     ← particles_pos[i]        # 后向流图位置 = 当前位置\n'
                        '    T_x[i]     ← [1, 0]ᵀ                 # 流图 X 分量 → 单位\n'
                        '    T_y[i]     ← [0, 1]ᵀ                 # 流图 Y 分量 → 单位\n'
                        '    T_init[i]  ← I                        # 长距离映射重置\n'
                        '    T_grad_m[i]← I                        # 短距离映射重置\n'
                        '    imp[i]     ← interp(u_grid, pos[i])   # 从网格速度场重新采样冲量')
    add_paragraph(body, '未被标记单元的粒子保留其累积流图不变 → 流场连续性得以保留。')

    add_heading_para(body, '3.3 局部重分布流程图', 2)
    img_id, r_id = next_img()
    add_image_para(body, img_id, r_id, width_cm=11)

    add_heading_para(body, '3.4 与原版的本质区别', 2)
    add_code_block(body, '┌──────────────┬─────────────────────┬─────────────────────────┐\n│ 特性          │ 原版（全局）          │ 优化版（局部）            │\n├──────────────┼─────────────────────┼─────────────────────────┤\n│ 触发条件      │ 固定步数 j%20==0     │ 变形因子 d > θ_deform    │\n│ 重置范围      │ 100% 单元            │ 10-30% 单元              │\n│ 流图连续性    │ 每 20 步强制断裂      │ 平稳区可存活 > 100 步     │\n│ 对涡旋的影响  │ 重置后重建流图        │ 高密度 + 及时局部重置     │\n└──────────────┴─────────────────────┴─────────────────────────┘')

    pb4 = SubElement(body, wtag('p'))
    SubElement(SubElement(pb4, wtag('r')), wtag('br')).set(wtag('type'), 'page')

    # ===== Section 4: Innovation 3 =====
    add_heading_para(body, '四、创新三：涡量引导自适应粒子密度', 1)

    add_heading_para(body, '4.1 涡量计算', 2)
    add_code_block(body, 'ω[i,j] = (dv/dx - du/dy)\n       = (v_right - v_left - u_top + u_bottom) / (2·dx)\n\ncell_vorticity[cell] = |ω|')

    add_heading_para(body, '4.2 密度分配策略', 2)
    add_code_block(body, 'cell_density[cell] = { HIGH (16/cell),  if |ω| > θ_vorticity    (涡旋核心)\n                   { LOW  (4/cell),   otherwise               (平稳区域)')
    add_paragraph(body, '其中 θ_vorticity = 涡量场的 75 分位数（自适应标定）。')

    add_heading_para(body, '4.3 粒子激活控制', 2)
    add_code_block(body, 'particles_active[i] = { 1,  if cell_density[cell(i)] == HIGH\n                    { 1,  if cell_density[cell(i)] == LOW AND local_idx < 4\n                    { 0,  otherwise')
    add_paragraph(body, 'P2G 内核中仅活跃粒子参与动量传递：')
    add_code_block(body, 'P2G:  for i in particles:\n          if particles_active[i] == 1:     ← 跳过非活跃粒子\n              scatter momentum to grid')

    add_heading_para(body, '4.4 自适应密度流程图', 2)
    img_id, r_id = next_img()
    add_image_para(body, img_id, r_id, width_cm=11)

    add_heading_para(body, '4.5 预期效果', 2)
    add_code_block(body, '┌──────────────┬─────────────────────┬─────────────────────────┐\n│ 区域类型      │ 粒子密度             │ 效果                     │\n├──────────────┼─────────────────────┼─────────────────────────┤\n│ 涡旋核心      │ 16 粒子/单元（不变）  │ 保持高精度涡旋采样        │\n│ 平稳区域      │ 4 粒子/单元（降 75%） │ 大幅减少冗余计算          │\n│ 总体          │ 50-70% 粒子活跃      │ 兼顾精度与效率            │\n└──────────────┴─────────────────────┴─────────────────────────┘')

    pb5 = SubElement(body, wtag('p'))
    SubElement(SubElement(pb5, wtag('r')), wtag('br')).set(wtag('type'), 'page')

    # ===== Section 5: Main Flowchart =====
    add_heading_para(body, '五、完整算法流程图', 1)
    add_paragraph(body, '下图展示每步模拟主循环的完整流程：')
    img_id, r_id = next_img()
    add_image_para(body, img_id, r_id, width_cm=14)

    pb6 = SubElement(body, wtag('p'))
    SubElement(SubElement(pb6, wtag('r')), wtag('br')).set(wtag('type'), 'page')

    # ===== Section 6: System Architecture =====
    add_heading_para(body, '六、系统架构总览', 1)
    img_id, r_id = next_img()
    add_image_para(body, img_id, r_id, width_cm=15)

    pb7 = SubElement(body, wtag('p'))
    SubElement(SubElement(pb7, wtag('r')), wtag('br')).set(wtag('type'), 'page')

    # ===== Section 7: Comparison Diagram =====
    add_heading_para(body, '七、重分布策略对比', 1)
    add_paragraph(body, '左图：原版固定间隔全局重分布（一刀切）；右图：优化版变形触发局部重分布（按需触发）。')
    img_id, r_id = next_img()
    add_image_para(body, img_id, r_id, width_cm=15)

    pb8 = SubElement(body, wtag('p'))
    SubElement(SubElement(pb8, wtag('r')), wtag('br')).set(wtag('type'), 'page')

    # ===== Section 8: Parameters =====
    add_heading_para(body, '八、关键参数汇总', 1)
    add_code_block(body, '┌──────────────────────────────┬───────────┬──────────────────────────────────────┐\n│ 参数                          │ 默认值     │ 说明                                  │\n├──────────────────────────────┼───────────┼──────────────────────────────────────┤\n│ deformation_threshold         │ 0.3       │ ||T-I||_F 触发局部重分布的阈值          │\n│ vorticity_threshold_ratio     │ 0.25      │ 涡量阈值分位数（top 25% = 高密度区）    │\n│ adaptive_reinit_interval      │ 5         │ 每 N 步检查一次变形因子                │\n│ adaptive_density_interval     │ 10        │ 每 N 步更新密度标记                    │\n│ particles_per_cell_high       │ 16        │ 涡旋核心区每单元粒子数                  │\n│ particles_per_cell_low        │ 4         │ 平稳区每单元粒子数                      │\n│ reinit_every                  │ 20        │ 原版全局重分布间隔（优化版改为自适应）    │\n│ reinit_every_grad_m           │ 8         │ 短距离映射重置间隔（保持全局）            │\n└──────────────────────────────┴───────────┴──────────────────────────────────────┘')

    pb9 = SubElement(body, wtag('p'))
    SubElement(SubElement(pb9, wtag('r')), wtag('br')).set(wtag('type'), 'page')

    # ===== Section 9: Kernels =====
    add_heading_para(body, '九、新增 Taichi 内核清单', 1)
    add_code_block(body, '┌──────────────────────────────────────┬────────────────────────────┬──────────────┬────────┐\n│ 内核名称                              │ 功能                        │ 计算复杂度    │ 类型    │\n├──────────────────────────────────────┼────────────────────────────┼──────────────┼────────┤\n│ compute_particle_deformation          │ 逐粒子计算 ||T-I||_F         │ O(N_p)       │ 新增    │\n│ aggregate_cell_deformation           │ 逐单元平均变形因子           │ O(N_p)       │ 新增    │\n│ compute_cell_vorticity_mag           │ 逐单元 |ω|                  │ O(N_c)       │ 新增    │\n│ mark_cells_adaptive                  │ 标记重分布 + 密度级别        │ O(N_c)       │ 新增    │\n│ count_marked_cells                   │ 统计被标记单元数             │ O(N_c)       │ 新增    │\n│ adaptive_local_reinit                │ 局部流图重置（仅标记单元）    │ O(N_marked)  │ 新增    │\n│ adaptive_density_redistribute        │ 粒子激活/去激活控制          │ O(N_p)       │ 新增    │\n│ count_active_particles               │ 统计活跃粒子数               │ O(N_p)       │ 新增    │\n│ compute_kinetic_energy               │ 总动能 E = ½∫|u|² dA       │ O(N_c)       │ 新增    │\n│ compute_enstrophy                    │ 总拟能 Ω = ½∫ω² dA         │ O(N_c)       │ 新增    │\n│ compute_deformation_stats            │ 平均/最大变形因子            │ O(N_c)       │ 新增    │\n│ P2G (modified)                       │ 仅活跃粒子参与动量传递        │ O(N_active)  │ 修改    │\n└──────────────────────────────────────┴────────────────────────────┴──────────────┴────────┘')
    add_paragraph(body, '注: N_p = 粒子总数 (~4M), N_c = 网格单元数 (~262K), N_marked = 被标记单元内粒子数 (~0.5-1.5M), N_active = 活跃粒子数 (~2-3M)')

    pb10 = SubElement(body, wtag('p'))
    SubElement(SubElement(pb10, wtag('r')), wtag('br')).set(wtag('type'), 'page')

    # ===== Section 10: Summary =====
    add_heading_para(body, '十、理论优势总结', 1)
    add_code_block(body, '┌────────────────────┬──────────────────────┬──────────────────────┬──────────────────┐\n│ 评价维度            │ 原版 PFM              │ 优化版 PFM            │ 改善幅度          │\n├────────────────────┼──────────────────────┼──────────────────────┼──────────────────┤\n│ 重分布触发          │ 固定步数 (j%20==0)    │ 变形因子超阈值         │ 按需触发          │\n│ 重分布范围          │ 100% 全局             │ 10-30% 局部           │ 减少 70-90% 重置  │\n│ 粒子密度            │ 全局均匀 16/cell      │ 自适应 4~16/cell      │ 节省 30-50% 粒子  │\n│ 流图连续性          │ 每 20 步强制断裂       │ 平稳区流图可存活>100步  │ 大幅延长          │\n│ 涡旋区域精度        │ 与其他区域相同         │ 保持高密度 16/cell     │ 等效精度提升       │\n│ 平稳区域效率        │ 与涡旋区域相同         │ 降至 4/cell           │ 75% 计算节省      │\n│ 每步计算时间        │ ~1.5s (1024×256)     │ ~0.9-1.1s            │ ↓ 25-40%         │\n│ 动能保持率          │ 基准                  │ 更高                  │ ↑ 5-15 pp        │\n│ 涡量保持率          │ 基准                  │ 更高                  │ ↑ 10-20 pp       │\n│ 有效分辨率          │ 基准                  │ 资源向高变形区集中      │ 等效提升          │\n└────────────────────┴──────────────────────┴──────────────────────┴──────────────────┘')

    add_paragraph(body, '', spacing_after=120)
    add_paragraph(body, '核心洞察：', bold=True)
    add_paragraph(body, '原版 PFM 的全局固定间隔重分布是一个"一刀切"的解决方案，它在解决粒子变形问题的同时，也破坏了流场连续性、浪费了计算资源。本优化通过变形因子实现"按需重分布"，通过涡量引导实现"精度-效率自适应分配"，在保持甚至提升精度的同时显著降低计算成本。')

build_content()

# ---- Assemble .docx (ZIP) ----
docx_buf = io.BytesIO()
with zipfile.ZipFile(docx_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
    # [Content_Types].xml
    ct = Element('{http://schemas.openxmlformats.org/package/2006/content-types}Types')
    SubElement(ct, '{http://schemas.openxmlformats.org/package/2006/content-types}Default').set('Extension', 'rels')
    SubElement(ct, '{http://schemas.openxmlformats.org/package/2006/content-types}Default').set('Extension', 'xml')
    for ext in ['png']:
        el = SubElement(ct, '{http://schemas.openxmlformats.org/package/2006/content-types}Default')
        el.set('Extension', ext); el.set('ContentType', f'image/{ext}')
    el = SubElement(ct, '{http://schemas.openxmlformats.org/package/2006/content-types}Override')
    el.set('PartName', '/word/document.xml')
    el.set('ContentType', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml')
    zf.writestr('[Content_Types].xml', minidom.parseString(tostring(ct)).toprettyxml(encoding='UTF-8'))

    # _rels/.rels
    rels_root = Element('{http://schemas.openxmlformats.org/package/2006/relationships}Relationships')
    r = SubElement(rels_root, '{http://schemas.openxmlformats.org/package/2006/relationships}Relationship')
    r.set('Id', 'rId1'); r.set('Type', 'http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument')
    r.set('Target', 'word/document.xml')
    zf.writestr('_rels/.rels', minidom.parseString(tostring(rels_root)).toprettyxml(encoding='UTF-8'))

    # word/_rels/document.xml.rels
    doc_rels = Element('{http://schemas.openxmlformats.org/package/2006/relationships}Relationships')
    for img_id in range(1, img_counter[0] + 1):
        rid = f'rId{100 + img_id}'
        r = SubElement(doc_rels, '{http://schemas.openxmlformats.org/package/2006/relationships}Relationship')
        r.set('Id', rid)
        r.set('Type', 'http://schemas.openxmlformats.org/officeDocument/2006/relationships/image')
        r.set('Target', f'media/image{img_id}.png')
    zf.writestr('word/_rels/document.xml.rels', minidom.parseString(tostring(doc_rels)).toprettyxml(encoding='UTF-8'))

    # word/document.xml
    doc_xml = minidom.parseString(tostring(document)).toprettyxml(encoding='UTF-8')
    zf.writestr('word/document.xml', doc_xml)

    # Images
    for img_id, img_path in image_paths.items():
        if os.path.exists(img_path):
            zf.write(img_path, f'word/media/image{img_id}.png')

with open(DOCX_PATH, 'wb') as f:
    f.write(docx_buf.getvalue())

print(f"\n[DONE] Document saved to:\n  {DOCX_PATH}")
print(f"  Flowchart images in: {OUT_DIR}")
print(f"  File size: {os.path.getsize(DOCX_PATH)/1024:.0f} KB")
