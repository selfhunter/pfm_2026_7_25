"""Volume rendering of .vti files — generates 3D rendered images like ParaView.

Usage:
  python vtk_visualize.py <log_directory>
  # or from project root: python 3D/vtk_visualize.py logs/<exp_name>
  # or with fields:   python 3D/vtk_visualize.py logs/<exp_name> --fields smoke vorticity
"""

import os
import re
import sys
import math
import vtk
import numpy as np
from vtk.util import numpy_support


def _make_smoke_tf(sr0, sr1):
    """Transfer functions for binary smoke (0 / 1)."""
    ctf = vtk.vtkColorTransferFunction()
    otf = vtk.vtkPiecewiseFunction()
    # Smoke is binary (0 or 1), so sharp transition
    ctf.AddRGBPoint(0.0, 0.0, 0.0, 0.0)
    ctf.AddRGBPoint(sr1, 1.0, 1.0, 1.0)
    # Opacity: invisible at 0, fully opaque white at 1
    otf.AddPoint(0.0, 0.0)
    otf.AddPoint(sr1 * 0.5, 0.0)  # stay transparent until mid-point
    otf.AddPoint(sr1, 1.0)        # fully opaque at max
    return ctf, otf


def _make_vorticity_tf(sr0, sr1, data):
    """Transfer functions for vorticity with adaptive opacity scaling."""
    if hasattr(data, 'GetPointData'):
        arr = data.GetPointData().GetArray('vorticity')
        if arr:
            vals = numpy_support.vtk_to_numpy(arr)
            if len(vals) > 0:
                p99 = np.percentile(vals, 99)
                # Use 99th percentile, but at least 10% of full range
                effective_max = max(p99, sr0 + (sr1 - sr0) * 0.1)
            else:
                effective_max = sr1
        else:
            effective_max = sr1
    else:
        effective_max = sr1

    r = float(max(effective_max - sr0, 1e-6))

    ctf = vtk.vtkColorTransferFunction()
    otf = vtk.vtkPiecewiseFunction()

    # Jet colormap
    ctf.AddRGBPoint(sr0, 0.0, 0.0, 0.5)
    ctf.AddRGBPoint(sr0 + r * 0.25, 0.0, 0.5, 1.0)
    ctf.AddRGBPoint(sr0 + r * 0.50, 0.0, 0.8, 0.8)
    ctf.AddRGBPoint(sr0 + r * 0.75, 1.0, 1.0, 0.0)
    ctf.AddRGBPoint(effective_max, 0.8, 0.2, 0.0)

    # Opacity: low vorticity = transparent
    otf.AddPoint(sr0, 0.0)
    otf.AddPoint(sr0 + r * 0.05, 0.0)
    otf.AddPoint(sr0 + r * 0.15, 0.05)
    otf.AddPoint(sr0 + r * 0.35, 0.15)
    otf.AddPoint(sr0 + r * 0.55, 0.35)
    otf.AddPoint(sr0 + r * 0.75, 0.55)
    otf.AddPoint(effective_max, 0.80)

    return ctf, otf


def volume_render_vti(vti_path, output_path, field='vorticity',
                      azim=30, elev=25, dist_factor=1.8,
                      resolution=(1920, 1080)):
    """Render a .vti file with GPU volume rendering and save to PNG.

    Produces a ParaView-style 3D volume rendering of the chosen scalar field.
    """
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(vti_path)
    reader.Update()
    data = reader.GetOutput()

    # ---- Volume mapper ----
    mapper = vtk.vtkSmartVolumeMapper()
    mapper.SetInputData(data)
    mapper.SelectScalarArray(field)
    mapper.SetScalarModeToUsePointFieldData()

    # ---- Scalar range ----
    arr = data.GetPointData().GetArray(field)
    if arr is None:
        print(f'  WARNING: field "{field}" not found in {vti_path}, skipping')
        return False
    sr0, sr1 = arr.GetRange()
    if sr1 - sr0 < 1e-12:
        sr1 = sr0 + 1.0

    # ---- Transfer functions ----
    if field == 'smoke':
        ctf, otf = _make_smoke_tf(sr0, sr1)
    else:
        ctf, otf = _make_vorticity_tf(sr0, sr1, data)

    # ---- Volume property ----
    prop = vtk.vtkVolumeProperty()
    prop.SetColor(ctf)
    prop.SetScalarOpacity(otf)
    prop.ShadeOn()
    prop.SetInterpolationTypeToLinear()
    prop.SetAmbient(0.30)
    prop.SetDiffuse(0.60)
    prop.SetSpecular(0.15)

    volume = vtk.vtkVolume()
    volume.SetMapper(mapper)
    volume.SetProperty(prop)

    # ---- Renderer ----
    renderer = vtk.vtkRenderer()
    renderer.AddVolume(volume)
    renderer.SetBackground(0.06, 0.06, 0.10)

    # ---- Camera ----
    nx, ny, nz = data.GetDimensions()
    center = (nx * 0.5, ny * 0.5, nz * 0.5)
    max_dim = max(nx, ny, nz)
    dist = max_dim * dist_factor

    az_rad = math.radians(azim)
    el_rad = math.radians(elev)

    cx = center[0] + dist * math.cos(el_rad) * math.sin(az_rad)
    cy = center[1] + dist * math.cos(el_rad) * math.cos(az_rad)
    cz = center[2] + dist * math.sin(el_rad)

    camera = renderer.GetActiveCamera()
    camera.SetPosition(cx, cy, cz)
    camera.SetFocalPoint(*center)
    camera.SetViewUp(0, 0, 1)
    camera.SetClippingRange(max_dim * 0.1, max_dim * 10.0)

    # ---- Off-screen render window ----
    ren_win = vtk.vtkRenderWindow()
    ren_win.SetOffScreenRendering(1)
    ren_win.AddRenderer(renderer)
    ren_win.SetSize(resolution[0], resolution[1])
    ren_win.SetMultiSamples(4)

    # ---- Capture to PNG ----
    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(ren_win)
    w2i.SetScale(1)
    w2i.SetInputBufferTypeToRGB()

    writer = vtk.vtkPNGWriter()
    writer.SetInputConnection(w2i.GetOutputPort())
    writer.SetFileName(output_path)

    ren_win.Render()
    writer.Write()
    ren_win.Finalize()
    return True


def visualize(logdir, fields=('vorticity', 'smoke'), azim=30, elev=25, dist_factor=1.8):
    """Read all .vti files under logdir/vtks/ and write volume-rendered images."""
    vtk_dir = os.path.join(logdir, 'vtks')
    if not os.path.isdir(vtk_dir):
        print(f'VTK directory not found: {vtk_dir}')
        return

    out_dir = os.path.join(logdir, '3d_output')
    os.makedirs(out_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(vtk_dir) if f.endswith('.vti')])
    if not files:
        print(f'No .vti files in {vtk_dir}')
        return

    for fname in files:
        vti_path = os.path.join(vtk_dir, fname)
        frame = int(re.search(r'(\d+)', fname).group())

        for field in fields:
            suffix = f'_{field}' if len(fields) > 1 else ''
            out_path = os.path.join(out_dir, f'frame_{frame:03d}{suffix}.png')
            ok = volume_render_vti(vti_path, out_path, field=field,
                                   azim=azim, elev=elev, dist_factor=dist_factor)
            if ok:
                print(f'  {out_path}')

    print(f'Done — {len(files)} frames, fields={fields} → {out_dir}')


if __name__ == '__main__':
    # Parse --fields argument
    fields = ('vorticity', 'smoke')
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    for a in sys.argv[1:]:
        if a.startswith('--fields='):
            fields = tuple(a.split('=', 1)[1].split())

    if args:
        log_path = args[0]
    else:
        sys.path.insert(0, os.path.dirname(__file__))
        from hyperparameters import exp_name
        for base in [os.getcwd(), os.path.join(os.path.dirname(__file__), '..', '..')]:
            candidate = os.path.join(base, 'logs', exp_name)
            if os.path.isdir(os.path.join(candidate, 'vtks')):
                log_path = candidate
                break
        else:
            print('Could not auto-detect log directory. Provide it as argument.')
            sys.exit(1)

    visualize(os.path.abspath(log_path), fields=fields)
